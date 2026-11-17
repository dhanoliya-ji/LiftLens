"""LiftLens: uplift modelling for email targeting on the Hillstrom randomised experiment.

    python liftlens.py          # downloads the data, runs everything, writes results/

Steps
  1. download the Hillstrom data (64,000 customers) and verify its checksum
  2. check randomisation balance, then a stratified train / validation / test split
  3. S, T and X-learners plus a response-model baseline, all on XGBoost
  4. hyperparameters and the final learner chosen by 5-fold cross-validated Qini (test set untouched)
  5. test set: Qini, uplift at 30%, gain over random targeting, deciles, 1,000 paired bootstrap resamples
  6. TreeSHAP explanation of the chosen learner's uplift through a surrogate model
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hillstrom.csv"
OUT = ROOT / "results"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
CSV = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
SOURCES = [f"http://www.minethatdata.com/{CSV}", f"https://raw.githubusercontent.com/bscan/uplift/master/{CSV}"]

# experiment settings
TEST_SIZE, VALID_SIZE, SPLIT_SEED = 0.20, 0.20, 42
GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.05], "min_child_weight": [50, 200]}
FIXED = {"n_estimators_max": 600, "early_stopping_rounds": 40, "subsample": 0.8, "colsample_bytree": 0.8}
BUDGET, BOOTSTRAPS, BOOT_SEED, TRAIN_SEEDS = 0.30, 1000, 7, [0, 1, 2, 3, 4]

# pre-treatment features only; visit, conversion and spend are outcomes and never enter X
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = {"zip_code": ["Rural", "Surburban", "Urban"], "channel": ["Multichannel", "Phone", "Web"]}


# ----------------------------------------------------------------------------- data
def load() -> pd.DataFrame:
    if not DATA.exists():
        for url in SOURCES:
            try:
                raw = urllib.request.urlopen(url, timeout=60).read()
            except Exception as e:  # noqa: BLE001
                print(f"download failed from {url}: {e}")
                continue
            if hashlib.sha256(raw).hexdigest() == SHA256:
                DATA.write_bytes(raw)
                break
    if not DATA.exists() or hashlib.sha256(DATA.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(f"could not get a verified copy of the data; place {CSV} at {DATA}")
    return pd.read_csv(DATA)


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).copy()
    X["log_history"] = np.log1p(df["history"])
    for col, levels in CATEGORICAL.items():
        for lvl in levels:
            X[f"{col}={lvl}"] = (df[col] == lvl).astype(float)
    return X


# ----------------------------------------------------------------------------- metrics
def qini_curve(y, uplift, t):
    """x = customers targeted, Q(x) = Y_t(x) - Y_c(x) * N_t(x) / N_c(x); tied scores move together."""
    o = np.argsort(-uplift, kind="mergesort")
    y, t, u = y[o], t[o], uplift[o]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    keep = np.r_[np.diff(u) != 0, True]
    return np.r_[0, np.arange(1, len(y) + 1)[keep]].astype(float), np.r_[0.0, q[keep]]


def qini(y, uplift, t) -> float:
    """Area between the model's Qini curve and random targeting, divided by the same area for a
    perfect ordering. 0 = random, 1 = perfect."""
    x, q = qini_curve(y, uplift, t)
    xp, qp = qini_curve(y, y * t - y * (1 - t), t)
    base = lambda xs: xs * qp[-1] / xp[-1]
    return float(np.trapezoid(q - base(x), x) / np.trapezoid(qp - base(xp), xp))


def uplift_at_k(y, uplift, t, k) -> float:
    top = np.argsort(-uplift, kind="mergesort")[: int(np.ceil(k * len(y)))]
    return float(y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean())


def ate(y, t) -> float:
    return float(y[t == 1].mean() - y[t == 0].mean())


def deciles(y, uplift, t):
    rows = []
    for d, part in enumerate(np.array_split(np.argsort(-uplift, kind="mergesort"), 10), start=1):
        yt, tt = y[part], t[part]
        rows.append({"decile": d, "customers": len(part), "treated": int(tt.sum()), "control": int((1 - tt).sum()),
                     "observed_uplift": float(yt[tt == 1].mean() - yt[tt == 0].mean()),
                     "predicted_uplift": float(uplift[part].mean())})
    return rows


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ----------------------------------------------------------------------------- learners
def _xgb(kind, p, seed):
    common = dict(tree_method="hist", n_estimators=p["n_estimators_max"], max_depth=p["max_depth"],
                  learning_rate=p["learning_rate"], min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                  colsample_bytree=p["colsample_bytree"], early_stopping_rounds=p["early_stopping_rounds"],
                  random_state=seed, n_jobs=2)
    if kind == "clf":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return xgb.XGBRegressor(objective="reg:squarederror", **common)


def _fit(m, X, y, Xv, yv):
    return m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)


def _p(m, X):
    return m.predict_proba(X)[:, 1]


def _with_t(X, t):
    Z = X.copy()
    Z["treatment"] = t
    return Z


class SLearner:
    """One model on [X, t]; uplift = f(x, 1) - f(x, 0)."""
    name = "S-learner"

    def __init__(self, p, seed=0):
        self.p, self.seed = p, seed

    def fit(self, X, y, t, Xv, yv, tv):
        self.m = _fit(_xgb("clf", self.p, self.seed), _with_t(X, t), y, _with_t(Xv, tv), yv)
        return self

    def predict(self, X):
        return _p(self.m, _with_t(X, 1)) - _p(self.m, _with_t(X, 0))


class TLearner(SLearner):
    """Separate models for emailed and not emailed; uplift = mu1(x) - mu0(x)."""
    name = "T-learner"

    def fit(self, X, y, t, Xv, yv, tv):
        self.mu1 = _fit(_xgb("clf", self.p, self.seed), X[t == 1], y[t == 1], Xv[tv == 1], yv[tv == 1])
        self.mu0 = _fit(_xgb("clf", self.p, self.seed), X[t == 0], y[t == 0], Xv[tv == 0], yv[tv == 0])
        return self

    def predict(self, X):
        return _p(self.mu1, X) - _p(self.mu0, X)


class XLearner(TLearner):
    """T-learner, then regress imputed individual effects and blend them by the propensity score
    (Kunzel et al., 2019). In a randomised test the propensity is the known treatment share."""
    name = "X-learner"

    def fit(self, X, y, t, Xv, yv, tv):
        super().fit(X, y, t, Xv, yv, tv)
        self.e = float(t.mean())
        d1, d1v = y[t == 1] - _p(self.mu0, X[t == 1]), yv[tv == 1] - _p(self.mu0, Xv[tv == 1])
        d0, d0v = _p(self.mu1, X[t == 0]) - y[t == 0], _p(self.mu1, Xv[tv == 0]) - yv[tv == 0]
        self.tau1 = _fit(_xgb("reg", self.p, self.seed), X[t == 1], d1, Xv[tv == 1], d1v)
        self.tau0 = _fit(_xgb("reg", self.p, self.seed), X[t == 0], d0, Xv[tv == 0], d0v)
        return self

    def predict(self, X):
        return self.e * self.tau0.predict(X) + (1 - self.e) * self.tau1.predict(X)


class ResponseModel(SLearner):
    """Baseline most campaigns use: P(visit | emailed). Ranks who responds, not who responds because of it."""
    name = "Response model"

    def fit(self, X, y, t, Xv, yv, tv):
        self.m = _fit(_xgb("clf", self.p, self.seed), X[t == 1], y[t == 1], Xv[tv == 1], yv[tv == 1])
        return self

    def predict(self, X):
        return _p(self.m, X)


LEARNERS = {"s_learner": SLearner, "t_learner": TLearner, "x_learner": XLearner, "response": ResponseModel}


def select(cls, X, y, t):
    """Grid search by 5-fold cross-validated Qini on train + validation."""
    splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, t * 2 + y))
    trials = []
    for values in itertools.product(*GRID.values()):
        p = {**dict(zip(GRID, values)), **FIXED}
        scores = [qini(y[va], cls(p).fit(X.iloc[tr], y[tr], t[tr], X.iloc[va], y[va], t[va]).predict(X.iloc[va]), t[va])
                  for tr, va in splits]
        trials.append({"params": dict(zip(GRID, values)), "cv_qini_mean": float(np.mean(scores)),
                       "cv_qini_std": float(np.std(scores))})
    best = max(trials, key=lambda r: r["cv_qini_mean"])
    print(f"  {cls.name:14s} CV Qini {best['cv_qini_mean']:.4f} +/- {best['cv_qini_std']:.4f} {best['params']}")
    return {**best["params"], **FIXED}, trials, best["cv_qini_mean"]


# ----------------------------------------------------------------------------- explanation and figures
def explain(X_train, u_train, X_test, u_test):
    """XGBoost surrogate on predicted uplift, exact TreeSHAP from pred_contribs, fidelity as test R^2."""
    sur = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, tree_method="hist", random_state=0, n_jobs=2)
    sur.fit(X_train, u_train)
    shap = sur.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)[:, :-1]
    mean_abs = np.abs(shap).mean(0)
    imp = [{"feature": X_test.columns[i], "mean_abs_shap": float(mean_abs[i]),
            "direction": float(np.corrcoef(X_test.iloc[:, i], shap[:, i])[0, 1])
            if X_test.iloc[:, i].std() > 0 and shap[:, i].std() > 0 else 0.0} for i in np.argsort(-mean_abs)]
    return float(r2_score(u_test, sur.predict(X_test))), imp, shap


def figures(y, t, preds, best_rows, best_name, shap, X):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, u in preds.items():
        x, q = qini_curve(y, u, t)
        ax.plot(x / len(y), q, label=f"{LEARNERS[name].name} (Qini {qini(y, u, t):.3f})")
    ax.plot([0, 1], [0, qini_curve(y, np.zeros(len(y)), t)[1][-1]], "k--", label="Random targeting")
    ax.set(xlabel="Share of customers emailed, ranked by score", ylabel="Incremental visits",
           title="Qini curves on the held-out test set")
    ax.legend(fontsize=8), ax.grid(alpha=0.3), fig.tight_layout(), fig.savefig(OUT / "qini_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    d = np.arange(1, 11)
    ax.bar(d - 0.2, [r["observed_uplift"] * 100 for r in best_rows], 0.4, label="Observed uplift")
    ax.bar(d + 0.2, [r["predicted_uplift"] * 100 for r in best_rows], 0.4, label="Predicted uplift")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xticks=d, xlabel="Decile of predicted uplift (1 = highest)", ylabel="Visit rate, treated minus control (pp)",
           title=f"{best_name}: uplift by decile, test set")
    ax.legend(), fig.tight_layout(), fig.savefig(OUT / "uplift_by_decile.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    order = np.argsort(-np.abs(shap).mean(0))[:8][::-1]
    for row, j in enumerate(order):
        v = X.iloc[:, j].to_numpy(float)
        c = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)
        idx = rng.choice(len(v), min(3000, len(v)), replace=False)
        ax.scatter(shap[idx, j] * 100, row + rng.uniform(-0.3, 0.3, len(idx)), c=c[idx], cmap="coolwarm", s=3)
    ax.set_yticks(range(len(order)), [X.columns[j] for j in order])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="SHAP value on predicted uplift (pp); colour = feature value, red high",
           title=f"{best_name}: per-customer SHAP values")
    fig.tight_layout(), fig.savefig(OUT / "shap_beeswarm.png", dpi=150)
    plt.close("all")


# ----------------------------------------------------------------------------- experiment
def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    df = load()
    X, t = features(df), (df["segment"] != "No E-Mail").to_numpy(int)  # treatment: any email
    y, y2 = df["visit"].to_numpy(int), df["conversion"].to_numpy(int)

    # randomisation check: standardised mean difference of every feature, emailed vs not
    smd = {c: float((X[c][t == 1].mean() - X[c][t == 0].mean()) / np.sqrt((X[c][t == 1].var() + X[c][t == 0].var()) / 2))
           for c in X.columns}

    # stratified split on treatment x outcome; the test set is only used at the end
    idx = np.arange(len(df))
    dev, test = train_test_split(idx, test_size=TEST_SIZE, stratify=t * 2 + y, random_state=SPLIT_SEED)
    tr, va = train_test_split(dev, test_size=VALID_SIZE, stratify=(t * 2 + y)[dev], random_state=SPLIT_SEED)
    Xtr, Xva, Xte = (X.iloc[i].reset_index(drop=True) for i in (tr, va, test))
    print(f"rows {len(df)} | train {len(tr)} valid {len(va)} test {len(test)} | max |SMD| {max(map(abs, smd.values())):.4f}")

    res = {"dataset": {"name": "Hillstrom MineThatData E-Mail Analytics Challenge (2008)", "rows": len(df),
                       "treatment": "any email", "outcome": "visit",
                       "split_sizes": {"train": len(tr), "valid": len(va), "test": len(test)},
                       "test_ate_visit": ate(y[test], t[test]), "test_ate_conversion": ate(y2[test], t[test])},
           "max_abs_smd": max(map(abs, smd.values())), "balance_smd": smd, "learners": {}}

    print("model selection (5-fold CV Qini on train + validation)")
    Xdev = pd.concat([Xtr, Xva], ignore_index=True)
    ydev, tdev = np.r_[y[tr], y[va]], np.r_[t[tr], t[va]]
    preds, preds_train = {}, {}
    for key, cls in LEARNERS.items():
        p, trials, cv_q = select(cls, Xdev, ydev, tdev)
        per_seed = [cls(p, seed=s).fit(Xtr, y[tr], t[tr], Xva, y[va], t[va]) for s in TRAIN_SEEDS]
        u_seeds = [m.predict(Xte) for m in per_seed]
        preds[key], preds_train[key] = u_seeds[0], per_seed[0].predict(Xtr)
        q_seeds = [qini(y[test], u, t[test]) for u in u_seeds]
        n_k = int(np.ceil(BUDGET * len(test)))
        res["learners"][key] = {
            "name": cls.name, "chosen_params": {k: p[k] for k in GRID}, "search": trials, "cv_qini": cv_q,
            "test_qini_by_seed": q_seeds, "test_qini_seed_mean": float(np.mean(q_seeds)),
            "test_qini_seed_std": float(np.std(q_seeds)),
            "test": {"qini": q_seeds[0], "uplift_at_30": uplift_at_k(y[test], u_seeds[0], t[test], BUDGET),
                     "incremental_at_k": uplift_at_k(y[test], u_seeds[0], t[test], BUDGET) * n_k},
            "test_conversion_qini": qini(y2[test], u_seeds[0], t[test]),
            "share_predicted_negative": float((u_seeds[0] < 0).mean())}

    # paired bootstrap: every learner scored on the same resamples of the test set
    yt, tt, n, n_k = y[test], t[test], len(test), int(np.ceil(BUDGET * len(test)))
    random_inc = ate(yt, tt) * n_k
    res["targeting"] = {"budget_share": BUDGET, "customers_targeted": n_k, "random_incremental_visits": random_inc}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {k: {"qini": [], "uplift_at_k": [], "gain": []} for k in preds}
    for _ in range(BOOTSTRAPS):
        i = rng.integers(0, n, n)
        rinc = ate(yt[i], tt[i]) * n_k
        for k, u in preds.items():
            ub = u[i] + rng.uniform(0, 1e-12, n)  # break ties created by duplicated rows
            uk = uplift_at_k(yt[i], ub, tt[i], BUDGET)
            boot[k]["qini"].append(qini(yt[i], ub, tt[i]))
            boot[k]["uplift_at_k"].append(uk)
            boot[k]["gain"].append(uk * n_k / rinc - 1)
    for k in preds:
        r, b = res["learners"][k], {m: np.array(v) for m, v in boot[k].items()}
        r["test"]["gain_vs_random"] = r["test"]["incremental_at_k"] / random_inc - 1
        r["bootstrap"] = {"qini_ci95": ci(b["qini"]), "uplift_at_k_ci95": ci(b["uplift_at_k"]),
                          "gain_vs_random_ci95": ci(b["gain"]), "p_gain_vs_random_le_0": float(np.mean(b["gain"] <= 0))}

    # the final learner is chosen by cross-validated Qini, never by test results
    best = max((k for k in preds if k != "response"), key=lambda k: res["learners"][k]["cv_qini"])
    diff = np.array(boot[best]["qini"]) - np.array(boot["response"]["qini"])
    res["best_learner"] = best
    res["best_vs_response"] = {"qini_diff": res["learners"][best]["test"]["qini"] - res["learners"]["response"]["test"]["qini"],
                               "qini_diff_ci95": ci(diff), "p_diff_le_0": float(np.mean(diff <= 0))}
    res["deciles"] = deciles(yt, preds[best], tt)
    top = np.argsort(-preds[best], kind="mergesort")[: n // 10]
    rng = np.random.default_rng(BOOT_SEED)
    tops = []
    for _ in range(BOOTSTRAPS):
        i = rng.integers(0, len(top), len(top))
        tops.append(yt[top][i][tt[top][i] == 1].mean() - yt[top][i][tt[top][i] == 0].mean())
    res["top_decile_uplift_ci95"] = ci(np.array(tops))

    r2, imp, shap = explain(Xtr, preds_train[best], Xte, preds[best])
    res["explanation"] = {"learner": best, "surrogate_r2_test": r2, "importance": imp}
    figures(yt, tt, preds, res["deciles"], LEARNERS[best].name, shap, Xte)
    res["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results.json", "w"), indent=2)

    print(f"\n{'learner':16s}{'test Qini':>10s}{'95% CI':>18s}{'gain vs random 30%':>20s}")
    for k, v in res["learners"].items():
        print(f"{v['name']:16s}{v['test']['qini']:10.4f}  [{v['bootstrap']['qini_ci95'][0]:.3f}, "
              f"{v['bootstrap']['qini_ci95'][1]:.3f}]{v['test']['gain_vs_random']:+19.1%}")
    print(f"chosen by CV: {LEARNERS[best].name} | surrogate R^2 {r2:.3f} | runtime {res['runtime_seconds']} s")


if __name__ == "__main__":
    main()
