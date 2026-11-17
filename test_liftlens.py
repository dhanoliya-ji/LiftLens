"""Uplift metrics checked on synthetic data where the email helps only customers with x > 0."""
import numpy as np

from liftlens import deciles, qini, uplift_at_k


def data(n=20000):
    rng = np.random.default_rng(0)
    x, t = rng.normal(size=n), rng.integers(0, 2, n)
    y = (rng.uniform(size=n) < 0.10 + 0.15 * t * (x > 0)).astype(int)
    return x, t, y


def test_qini():
    x, t, y = data()
    assert qini(y, x, t) > 0.2                                   # true effect ranks well
    assert qini(y, -x, t) < 0                                    # reversed ranking is worse than random
    assert abs(qini(y, np.random.default_rng(1).normal(size=len(y)), t)) < 0.1  # random is about 0
    assert np.isclose(qini(y, y * t - y * (1 - t), t), 1.0)      # perfect ordering scores 1


def test_uplift_at_k_and_deciles():
    x, t, y = data()
    top = np.argsort(-x)[: int(np.ceil(0.3 * len(x)))]
    assert np.isclose(uplift_at_k(y, x, t, 0.3), y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean())
    rows = deciles(y, x, t)
    assert sum(r["customers"] for r in rows) == len(y) and rows[0]["observed_uplift"] > rows[-1]["observed_uplift"]
