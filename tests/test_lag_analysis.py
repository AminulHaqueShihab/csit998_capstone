"""Tests for the lag analysis helpers. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import lag_analysis


def test_aligned_pairs_sentiment_before_move_for_positive_k():
    x = np.arange(10)
    y = np.arange(10) * 10
    ax, ay = lag_analysis.aligned(x, y, 2)       # sentiment on day i, move on day i+2
    assert ax[0] == 0 and ay[0] == 20 and len(ax) == len(ay) == 8
    ax, ay = lag_analysis.aligned(x, y, -2)      # sentiment on day i, move on day i-2 (sentiment reacts)
    assert ax[0] == 2 and ay[0] == 0 and len(ax) == len(ay) == 8
    ax, ay = lag_analysis.aligned(x, y, 0)
    assert (ax == x).all() and (ay == y).all()


def test_permutation_test_finds_planted_lead_lag_only_at_the_right_lag():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    y = np.zeros(n)
    y[1:] = 0.5 * x[:-1] + rng.normal(size=n - 1)  # y_t depends on x_{t-1}: sentiment BEFORE the move
    for k, should_find in ((1, True), (0, False), (-1, False)):
        ax, ay = lag_analysis.aligned(x, y, k)
        rho, p, lo, hi = lag_analysis.spearman_perm(ax, ay, 2000, np.random.default_rng(1))
        assert (p < 0.01) == should_find
        if should_find:
            assert rho > 0.3
        assert lo < 0 < hi


def test_circular_shift_null_is_not_fooled_by_autocorrelated_series():
    rng = np.random.default_rng(3)

    def ar1(n, phi=0.97):
        e = rng.normal(size=n)
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = phi * s[i - 1] + e[i]
        return s
    pvals = []
    for _ in range(20):
        pvals.append(lag_analysis.spearman_perm(ar1(800), ar1(800), 1000, rng)[1])
    assert np.mean(np.array(pvals) < 0.05) < 0.5  # independent series: not mostly "significant"


def test_surprise_uses_only_past_days():
    n = 40
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    vader = pd.DataFrame({"date": idx, "vader_mean": np.linspace(-0.2, 0.3, n), "vader_pos_count": 5,
                          "vader_neg_count": 3, "article_count": 10})
    finbert = pd.DataFrame({"date": idx, "finbert_mean": np.linspace(0.3, -0.1, n)})
    price = pd.DataFrame({"date": idx, "close": 100 + np.arange(n, dtype=float)})
    base = lag_analysis.build_daily(price, vader, finbert)
    changed = vader.copy()
    changed.loc[25:, "vader_mean"] += 5.0       # change the future only
    new = lag_analysis.build_daily(price, changed, finbert)
    pd.testing.assert_series_equal(base["vader_surprise"].iloc[:25], new["vader_surprise"].iloc[:25])
    expected = base["vader_mean"].iloc[20] - base["vader_mean"].iloc[13:20].mean()
    assert base["vader_surprise"].iloc[20] == pytest.approx(expected)


def test_lag_table_shape_and_flags(monkeypatch):
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame({m: rng.normal(size=n) for m in lag_analysis.MEASURES + lag_analysis.TARGETS}, index=idx)
    t = lag_analysis.lag_table(df, n_perm=200, seed=1)
    assert len(t) == len(lag_analysis.TARGETS) * len(lag_analysis.MEASURES) * len(lag_analysis.LAGS)
    assert set(t["lag_k"]) == set(lag_analysis.LAGS)
    assert t["p_perm"].between(0, 1).all() and (t["bonferroni_sig"] <= t["nominal_p05"]).all()
