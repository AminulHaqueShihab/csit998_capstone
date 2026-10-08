"""Tests for the split, scaling and sequence stage. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import build_modelling_table, build_sentiment, config, features, make_splits


def make_table(n=700):
    rng = np.random.default_rng(3)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    price = pd.DataFrame({"date": dates, "asset": "BTC", "open": close, "high": close * 1.02,
                          "low": close * 0.98, "close": close, "volume": rng.uniform(1e3, 5e3, n)})
    news_dates = rng.choice(dates, 3000)
    news = pd.DataFrame({"date_utc": news_dates, "vader_compound": rng.uniform(-1, 1, 3000),
                         "news_source": rng.choice(list("abcde"), 3000), "gdelt_tone": rng.normal(0, 2, 3000)})
    daily = build_sentiment.aggregate_daily(news, pd.DatetimeIndex(dates))
    return build_modelling_table.build(price, daily)


@pytest.fixture
def small_splits(monkeypatch):
    monkeypatch.setattr(config, "SPLIT_DATES", {
        "train": ("2021-01-01", "2022-06-30"),
        "validation": ("2022-07-01", "2022-10-31"),
        "test": ("2022-11-01", "2022-11-30"),
    })
    table = make_table()
    return table[table["date"] <= "2022-11-30"].reset_index(drop=True)


def test_splits_are_back_to_back_and_cover_all_dates():
    table = make_table(300)
    dates = table["date"]
    labels = make_splits.split_of(dates.clip(upper=pd.Timestamp("2026-08-30")))
    assert labels.notna().all()
    order = {"train": 0, "validation": 1, "test": 2}
    codes = labels.map(order)
    assert codes.is_monotonic_increasing


def test_split_rejects_overlap_or_gap(monkeypatch):
    monkeypatch.setattr(config, "SPLIT_DATES", {"train": ("2021-01-01", "2022-01-31"),
                                                "validation": ("2022-01-31", "2022-06-30"),
                                                "test": ("2022-07-01", "2022-12-31")})
    with pytest.raises(config.PipelineError):
        make_splits.split_of(pd.Series(pd.date_range("2021-01-01", periods=10)))


def test_scaler_uses_training_rows_only(small_splits):
    arrays, tabular, sc_a, sc_b, t = make_splits.build(small_splits, seq_len=7)
    altered = small_splits.copy()
    altered.loc[altered["date"] >= "2022-07-01", "volume"] *= 50  # change validation/test only
    _, _, sc_a2, _, _ = make_splits.build(altered, seq_len=7)
    assert sc_a == sc_a2  # scaler parameters unchanged by non-training data
    train = t[t["split"] == "train"]
    for c in features.MODEL_PRICE_FEATURES:
        assert sc_a["mean"][c] == pytest.approx(train[c].mean())


def test_train_features_are_standardised_and_flag_untouched(small_splits):
    arrays, *_ = make_splits.build(small_splits, seq_len=7)
    flat = arrays["X_b_train"].reshape(-1, arrays["X_b_train"].shape[2])
    cols = list(arrays["features_b"])
    i = cols.index("return_1d")
    assert abs(flat[:, i].mean()) < 0.1 and 0.8 < flat[:, i].std() < 1.2
    flag = flat[:, cols.index("no_news_flag")]
    assert set(np.unique(flag)) <= {0.0, 1.0}


def test_window_alignment_no_future_rows(small_splits):
    seq = 7
    arrays, tabular, sc_a, _, t = make_splits.build(small_splits, seq_len=seq)
    scaled = make_splits.apply_scaler(t, sc_a)
    for name in ("train", "validation", "test"):
        dates = pd.to_datetime(arrays[f"dates_{name}"])
        for k in (0, len(dates) // 2, len(dates) - 1):
            end_row = t.index[t["date"] == dates[k]][0]
            np.testing.assert_allclose(arrays[f"X_a_{name}"][k], scaled[end_row - seq + 1: end_row + 1])
            assert arrays[f"y_{name}"][k] == t.loc[end_row, "target_direction"]
            # the window's last input row is day t; its target date is t+1, which is not an input
            assert t.loc[end_row, "target_date"] == dates[k] + pd.Timedelta(days=1)


def test_identical_dates_for_model_a_and_b_and_ordered_disjoint_splits(small_splits):
    arrays, tabular, *_ = make_splits.build(small_splits, seq_len=7)
    for name in ("train", "validation", "test"):
        assert arrays[f"X_a_{name}"].shape[:2] == arrays[f"X_b_{name}"].shape[:2]
        assert len(arrays[f"y_{name}"]) == len(arrays[f"dates_{name}"]) == len(tabular[name])
    assert arrays["dates_train"][-1] < arrays["dates_validation"][0]
    assert arrays["dates_validation"][-1] < arrays["dates_test"][0]
    assert arrays["X_b_train"].shape[2] == arrays["X_a_train"].shape[2] + len(features.MODEL_SENTIMENT_FEATURES)
    # model B = model A features + sentiment, in that order
    assert list(arrays["features_b"][: len(arrays["features_a"])]) == list(arrays["features_a"])


def test_gap_in_table_is_rejected(small_splits):
    broken = small_splits.drop(index=100).reset_index(drop=True)
    with pytest.raises(config.PipelineError):
        make_splits.build(broken, seq_len=7)


@pytest.mark.skipif(not config.SEQUENCES.exists(), reason="splits not built")
def test_built_artefacts_consistent():
    z = np.load(config.SEQUENCES)
    for n in ("train", "validation", "test"):
        assert not np.isnan(z[f"X_a_{n}"]).any() and not np.isnan(z[f"X_b_{n}"]).any()
        assert set(np.unique(z[f"y_{n}"])) <= {0, 1}
    assert z["X_a_train"].shape[1] == config.SEQ_LEN
