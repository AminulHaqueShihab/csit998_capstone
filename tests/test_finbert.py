"""Tests for the FinBERT feature stage (no model download needed). Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import (build_finbert_features, build_modelling_table, build_sentiment, config, evaluate_finbert,
                 features)
from tests.test_splits import make_table


def make_news():
    return pd.DataFrame({
        "date_utc": pd.to_datetime(["2022-03-01", "2022-03-01", "2022-03-03"]),
        "finbert_score": [0.8, -0.4, 0.1],
        "finbert_label": ["positive", "negative", "neutral"],
    })


def test_daily_finbert_counts_and_explicit_zero_news_days():
    dates = pd.date_range("2022-02-28", "2022-03-04", freq="D")
    d = build_finbert_features.aggregate_daily_finbert(make_news(), dates).set_index("date")
    assert d.loc["2022-03-01", "article_count"] == 2
    assert d.loc["2022-03-01", "finbert_pos_count"] == 1 and d.loc["2022-03-01", "finbert_neg_count"] == 1
    assert d.loc["2022-03-01", "finbert_mean"] == pytest.approx(0.2)
    assert d.loc["2022-03-02", "no_news_flag"] == 1 and np.isnan(d.loc["2022-03-02", "finbert_mean"])
    assert (d["article_count"] == d[["finbert_pos_count", "finbert_neg_count", "finbert_neu_count"]].sum(axis=1)).all()


def test_finbert_features_mirror_vader_structure_and_keep_identical_samples():
    n = 500
    rng = np.random.default_rng(1)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    price = pd.DataFrame({"date": dates, "asset": "BTC", "open": close, "high": close * 1.02,
                          "low": close * 0.98, "close": close, "volume": rng.uniform(1e3, 5e3, n)})
    news = pd.DataFrame({
        "date_utc": rng.choice(dates, 2000), "vader_compound": rng.uniform(-1, 1, 2000),
        "news_source": rng.choice(list("abc"), 2000), "gdelt_tone": rng.normal(size=2000),
        "finbert_score": rng.uniform(-1, 1, 2000),
    })
    news["finbert_label"] = np.where(news["finbert_score"] > 0.3, "positive",
                                     np.where(news["finbert_score"] < -0.3, "negative", "neutral"))
    vader_daily = build_sentiment.aggregate_daily(news, pd.DatetimeIndex(dates))
    fb_daily = build_finbert_features.aggregate_daily_finbert(news, pd.DatetimeIndex(dates))
    table = build_modelling_table.build(price, vader_daily)
    t = evaluate_finbert.add_finbert_inputs(features.add_model_inputs(table), fb_daily)
    assert (t["fb_article_count"] == t["article_count"]).all()  # same headlines, same dates
    assert set(features.FINBERT_SENTIMENT_FEATURES) <= set(t.columns)
    assert ((t["finbert_pos_share"] >= 0) & (t["finbert_pos_share"] <= 1)).all()
    assert len(features.FINBERT_SENTIMENT_FEATURES) == len(features.MODEL_SENTIMENT_FEATURES)
    assert not t[features.FINBERT_SENTIMENT_FEATURES].isna().any().any()


def test_mismatched_headline_sets_are_rejected():
    table = make_table(300)
    fb_daily = pd.DataFrame({"date": table["date"], "finbert_mean": 0.0, "finbert_pos_count": 0,
                             "finbert_neg_count": 0, "article_count": table["article_count"] + 1})
    with pytest.raises(config.PipelineError):
        evaluate_finbert.add_finbert_inputs(features.add_model_inputs(table), fb_daily)
