"""Causal feature and target construction (PF-01, TP-01..03).

Every feature in row t uses only rows <= t. The target for row t is built from the close of
day t+1 and is never available as an input. Verified in tests/test_pipeline.py.
"""
import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "close", "volume", "return_1d", "log_return_1d", "close_lag_1", "close_lag_7",
    "sma_7", "sma_14", "sma_30", "close_over_sma_7", "close_over_sma_30",
    "momentum_7", "momentum_14", "rsi_14", "volatility_14", "hl_range_pct", "volume_ratio_7",
]
SENTIMENT_FEATURES = [
    "vader_mean", "vader_pos_count", "vader_neg_count", "vader_neu_count", "article_count",
    "unique_source_count", "gdelt_tone_mean", "no_news_flag",
]
WARMUP_DAYS = 30  # longest look-back window (sma_30)

# Inputs actually fed to the models. Price *levels* (close, lags, SMAs) are left out: BTC trades at very
# different levels across the study period, so a scaler fitted on 2021-2024 would see out-of-range values
# in 2025-2026. Stationary ratios/returns are used instead. Sentiment *counts* are replaced by shares plus
# log(article_count) so day-to-day volume differences and the 2024-01 export step weigh less.
MODEL_PRICE_FEATURES = [
    "return_1d", "log_return_1d", "close_over_sma_7", "close_over_sma_30", "momentum_7", "momentum_14",
    "rsi_14", "volatility_14", "hl_range_pct", "volume_ratio_7", "log_volume",
]
MODEL_SENTIMENT_FEATURES = [  # VADER only (mandatory method); GDELT tone is an optional variant
    "vader_mean", "vader_pos_share", "vader_neg_share", "log_article_count", "no_news_flag",
]
OPTIONAL_GDELT_FEATURES = ["gdelt_tone_mean"]
# SF-03: same structure as the VADER features, scored by FinBERT instead (same items, dates and aggregation)
FINBERT_SENTIMENT_FEATURES = ["finbert_mean", "finbert_pos_share", "finbert_neg_share", "log_article_count",
                              "no_news_flag"]
BINARY_FEATURES = {"no_news_flag"}  # passed through unscaled


def add_model_inputs(table: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (hence causal) transforms that build the model input columns from the modelling table."""
    t = table.copy()
    n = t["article_count"].where(t["article_count"] > 0)
    t["vader_pos_share"] = (t["vader_pos_count"] / n).fillna(0.0)
    t["vader_neg_share"] = (t["vader_neg_count"] / n).fillna(0.0)
    t["log_article_count"] = np.log1p(t["article_count"])
    t["log_volume"] = np.log(t["volume"])
    return t


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    out = 100 - 100 / (1 + gain / loss)
    return out.where(loss != 0, 100.0).where(gain.notna() & loss.notna())


def price_features(price: pd.DataFrame) -> pd.DataFrame:
    """Return one row per date with causal price features. `price` must be date-sorted."""
    p = price.sort_values("date").reset_index(drop=True)
    f = pd.DataFrame({"date": p["date"], "asset": p["asset"]})
    close = p["close"]
    f["close"] = close
    f["volume"] = p["volume"]
    f["return_1d"] = close.pct_change()
    f["log_return_1d"] = np.log(close / close.shift(1))
    f["close_lag_1"] = close.shift(1)
    f["close_lag_7"] = close.shift(7)
    for w in (7, 14, 30):
        f[f"sma_{w}"] = close.rolling(w).mean()
    f["close_over_sma_7"] = close / f["sma_7"]
    f["close_over_sma_30"] = close / f["sma_30"]
    f["momentum_7"] = close / close.shift(7) - 1
    f["momentum_14"] = close / close.shift(14) - 1
    f["rsi_14"] = rsi(close, 14)
    f["volatility_14"] = f["return_1d"].rolling(14).std()
    f["hl_range_pct"] = (p["high"] - p["low"]) / close
    f["volume_ratio_7"] = p["volume"] / p["volume"].rolling(7).mean()
    return f


def add_target(table: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """target_direction = 1 if Close(t+1) > Close(t) else 0, using the next *calendar* day.

    Rows whose next calendar day has no price (including the final row) are dropped, never filled.
    """
    close_by_date = price.set_index("date")["close"]
    nxt = table["date"] + pd.Timedelta(days=1)
    next_close = nxt.map(close_by_date)
    out = table.copy()
    out["target_date"] = nxt
    out["target_direction"] = (next_close > out["close"]).astype("float").where(next_close.notna())
    return out.dropna(subset=["target_direction"]).astype({"target_direction": int})


def add_sentiment(table: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Join day-t sentiment onto row t (it predicts t+1: fixed one-day lag, TP-03).

    Days without news get 0 for the mean score and tone (neutral) with no_news_flag = 1; the fill
    value is a constant, so nothing is estimated from other (possibly future) rows.
    """
    cols = ["date"] + SENTIMENT_FEATURES
    merged = table.merge(daily[cols], on="date", how="left", validate="one_to_one")
    for c in ("vader_mean", "gdelt_tone_mean"):
        merged[c] = merged[c].fillna(0.0)
    return merged
