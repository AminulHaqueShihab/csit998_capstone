# Data dictionary: `data/processed/modelling_table_v1.csv` (DS-01)

One row per UTC day `t`. All inputs are known by the prediction cutoff (23:59:59 UTC on day `t`);
the target looks one day ahead. Features are **unscaled**: scaling is fitted on training data only,
at training time (PF-02).

| Column | Type | Description |
|---|---|---|
| `date` | date (UTC) | Day `t` |
| `asset` | str | `BTC` |
| `prediction_cutoff_utc` | datetime | `date` 23:59:59 UTC. Nothing after this time is used |
| `target_date` | date | `date` + 1 day |
| **`target_direction`** | int (0/1) | **Target.** 1 if Close(t+1) > Close(t), else 0 |

## Price features (from Binance BTCUSDT, causal: rows <= t only)

| Column | Definition |
|---|---|
| `close`, `volume` | Day t close (USDT) and base-asset volume |
| `return_1d`, `log_return_1d` | Simple and log return vs the previous day's close |
| `close_lag_1`, `close_lag_7` | Close 1 and 7 days earlier |
| `sma_7`, `sma_14`, `sma_30` | Simple moving averages of close |
| `close_over_sma_7`, `close_over_sma_30` | Close divided by the moving average (trend position) |
| `momentum_7`, `momentum_14` | Close(t) / Close(t-n) - 1 |
| `rsi_14` | Relative Strength Index, 14-day simple-average version, 0-100 |
| `volatility_14` | Rolling standard deviation of `return_1d` over 14 days |
| `hl_range_pct` | (High - Low) / Close on day t |
| `volume_ratio_7` | Volume divided by its 7-day average |

The first 30 days of the price history are dropped (not enough look-back for `sma_30`).

## Sentiment features (from GDELT headlines available during UTC day t)

| Column | Definition |
|---|---|
| `vader_mean` | Mean VADER compound score (-1..1) of that day's retained headlines; 0 on days without news |
| `vader_pos_count`, `vader_neg_count`, `vader_neu_count` | Headlines with compound >= 0.05, <= -0.05, and in between |
| `article_count` | Retained, de-duplicated, English, Bitcoin-relevant headlines |
| `unique_source_count` | Distinct news sources that day |
| `gdelt_tone_mean` | Mean GDELT V2Tone "tone" value; 0 on days without news |
| `no_news_flag` | 1 if `article_count` is 0 (the zero fills above are constants, not estimates) |

Sentiment of day `t` is paired with the target of day `t+1` (fixed one-day lag, TP-03).

# Intermediate tables

| File | Contents |
|---|---|
| `btcusdt_1d_clean.csv` | Validated Binance daily bars (UTC), with `extreme_move_flag` (|return| > 10%, audit only) |
| `news_clean.csv.gz` | One row per retained headline: `available_at_utc`, `published_at_utc` (often blank), `date_utc`, `headline`, `article_url`, `news_source`, `asset_relevance`, GDELT tone fields |
| `news_scored.csv.gz` | `news_clean` plus `vader_compound` |
| `daily_sentiment.csv` | One row per calendar day with all sentiment aggregates (zero-news days explicit, means empty) |
