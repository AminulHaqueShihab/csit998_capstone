"""Stage 4: build the modelling table (price + sentiment features, next-day direction target).

Run: python -m src.build_modelling_table
"""
import pandas as pd

from . import config, features
from .config import PipelineError


def build(price: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    table = features.price_features(price)
    table = table.iloc[features.WARMUP_DAYS:].reset_index(drop=True)  # drop rows lacking full look-back
    table = features.add_sentiment(table, daily)
    table = features.add_target(table, price)
    table["prediction_cutoff_utc"] = table["date"] + pd.Timedelta(hours=23, minutes=59, seconds=59)
    columns = (["date", "asset", "prediction_cutoff_utc"] + features.PRICE_FEATURES
               + features.SENTIMENT_FEATURES + ["target_date", "target_direction"])
    return table[columns]


def check_cutoff(table: pd.DataFrame, news: pd.DataFrame) -> None:
    """TP-01: every news item counted in day t was available on or before day t's cutoff."""
    cutoff = news["date_utc"] + pd.Timedelta(hours=23, minutes=59, seconds=59)
    available = news["available_at_utc"].dt.tz_localize(None)
    if (available > cutoff).any():
        raise PipelineError("[build_modelling_table] a news item is dated before it became available")


def main():
    for p, hint in ((config.PRICE_CLEAN, "prepare_price"), (config.DAILY_SENTIMENT, "build_sentiment"),
                    (config.NEWS_CLEAN, "prepare_news")):
        config.require_file(p, "build_modelling_table", f"Run `python -m src.{hint}` first.")
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"])
    daily = pd.read_csv(config.DAILY_SENTIMENT, parse_dates=["date"])
    news = pd.read_csv(config.NEWS_CLEAN, parse_dates=["date_utc", "available_at_utc"])

    table = build(price, daily)
    check_cutoff(table, news)
    if table.isna().any().any():
        bad = table.columns[table.isna().any()].tolist()
        raise PipelineError(f"[build_modelling_table] unexpected missing values in {bad}")
    table.to_csv(config.MODELLING_TABLE, index=False)

    counts = table["target_direction"].value_counts().to_dict()
    report = {
        "rows": len(table),
        "date_min": table["date"].min(), "date_max": table["date"].max(),
        "last_target_date": table["target_date"].max(),
        "class_balance": {"up(1)": counts.get(1, 0), "not_up(0)": counts.get(0, 0),
                          "share_up": round(counts.get(1, 0) / len(table), 4)},
        "warmup_days_dropped": features.WARMUP_DAYS,
        "days_without_news_in_table": int(table["no_news_flag"].sum()),
        "price_features": features.PRICE_FEATURES,
        "sentiment_features": features.SENTIMENT_FEATURES,
        "cutoff_check": "passed: every news item used on day t was available by 23:59:59 UTC on day t",
        "output": str(config.MODELLING_TABLE.relative_to(config.ROOT)),
        "output_sha256": config.sha256_file(config.MODELLING_TABLE),
    }
    config.write_report("modelling_table_report.json", report)
    print(f"[build_modelling_table] {len(table)} rows {table['date'].min().date()}..{table['date'].max().date()}, "
          f"share up = {report['class_balance']['share_up']}")


if __name__ == "__main__":
    main()
