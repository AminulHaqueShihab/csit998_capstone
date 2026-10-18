"""Stage 9: aggregate FinBERT headline scores to one row per UTC day.

Run: python -m src.build_finbert_features
"""
import numpy as np
import pandas as pd

from . import config
from .config import PipelineError


def aggregate_daily_finbert(news: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    g = news.groupby("date_utc")
    daily = pd.DataFrame({
        "article_count": g.size(),
        "finbert_mean": g["finbert_score"].mean(),
        "finbert_pos_count": g["finbert_label"].apply(lambda s: int((s == "positive").sum())),
        "finbert_neg_count": g["finbert_label"].apply(lambda s: int((s == "negative").sum())),
    }).reindex(dates)
    daily.index.name = "date"
    for c in ("article_count", "finbert_pos_count", "finbert_neg_count"):
        daily[c] = daily[c].fillna(0).astype(int)
    daily["finbert_neu_count"] = daily["article_count"] - daily["finbert_pos_count"] - daily["finbert_neg_count"]
    daily["no_news_flag"] = (daily["article_count"] == 0).astype(int)
    daily.insert(0, "asset", config.ASSET)
    return daily.reset_index()


def main():
    for p, hint in ((config.NEWS_CLEAN, "prepare_news"), (config.NEWS_FINBERT, "score_finbert"),
                    (config.NEWS_SCORED, "build_sentiment"), (config.PRICE_CLEAN, "prepare_price")):
        config.require_file(p, "build_finbert_features", f"Run `python -m src.{hint}` first.")
    news = pd.read_csv(config.NEWS_CLEAN, parse_dates=["date_utc"], dtype={"article_id": str})
    fb = pd.read_csv(config.NEWS_FINBERT, dtype={"article_id": str})
    if len(fb) != len(news) or not (fb["article_id"].to_numpy() == news["article_id"].to_numpy()).all():
        raise PipelineError("[build_finbert_features] FinBERT scores are not aligned with news_clean; re-run score_finbert")
    news = pd.concat([news, fb.drop(columns="article_id")], axis=1)
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"])
    dates = pd.date_range(price["date"].min(), price["date"].max(), freq="D")
    daily = aggregate_daily_finbert(news, dates)
    daily.to_csv(config.DAILY_FINBERT, index=False)

    vader = pd.read_csv(config.NEWS_SCORED, usecols=["vader_compound"])["vader_compound"].to_numpy()
    v_label = np.where(vader >= config.VADER_POS_THRESHOLD, "positive",
                       np.where(vader <= config.VADER_NEG_THRESHOLD, "negative", "neutral"))
    agree = float((v_label == news["finbert_label"].to_numpy()).mean())
    vdaily = pd.read_csv(config.DAILY_SENTIMENT, parse_dates=["date"])
    both = daily.merge(vdaily[["date", "vader_mean"]], on="date")
    both = both[daily["no_news_flag"].to_numpy() == 0]
    report = {
        "headlines": len(news), "days": len(daily), "days_without_news": int(daily["no_news_flag"].sum()),
        "label_agreement_vader_vs_finbert": round(agree, 4),
        "headline_score_correlation": round(float(np.corrcoef(vader, news["finbert_score"])[0, 1]), 4),
        "daily_mean_correlation": round(float(both["finbert_mean"].corr(both["vader_mean"])), 4),
        "finbert_label_share": news["finbert_label"].value_counts(normalize=True).round(4).to_dict(),
        "vader_label_share": pd.Series(v_label).value_counts(normalize=True).round(4).to_dict(),
        "output": str(config.DAILY_FINBERT.relative_to(config.ROOT)),
    }
    config.write_report("finbert_features_report.json", report)
    print(f"[build_finbert_features] {len(daily)} days; label agreement with VADER {agree:.1%}; "
          f"daily-mean correlation {report['daily_mean_correlation']}")


if __name__ == "__main__":
    main()
