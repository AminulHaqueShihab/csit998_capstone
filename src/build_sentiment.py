"""Stage 3: score headlines with VADER and aggregate to one row per UTC day.

Run: python -m src.build_sentiment
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config
from .config import PipelineError


def score_vader(headlines: pd.Series) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()
    return headlines.map(lambda t: analyzer.polarity_scores(t)["compound"])


def aggregate_daily(news: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    news = news.copy()
    news["is_pos"] = news["vader_compound"] >= config.VADER_POS_THRESHOLD
    news["is_neg"] = news["vader_compound"] <= config.VADER_NEG_THRESHOLD
    grouped = news.groupby("date_utc")
    daily = pd.DataFrame({
        "article_count": grouped.size(),
        "unique_source_count": grouped["news_source"].nunique(),
        "vader_mean": grouped["vader_compound"].mean(),
        "vader_pos_count": grouped["is_pos"].sum(),
        "vader_neg_count": grouped["is_neg"].sum(),
        "gdelt_tone_mean": grouped["gdelt_tone"].mean(),
    })
    daily = daily.reindex(dates)
    daily.index.name = "date"
    for col in ["article_count", "unique_source_count", "vader_pos_count", "vader_neg_count"]:
        daily[col] = daily[col].fillna(0).astype(int)
    daily["vader_neu_count"] = daily["article_count"] - daily["vader_pos_count"] - daily["vader_neg_count"]
    daily["no_news_flag"] = (daily["article_count"] == 0).astype(int)
    daily.insert(0, "asset", config.ASSET)
    return daily.reset_index()


def main():
    config.require_file(config.NEWS_CLEAN, "build_sentiment", "Run `python -m src.prepare_news` first.")
    config.require_file(config.PRICE_CLEAN, "build_sentiment", "Run `python -m src.prepare_price` first.")
    news = pd.read_csv(config.NEWS_CLEAN, parse_dates=["date_utc"])
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"])

    news["vader_compound"] = score_vader(news["headline"])
    news.to_csv(config.NEWS_SCORED, index=False)

    dates = pd.date_range(price["date"].min(), price["date"].max(), freq="D")
    daily = aggregate_daily(news, dates)
    daily.to_csv(config.DAILY_SENTIMENT, index=False)

    no_news = daily[daily["no_news_flag"] == 1]["date"].dt.strftime("%Y-%m-%d").tolist()
    report = {
        "scored_articles": len(news),
        "vader_version": "vaderSentiment 3.3.2",
        "vader_thresholds": {"positive": config.VADER_POS_THRESHOLD, "negative": config.VADER_NEG_THRESHOLD},
        "days": len(daily),
        "days_without_news": len(no_news),
        "days_without_news_list": no_news,
        "articles_per_day": daily["article_count"].describe().round(2).to_dict(),
        "days_with_fewer_than_5_articles": int((daily["article_count"] < 5).sum()),
        "vader_mean_overall": float(news["vader_compound"].mean()),
        "output": str(config.DAILY_SENTIMENT.relative_to(config.ROOT)),
    }
    config.write_report("sentiment_report.json", report)
    print(f"[build_sentiment] {len(daily)} days, {len(no_news)} without news, "
          f"median {daily['article_count'].median():.0f} articles/day")


if __name__ == "__main__":
    main()
