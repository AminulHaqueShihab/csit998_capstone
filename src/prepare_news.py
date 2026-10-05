"""Stage 2: clean GDELT Bitcoin headlines.

Steps (every drop is counted in reports/news_quality_report.json):
  1. parse timestamps to UTC; compute the time each item became available
  2. decode HTML entities, normalise whitespace, drop empty/very short headlines
  3. keep English only (no machine-translation flag + detector fallback)
  4. label asset relevance (BTC / ETH / BTC+ETH / irrelevant); drop irrelevant
  5. drop duplicate URLs and same-day duplicate headlines

Availability time: GDELT's own timestamp (`gdelt_timestamp`, 15-minute UTC batch) is when the
item entered the dataset. `precise_publication_timestamp` is usually missing and can be earlier.
Both are valid, so the later one is used: that is the conservative choice for a no-leakage cutoff.

Run: python -m src.prepare_news
"""
import html
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from . import config
from .config import PipelineError

BTC_RE = re.compile(r"\b(bitcoins?|btc|xbt)\b|₿", re.IGNORECASE)
ETH_RE = re.compile(r"\b(ethereum|ether|eth)\b", re.IGNORECASE)
ENGLISH_HINT_RE = re.compile(r"\b(the|of|to|in|for|on|as|is|at|by|with|and|a|an|from|after|over)\b", re.IGNORECASE)
TRACKING_PARAMS = {"fbclid", "gclid", "ref", "ref_src", "cmpid", "ocid", "src"}


def clean_headline(text: str) -> str:
    text = html.unescape(html.unescape(text or ""))  # some titles are double-encoded
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return (url or "").strip().lower()
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))


def headline_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def asset_relevance(text: str) -> str:
    btc, eth = bool(BTC_RE.search(text)), bool(ETH_RE.search(text))
    if btc and eth:
        return "BTC+ETH"
    return "BTC" if btc else ("ETH" if eth else "irrelevant")


def is_english(text: str) -> bool:
    """True unless the detector is confident the text is another language.

    Headlines with common English function words are accepted without running the (slow) detector.
    """
    if ENGLISH_HINT_RE.search(text):
        return True
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    try:
        top = detect_langs(text)[0]
    except LangDetectException:
        return False
    return top.lang == "en" or top.prob < config.LANGDETECT_MIN_PROB


def parse_v2tone(series: pd.Series) -> pd.DataFrame:
    parts = series.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    parts = parts.iloc[:, :4]
    parts.columns = ["gdelt_tone", "gdelt_positive", "gdelt_negative", "gdelt_polarity"]
    return parts


def load_gdelt(files) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False, encoding_errors="replace")
        missing = {"gdelt_timestamp", "headline", "article_url", "news_source", "TranslationInfo", "V2Tone",
                   "precise_publication_timestamp"} - set(df.columns)
        if missing:
            raise PipelineError(f"[prepare_news] {f.name} is missing columns: {sorted(missing)}")
        df["source_file"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame):
    stats = {"rows_loaded": len(raw)}
    drops = {}
    df = raw.copy()

    df["gdelt_ts"] = pd.to_datetime(df["gdelt_timestamp"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    precise = pd.to_datetime(df["precise_publication_timestamp"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    # ignore implausible precise timestamps (before 2000 or more than a day after GDELT saw the item)
    valid_precise = precise.where((precise >= pd.Timestamp("2000-01-01", tz="UTC"))
                                  & (precise <= df["gdelt_ts"] + pd.Timedelta(days=1)))
    stats["precise_timestamp_present"] = int(valid_precise.notna().sum())
    df["published_at_utc"] = valid_precise
    df["available_at_utc"] = pd.concat([df["gdelt_ts"], valid_precise], axis=1).max(axis=1)

    def drop(mask, reason):
        nonlocal df
        drops[reason] = int(mask.sum())
        df = df[~mask]

    drop(df["gdelt_ts"].isna(), "invalid_timestamp")
    df = df.assign(headline=df["headline"].map(clean_headline))
    drop(df["headline"].str.len() < config.MIN_HEADLINE_CHARS, "empty_or_too_short_headline")
    drop(df["TranslationInfo"].str.strip() != "", "machine_translated_source")
    english = df["headline"].map(is_english)
    drop(~english, "detected_non_english")
    df = df.assign(asset_relevance=df["headline"].map(asset_relevance))
    drop(df["asset_relevance"].isin(["ETH", "irrelevant"]), "not_bitcoin_relevant")

    df = df.sort_values(["available_at_utc", "article_url"], kind="stable")
    df["url_key"] = df["article_url"].map(canonical_url)
    drop(df.duplicated("url_key", keep="first"), "duplicate_url")
    df["date_utc"] = df["available_at_utc"].dt.floor("D").dt.tz_localize(None)
    df["hl_key"] = df["headline"].map(headline_key)
    drop(df.duplicated(["date_utc", "hl_key"], keep="first"), "duplicate_headline_same_day")

    df["article_id"] = pd.util.hash_pandas_object(df["url_key"], index=False).astype("uint64").astype(str)
    out = pd.concat([df, parse_v2tone(df["V2Tone"])], axis=1)[[
        "article_id", "available_at_utc", "published_at_utc", "date_utc", "headline", "article_url",
        "news_source", "asset_relevance", "gdelt_tone", "gdelt_positive", "gdelt_negative",
        "gdelt_polarity", "source_file",
    ]].reset_index(drop=True)
    stats.update({"drops": drops, "rows_clean": len(out)})
    return out, stats


def main():
    files = sorted(config.GDELT_DIR.glob("gdelt_btc_news_*_raw.csv"))
    if not files:
        raise PipelineError(f"[prepare_news] no GDELT files in {config.GDELT_DIR}. "
                            "Copy gdelt_btc_news_*_raw.csv into data/raw/gdelt first.")
    out, stats = clean(load_gdelt(files))
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.NEWS_CLEAN, index=False)

    by_month = out.groupby(out["date_utc"].dt.to_period("M")).size()
    report = {
        "inputs": [f.name for f in files],
        **stats,
        "date_min": out["date_utc"].min(), "date_max": out["date_utc"].max(),
        "asset_relevance_counts": out["asset_relevance"].value_counts().to_dict(),
        "articles_per_month_min": int(by_month.min()), "articles_per_month_max": int(by_month.max()),
        "timestamp_policy": "available_at_utc = max(gdelt_timestamp, precise_publication_timestamp when valid); "
                            "an item belongs to the UTC date of available_at_utc.",
        "output": str(config.NEWS_CLEAN.relative_to(config.ROOT)),
    }
    config.write_report("news_quality_report.json", report)
    print(f"[prepare_news] loaded {stats['rows_loaded']}, kept {stats['rows_clean']}; drops: {stats['drops']}")


if __name__ == "__main__":
    main()
