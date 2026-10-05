"""Unit tests for cleaning rules and leakage controls. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import build_modelling_table, build_sentiment, config, features, prepare_news, prepare_price


def make_price(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates, "asset": "BTC", "open": close * 0.99, "high": close * 1.02,
        "low": close * 0.97, "close": close, "volume": rng.uniform(1e3, 5e3, n),
    })


# ---------- leakage: price features and target ----------

def test_price_features_use_only_past_rows():
    price = make_price()
    full = features.price_features(price)
    for k in (40, 75, 110):
        truncated = features.price_features(price.iloc[: k + 1])
        pd.testing.assert_series_equal(full.iloc[k], truncated.iloc[k], check_names=False)


def test_future_change_does_not_alter_past_features():
    price = make_price()
    base = features.price_features(price)
    altered = price.copy()
    altered.loc[90:, "close"] *= 3  # change the future only
    new = features.price_features(altered)
    feature_cols = [c for c in features.PRICE_FEATURES if c != "hl_range_pct"]
    pd.testing.assert_frame_equal(base.loc[:89, feature_cols], new.loc[:89, feature_cols])


def test_target_is_next_day_direction_and_last_row_dropped():
    price = make_price()
    table = features.add_target(features.price_features(price), price)
    assert table["date"].max() == price["date"].max() - pd.Timedelta(days=1)
    nxt = price.set_index("date")["close"]
    for _, row in table.sample(15, random_state=1).iterrows():
        expected = int(nxt[row["date"] + pd.Timedelta(days=1)] > row["close"])
        assert row["target_direction"] == expected


def test_target_not_filled_across_missing_calendar_day():
    price = make_price().drop(index=[60]).reset_index(drop=True)  # a missing date
    table = features.add_target(features.price_features(price), price)
    missing_prev = price.loc[59, "date"]
    assert missing_prev not in set(table["date"])  # next calendar day absent -> row dropped


def test_rsi_bounds():
    r = features.price_features(make_price(200))["rsi_14"].dropna()
    assert ((r >= 0) & (r <= 100)).all()


# ---------- sentiment aggregation and cutoff ----------

def make_news():
    ts = pd.to_datetime(["2022-03-01 23:59:59", "2022-03-02 00:00:00", "2022-03-02 12:00:00"], utc=True)
    return pd.DataFrame({
        "available_at_utc": ts, "date_utc": ts.floor("D").tz_localize(None),
        "vader_compound": [0.6, -0.5, 0.0], "news_source": ["a", "b", "a"],
        "gdelt_tone": [1.0, -1.0, 0.0],
    })


def test_daily_aggregation_respects_utc_cutoff_and_flags_empty_days():
    dates = pd.date_range("2022-02-28", "2022-03-04", freq="D")
    daily = build_sentiment.aggregate_daily(make_news(), dates).set_index("date")
    assert daily.loc["2022-03-01", "article_count"] == 1  # 23:59:59 stays on day t
    assert daily.loc["2022-03-02", "article_count"] == 2  # midnight belongs to the next day
    assert daily.loc["2022-02-28", "no_news_flag"] == 1
    assert daily.loc["2022-03-02", "vader_pos_count"] == 0 and daily.loc["2022-03-02", "vader_neg_count"] == 1
    assert np.isnan(daily.loc["2022-02-28", "vader_mean"])  # never invented
    assert (daily["article_count"] == daily[["vader_pos_count", "vader_neg_count", "vader_neu_count"]].sum(axis=1)).all()


def test_modelling_table_inputs_not_after_cutoff():
    price = make_price(150)
    news = make_news()
    daily = build_sentiment.aggregate_daily(news, pd.DatetimeIndex(price["date"]))
    table = build_modelling_table.build(price, daily)
    build_modelling_table.check_cutoff(table, news)
    assert (table["target_date"] == table["date"] + pd.Timedelta(days=1)).all()
    assert not table.isna().any().any()


def test_check_cutoff_catches_item_dated_before_it_existed():
    news = make_news()
    news.loc[0, "date_utc"] = pd.Timestamp("2022-02-28")
    with pytest.raises(config.PipelineError):
        build_modelling_table.check_cutoff(None, news)


# ---------- cleaning rules ----------

def test_clean_headline_decodes_double_encoded_entities():
    assert prepare_news.clean_headline("Bitcoin &amp;#x2019;s  rally") == "Bitcoin ’s rally"


@pytest.mark.parametrize("text,label", [
    ("Bitcoin hits new high", "BTC"), ("BTC/USD forecast", "BTC"),
    ("Bitcoin and Ethereum surge", "BTC+ETH"), ("Ethereum upgrade", "ETH"),
    ("Latest from NewsBTC", "irrelevant"), ("HitBTC token volume", "irrelevant"),
])
def test_asset_relevance(text, label):
    assert prepare_news.asset_relevance(text) == label


def test_canonical_url_strips_tracking_and_www():
    a = prepare_news.canonical_url("https://www.Example.com/news/x/?utm_source=a&id=7#top")
    b = prepare_news.canonical_url("https://example.com/news/x?id=7")
    assert a == b


def test_clean_pipeline_drop_reasons_and_availability_time():
    raw = pd.DataFrame({
        "gdelt_timestamp": ["20220301120000", "20220301120000", "20220301130000", "20220301140000",
                            "20220301150000", "20220301150000", "notatime"],
        "precise_publication_timestamp": ["", "", "", "", "20220301100000", "", ""],
        "headline": ["Bitcoin climbs as ETF hopes rise", "Bitcoin climbs as ETF hopes rise",
                     "Bitcoin climbs as ETF hopes rise!", "Bitcoin sube tras anuncio",
                     "Ethereum merge delayed again", "Bitcoin fees drop to yearly low",
                     "Bitcoin whatever headline here"],
        "article_url": ["https://a.com/1", "https://a.com/1?utm_x=1", "https://b.com/2", "https://c.com/3",
                        "https://d.com/4", "https://e.com/5", "https://f.com/6"],
        "news_source": list("aabcdef"),
        "TranslationInfo": ["", "", "", "srclc:spa;eng:GT-SPA 1.0", "", "", ""],
        "V2Tone": ["1,2,1,3,0,0,10"] * 7,
        "source_file": "t.csv",
    })
    out, stats = prepare_news.clean(raw)
    assert stats["drops"]["invalid_timestamp"] == 1
    assert stats["drops"]["machine_translated_source"] == 1
    assert stats["drops"]["not_bitcoin_relevant"] == 1  # ETH-only
    assert stats["drops"]["duplicate_url"] == 1
    assert stats["drops"]["duplicate_headline_same_day"] == 1  # same headline modulo punctuation
    assert len(out) == 2
    # precise timestamp earlier than gdelt: the later (gdelt) time is the availability time
    assert (out["available_at_utc"] >= pd.Timestamp("2022-03-01 12:00", tz="UTC")).all()


# ---------- price cleaning ----------

def test_epoch_units_ms_and_us_both_parse_to_same_date():
    s = pd.Series(["1609459200000", "1735689600000000"])
    out = prepare_price._epoch_to_utc(s)
    assert out.iloc[0] == pd.Timestamp("2021-01-01", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("2025-01-01", tz="UTC")


def test_price_cleaning_rejects_bad_rows_and_dedupes():
    def row(ts, o, h, l, c, v):
        return [str(ts), o, h, l, c, v, str(ts + 86399999), "1", "1", "1", "1", "0", "f.zip"]
    base = 1609459200000
    rows = [row(base, "10", "12", "9", "11", "5"),
            row(base + 86400000, "11", "10", "9", "11", "5"),   # high < open/close -> OHLC violation
            row(base + 2 * 86400000, "11", "13", "10", "12", "-1"),  # negative volume
            row(base + 3 * 86400000, "12", "14", "11", "13", "5"),
            row(base + 3 * 86400000, "12", "14", "11", "13", "5")]  # duplicate day
    raw = pd.DataFrame(rows, columns=prepare_price.KLINE_COLS + ["source_file"])
    out, rejected, counts = prepare_price.clean(raw)
    assert len(out) == 2 and counts["duplicate_dates_removed"] == 1
    assert set(rejected["reason"]) == {"ohlc_logic_violation", "negative_volume"}


# ---------- processed artefacts (skipped until the pipeline has been run) ----------

@pytest.mark.skipif(not config.MODELLING_TABLE.exists(), reason="modelling table not built")
def test_built_modelling_table_integrity():
    t = pd.read_csv(config.MODELLING_TABLE, parse_dates=["date", "target_date", "prediction_cutoff_utc"])
    assert t["date"].is_unique and t["date"].is_monotonic_increasing
    assert not t.isna().any().any()
    assert set(t["target_direction"].unique()) <= {0, 1}
    assert (t["target_date"] == t["date"] + pd.Timedelta(days=1)).all()
    assert (t["prediction_cutoff_utc"] < t["target_date"]).all()
