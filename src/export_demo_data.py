"""Stage 14: export saved results as small JSON files for the demo web app.

Run: python -m src.export_demo_data
"""
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from . import config
from .config import PipelineError

RESULTS = config.ROOT / "results"
REPORTS = config.REPORTS
FORBIDDEN_KEYS = {"headline", "article_url", "url", "text", "title"}


def clean(obj, ndigits: int = 4):
    """Round floats, turn NaN/inf into None and numpy scalars into Python types (valid JSON)."""
    if isinstance(obj, dict):
        return {str(k): clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, ndigits) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, ndigits)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def read_json(path: Path):
    config.require_file(path, "export_demo_data", "Run the earlier pipeline stage that creates it first.")
    return json.loads(path.read_text())


def price_sentiment() -> list:
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"])[["date", "close"]]
    v = pd.read_csv(config.DAILY_SENTIMENT, parse_dates=["date"])[
        ["date", "vader_mean", "article_count", "no_news_flag"]]
    f = pd.read_csv(config.DAILY_FINBERT, parse_dates=["date"])[["date", "finbert_mean"]]
    d = price.merge(v, on="date").merge(f, on="date").sort_values("date")
    return [{"date": r.date.strftime("%Y-%m-%d"), "close": r.close,
             "vader": None if r.no_news_flag else r.vader_mean,
             "finbert": None if r.no_news_flag else r.finbert_mean,
             "articles": int(r.article_count), "noNews": bool(r.no_news_flag)} for r in d.itertuples()]


def _prediction_rows(frame: pd.DataFrame, names: dict, price: pd.Series) -> list:
    missing = set(names) - set(frame.columns)
    if missing:
        raise PipelineError(f"[export_demo_data] missing prediction columns {sorted(missing)}")
    rows = []
    for r in frame.itertuples(index=False):
        nxt = price.get(r.date + pd.Timedelta(days=1))
        row = {"date": r.date.strftime("%Y-%m-%d"), "up": int(r.y_true), "close": float(price[r.date]),
               "closeNext": None if nxt is None else float(nxt)}
        for col, key in names.items():
            row[key] = float(getattr(r, col))
        rows.append(row)
    return rows


def predictions() -> dict:
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"]).set_index("date")["close"]
    frozen = pd.read_csv(RESULTS / "finbert_test_predictions.csv", parse_dates=["date"])
    wf = pd.read_csv(RESULTS / "walk_forward" / "walk_forward_predictions.csv", parse_dates=["date"])
    names_frozen = {"proba_naive_majority": "naive", "proba_xgboost_price": "xgboost",
                    "proba_lstm_a_price_only": "lstmA", "proba_lstm_b_price_plus_sentiment": "lstmBVader",
                    "proba_lstm_b_finbert": "lstmBFinbert"}
    names_wf = {"proba_naive_majority": "naive", "proba_xgboost_price": "xgboost",
                "proba_lstm_a_price_only": "lstmA", "proba_lstm_b_vader": "lstmBVader",
                "proba_lstm_b_finbert": "lstmBFinbert"}
    sets = {
        "frozen": {"label": "Frozen test split", "rows": _prediction_rows(frozen, names_frozen, price)},
        "walkForward": {"label": "Walk-forward (retrained every 6 months)",
                        "rows": _prediction_rows(wf, names_wf, price)},
    }
    for s_ in sets.values():
        s_["from"], s_["to"] = s_["rows"][0]["date"], s_["rows"][-1]["date"]
    return {"threshold": 0.5, "sets": sets}


def meta(now: str) -> dict:
    price = read_json(REPORTS / "price_quality_report.json")
    sent = read_json(REPORTS / "sentiment_report.json")
    settings = read_json(RESULTS / "model_settings.json")
    return {
        "generatedAt": now,
        "disclaimer": "Experimental research output for CSIT998. Not financial advice.",
        "data": {"start": price["date_min"][:10], "end": price["date_max"][:10], "priceDays": price["rows_clean"],
                 "noNewsDays": sent["days_without_news_list"]},
        "model": {"seeds": len(settings["seeds"]), "seqLen": settings["seq_len"], "lstm": settings["lstm_hyperparameters"],
                  "xgboost": settings["xgboost_hyperparameters"], "frozen": "2026-10-08",
                  "modellingTableSha256": settings["modelling_table_sha256"]},
        "splits": {k: list(v) for k, v in config.SPLIT_DATES.items()},
    }


def assert_no_text(obj, path="root"):
    """Safety check: nothing that looks like headline text or URLs leaves the research repo."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise PipelineError(f"[export_demo_data] forbidden key '{k}' at {path}")
            assert_no_text(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            assert_no_text(v, f"{path}[{i}]")
    elif isinstance(obj, str) and obj.startswith(("http://", "https://")):
        raise PipelineError(f"[export_demo_data] URL found at {path}")


def build_all() -> dict:
    return {
        "meta": meta(time.strftime("%Y-%m-%d")),
        "price_sentiment": price_sentiment(),
        "predictions": predictions(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(config.ROOT / "web" / "public" / "data"),
                        help="folder for the JSON files (default: web/public/data; created if missing)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, payload in build_all().items():
        payload = clean(payload)
        assert_no_text(payload)
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        total += path.stat().st_size
        print(f"[export_demo_data] {path.name:<22}{path.stat().st_size / 1024:>8.1f} KB")
    print(f"[export_demo_data] total {total / 1024:.0f} KB -> {out}")


if __name__ == "__main__":
    main()
