"""Tests for the demo-data export. Run: pytest -q"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src import config, export_demo_data as ex


def test_clean_makes_valid_json_and_rounds():
    obj = {"a": float("nan"), "b": [1.234567, float("inf")], "c": np.float64(0.123456), "d": np.int64(3),
           "e": pd.Timestamp("2024-05-06")}
    out = ex.clean(obj)
    assert out == {"a": None, "b": [1.2346, None], "c": 0.1235, "d": 3, "e": "2024-05-06"}
    json.dumps(out, allow_nan=False)


def test_no_text_guard_blocks_headlines_and_urls():
    with pytest.raises(config.PipelineError):
        ex.assert_no_text({"rows": [{"headline": "Bitcoin jumps"}]})
    with pytest.raises(config.PipelineError):
        ex.assert_no_text({"x": ["https://example.com/a"]})
    ex.assert_no_text({"rows": [{"date": "2024-01-01", "close": 1.0}]})


@pytest.mark.skipif(not (config.ROOT / "results" / "walk_forward" / "walk_forward_predictions.csv").exists(),
                    reason="pipeline results not built")
def test_real_export_is_valid_small_and_text_free():
    data = ex.build_all()
    assert set(data) == {"meta", "price_sentiment", "predictions"}
    for name, payload in data.items():
        payload = ex.clean(payload)
        ex.assert_no_text(payload)
        assert len(json.dumps(payload, allow_nan=False)) < 2_000_000, name
    ps = data["price_sentiment"]
    assert len(ps) == data["meta"]["data"]["priceDays"] and ps[0]["date"] < ps[-1]["date"]
    assert all(r["vader"] is None for r in ps if r["noNews"])
    sets = data["predictions"]["sets"]
    assert set(sets) == {"frozen", "walkForward"}
    for s_ in sets.values():
        dates = [r["date"] for r in s_["rows"]]
        assert dates == sorted(dates) and len(set(dates)) == len(dates)
        assert all(0 <= r[k] <= 1 for r in s_["rows"] for k in ("naive", "xgboost", "lstmA", "lstmBVader", "lstmBFinbert"))
        assert all(r["up"] in (0, 1) for r in s_["rows"])
    # the frozen split lies inside the walk-forward coverage
    assert sets["walkForward"]["from"] <= sets["frozen"]["from"] and sets["frozen"]["to"] <= sets["walkForward"]["to"]
