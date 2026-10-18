"""Tests for the walk-forward fold construction. Run: pytest -q"""
import sys

import numpy as np
import pandas as pd
import pytest

from src import config, features, modelling, walk_forward
from tests.test_splits import make_table

COLS = features.MODEL_PRICE_FEATURES + features.MODEL_SENTIMENT_FEATURES


@pytest.fixture
def t():
    return features.add_model_inputs(make_table(900))  # dates 2021-01-31 onwards


def fold(t, start="2022-06-01", end="2022-08-31", **kw):
    return walk_forward.fold_data(t, COLS, 7, start, end, **kw)


def test_train_validation_test_are_ordered_disjoint_and_cover_the_fold(t):
    f = fold(t)
    tr, va, te = (pd.to_datetime(f[f"dates_{n}"]) for n in ("train", "validation", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert te.min() == pd.Timestamp("2022-06-01") and te.max() == pd.Timestamp("2022-08-31")
    assert len(va) == walk_forward.VAL_DAYS and va.max() == pd.Timestamp("2022-05-31")
    assert len(te) == (te.max() - te.min()).days + 1  # no gaps
    assert f["X_train"].shape[1:] == (7, len(COLS))


def test_nothing_from_the_test_fold_reaches_the_scaler_or_training_data(t):
    base = fold(t)
    altered = t.copy()
    altered.loc[altered["date"] >= "2022-06-01", "volume"] *= 100   # change test-fold rows only
    altered.loc[altered["date"] >= "2022-06-01", "log_volume"] = np.log(altered["volume"])
    new = fold(altered)
    assert base["scaler"] == new["scaler"]
    np.testing.assert_allclose(base["X_train"], new["X_train"])
    np.testing.assert_allclose(base["X_validation"], new["X_validation"])
    assert not np.allclose(base["X_test"], new["X_test"])


def test_scaler_is_fitted_on_training_rows_only_not_validation(t):
    f = fold(t)
    val_start = pd.Timestamp("2022-06-01") - pd.Timedelta(days=walk_forward.VAL_DAYS)
    train_rows = t[t["date"] < val_start]
    assert f["scaler"]["mean"]["return_1d"] == pytest.approx(train_rows["return_1d"].mean())


def test_consecutive_folds_have_disjoint_test_windows_and_growing_training(t):
    a = fold(t, "2022-03-01", "2022-05-31")
    b = fold(t, "2022-06-01", "2022-08-31")
    assert set(a["dates_test"]).isdisjoint(set(b["dates_test"]))
    assert len(b["y_train"]) > len(a["y_train"])
    assert pd.Timestamp(b["dates_validation"].max()) == pd.Timestamp("2022-05-31")  # validation ends the day before the fold


def test_labels_are_next_day_direction(t):
    f = fold(t)
    meta = f["meta_test"]
    assert (f["y_test"] == meta["target_direction"].to_numpy()).all()
    assert (meta["target_date"] == meta["date"] + pd.Timedelta(days=1)).all()


def test_defined_folds_are_back_to_back_and_inside_the_data():
    starts = [pd.Timestamp(a) for a, _ in walk_forward.FOLDS]
    ends = [pd.Timestamp(b) for _, b in walk_forward.FOLDS]
    for e, s in zip(ends[:-1], starts[1:]):
        assert s == e + pd.Timedelta(days=1)
    assert ends[-1] == pd.Timestamp(config.SPLIT_DATES["test"][1])


def test_refuses_second_run_without_force(monkeypatch, tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}")
    out = tmp_path / "wf"
    out.mkdir()
    (out / "walk_forward_results.json").write_text("{}")
    (tmp_path / "daily.csv").write_text("x")
    monkeypatch.setattr(modelling, "MODEL_CONFIG_FILE", frozen)
    monkeypatch.setattr(walk_forward, "OUT_DIR", out)
    monkeypatch.setattr(config, "DAILY_FINBERT", tmp_path / "daily.csv")
    monkeypatch.setattr(sys, "argv", ["walk_forward"])
    with pytest.raises(config.PipelineError, match="already exist"):
        walk_forward.main()
