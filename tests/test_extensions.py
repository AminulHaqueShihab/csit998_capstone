"""Tests for multi-day targets, purging, large-move filtering and block bootstrap. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config, evaluate_extensions, modelling


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(config, "SPLIT_DATES", {
        "train": ("2022-01-01", "2022-03-31"),
        "validation": ("2022-04-01", "2022-04-30"),
        "test": ("2022-05-01", "2022-05-31"),
    })
    dates = pd.date_range("2022-01-01", "2022-06-30", freq="D")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates)))), index=dates)
    d = pd.date_range("2022-01-15", "2022-05-31", freq="D")  # window end days (first 14 days dropped)
    split = pd.Series(np.where(d <= "2022-03-31", "train", np.where(d <= "2022-04-30", "validation", "test")))
    meta = pd.DataFrame({"date": d, "split": split.values})
    return meta, close


def test_labels_match_future_close(setup):
    meta, close = setup
    for h in (1, 3, 7):
        s = evaluate_extensions.task_samples(meta, close, h, 0.0)
        rows = s[s["purge_ok"]].sample(10, random_state=0)
        for _, r in rows.iterrows():
            assert r["y"] == int(close[r["date"] + pd.Timedelta(days=h)] > close[r["date"]])


def test_purge_drops_last_h_minus_1_days_of_each_split(setup):
    meta, close = setup
    daily = evaluate_extensions.task_samples(meta, close, 1, 0.0)
    assert daily["purge_ok"].all()  # h=1 identical to the main experiment: nothing dropped
    weekly = evaluate_extensions.task_samples(meta, close, 7, 0.0)
    for split, end in (("train", "2022-03-31"), ("validation", "2022-04-30"), ("test", "2022-05-31")):
        kept = weekly[(weekly["split"] == split) & weekly["purge_ok"]]["date"]
        assert kept.max() == pd.Timestamp(end) - pd.Timedelta(days=6)
        # no kept label window reaches more than one day beyond its split
        assert (kept + pd.Timedelta(days=7) <= pd.Timestamp(end) + pd.Timedelta(days=1)).all()


def test_large_move_filter_applies_to_training_and_validation_only(setup):
    meta, close = setup
    s = evaluate_extensions.task_samples(meta, close, 1, 0.02)
    ret1 = (close.shift(-1) / close - 1)
    ret1 = s["date"].map(ret1)
    train_like = s["split"] != "test"
    assert (s.loc[train_like & s["use"], "date"].map(ret1.set_axis(s["date"]).to_dict()).abs() >= 0.02).all()
    assert s.loc[~train_like, "use"].all()  # every test day is scored
    small_train = train_like & (ret1.abs() < 0.02)
    assert small_train.any() and not s.loc[small_train, "use"].any()
    assert ((s["big"]) == (ret1.abs() >= 0.02)).all()


def test_chance_interval_widens_with_horizon():
    d1 = evaluate_extensions.chance_halfwidth(300, 1)
    d7 = evaluate_extensions.chance_halfwidth(300, 7)
    assert d7 == pytest.approx(d1 * np.sqrt(7))


def test_block_bootstrap_widens_interval_for_autocorrelated_outcomes():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 60)
    y = np.repeat(base, 7)  # outcomes persist for a week
    # each model's errors persist for a week too, so the correctness difference is autocorrelated
    pa = np.clip(y + np.repeat(rng.normal(0, 0.8, 60), 7), 0, 1)
    pb = np.clip(y + np.repeat(rng.normal(0, 0.8, 60), 7), 0, 1)
    plain = modelling.paired_bootstrap_diff(y, pa, pb, n_boot=600)
    block = modelling.paired_block_bootstrap_diff(y, pa, pb, block=7, n_boot=600)
    assert block["point"] == plain["point"]
    assert (block["ci95_high"] - block["ci95_low"]) > (plain["ci95_high"] - plain["ci95_low"])


def test_block_bootstrap_null_and_real_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    same = modelling.paired_block_bootstrap_diff(y, y * 0.9, y * 0.9, block=3, n_boot=300)
    assert same["point"] == 0 and not same["ci_excludes_zero"]
    coin = rng.random(300)
    better = modelling.paired_block_bootstrap_diff(y, coin, y.astype(float), block=3, n_boot=300)
    assert better["ci_excludes_zero"] and better["point"] > 0.3


def test_refuses_second_run_without_force(monkeypatch, tmp_path):
    import sys
    frozen, final = tmp_path / "frozen.json", tmp_path / "final_results.json"
    frozen.write_text("{}")
    final.write_text("{}")
    out = tmp_path / "ext.json"
    out.write_text("{}")
    monkeypatch.setattr(modelling, "MODEL_CONFIG_FILE", frozen)
    monkeypatch.setattr(evaluate_extensions, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate_extensions, "OUT", out)
    monkeypatch.setattr(config, "DAILY_FINBERT", tmp_path / "daily.csv")
    (tmp_path / "daily.csv").write_text("x")
    monkeypatch.setattr(sys, "argv", ["evaluate_extensions"])
    with pytest.raises(config.PipelineError, match="already exist"):
        evaluate_extensions.main()
