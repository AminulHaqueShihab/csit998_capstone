"""Tests for metrics, training determinism, bootstrap and the test-set guards. Run: pytest -q"""
import sys

import numpy as np
import pytest

from src import config, evaluate_final, modelling

FAST = {"hidden": 8, "layers": 1, "dropout": 0.1, "lr": 1e-2, "weight_decay": 0.0,
        "max_epochs": 6, "patience": 3, "batch_size": 32}


def toy_data(n=120, steps=5, feats=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, steps, feats)).astype("float32")
    y = (X[:, -1, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_direction_metrics_on_known_case():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.1, 0.8, 0.7, 0.3])  # predictions 1,0,0,1,1,0
    m = modelling.direction_metrics(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_naive_baseline_uses_training_majority():
    assert modelling.naive_majority(np.array([1, 1, 0])) == 1
    assert modelling.naive_majority(np.array([0, 0, 1])) == 0


def test_lstm_training_is_reproducible_and_respects_budget():
    X, y = toy_data()
    m1, i1 = modelling.train_lstm(X[:80], y[:80], X[80:], y[80:], FAST, seed=7)
    m2, i2 = modelling.train_lstm(X[:80], y[:80], X[80:], y[80:], FAST, seed=7)
    np.testing.assert_allclose(modelling.predict_proba(m1, X[80:]), modelling.predict_proba(m2, X[80:]))
    assert i1["epochs_run"] <= FAST["max_epochs"] and 1 <= i1["best_epoch"] <= i1["epochs_run"]
    m3, _ = modelling.train_lstm(X[:80], y[:80], X[80:], y[80:], FAST, seed=8)
    assert not np.allclose(modelling.predict_proba(m1, X[80:]), modelling.predict_proba(m3, X[80:]))


def test_lstm_output_is_probability_for_any_feature_count():
    for feats in (11, 16):  # Model A and Model B input widths
        X, y = toy_data(feats=feats)
        m, _ = modelling.train_lstm(X[:80], y[:80], X[80:], y[80:], FAST, seed=0)
        p = modelling.predict_proba(m, X[80:])
        assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()


def test_xgboost_trains_and_predicts():
    X, y = toy_data()
    Xf = X.reshape(len(X), -1)
    m, info = modelling.train_xgb(Xf[:80], y[:80], Xf[80:], y[80:], {"max_depth": 2, "learning_rate": 0.1}, seed=0)
    p = m.predict_proba(Xf[80:])[:, 1]
    assert p.shape == (40,) and info["best_iteration"] >= 0


def test_bootstrap_detects_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    perfect = y.astype(float)
    coin = rng.random(400)
    big = modelling.paired_bootstrap_diff(y, coin, perfect, n_boot=300)
    assert big["ci_excludes_zero"] and big["point"] > 0.3
    same = modelling.paired_bootstrap_diff(y, coin, coin, n_boot=300)
    assert same["point"] == 0 and not same["ci_excludes_zero"]


def test_verdict_wording():
    assert evaluate_final.verdict(0.0, {"ci_excludes_zero": False}) == "unchanged"
    assert evaluate_final.verdict(0.02, {"ci_excludes_zero": True}) == "improved"
    assert "includes 0" in evaluate_final.verdict(-0.02, {"ci_excludes_zero": False})


def test_evaluate_final_refuses_without_frozen_config(monkeypatch, tmp_path):
    monkeypatch.setattr(modelling, "MODEL_CONFIG_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["evaluate_final"])
    with pytest.raises(config.PipelineError):
        evaluate_final.main()


def test_evaluate_final_refuses_second_run_without_force(monkeypatch, tmp_path):
    frozen, final = tmp_path / "frozen.json", tmp_path / "final.json"
    frozen.write_text("{}")
    final.write_text("{}")
    monkeypatch.setattr(modelling, "MODEL_CONFIG_FILE", frozen)
    monkeypatch.setattr(evaluate_final, "FINAL", final)
    monkeypatch.setattr(sys, "argv", ["evaluate_final"])
    with pytest.raises(config.PipelineError, match="evaluated once"):
        evaluate_final.main()
