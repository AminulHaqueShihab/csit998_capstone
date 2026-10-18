"""Stage 10: repeat the main experiment with FinBERT instead of VADER (secondary result).

Run: python -m src.evaluate_finbert
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from . import config, features, make_splits, modelling
from .config import PipelineError

RESULTS = config.ROOT / "results"
MODELS = config.ROOT / "models"
OUT = RESULTS / "finbert_results.json"
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def add_finbert_inputs(t: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Join day-t FinBERT aggregates onto the modelling rows (same one-day lag rule as VADER)."""
    d = daily[["date", "finbert_mean", "finbert_pos_count", "finbert_neg_count", "article_count"]].rename(
        columns={"article_count": "fb_article_count"})
    t = t.merge(d, on="date", how="left", validate="one_to_one")
    if (t["fb_article_count"] != t["article_count"]).any():
        raise PipelineError("[evaluate_finbert] FinBERT and VADER daily article counts differ: different headlines")
    n = t["article_count"].where(t["article_count"] > 0)
    t["finbert_mean"] = t["finbert_mean"].fillna(0.0)
    t["finbert_pos_share"] = (t["finbert_pos_count"] / n).fillna(0.0)
    t["finbert_neg_share"] = (t["finbert_neg_count"] / n).fillna(0.0)
    return t


def summarise(per_seed, ens):
    return {**{f"{m}_mean": float(np.mean([p[m] for p in per_seed])) for m in METRICS},
            **{f"{m}_std": float(np.std([p[m] for p in per_seed])) for m in METRICS},
            **{f"{m}_ensemble": ens[m] for m in METRICS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for p, hint in ((modelling.MODEL_CONFIG_FILE, "tune"), (RESULTS / "final_results.json", "evaluate_final"),
                    (config.DAILY_FINBERT, "build_finbert_features")):
        config.require_file(p, "evaluate_finbert", f"Run `python -m src.{hint}` first.")
    if OUT.exists() and not args.force:
        raise PipelineError("[evaluate_finbert] results already exist; use --force only for a documented re-run")

    frozen = json.loads(modelling.MODEL_CONFIG_FILE.read_text())
    seeds, lp, xp = frozen["final_seeds"], frozen["lstm"], frozen["xgboost"]
    table = pd.read_csv(config.MODELLING_TABLE, parse_dates=["date", "target_date", "prediction_cutoff_utc"])
    daily = pd.read_csv(config.DAILY_FINBERT, parse_dates=["date"])
    t = add_finbert_inputs(make_splits.prepare_table(table), daily)

    cols = features.MODEL_PRICE_FEATURES + features.FINBERT_SENTIMENT_FEATURES
    X, scaler, meta = make_splits.windows_for(t, cols, config.SEQ_LEN)

    z = modelling.load_sequences()  # identical samples as the main experiment?
    for name in config.SPLIT_DATES:
        m = (meta["split"] == name).to_numpy()
        assert (meta.loc[m, "date"].dt.strftime("%Y-%m-%d").to_numpy() == z[f"dates_{name}"]).all()
        assert (meta.loc[m, "target_direction"].to_numpy() == z[f"y_{name}"]).all()
    y = {n: z[f"y_{n}"] for n in config.SPLIT_DATES}

    MODELS.mkdir(exist_ok=True)
    t0 = time.time()
    lstm = {"validation": [], "test": []}
    for seed in seeds:
        model, _ = modelling.train_lstm(X["train"], y["train"], X["validation"], y["validation"], lp, seed)
        torch.save(model.state_dict(), MODELS / f"lstm_b_finbert_seed{seed}.pt")
        for split in lstm:
            lstm[split].append(modelling.predict_proba(model, X[split]))
    print(f"[evaluate_finbert] LSTM trained ({time.time() - t0:.0f}s)")

    tab = {n: meta[meta["split"] == n][cols].to_numpy(dtype=float) for n in config.SPLIT_DATES}
    xgbp = {"validation": [], "test": []}
    for seed in seeds:
        model, _ = modelling.train_xgb(tab["train"], y["train"], tab["validation"], y["validation"], xp, seed)
        for split in xgbp:
            xgbp[split].append(model.predict_proba(tab[split])[:, 1])

    results = {}
    for name, probs in (("lstm_b_finbert", lstm), ("xgboost_finbert_extra", xgbp)):
        ens = {s: np.mean(probs[s], axis=0) for s in probs}
        results[name] = {
            "test": summarise([modelling.direction_metrics(y["test"], p) for p in probs["test"]],
                              modelling.direction_metrics(y["test"], ens["test"])),
            "validation_ensemble": modelling.direction_metrics(y["validation"], ens["validation"]),
            "validation_log_loss_ensemble": modelling.log_loss_value(y["validation"], ens["validation"]),
            "confusion_ensemble_test": {k: modelling.direction_metrics(y["test"], ens["test"])[k]
                                        for k in ("tn", "fp", "fn", "tp")},
        }
        results[name]["_ens_test"] = ens["test"]

    # compare with the main experiment's saved ensemble predictions (same test days)
    main_pred = pd.read_csv(RESULTS / "test_predictions.csv")
    assert (main_pred["y_true"].to_numpy() == y["test"]).all()
    a = main_pred["proba_lstm_a_price_only"].to_numpy()
    b_vader = main_pred["proba_lstm_b_price_plus_sentiment"].to_numpy()
    f = results["lstm_b_finbert"]["_ens_test"]
    comparisons = {}
    for label, other in (("finbert_B_minus_price_only_A", a), ("finbert_B_minus_vader_B", b_vader)):
        comparisons[label] = {m: modelling.paired_bootstrap_diff(y["test"], other, f, metric=m)
                              for m in ("accuracy", "balanced_accuracy")}

    pred = main_pred.copy()
    pred["proba_lstm_b_finbert"] = f
    pred["proba_xgboost_finbert_extra"] = results["xgboost_finbert_extra"]["_ens_test"]
    pred.to_csv(RESULTS / "finbert_test_predictions.csv", index=False)
    for r in results.values():
        r.pop("_ens_test")

    OUT.write_text(json.dumps({
        "reruns": 0, "seeds": seeds, "lstm_hyperparameters_frozen": lp, "xgboost_hyperparameters_frozen": xp,
        "features": cols, "finbert_model": config.FINBERT_MODEL,
        "results": results, "comparisons_ensemble_bootstrap": comparisons,
        "note": "Secondary, pre-declared experiment on the same test period as the main result; settings were not "
                "re-tuned for FinBERT. Differences inside the 95% interval are not evidence of improvement.",
    }, indent=2, default=float))

    print(f"\n{'model':<26}{'acc':>8}{'bal.acc':>9}{'f1':>7}   (test, mean over seeds)")
    for n, r in results.items():
        print(f"{n:<26}{r['test']['accuracy_mean']:>8.3f}{r['test']['balanced_accuracy_mean']:>9.3f}"
              f"{r['test']['f1_mean']:>7.3f}   ± {r['test']['accuracy_std']:.3f} acc")
    for label, c in comparisons.items():
        d = c["accuracy"]
        print(f"{label}: accuracy {d['point']:+.4f} (95% CI {d['ci95_low']:+.3f}..{d['ci95_high']:+.3f})")


if __name__ == "__main__":
    main()
