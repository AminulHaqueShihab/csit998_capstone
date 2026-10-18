"""Stage 7: final training and the single evaluation on the test split.

Run: python -m src.evaluate_final
"""
import argparse
import importlib.metadata as md
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import config, features, modelling
from .config import PipelineError

RESULTS = config.ROOT / "results"
MODELS = config.ROOT / "models"
FINAL = RESULTS / "final_results.json"
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def summarise(per_seed: list, ensemble: dict) -> dict:
    out = {}
    for m in METRICS:
        vals = np.array([p[m] for p in per_seed])
        out[f"{m}_mean"], out[f"{m}_std"], out[f"{m}_ensemble"] = float(vals.mean()), float(vals.std(ddof=0)), ensemble[m]
    return out


def verdict(delta: float, ci: dict) -> str:
    if abs(delta) < 1e-12:
        return "unchanged"
    direction = "improved" if delta > 0 else "worsened"
    return direction if ci["ci_excludes_zero"] else f"{direction} (point estimate only; 95% interval includes 0)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-run even though test results already exist")
    args = parser.parse_args()
    config.require_file(modelling.MODEL_CONFIG_FILE, "evaluate_final", "Run `python -m src.tune` first.")
    reruns = 0
    if FINAL.exists():
        if not args.force:
            raise PipelineError("[evaluate_final] test results already exist. The test set is evaluated once; "
                                "use --force only for a documented re-run.")
        reruns = json.loads(FINAL.read_text()).get("reruns", 0) + 1
    frozen = json.loads(modelling.MODEL_CONFIG_FILE.read_text())
    seeds, lp, xp = frozen["final_seeds"], frozen["lstm"], frozen["xgboost"]
    z = modelling.load_sequences()
    MODELS.mkdir(exist_ok=True)
    y_val, y_test = z["y_validation"], z["y_test"]
    t0 = time.time()

    probs = {}  # name -> {"validation": [per seed], "test": [per seed]}
    # --- naive majority class (training labels)
    majority = modelling.naive_majority(z["y_train"])
    probs["naive_majority"] = {s: [np.full(len(y), float(majority)) for _ in [0]] for s, y in
                               (("validation", y_val), ("test", y_test))}
    # --- LSTM Model A and B
    for name, key in (("lstm_a_price_only", "a"), ("lstm_b_price_plus_sentiment", "b")):
        probs[name] = {"validation": [], "test": []}
        for seed in seeds:
            model, info = modelling.train_lstm(z[f"X_{key}_train"], z["y_train"], z[f"X_{key}_validation"],
                                               y_val, lp, seed)
            torch.save(model.state_dict(), MODELS / f"{name}_seed{seed}.pt")
            probs[name]["validation"].append(modelling.predict_proba(model, z[f"X_{key}_validation"]))
            probs[name]["test"].append(modelling.predict_proba(model, z[f"X_{key}_test"]))
        print(f"[evaluate_final] {name} trained ({time.time() - t0:.0f}s)")
    # --- XGBoost (price) and extra price+sentiment variant
    for name, sentiment in (("xgboost_price", False), ("xgboost_price_plus_sentiment_extra", True)):
        Xtr, ytr, _ = modelling.load_tabular("train", sentiment)
        Xva, yva, _ = modelling.load_tabular("validation", sentiment)
        Xte, yte, dates_test = modelling.load_tabular("test", sentiment)
        assert (yte == y_test).all() and (yva == y_val).all()  # same samples as the LSTMs
        probs[name] = {"validation": [], "test": []}
        for seed in seeds:
            model, _ = modelling.train_xgb(Xtr, ytr, Xva, yva, xp, seed)
            probs[name]["validation"].append(model.predict_proba(Xva)[:, 1])
            probs[name]["test"].append(model.predict_proba(Xte)[:, 1])
    dates_test = pd.to_datetime(z["dates_test"])

    # --- metrics
    table, ensembles, per_seed_rows, confusion = {}, {"validation": {}, "test": {}}, [], {}
    for name, by_split in probs.items():
        for split, y in (("validation", y_val), ("test", y_test)):
            ensembles[split][name] = np.mean(by_split[split], axis=0)
            per_seed = [modelling.direction_metrics(y, p) for p in by_split[split]]
            if split == "test":
                for s, m in zip(seeds if len(per_seed) > 1 else [None], per_seed):
                    per_seed_rows.append({"model": name, "seed": s, **m})
                ens = modelling.direction_metrics(y, ensembles["test"][name])
                table[name] = {"n_seeds": len(per_seed), **summarise(per_seed, ens)}
                confusion[name] = {k: ens[k] for k in ("tn", "fp", "fn", "tp")}
    validation = {n: modelling.direction_metrics(y_val, ensembles["validation"][n]) for n in probs}

    # --- sentiment contribution: Model B - Model A (EV-03)
    a, b = "lstm_a_price_only", "lstm_b_price_plus_sentiment"
    a_seed = [modelling.direction_metrics(y_test, p) for p in probs[a]["test"]]
    b_seed = [modelling.direction_metrics(y_test, p) for p in probs[b]["test"]]
    contribution = {}
    boot = {m: modelling.paired_bootstrap_diff(y_test, ensembles["test"][a], ensembles["test"][b], metric=m)
            for m in ("accuracy", "balanced_accuracy")}
    for m in METRICS:
        paired = np.array([bm[m] - am[m] for am, bm in zip(a_seed, b_seed)])
        ens_delta = table[b][f"{m}_ensemble"] - table[a][f"{m}_ensemble"]
        entry = {"mean_paired_seed_delta": float(paired.mean()), "std_paired_seed_delta": float(paired.std(ddof=0)),
                 "ensemble_delta": float(ens_delta)}
        if m in boot:
            entry["bootstrap_ensemble"] = boot[m]
            entry["verdict"] = verdict(ens_delta, boot[m])
        else:
            entry["verdict"] = "improved" if ens_delta > 1e-12 else ("worsened" if ens_delta < -1e-12 else "unchanged")
        contribution[m] = entry

    # --- outputs
    pred = pd.DataFrame({"date": dates_test, "y_true": y_test})
    for n in probs:
        pred[f"proba_{n}"] = ensembles["test"][n]
    pred.to_csv(RESULTS / "test_predictions.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(RESULTS / "test_metrics_per_seed.csv", index=False)
    pd.DataFrame(table).T.rename_axis("model").to_csv(RESULTS / "model_comparison.csv")

    settings = {
        "evaluated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "threshold": modelling.THRESHOLD,
        "lstm_hyperparameters": lp, "xgboost_hyperparameters": xp, "seq_len": config.SEQ_LEN,
        "split_dates": config.SPLIT_DATES,
        "model_a_features": features.MODEL_PRICE_FEATURES,
        "model_b_features": features.MODEL_PRICE_FEATURES + features.MODEL_SENTIMENT_FEATURES,
        "modelling_table_sha256": config.sha256_file(config.MODELLING_TABLE),
        "sequences_sha256": config.sha256_file(config.SEQUENCES),
        "versions": {p: md.version(p) for p in ("torch", "xgboost", "scikit-learn", "numpy", "pandas")},
        "python": sys.version.split()[0],
        "model_files": f"models/<name>_seed<seed>.pt (LSTM state dicts)",
    }
    (RESULTS / "model_settings.json").write_text(json.dumps(settings, indent=2))
    FINAL.write_text(json.dumps({
        "reruns": reruns, "test_n": int(len(y_test)), "test_share_up": float(y_test.mean()),
        "comparison": table, "confusion_matrices_ensemble": confusion, "validation_ensemble": validation,
        "sentiment_contribution_B_minus_A": contribution,
        "notes": ["Test set evaluated once after hyperparameters were frozen on validation data.",
                  "'ensemble' = average probability over the 10 seeds; '_mean/_std' = across single-seed models.",
                  "Bootstrap interval resamples test days (paired) for the ensemble predictions of A and B.",
                  "Correlation is not causation: differences describe prediction performance only."],
    }, indent=2, default=float))

    print(f"\n{'model':<38}{'acc':>8}{'bal.acc':>9}{'prec':>7}{'rec':>7}{'f1':>7}   (mean over seeds, ± std)")
    for n, r in table.items():
        print(f"{n:<38}{r['accuracy_mean']:>8.3f}{r['balanced_accuracy_mean']:>9.3f}{r['precision_mean']:>7.3f}"
              f"{r['recall_mean']:>7.3f}{r['f1_mean']:>7.3f}   ± {r['accuracy_std']:.3f} acc")
    print("\nB - A:", {m: (round(c['ensemble_delta'], 4), c['verdict']) for m, c in contribution.items()
                      if m in ('accuracy', 'balanced_accuracy')})


if __name__ == "__main__":
    main()
