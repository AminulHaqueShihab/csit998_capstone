"""Stage 6: choose hyperparameters on the validation split only, then freeze them.

Run: python -m src.tune
"""
import itertools
import json
import time

import numpy as np
import pandas as pd

from . import config, modelling

RESULTS = config.ROOT / "results"
LSTM_BUDGET = {"max_epochs": 150, "patience": 15, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4}
LSTM_GRID = {"hidden": [16, 32, 64], "layers": [1, 2], "dropout": [0.2, 0.4]}
XGB_GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]}
TUNING_SEEDS = [0, 1, 2]
FINAL_SEEDS = list(range(100, 110))  # different from tuning seeds, fixed here before any test evaluation


def main():
    z = modelling.load_sequences()
    RESULTS.mkdir(exist_ok=True)
    rows = []

    t0 = time.time()
    for hidden, layers, dropout in itertools.product(LSTM_GRID["hidden"], LSTM_GRID["layers"], LSTM_GRID["dropout"]):
        params = {**LSTM_BUDGET, "hidden": hidden, "layers": layers, "dropout": dropout}
        for model_name, key in (("A", "a"), ("B", "b")):
            for seed in TUNING_SEEDS:
                model, info = modelling.train_lstm(z[f"X_{key}_train"], z["y_train"], z[f"X_{key}_validation"],
                                                   z["y_validation"], params, seed)
                proba = modelling.predict_proba(model, z[f"X_{key}_validation"])
                m = modelling.direction_metrics(z["y_validation"], proba)
                rows.append({"family": "lstm", "model": model_name, "hidden": hidden, "layers": layers,
                             "dropout": dropout, "seed": seed, "val_loss": info["best_val_loss"],
                             "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
                             "best_epoch": info["best_epoch"]})
        print(f"[tune] lstm hidden={hidden} layers={layers} dropout={dropout} done ({time.time() - t0:.0f}s)")

    Xtr, ytr, _ = modelling.load_tabular("train", with_sentiment=False)
    Xva, yva, _ = modelling.load_tabular("validation", with_sentiment=False)
    for depth, lr in itertools.product(XGB_GRID["max_depth"], XGB_GRID["learning_rate"]):
        params = {"max_depth": depth, "learning_rate": lr}
        model, info = modelling.train_xgb(Xtr, ytr, Xva, yva, params, seed=0)
        m = modelling.direction_metrics(yva, model.predict_proba(Xva)[:, 1])
        rows.append({"family": "xgboost", "model": "A", "max_depth": depth, "learning_rate": lr, "seed": 0,
                     "val_loss": info["best_val_loss"], "val_accuracy": m["accuracy"],
                     "val_balanced_accuracy": m["balanced_accuracy"], "best_epoch": info["best_iteration"]})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "tuning_results.csv", index=False)

    lstm = results[results["family"] == "lstm"]
    best_lstm = (lstm.groupby(["hidden", "layers", "dropout"])["val_loss"].mean().sort_values().head(1))
    h, l, d = best_lstm.index[0]
    xgb = results[results["family"] == "xgboost"].sort_values("val_loss").iloc[0]
    frozen = {
        "frozen_before_test_evaluation": True,
        "selection_rule": "LSTM: lowest mean validation loss over Models A and B and 3 seeds (one shared config). "
                          "XGBoost: lowest validation log loss on price features. Threshold fixed at 0.5.",
        "lstm": {**LSTM_BUDGET, "hidden": int(h), "layers": int(l), "dropout": float(d),
                 "mean_val_loss": float(best_lstm.iloc[0])},
        "xgboost": {"max_depth": int(xgb["max_depth"]), "learning_rate": float(xgb["learning_rate"]),
                    "val_loss": float(xgb["val_loss"])},
        "final_seeds": FINAL_SEEDS, "tuning_seeds": TUNING_SEEDS,
        "seq_len": config.SEQ_LEN, "split_dates": config.SPLIT_DATES,
        "val_loss_by_lstm_model": lstm[(lstm.hidden == h) & (lstm.layers == l) & (lstm.dropout == d)]
        .groupby("model")["val_loss"].mean().round(5).to_dict(),
    }
    (RESULTS / "frozen_config.json").write_text(json.dumps(frozen, indent=2))
    print("[tune] frozen config:", json.dumps({k: frozen[k] for k in ("lstm", "xgboost", "val_loss_by_lstm_model")}))


if __name__ == "__main__":
    main()
