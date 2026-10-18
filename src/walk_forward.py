"""Stage 13: expanding-window walk-forward evaluation over eight test folds.

Run: python -m src.walk_forward   (once; --force to re-run)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config, evaluate_finbert, features, make_splits, modelling
from .config import PipelineError

OUT_DIR = config.ROOT / "results" / "walk_forward"
FOLDS = [("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"), ("2024-01-01", "2024-06-30"),
         ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"),
         ("2026-01-01", "2026-06-30"), ("2026-07-01", "2026-08-30")]
VAL_DAYS = 90
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
MODELS = ["naive_majority", "xgboost_price", "lstm_a_price_only", "lstm_b_vader", "lstm_b_finbert"]


def fold_data(t: pd.DataFrame, cols, seq_len: int, fold_start, fold_end, val_days: int = VAL_DAYS):
    """Scaled windows for one fold. Scaler uses training rows only; windows end on day t."""
    start, end = pd.Timestamp(fold_start), pd.Timestamp(fold_end)
    pool_end = start - pd.Timedelta(days=1)
    val_start = pool_end - pd.Timedelta(days=val_days - 1)
    scaler = make_splits.fit_scaler(t[t["date"] < val_start], cols)
    X_all, idx = make_splits.make_windows(make_splits.apply_scaler(t, scaler), seq_len)
    meta = t.iloc[idx].reset_index(drop=True)
    d = meta["date"]
    masks = {"train": (d < val_start).to_numpy(), "validation": ((d >= val_start) & (d <= pool_end)).to_numpy(),
             "test": ((d >= start) & (d <= end)).to_numpy()}
    if not masks["train"].any() or not masks["test"].any():
        raise PipelineError(f"[walk_forward] fold {fold_start}..{fold_end} has no training or test samples")
    out = {"scaler": scaler}
    for name, m in masks.items():
        out[f"X_{name}"] = X_all[m]
        out[f"y_{name}"] = meta.loc[m, "target_direction"].to_numpy(dtype="int64")
        out[f"dates_{name}"] = d[m].to_numpy()
        out[f"meta_{name}"] = meta[m]
    return out


def summarise(per_seed, ens):
    return {**{f"{m}_mean": float(np.mean([p[m] for p in per_seed])) for m in METRICS},
            **{f"{m}_std": float(np.std([p[m] for p in per_seed])) for m in METRICS},
            **{f"{m}_ensemble": ens[m] for m in METRICS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_json = OUT_DIR / "walk_forward_results.json"
    for p, hint in ((modelling.MODEL_CONFIG_FILE, "tune"), (config.DAILY_FINBERT, "build_finbert_features")):
        config.require_file(p, "walk_forward", f"Run `python -m src.{hint}` first.")
    if out_json.exists() and not args.force:
        raise PipelineError("[walk_forward] results already exist; use --force only for a documented re-run")

    frozen = json.loads(modelling.MODEL_CONFIG_FILE.read_text())
    seeds, lp, xp = frozen["final_seeds"], frozen["lstm"], frozen["xgboost"]
    table = pd.read_csv(config.MODELLING_TABLE, parse_dates=["date", "target_date", "prediction_cutoff_utc"])
    daily = pd.read_csv(config.DAILY_FINBERT, parse_dates=["date"])
    t = evaluate_finbert.add_finbert_inputs(make_splits.prepare_table(table), daily)
    price_cols = features.MODEL_PRICE_FEATURES
    sets = {"a": price_cols, "b": price_cols + features.MODEL_SENTIMENT_FEATURES,
            "f": price_cols + features.FINBERT_SENTIMENT_FEATURES}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_out, pooled_y, pooled_dates, pooled_fold = [], [], [], []
    pooled = {m: [[] for _ in seeds] for m in MODELS}  # model -> per-seed list of per-fold probabilities
    t0 = time.time()
    for fi, (fs, fe) in enumerate(FOLDS, 1):
        D = {k: fold_data(t, cols, config.SEQ_LEN, fs, fe) for k, cols in sets.items()}
        y = {n: D["a"][f"y_{n}"] for n in ("train", "validation", "test")}
        for k in ("b", "f"):  # identical samples across feature sets
            assert all((D[k][f"y_{n}"] == y[n]).all() for n in y)
        probs = {}
        majority = modelling.naive_majority(y["train"])
        probs["naive_majority"] = [np.full(len(y["test"]), float(majority))]
        for name, key in (("lstm_a_price_only", "a"), ("lstm_b_vader", "b"), ("lstm_b_finbert", "f")):
            probs[name] = []
            for seed in seeds:
                model, _ = modelling.train_lstm(D[key]["X_train"], y["train"], D[key]["X_validation"],
                                                y["validation"], lp, seed)
                probs[name].append(modelling.predict_proba(model, D[key]["X_test"]))
        tab = {n: D["a"][f"meta_{n}"][price_cols].to_numpy(dtype=float) for n in ("train", "validation", "test")}
        probs["xgboost_price"] = []
        for seed in seeds:
            model, _ = modelling.train_xgb(tab["train"], y["train"], tab["validation"], y["validation"], xp, seed)
            probs["xgboost_price"].append(model.predict_proba(tab["test"])[:, 1])

        fold_res = {"fold": fi, "test_start": fs, "test_end": fe, "n_train": int(len(y["train"])),
                    "n_validation": int(len(y["validation"])), "n_test": int(len(y["test"])),
                    "share_up_test": round(float(y["test"].mean()), 4), "models": {}}
        for name in MODELS:
            ens = np.mean(probs[name], axis=0)
            per_seed = [modelling.direction_metrics(y["test"], p) for p in probs[name]]
            fold_res["models"][name] = summarise(per_seed, modelling.direction_metrics(y["test"], ens))
            for si in range(len(seeds)):
                pooled[name][si].append(probs[name][min(si, len(probs[name]) - 1)])
        folds_out.append(fold_res)
        pooled_y.append(y["test"])
        pooled_dates.append(D["a"]["dates_test"])
        pooled_fold.append(np.full(len(y["test"]), fi))
        print(f"[walk_forward] fold {fi}/{len(FOLDS)} {fs}..{fe} n_test={len(y['test'])} ({time.time() - t0:.0f}s)",
              flush=True)

    y_all = np.concatenate(pooled_y)
    ens_all, pooled_res = {}, {}
    for name in MODELS:
        per_seed_all = [np.concatenate(pooled[name][si]) for si in range(len(seeds))]
        ens_all[name] = np.mean(per_seed_all, axis=0)
        per_seed = [modelling.direction_metrics(y_all, p) for p in per_seed_all]
        ens_m = modelling.direction_metrics(y_all, ens_all[name])
        pooled_res[name] = {**summarise(per_seed, ens_m),
                            "confusion_ensemble": {k: ens_m[k] for k in ("tn", "fp", "fn", "tp")}}
    n = len(y_all)
    contribution = {}
    for label, key in (("vader_B_minus_A", "lstm_b_vader"), ("finbert_B_minus_A", "lstm_b_finbert")):
        contribution[label] = {m: modelling.paired_block_bootstrap_diff(
            y_all, ens_all["lstm_a_price_only"], ens_all[key], block=7, metric=m)
            for m in ("accuracy", "balanced_accuracy")}
    fold_wins = {}
    for name in ("xgboost_price", "lstm_a_price_only", "lstm_b_vader", "lstm_b_finbert"):
        fold_wins[name] = {
            "folds_ensemble_accuracy_above_naive": int(sum(
                f["models"][name]["accuracy_ensemble"] > f["models"]["naive_majority"]["accuracy_ensemble"]
                for f in folds_out)),
            "folds_ensemble_accuracy_above_50pct": int(sum(f["models"][name]["accuracy_ensemble"] > 0.5
                                                           for f in folds_out))}
    fold_wins["folds_vader_B_beats_A"] = int(sum(
        f["models"]["lstm_b_vader"]["accuracy_ensemble"] > f["models"]["lstm_a_price_only"]["accuracy_ensemble"]
        for f in folds_out))
    fold_wins["folds_finbert_B_beats_A"] = int(sum(
        f["models"]["lstm_b_finbert"]["accuracy_ensemble"] > f["models"]["lstm_a_price_only"]["accuracy_ensemble"]
        for f in folds_out))

    out_json.write_text(json.dumps({
        "design": {"folds": FOLDS, "validation_days": VAL_DAYS, "seeds": seeds, "seq_len": config.SEQ_LEN,
                   "lstm_hyperparameters_frozen": lp, "xgboost_hyperparameters_frozen": xp},
        "pooled_n_test": int(n), "pooled_share_up": float(y_all.mean()),
        "pooled_chance_ci95_halfwidth": float(1.96 * np.sqrt(0.25 / n)),
        "pooled": pooled_res, "sentiment_contribution_pooled": contribution, "fold_counts": fold_wins,
        "folds": folds_out,
        "caveat": "Frozen settings were tuned on 2025-01..2025-09 validation data that lies inside later folds' test "
                  "windows; expected effect is negligible. The primary result is the single frozen test split.",
    }, indent=2, default=float))

    pred = pd.DataFrame({"date": np.concatenate(pooled_dates), "fold": np.concatenate(pooled_fold), "y_true": y_all})
    for name in MODELS:
        pred[f"proba_{name}"] = ens_all[name]
    pred.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    rows = [{"fold": f["fold"], "test_start": f["test_start"], "test_end": f["test_end"], "n_test": f["n_test"],
             "share_up_test": f["share_up_test"], "model": m, "accuracy_mean": r["accuracy_mean"],
             "accuracy_std": r["accuracy_std"], "accuracy_ensemble": r["accuracy_ensemble"],
             "balanced_accuracy_mean": r["balanced_accuracy_mean"]} for f in folds_out
            for m, r in f["models"].items()]
    pd.DataFrame(rows).to_csv(OUT_DIR / "walk_forward_by_fold.csv", index=False)

    print(f"\npooled over {n} test days (chance band ±{100 * 1.96 * np.sqrt(0.25 / n):.1f} pts), mean over seeds")
    print(f"{'model':<20}{'acc':>8}{'bal.acc':>9}{'f1':>7}")
    for m in MODELS:
        r = pooled_res[m]
        print(f"{m:<20}{r['accuracy_mean']:>8.3f}{r['balanced_accuracy_mean']:>9.3f}{r['f1_mean']:>7.3f}")
    for label, c in contribution.items():
        d = c["accuracy"]
        print(f"{label}: accuracy {d['point']:+.4f} (95% CI {d['ci95_low']:+.3f}..{d['ci95_high']:+.3f})")


if __name__ == "__main__":
    main()
