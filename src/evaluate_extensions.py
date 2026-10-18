"""Stage 11: other horizons and large-move training (exploratory, not the primary target).

Run: python -m src.evaluate_extensions   (once; --force to re-run)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config, evaluate_finbert, features, make_splits, modelling
from .config import PipelineError

RESULTS = config.ROOT / "results"
OUT = RESULTS / "extension_results.json"
TASKS = [
    {"name": "daily_direction", "horizon": 1, "min_move": 0.0},
    {"name": "3day_direction", "horizon": 3, "min_move": 0.0},
    {"name": "weekly_direction", "horizon": 7, "min_move": 0.0},
    {"name": "daily_large_moves_1pct", "horizon": 1, "min_move": 0.01},
    {"name": "daily_large_moves_2pct", "horizon": 1, "min_move": 0.02},
]
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
MODELS = ["naive_majority", "xgboost_price", "lstm_a_price_only", "lstm_b_vader", "lstm_b_finbert"]


def task_samples(meta: pd.DataFrame, close: pd.Series, horizon: int, min_move: float) -> pd.DataFrame:
    """Per-window labels and masks for one task. `meta` has date and split for each window (day t)."""
    d = meta["date"]
    cur = d.map(close)
    ret_h = d.add(pd.Timedelta(days=horizon)).map(close) / cur - 1
    ret_1 = d.add(pd.Timedelta(days=1)).map(close) / cur - 1
    split_end = meta["split"].map({n: pd.Timestamp(b) for n, (_, b) in config.SPLIT_DATES.items()})
    # purge: label's end date may be at most one day past the split (same slack as the main daily experiment)
    purge_ok = (d + pd.Timedelta(days=horizon - 1) <= split_end) & ret_h.notna()
    big = ret_1.abs() >= min_move if min_move > 0 else pd.Series(True, index=meta.index)
    train_like = meta["split"] != "test"
    use = purge_ok & (big | ~train_like)  # large-move filter applies to training/validation only
    return pd.DataFrame({"date": d, "split": meta["split"], "y": (ret_h > 0).astype(int), "ret_h": ret_h,
                         "purge_ok": purge_ok, "big": big.fillna(False), "use": use})


def chance_halfwidth(n: int, horizon: int) -> float:
    """95% half-width for an accuracy of a no-skill model, using n/horizon independent outcomes."""
    return float(1.96 * np.sqrt(0.25 / max(n / horizon, 1)))


def summarise(per_seed, ens):
    return {**{f"{m}_mean": float(np.mean([p[m] for p in per_seed])) for m in METRICS},
            **{f"{m}_std": float(np.std([p[m] for p in per_seed])) for m in METRICS},
            **{f"{m}_ensemble": ens[m] for m in METRICS}}


def run_task(task, meta, close, X, tab, lp, xp, seeds):
    s = task_samples(meta, close, task["horizon"], task["min_move"])
    in_split = {n: (meta["split"] == n).to_numpy() for n in config.SPLIT_DATES}
    use = {n: s["use"].to_numpy()[in_split[n]] for n in config.SPLIT_DATES}
    y = {n: s["y"].to_numpy()[in_split[n]][use[n]] for n in config.SPLIT_DATES}
    for n in ("train", "validation"):
        if len(np.unique(y[n])) < 2:
            raise PipelineError(f"[evaluate_extensions] {task['name']}: {n} has a single class")
    Xs = {k: {n: X[k][n][use[n]] for n in config.SPLIT_DATES} for k in X}
    Ts = {n: tab[n][use[n]] for n in config.SPLIT_DATES}
    test_big = s["big"].to_numpy()[in_split["test"]][use["test"]]
    dates_test = s["date"].to_numpy()[in_split["test"]][use["test"]]

    probs = {}
    majority = modelling.naive_majority(y["train"])
    probs["naive_majority"] = {"validation": [np.full(len(y["validation"]), float(majority))],
                               "test": [np.full(len(y["test"]), float(majority))]}
    for name, key in (("lstm_a_price_only", "a"), ("lstm_b_vader", "b"), ("lstm_b_finbert", "f")):
        probs[name] = {"validation": [], "test": []}
        for seed in seeds:
            model, _ = modelling.train_lstm(Xs[key]["train"], y["train"], Xs[key]["validation"], y["validation"],
                                            lp, seed)
            for split in ("validation", "test"):
                probs[name][split].append(modelling.predict_proba(model, Xs[key][split]))
    n_price = len(features.MODEL_PRICE_FEATURES)
    probs["xgboost_price"] = {"validation": [], "test": []}
    for seed in seeds:
        model, _ = modelling.train_xgb(Ts["train"][:, :n_price], y["train"], Ts["validation"][:, :n_price],
                                       y["validation"], xp, seed)
        for split in ("validation", "test"):
            probs["xgboost_price"][split].append(model.predict_proba(Ts[split][:, :n_price])[:, 1])

    h = task["horizon"]
    out = {"task": task, "n": {k: int(len(v)) for k, v in y.items()},
           "share_up": {k: round(float(v.mean()), 4) for k, v in y.items()},
           "chance_ci95_halfwidth_test": chance_halfwidth(len(y["test"]), h), "models": {}}
    ens_test = {}
    for name, p in probs.items():
        ens = {k: np.mean(v, axis=0) for k, v in p.items()}
        ens_test[name] = ens["test"]
        per_seed = [modelling.direction_metrics(y["test"], q) for q in p["test"]]
        entry = {"test_all_days": summarise(per_seed, modelling.direction_metrics(y["test"], ens["test"])),
                 "validation_accuracy_ensemble": modelling.direction_metrics(y["validation"], ens["validation"])[
                     "accuracy"],
                 "validation_log_loss_ensemble": modelling.log_loss_value(y["validation"], ens["validation"])}
        if task["min_move"] > 0 and test_big.any():
            sub = [modelling.direction_metrics(y["test"][test_big], q[test_big]) for q in p["test"]]
            entry["test_large_move_days_only"] = summarise(
                sub, modelling.direction_metrics(y["test"][test_big], ens["test"][test_big]))
        if h > 1:  # non-overlapping outcomes: every h-th test day
            sel = np.arange(0, len(y["test"]), h)
            entry["test_non_overlapping"] = {
                "n": int(len(sel)), **modelling.direction_metrics(y["test"][sel], ens["test"][sel])}
        out["models"][name] = entry
    if task["min_move"] > 0:
        out["test_large_move_days"] = {"n": int(test_big.sum()), "share_of_test": round(float(test_big.mean()), 4),
                                       "share_up": round(float(y["test"][test_big].mean()), 4)}
    out["sentiment_contribution"] = {
        f"{label}_minus_A": {m: modelling.paired_block_bootstrap_diff(
            y["test"], ens_test["lstm_a_price_only"], ens_test[key], block=h, metric=m)
            for m in ("accuracy", "balanced_accuracy")}
        for label, key in (("vader_B", "lstm_b_vader"), ("finbert_B", "lstm_b_finbert"))}
    pred = pd.DataFrame({"date": dates_test, "y_true": y["test"], "large_move_day": test_big})
    for name, v in ens_test.items():
        pred[f"proba_{name}"] = v
    pred.to_csv(RESULTS / f"extension_predictions_{task['name']}.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for p, hint in ((modelling.MODEL_CONFIG_FILE, "tune"), (RESULTS / "final_results.json", "evaluate_final"),
                    (config.DAILY_FINBERT, "build_finbert_features")):
        config.require_file(p, "evaluate_extensions", f"Run `python -m src.{hint}` first.")
    if OUT.exists() and not args.force:
        raise PipelineError("[evaluate_extensions] results already exist; use --force only for a documented re-run")

    frozen = json.loads(modelling.MODEL_CONFIG_FILE.read_text())
    seeds, lp, xp = frozen["final_seeds"], frozen["lstm"], frozen["xgboost"]
    table = pd.read_csv(config.MODELLING_TABLE, parse_dates=["date", "target_date", "prediction_cutoff_utc"])
    daily = pd.read_csv(config.DAILY_FINBERT, parse_dates=["date"])
    t = evaluate_finbert.add_finbert_inputs(make_splits.prepare_table(table), daily)
    price = pd.read_csv(config.PRICE_CLEAN, parse_dates=["date"])
    close = price.set_index("date")["close"]

    price_cols = features.MODEL_PRICE_FEATURES
    cols = {"a": price_cols, "b": price_cols + features.MODEL_SENTIMENT_FEATURES,
            "f": price_cols + features.FINBERT_SENTIMENT_FEATURES}
    X, meta = {}, None
    for key, c in cols.items():
        X[key], _, meta = make_splits.windows_for(t, c, config.SEQ_LEN)
    meta = meta.copy()
    tab = {n: meta[meta["split"] == n][price_cols].to_numpy(dtype=float) for n in config.SPLIT_DATES}

    results, t0 = {}, time.time()
    for task in TASKS:
        results[task["name"]] = run_task(task, meta, close, X, tab, lp, xp, seeds)
        print(f"[evaluate_extensions] {task['name']} done ({time.time() - t0:.0f}s)", flush=True)

    OUT.write_text(json.dumps({
        "status": "extension experiment: the approved primary target remains next-day direction",
        "seeds": seeds, "lstm_hyperparameters_frozen": lp, "xgboost_hyperparameters_frozen": xp,
        "multiple_comparisons_note": "5 tasks x 5 models were examined on the same test period; with this many "
                                     "comparisons one nominally 'significant' difference is expected by chance.",
        "tasks": results}, indent=2, default=float))

    rows = []
    for tname, r in results.items():
        for mname, m in r["models"].items():
            a = m["test_all_days"]
            rows.append({"task": tname, "horizon_days": r["task"]["horizon"], "min_move": r["task"]["min_move"],
                         "model": mname, "n_test": r["n"]["test"], "share_up_test": r["share_up"]["test"],
                         "accuracy_mean": a["accuracy_mean"], "accuracy_std": a["accuracy_std"],
                         "accuracy_ensemble": a["accuracy_ensemble"], "balanced_accuracy_mean": a["balanced_accuracy_mean"],
                         "f1_mean": a["f1_mean"], "chance_ci95_halfwidth": r["chance_ci95_halfwidth_test"],
                         "large_move_days_accuracy_mean": m.get("test_large_move_days_only", {}).get("accuracy_mean")})
    pd.DataFrame(rows).to_csv(RESULTS / "extension_comparison.csv", index=False)

    print(f"\n{'task':<24}{'n_test':>7}" + "".join(f"{m[:12]:>14}" for m in MODELS) + "   (test accuracy, all days)")
    for tname, r in results.items():
        print(f"{tname:<24}{r['n']['test']:>7}" + "".join(
            f"{r['models'][m]['test_all_days']['accuracy_mean']:>14.3f}" for m in MODELS))


if __name__ == "__main__":
    main()
