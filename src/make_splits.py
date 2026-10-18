"""Stage 5: chronological split, training-only scaling and LSTM sequences.

Run: python -m src.make_splits
"""
import json

import numpy as np
import pandas as pd

from . import config, features
from .config import PipelineError


def split_of(dates: pd.Series) -> pd.Series:
    """Label each date train / validation / test. Fails if any date is uncovered or splits overlap."""
    bounds = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in config.SPLIT_DATES.items()}
    names = list(bounds)
    for prev, nxt in zip(names, names[1:]):
        if bounds[nxt][0] != bounds[prev][1] + pd.Timedelta(days=1):
            raise PipelineError(f"[make_splits] {prev} and {nxt} are not back-to-back or they overlap")
    label = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, (a, b) in bounds.items():
        label[(dates >= a) & (dates <= b)] = name
    if label.isna().any():
        raise PipelineError("[make_splits] some dates fall outside the configured split ranges: "
                            f"{dates[label.isna()].min().date()}..{dates[label.isna()].max().date()}")
    return label


def fit_scaler(train: pd.DataFrame, cols) -> dict:
    """Mean/std from the training rows only. Binary columns pass through (mean 0, std 1)."""
    scaler = {"columns": list(cols), "mean": {}, "std": {}}
    for c in cols:
        if c in features.BINARY_FEATURES:
            scaler["mean"][c], scaler["std"][c] = 0.0, 1.0
        else:
            std = float(train[c].std(ddof=0))
            scaler["mean"][c], scaler["std"][c] = float(train[c].mean()), (std if std > 0 else 1.0)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    cols = scaler["columns"]
    mean = np.array([scaler["mean"][c] for c in cols])
    std = np.array([scaler["std"][c] for c in cols])
    return ((df[cols].to_numpy(dtype=float) - mean) / std).astype("float32")


def make_windows(values: np.ndarray, seq_len: int):
    """Windows of `seq_len` consecutive rows; returns (X, end_index). Needs date-sorted, gap-free rows."""
    n = len(values)
    if n < seq_len:
        raise PipelineError(f"[make_splits] only {n} rows, need at least {seq_len}")
    end = np.arange(seq_len - 1, n)
    X = np.stack([values[i - seq_len + 1: i + 1] for i in end])
    return X, end


def windows_for(t: pd.DataFrame, cols, seq_len: int):
    """Scaler (train rows only) + windows for one feature set. Returns (X per split, scaler, window meta)."""
    scaler = fit_scaler(t[t["split"] == "train"], cols)
    X_all, end = make_windows(apply_scaler(t, scaler), seq_len)
    meta = t.iloc[end].reset_index(drop=True)  # day t, split and target for each window
    by_split = {name: X_all[(meta["split"] == name).to_numpy()] for name in config.SPLIT_DATES}
    return by_split, scaler, meta


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.sort_values("date").reset_index(drop=True)
    gaps = table["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise PipelineError("[make_splits] modelling table has missing calendar days; windows would span a gap")
    t = features.add_model_inputs(table)
    t["split"] = split_of(t["date"])
    return t


def build(table: pd.DataFrame, seq_len: int = config.SEQ_LEN):
    t = prepare_table(table)
    cols_a = features.MODEL_PRICE_FEATURES
    cols_b = features.MODEL_PRICE_FEATURES + features.MODEL_SENTIMENT_FEATURES
    xa, scaler_a, meta = windows_for(t, cols_a, seq_len)
    xb, scaler_b, _ = windows_for(t, cols_b, seq_len)

    arrays, tabular = {}, {}
    for name in config.SPLIT_DATES:
        m = (meta["split"] == name).to_numpy()
        arrays[f"X_a_{name}"], arrays[f"X_b_{name}"] = xa[name], xb[name]
        arrays[f"y_{name}"] = meta.loc[m, "target_direction"].to_numpy(dtype="int64")
        arrays[f"dates_{name}"] = meta.loc[m, "date"].dt.strftime("%Y-%m-%d").to_numpy().astype("U10")
        tabular[name] = meta[m]
    arrays["features_a"] = np.array(cols_a, dtype="U32")
    arrays["features_b"] = np.array(cols_b, dtype="U32")
    return arrays, tabular, scaler_a, scaler_b, t


def main():
    config.require_file(config.MODELLING_TABLE, "make_splits", "Run `python -m src.build_modelling_table` first.")
    table = pd.read_csv(config.MODELLING_TABLE, parse_dates=["date", "target_date", "prediction_cutoff_utc"])
    arrays, tabular, scaler_a, scaler_b, t = build(table)

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.SEQUENCES, **arrays)
    for path, sc in ((config.SCALER_A, scaler_a), (config.SCALER_B, scaler_b)):
        sc["fitted_on"] = {"split": "train", "rows": int((t["split"] == "train").sum()),
                           "date_min": str(t.loc[t["split"] == "train", "date"].min().date()),
                           "date_max": str(t.loc[t["split"] == "train", "date"].max().date())}
        path.write_text(json.dumps(sc, indent=2))
    for name, df in tabular.items():  # same dates for naive / XGBoost / LSTM, unscaled
        cols = ["date", "split"] + features.MODEL_PRICE_FEATURES + features.MODEL_SENTIMENT_FEATURES + [
            "target_direction"]
        df[cols].to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
    config.SPLIT_DEFINITION.write_text(json.dumps({
        "frozen": "2026-10-08, before any model was trained",
        "split_dates_utc": config.SPLIT_DATES, "seq_len_days": config.SEQ_LEN,
        "sample_definition": "sample for day t = features of days t-SEQ_LEN+1..t; target = Close(t+1) > Close(t)",
        "note": "Test results must not be used for feature selection, thresholds, scaling, tuning or early stopping.",
    }, indent=2))

    def balance(y):
        return {"n": int(len(y)), "up": int(y.sum()), "share_up": round(float(y.mean()), 4)}

    report = {
        "seq_len": config.SEQ_LEN,
        "splits": {n: {"first_day": str(arrays[f"dates_{n}"][0]), "last_day": str(arrays[f"dates_{n}"][-1]),
                       **balance(arrays[f"y_{n}"]),
                       "days_without_news": int(tabular[n]["no_news_flag"].sum())}
                   for n in config.SPLIT_DATES},
        "shapes": {k: list(v.shape) for k, v in arrays.items() if k.startswith("X_")},
        "model_a_features": features.MODEL_PRICE_FEATURES,
        "model_b_features": features.MODEL_PRICE_FEATURES + features.MODEL_SENTIMENT_FEATURES,
        "scaler": "z-score, mean/std from training rows only; binary flag unscaled; saved in scaler_model_*.json",
        "train_rows_dropped_for_lookback": config.SEQ_LEN - 1,
        "sequences_sha256": config.sha256_file(config.SEQUENCES),
    }
    config.write_report("split_report.json", report)
    for n, s in report["splits"].items():
        print(f"[make_splits] {n:<10} {s['first_day']}..{s['last_day']}  n={s['n']}  share_up={s['share_up']}")
    print(f"[make_splits] X_a {report['shapes']['X_a_train']}  X_b {report['shapes']['X_b_train']} (train)")


if __name__ == "__main__":
    main()
