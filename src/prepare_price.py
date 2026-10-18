"""Stage 1: verify, parse and validate Binance BTCUSDT daily klines.

Run: python -m src.prepare_price
"""
import io
import zipfile

import pandas as pd

from . import config
from .config import PipelineError

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]
NUMERIC = ["open", "high", "low", "close", "volume", "quote_asset_volume",
           "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]


def verify_checksums(zip_paths):
    """Compare every zip with its .CHECKSUM file (sha256). Fail loudly on mismatch."""
    bad = []
    for z in zip_paths:
        checksum_file = z.with_name(z.name + ".CHECKSUM")
        if not checksum_file.exists():
            bad.append(f"{z.name}: no .CHECKSUM file")
            continue
        expected = checksum_file.read_text().split()[0].lower()
        if config.sha256_file(z) != expected:
            bad.append(f"{z.name}: sha256 mismatch")
    if bad:
        raise PipelineError("[prepare_price] checksum verification failed: " + "; ".join(bad))


def _epoch_to_utc(series: pd.Series) -> pd.Series:
    """Binance spot files use ms until 2024 and microseconds from 2025: detect per value."""
    values = pd.to_numeric(series, errors="coerce")
    unit_us = values > 1e14
    ms = pd.to_datetime(values.where(~unit_us), unit="ms", utc=True, errors="coerce")
    us = pd.to_datetime(values.where(unit_us), unit="us", utc=True, errors="coerce")
    return ms.fillna(us)


def load_raw(zip_paths) -> pd.DataFrame:
    frames = []
    for z in zip_paths:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    df = pd.read_csv(io.TextIOWrapper(f), header=None, names=KLINE_COLS, dtype=str)
                df["source_file"] = z.name
                frames.append(df)
    if not frames:
        raise PipelineError(f"[prepare_price] no kline csv found in {config.BINANCE_DIR}")
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame):
    raw = raw.copy()
    raw["open_time_utc"] = _epoch_to_utc(raw["open_time"])
    raw["close_time_utc"] = _epoch_to_utc(raw["close_time"])
    for c in NUMERIC:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    rejected = []

    def reject(mask, reason):
        if mask.any():
            part = raw[mask].copy()
            part["reason"] = reason
            rejected.append(part)
        return raw[~mask]

    n_raw = len(raw)
    raw = reject(raw["open_time_utc"].isna() | raw["close_time_utc"].isna(), "invalid_timestamp")
    raw = reject(raw[NUMERIC].isna().any(axis=1), "non_numeric_value")
    raw = reject(raw["volume"] < 0, "negative_volume")
    raw = reject((raw[["open", "high", "low", "close"]] <= 0).any(axis=1), "non_positive_price")
    raw = reject(
        (raw["low"] > raw[["open", "close", "high"]].min(axis=1))
        | (raw["high"] < raw[["open", "close", "low"]].max(axis=1)),
        "ohlc_logic_violation",
    )
    # daily bar must open at 00:00:00 UTC
    raw = reject(raw["open_time_utc"] != raw["open_time_utc"].dt.floor("D"), "bar_not_utc_midnight")

    raw["date"] = raw["open_time_utc"].dt.floor("D").dt.tz_localize(None)
    n_dups = int(raw.duplicated("date", keep="first").sum())
    raw = raw.sort_values(["date", "source_file"]).drop_duplicates("date", keep="first")

    raw["asset"] = config.ASSET
    raw["daily_return_pct"] = raw["close"].pct_change() * 100
    raw["extreme_move_flag"] = raw["daily_return_pct"].abs() > config.EXTREME_MOVE_PCT
    out = raw[[
        "date", "open_time_utc", "close_time_utc", "asset", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
        "extreme_move_flag",
    ]].reset_index(drop=True)
    rejected_df = pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame()
    return out, rejected_df, {"raw_rows": n_raw, "duplicate_dates_removed": n_dups}


def main():
    zips = sorted(config.BINANCE_DIR.glob("BTCUSDT-1d-*.zip"))
    if not zips:
        raise PipelineError(f"[prepare_price] no zips in {config.BINANCE_DIR}. Copy Binance klines into data/raw/binance first.")
    verify_checksums(zips)
    out, rejected, counts = clean(load_raw(zips))

    full = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    missing = [d.strftime("%Y-%m-%d") for d in full.difference(out["date"])]

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PRICE_CLEAN, index=False)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    if len(rejected):
        rejected.to_csv(config.REPORTS / "price_rejected.csv", index=False)
    extreme = out[out["extreme_move_flag"]].copy()
    extreme["daily_return_pct"] = out["close"].pct_change().mul(100)[extreme.index]
    extreme[["date", "open", "close", "daily_return_pct"]].to_csv(
        config.REPORTS / "price_extreme_moves.csv", index=False)

    report = {
        "zips_verified": len(zips),
        "checksums": "all sha256 verified",
        **counts,
        "rows_rejected": 0 if rejected.empty else len(rejected),
        "rejection_reasons": {} if rejected.empty else rejected["reason"].value_counts().to_dict(),
        "rows_clean": len(out),
        "date_min": out["date"].min(), "date_max": out["date"].max(),
        "missing_dates": missing,
        "extreme_moves_flagged": int(out["extreme_move_flag"].sum()),
        "extreme_move_policy": "Retained as genuine market moves; audit list in reports/price_extreme_moves.csv. "
                               "Only confirmed source errors may be corrected, with an audit record.",
        "output": str(config.PRICE_CLEAN.relative_to(config.ROOT)),
        "output_sha256": config.sha256_file(config.PRICE_CLEAN),
    }
    config.write_report("price_quality_report.json", report)
    print(f"[prepare_price] {len(out)} rows, {out['date'].min().date()} .. {out['date'].max().date()}, "
          f"missing dates: {len(missing)}, rejected: {report['rows_rejected']}, extreme: {report['extreme_moves_flagged']}")


if __name__ == "__main__":
    main()
