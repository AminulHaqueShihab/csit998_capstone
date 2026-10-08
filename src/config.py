"""Shared paths and constants for the data pipeline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

BINANCE_DIR = RAW / "binance" / "BTCUSDT" / "1d"
GDELT_DIR = RAW / "gdelt"

PRICE_CLEAN = PROCESSED / "btcusdt_1d_clean.csv"
NEWS_CLEAN = PROCESSED / "news_clean.csv.gz"
NEWS_SCORED = PROCESSED / "news_scored.csv.gz"
DAILY_SENTIMENT = PROCESSED / "daily_sentiment.csv"
MODELLING_TABLE = PROCESSED / "modelling_table_v1.csv"

SPLITS_DIR = PROCESSED / "splits"
SEQUENCES = SPLITS_DIR / "sequences.npz"
SCALER_A = SPLITS_DIR / "scaler_model_a.json"
SCALER_B = SPLITS_DIR / "scaler_model_b.json"
SPLIT_DEFINITION = SPLITS_DIR / "split_definition.json"

# Chronological split, fixed BEFORE any modelling (DS-02). Dates are the day t of the last input row.
# Test = Oct 2025 .. end of data (~1 year). Do not change after model design starts.
SPLIT_DATES = {
    "train": ("2021-01-01", "2024-12-31"),
    "validation": ("2025-01-01", "2025-09-30"),
    "test": ("2025-10-01", "2026-08-30"),
}
SEQ_LEN = 14  # days of history per LSTM sample; identical for Model A and Model B (ML-05)

ASSET = "BTC"
EXTREME_MOVE_PCT = 10.0  # |daily return| above this is audited, never auto-corrected
VADER_POS_THRESHOLD = 0.05  # standard VADER cut-offs for pos/neg/neutral
VADER_NEG_THRESHOLD = -0.05
MIN_HEADLINE_CHARS = 10
LANGDETECT_MIN_PROB = 0.95  # only reject as non-English when detector is this sure


class PipelineError(RuntimeError):
    """Raised when an input is missing or invalid; message names the failed stage."""


def require_file(path: Path, stage: str, hint: str) -> Path:
    if not Path(path).exists():
        raise PipelineError(f"[{stage}] missing input: {path}. {hint}")
    return Path(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report(name: str, payload: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
