# Cryptocurrency Price-Direction Prediction (CSIT998 Group 16)

Does adding media sentiment improve next-day Bitcoin price-direction prediction compared with
price-only LSTM models? Target: 1 if Close(t+1) > Close(t) for the next UTC day, else 0.
Research use only; outputs are experimental and **not financial advice**.

Requirements: `Requirements Draft_01-09-26.pdf`. Proposal: `CSIT998_Group16_Project Proposal_latex.pdf`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(If this folder is inside iCloud-synced Documents, create the virtual environment outside it, e.g.
`~/.venvs/crypto-price-prediction`, to avoid slow file downloads.)

## Data preparation (stages run independently, in this order)

| Stage | Command | Input | Output |
|---|---|---|---|
| 1. Price | `python -m src.prepare_price` | `data/raw/binance/BTCUSDT/1d/*.zip` | `data/processed/btcusdt_1d_clean.csv`, `reports/price_*` |
| 2. News | `python -m src.prepare_news` | `data/raw/gdelt/gdelt_btc_news_*_raw.csv` | `data/processed/news_clean.csv.gz`, `reports/news_quality_report.json` |
| 3. Sentiment | `python -m src.build_sentiment` | stage 1 + 2 outputs | `data/processed/daily_sentiment.csv`, `reports/sentiment_report.json` |
| 4. Modelling table | `python -m src.build_modelling_table` | stage 1 + 3 outputs | `data/processed/modelling_table_v1.csv`, `reports/modelling_table_report.json` |

Tests (leakage controls, target construction, cleaning rules): `pytest -q`

Missing files or columns stop a stage with a message naming the stage and the fix.

## Layout

```
data/raw/        read-only source data (Binance zips; GDELT csv are git-ignored, see docs/data_sources.md)
data/processed/  cleaned tables and the modelling table
src/             one module per pipeline stage + features.py
tests/           pytest checks
reports/         quality, audit and summary JSON/CSV produced by each stage
docs/            data sources, data dictionary
```

## Key design rules

- All timestamps UTC. Prediction cutoff is 23:59:59 UTC on day t; day-t news and prices predict day t+1.
- Features use rows <= t only (tested). Scaling parameters are fitted on training data only (later stage).
- No imputation of prices; days without news are explicit (`no_news_flag`), never invented.
- Raw data is never modified. See `docs/data_sources.md` for provenance and limits.
