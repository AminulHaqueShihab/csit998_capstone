# Cryptocurrency Price-Direction Prediction (CSIT998 Group 16)

Does adding media sentiment improve next-day Bitcoin price-direction prediction compared with
price-only LSTM models? Target: 1 if Close(t+1) > Close(t) for the next UTC day, else 0.

**Live demo:** https://crypto-price-prediction-demo.vercel.app/ (saved predictions against the actual price; not financial advice)


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

| 5. Split, scaling, sequences | `python -m src.make_splits` | modelling table | `data/processed/splits/` (`sequences.npz`, `train/validation/test.csv`, `scaler_model_a/b.json`, `split_definition.json`), `reports/split_report.json` |

Split (fixed before modelling): train 2021-01 to 2024-12, validation 2025-01 to 2025-09, test 2025-10 to 2026-08.
Each sample is 14 days of features ending on day t, predicting day t+1. Scalers are fitted on training rows only.
Model A uses price features; Model B uses the same price features plus VADER sentiment features.

Tests (leakage controls, target construction, cleaning rules): `pytest -q`

Missing files or columns stop a stage with a message naming the stage and the fix.

## Modelling (run after the data stages)

| Stage | Command | Output |
|---|---|---|
| 6. Tune on validation only, freeze settings | `python -m src.tune` | `results/tuning_results.csv`, `results/frozen_config.json` |
| 7. Train 10 seeds per model, evaluate on test once | `python -m src.evaluate_final` | `results/model_comparison.csv`, `results/final_results.json`, `results/test_predictions.csv`, `results/model_settings.json`, `models/*.pt` |
| 8-10. Optional FinBERT experiment | `python -m src.score_finbert`, `python -m src.build_finbert_features`, `python -m src.evaluate_finbert` | `data/processed/daily_finbert.csv`, `results/finbert_results.json` |
| 11. Extension: 3-day / weekly direction and large-move training (not the approved primary target) | `python -m src.evaluate_extensions` | `results/extension_results.json`, `results/extension_comparison.csv` |
| 12. Lag and feature analysis (EX-01) | `python -m src.lag_analysis` | `results/lag_analysis/`, `docs/lag_analysis.md` |
| 13. Walk-forward evaluation (8 expanding-window folds, 2023-2026) | `python -m src.walk_forward` | `results/walk_forward/` |
| 14. Export data for the demo web app | `python -m src.export_demo_data` | `web/public/data/` (price, sentiment and saved predictions; no headline text) |
| 15. Report figures and the LaTeX report | `python -m src.make_report_figures`, then `cd report && latexmk -pdf main.tex` | `report/figures/*.pdf`, `report/main.pdf` |

`evaluate_final` refuses to run without frozen settings and refuses a second run unless `--force` is given (the test set is evaluated once).

macOS note: XGBoost needs OpenMP (`brew install libomp`). If Python crashes (segfault) when PyTorch and XGBoost are used together,
make PyTorch use the same OpenMP library: replace `<venv>/lib/python3.13/site-packages/torch/lib/libomp.dylib` with a symlink to
`/opt/homebrew/opt/libomp/lib/libomp.dylib`.

## Layout

```
data/raw/        read-only source data (Binance zips; GDELT csv are git-ignored, see docs/data_sources.md)
data/processed/  cleaned tables and the modelling table
src/             one module per pipeline stage + features.py
web/             Next.js demo interface, deployed at https://crypto-price-prediction-demo.vercel.app/ (see web/README.md)
report/          LaTeX source of the final report (main.tex, chapters/, figures/)
tests/           pytest checks
reports/         quality, audit and summary JSON/CSV produced by each stage
docs/            data sources, data dictionary
```


