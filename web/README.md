# Bitcoin direction demo (web)

Demo interface for the CSIT998 Group 16 capstone. It shows saved model predictions of whether Bitcoin closes higher
the next day, against the actual price. It loads saved results only (no retraining, no live data).


## What it shows
- **Date selector:** from/to dates and presets (test period, last 3 or 6 months, last year, all data).
- **Model selector:** naive baseline, XGBoost, LSTM A (price only), LSTM B (price + VADER), LSTM B (price + FinBERT).
- **Prediction set:** the frozen test split (Oct 2025 to Aug 2026) or walk-forward predictions (2023 to 2026, each
  model retrained every six months on earlier data only).
- **Price chart** with a marker on each day: triangle up/down = predicted up/not up, filled = right, outline = wrong.
  Hover for the close, the prediction and probability, and what actually happened.
- **Summary numbers** for the selected dates (accuracy with the "guessing" range, days, predicted-up and actual-up share).
- **News sentiment chart** (VADER or FinBERT) for the same dates, with days without news marked.
- **Table comparing all models** on exactly the same days.

## Stack
Next.js (App Router, TypeScript, static export) · Tailwind CSS · shadcn/ui Recharts. No server and no database: the page is built once from JSON files.

## Run
```bash
pnpm install
pnpm dev            # http://localhost:3000
pnpm build          # static site in ./out
pnpm lint
pnpm test           # scoring and date helpers (Node's built-in test runner)
```

## Data
`public/data/*.json` (about 450 KB) is produced by the Python pipeline in the repo root, not edited by hand:

```bash
python -m src.export_demo_data      # writes web/public/data/
```




