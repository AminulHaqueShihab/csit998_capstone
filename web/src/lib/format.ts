export const pct = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? "n/a" : `${(x * 100).toFixed(digits)}%`;

export const usd = (x: number) => `$${x.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

export const MODEL_LABELS = {
  naive: "Naive baseline",
  xgboost: "XGBoost (price)",
  lstmA: "LSTM A: price only",
  lstmBVader: "LSTM B: price + VADER",
  lstmBFinbert: "LSTM B: price + FinBERT",
} as const;

export const MODEL_SHORT = {
  naive: "Naive",
  xgboost: "XGBoost",
  lstmA: "LSTM A (price)",
  lstmBVader: "LSTM B (VADER)",
  lstmBFinbert: "LSTM B (FinBERT)",
} as const;
