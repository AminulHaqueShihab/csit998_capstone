// Shapes of the JSON files exported by the Python pipeline (src/export_demo_data.py).

export type ModelKey = "naive" | "xgboost" | "lstmA" | "lstmBVader" | "lstmBFinbert";

export interface PriceRow {
  date: string;
  close: number;
  vader: number | null;
  finbert: number | null;
  articles: number;
  noNews: boolean;
}

export interface PredRow extends Record<ModelKey, number> {
  date: string;
  up: 0 | 1; // did the NEXT day close higher than this day?
  close: number;
  closeNext: number | null;
}

export interface PredSet {
  label: string;
  from: string;
  to: string;
  rows: PredRow[];
}

export interface Predictions {
  threshold: number;
  sets: { frozen: PredSet; walkForward: PredSet };
}

export interface Meta {
  generatedAt: string;
  disclaimer: string;
  data: { start: string; end: string; priceDays: number; noNewsDays: string[] };
  model: {
    seeds: number;
    seqLen: number;
    lstm: Record<string, number>;
    xgboost: Record<string, number>;
    frozen: string;
    modellingTableSha256: string;
  };
  splits: Record<string, [string, string]>;
}
