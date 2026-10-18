import type { ModelKey, PredRow } from "./types";

// Pure helpers (no React, no path aliases) so they can be tested with `node --test`.

export type Preset = "test" | "3m" | "6m" | "1y" | "all";

export function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

export function presetRange(
  preset: Preset,
  data: { start: string; end: string },
  test: { from: string; to: string },
): { from: string; to: string } {
  if (preset === "all") return { from: data.start, to: data.end };
  if (preset === "test") return { from: test.from, to: test.to };
  const days = preset === "3m" ? 91 : preset === "6m" ? 182 : 365;
  const from = addDays(data.end, -days);
  return { from: from < data.start ? data.start : from, to: data.end };
}

export const predictsUp = (r: PredRow, model: ModelKey, threshold = 0.5): 0 | 1 => (r[model] >= threshold ? 1 : 0);

export interface Score {
  n: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  accuracy: number | null;
  balancedAccuracy: number | null;
  predictedUpShare: number | null;
  actualUpShare: number | null;
}

export function scoreModel(rows: PredRow[], model: ModelKey, threshold = 0.5): Score {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const r of rows) {
    const p = predictsUp(r, model, threshold);
    if (p === 1 && r.up === 1) tp++;
    else if (p === 0 && r.up === 0) tn++;
    else if (p === 1) fp++;
    else fn++;
  }
  const n = rows.length;
  const up = tp + fn;
  const notUp = tn + fp;
  return {
    n, tp, tn, fp, fn,
    accuracy: n ? (tp + tn) / n : null,
    balancedAccuracy: up && notUp ? (tp / up + tn / notUp) / 2 : null,
    predictedUpShare: n ? (tp + fp) / n : null,
    actualUpShare: n ? up / n : null,
  };
}

// 95% range of accuracy that pure guessing would produce on n independent days.
export const chanceHalfWidth = (n: number): number | null => (n > 0 ? 1.96 * Math.sqrt(0.25 / n) : null);
