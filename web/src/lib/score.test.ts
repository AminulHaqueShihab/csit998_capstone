import assert from "node:assert/strict";
import { test } from "node:test";
import { addDays, chanceHalfWidth, predictsUp, presetRange, scoreModel } from "./score.ts";

const row = (up: 0 | 1, p: number) =>
  ({ date: "2025-01-01", up, close: 1, closeNext: 1, naive: 0, xgboost: 0, lstmA: p, lstmBVader: p, lstmBFinbert: p }) as never;

test("scoreModel counts the confusion matrix and accuracy", () => {
  // predictions: up, not-up, not-up, up, up, not-up   actual: up, up, not-up, not-up, up, not-up
  const rows = [row(1, 0.9), row(1, 0.2), row(0, 0.1), row(0, 0.8), row(1, 0.7), row(0, 0.3)];
  const s = scoreModel(rows, "lstmA");
  assert.deepEqual([s.tp, s.fn, s.tn, s.fp, s.n], [2, 1, 2, 1, 6]);
  assert.equal(s.accuracy, 4 / 6);
  assert.equal(s.balancedAccuracy, (2 / 3 + 2 / 3) / 2);
  assert.equal(s.predictedUpShare, 3 / 6);
  assert.equal(s.actualUpShare, 3 / 6);
});

test("probability of exactly 0.5 counts as a prediction of up", () => {
  assert.equal(predictsUp(row(1, 0.5), "lstmA"), 1);
  assert.equal(predictsUp(row(1, 0.4999), "lstmA"), 0);
});

test("empty and one-sided ranges do not divide by zero", () => {
  assert.equal(scoreModel([], "lstmA").accuracy, null);
  assert.equal(scoreModel([row(1, 0.9), row(1, 0.1)], "lstmA").balancedAccuracy, null); // only "up" days: undefined
  assert.equal(scoreModel([row(0, 0.9), row(0, 0.1)], "lstmA").balancedAccuracy, null);
  assert.equal(chanceHalfWidth(0), null);
});

test("chance band shrinks with more days", () => {
  assert.ok(chanceHalfWidth(100)! > chanceHalfWidth(400)!);
  assert.ok(Math.abs(chanceHalfWidth(334)! - 0.0536) < 0.001);
});

test("date helpers and presets stay inside the data period", () => {
  assert.equal(addDays("2026-03-01", -1), "2026-02-28");
  assert.equal(addDays("2024-02-28", 1), "2024-02-29");
  const data = { start: "2021-01-01", end: "2026-08-31" };
  const t = { from: "2025-10-01", to: "2026-08-30" };
  assert.deepEqual(presetRange("test", data, t), { from: "2025-10-01", to: "2026-08-30" });
  assert.deepEqual(presetRange("all", data, t), { from: "2021-01-01", to: "2026-08-31" });
  assert.deepEqual(presetRange("3m", data, t), { from: "2026-06-01", to: "2026-08-31" });
  assert.deepEqual(presetRange("1y", { start: "2026-01-01", end: "2026-08-31" }, t).from, "2026-01-01");
});
