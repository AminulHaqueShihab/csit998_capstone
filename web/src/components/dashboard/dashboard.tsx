"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Segmented } from "@/components/segmented";
import { MarkerLegend, PredictionChart, type ChartPoint } from "./prediction-chart";
import { SentimentChart, type SentimentPoint } from "./sentiment-chart";
import { MODEL_LABELS, MODEL_SHORT, pct } from "@/lib/format";
import { chanceHalfWidth, predictsUp, presetRange, scoreModel, type Preset } from "@/lib/score";
import type { Meta, ModelKey, Predictions, PriceRow } from "@/lib/types";

const MODELS = Object.keys(MODEL_SHORT) as ModelKey[];
type SetKey = "frozen" | "walkForward";
type Source = "vader" | "finbert";

const PRESETS: { id: Preset; label: string }[] = [
  { id: "test", label: "Test period" },
  { id: "3m", label: "Last 3 months" },
  { id: "6m", label: "Last 6 months" },
  { id: "1y", label: "Last year" },
  { id: "all", label: "All data" },
];

export function Dashboard({ meta, prices, predictions }: { meta: Meta; prices: PriceRow[]; predictions: Predictions }) {
  const [setKey, setSetKey] = useState<SetKey>("frozen");
  const [model, setModel] = useState<ModelKey>("lstmA");
  const [source, setSource] = useState<Source>("vader");
  const test = { from: predictions.sets.frozen.from, to: predictions.sets.frozen.to };
  const [range, setRange] = useState(test);
  const set = predictions.sets[setKey];

  const lo = range.from <= range.to ? range.from : range.to;
  const hi = range.from <= range.to ? range.to : range.from;

  const rows = useMemo(() => set.rows.filter((r) => r.date >= lo && r.date <= hi), [set, lo, hi]);
  const byDate = useMemo(() => new Map(set.rows.map((r) => [r.date, r])), [set]);

  const chartData: ChartPoint[] = useMemo(
    () =>
      prices
        .filter((p) => p.date >= lo && p.date <= hi)
        .map((p) => {
          const r = byDate.get(p.date);
          const pred = r ? predictsUp(r, model, predictions.threshold) : null;
          return {
            date: p.date,
            close: p.close,
            closeNext: r ? r.closeNext : null,
            pred,
            up: r ? r.up : null,
            prob: r ? r[model] : null,
            correct: r && pred !== null ? pred === r.up : null,
          };
        }),
    [prices, byDate, model, lo, hi, predictions.threshold],
  );

  const sentimentData: SentimentPoint[] = useMemo(() => {
    const withAvg = prices.map((p, i) => {
      const win = prices
        .slice(Math.max(0, i - 6), i + 1)
        .map((x) => x[source])
        .filter((v): v is number => v !== null);
      return { date: p.date, raw: p[source], noNews: p.noNews, avg7: win.length ? win.reduce((a, b) => a + b, 0) / win.length : null };
    });
    return withAvg.filter((p) => p.date >= lo && p.date <= hi);
  }, [prices, source, lo, hi]);

  const s = useMemo(() => scoreModel(rows, model, predictions.threshold), [rows, model, predictions.threshold]);
  const half = chanceHalfWidth(s.n);
  const covered = chartData.some((p) => p.pred !== null);

  const stat = (label: string, value: string, note?: string, highlight = false) => (
    <div className={`space-y-0.5 px-4 py-3 ${highlight ? "bg-secondary/60" : ""}`}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${highlight ? "text-secondary-foreground" : ""}`}>{value}</div>
      {note ? <div className="text-xs text-muted-foreground">{note}</div> : null}
    </div>
  );

  return (
    <div className="space-y-5">
      <Card>
        <CardContent className="grid gap-x-8 gap-y-4 pt-4 md:grid-cols-2">
          <div className="space-y-1.5">
            <div className="text-xs font-medium text-muted-foreground">Dates</div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                id="from"
                aria-label="From date"
                type="date"
                min={meta.data.start}
                max={meta.data.end}
                value={range.from}
                onChange={(e) => e.target.value && setRange((r) => ({ ...r, from: e.target.value }))}
                className="block h-8 rounded-lg border border-input bg-background px-2 text-sm"
              />
              <span className="text-muted-foreground">to</span>
              <input
                id="to"
                aria-label="To date"
                type="date"
                min={meta.data.start}
                max={meta.data.end}
                value={range.to}
                onChange={(e) => e.target.value && setRange((r) => ({ ...r, to: e.target.value }))}
                className="block h-8 rounded-lg border border-input bg-background px-2 text-sm"
              />
            </div>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Date range presets">
              {PRESETS.map((p) => (
                <Button key={p.id} variant="ghost" size="xs" className="text-muted-foreground hover:text-secondary-foreground" onClick={() => setRange(presetRange(p.id, meta.data, test))}>
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="space-y-4">
            <div className="space-y-1.5">
              <div className="text-xs font-medium text-muted-foreground">Model</div>
              <Segmented<ModelKey> label="Model" value={model} onChange={setModel} options={MODELS.map((m) => ({ value: m, label: MODEL_SHORT[m] }))} />
            </div>
            <div className="space-y-1.5">
              <div className="text-xs font-medium text-muted-foreground">Predictions ({set.from} to {set.to})</div>
              <Segmented<SetKey>
                label="Prediction set"
                value={setKey}
                onChange={setSetKey}
                options={[
                  { value: "frozen", label: "Frozen test split" },
                  { value: "walkForward", label: "Walk-forward" },
                ]}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="grid grid-cols-2 divide-x divide-y overflow-hidden p-0 lg:grid-cols-4 lg:divide-y-0" aria-label="Summary for the selected dates">
          {stat("Accuracy", pct(s.accuracy), half !== null ? `Guessing: 50% ± ${(half * 100).toFixed(1)} pts` : "No predictions in range", true)}
          {stat("Days predicted", String(s.n), `${s.tp + s.tn} right · ${s.fp + s.fn} wrong`)}
          {stat("Predicted up", pct(s.predictedUpShare), "share of days")}
          {stat("Actually up", pct(s.actualUpShare), "share of days")}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Bitcoin price and predictions</CardTitle>
          <CardDescription>{MODEL_LABELS[model]} · each marker is that day&apos;s call for the next day.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {!covered ? (
            <p className="rounded-md border border-dashed px-3 py-2 text-sm text-muted-foreground">
              No predictions fall in this date range for the selected set. The price is still shown.
            </p>
          ) : null}
          <PredictionChart data={chartData} />
          <MarkerLegend />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle>News sentiment</CardTitle>
            <CardDescription>Daily headline score, −1 to +1. Grey bands: no news.</CardDescription>
          </div>
          <Segmented<Source>
            label="Sentiment method"
            value={source}
            onChange={setSource}
            options={[
              { value: "vader", label: "VADER" },
              { value: "finbert", label: "FinBERT" },
            ]}
          />
        </CardHeader>
        <CardContent>
          <SentimentChart data={sentimentData} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>All models on the same days</CardTitle>
          <CardDescription>{s.n} days with predictions, {lo} to {hi}.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Accuracy</TableHead>
                <TableHead className="text-right">Balanced accuracy</TableHead>
                <TableHead className="text-right">Predicted up</TableHead>
                <TableHead className="text-right">Right / wrong</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {MODELS.map((m) => {
                const x = scoreModel(rows, m, predictions.threshold);
                return (
                  <TableRow key={m} data-state={m === model ? "selected" : undefined} className={m === model ? "bg-secondary/60" : undefined}>
                    <TableCell className="font-medium">{MODEL_LABELS[m]}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(x.accuracy)}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(x.balancedAccuracy)}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(x.predictedUpShare)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {x.tp + x.tn} / {x.fp + x.fn}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
