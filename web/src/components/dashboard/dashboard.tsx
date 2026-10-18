"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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

  const stat = (label: string, value: string, note?: string) => (
    <Card size="sm">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
        {note ? <p className="text-xs text-muted-foreground">{note}</p> : null}
      </CardHeader>
    </Card>
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="space-y-4 pt-4">
          <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
            <div className="space-y-1">
              <label htmlFor="from" className="text-xs font-medium text-muted-foreground">From</label>
              <input
                id="from"
                type="date"
                min={meta.data.start}
                max={meta.data.end}
                value={range.from}
                onChange={(e) => e.target.value && setRange((r) => ({ ...r, from: e.target.value }))}
                className="block h-8 rounded-lg border border-input bg-background px-2 text-sm"
              />
            </div>
            <div className="space-y-1">
              <label htmlFor="to" className="text-xs font-medium text-muted-foreground">To</label>
              <input
                id="to"
                type="date"
                min={meta.data.start}
                max={meta.data.end}
                value={range.to}
                onChange={(e) => e.target.value && setRange((r) => ({ ...r, to: e.target.value }))}
                className="block h-8 rounded-lg border border-input bg-background px-2 text-sm"
              />
            </div>
            <div className="flex flex-wrap gap-2" role="group" aria-label="Date range presets">
              {PRESETS.map((p) => (
                <Button key={p.id} variant="outline" size="sm" onClick={() => setRange(presetRange(p.id, meta.data, test))}>
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Model</div>
              <Segmented<ModelKey> label="Model" value={model} onChange={setModel} options={MODELS.map((m) => ({ value: m, label: MODEL_SHORT[m] }))} />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Predictions</div>
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
            <Badge variant="outline" className="self-end">
              Predictions cover {set.from} to {set.to}
            </Badge>
          </div>
        </CardContent>
      </Card>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Summary for the selected dates">
        {stat("Accuracy", pct(s.accuracy), half !== null ? `Guessing: 50% ± ${(half * 100).toFixed(1)} pts` : "No predictions in range")}
        {stat("Days predicted", String(s.n), `${s.tp + s.tn} right · ${s.fp + s.fn} wrong`)}
        {stat("Predicted up", pct(s.predictedUpShare), "share of days")}
        {stat("Actually up", pct(s.actualUpShare), "share of days")}
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Bitcoin price and predictions</CardTitle>
          <CardDescription>
            {MODEL_LABELS[model]} · {lo} to {hi}. Each marker is that day&apos;s prediction for whether the next day closes higher.
          </CardDescription>
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
            <CardDescription>Daily average score of Bitcoin headlines (−1 negative to +1 positive). Grey bands: no news data.</CardDescription>
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
          <CardDescription>
            {s.n} days with predictions between {lo} and {hi}.
          </CardDescription>
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
                  <TableRow key={m} data-state={m === model ? "selected" : undefined}>
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
