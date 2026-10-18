"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { pct, usd } from "@/lib/format";

export interface ChartPoint {
  date: string;
  close: number;
  closeNext: number | null;
  pred: 0 | 1 | null; // predicted direction of the NEXT day (null = no prediction on this date)
  up: 0 | 1 | null; // what actually happened the next day
  prob: number | null;
  correct: boolean | null;
}

const config = { close: { label: "BTC close (USDT)", color: "var(--chart-1)" } } satisfies ChartConfig;
const GOOD = "#059669";
const BAD = "#e11d48";

// Triangle up = predicted up, triangle down = predicted not up. Filled = right, outline = wrong
// (so the chart does not rely on colour alone).
function PredictionDot({ cx, cy, payload, size = 3.5 }: { cx?: number; cy?: number; payload?: ChartPoint; size?: number }) {
  if (cx === undefined || cy === undefined || !payload || payload.pred === null) return null;
  const s = size;
  const d =
    payload.pred === 1
      ? `M${cx},${cy - s}L${cx + s},${cy + s}L${cx - s},${cy + s}Z`
      : `M${cx},${cy + s}L${cx + s},${cy - s}L${cx - s},${cy - s}Z`;
  const color = payload.correct ? GOOD : BAD;
  return <path d={d} fill={payload.correct ? color : "var(--background)"} stroke={color} strokeWidth={1.4} />;
}

function Tip({ active, payload }: { active?: boolean; payload?: { payload: ChartPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const move = p.closeNext !== null ? p.closeNext / p.close - 1 : null;
  return (
    <div className="grid min-w-[12rem] gap-1 rounded-lg border bg-background px-3 py-2 text-xs shadow-xl">
      <div className="font-medium">{p.date}</div>
      <div className="flex justify-between gap-4">
        <span className="text-muted-foreground">Close</span>
        <span className="tabular-nums">{usd(p.close)}</span>
      </div>
      {p.pred !== null ? (
        <>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Predicted next day</span>
            <span>{p.pred ? "Up" : "Not up"} ({pct(p.prob, 0)} up)</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Actual next day</span>
            <span className="tabular-nums">
              {p.up ? "Up" : "Not up"}
              {move !== null ? ` (${move >= 0 ? "+" : ""}${(move * 100).toFixed(1)}%)` : ""}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Result</span>
            <span className="font-medium" style={{ color: p.correct ? GOOD : BAD }}>
              {p.correct ? "Right" : "Wrong"}
            </span>
          </div>
        </>
      ) : (
        <div className="text-muted-foreground">No prediction for this date</div>
      )}
    </div>
  );
}

export function PredictionChart({ data }: { data: ChartPoint[] }) {
  const days = data.length;
  const short = days <= 140;
  return (
    <ChartContainer config={config} className="h-[340px] w-full">
      <LineChart data={data} syncId="demo" margin={{ left: 4, right: 12, top: 10 }}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          minTickGap={44}
          tickFormatter={(d: string) => (short ? d.slice(5) : d.slice(0, 7))}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          width={56}
          domain={[(min: number) => Math.floor(min * 0.97), (max: number) => Math.ceil(max * 1.03)]}
          tickFormatter={(v: number) => `$${Math.round(v / 1000)}k`}
        />
        <ChartTooltip content={<Tip />} />
        <Line
          dataKey="close"
          stroke="var(--color-close)"
          strokeWidth={1.6}
          strokeOpacity={days > 300 ? 0.7 : 1}
          dot={<PredictionDot size={days <= 120 ? 4 : days <= 400 ? 3 : 2.2} />}
          activeDot={{ r: 4 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ChartContainer>
  );
}

export function MarkerLegend() {
  const item = (label: string, path: string, fill: string, stroke: string) => (
    <span className="inline-flex items-center gap-1.5">
      <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden>
        <path d={path} fill={fill} stroke={stroke} strokeWidth="1.4" />
      </svg>
      {label}
    </span>
  );
  const up = "M0,-5L5,5L-5,5Z";
  const down = "M0,5L5,-5L-5,-5Z";
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground" aria-label="Marker key">
      {item("Predicted up, right", up, GOOD, GOOD)}
      {item("Predicted up, wrong", up, "var(--background)", BAD)}
      {item("Predicted not up, right", down, GOOD, GOOD)}
      {item("Predicted not up, wrong", down, "var(--background)", BAD)}
    </div>
  );
}
