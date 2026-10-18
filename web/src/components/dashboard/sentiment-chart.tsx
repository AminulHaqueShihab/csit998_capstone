"use client";

import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

export interface SentimentPoint {
  date: string;
  raw: number | null;
  avg7: number | null;
  noNews: boolean;
}

const config = {
  raw: { label: "Daily average", color: "var(--chart-3)" },
  avg7: { label: "7-day average", color: "var(--chart-2)" },
} satisfies ChartConfig;

export function SentimentChart({ data }: { data: SentimentPoint[] }) {
  const gaps: { start: string; end: string }[] = [];
  let open: string | null = null;
  data.forEach((r, i) => {
    if (r.noNews && open === null) open = r.date;
    if (open !== null && (!r.noNews || i === data.length - 1)) {
      gaps.push({ start: open, end: r.noNews ? r.date : data[i - 1].date });
      open = null;
    }
  });
  const short = data.length <= 140;
  return (
    <ChartContainer config={config} className="h-[200px] w-full">
      <LineChart data={data} syncId="demo" margin={{ left: 4, right: 12, top: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={44} tickFormatter={(d: string) => (short ? d.slice(5) : d.slice(0, 7))} />
        <YAxis tickLine={false} axisLine={false} width={56} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(1)} />
        <ReferenceLine y={0} stroke="var(--border)" />
        {gaps.map((g) => (
          <ReferenceArea key={g.start} x1={g.start} x2={g.end} fill="var(--muted-foreground)" fillOpacity={0.18} />
        ))}
        <ChartTooltip content={<ChartTooltipContent indicator="line" formatter={(v) => Number(v).toFixed(3)} />} />
        <Line dataKey="raw" stroke="var(--color-raw)" strokeOpacity={0.4} strokeWidth={1} dot={false} connectNulls={false} isAnimationActive={false} />
        <Line dataKey="avg7" stroke="var(--color-avg7)" strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} />
      </LineChart>
    </ChartContainer>
  );
}
