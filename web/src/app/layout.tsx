import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { TriangleAlert } from "lucide-react";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";
import { getMeta } from "@/lib/data";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Bitcoin Direction Demo",
  description: "CSIT998 Group 16 demo: model predictions of next-day Bitcoin direction against the actual price. Experimental, not financial advice.",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const meta = await getMeta();
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full bg-background text-foreground">
        <TooltipProvider>
          <div
            role="note"
            className="flex items-center justify-center gap-2 border-b border-amber-300/60 bg-amber-100 px-4 py-2 text-center text-xs font-medium text-amber-950 dark:border-amber-500/30 dark:bg-amber-950 dark:text-amber-100"
          >
            <TriangleAlert className="size-4 shrink-0" aria-hidden />
            Experimental research output for a university project. Not financial advice, and not a trading tool.
          </div>
          <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 md:px-8">
            <header className="flex flex-wrap items-end justify-between gap-2">
              <div>
                <h1 className="text-xl font-semibold tracking-tight">Bitcoin next-day direction: demo</h1>
                <p className="text-sm text-muted-foreground">CSIT998 · Group 16 · predicted vs actual</p>
              </div>
              <p className="text-xs text-muted-foreground">
                Data {meta.data.start} to {meta.data.end} · models frozen {meta.model.frozen} · {meta.model.seeds} seeds each
              </p>
            </header>
            <main>{children}</main>
            <footer className="border-t pt-4 text-xs text-muted-foreground">
              Prices: Binance BTCUSDT daily close (UTC). Sentiment: GDELT news headlines scored with VADER and FinBERT. A model
              predicts &quot;up&quot; when its probability is 50% or more. Results exported {meta.generatedAt}.
            </footer>
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
