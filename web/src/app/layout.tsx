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
          <div role="note" className="flex items-center justify-center gap-2 bg-secondary px-4 py-1.5 text-center text-xs text-secondary-foreground">
            <TriangleAlert className="size-3.5 shrink-0" aria-hidden />
            Experimental university research. Not financial advice.
          </div>
          <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 md:px-8">
            <header className="flex items-center gap-3 border-b pb-4">
              <div className="h-9 w-1.5 rounded-full bg-primary" aria-hidden />
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-primary">Bitcoin next-day direction</h1>
                <p className="text-sm text-muted-foreground">Predicted vs actual · CSIT998 Group 16</p>
              </div>
            </header>
            <main>{children}</main>
            <footer className="border-t pt-4 text-xs text-muted-foreground">
              Prices: Binance BTCUSDT daily close (UTC). Sentiment: GDELT headlines scored with VADER and FinBERT. A model
              predicts &quot;up&quot; at 50% or more. Data {meta.data.start} to {meta.data.end} · {meta.model.seeds} seeds per model · exported {meta.generatedAt}.
            </footer>
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
