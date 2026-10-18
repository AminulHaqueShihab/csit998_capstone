import { promises as fs } from "node:fs";
import path from "node:path";
import type { Meta, Predictions, PriceRow } from "./types";

// Runs at build time only (static export). Files are produced by `python -m src.export_demo_data`.
async function read<T>(name: string): Promise<T> {
  const file = path.join(process.cwd(), "public", "data", `${name}.json`);
  return JSON.parse(await fs.readFile(file, "utf8")) as T;
}

export const getMeta = () => read<Meta>("meta");
export const getPriceSentiment = () => read<PriceRow[]>("price_sentiment");
export const getPredictions = () => read<Predictions>("predictions");
