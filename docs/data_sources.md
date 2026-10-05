# Data source register (DR-04)

| | BTCUSDT daily prices | GDELT Bitcoin news |
|---|---|---|
| Source | Binance public data, spot daily klines | GDELT Global Knowledge Graph v2.1 (`gdelt-bq.gdeltv2.gkg_partitioned` on Google BigQuery) |
| Owner | Binance | The GDELT Project |
| URL | https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1d/ | https://www.gdeltproject.org/ |
| Raw location | `data/raw/binance/BTCUSDT/1d/*.zip` (+ `.CHECKSUM`) | `data/raw/gdelt/gdelt_btc_news_2021_2023_raw.csv`, `gdelt_btc_news_2024_2026_raw.csv` |
| Retrieved | Monthly zips for 2021-01 .. 2026-08 (68 files) | 2024-2026 file: supplied by the team (original query not on record). 2021-2023 file: BigQuery export, 2026-10-04 |
| Coverage | 2021-01-01 .. 2026-08-31, 2,069 daily bars, no missing dates | 2021-01-01 .. 2026-08-31 (headlines containing "bitcoin" or "btc") |
| Rows (raw) | 2,069 | 540,518 (279,049 for 2021-2023, 261,469 for 2024-2026) |
| Licence / terms | Binance market-data terms; public download | GDELT is open for any use with attribution; headlines belong to their publishers |
| Redistribution | Small; can be committed | Raw export is ~170 MB, so it is **not committed**; only derived daily aggregates are shared. Do not redistribute headline text publicly |
| Integrity | Every zip verified against its `.CHECKSUM` (sha256) in `src.prepare_price` | Row counts and date coverage checked in `reports/news_quality_report.json` |

## 2021-2023 GDELT query (BigQuery)

```sql
SELECT
  DATE AS gdelt_timestamp,
  REGEXP_EXTRACT(Extras, r'<PAGE_PRECISEPUBTIMESTAMP>(\d+)</PAGE_PRECISEPUBTIMESTAMP>') AS precise_publication_timestamp,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS headline,
  DocumentIdentifier AS article_url,
  SourceCommonName AS news_source,
  TranslationInfo,
  V2Tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('2021-01-01') AND _PARTITIONTIME < TIMESTAMP('2024-01-01')
  AND REGEXP_CONTAINS(LOWER(REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>')), r'bitcoin|btc')
```

Scanned 223.5 GB (free tier). For January 2024 this query returned 17,486 rows against 16,780 in
the team's file (+4%), so the original filter was slightly narrower. The two files are treated as
one source; the join point (2023-12 / 2024-01) is checked in the quality report.

## Known limitations

- `precise_publication_timestamp` is blank for about half of the rows. `gdelt_timestamp` (when GDELT
  first saw the item, 15-minute UTC batches) is the reliable time. The later of the two is used as the
  availability time, which is the conservative choice for the no-leakage rule.
- GDELT contains many non-English and machine-translated sources. Only English-source headlines are kept
  because VADER is an English lexicon.
- Headlines only (no article body). Headline sentiment is a coarse proxy for market mood.
- Duplicate and syndicated stories are removed by URL and by same-day headline text; paraphrased
  syndication is not caught.
- Coverage per month varies (see `reports/news_quality_report.json`); days without any retained article are
  kept explicitly with `no_news_flag = 1`.

## Findings from the first full run (2026-10-05)

- **News gap:** no retained article for 17 consecutive days, 2025-06-15 to 2025-07-01 (GDELT/export gap).
  These days are kept with `no_news_flag = 1`; they must not be read as "neutral news".
- **Volume step at the file join:** mean articles/day is about 204 in 2023-12 and about 302 in 2024-01, after
  the two exports were merged. The two files probably used slightly different filters, so `article_count`
  and the pos/neg/neutral counts are not on one scale across 2021-2023 vs 2024+. Prefer ratio or standardised
  features, or test the sensitivity of results to this, before reading volume effects as real.
- **Same-day link:** `vader_mean` correlates 0.12 with the same day's return but about 0.01 with next-day
  direction (descriptive only, not a causal claim; formal lag analysis is requirement EX-01).
