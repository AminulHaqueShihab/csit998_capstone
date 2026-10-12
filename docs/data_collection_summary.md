# Data collection summary (for the report)

Items in [square brackets] are not recorded in the repository and must be filled in by the person who collected that data.

## 1. Overview
The study uses two data sources, matched by UTC calendar day:
- **Bitcoin price data:** Binance daily BTCUSDT candles.
- **Media sentiment data:** Bitcoin-related news headlines from the GDELT Global Knowledge Graph.

Together they cover **1 January 2021 to 31 August 2026** (2,069 days). This differs from the originally planned
period (January 2021 to May 2025) because news data was available through August 2026.

## 2. Price data: Binance BTCUSDT daily klines
| Item | Detail |
|---|---|
| Source and owner | Binance, public market data archive (data.binance.vision, spot, monthly klines, BTCUSDT, 1-day interval) [1][2] |
| Coverage | 2021-01-01 to 2026-08-31 |
| Files | 68 monthly zip files, each with a SHA-256 checksum file |
| Records | 2,069 daily bars, with no missing dates |
| Fields | Open time, open, high, low, close, volume, close time, quote-asset volume, number of trades, taker-buy base volume and taker-buy quote volume |
| Time convention | UTC daily bars, 00:00:00 to 23:59:59.999 |
| Retrieval date | [to be confirmed by the data lead] |
| Licence | Publicly available market data under Binance's data terms |

**Quality checks:**
- Every file was verified against its checksum, and all 68 passed.
- No duplicate days and no rejected rows.
- All open, high, low and close values passed the validity checks (high is not below any other price, low is not
  above any other price, prices are positive, volume is non-negative).
- 24 days with moves above 10% were reviewed. They match known events (for example the May 2021 crash and the
  November 2022 FTX collapse), so they were kept as genuine market movements.
- Binance changed its timestamp unit from milliseconds to microseconds in 2025. The pipeline handles both.

## 3. News data: GDELT Global Knowledge Graph (GKG 2.1)
| Item | Detail |
|---|---|
| Source and owner | The GDELT Project [3][4], accessed through Google BigQuery (public table `gdelt-bq.gdeltv2.gkg_partitioned`) [5] |
| Selection filter | Articles whose headline contains "bitcoin" or "btc" (case-insensitive) |
| Fields collected | GDELT timestamp (the 15-minute UTC batch in which GDELT recorded the item), precise publication timestamp (when available), headline, article URL, news source, translation information and GDELT tone (V2Tone) [4] |
| Coverage | 2021-01-01 to 2026-08-31 |
| Records collected | **540,518 headlines** in two exports: 279,049 for 2021-2023 and 261,469 for 2024-2026 |
| Retrieval | 2021-2023 extracted from BigQuery on 4 October 2026 (about 223 GB scanned, free tier). 2024-2026 extracted earlier by the team [date and exact query to be confirmed] |
| Licence | GDELT data is open for use with attribution [6]. The headline text belongs to the original publishers |

**Known limitations of this data:**
- **Two exports with slightly different filters.** For January 2024, the 2021-2023 query returns about 4% more rows
  than the team's earlier 2024-2026 export. This produced a visible jump in daily article volume at 2024-01 (about
  204 articles per day in December 2023 against about 302 in January 2024). Raw article counts are therefore not
  directly comparable across the two periods.
- **Missing publication times.** About half of the records have no precise publication timestamp. The GDELT batch
  timestamp is always present and is used as the reliable time.
- **Headlines only.** There is no article body, so headline sentiment is a coarse proxy for market mood.
- **Many non-English and machine-translated sources.** About 39% of the headlines came from machine-translated sources.
- **Coverage gap.** No usable news for 17 consecutive days, 15 June to 1 July 2025.
- **Publisher duplicates.** Many headlines are syndicated copies of the same story.

## 4. How the collected data becomes the analysis data
The collected 540,518 headlines are cleaned (see the preprocessing summary) to **250,475** English, Bitcoin-relevant,
de-duplicated headlines. Each headline is assigned to the UTC day on which it became available. Price and news are
then joined, with a prediction cutoff of 23:59:59 UTC on day t to predict the price direction of day t+1.

## 5. Data considered but not used
| Data | Reason excluded |
|---|---|
| Bitcoin tweet dataset (about 2 GB, about 4.7 million parsed records, March 2025 to March 2026) | Twitter/X is outside the approved scope. Timestamps could not be verified and it has no human-labelled sentiment. Covers only one year |
| NewsAPI sample (97 articles) | Covers only about 7 hours, far too small |
| CoinMarketCap and `btc-usd-max` price files | Not the approved price source (Binance). Used only as a cross-check |
| Other small sentiment files (2019-2024) | Sparse coverage and unknown provenance |
| A multi-coin dataset covering 31 days | Too short, and appears synthetic |

## 6. Data handling, ethics and sharing
- Raw price files are kept as read-only inputs, with processed data produced by versioned scripts.
- No credentials or API keys are used or stored.
- Binance market data and GDELT metadata are public. The raw GDELT export (about 170 MB, containing publishers'
  headline text) is not committed to the repository, and only derived daily aggregates are shared.
- A provenance register for each dataset is in `docs/data_sources.md`.

## References
Please verify each link and add the access date before submission.

[1] Binance. (n.d.). *Binance public data: Spot market data, klines* [Data set]. https://data.binance.vision

[2] Binance. (n.d.). *binance-public-data* [Documentation and download scripts]. GitHub. https://github.com/binance/binance-public-data

[3] Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global data on events, location and tone, 1979-2012. *Paper presented at the ISA Annual Convention*, San Francisco, CA.

[4] The GDELT Project. (2015). *The GDELT Global Knowledge Graph (GKG) Data Format Codebook V2.1*. http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf

[5] The GDELT Project. (n.d.). *GDELT 2.0 on Google BigQuery* (`gdelt-bq.gdeltv2`). https://www.gdeltproject.org/data.html#googlebigquery

[6] The GDELT Project. (n.d.). *About GDELT: Terms of use and citation*. https://www.gdeltproject.org/about.html
