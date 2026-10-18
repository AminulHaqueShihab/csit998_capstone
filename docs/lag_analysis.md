# Lag and feature analysis (requirement EX-01)

Descriptive only. Correlation is not causation, and nothing here is tuned to improve a model.
Code: `src/lag_analysis.py`. Tables and figure: `results/lag_analysis/`. Summary numbers: `reports/lag_analysis_summary.json`.

## Method
- **Sample:** 2021-01-10 to 2025-06-14 (1,617 days). It stops before the 17-day news gap, and the test period
  (October 2025 onward) is not used.
- **Measures (day t):** mean VADER score, mean FinBERT score, VADER net share (positive minus negative share),
  the "surprise" of each mean (today minus the mean of the previous 7 days, today excluded) and log article count.
- **Targets:** daily return and absolute daily return (volatility) on day t+k, for k from -5 to +5.
  k > 0 means the sentiment comes **before** the price move (this is what prediction needs).
  k < 0 means the sentiment comes **after** the move (sentiment reacts to price).
- **Statistic:** Spearman rank correlation.
- **Significance:** circular-shift permutation test (5,000 shifts), which keeps the autocorrelation of both
  series, so it is not fooled by slowly varying series. 132 tests were run, so about 7 nominal p < 0.05 results
  are expected by chance. The Bonferroni threshold is p < 0.05 / 132 = 0.00038.

## Findings
1. **Sentiment follows price; it does not lead it.** 25 of the 132 tests pass the Bonferroni threshold, and every one
   is a link between sentiment and the return on the same day or **earlier** days (k from -5 to 0). The strongest is at
   k = -1: day-t sentiment correlates with the **previous day's** return at 0.45 for FinBERT and 0.26 to 0.28 for VADER.
   Headlines published on a day largely describe the previous day's price move.
2. **No lead at all.** For k = +1 to +5, none of the 6 measures x 2 targets x 5 lags (60 tests) reaches even
   nominal significance. Correlations at k = +1 are between -0.012 and +0.007. This matches the model results:
   sentiment cannot predict what it only echoes.
3. **FinBERT reflects price moves more than VADER does** (same-day 0.23 vs 0.09; previous-day 0.45 vs 0.26), which fits
   FinBERT being a financial-domain model. This does not make it more predictive.
4. **Attention (article count) does not predict volatility.** Article counts tend to be higher on days of larger absolute
   moves (correlation 0.16 same day), but even this same-day link is not significant under the permutation test (the
   series drifts and changes level at the 2024-01 data join, so its chance band is wide), and none of the lead links
   (k > 0) is significant.
5. **Quintiles show no pattern.** Next-day share of up days is between 0.48 and 0.53 in every sentiment quintile for
   both VADER and FinBERT, with no monotonic trend.
6. **The same-day link is not stable across years** (FinBERT 0.11 to 0.32, VADER about -0.02 to +0.12), and next-day
   correlations change sign from year to year (all between -0.09 and +0.03).

## Price features
Univariate screening of all 18 model features against next-day direction (same sample):
- 5 of 18 reach nominal p < 0.05, against 0.9 expected by chance. Three pass the Bonferroni threshold (p < 0.0028):
  `return_1d`, `log_return_1d` and `close_over_sma_7`. All are **negative** associations (a weak short-term reversal:
  after an up day, a slightly higher chance of a down day), with AUC about 0.45 (0.55 if the sign is flipped).
- No sentiment feature is significant.
- This is a small effect (the largest deviation from AUC 0.5 is 0.05). It was not exploited on the held-out test period
  (all models scored about 48%), so it should not be treated as a stable signal.

## Use in the report
- Report finding 1 and 2 together: sentiment is strongly associated with the **previous** day's price move
  (a reaction), and has no association with the **next** day's move (no prediction). This explains why adding
  sentiment did not improve the models.
- Say clearly that the analysis is descriptive, run on 2021-2025 data before the held-out test period, and that 132 tests
  were corrected for.
- Limitation: headlines are assigned to the UTC day on which GDELT recorded them, so a headline at 23:59 on day t
  reflecting a move that day still counts for day t. Intraday timing is not modelled.
