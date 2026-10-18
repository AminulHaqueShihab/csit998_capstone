# Walk-forward evaluation

Code: `src/walk_forward.py`. Outputs: `results/walk_forward/` (`walk_forward_results.json`, `walk_forward_by_fold.csv`,
`walk_forward_predictions.csv`). The primary result remains the single frozen test split (`src/evaluate_final`).

## Design (fixed before running)
- Eight consecutive test folds of about six months, 2023-01-01 to 2026-08-30 (1,338 test days in total).
- For each fold, models are retrained on all earlier data. The last 90 days before the fold are the validation set
  (early stopping); the earlier part is the training set. Scalers are refitted on each fold's training rows only.
- Same features, 14-day windows, frozen LSTM and XGBoost settings (no re-tuning) and 10 seeds as the main experiment.
- Models: naive majority class, XGBoost (price), LSTM A (price), LSTM B (price + VADER), LSTM B (price + FinBERT).
- Pooled uncertainty: 95% block bootstrap (7-day blocks) for B minus A.

## Results (pooled, mean over 10 seeds; no-skill band for 1,338 days is 50% +/- 2.7 points)
| Model | Accuracy | Balanced accuracy | F1 |
|---|---|---|---|
| Naive majority | 49.4% | 50.0% | 0.00 |
| XGBoost (price) | 48.9% | 49.1% | 0.385 |
| LSTM A (price only) | 51.2% | 51.3% | 0.467 |
| LSTM B (price + VADER) | 50.4% | 50.4% | 0.482 |
| LSTM B (price + FinBERT) | 50.9% | 50.9% | 0.485 |

Sentiment contribution (B minus A, accuracy): VADER -1.4 points (95% CI -4.0 to +1.0); FinBERT +0.8 points
(95% CI -1.6 to +3.1). Both intervals include zero. Balanced accuracy gives the same conclusion.

By fold (ensemble accuracy), the models range from about 42% to 58%; the two newest folds (2026) are the weakest
for every model, and the naive baseline wins the 2026-01 to 2026-06 fold. Counts of folds above 50%:
LSTM A 5/8, LSTM B VADER 3/8, LSTM B FinBERT 6/8, XGBoost 2/8. B beats A in 3/8 folds with VADER and 4/8 with FinBERT.

## Reading the results
- Pooled accuracy for the LSTMs is 50-51%, inside the band expected from chance, so there is no reliable evidence of skill.
- Sentiment (VADER or FinBERT) adds no reliable improvement, which agrees with the single-split result and the lag analysis.
- Performance changes a lot between six-month periods, so any single test window (including the main one) is a noisy
  estimate. This is the reason to report ranges and intervals and not a single accuracy.

## Caveats
- The frozen settings were chosen on 2025-01 to 2025-09 validation data, which lies inside the 2025 folds' test windows.
  Tuning losses were almost flat, so the effect is expected to be small, but folds 5 and 6 are not fully clean.
- The last fold has only 61 days.
- Exploratory extra evidence; the headline result stays the pre-declared frozen split.
