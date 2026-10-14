# AtheroVision model card

> **Educational decision-support prototype only.** Predictions are model estimates, not confirmed
> diagnoses. This application is not a substitute for professional clinical assessment or diagnostic imaging.

Every number below is copied from the project's saved evaluation reports
(`cardio_ai_hackathon/reports/final_ml_evaluation/bootstrap_confidence_intervals.csv` and, for LAD,
`reports/lad_calibration/calibration_comparison.csv`). Nothing was retrained to produce this card.

## What the models do

Four separate binary classifiers, trained on the same 303 patients:

| Target | Question | Model | Inputs |
|---|---|---|---|
| CAD | Coronary artery disease on angiography? | Logistic regression (`logreg_reduced`) | 48 |
| LAD | ≥50% stenosis in the left anterior descending artery? | Random forest + sigmoid calibration (`rf_reduced_sigmoid`) | 48 |
| LCX | ≥50% stenosis in the left circumflex artery? | Logistic regression (`logreg_clinical`) | 7 |
| RCA | ≥50% stenosis in the right coronary artery? | Logistic regression (`logreg_clinical`) | 7 |

**Data:** the Z-Alizadeh Sani dataset (extension), UCI Machine Learning Repository. It has 303 patients from one centre, with 54 clinical, ECG, echo and laboratory inputs and angiography outcomes.

## How well they do (cross-validated)

Repeated stratified 5-fold cross-validation (5 repeats), with 95% bootstrap intervals. Sensitivity and specificity use the fixed 0.5 threshold.

| Target | Prevalence | ROC-AUC | Sensitivity | Specificity | Brier | Calibration slope (ideal 1) |
|---|---|---|---|---|---|---|
| CAD | 71% | **0.92** (0.89–0.95) | 0.90 (0.86–0.94) | 0.74 (0.66–0.83) | 0.10 | 0.80 (0.65–1.03) |
| LAD | 58% | **0.84** (0.80–0.89) | 0.86 (0.81–0.91) | 0.67 (0.60–0.75) | 0.15 | 1.00 (0.81–1.26) |
| LCX | 39% | **0.74** (0.68–0.80) | 0.52 (0.44–0.61) | 0.80 (0.74–0.85) | 0.20 | 0.87 (0.62–1.18) |
| RCA | 38% | **0.74** (0.68–0.79) | 0.43 (0.35–0.52) | 0.81 (0.76–0.86) | 0.20 | 0.94 (0.71–1.23) |

How to read this:

- **CAD** is the strongest model. A calibration slope a little below 1 means its probabilities are somewhat too extreme. Read 95% as "very likely", not as an exact 95%.
- **LAD** was recalibrated (sigmoid), which brought its slope from 1.82 to 1.00. Its probabilities are the best-calibrated of the four.
- **LCX and RCA** rank patients only moderately well. At the 0.5 threshold they **miss about half of the stenotic vessels** (sensitivity 0.43–0.52), so the app flags them from 30% instead (see below). A low LCX or RCA score still does not rule out disease.

## Flagging cut-offs (what the app uses)

The table above uses the 0.5 cut-off for every target. At 0.5, LCX and RCA miss about half of the
narrowed arteries, which is the wrong trade-off for a screening-style tool. The app therefore flags
each target at the **highest cut-off (0.05 steps, never above 0.5) that catches at least 80% of
diseased patients in cross-validation** (`cardio_ai_hackathon/api/operating_points.py`). The
models and their probabilities are unchanged; only the flag moves.

| Target | Cut-off | Catches (sensitivity) | Correctly clears (specificity) |
|---|---|---|---|
| CAD | 50% (unchanged) | 90% | 74% |
| LAD | 50% (unchanged) | 86% | 67% |
| LCX | **30%** | **84%** (52% at 0.5) | 54% (80% at 0.5) |
| RCA | **30%** | **83%** (43% at 0.5) | 53% (81% at 0.5) |

In patient numbers (each patient's mean cross-validated prediction, 303 patients), LCX misses fall
from 57 to 19 and RCA misses from 65 to 19. In exchange, false alarms rise from 36 to 86 (LCX) and
from 35 to 89 (RCA). The cut-off was chosen on the same cross-validation results it is reported on, so the
figures at 30% are slightly optimistic. It has not been validated on new patients.

## Reality check (per prediction)

For each score, the app shows how many of the 40 dataset patients with the closest *cross-validated*
score actually had the condition on angiography, with a Wilson 95% interval
(`cardio_ai_hackathon/api/reality_check.py`). It turns a probability into an observed frequency.
It uses the same 303 patients, so it is an internal check, not external validation. When a dataset patient is loaded, that patient is left out of their own comparison.

## Explanations

- **CAD, LCX, RCA (logistic regression):** each contribution is exact and in log-odds. They add up to the model's output.
- **LAD (random forest):** each contribution is approximate. It is the change in probability when that one input is replaced by values from the training data, and the contributions do not add up exactly.
- **All of them** are statistical associations in this dataset, **not causes**, and not medical advice.

## Known limitations

1. **No external validation.** All of the metrics come from 303 patients at one centre. Performance on other populations, hospitals or equipment is unknown.
2. **Model choice was made on the same folds.** Six candidates were compared per target on the same cross-validation folds, so the reported means are probably slightly optimistic.
3. **Cut-offs come from a fixed rule, not clinical tuning.** CAD and LAD use 0.5; LCX and RCA use 0.3 so that at least 80% of narrowed arteries are caught, with many more false alarms. `reports/final_ml_evaluation/threshold_tradeoffs.csv` shows the full trade-off.
4. **The deployed models were refitted on all 303 patients.** A live prediction for a dataset patient is therefore in-sample and optimistic. The app's **Dataset patient** loader also shows the cross-validated (held-out) prediction and the true angiography result for exactly this reason.
5. **The models are independent.** The CAD model and the vessel models can disagree. Out of fold, 10 patients had a vessel predicted stenotic with CAD predicted negative, and 18 had CAD predicted with no vessel predicted. The app flags disagreement and never forces agreement.
6. **Several inputs need specialist tests.** For CAD and LAD, echo ejection fraction and regional wall-motion abnormality (RWMA) are among the five strongest single predictors (`reports/audit_step2/univariate_auc_screen.csv`); typical chest pain is the strongest. That makes AtheroVision a tool for after the cardiology work-up, not for first-contact screening.
7. **Vessel colouring is a whole-vessel score.** The dataset has no lesion locations, so the 3D heart shows a probability per artery, never where or how severe a lesion is.
8. **Some inputs are not used.** CHF, exertional chest pain, low-threshold angina, neutrophils and obesity do not affect any model. The form marks them "not used by models". The LCX and RCA models use only 7 clinical inputs, so ECG, echo and lab values do not change those two scores.
9. **There are data quirks.** Row 93 (CAD = Normal, LAD = Stenotic) contradicts the dataset's own definition of CAD and is kept as recorded. The app shows a ⚠ Label conflict note when this patient is loaded. Sex is recorded as "Fmale", presumed to mean female. The definitions of NYHA function class and RWMA are undocumented.

## Intended use

Teaching and demonstration of how clinical features relate to angiographic coronary disease in one
published dataset. AtheroVision is not for diagnosis, triage or treatment decisions.
