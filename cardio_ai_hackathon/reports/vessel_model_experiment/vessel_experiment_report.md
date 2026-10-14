# Experiment 2: LCX and RCA candidate models vs the clinical baseline

> **Educational prototype. Not a diagnostic tool.** Internal repeated cross-validation on 303 patients from one centre. This is not external or clinical validation, and nothing here changes the deployed models.

## Protocol

- **Outer validation:** RepeatedStratifiedKFold, 5 folds × 5 repeats, `random_state=42`. The splits are created separately for each target and stratified on it; within a target every candidate uses identical splits.
- **Preprocessing:** `train_phase1.make_pipeline`, refitted inside every training fold. No hyperparameter was tuned; the candidates below were fixed before any result was seen.
- **Inputs:** the project's existing feature sets, `clinical` (7) and `reduced` (48, the inference schema). `Cath`, `LAD`, `LCX` and `RCA` are never inputs (checked by the script).
- **Labels:** 1 = `Stenotic`, 0 = `Normal`. Threshold 0.5 for the threshold-based metrics.
- **Uncertainty:** 2000-replicate patient-level bootstrap (seed 20261012). Each resampled patient keeps all 5 out-of-fold predictions together; differences from the baseline use the same resamples (paired).

| Candidate | Inputs | Settings | Purpose |
|---|---|---|---|
| `baseline` | clinical (7) | cloned from deployed file: LogisticRegression(C=1.0, max_iter=5000) | current deployed model (cloned settings) |
| `lr_balanced_clinical` | clinical (7) | LogisticRegression(class_weight='balanced', max_iter=5000) | class weighting |
| `lr_strongreg_clinical` | clinical (7) | LogisticRegression(C=0.1, max_iter=5000) | stronger L2 regularization |
| `lr_strongreg_reduced` | reduced (48) | LogisticRegression(C=0.1, max_iter=5000) | stronger L2 regularization on the 48-input set |
| `rf_conservative_reduced` | reduced (48) | RandomForestClassifier(max_depth=5, min_samples_leaf=10, n_estimators=500, n_jobs=-1, random_state=42) | conservative random forest |

**Baseline reproduction:** LCX: REPRODUCED (OOF diff 9.7e-17, fold AUC diff 1.1e-16); RCA: REPRODUCED (OOF diff 9.7e-17, fold AUC diff 1.1e-16)

## LCX

Values are repeat-averaged out-of-fold metrics with 95% bootstrap intervals.

| Candidate | ROC-AUC | PR-AUC | Brier | Log loss | Sens. | Spec. | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.739 (0.684–0.793) | 0.595 (0.511–0.686) | 0.202 (0.181–0.223) | 0.587 | 0.52 | 0.80 | 0.62 | 0.57 | 0.69 |
| `lr_balanced_clinical` | 0.740 (0.684–0.794) | 0.597 (0.511–0.687) | 0.207 (0.186–0.230) | 0.604 | 0.72 | 0.66 | 0.58 | 0.64 | 0.68 |
| `lr_strongreg_clinical` | 0.740 (0.684–0.794) | 0.604 (0.517–0.693) | 0.201 (0.185–0.219) | 0.587 | 0.44 | 0.82 | 0.61 | 0.51 | 0.67 |
| `lr_strongreg_reduced` | 0.709 (0.651–0.764) | 0.585 (0.500–0.675) | 0.212 (0.190–0.234) | 0.611 | 0.46 | 0.80 | 0.60 | 0.52 | 0.67 |
| `rf_conservative_reduced` | 0.733 (0.677–0.785) | 0.594 (0.513–0.694) | 0.209 (0.195–0.223) | 0.605 | 0.27 | 0.90 | 0.63 | 0.37 | 0.65 |

**Paired difference vs baseline** (candidate − baseline; 95% bootstrap CI; number of the 5 repeats where the candidate scored higher):

| Candidate | ROC-AUC | PR-AUC | Brier (lower is better) | Log loss (lower is better) | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| `lr_balanced_clinical` | +0.000 (-0.001 to +0.002); 4/5 | +0.002 (-0.001 to +0.004); 3/5 | +0.006 (-0.003 to +0.015); 5/5 | +0.018 (-0.003 to +0.040); 5/5 | +0.198 (+0.145 to +0.259); 5/5 | -0.136 (-0.176 to -0.098); 0/5 |
| `lr_strongreg_clinical` | +0.000 (-0.010 to +0.012); 2/5 | +0.009 (-0.010 to +0.026); 5/5 | -0.000 (-0.007 to +0.006); 2/5 | -0.000 (-0.017 to +0.016); 2/5 | -0.086 (-0.132 to -0.043); 0/5 | +0.023 (+0.006 to +0.043); 5/5 |
| `lr_strongreg_reduced` | -0.030 (-0.072 to +0.008); 0/5 | -0.010 (-0.072 to +0.050); 1/5 | +0.010 (-0.003 to +0.025); 5/5 | +0.024 (-0.009 to +0.059); 5/5 | -0.064 (-0.138 to +0.008); 0/5 | +0.005 (-0.035 to +0.045); 2/5 |
| `rf_conservative_reduced` | -0.006 (-0.045 to +0.030); 2/5 | -0.001 (-0.059 to +0.067); 3/5 | +0.008 (-0.006 to +0.021); 5/5 | +0.019 (-0.015 to +0.051); 5/5 | -0.257 (-0.332 to -0.185); 0/5 | +0.104 (+0.063 to +0.148); 5/5 |

**ROC-AUC, read from the intervals:** no candidate has an ROC-AUC difference interval entirely above 0.

## RCA

Values are repeat-averaged out-of-fold metrics with 95% bootstrap intervals.

| Candidate | ROC-AUC | PR-AUC | Brier | Log loss | Sens. | Spec. | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.737 (0.681–0.790) | 0.600 (0.509–0.695) | 0.197 (0.177–0.218) | 0.570 | 0.43 | 0.81 | 0.58 | 0.50 | 0.67 |
| `lr_balanced_clinical` | 0.737 (0.681–0.791) | 0.600 (0.507–0.695) | 0.206 (0.185–0.227) | 0.595 | 0.67 | 0.63 | 0.52 | 0.59 | 0.65 |
| `lr_strongreg_clinical` | 0.737 (0.679–0.791) | 0.591 (0.499–0.689) | 0.200 (0.183–0.216) | 0.581 | 0.36 | 0.87 | 0.62 | 0.45 | 0.68 |
| `lr_strongreg_reduced` | 0.727 (0.673–0.782) | 0.593 (0.509–0.690) | 0.202 (0.182–0.223) | 0.587 | 0.44 | 0.81 | 0.59 | 0.50 | 0.67 |
| `rf_conservative_reduced` | 0.719 (0.662–0.773) | 0.583 (0.495–0.680) | 0.208 (0.193–0.223) | 0.602 | 0.23 | 0.92 | 0.65 | 0.34 | 0.66 |

**Paired difference vs baseline** (candidate − baseline; 95% bootstrap CI; number of the 5 repeats where the candidate scored higher):

| Candidate | ROC-AUC | PR-AUC | Brier (lower is better) | Log loss (lower is better) | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| `lr_balanced_clinical` | +0.001 (-0.002 to +0.003); 5/5 | +0.000 (-0.003 to +0.003); 3/5 | +0.008 (-0.003 to +0.020); 5/5 | +0.024 (-0.002 to +0.050); 5/5 | +0.235 (+0.173 to +0.304); 5/5 | -0.178 (-0.224 to -0.132); 0/5 |
| `lr_strongreg_clinical` | +0.000 (-0.013 to +0.014); 3/5 | -0.009 (-0.028 to +0.010); 0/5 | +0.002 (-0.005 to +0.010); 5/5 | +0.011 (-0.007 to +0.028); 5/5 | -0.075 (-0.123 to -0.035); 0/5 | +0.057 (+0.032 to +0.086); 5/5 |
| `lr_strongreg_reduced` | -0.010 (-0.056 to +0.037); 0/5 | -0.007 (-0.075 to +0.066); 1/5 | +0.005 (-0.011 to +0.021); 5/5 | +0.016 (-0.021 to +0.054); 5/5 | +0.005 (-0.083 to +0.096); 4/5 | +0.002 (-0.044 to +0.052); 3/5 |
| `rf_conservative_reduced` | -0.018 (-0.055 to +0.024); 0/5 | -0.018 (-0.077 to +0.048); 1/5 | +0.010 (-0.003 to +0.024); 5/5 | +0.032 (+0.002 to +0.062); 5/5 | -0.204 (-0.285 to -0.125); 0/5 | +0.113 (+0.068 to +0.161); 5/5 |

**ROC-AUC, read from the intervals:** no candidate has an ROC-AUC difference interval entirely above 0.

## How to read this, and its limits

- **Discrimination vs calibration vs threshold.** ROC-AUC and PR-AUC measure ranking. Brier and log loss measure probability accuracy. Sensitivity, specificity, precision, F1 and accuracy depend on the 0.5 threshold.
- **Class weighting** (`lr_balanced_clinical`) deliberately pushes probabilities up for the minority (Stenotic) class. Expect more sensitivity, less specificity and worse calibration at 0.5, even with similar ROC-AUC. That's an operating-point change, not better discrimination.
- **Selection optimism.** Four new candidates per target (eight comparisons) were scored on the same folds used to judge them, on top of the six Phase 1 candidates already compared on these folds. If you pick whichever looks best here, its estimate is biased upwards. A small positive difference with an interval crossing 0 is not evidence of improvement. Adopting a candidate would need confirmation, ideally by nested cross-validation.
- **What the intervals capture.** They reflect only which patients were evaluated. They don't include retraining variability or the selection steps above, so treat them as lower bounds on the uncertainty. There are no corrections for multiple comparisons.
- **Small sample.** LCX has 119 and RCA 114 positives. Several categorical levels in the 48-input set occur in 5–15 patients (`rare_categories.csv`), so some training folds miss them entirely.
- **Coherence.** These are independent vessel models; any change could alter agreement with the CAD model.

## Warnings during fitting

- None.

All warnings are listed in full in `fit_warnings.csv`; none were suppressed.

## Files

`model_comparison.csv` (all metrics with CIs and repeat SDs), `paired_differences_vs_baseline.csv` (every metric, with CI, repeat and fold win counts), `repeat_metrics.csv`, `fold_metrics.csv`, `oof_predictions.csv` (target × candidate × patient × repeat, with outer fold IDs), `rare_categories.csv`, `fit_warnings.csv`, `run_config.json` (seeds, settings, feature sets, input file hashes) and `vessel_experiment_log.txt`.

**No candidate is adopted by this script.** The decision is left to review of these results.
