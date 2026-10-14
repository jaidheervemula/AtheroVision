# ML evaluation audit

Project: Multimodal AI Hackathon 2026, Track A (CAD and LAD/LCX/RCA stenosis).
Audit date: 2026-10-10. Scope: the evidence behind the four selected models.

> Educational decision-support prototype. Not a diagnostic tool. No external or clinical
> validation has been performed; nothing in this audit changes that.

**This was an audit only.** No model was trained, refitted or tuned, no threshold was
changed, and no existing data, model, code or report was modified. The only new file is
this report. The audit did two things only:

- It regenerated the cross-validation fold *assignments*, which needs no model fitting.
- It computed descriptive summaries from the saved out-of-fold predictions.

Labels used: **[Verified]** = checked directly against files or recomputed from them;
**[Not verifiable]** = the saved files do not contain enough to check;
**[Recommendation]** = advice, not a fact.

Selected models (from `config/model_selection.json`):

| Task | Model | Inputs |
|---|---|---|
| Cath | `logreg_reduced` | 48 |
| LAD | `rf_reduced` | 48 |
| LCX | `logreg_clinical` | 7 |
| RCA | `logreg_clinical` | 7 |

---

## 1. Evaluation protocol actually used

All of the following are **[Verified]** from `scripts/train_phase1.py` (lines 61–65, 171–182, 345–376), `models/phase1_candidates/metadata.json` and `reports/phase1_modeling/phase1_output.txt`.

**Data**

- All 303 patients. Row 93 is kept with its labels as recorded.
- Each task is its own binary problem, with 1 = `CAD` (Cath) or `Stenotic` (vessels).

**Splitting**

- `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)`, created separately for each task and stratified on that task's label.
- That gives 25 train/test splits per task (about 242 training and 61 test patients each).
- Within a task, all six candidate models used identical splits.

**Fitting**

- In every split, a fresh pipeline is fitted on the training part only. The pipeline contains imputation, one-hot encoding, scaling and the model, so no preprocessing is fitted on test patients.
- No hyperparameters were tuned. Every model uses fixed settings: L2 logistic regression with C = 1.0; a random forest with 300 trees, `min_samples_leaf=3` and seed 42.

**Metrics**

- Metrics are computed on each test fold. The threshold-based ones use a fixed threshold of 0.5.
- `cv_summary.csv` reports the mean, SD, minimum and maximum across the 25 folds.

**Model selection**

- Six candidates per task were compared on these same folds.
- The final choice used a rule whose numeric thresholds were set after the results were seen.
- RCA was revised from `logreg_reduced` to `logreg_clinical` after review.

**Saved models**

- Each saved `.joblib` pipeline was then **refitted on all 303 patients**.
- So the deployed models are not any of the models that were scored in cross-validation.

**Sensitivity analysis**

- The same CV was re-run without row 93 (302 patients, new splits), but only for `logreg_all` and `rf_all`. See section 6.

## 2. Are the out-of-fold predictions genuinely out-of-fold?

**What was verified [Verified]**

- `oof_predictions.csv` has, for each selected model and task: 303 rows, one per patient, no missing values, and `y_true` identical to the labels recomputed from the Excel file.
- The code stores each prediction only for that split's test patients (`oof[test_idx, repeat] = proba`), from a pipeline fitted on `train_idx`.
- Regenerating the splits with the same settings (splitting only, no fitting) shows:
  - every patient is a test patient exactly once per repeat;
  - training and test parts never overlap;
  - the regenerated per-fold `n_test` and `n_test_positive` match `cv_fold_metrics.csv` for all 25 folds of all four selected models.
- `oof_proba_mean` is the average of each patient's five predictions (one per repeat). Each of those five came from a model that did not see the patient.
- Independent evidence that these are not in-sample scores: pooled AUC from the saved out-of-fold predictions vs AUC of the saved refitted model on its own training data.

  | Task | Out-of-fold AUC | In-sample AUC (refitted model) |
  |---|---|---|
  | Cath | 0.925 | 0.977 |
  | LAD | 0.852 | 0.999 |
  | LCX | 0.743 | 0.770 |
  | RCA | 0.738 | 0.762 |

  The in-sample random forest is almost perfect, while its out-of-fold scores are not. Leaked predictions would not show that gap.

**What could not be verified [Not verifiable]**

- The script saved neither the fold assignments nor the five individual per-repeat predictions; only their mean and SD.
- So it is not possible to recompute the per-fold metrics from saved data, or to prove prediction by prediction which model produced each value.
- The conclusion "genuinely out-of-fold" therefore rests on the code, the matching fold counts and the in-sample gap. It is strong evidence, but indirect.

**Points to remember when using these predictions**

- `oof_proba_mean` is an average over five models trained on about 80% of the data. It is slightly smoother than any single model's output, and it is not the output of the deployed model refitted on 100%.
- `confusion_matrices_oof.csv` thresholds this average at 0.5, so it summarises the averaged predictions, not one particular model.

## 3. Metrics available for the selected models

All values below are **[Verified]** from `cv_summary.csv` and `cv_fold_metrics.csv`. Each is the mean ± SD over 25 test folds; threshold-based metrics use 0.5.

| Metric | Cath (`logreg_reduced`) | LAD (`rf_reduced`) | LCX (`logreg_clinical`) | RCA (`logreg_clinical`) |
|---|---|---|---|---|
| ROC-AUC | 0.922 ± 0.034 | 0.853 ± 0.046 | 0.743 ± 0.078 | 0.740 ± 0.047 |
| PR-AUC | 0.968 ± 0.015 | 0.884 ± 0.043 | 0.632 ± 0.107 | 0.622 ± 0.056 |
| Brier (lower is better) | 0.103 ± 0.028 | 0.164 ± 0.015 | 0.202 ± 0.028 | 0.197 ± 0.016 |
| Accuracy | 0.856 ± 0.047 | 0.785 ± 0.044 | 0.689 ± 0.075 | 0.669 ± 0.042 |
| Balanced accuracy | 0.823 ± 0.060 | 0.765 ± 0.051 | 0.659 ± 0.079 | 0.621 ± 0.045 |
| Precision | 0.899 ± 0.041 | 0.781 ± 0.050 | 0.627 ± 0.116 | 0.588 ± 0.084 |
| Recall (sensitivity) | 0.901 ± 0.054 | 0.886 ± 0.057 | 0.522 ± 0.108 | 0.431 ± 0.093 |
| Specificity | 0.745 ± 0.118 | 0.643 ± 0.109 | 0.797 ± 0.074 | 0.812 ± 0.069 |
| F1 | 0.899 ± 0.034 | 0.828 ± 0.034 | 0.568 ± 0.106 | 0.491 ± 0.076 |
| Prevalence | 0.713 | 0.584 | 0.393 | 0.376 |

Notes on reading these numbers:

- **Prevalence.** Cath's high AUC partly reflects an easier task with 71% prevalence. Cath and LAD precision and F1 are inflated by their high prevalence.
- **SD.** The "±" is the spread across overlapping folds. It is not a confidence interval or a standard error.
- **Dummy baseline.** The no-skill "always predict prevalence" baseline is in the same files and scores AUC 0.500 for every task.

## 4. Can calibration and threshold trade-offs be computed without retraining?

**Yes, from `oof_predictions.csv` [Verified]. I computed descriptive checks during this audit.**

**Calibration-in-the-large** (mean predicted probability vs observed prevalence):

| Task | Mean prediction | Prevalence |
|---|---|---|
| Cath | 0.711 | 0.713 |
| LAD | 0.580 | 0.584 |
| LCX | 0.392 | 0.393 |
| RCA | 0.377 | 0.376 |

**Calibration slope** (1.0 is ideal; under 1 means over-confident; over 1 means predictions are squeezed towards the middle):

| Task | Slope |
|---|---|
| Cath | 0.88 |
| LAD | **1.90** |
| LCX | 0.89 |
| RCA | 0.96 |

- **LAD.** The random forest is clearly under-dispersed. In the lowest fifth of predictions, the mean prediction was 0.27 but the observed rate was 0.10; in the highest fifth, 0.83 vs 0.92. Its probabilities should not be read at face value.
- **The three logistic models** look reasonably calibrated on these data.

**Threshold trade-offs** (sensitivity / specificity at several thresholds):

| Task | Threshold 0.3 | Threshold 0.5 | Threshold 0.7 |
|---|---|---|---|
| Cath | 0.94 / 0.60 | 0.90 / 0.75 | 0.84 / 0.85 |
| LAD | 0.98 / 0.26 | 0.89 / 0.63 | 0.50 / 0.91 |
| LCX | 0.84 / 0.53 | 0.52 / 0.80 | 0.11 / 0.92 |
| RCA | 0.83 / 0.53 | 0.43 / 0.81 | 0.14 / 0.96 |

**Caveats**

1. **Choosing a threshold on these predictions and then reporting performance at it on the same predictions is optimistic.** It is a second round of selection on the same data. The results are fine as a descriptive trade-off table, not as validated operating points.
2. **Each calibration bin holds only about 60 patients**, so calibration curves will have wide uncertainty.
3. **These are the cross-validation models' averaged outputs**, not the deployed refitted model's. The deployed model's calibration is assumed to be similar, but that was not measured.

## 5. Was a separate locked test set used?

**No [Verified].** All 303 patients were used both for cross-validation and for refitting the saved models, and they also informed model selection. There is no held-out test set.

The cross-validation folds are not an independent final test set: the same patients were used to compare candidates and to choose the selection rule.

Splitting off a "test set" from these 303 patients now would not be independent either, because every patient has already influenced the choices.

## 6. Remaining leakage risks and evaluation limitations

**Controlled [Verified]**

- **Target leakage.**
  - The script stops if a target column reaches the features.
  - `feature_names_in_` of all four saved pipelines contains no target.
  - The inference and explanation tests check both of these.
- **Preprocessing leakage.** All learned preprocessing is inside the Pipeline and is refitted on every training fold.

**Minor, label-free decisions made on the full dataset [Verified]**

These used no labels, so their risk is low. They should still be disclosed.

- Excluding constant and rare features (Exertional CP, CHF, LowTH Ang).
- Defining the "reduced" feature set (BMI is derived from weight and length; Obesity means BMI ≥ 25; Lymph and Neut are correlated).
- The encoding fix that sends 0/1 columns through the categorical branch.

**Selection optimism (the main limitation) [Verified that it occurred; its size is Not verifiable from current files]**

- **Several selection steps used the same folds.**
  - Six candidates per task were compared on the same 25 folds.
  - The selection-rule thresholds were set after the results were seen.
  - RCA was changed after review.
  - Phase 1 was also run twice: once before and once after the encoding fix.
- **Consequence.** The reported means for the *selected* models are probably somewhat optimistic. How much cannot be measured without nested cross-validation.

**Sensitivity analysis gap [Verified]**

- `sensitivity_inconsistent_row.csv` covers `logreg_all` and `rf_all` only; none of the four selected models were included.
- For those two models, removing row 93 changed AUC by −0.005 to +0.018.
- The effect on the selected models is **not verified**.

**Row 93 treatment [Verified, unchanged]**

- Recorded as Cath = Normal, LAD = Stenotic, LCX = Normal, RCA = Normal. It is kept as recorded in all analyses.
- Its out-of-fold Cath prediction is 0.979, a strong CAD score for a patient labelled Normal. That is consistent with the label conflict, but it does not prove which label is correct.

**Other limitations**

- **Single centre and time period; no external, temporal or clinical validation.**
- **No patient IDs.** Exact duplicate rows were ruled out, but near-duplicates or repeat visits cannot be checked.
- **Thresholds and calibration.**
  - Threshold-based metrics depend on the untuned 0.5 threshold.
  - The LAD probabilities are poorly calibrated (slope 1.90).
- **Independent models.**
  - Out-of-fold, 10 patients had a vessel predicted stenotic while CAD was predicted negative.
  - 18 patients had CAD predicted with no vessel predicted.
- **Uncertainty.** Fold SDs understate true uncertainty, because folds share training data. No confidence intervals have been computed yet.

## 7. Most defensible next evaluation protocol [Recommendation]

**A. Without retraining** (uses only the existing out-of-fold file):

1. **Bootstrap 95% confidence intervals** for pooled out-of-fold AUC, PR-AUC and Brier score, per selected model (for example, 2,000 resamples of patients).
2. **Calibration report:** reliability plots (quintiles), calibration slope and intercept, and Brier score. Flag LAD's under-dispersion explicitly in the documentation and in the UI.
3. **Threshold trade-off tables:** sensitivity, specificity, PPV, NPV and the number flagged across thresholds.
   - Present them as descriptive. Keep 0.5 as the documented default unless a threshold rule is fixed *before* looking. Any chosen threshold must be labelled "chosen on the same data".

**B. With retraining** (needs your explicit approval; the deployed models would not change):

4. **Nested repeated cross-validation of the whole selection procedure.**
   - Outer loop: 5 folds × several repeats.
   - Inner loop: the same six candidates compared with the same rule.
   - This estimates how much selection inflated the reported AUC, and is the honest replacement for a held-out test set with only 303 patients.
5. **Re-run the row 93 sensitivity analysis for the four selected models.**
6. **Save fold IDs and per-repeat predictions in every future run**, so that results can be audited directly.
7. **Optionally**, evaluate a recalibrated LAD model (calibration fitted inside the training folds) and compare it with the current one. Only adopt it if you decide to change the deployed model.

**Do not:**

- Carve a new "test set" from these 303 patients and present it as independent.
- Report thresholds tuned on the out-of-fold data as validated performance.
- Describe any of this as external or clinical validation.

## Unresolved questions

- The size of the selection optimism. This needs nested cross-validation (B4).
- Row 93's effect on the four selected models (B5).
- The calibration of the deployed refitted models themselves. This cannot be measured without new data. The cross-validation models' calibration is the best available proxy.
- The original meaning of the conflicting row 93 labels, and of undocumented codes (Function Class, Region RWMA). The dataset documentation doesn't resolve these.

## Files inspected (all read-only)

- `scripts/train_phase1.py` (matches the tested version)
- `reports/phase1_modeling/`: `cv_summary.csv`, `cv_fold_metrics.csv`, `oof_predictions.csv`, `sensitivity_inconsistent_row.csv`, `confusion_matrices_oof.csv`, `paired_auc_comparison.csv`, `phase1_output.txt`
- `config/model_selection.json`, `reports/model_selection/model_selection_report.md`
- `models/phase1_candidates/metadata.json`, and the four selected `.joblib` pipelines (loaded only to compute in-sample AUC for the comparison in section 2)
- `data/raw/extention of Z-Alizadeh sani dataset.xlsx`

All 55 project files (excluding `.venv` and `__pycache__`) had identical SHA-256 hashes before and after the audit. The only addition is this report.
