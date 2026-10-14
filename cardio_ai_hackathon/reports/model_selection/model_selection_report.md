# Model selection report (provisional)

Generated 2026-10-10T14:39:50+00:00 by `scripts/build_model_selection.py`. Nothing was retrained; all numbers come from the saved Phase 1 reports.

> Educational decision-support prototype. Cross-validation on 303 patients from one centre does not establish clinical validity or diagnostic accuracy.

## Selected pipelines

| Task | Model | Inputs | Positive label | ROC-AUC | PR-AUC | Brier | Bal. acc. (0.5) | Rule check |
|---|---|---|---|---|---|---|---|---|
| Cath | `logreg_reduced` (LogisticRegression) | 48 | CAD | 0.922 ± 0.034 | 0.968 ± 0.015 | 0.103 ± 0.028 | 0.823 ± 0.060 | agrees |
| LAD | `rf_reduced` (RandomForestClassifier) | 48 | Stenotic | 0.853 ± 0.046 | 0.884 ± 0.043 | 0.164 ± 0.015 | 0.765 ± 0.051 | agrees |
| LCX | `logreg_clinical` (LogisticRegression) | 7 | Stenotic | 0.743 ± 0.078 | 0.632 ± 0.107 | 0.202 ± 0.028 | 0.659 ± 0.079 | agrees |
| RCA | `logreg_clinical` (LogisticRegression) | 7 | Stenotic | 0.740 ± 0.047 | 0.622 ± 0.056 | 0.197 ± 0.016 | 0.621 ± 0.045 | agrees |

Values are mean ± SD over 25 test folds. Brier: lower is better.

## Cath

**Selected:** `logreg_reduced` -> `models/phase1_candidates/Cath__logreg_reduced.joblib`

**Rationale:** Top mean AUC is rf_reduced, but logreg_reduced is within noise of it and has a lower (better) Brier score. Logistic regression is simpler and its explanations are exact, so it is preferred.

| Candidate | ROC-AUC | Brier |
|---|---|---|
| dummy_prior | 0.500 ± 0.000 | 0.205 ± 0.003 |
| logreg_clinical | 0.906 ± 0.037 | 0.104 ± 0.021 |
| logreg_all | 0.919 ± 0.035 | 0.106 ± 0.029 |
| rf_all | 0.922 ± 0.039 | 0.114 ± 0.015 |
| logreg_reduced (selected) | 0.922 ± 0.034 | 0.103 ± 0.028 |
| rf_reduced | 0.927 ± 0.036 | 0.112 ± 0.015 |

## LAD

**Selected:** `rf_reduced` -> `models/phase1_candidates/LAD__rf_reduced.joblib`

**Rationale:** rf_reduced has the highest mean AUC and beats logistic regression on most folds, which suggests real non-linear signal, so the more complex model is justified for this task.

| Candidate | ROC-AUC | Brier |
|---|---|---|
| dummy_prior | 0.500 ± 0.000 | 0.243 ± 0.001 |
| logreg_clinical | 0.796 ± 0.066 | 0.176 ± 0.031 |
| logreg_all | 0.820 ± 0.056 | 0.180 ± 0.034 |
| rf_all | 0.850 ± 0.047 | 0.166 ± 0.014 |
| logreg_reduced | 0.823 ± 0.055 | 0.177 ± 0.033 |
| rf_reduced (selected) | 0.853 ± 0.046 | 0.164 ± 0.015 |

## LCX

**Selected:** `logreg_clinical` -> `models/phase1_candidates/LCX__logreg_clinical.joblib`

**Rationale:** logreg_clinical (7 inputs) has the highest mean AUC; the 52-feature logistic regression does worse, consistent with overfitting on this weakest task.

| Candidate | ROC-AUC | Brier |
|---|---|---|
| dummy_prior | 0.500 ± 0.000 | 0.239 ± 0.001 |
| logreg_clinical (selected) | 0.743 ± 0.078 | 0.202 ± 0.028 |
| logreg_all | 0.685 ± 0.077 | 0.235 ± 0.038 |
| rf_all | 0.736 ± 0.068 | 0.205 ± 0.015 |
| logreg_reduced | 0.693 ± 0.072 | 0.231 ± 0.037 |
| rf_reduced | 0.731 ± 0.068 | 0.205 ± 0.015 |

## RCA

**Selected:** `logreg_clinical` -> `models/phase1_candidates/RCA__logreg_clinical.joblib`

**Rationale:** logreg_clinical (7 inputs) has the highest mean AUC and the lowest Brier score; the larger logistic regressions do not improve on it.

| Candidate | ROC-AUC | Brier |
|---|---|---|
| dummy_prior | 0.500 ± 0.000 | 0.235 ± 0.001 |
| logreg_clinical (selected) | 0.740 ± 0.047 | 0.197 ± 0.016 |
| logreg_all | 0.718 ± 0.059 | 0.217 ± 0.028 |
| rf_all | 0.719 ± 0.061 | 0.205 ± 0.012 |
| logreg_reduced | 0.726 ± 0.060 | 0.213 ± 0.027 |
| rf_reduced | 0.709 ± 0.055 | 0.207 ± 0.011 |

## Verified artifact contract

- Each artifact is a scikit-learn `Pipeline` (`preprocess` -> `model`) saved with joblib.
- Input: a pandas DataFrame whose column names match `input_features` exactly (extra columns are ignored; a missing column raises an error).
- Output: `predict_proba(X)[:, 1]` is the probability of the positive label (`CAD` for Cath, `Stenotic` for LAD/LCX/RCA); `classes_` is `[0, 1]`.
- Preprocessing (imputation, encoding, scaling) is inside the pipeline and must not be rebuilt separately.

## Revision history

- 2026-10-10: RCA changed from `logreg_reduced` to `logreg_clinical`. Selection-rule review: logreg_clinical has higher mean CV ROC-AUC (about 0.740 vs 0.726) and logreg_reduced had a worse Brier score in 19/25 paired folds. Decided after Phase 1 results were seen.

## Known limitations

- Performance comes from repeated stratified 5-fold CV (5 repeats) on 303 patients from one centre. There is no external validation; the models are not clinically validated and are for education/decision-support demonstration only.
- Each selection was made by comparing six candidates on the same folds, so the reported means are likely slightly optimistic.
- Threshold-based metrics use an untuned 0.5 threshold.
- Calibration was only summarised by the Brier score; probabilities are model scores, not validated risk estimates.
- Saved pipelines were refitted on all 303 patients. Predictions for those same patients are in-sample and optimistic; dataset patients shown in a demo should use the stored out-of-fold probabilities or be labelled as training patients.
- Row 93 (Cath=Normal, LAD=Stenotic) contradicts the dataset definition of CAD. It is kept as recorded; removing it changed CV AUC by at most 0.018.
- The four models are independent: out-of-fold, 10 patient(s) had a vessel predicted stenotic while CAD was predicted negative, and 18 had CAD predicted with no vessel predicted.
- The LCX model uses only 7 clinical inputs; ECG, echo and laboratory values do not affect its output.
- The dataset has no lesion location data, so vessel colouring shows whole-vessel model scores, never a lesion position or severity. Region RWMA is not a location.
- Sex level 'Fmale' is used as recorded (presumed 'Female', unconfirmed). 'Obesity' means BMI >= 25. Function Class and Region RWMA definitions are undocumented.
- The pipelines silently accept unknown categories and missing numeric values (encoded as zeros / median-imputed). Inputs must be validated before prediction.
