# Experiment 3: out-of-fold error analysis

> **Educational prototype. Not a diagnostic tool.** Analysis of internal cross-validation predictions for 303 patients from one centre. Subgroup results are exploratory descriptions, not clinical conclusions. Nothing here is clinical validation.

## What was analysed

- Out-of-fold predictions only (`reports/final_ml_evaluation/oof_per_repeat_selected.csv`). Each patient has 5 predictions, one per CV repeat, each from a model that never saw them. No deployed-model (in-sample) prediction is used.
- No model was fitted and nothing existing was changed. The dataset is used only to describe patients and define subgroups; `Cath`, `LAD`, `LCX` and `RCA` serve only as the true labels being compared.
- Selected models: Cath `logreg_reduced`, LAD `rf_reduced`, LCX `logreg_clinical`, RCA `logreg_clinical`. Positive labels: `CAD` (Cath) and `Stenotic` (vessels). Threshold 0.5 from `config/model_selection.json`.
- **Alignment checks:** every one passed (`data_checks.csv`).
  - Labels match the dataset.
  - Each patient is predicted exactly once per repeat, using the regenerated Phase 1 splits.
  - For LAD, LCX and RCA, the regenerated fold IDs equal the fold IDs saved independently by the calibration and vessel experiments.
  - Row 93 (Cath = Normal but LAD = Stenotic) is kept as recorded and flagged in `patient_error_frequency.csv`.

**Two kinds of summary** (never mixed):
- **Per-repeat / repeat-averaged:** metrics on one repeat's 303 predictions, then averaged.
- **Patient-level:** how many of a patient's 5 repeats were misclassified.

The 5 predictions for one patient are never counted as 5 patients.

## Overall errors at threshold 0.5

| Task | Positives / negatives | FP per repeat | FN per repeat | Sensitivity | Specificity | Precision | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Cath | 216 / 87 | 22.2 | 21.4 | 0.901 (0.89–0.91) | 0.745 (0.69–0.80) | 0.898 | 0.899 | 0.856 | 0.920 |
| LAD | 177 / 126 | 45.0 | 20.2 | 0.886 (0.88–0.89) | 0.643 (0.61–0.67) | 0.777 | 0.828 | 0.785 | 0.847 |
| LCX | 119 / 184 | 37.4 | 56.8 | 0.523 (0.50–0.56) | 0.797 (0.78–0.81) | 0.624 | 0.569 | 0.689 | 0.739 |
| RCA | 114 / 189 | 35.6 | 64.8 | 0.432 (0.43–0.44) | 0.812 (0.80–0.84) | 0.581 | 0.495 | 0.669 | 0.737 |

Values are repeat-averaged; brackets give the lowest and highest of the 5 repeats. Recall equals sensitivity. Confusion matrices for each repeat, the repeat average and the patient-mean probability are in `confusion_matrices.csv`. Bootstrap CIs for the overall metrics are in `reports/final_ml_evaluation/bootstrap_confidence_intervals.csv`.

## Patient-level error stability

| Task | Always correct | Missed in all 5 repeats (persistent FN) | Falsely flagged in all 5 (persistent FP) | Unstable (wrong in 1–4 repeats) |
|---|---|---|---|---|
| Cath | 241 | 13 | 14 | 35 |
| LAD | 219 | 14 | 35 | 35 |
| LCX | 181 | 42 | 28 | 52 |
| RCA | 186 | 60 | 26 | 31 |

Patients misclassified in all 5 repeats for two or more tasks: **70**. Persistent errors mean the model consistently places these patients on the wrong side of 0.5. Unstable patients sit close to the threshold, where small training differences flip the result. Row 93's per-task status is in `patient_error_frequency.csv` (`inconsistent_target_row = True`).

**Descriptive profile of persistent errors** (`persistent_error_profiles.csv`). For each task, patients missed in all 5 repeats are compared with correctly classified patients of the same true class (medians and percentages). These groups are small, so read them as hints for further investigation, not explanations.

- **Cath**: 13 persistent FN vs 187 correctly detected positives. Typical chest pain 15% vs 78%, Region RWMA ≥ 1 0% vs 43%, male 69% vs 58%, DM 8% vs 41%, median age 50 vs 62.
- **LAD**: 14 persistent FN vs 147 correctly detected positives. Typical chest pain 7% vs 84%, Region RWMA ≥ 1 7% vs 49%, male 71% vs 61%, DM 21% vs 37%, median age 53 vs 62.
- **LCX**: 42 persistent FN vs 49 correctly detected positives. Typical chest pain 50% vs 88%, Region RWMA ≥ 1 33% vs 31%, male 52% vs 73%, DM 26% vs 51%, median age 56 vs 68.
- **RCA**: 60 persistent FN vs 45 correctly detected positives. Typical chest pain 58% vs 96%, Region RWMA ≥ 1 30% vs 36%, male 63% vs 73%, DM 20% vs 78%, median age 59 vs 64.

## Subgroups (exploratory)

Each subgroup's metrics are repeat-averaged, with 95% patient-level bootstrap intervals (1000 resamples within the subgroup, seed 20261013). A metric is **withheld** when its subgroup has fewer than 10 patients of the relevant class. Subgroups overlap and many are compared, so some differences will occur by chance. Where intervals overlap substantially, don't read the gap as a real difference. Full table: `subgroup_metrics.csv`.

### Cath

| Subgroup | Level | n (pos/neg) | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| Age group | <50 | 58 (29/29) | 0.77 (0.62–0.90) | 0.84 (0.72–0.95) | 0.91 (0.83–0.97) |
| Age group | 50-59 | 104 (65/39) | 0.88 (0.81–0.95) | 0.83 (0.71–0.92) | 0.92 (0.87–0.97) |
| Age group | 60-69 | 85 (70/15) | 0.93 (0.87–0.98) | 0.52 (0.31–0.70) | 0.89 (0.80–0.95) |
| Age group | 70+ | 56 (52/4) | 0.96 (0.91–1.00) | withheld | withheld |
| Sex (as recorded) | Fmale | 127 (86/41) | 0.92 (0.87–0.97) | 0.83 (0.74–0.92) | 0.95 (0.91–0.98) |
| Sex (as recorded) | Male | 176 (130/46) | 0.89 (0.84–0.93) | 0.67 (0.54–0.80) | 0.89 (0.84–0.94) |
| Diabetes (DM) | No | 213 (136/77) | 0.86 (0.80–0.91) | 0.75 (0.66–0.83) | 0.89 (0.85–0.93) |
| Diabetes (DM) | Yes | 90 (80/10) | 0.97 (0.93–1.00) | 0.72 (0.45–0.94) | 0.97 (0.93–1.00) |
| Hypertension (HTN) | No | 124 (69/55) | 0.85 (0.76–0.92) | 0.81 (0.71–0.89) | 0.91 (0.86–0.95) |
| Hypertension (HTN) | Yes | 179 (147/32) | 0.93 (0.88–0.96) | 0.64 (0.47–0.79) | 0.91 (0.85–0.95) |
| Typical chest pain | No | 139 (62/77) | 0.73 (0.63–0.83) | 0.81 (0.74–0.88) | 0.86 (0.79–0.91) |
| Typical chest pain | Yes | 164 (154/10) | 0.97 (0.95–0.99) | 0.22 (0.05–0.43) | 0.85 (0.72–0.95) |
| Current smoker | No | 240 (167/73) | 0.90 (0.86–0.94) | 0.77 (0.68–0.85) | 0.93 (0.90–0.96) |
| Current smoker | Yes | 63 (49/14) | 0.90 (0.81–0.98) | 0.61 (0.37–0.85) | 0.87 (0.75–0.96) |
| Dyslipidaemia (DLP) | No | 191 (137/54) | 0.91 (0.86–0.95) | 0.70 (0.59–0.80) | 0.91 (0.87–0.95) |
| Dyslipidaemia (DLP) | Yes | 112 (79/33) | 0.89 (0.83–0.96) | 0.81 (0.70–0.91) | 0.93 (0.89–0.97) |
| BMI group | <25 | 92 (67/25) | 0.93 (0.87–0.97) | 0.74 (0.58–0.89) | 0.94 (0.89–0.98) |
| BMI group | 25-30 | 140 (104/36) | 0.90 (0.84–0.95) | 0.77 (0.65–0.89) | 0.90 (0.85–0.95) |
| BMI group | >=30 | 71 (45/26) | 0.87 (0.77–0.96) | 0.72 (0.57–0.86) | 0.93 (0.87–0.97) |
| EF-TTE | <50% | 127 (109/18) | 0.93 (0.88–0.97) | 0.59 (0.36–0.81) | 0.91 (0.86–0.96) |
| EF-TTE | >=50% | 176 (107/69) | 0.87 (0.82–0.93) | 0.79 (0.71–0.86) | 0.91 (0.87–0.95) |
| Region RWMA | 0 | 217 (134/83) | 0.85 (0.79–0.90) | 0.76 (0.67–0.83) | 0.89 (0.85–0.93) |
| Region RWMA | >=1 | 86 (82/4) | 0.98 (0.96–1.00) | withheld | withheld |

### LAD

| Subgroup | Level | n (pos/neg) | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| Age group | <50 | 58 (22/36) | 0.78 (0.61–0.94) | 0.84 (0.72–0.95) | 0.88 (0.78–0.97) |
| Age group | 50-59 | 104 (54/50) | 0.90 (0.82–0.98) | 0.74 (0.62–0.85) | 0.89 (0.83–0.94) |
| Age group | 60-69 | 85 (58/27) | 0.88 (0.80–0.95) | 0.47 (0.29–0.64) | 0.73 (0.61–0.84) |
| Age group | 70+ | 56 (43/13) | 0.92 (0.85–0.97) | 0.06 (0.00–0.15) | 0.70 (0.55–0.83) |
| Sex (as recorded) | Fmale | 127 (67/60) | 0.90 (0.83–0.96) | 0.64 (0.53–0.74) | 0.83 (0.76–0.90) |
| Sex (as recorded) | Male | 176 (110/66) | 0.88 (0.82–0.93) | 0.65 (0.55–0.75) | 0.86 (0.80–0.91) |
| Diabetes (DM) | No | 213 (116/97) | 0.86 (0.80–0.92) | 0.72 (0.64–0.80) | 0.88 (0.84–0.93) |
| Diabetes (DM) | Yes | 90 (61/29) | 0.93 (0.87–0.98) | 0.39 (0.23–0.58) | 0.70 (0.59–0.82) |
| Hypertension (HTN) | No | 124 (59/65) | 0.87 (0.77–0.95) | 0.79 (0.69–0.88) | 0.90 (0.84–0.95) |
| Hypertension (HTN) | Yes | 179 (118/61) | 0.89 (0.84–0.94) | 0.49 (0.37–0.61) | 0.79 (0.71–0.85) |
| Typical chest pain | No | 139 (47/92) | 0.64 (0.52–0.77) | 0.82 (0.75–0.89) | 0.81 (0.73–0.88) |
| Typical chest pain | Yes | 164 (130/34) | 0.97 (0.95–0.99) | 0.16 (0.07–0.29) | 0.72 (0.63–0.81) |
| Current smoker | No | 240 (135/105) | 0.89 (0.84–0.93) | 0.64 (0.55–0.72) | 0.84 (0.78–0.88) |
| Current smoker | Yes | 63 (42/21) | 0.87 (0.77–0.95) | 0.68 (0.48–0.85) | 0.87 (0.78–0.96) |
| Dyslipidaemia (DLP) | No | 191 (115/76) | 0.88 (0.82–0.93) | 0.66 (0.56–0.76) | 0.88 (0.82–0.92) |
| Dyslipidaemia (DLP) | Yes | 112 (62/50) | 0.89 (0.82–0.96) | 0.62 (0.49–0.73) | 0.80 (0.72–0.88) |
| BMI group | <25 | 92 (58/34) | 0.90 (0.81–0.97) | 0.61 (0.45–0.76) | 0.85 (0.77–0.93) |
| BMI group | 25-30 | 140 (83/57) | 0.87 (0.81–0.94) | 0.64 (0.52–0.76) | 0.85 (0.79–0.90) |
| BMI group | >=30 | 71 (36/35) | 0.88 (0.77–0.97) | 0.69 (0.53–0.83) | 0.82 (0.72–0.91) |
| EF-TTE | <50% | 127 (99/28) | 0.96 (0.92–0.99) | 0.45 (0.28–0.63) | 0.85 (0.77–0.92) |
| EF-TTE | >=50% | 176 (78/98) | 0.79 (0.70–0.87) | 0.70 (0.61–0.78) | 0.80 (0.73–0.86) |
| Region RWMA | 0 | 217 (102/115) | 0.82 (0.74–0.88) | 0.68 (0.59–0.75) | 0.81 (0.76–0.86) |
| Region RWMA | >=1 | 86 (75/11) | 0.98 (0.95–1.00) | 0.27 (0.04–0.53) | 0.81 (0.66–0.92) |

### LCX

| Subgroup | Level | n (pos/neg) | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| Age group | <50 | 58 (12/46) | 0.08 (0.00–0.20) | 0.99 (0.97–1.00) | 0.71 (0.55–0.87) |
| Age group | 50-59 | 104 (33/71) | 0.31 (0.17–0.45) | 0.94 (0.90–0.98) | 0.76 (0.66–0.85) |
| Age group | 60-69 | 85 (42/43) | 0.62 (0.48–0.75) | 0.65 (0.52–0.78) | 0.65 (0.53–0.76) |
| Age group | 70+ | 56 (32/24) | 0.78 (0.65–0.90) | 0.25 (0.11–0.40) | 0.54 (0.40–0.69) |
| Sex (as recorded) | Fmale | 127 (42/85) | 0.40 (0.27–0.53) | 0.85 (0.78–0.92) | 0.74 (0.64–0.82) |
| Sex (as recorded) | Male | 176 (77/99) | 0.59 (0.49–0.69) | 0.75 (0.67–0.82) | 0.73 (0.65–0.80) |
| Diabetes (DM) | No | 213 (73/140) | 0.44 (0.34–0.55) | 0.84 (0.79–0.89) | 0.74 (0.67–0.81) |
| Diabetes (DM) | Yes | 90 (46/44) | 0.65 (0.52–0.76) | 0.65 (0.52–0.77) | 0.68 (0.56–0.79) |
| Hypertension (HTN) | No | 124 (36/88) | 0.32 (0.20–0.46) | 0.89 (0.83–0.95) | 0.73 (0.64–0.83) |
| Hypertension (HTN) | Yes | 179 (83/96) | 0.61 (0.52–0.71) | 0.71 (0.62–0.79) | 0.71 (0.63–0.79) |
| Typical chest pain | No | 139 (33/106) | 0.25 (0.12–0.38) | 0.95 (0.92–0.98) | 0.78 (0.67–0.87) |
| Typical chest pain | Yes | 164 (86/78) | 0.63 (0.53–0.72) | 0.58 (0.49–0.68) | 0.60 (0.52–0.69) |
| Current smoker | No | 240 (91/149) | 0.51 (0.42–0.60) | 0.80 (0.74–0.86) | 0.75 (0.68–0.81) |
| Current smoker | Yes | 63 (28/35) | 0.56 (0.39–0.72) | 0.78 (0.64–0.90) | 0.70 (0.56–0.82) |
| Dyslipidaemia (DLP) | No | 191 (76/115) | 0.47 (0.37–0.56) | 0.76 (0.69–0.83) | 0.72 (0.64–0.79) |
| Dyslipidaemia (DLP) | Yes | 112 (43/69) | 0.61 (0.47–0.75) | 0.86 (0.79–0.93) | 0.77 (0.67–0.86) |
| BMI group | <25 | 92 (39/53) | 0.53 (0.39–0.67) | 0.64 (0.51–0.76) | 0.65 (0.54–0.76) |
| BMI group | 25-30 | 140 (58/82) | 0.51 (0.40–0.63) | 0.84 (0.77–0.91) | 0.76 (0.68–0.83) |
| BMI group | >=30 | 71 (22/49) | 0.54 (0.35–0.72) | 0.90 (0.82–0.96) | 0.79 (0.68–0.90) |
| EF-TTE | <50% | 127 (59/68) | 0.59 (0.47–0.71) | 0.66 (0.57–0.77) | 0.67 (0.58–0.76) |
| EF-TTE | >=50% | 176 (60/116) | 0.46 (0.35–0.57) | 0.87 (0.81–0.92) | 0.78 (0.70–0.85) |
| Region RWMA | 0 | 217 (76/141) | 0.52 (0.41–0.63) | 0.85 (0.79–0.90) | 0.78 (0.71–0.84) |
| Region RWMA | >=1 | 86 (43/43) | 0.53 (0.40–0.64) | 0.63 (0.49–0.75) | 0.60 (0.47–0.71) |

### RCA

| Subgroup | Level | n (pos/neg) | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| Age group | <50 | 58 (11/47) | 0.18 (0.00–0.44) | 0.97 (0.93–1.00) | 0.82 (0.69–0.93) |
| Age group | 50-59 | 104 (33/71) | 0.32 (0.16–0.47) | 0.90 (0.84–0.95) | 0.75 (0.65–0.83) |
| Age group | 60-69 | 85 (42/43) | 0.52 (0.38–0.68) | 0.67 (0.53–0.80) | 0.66 (0.53–0.77) |
| Age group | 70+ | 56 (28/28) | 0.54 (0.37–0.70) | 0.54 (0.37–0.71) | 0.57 (0.41–0.71) |
| Sex (as recorded) | Fmale | 127 (39/88) | 0.36 (0.23–0.51) | 0.88 (0.81–0.94) | 0.80 (0.73–0.87) |
| Sex (as recorded) | Male | 176 (75/101) | 0.47 (0.36–0.58) | 0.75 (0.67–0.83) | 0.68 (0.60–0.76) |
| Diabetes (DM) | No | 213 (62/151) | 0.19 (0.10–0.28) | 0.88 (0.84–0.93) | 0.70 (0.63–0.77) |
| Diabetes (DM) | Yes | 90 (52/38) | 0.72 (0.59–0.82) | 0.52 (0.38–0.66) | 0.67 (0.55–0.78) |
| Hypertension (HTN) | No | 124 (36/88) | 0.22 (0.09–0.36) | 0.92 (0.86–0.97) | 0.75 (0.66–0.83) |
| Hypertension (HTN) | Yes | 179 (78/101) | 0.53 (0.42–0.64) | 0.72 (0.64–0.80) | 0.71 (0.64–0.78) |
| Typical chest pain | No | 139 (31/108) | 0.11 (0.02–0.23) | 0.95 (0.90–0.98) | 0.78 (0.70–0.85) |
| Typical chest pain | Yes | 164 (83/81) | 0.55 (0.45–0.66) | 0.63 (0.54–0.72) | 0.64 (0.55–0.72) |
| Current smoker | No | 240 (88/152) | 0.46 (0.36–0.56) | 0.82 (0.76–0.88) | 0.77 (0.71–0.82) |
| Current smoker | Yes | 63 (26/37) | 0.34 (0.17–0.52) | 0.76 (0.63–0.88) | 0.58 (0.44–0.72) |
| Dyslipidaemia (DLP) | No | 191 (69/122) | 0.36 (0.25–0.46) | 0.83 (0.76–0.89) | 0.72 (0.64–0.79) |
| Dyslipidaemia (DLP) | Yes | 112 (45/67) | 0.54 (0.41–0.69) | 0.78 (0.69–0.88) | 0.76 (0.67–0.84) |
| BMI group | <25 | 92 (35/57) | 0.46 (0.29–0.63) | 0.74 (0.63–0.83) | 0.70 (0.59–0.79) |
| BMI group | 25-30 | 140 (55/85) | 0.43 (0.30–0.54) | 0.83 (0.75–0.90) | 0.74 (0.65–0.81) |
| BMI group | >=30 | 71 (24/47) | 0.41 (0.22–0.61) | 0.87 (0.77–0.94) | 0.77 (0.66–0.87) |
| EF-TTE | <50% | 127 (55/72) | 0.48 (0.36–0.61) | 0.67 (0.57–0.78) | 0.64 (0.56–0.74) |
| EF-TTE | >=50% | 176 (59/117) | 0.39 (0.28–0.51) | 0.90 (0.84–0.95) | 0.79 (0.72–0.85) |
| Region RWMA | 0 | 217 (77/140) | 0.42 (0.31–0.53) | 0.86 (0.80–0.91) | 0.78 (0.72–0.84) |
| Region RWMA | >=1 | 86 (37/49) | 0.45 (0.30–0.61) | 0.68 (0.56–0.80) | 0.62 (0.50–0.73) |

## Limitations

- 303 patients from one centre. Subgroups of 50–100 patients give very wide intervals; no subgroup result here is a clinical finding.
- The subgroup variables (age, sex, risk factors, EF, RWMA) are partly the models' own inputs, so performance differences can reflect how informative those inputs are, not bias in the usual sense.
- The threshold 0.5 is not tuned. Error counts shift with any other threshold (`reports/final_ml_evaluation/threshold_tradeoffs.csv`).
- No multiple-comparison correction. The selected models were chosen on these same folds, so overall performance is somewhat optimistic.
- Persistent errors may include label noise (for example row 93); this analysis can't tell model error from label error.

## Files

`data_checks.csv`, `confusion_matrices.csv`, `per_repeat_metrics.csv`, `error_summary.csv`, `subgroup_metrics.csv`, `patient_error_frequency.csv` (303 patients × 4 tasks, with fold IDs), `persistent_error_profiles.csv` and `analysis_log.txt`.
