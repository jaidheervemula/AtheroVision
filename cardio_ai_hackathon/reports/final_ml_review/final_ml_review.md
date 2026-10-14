# Final ML review: frozen baseline for integration

**Date:** 2026-10-10 · **Type:** read-only audit before the ML freeze · **Project:** Multimodal AI Hackathon 2026, Track A

> **Educational decision-support prototype. Not a diagnostic tool.** All performance figures
> are internal cross-validation estimates on 303 patients from one centre. There is no
> independent test set and no external or clinical validation. Nothing here claims that
> this project performs better than any other.

## 1. Status

**The four models below are frozen as the ML baseline for integration.**

This review changed nothing:
- no model was trained;
- no threshold was changed;
- no calibrated or replacement artifact was created;
- no existing file was modified.

The 34 files it relied on were copied from the laptop and confirmed SHA-256-identical before any check. A hash snapshot of all 103 project files was taken before the review and compared afterwards (section 8).

`config/model_selection.json` still says `"status": "provisional"`. That's expected: this read-only review can't edit it, so the freeze is recorded here and in `review_config.json`.

| Result | Count | Meaning |
|---|---|---|
| PASS | 22 | Verified against files or recomputed |
| WARNING | 14 | Verified, but a limitation or caveat the integration must respect |
| NOT VERIFIED | 1 | Couldn't be checked by the reviewer (section 7) |
| FAIL | 0 | — |

Every check, with its evidence, is in `review_log.txt`; per-artifact detail is in `artifact_audit.csv`.

## 2. Frozen artifacts

| Task | Model | Inputs | Artifact (SHA-256 prefix) | Positive / negative label | ROC-AUC, repeat-averaged OOF (95% CI) | ROC-AUC, 25-fold mean |
|---|---|---|---|---|---|---|
| Cath (overall CAD) | `logreg_reduced` (LogisticRegression) | 48 | `models/phase1_candidates/Cath__logreg_reduced.joblib` (`b27f7886cff8`) | `CAD` / `Normal` | 0.920 (0.889–0.949) | 0.922 |
| LAD | `rf_reduced` (RandomForestClassifier) | 48 | `models/phase1_candidates/LAD__rf_reduced.joblib` (`04ee2c01869a`) | `Stenotic` / `Normal` | 0.847 (0.801–0.887) | 0.853 |
| LCX | `logreg_clinical` (LogisticRegression) | 7 | `models/phase1_candidates/LCX__logreg_clinical.joblib` (`44d4a2b34f97`) | `Stenotic` / `Normal` | 0.739 (0.681–0.796) | 0.743 |
| RCA | `logreg_clinical` (LogisticRegression) | 7 | `models/phase1_candidates/RCA__logreg_clinical.joblib` (`dfa6626bc904`) | `Stenotic` / `Normal` | 0.737 (0.680–0.790) | 0.740 |

**Verified for each artifact (PASS):**
- the file exists, and its SHA-256 equals both the config and the laptop copy;
- it loads without warnings as a `Pipeline(preprocess → model)`;
- its estimator class matches the config;
- `classes_ = [0, 1]`, and `predict_proba[:, 1]` is the positive class;
- its input features equal both the config and `metadata.json`;
- **no input is `Cath`, `LAD`, `LCX` or `RCA`**;
- the raw labels in the dataset are exactly the configured positive and negative labels.

The inference schema has 48 fields, exactly the union of the four models' inputs, with no target columns.

**Which ROC-AUC to quote:**
- The quoted figures (0.920 / 0.847 / 0.739 / 0.737) were recomputed here from the out-of-fold predictions. They're the **repeat-averaged** estimand: computed on each repeat's 303 predictions, then averaged over the 5 repeats.
- `config/model_selection.json` stores the **25-fold mean** instead (0.922 / 0.853 / 0.743 / 0.740).
- Both are correct; documentation should name the estimand it quotes (WARNING).

## 3. Evidence

**Evaluation protocol (PASS):**
- RepeatedStratifiedKFold, 5 folds × 5 repeats, `random_state=42`, stratified separately on each target.
- All preprocessing is fitted inside the training folds.
- The threshold is a fixed, untuned 0.5. These settings match in `scripts/train_phase1.py`, `metadata.json` and the config.
- The final evaluation regenerated the per-repeat out-of-fold predictions and reproduced the saved out-of-fold means and fold AUCs for all four tasks, on the laptop, with existing files unchanged.

**Experiment 1, LAD sigmoid calibration (PASS: recorded as an alternative, not deployed):**
- Brier score improved from 0.164 to 0.154, log loss from 0.504 to 0.476, and calibration slope from 1.82 to 1.00.
- ROC-AUC changed by −0.002 at repeat level; per-fold ROC-AUC was identical.
- The baseline was reproduced exactly.
- The config still selects `rf_reduced`, the LAD artifact hash is unchanged, and no calibrated artifact exists in `models/`.
- One `unknown category` warning (CVA, in an inner split) was documented; it's not leakage (WARNING).

**Experiment 2, LCX/RCA alternatives (PASS: none adopted):**
- None of the 8 candidate comparisons had an ROC-AUC difference interval entirely above 0.
- Class weighting and stronger regularisation (C = 0.1) on the clinical inputs changed ROC-AUC by +0.000 to +0.001.
- The 48-input models and the conservative random forest had lower point estimates (−0.006 to −0.030), with intervals crossing 0.
- The baselines were reproduced, and the config is unchanged.

**Experiment 3, out-of-fold error analysis (PASS):**
- 33 of 33 alignment checks passed on the laptop, with existing files unchanged.
- At threshold 0.5 (repeat-averaged), sensitivity and specificity were:

  | Task | Sensitivity | Specificity |
  |---|---|---|
  | Cath | 0.901 | 0.745 |
  | LAD | 0.886 | 0.643 |
  | LCX | 0.523 | 0.797 |
  | RCA | 0.432 | 0.812 |

- Patients missed in all 5 repeats: 42 for LCX and 60 for RCA. LCX and RCA sensitivity at 0.5 is low (WARNING).

**Row 93 (WARNING, preserved):**
- Recorded as Cath = Normal and LAD = Stenotic, which contradicts the dataset's own CAD definition. It is kept unchanged everywhere.
- Excluding it in the sensitivity analysis changed the selected models' ROC-AUC by at most 0.008.

**Tests (PASS in the audit replica):**
- `python -m unittest discover -s tests` ran 39 tests: OK.
- `scripts/check_inference.py` and `scripts/check_explain.py` both passed.
- The local Windows `.venv` run is NOT VERIFIED (section 7).

## 4. Integration contract

What the backend and dashboard must follow:

1. **Loading**
   - Always load models through `cad.inference.CADPredictor()`, which reads `config/model_selection.json`, checks every artifact's SHA-256, classes and feature list, and refuses to start on a mismatch.
   - Never load `.joblib` files by name, and never use the other 16 Phase 1 candidates.
2. **Input**
   - One dict with all 48 schema fields, using exact, case-sensitive names and values.
   - 0/1 fields must be numbers, and categories must match exactly. For example, `Sex` is `Fmale`/`Male` as recorded, Y/N fields are `N`/`Y`, `VHD` is `N`/`mild`/`Moderate`/`Severe`.
   - Numbers must lie within the training range unless `allow_out_of_range=True`, which turns the rejection into a warning.
   - Missing values, unknown fields, unknown categories and target columns are rejected with `InputValidationError`, which lists every problem. The UI must show these errors, not substitute defaults.
3. **Output**
   - For each task: `probability` (positive class), `predicted_label` at threshold 0.5, `positive_label`, the model name and type, the inputs used, and `cv_roc_auc` (25-fold mean).
   - Also returned: an `agreement` block, `warnings`, `limitations` and `disclaimer`.
   - The probability is the primary output. The 0.5 label is a fixed, untuned convenience.
4. **Explanations** (`cad.explain.CADExplainer`)
   - **Cath, LCX and RCA** use an exact additive decomposition in **log-odds** (not percentage points).
   - **LAD** uses an **approximate, non-additive** single-feature replacement in probability units; it reports `non_additive_gap` and must be labelled approximate.
   - All contributions are associations with the model's output, not causes or diagnoses.
5. **Required UI statements**
   - Show the disclaimer at all times.
   - LAD probabilities are under-dispersed (calibration slope 1.82), so they are scores, not risk estimates.
   - LCX and RCA miss many stenotic vessels at 0.5, so a "Normal" result doesn't rule out stenosis.
   - CAD and vessel models can disagree, and disagreements are shown, not corrected.
   - The vessel colour on the 3D model is a whole-vessel model score. The dataset contains no lesion location, so the UI must never show one.
6. **Training-set patients.** For any demo patient from the dataset, show the stored out-of-fold probability (`reports/final_ml_evaluation/oof_per_repeat_selected.csv`). The deployed models have seen these patients, so their in-sample scores are optimistic.
7. **Change control**
   - Any change to the config, an artifact, the threshold or the input schema reopens this review.
   - Adopting LAD calibration would need a new artifact, an updated config and a new review.

## 5. Limitations (all apply to every figure above)

- **Sample.** 303 patients from one centre and one time period; no patient IDs, so near-duplicate patients can't be ruled out.
- **No independent test set.** All patients were used for cross-validation, model selection and the final refit.
- **No external or clinical validation.** Generalisation to other hospitals, populations or equipment is unknown.
- **Model-selection optimism.**
  - Six Phase 1 candidates per task were compared on the same folds.
  - The selection-rule thresholds were set after the results were seen.
  - RCA was revised after review.
  - 8 further LCX/RCA comparisons and 1 LAD calibration comparison were made on the same folds.
  - Bootstrap intervals cover patient sampling only, not retraining or selection, so treat them as lower bounds on the uncertainty. Nested cross-validation was not performed.
- **Untuned 0.5 threshold.** Sensitivity, specificity, precision and F1 all depend on it.
- **Rare categories.** 14 categorical levels in the 48-input schema occur in fewer than 15 patients (for example CVA = Y in 5, Weak Peripheral Pulse = Y in 5, CRF = Y in 6). CHF and LowTH Ang were excluded as too rare.
- **Row 93 label inconsistency.** Kept as recorded; this analysis can't separate model error from label error.
- **LAD explanations.** Approximate and non-additive; they ignore correlations between features.
- **Calibration.** LAD is under-dispersed. Cath and LCX lean towards over-confidence, but their intervals include ideal calibration.
- **Independent models.** In the out-of-fold predictions, 10 patients had a vessel flagged while CAD was predicted negative, and 18 had CAD predicted with no vessel flagged.
- **Undocumented dataset fields.** Function Class and Region RWMA have undocumented definitions, and the `Fmale` spelling is unconfirmed.

## 6. Alternatives on record (not deployed)

| Alternative | Where | Status |
|---|---|---|
| LAD `rf_reduced` + sigmoid calibration | `reports/lad_calibration/` | Better probability quality, same discrimination. **Not adopted.** Candidate for a future, separately reviewed change. |
| LCX/RCA: class weighting, C = 0.1 (7 or 48 inputs), conservative random forest | `reports/vessel_model_experiment/` | No clear ROC-AUC gain. **Not adopted.** |

## 7. NOT VERIFIED

**Running the tests in the local Windows `.venv`.**
- I can't execute your Windows Python environment, and no saved local test log exists.
- The tests passed in an audit replica with identical project files and the same scikit-learn 1.9.1, but slightly different Python and pandas patch versions (replica 3.13.16 / 3.0.5; laptop 3.13.5 / 3.0.6).
- To verify, run these from the project root with `.venv` active:

```powershell
python -m unittest discover -s tests -v
python scripts\check_inference.py
python scripts\check_explain.py
```

**Expected:** `Ran 39 tests ... OK`, `SMOKE TEST PASSED`, `EXPLANATION CHECK PASSED`. Until then, this item stays NOT VERIFIED.

## 8. Files

- `final_ml_review.md`: this document.
- `artifact_audit.csv`: one row per frozen artifact, with path, hash, schema and every check.
- `review_config.json`: protocol, frozen selection, ROC-AUC estimands, check counts and the SHA-256 of key files.
- `review_log.txt`: all 37 checks, with status and evidence.

**Read-only confirmation:** all 103 pre-existing project files had identical SHA-256 hashes before and after this review. The only additions are the four files above, in `reports/final_ml_review/`.
