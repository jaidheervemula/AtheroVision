# Inference layer (`src/cad/inference.py`)

> **Educational decision-support prototype built for a hackathon. Not a diagnostic tool.**
> The models were developed on 303 patients from one centre, with no external validation
> and no clinical validation. Outputs are model scores, not validated risk estimates.

## What it does

- Reads `config/model_selection.json` (written by `scripts/build_model_selection.py`).
- Loads the four selected pipelines from the paths in that config and refuses to run if a
  file's SHA-256, classes (`[0, 1]`) or input feature list do not match the config.
- Validates input **before** prediction. The saved pipelines would otherwise silently
  median-fill missing numbers and zero-encode unknown categories.
- Calls each pipeline's own `predict_proba()`; preprocessing is never rebuilt here.
  Column 1 is the positive class: `CAD` for Cath, `Stenotic` for LAD, LCX and RCA.

| Task | Model | Inputs used |
|---|---|---|
| Cath (overall CAD) | `logreg_reduced` | 48 |
| LAD | `rf_reduced` | 48 |
| LCX | `logreg_clinical` | 7 |
| RCA | `logreg_clinical` | 7 |

## Run the checks (Windows, project folder, `.venv` active)

```powershell
python -m unittest discover -s tests -v
python scripts\check_inference.py
python scripts\check_explain.py
```

No extra packages are needed (the tests use Python's built-in `unittest`; SHAP is not
installed and not used). The unittest command runs both `tests\test_inference.py` and
`tests\test_explain.py`.

## Usage

```python
import sys; sys.path.insert(0, "src")
from cad.inference import CADPredictor, InputValidationError

predictor = CADPredictor()
try:
    result = predictor.predict(patient)          # patient: dict with all 48 fields
except InputValidationError as e:
    print(e.errors)                              # list of every problem found
```

## Input schema

One dict with **all 48 fields** in `config/model_selection.json -> input_schema`
(the union of the four models' inputs; field names are case-sensitive).

- **Numeric fields** (e.g. `Age`, `BMI`, `EF-TTE`, `Region RWMA`): real numbers, whole
  numbers for integer fields, inside the training range. Out-of-range values are rejected
  unless `allow_out_of_range=True`, which accepts them with a warning.
- **0/1 fields** (e.g. `DM`, `HTN`, `Typical Chest Pain`, `Q Wave`): the number `0` or `1`.
  Text such as `"1"` and `True`/`False` are rejected.
- **Text categories** (exact, case-sensitive): `Sex` = `Fmale`/`Male` (spelling as recorded
  in the dataset), Y/N fields = `N`/`Y`, `BBB` = `LBBB`/`N`/`RBBB`,
  `VHD` = `Moderate`/`N`/`Severe`/`mild`.
- `Cath`, `LAD`, `LCX`, `RCA` are prediction targets and are rejected as inputs.
- Missing values (`None`, `NaN`) and unknown fields are rejected; all problems are
  reported together, with "did you mean" hints for near misses.

## Output format

```text
{
  "disclaimer": "...",
  "threshold": 0.5,
  "threshold_note": "fixed threshold, not tuned or clinically optimised",
  "predictions": {
    "Cath": {"display_name": "Overall CAD", "probability": 0.0-1.0,
             "predicted_positive": true/false, "predicted_label": "CAD"/"Normal",
             "positive_label": "CAD", "negative_label": "Normal",
             "model_name": "...", "model_type": "...", "n_input_features": 48,
             "input_features_used": [...], "cv_roc_auc": {"mean": ..., "sd": ...}},
    "LAD": {...}, "LCX": {...}, "RCA": {...}
  },
  "agreement": {"cad_predicted_positive": ..., "vessels_predicted_positive": [...],
                "models_disagree": true/false, "note": "..."},
  "warnings": [...],
  "limitations": [...]
}
```

The four models are independent. If overall CAD and the vessel models disagree, the
`agreement` section says so; outputs are never adjusted to agree.

## Limitations

- 303 patients, one centre, no external or clinical validation.
- The default 0.5 threshold is not tuned; treat the probability as the main output.
- Patients from the training dataset get optimistic (in-sample) scores from these
  refitted models; use the stored out-of-fold probabilities when demonstrating them.

## Explanations (`src/cad/explain.py`)

```python
from cad.explain import CADExplainer
explainer = CADExplainer(predictor)              # reuses the loaded models
report = explainer.explain(patient, top_k=10)    # same validation as predict()
```

| Task | Model | Method | Units | Additive? |
|---|---|---|---|---|
| Cath, LCX, RCA | logistic regression | exact decomposition: `coef x (encoded value - training-average encoded value)`, summed back to each input | log-odds | Yes: reference log-odds + contributions = model log-odds |
| LAD | random forest | single-feature background replacement: replace one input at a time with each training patient's value, average the model output | probability (0-1) | No: approximate; the gap is reported as `non_additive_gap` |

Each task entry contains `probability` (identical to `predict()`), `method`, `units`,
`additive`, a reference value, `contributions` ranked by absolute size (feature,
patient value, contribution, "raises/lowers model output"), `features_not_used_by_model`
and `limitations`.

Read log-odds contributions as log-odds, not percentage points. Every contribution
describes how the **model's output** responds to an input; it is a statistical
association from 303 patients, not a cause, a medical reason or a diagnosis.
The reference values are computed from the training file (read-only, hash-checked).
Explaining all four models takes about 2 seconds, mostly for LAD.
