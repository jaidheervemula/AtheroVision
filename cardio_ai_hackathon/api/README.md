# AtheroVision model service (`api/`)

> Educational decision-support prototype. Not a diagnostic tool.

A thin FastAPI layer that serves the four selected models from `src/cad` to the AtheroVision
frontend. It imports `cad.inference.CADPredictor` and `cad.explain.CADExplainer` and
**adds no modelling logic**: the same files, hash checks, validation and explanation
methods are used. Nothing in `src/`, `models/`, `config/` or `scripts/` is changed.

## Run (Windows, project folder)

```powershell
.venv\Scripts\activate
pip install -r api\requirements.txt          # once
python -m uvicorn api.main:app --port 8000
```

Then start the frontend (`npm run dev` in the frontend folder).
Open http://localhost:8000/docs for interactive API documentation.

## Tests

```powershell
python -m unittest api.test_api -v
```

They load the real models and use the frontend's example patients
(`api/frontend_examples.json`), and check that API probabilities equal
`CADPredictor.predict()` exactly and that log-odds explanations reconstruct each probability.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, selected models, training library versions |
| GET | `/metrics` | CV metrics per target (mean and SD), validation scheme, n, exact training ranges, limitations |
| POST | `/predict` | `{"features": {...}, "explain": true}` → probabilities, labels, explanations, warnings |
| GET | `/dataset/{row}` | a real training patient (row 0–302): inputs in the frontend's format, the angiography outcome (comparison only, never an input) and the cross-validated probability from the saved evaluation reports (`api/dataset.py`) |

## What the adapter does (`api/adapter.py`)

| Frontend sends | Model receives |
|---|---|
| 55 form fields | the 48 fields in `config/model_selection.json → input_schema`; the rest are listed back as `unused_inputs` (Weight/Height are used only to compute BMI) |
| `Sex: "Female"` | `"Fmale"` (dataset spelling) |
| switches as `0/1` | `0/1` or `"N"/"Y"`, whichever the schema stores |
| `Cath`, `LAD`, `LCX`, `RCA` | rejected with **422** (prediction targets; leakage prevention) |

- Out-of-range values are predicted with `allow_out_of_range=True` and returned as
  `warnings`, so the interface can show the score with a reliability notice.
- Invalid values (e.g. a non-whole number for an integer field) return **422** with every
  problem listed.
- `explain: false` skips explanations (~0.1 s instead of ~2 s; the LAD random forest is the
  slow part). The frontend uses it for instant updates and for the what-if simulator.
- Recent results are cached (64 entries), so returning to an earlier input is instant.

## Flagging cut-offs (`api/operating_points.py`)

Probabilities are never changed. Each target's `label` is set at its own cut-off: the highest one
(0.05 steps, at most 0.5) that catches at least 80% of positive patients in the saved
cross-validation predictions. Result: CAD 0.5, LAD 0.5, LCX 0.3, RCA 0.3. Every prediction also
carries `default_threshold`/`default_label` (the package's 0.5 decision) and an `operating_point`
with the reason. `/metrics` returns `operating_points` with the cross-validated sensitivity and
specificity at both cut-offs.

## Reality check (`api/reality_check.py`)

Each prediction carries `reality_check`: among the 40 dataset patients whose cross-validated score
is closest, how many truly had the condition (`positives`/`n`, `rate`, Wilson 95% `ci_low`–`ci_high`,
and the score range `score_min`–`score_max`). It is computed from saved predictions only. Send `"dataset_row": <row>` with an unedited dataset patient to leave that patient out of their own comparison (`excluded_row` echoes it).

## Explanation units (as produced by `cad.explain`)

| Target | Model | Units | Additive |
|---|---|---|---|
| CAD (Cath), LCX, RCA | logistic regression | log-odds | yes: `base_value + Σ contributions = model log-odds` |
| LAD | random forest | probability (0–1) | no: approximate single-feature replacement |
