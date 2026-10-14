# AtheroVision frontend

React + TypeScript + Three.js (React Three Fiber) UI for the Track A cardiovascular risk project:
patient input form, CAD + LAD/LCX/RCA probabilities, an interactive 3D heart whose coronary
vessels are coloured by predicted stenosis risk, and SHAP-style feature contribution views.

This folder is self-contained and never touches the ML code.

## Run

1. Start the model service in the ML project (`cardio_ai_hackathon`), see `api/README.md` there:

   ```powershell
   .venv\Scripts\activate
   python -m uvicorn api.main:app --port 8000
   ```

2. Start the frontend:

   ```bash
   npm install
   npm run dev          # http://localhost:5173, predictions from the trained models
   ```

The header shows **Live model** when connected and **Model offline** (with the command to start
it) when the service can't be reached.

**Demo mode** (no Python needed): `npm run dev:demo` uses `src/api/mockModel.ts`, a hand-weighted
model with placeholder weights. Every screen and printed report is then labelled *DEMO DATA*.

`npm run build` outputs a static site to `dist/` (`npm run build:demo` for a demo build).
`npm test` runs the frontend unit tests.

## API contract (served by `api/` in the ML project)

The model service wraps `src/cad` (`CADPredictor`, `CADExplainer`) without changing it; see
`api/README.md` in the ML project. Types are in `src/types.ts`.

### `POST /predict`

```json
{ "features": { "Age": 61, "Sex": "Male", "DM": 0, "HTN": 1, "Typical Chest Pain": 1, "...": "all 55 form fields" },
  "explain": true }
```

Keys are the dataset column names. Switches are sent as `0/1`, `Sex` as `"Male"`/`"Female"`.
The service re-encodes to the models' schema (48 fields; `"Fmale"`, `"Y"/"N"` where the data
stores text) and never sends `Cath`, `LAD`, `LCX` or `RCA`. Out-of-range values are predicted and
listed in `warnings`; invalid values return **422** with `detail.errors` listing every problem.

`explain: false` returns probabilities only (~0.1 s). The UI asks for those first, then the
explanation (~2 s, mostly the LAD random forest), and the what-if panel never asks for explanations.

Response for the *Typical angina* example (real output, trimmed):

```json
{
  "model_version": "CAD logreg_reduced, LAD rf_reduced_sigmoid, LCX logreg_clinical, RCA logreg_clinical (selected 2026-10-10)",
  "explained": true,
  "predictions": {
    "CAD": {
      "probability": 0.9967,
      "label": 1,
      "threshold": 0.5,
      "predicted_label": "CAD",
      "model_name": "logreg_reduced",
      "model_type": "LogisticRegression",
      "n_input_features": 48,
      "units": "log-odds",
      "additive": true,
      "method": "Exact additive decomposition of the logistic-regression log-...",
      "base_value": 2.5238,
      "reference_probability": 0.9258,
      "reference_note": "...",
      "non_additive_gap": null,
      "features_not_used": [],
      "contributions": [
        {
          "feature": "Typical Chest Pain",
          "value": 1,
          "shap": 1.0717
        },
        {
          "feature": "Tinversion",
          "value": 1,
          "shap": 0.8018
        },
        {
          "...": "one entry per input the model uses"
        }
      ]
    },
    "LAD": {
      "probability": 0.9588,
      "label": 1,
      "threshold": 0.5,
      "predicted_label": "Stenotic",
      "model_name": "rf_reduced_sigmoid",
      "model_type": "RandomForestClassifier",
      "n_input_features": 48,
      "units": "probability",
      "additive": false,
      "method": "Approximate local impact: change in predicted probability wh...",
      "base_value": 0.5807,
      "reference_probability": 0.5807,
      "reference_note": "...",
      "non_additive_gap": 0.2241,
      "features_not_used": [],
      "contributions": [
        {
          "feature": "Typical Chest Pain",
          "value": 1,
          "shap": 0.0336
        },
        {
          "feature": "Age",
          "value": 61,
          "shap": 0.0182
        },
        {
          "...": "one entry per input the model uses"
        }
      ]
    },
    "LCX": "{ ...same shape, log-odds, 7 inputs... }",
    "RCA": "{ ...same shape... }"
  },
  "agreement": {
    "models_disagree": false,
    "note": "The overall CAD model and the vessel models agree at this threshold."
  },
  "warnings": [],
  "unused_inputs": [
    "CHF",
    "Exertional CP",
    "LowTH Ang",
    "Neut",
    "Obesity"
  ],
  "disclaimer": "...",
  "limitations": [
    "..."
  ],
  "threshold_note": "..."
}
```

- `units: "log-odds"`, `additive: true` (logistic regression: CAD, LCX, RCA): `base_value + Σ shap`
  equals the model's log-odds exactly.
- `units: "probability"`, `additive: false` (random forest: LAD): approximate single-input impacts;
  the UI shows them in percentage points and says they don't add up.
- `features_not_used`: inputs a model ignores (LCX and RCA use 7 inputs).

### `GET /metrics`

Mean and SD of accuracy, precision, recall, F1, ROC-AUC (plus PR-AUC, Brier, balanced accuracy,
specificity) per target from repeated stratified 5-fold CV (5 repeats, 303 patients), the selected
models, the exact training range of every numeric input (`feature_ranges`, used for the
"outside the training data" notice) and the project's stated limitations.

### `GET /dataset/{row}`

A real training patient (row 0–302): `features` in the form's encoding, `actual` angiography outcomes
(1 = CAD / stenotic; shown for comparison only, never sent back as inputs) and `cv`, the mean
out-of-fold probability per target from the saved evaluation reports (the fair, held-out estimate).

### `GET /health`

Model names, config date and the library versions the models were trained with.

## Features

- Clinical header: patient banner (case ID, demographics, key vitals with range flags, alert count,
  last-updated time), model status panel, keyboard shortcuts, **New case**, **Export report**
- **Export report**: prints a clean A4 clinical report (use "Save as PDF") with the assessment
  summary, a snapshot of the 3D heart, per-vessel results, top SHAP drivers, alerts and all inputs
- Plain-language assessment summary above the risk numbers
- Keyboard shortcuts: 1–4 select CAD/LAD/LCX/RCA, R reset view, G glow, E export, N new case, ? help
- Patient form (55 dataset features, 3 example patients), live predictions
- **Dataset patient** loader (row number or Random): loads a real patient and compares the live and
  cross-validated predictions with the angiography result
- 3D heart: risk-coloured LAD/LCX/RCA, territory highlighting, heartbeat at the patient's pulse rate,
  bloom glow on the arteries (**Glow** button turns it off on slow laptops)
- Optional real anatomical model: see `public/models/README.md`
- HUD: schematic ECG strip whose shape follows the ECG inputs (ST↑/↓, T inversion, Q waves, LVH,
  poor R progression, BBB), vitals readouts with range flags
- What-if simulator: quit smoking / LDL / HDL / BP / glucose / weight levers, re-runs the model,
  shows now-vs-simulated per target and can paint the scenario on the heart
- Clinical alerts: ECG rhythm label (bradycardia/tachycardia), danger/warning flags for extreme values,
  and an "outside training data" reliability notice

## Robustness

- Number fields only update the prediction with a valid, in-range value; a blank or half-typed
  entry shows the allowed range and reverts when you leave the field.
- The current case and what-if scenario survive a page refresh (stored in this browser only).
- The dashboard loads first; the 3D viewer is a separate download with a loading state, so the
  page never flashes a placeholder heart. If `heart.glb` is missing, the procedural heart is used.
- If the frame rate drops, glow is switched off for that visit and a notice explains why.
  A manual Glow toggle is remembered; the automatic one is not.
- Vessel labels are pushed apart when they would overlap.
- The 3D view has a text description for screen readers, and vessels can be chosen with keys 1–4.

## Design

Clinical-workstation look: colour carries meaning only (risk scale on the arteries and numbers,
green for the live ECG signal, amber/red for alerts); everything else is neutral so the heart is
the focal point. Type is IBM Plex Sans with its condensed cut for monitor-style readouts,
bundled locally (`@fontsource`) so the app looks the same offline.

## Code map

| Path | What it does |
| --- | --- |
| `src/types.ts` | The prediction contract above, as TypeScript types |
| `src/features.ts` | Form schema (55 features, groups, units, reference ranges) and example patients |
| `src/api/client.ts` | `predict()` / `fetchMetrics()`; switches between mock and API |
| `src/api/mockModel.ts` | Placeholder model; exact linear SHAP so the charts behave like the real thing |
| `src/three/anatomy.ts` | Procedural myocardium, coronary paths (LAD/LCX/RCA + branches), vessel territories |
| `src/three/HeartScene.tsx` | Canvas, risk-coloured clickable vessels, territory tinting, projected labels |
| `src/three/CoronaryHeart.tsx` | Renders the BodyParts3D heart: risk-coloured LAD/LCX/RCA meshes, territories |
| `src/three/AnatomicalHeart.tsx` | Fallback for a generic `.glb` heart with no coronary nodes (auto-fit + Align) |
| `tools/build_heart_model.py` | Rebuilds `public/models/heart.glb` from BodyParts3D meshes |
| `src/components/*` | Form, risk summary, SHAP chart, measurements, metrics, ECG, vitals, what-if, alerts, align panel |
| `src/risk.ts` | Single green→red colour scale shared by the 3D view and charts |

## 3D model notes

The default heart is **real anatomy from BodyParts3D** (`public/models/heart.glb`):
ventricular wall, both atria, aorta, pulmonary trunk, venae cavae, and the actual coronary
artery meshes grouped into the three model targets:

| Node in heart.glb | BodyParts3D segments |
| --- | --- |
| `LAD` | trunk of the anterior interventricular branch + diagonal, septal, conus and right anterior branches |
| `LCX` | circumflex branch of the left coronary artery |
| `RCA` | trunk of the right coronary artery + marginal, conus, anterior/posterior ventricular, posterior descending and its septal branches |
| `LeftMain` | trunk of the left coronary artery (coloured by the higher of LAD/LCX) |
| `Myocardium_LAD/LCX/RCA` | ventricular wall split by **nearest coronary artery** — the territory each vessel supplies |

Each vessel mesh is coloured from exactly one model output, so the 3D view always matches the
dashboard. Selecting a vessel (or clicking its territory) highlights that vessel and the muscle
it supplies. Predictions are vessel-level: colour does not indicate where along a vessel a
lesion is. The territory split is a geometric approximation, not a perfusion map.

**Rebuilding the model** (reproducible): `tools/build_heart_model.py` takes a folder of
BodyParts3D `.obj` files (from https://github.com/olivercase/body_parts_3d_api, `meshes/`),
groups the segments, converts BodyParts3D coordinates (+x left, −y anterior, +z superior, mm)
to the app frame, splits the territories and writes the GLB. See the script header for usage.

**Licence / attribution (required):** BodyParts3D © The Database Center for Life Science
(DBCLS), licensed CC BY-SA 2.1 Japan. `heart.glb` is a derivative and is shared under the same
licence. The viewer shows this credit; keep it in your documentation and demo video.

**Fallbacks:** the *Anatomical / Procedural* toggle switches to a procedural heart (generated
in code, with its own vessel paths). If `heart.glb` is replaced by a generic heart model with no
`LAD`/`LCX`/`RCA` nodes, the app overlays the procedural vessels on it and shows an **Align**
panel to line them up.

## Clinical safety

- A sticky banner (and every printed report) shows, verbatim: *"Educational decision-support
  prototype only. Predictions are model estimates, not confirmed diagnoses. This application is
  not a substitute for professional clinical assessment or diagnostic imaging."* (`src/disclaimer.ts`)
- Explanations state that contributions are statistical associations learned from 303 patients,
  not causes, and which units/method each model's explanation uses.
- Vessel colours are whole-vessel model scores; the app never shows or implies a lesion location.
- Inputs outside the training range get a reliability notice; the CAD and vessel models are
  independent and any disagreement between them is shown, not hidden.
- Demo mode is labelled *DEMO DATA* on every screen and report.
- No secrets reach the browser: the frontend only knows the model service URL.
