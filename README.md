# AtheroVision: coronary artery disease risk prototype
**🔗 Live demo:** https://athero-vision.vercel.app
*The model server runs on a free tier: if it has been idle, the first prediction can take up to a minute to wake up. After that it's instant.*

**🎬 Demo video:** https://www.youtube.com/watch?v=lBzC9CKA5Do

> **Educational decision-support prototype only.** Predictions are model estimates, not confirmed
> diagnoses. This application is not a substitute for professional clinical assessment or diagnostic imaging.

AtheroVision predicts the probability of coronary artery disease (CAD) and of disease in each main
coronary artery (LAD, LCX, RCA) from clinical inputs. It shows the results on an interactive 3D heart,
with per-feature contributions. Contributions are statistical associations learned by the models, not causes.
See **[MODEL_CARD.md](MODEL_CARD.md)** for how accurate the models are and what their limits are.

```
AtheroVision-Project/
├── start.bat / start.ps1      one-click setup + start (Windows)
├── README.md                  this file
├── MODEL_CARD.md              model performance, explanations, limitations
├── PRESENTATION.md            demo + video script, slide text, likely judge questions
├── docs/                      AtheroVision-Documentation.pdf (the 6-page project documentation)
├── .github/workflows/         automatic tests on GitHub
├── requirements-backend.txt   exact Python package versions for the backend
├── cardio_ai_hackathon/       ML project + FastAPI model service (Python)
│   ├── api/                   FastAPI service (main.py, adapter.py, dataset.py, test_api.py)
│   ├── src/cad/               inference.py (CADPredictor), explain.py (CADExplainer)
│   ├── models/                trained models (the 4 deployed ones are listed below)
│   ├── config/                model_selection.json (model choice, input schema, hashes)
│   ├── data/raw/              Z-Alizadeh Sani dataset (needed by the explainer and the dataset loader)
│   ├── reports/  scripts/  tests/
└── frontend/                  React + TypeScript + Vite web app with the 3D heart
```

## Quick start (Windows)

1. Install **Python 3.12 or 3.13** (https://www.python.org/downloads/, tick "Add python.exe to PATH") and **Node.js LTS** (https://nodejs.org/).
2. Unzip the project (right-click → Extract All).
3. Double-click **`start.bat`**.

The first run installs everything, which needs internet and takes a few minutes. After that it starts in a few seconds.
It opens two windows, "AtheroVision model service" and "AtheroVision web app", and then the browser at **http://localhost:5173**.
To stop AtheroVision, close those two windows.

If `start.bat` doesn't work on a computer, use the manual steps below. They do the same thing.

## Using the app

- **Guided demo:** click **Guided demo** in the top bar, or press **P**. Eight steps walk through every feature with captions; use **→** for next, **←** for back and **Esc** to close. The script for presenting is in [PRESENTATION.md](PRESENTATION.md).
- **Calibration:** the **Model performance** card shows predicted against observed risk for each target (hover a point; "Show as table" for the numbers).
- **Reality check:** under the vessel scores, for the selected target. It shows how many of the 40 dataset patients with the closest held-out score actually had the condition on angiography, with a 95% interval.

- **Example patients:** *Low risk*, *Typical angina* and *Multi-vessel* are made-up example cases.
- **Dataset patient:** type a row number (0–302) or click **Random** to load a real patient from the training data.
  The risk panel then shows that patient's **angiography result** next to two scores:
  - **Live:** what the deployed model predicts now. The deployed models were trained on this patient, so the live score is optimistic.
  - **Cross-val.:** what the model predicted when this patient was held out of training. This is the fair estimate. ✓/✗ shows whether it matched angiography at the 50% threshold.
  Editing any field clears the comparison, because an edited patient no longer has a known outcome.
  If a patient's labels contradict each other (row 93: no CAD recorded, but the LAD recorded as stenotic), a **⚠ Label conflict** note explains it.
- **Flagging cut-offs:** CAD and LAD are flagged from 50%. LCX and RCA are flagged from **30%**, because at 50% they missed about half of the narrowed arteries. The tick on each bar marks its cut-off. The **Model performance** card shows what each cut-off catches and clears. Details are in MODEL_CARD.md.
- **What-if simulator:** change one risk factor and compare the result against the current patient.
- **Export report:** prints a one-page report (use "Save as PDF" in the print dialog).

## Manual setup (once)

If PowerShell says *"running scripts is disabled on this system"*, run this first in that window:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**Backend:** open PowerShell in `cardio_ai_hackathon`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r ..\requirements-backend.txt
```

The exact package versions matter. The models are scikit-learn 1.9.1 pipelines, and the loader checks every
model file against the SHA-256 hashes in `config/model_selection.json`.

**Frontend:** open PowerShell in `frontend`:

```powershell
Copy-Item .env.example .env
npm install
```

`.env` sets `VITE_API_URL=http://localhost:8000`, the address of the backend. The app and the frontend tests need it.

## Manual start (two PowerShell windows)

**Window 1: backend.** In `cardio_ai_hackathon`:

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8000
```

Wait for `Application startup complete`. API documentation: http://localhost:8000/docs.

**Window 2: frontend.** In `frontend`, run `npm run dev`. Then open **http://localhost:5173**.

**Demo mode (no backend):** `npm run dev:demo` uses a built-in demo model, and every result is labelled DEMO DATA.

## Tests

```powershell
# Backend, in cardio_ai_hackathon with .venv activated
python -m unittest discover -s tests -v      # ML package: inference + explanations (39 tests)
python -m unittest api.test_api -v           # model service, dataset loader, cut-offs, reality check, calibration (27 tests)

# Frontend, in frontend (after creating .env)
npm test                                     # 28 tests
npm run build                                # type-check + production build
```

## Working on it together (GitHub)

Once the project is on GitHub, `.github/workflows/tests.yml` runs all the backend tests (on Windows) and the frontend tests and build on every push. The results appear in the repository's **Actions** tab.

Use one shared repository instead of sending ZIPs, so nobody's changes get lost. The `.gitignore` at the top level
already leaves out `.venv`, `node_modules`, `dist` and `.env`. The models and the dataset are small enough to include (about 25 MB).

Easiest way, with **GitHub Desktop** (https://desktop.github.com):

1. File → Add local repository → choose the `AtheroVision-Project` folder → **create a repository** there.
2. Make the first commit, then **Publish repository**. Keep "private" ticked if you don't want it public.
3. On github.com, open the repository → Settings → Collaborators → add your teammate.
4. Your teammate uses File → Clone repository. After that, each of you **Fetch/Pull** before working and **Commit + Push** after.

Or from PowerShell, if Git is installed, in the `AtheroVision-Project` folder, after creating an empty repository on github.com:

```powershell
git init
git add .
git commit -m "AtheroVision"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Deployed models

| Target | Model | File |
|---|---|---|
| CAD (Cath) | logistic regression (reduced) | `models/phase1_candidates/Cath__logreg_reduced.joblib` |
| LAD | random forest, sigmoid-calibrated | `models/deployment/LAD__rf_reduced_sigmoid.joblib` |
| LCX | logistic regression (clinical) | `models/phase1_candidates/LCX__logreg_clinical.joblib` |
| RCA | logistic regression (clinical) | `models/phase1_candidates/RCA__logreg_clinical.joblib` |

The other files in `models/phase1_candidates/` are the candidates that were compared during model selection.
`config/model_selection.json` records which models were chosen and why.
Cath, LAD, LCX and RCA are prediction targets only. The API rejects them as inputs with HTTP 422 to prevent leakage.
The dataset loader returns them separately, for comparison only.

## Not included in the ZIP

| Left out | How to get it back |
|---|---|
| `cardio_ai_hackathon/.venv/` (Python environment) | `start.bat`, or the manual backend setup |
| `frontend/node_modules/` (downloaded packages) | `start.bat`, or `npm install` |
| `frontend/dist/` (build output), `__pycache__/` | Recreated automatically |
| `frontend/.env` (no secrets, only the backend URL) | `start.bat`, or `Copy-Item .env.example .env` |

No API keys, passwords or tokens are used anywhere. The frontend only knows the backend URL.
In six log files under `cardio_ai_hackathon/reports/`, the Windows user folder in file paths was replaced with `C:\Users\<user>`.

## Credits and licences

- **Dataset:** Z-Alizadeh Sani dataset (extension), UCI Machine Learning Repository. 303 patients.
- **3D heart model** (`frontend/public/models/heart.glb`): derived from BodyParts3D, © The Database Center for
  Life Science (DBCLS), licensed CC BY-SA 2.1 Japan. See `frontend/public/models/README.md`.
