# AtheroVision: presentation pack

Everything here can be said on stage. Every number comes from the project's saved evaluation
reports or the running app (see MODEL_CARD.md for sources).

## Before you present (checklist)

- [ ] Start AtheroVision (`start.bat`) **10 minutes before**, open http://localhost:5173 and check that the header shows **Live model**.
- [ ] Press **N** (new case), then **P** to check that the guided demo starts. Press **Esc** to close it.
- [ ] Laptop on charge. Close other apps. Browser zoom at 100%. Turn notifications off.
- [ ] Have the **demo video** (the required deliverable below) downloaded to the desktop as a backup, in case the laptop or projector fails.
- [ ] If the 3D view is slow on the projector laptop, click **Glow** to turn it off.

## Demonstration video (required deliverable: 3–10 minutes, on YouTube)

The brief asks the video to show the working system, the feature-input workflow, the 3D interactions **and the technical implementation**. Target about **6 minutes**.

**Recording:**
- Use OBS Studio (free) or the Windows Game Bar (**Win + G**, then the record button).
- Record at 1080p with the browser full-screen (F11) and the zoom at 100%.
- Use a headset microphone if you have one.
- Upload to YouTube as **Unlisted** (anyone with the link can watch) and put the link in your submission.

| Time | On screen | What to say / do |
|---|---|---|
| 0:00–0:30 | Title slide, then the app | "AtheroVision predicts coronary artery disease and narrowing of the LAD, LCX and RCA from routine clinical data, and maps it onto a 3D heart. Everything runs locally; the bar at the top says it's decision support, not a diagnosis." |
| 0:30–1:30 | **Input workflow**: click *New case* (N), type a few values (age, chest pain, LDL, ECG findings) | "Every field is from the dataset. Watch the scores and the arteries update as I type. Values outside the training range get a warning instead of a silent guess." Then pick *Typical angina*. |
| 1:30–2:30 | **3D interaction**: rotate, zoom, click the LAD, then the RCA territory | "These are the real LAD, LCX and RCA from BodyParts3D. The colour is the predicted probability. Clicking an artery selects it, and the risk panel, explanation and measurements follow. The heart beats at the patient's pulse." |
| 2:30–3:30 | Press **P** and walk through guided-demo steps 2–4 | SHAP explanation, reality check, screening cut-offs (use the guided demo's captions). |
| 3:30–4:15 | Guided-demo steps 5–7 | What-if, then real patient row 32 (all ✓), then row 93 (label conflict). |
| 4:15–5:30 | **Technical implementation**: show the architecture diagram (documentation page 2), then http://localhost:8000/docs, then the terminal running the tests | "Three layers: React and Three.js in the browser, a FastAPI service, and the scikit-learn package. Models are hash-checked at start-up, and targets are rejected as inputs to prevent leakage. Validation is 5×5 repeated cross-validation with bootstrap intervals and calibration." Run `python -m unittest api.test_api` on screen (about 8 s). |
| 5:30–6:00 | Model performance panel, then export the report (E) | "AUC 0.92 for CAD, 0.84 for LAD, 0.74 for LCX and RCA, calibration shown here. Limits: one centre, 303 patients, no external validation. Thank you." |

**Tips:**
- Rehearse once without recording.
- If a step goes wrong, pause, then cut it out later.
- Keep the mouse still while you talk.

## 3-minute demo (press P, then → for each step)

| # | On screen | What to say (about 20 seconds each) |
|---|---|---|
| 1 | Typical angina patient, heart coloured | "Here's a 61-year-old smoker with chest pain. Four models trained on 303 angiography patients score overall CAD and each coronary artery. The heart shows where the risk is." |
| 2 | Why this prediction | "Every score is explained. These bars show what pushed it up or down. They're associations in the data, not causes, and we say so on screen." |
| 3 | Reality check (LCX) | "A probability alone is hard to trust. So we find the 40 patients in our data with the closest held-out score and count how many really had disease. For this LCX score, 28 of 40 did." |
| 4 | Vessel cut-offs | "At 50%, the LCX and RCA models missed half of the narrowed arteries. For screening that's wrong, so we flag them from 30%. They now catch 84% and 83%, and we show the cost in false alarms." |
| 5 | What-if | "Quit smoking, lower LDL and blood pressure: the models re-run instantly, and the heart shows the scenario." |
| 6 | Dataset row 32 | "This is a real patient with narrowing in all three arteries. 'Cross-val.' is what each model predicted when this patient was held out of training: the fair test. All four are right, and at the old 50% cut-off we'd have missed the LCX and the RCA." |
| 7 | Row 93, label conflict | "And when the data is wrong, we say so. Row 93 contradicts the dataset's own definition of CAD. We flag it instead of hiding a miss." |
| 8 | Model performance | "Our limits are on screen: AUC 0.92 for CAD, 0.84 for LAD, 0.74 for LCX and RCA. 303 patients from one centre, no external validation. This is decision support, not a diagnosis." |

Finish with: **Export report** (press **E**) to show the one-page printable summary.

## Slides (5)

1. **Problem.** Coronary angiography is invasive and costly. Can routine clinical data say *who* has disease and *where*?
2. **What we built.** Clinical form → four independent models (CAD, LAD, LCX, RCA) → explanations → 3D heart → printable report. It runs locally, and no patient data leaves the machine.
3. **How we kept it honest.**
   - 5×5 repeated cross-validation with bootstrap confidence intervals.
   - The targets can never be inputs; the API rejects them.
   - Held-out scores shown next to live ones.
   - A reality check for every score.
   - Data errors flagged.
4. **Results.**

   | | AUC (95% CI) | Catches | Clears |
   |---|---|---|---|
   | CAD | 0.92 (0.89–0.95) | 90% | 74% |
   | LAD | 0.84 (0.80–0.89) | 86% | 67% |
   | LCX | 0.74 (0.68–0.80) | 84% at 30% | 54% |
   | RCA | 0.74 (0.68–0.79) | 83% at 30% | 53% |

5. **Limits and next steps.**
   - Limits: one centre, 303 patients, no external validation; LCX and RCA are only moderately accurate; the strongest inputs include echo results.
   - Next: external validation, more patients, lesion-level data, and a prospective check of the 30% cut-off.

## Likely judge questions

**Why not deep learning?**
There are 303 patients with tabular data. Logistic regression is competitive at this size, every prediction is exactly explainable, and it's far less prone to overfitting.

**How do you know it isn't overfitting?**
- Every metric comes from 5×5 repeated cross-validation, with bootstrap 95% intervals.
- The app shows each dataset patient's held-out prediction next to the live one.
- We picked models by comparing candidates on the same folds, so the numbers are probably slightly optimistic, and we say that in the model card.

**Any data leakage?**
- LAD, LCX, RCA and Cath are never inputs; the API rejects them with an error.
- Preprocessing is fitted inside each training fold.
- Model files are checked against recorded fingerprints at start-up.

**Why 30% for LCX and RCA?**
- It comes from a fixed rule: the highest cut-off that catches at least 80% of diseased patients.
- For screening, a missed narrowing is worse than a false alarm, which leads to a further test.
- The cost is visible in the app: specificity falls from about 80% to about 54%.

**The CAD score says 100%. Is it really?**
- The CAD model is somewhat overconfident at the extremes (calibration slope 0.80).
- That's why the reality check exists: of the 40 patients with the closest scores, how many really had CAD.

**Could a hospital use this?**
- Not yet. It needs external validation first.
- It also relies on echo results, so it fits *after* a cardiology work-up, not at first contact.

**Why is AtheroVision multimodal?**
- **Inputs:** five clinical sources, namely demographics, clinical history and examination, ECG findings, laboratory results and echocardiography, as Track A specifies. The CAD and LAD models use features from all five; the LCX and RCA models use 7 history and demographic inputs.
- **Outputs:** three complementary forms, namely numerical risk probabilities, interpretable explanations, and an interactive 3D anatomical heart.
- **What we don't claim:** the models run on structured clinical features, not raw ECG signals or medical images.
- If a judge challenges the term: *"Our models use structured clinical features from multiple sources, as specified by Track A. We then connect their predictions to explainability and interactive 3D anatomy. We don't claim raw ECG or medical-image inference."*

**What would you do with more time?**
- Validate on another hospital's data.
- Get lesion-level labels, so the heart can show *where* in the artery the narrowing is.
- Re-check calibration and the cut-offs on new patients.

**Where's the code?**
GitHub, with automatic tests on every push (`.github/workflows/tests.yml`). 94 tests cover the models, the API and the interface.
