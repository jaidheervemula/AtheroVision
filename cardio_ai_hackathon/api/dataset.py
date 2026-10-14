"""
Real patients from the training dataset, for demonstrations ("load dataset patient").

Read-only: the dataset and the saved cross-validation reports are read, never changed, and
no model is fitted here.

For each row this returns three separate things:
  * features  - the patient's inputs in the frontend's encoding (what the form sends);
  * actual    - the angiography outcomes (Cath, LAD, LCX, RCA). These are prediction
                targets and are returned ONLY for comparison; they are never model inputs
                (the /predict endpoint rejects them with 422);
  * cv        - the cross-validated probability for this patient: the mean of the
                out-of-fold predictions saved during evaluation, i.e. what each model predicted
                when this patient was held out of training (5 repeats of 5-fold CV).

Why `cv` matters: the deployed models were fitted on all 303 patients, so a live
prediction for a dataset patient is in-sample and optimistic. The cross-validated value is
the honest estimate of how the model does on a patient it has not seen.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

from .adapter import FRONTEND_TARGET

TARGET_POSITIVE = {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}

# Saved out-of-fold predictions of the selected models (reports from the evaluation runs).
# Cath, LCX, RCA: the selected logistic regressions. LAD: the deployed sigmoid-calibrated forest.
OOF_SELECTED = Path("reports/final_ml_evaluation/oof_per_repeat_selected.csv")
OOF_LAD_SIGMOID = Path("reports/lad_calibration/lad_calibration_predictions.csv")

LABEL_CONFLICT_NOTE = (
    "Label conflict in the dataset: this patient is recorded as having no CAD, yet at least one artery "
    "is recorded as stenotic. In this dataset CAD means stenosis in any artery, so the CAD label is "
    "probably a recording error. It is kept as recorded (the project audit found that removing it changes "
    "cross-validated AUC by at most 0.018), so a CAD 'miss' here may be the label, not the model.")

IN_SAMPLE_NOTE = (
    "This patient is one of the 303 the models were trained on, so the live score is in-sample "
    "and optimistic. The cross-validated value is what each model predicted when this patient "
    "was held out of training, which is the fair estimate.")


def _to_frontend(value, column: str):
    """Dataset encoding -> the frontend form's encoding (inverse of adapter.to_model_input)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if column == "Sex":
        return "Female" if value == "Fmale" else value
    if value in ("Y", "N") and column not in ("BBB", "VHD"):
        return 1 if value == "Y" else 0
    if hasattr(value, "item"):                     # numpy scalar -> plain Python (int stays int)
        value = value.item()
    return value


def _mean_oof(project_root: Path) -> dict[int, dict[str, dict]]:
    """{row: {frontend_target: {"probability": mean, "min": .., "max": .., "repeats": n}}}"""
    per_row: dict[int, dict[str, list[float]]] = {}

    def add(row, target, p):
        per_row.setdefault(row, {}).setdefault(target, []).append(p)

    sel = project_root / OOF_SELECTED
    if sel.exists():
        with sel.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["task"] in ("Cath", "LCX", "RCA"):
                    add(int(r["row_index"]), FRONTEND_TARGET[r["task"]], float(r["oof_proba"]))
    lad = project_root / OOF_LAD_SIGMOID
    if lad.exists():
        with lad.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                add(int(r["row_index"]), "LAD", float(r["p_sigmoid"]))

    return {row: {t: {"probability": sum(v) / len(v), "min": min(v), "max": max(v), "repeats": len(v)}
                  for t, v in targets.items()}
            for row, targets in per_row.items()}


class DatasetPatients:
    """Rows of the (hash-checked) training dataset in the frontend's format."""

    def __init__(self, data_path: Path, project_root: Path, frontend_fields: list[str] | None = None):
        df = pd.read_excel(data_path, sheet_name=0)
        self.targets = df[list(TARGET_POSITIVE)].copy()
        self.inputs = df.drop(columns=list(TARGET_POSITIVE))
        self.fields = frontend_fields or list(self.inputs.columns)
        self.cv = _mean_oof(project_root)

    def __len__(self):
        return len(self.inputs)

    def get(self, row: int) -> dict:
        if not 0 <= row < len(self):
            raise IndexError(row)
        rec = self.inputs.iloc[row]
        features = {c: _to_frontend(rec[c], c) for c in self.fields if c in rec.index}
        actual = {FRONTEND_TARGET[t]: int(self.targets.iloc[row][t] == pos) for t, pos in TARGET_POSITIVE.items()}
        # CAD is defined as stenosis in any artery, so CAD and the vessel labels must agree.
        any_vessel = any(actual[v] for v in ("LAD", "LCX", "RCA"))
        conflict = LABEL_CONFLICT_NOTE if actual["CAD"] != int(any_vessel) else None
        return {
            "row": row,
            "n_rows": len(self),
            "features": features,
            "actual": actual,
            "actual_source": "Coronary angiography (dataset labels). Shown for comparison only; never a model input.",
            "cv": self.cv.get(row, {}),
            "cv_source": "Mean out-of-fold probability over 5 repeats of 5-fold cross-validation "
                         "(reports/final_ml_evaluation and reports/lad_calibration).",
            "note": IN_SAMPLE_NOTE,
            "label_conflict": conflict,
        }
