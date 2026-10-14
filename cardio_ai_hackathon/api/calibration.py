"""
Calibration ("risk estimation quality") of the deployed models, read from the saved reports.

Read-only; nothing is recomputed. For each target: 10 groups of ~30 patients sorted by their
cross-validated (out-of-fold) probability, each with the mean predicted probability and the
observed proportion with a 95% Wilson interval, plus the repeat-averaged Brier score and
calibration slope.

Sources (written by the ML pipeline's evaluation scripts):
  * Cath, LCX, RCA: reports/final_ml_evaluation/calibration_bins.csv and bootstrap_confidence_intervals.csv
  * LAD (the deployed sigmoid-calibrated forest): reports/lad_calibration/calibration_bins.csv
    (approach = "sigmoid") and calibration_comparison.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from .adapter import FRONTEND_TARGET

FINAL = Path("reports/final_ml_evaluation")
LAD = Path("reports/lad_calibration")


def _rows(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _bin(r) -> dict:
    return {"n": int(r["n_patients"]), "mean_predicted": float(r["mean_predicted"]),
            "observed": float(r["observed_rate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])}


def load_calibration(project_root: Path) -> dict:
    out: dict[str, dict] = {}
    for r in _rows(project_root / FINAL / "calibration_bins.csv"):
        if r["task"] in ("Cath", "LCX", "RCA"):
            out.setdefault(FRONTEND_TARGET[r["task"]], {"bins": []})["bins"].append(_bin(r))
    for r in _rows(project_root / LAD / "calibration_bins.csv"):
        if r["approach"] == "sigmoid":
            out.setdefault("LAD", {"bins": []})["bins"].append(_bin(r))

    for r in _rows(project_root / FINAL / "bootstrap_confidence_intervals.csv"):
        t = FRONTEND_TARGET.get(r["task"])
        if t in out and t != "LAD" and r["metric"] in ("brier", "calibration_slope"):
            out[t][r["metric"]] = {"estimate": float(r["estimate"]), "ci_low": float(r["ci95_low"]),
                                   "ci_high": float(r["ci95_high"])}
    for r in _rows(project_root / LAD / "calibration_comparison.csv"):
        if "LAD" in out and r["metric"] in ("brier", "calibration_slope"):
            out["LAD"][r["metric"]] = {"estimate": float(r["sigmoid"]), "ci_low": float(r["sigmoid_ci95_low"]),
                                       "ci_high": float(r["sigmoid_ci95_high"])}
    for t in out:
        out[t]["bins"].sort(key=lambda b: b["mean_predicted"])
    return out


NOTE = ("Each point is a group of about 30 patients, sorted by their cross-validated probability: "
        "the mean predicted probability (x) against the share who actually had the condition (y), "
        "with a 95% interval. Points on the diagonal mean the probabilities can be read at face value.")
