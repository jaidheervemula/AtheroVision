"""
Decision cut-offs ("operating points") used to flag a target as positive.

The models and their probabilities are unchanged; this only decides at which probability a
result is *flagged*. The project's default is 0.5 for every target (untuned). At 0.5 the LCX
and RCA models miss about half of the stenotic vessels in cross-validation, which is the wrong
trade-off for a screening-style tool, where a missed stenosis costs more than a false alarm.

Rule, fixed before looking at individual cut-offs and applied the same way to every target:
    use the highest cut-off on a 0.05 grid, at or below 0.5, whose cross-validated sensitivity
    is at least MIN_SENSITIVITY (0.80). Cut-offs are never raised above the 0.5 default.
Result with the saved predictions: CAD 0.50, LAD 0.50 (both already >= 0.80), LCX 0.30, RCA 0.30.

Source: the saved out-of-fold predictions of the deployed models (5 repeats of 5-fold CV):
  * Cath, LCX, RCA: reports/final_ml_evaluation/oof_per_repeat_selected.csv
  * LAD (sigmoid-calibrated forest): reports/lad_calibration/lad_calibration_predictions.csv
Metrics are computed per repeat and averaged, as in reports/final_ml_evaluation/threshold_tradeoffs.csv.

Caveat (also in MODEL_CARD.md): the cut-off is chosen on the same cross-validation results it
is reported on, so the sensitivity/specificity at the chosen cut-off are slightly optimistic.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .adapter import FRONTEND_TARGET

DEFAULT_THRESHOLD = 0.5
MIN_SENSITIVITY = 0.80
GRID = [round(0.05 * i, 2) for i in range(1, 11)]          # 0.05 ... 0.50

OOF_SELECTED = Path("reports/final_ml_evaluation/oof_per_repeat_selected.csv")
OOF_LAD_SIGMOID = Path("reports/lad_calibration/lad_calibration_predictions.csv")


def _load_oof(project_root: Path) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """{frontend_target: {repeat: [(y_true, probability), ...]}}"""
    out: dict[str, dict[int, list[tuple[int, float]]]] = {}
    sel = project_root / OOF_SELECTED
    if sel.exists():
        with sel.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["task"] in ("Cath", "LCX", "RCA"):
                    out.setdefault(FRONTEND_TARGET[r["task"]], {}).setdefault(int(r["repeat"]), []).append(
                        (int(r["y_true"]), float(r["oof_proba"])))
    lad = project_root / OOF_LAD_SIGMOID
    if lad.exists():
        with lad.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.setdefault("LAD", {}).setdefault(int(r["repeat"]), []).append(
                    (int(r["y_true"]), float(r["p_sigmoid"])))
    return out


def metrics_at(repeats: dict[int, list[tuple[int, float]]], threshold: float) -> dict[str, float]:
    """Sensitivity, specificity, precision, NPV and accuracy at `threshold`, averaged over repeats."""
    keys = ("sensitivity", "specificity", "precision", "npv", "accuracy")
    sums = dict.fromkeys(keys, 0.0)
    for rows in repeats.values():
        tp = sum(1 for y, p in rows if y == 1 and p >= threshold)
        fn = sum(1 for y, p in rows if y == 1 and p < threshold)
        fp = sum(1 for y, p in rows if y == 0 and p >= threshold)
        tn = sum(1 for y, p in rows if y == 0 and p < threshold)
        sums["sensitivity"] += tp / (tp + fn) if tp + fn else 0.0
        sums["specificity"] += tn / (tn + fp) if tn + fp else 0.0
        sums["precision"] += tp / (tp + fp) if tp + fp else 0.0
        sums["npv"] += tn / (tn + fn) if tn + fn else 0.0
        sums["accuracy"] += (tp + tn) / len(rows)
    n = len(repeats)
    return {k: v / n for k, v in sums.items()}


def compute_operating_points(project_root: Path) -> dict[str, dict]:
    oof = _load_oof(project_root)
    points = {}
    for target in ("CAD", "LAD", "LCX", "RCA"):
        repeats = oof.get(target)
        if not repeats:                       # reports missing: keep the project default
            points[target] = {"threshold": DEFAULT_THRESHOLD, "mode": "default", "cv": None, "cv_at_default": None,
                              "reason": "Cross-validation predictions not found; using the 0.5 default."}
            continue
        at_default = metrics_at(repeats, DEFAULT_THRESHOLD)
        chosen = DEFAULT_THRESHOLD
        if at_default["sensitivity"] < MIN_SENSITIVITY:
            ok = [t for t in GRID if metrics_at(repeats, t)["sensitivity"] >= MIN_SENSITIVITY]
            chosen = max(ok) if ok else DEFAULT_THRESHOLD
        cv = metrics_at(repeats, chosen)
        if chosen == DEFAULT_THRESHOLD:
            reason = (f"0.5 default kept: it already catches {at_default['sensitivity']:.0%} of positive "
                      f"patients in cross-validation.")
        else:
            reason = (f"Screening cut-off {chosen:.2f}: catches {cv['sensitivity']:.0%} of stenotic {target} "
                      f"in cross-validation (vs {at_default['sensitivity']:.0%} at 0.5), at the cost of more "
                      f"false alarms (specificity {cv['specificity']:.0%} vs {at_default['specificity']:.0%}).")
        points[target] = {"threshold": chosen, "mode": "default" if chosen == DEFAULT_THRESHOLD else "screening",
                          "cv": cv, "cv_at_default": at_default, "reason": reason}
    return points


RULE = (f"Each target is flagged at the highest cut-off (0.05 steps, never above 0.5) that catches at least "
        f"{MIN_SENSITIVITY:.0%} of positive patients in cross-validation. The probabilities themselves are "
        f"unchanged. Because the cut-off was chosen on the same cross-validation results, the metrics at the "
        f"chosen cut-off are slightly optimistic.")
