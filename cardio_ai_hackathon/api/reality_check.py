"""
"Reality check" for a prediction: what actually happened to similar-scoring patients.

For a probability p and a target, take the K dataset patients whose *cross-validated* score
(predicted while they were held out of training; see api/dataset.py) is closest to p, and report
how many of them truly had the disease on angiography, with a Wilson 95% interval.

This turns a model score into an observed frequency ("of 40 patients scored like this, 12 had
LCX stenosis"), which is easier to interpret and is honest about uncertainty. It uses only saved
cross-validation predictions and the dataset labels; no model is refitted. Limitations: it is
the same 303 patients the models were developed on, so it is an internal check, not external
validation, and at the extremes the neighbours span a wider score range.
"""

from __future__ import annotations

import math

K_NEIGHBOURS = 40


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class RealityCheck:
    def __init__(self, dataset, k: int = K_NEIGHBOURS):
        """`dataset` is an api.dataset.DatasetPatients."""
        self.k = k
        self.points: dict[str, list[tuple[float, int, int]]] = {}     # (cv score, outcome, row)
        for row in range(len(dataset)):
            d = dataset.get(row)
            for target, cv in d["cv"].items():
                self.points.setdefault(target, []).append((cv["probability"], d["actual"][target], row))

    def check(self, target: str, probability: float, exclude_row: int | None = None) -> dict | None:
        """`exclude_row`: when the patient IS a dataset patient, leave them out of their own comparison."""
        pts = [p for p in self.points.get(target, []) if p[2] != exclude_row]
        if not pts:
            return None
        nearest = sorted(pts, key=lambda sp: (abs(sp[0] - probability), sp[0]))[: self.k]
        n = len(nearest)
        k = sum(y for _, y, _ in nearest)
        lo, hi = wilson(k, n)
        return {
            "n": n,
            "positives": k,
            "rate": k / n,
            "ci_low": lo,
            "ci_high": hi,
            "score_min": min(s for s, _, _ in nearest),
            "score_max": max(s for s, _, _ in nearest),
            "excluded_row": exclude_row,
        }

    def add_to_response(self, response: dict, exclude_row: int | None = None) -> dict:
        for target, pred in response["predictions"].items():
            rc = self.check(target, pred["probability"], exclude_row)
            if rc is not None:
                pred["reality_check"] = rc
        return response
