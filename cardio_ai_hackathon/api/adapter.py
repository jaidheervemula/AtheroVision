"""
Translation between the AtheroVision frontend and the cad package.

Pure functions only (no web framework), so they can be unit-tested on their own.
Nothing here changes how a prediction is made: inputs are only re-encoded into the
exact form `cad.inference.CADPredictor` validates, and outputs are only re-shaped.

Frontend -> model
  * the frontend sends 55 fields; the four models together use 48 (config input_schema).
    The rest (e.g. Weight/Height, used only to compute BMI) are not sent to the models
    and are listed back as `unused_inputs`.
  * encodings: Sex "Female" -> "Fmale" (the dataset's spelling); 0/1 switches -> "N"/"Y"
    for fields the dataset stores as text; whole numbers stay whole for integer fields.

Model -> frontend
  * task "Cath" is shown as "CAD".
  * each task carries its explanation in its own units: log-odds (logistic regression,
    exact and additive) or probability (random forest, approximate, not additive).
"""

from __future__ import annotations

import math

FRONTEND_TARGET = {"Cath": "CAD", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}
TARGET_FIELDS = ("Cath", "LAD", "LCX", "RCA")
SEX_TO_MODEL = {"Female": "Fmale", "Fmale": "Fmale", "Male": "Male"}
YN_TO_MODEL = {1: "Y", 0: "N", "Y": "Y", "N": "N", True: "Y", False: "N"}

# Fields the frontend collects only to derive another input (BMI); not "unused" for the user.
DERIVATION_INPUTS = {"Weight", "Length"}


class AdapterError(ValueError):
    """Raised when frontend values cannot be re-encoded. `.errors` lists every problem."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def to_model_input(features: dict, schema: dict) -> tuple[dict, list[str]]:
    """Re-encode frontend features into the model's input schema.

    Returns (patient, unused_inputs). Raises AdapterError for values that cannot be
    re-encoded; range and category checks are left to CADPredictor.validate_input.
    """
    if not isinstance(features, dict):
        raise AdapterError(["'features' must be an object of field -> value."])
    patient, errors = {}, []
    # Leakage prevention: targets are rejected loudly (as CADPredictor does), never silently dropped.
    for t in TARGET_FIELDS:
        if t in features:
            errors.append(f"{t!r} is a prediction target and cannot be supplied as an input.")
    for name, spec in schema.items():
        if name not in features:
            errors.append(f"Missing field {name!r}.")
            continue
        raw = features[name]
        if spec["kind"] == "numeric":
            n = _as_number(raw)
            if n is None or not math.isfinite(n):
                errors.append(f"{name!r} must be a number, got {raw!r}.")
                continue
            patient[name] = int(n) if spec["dtype"] == "integer" and float(n).is_integer() else float(n)
            continue
        allowed = spec["allowed_values"]
        if all(isinstance(a, int) for a in allowed):              # stored as 0/1 numbers
            n = _as_number(raw)
            patient[name] = int(n) if n is not None and float(n).is_integer() else raw
        elif name == "Sex":
            patient[name] = SEX_TO_MODEL.get(raw, raw)
        elif set(allowed) == {"N", "Y"}:                          # stored as Y/N text
            key = int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else raw
            patient[name] = YN_TO_MODEL.get(key, raw)
        else:                                                     # BBB, VHD: same labels
            patient[name] = raw
    if errors:
        raise AdapterError(errors)
    unused = sorted(k for k in features if k not in schema and k not in DERIVATION_INPUTS)
    return patient, unused


def _task_summary(task: str, prediction: dict, threshold: float) -> dict:
    p = prediction["probability"]
    return {
        "probability": p,
        "label": 1 if prediction["predicted_positive"] else 0,
        "threshold": threshold,
        "predicted_label": prediction["predicted_label"],
        "model_name": prediction["model_name"],
        "model_type": prediction["model_type"],
        "n_input_features": prediction["n_input_features"],
    }


def _explanation(expl: dict, features: dict) -> dict:
    """Map one cad.explain task entry onto the frontend's contribution fields."""
    log_odds = expl["units"] == "log-odds"
    return {
        "units": "log-odds" if log_odds else "probability",
        "additive": bool(expl["additive"]),
        "method": expl["method_description"],
        # base_value is in the same units as the contributions.
        "base_value": expl["reference_log_odds"] if log_odds else expl["reference_probability"],
        "reference_probability": expl["reference_probability"],
        "reference_note": expl["reference_note"],
        "non_additive_gap": expl.get("non_additive_gap"),
        "features_not_used": list(expl["features_not_used_by_model"]),
        "contributions": [
            # Show the value as the user entered it (e.g. "Female", 1), not the model encoding.
            {"feature": c["feature"], "value": features.get(c["feature"], c["value"]), "shap": c["contribution"]}
            for c in expl["contributions"]
        ],
        "limitations": list(expl.get("limitations", [])),
    }


def model_version(config: dict) -> str:
    names = ", ".join(f"{FRONTEND_TARGET[t]} {config['tasks'][t]['model_name']}" for t in ("Cath", "LAD", "LCX", "RCA"))
    return f"{names} (selected {config.get('created_utc', '')[:10]})"


def _agreement(predictions: dict) -> dict:
    """Same rule as cad.inference, applied to the flags at each target's own cut-off."""
    cad = bool(predictions["CAD"]["label"])
    vessels = [v for v in ("LAD", "LCX", "RCA") if predictions[v]["label"]]
    disagree = (cad and not vessels) or (not cad and bool(vessels))
    if not disagree:
        note = "The overall CAD model and the vessel models agree at these cut-offs."
    elif cad:
        note = ("Overall CAD is flagged but no vessel model is. The models are independent; "
                "this disagreement is shown, not corrected.")
    else:
        note = (f"{', '.join(vessels)} flagged while overall CAD is not. The models are independent; "
                "this disagreement is shown, not corrected.")
    return {"models_disagree": bool(disagree), "note": note}


def build_response(config: dict, features: dict, prediction: dict, explanation: dict | None,
                   unused_inputs: list[str], operating_points: dict | None = None) -> dict:
    """Assemble the frontend's PredictionResponse from cad.predict() and cad.explain() output.

    `operating_points` (api/operating_points.py) gives each target's flagging cut-off. The
    probability is never changed; only `label`/`threshold` follow the cut-off. The package's own
    0.5 decision is kept as `default_label`/`default_threshold`.
    """
    default_threshold = prediction["threshold"]
    predictions = {}
    for task, fe in FRONTEND_TARGET.items():
        entry = _task_summary(task, prediction["predictions"][task], default_threshold)
        entry["default_threshold"] = default_threshold
        entry["default_label"] = entry["label"]
        op = (operating_points or {}).get(fe)
        if op:
            entry["threshold"] = op["threshold"]
            entry["label"] = 1 if entry["probability"] >= op["threshold"] else 0
            entry["operating_point"] = {"mode": op["mode"], "reason": op["reason"],
                                        "cv_sensitivity": (op["cv"] or {}).get("sensitivity"),
                                        "cv_specificity": (op["cv"] or {}).get("specificity")}
        if explanation is not None:
            entry.update(_explanation(explanation["explanations"][task], features))
        predictions[fe] = entry
    return {
        "model_version": model_version(config),
        "explained": explanation is not None,
        "predictions": predictions,
        "agreement": _agreement(predictions),
        "threshold_note": (prediction["threshold_note"] if not operating_points else
                           "Each target is flagged at its own cut-off (see operating_point); the probability is "
                           "the primary output. " + prediction["threshold_note"]),
        "warnings": list(prediction["warnings"]),
        "unused_inputs": unused_inputs,
        "disclaimer": prediction["disclaimer"],
        "limitations": list(prediction["limitations"]),
    }


METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
               "balanced_accuracy", "specificity")


def build_metrics(config: dict, n_samples: int, operating_points: dict | None = None, rule: str | None = None) -> dict:
    """Cross-validated metrics and exact training ranges from the model-selection config."""
    v = config.get("validation", {})
    validation = (f"Repeated stratified {v.get('n_splits', '?')}-fold CV, {v.get('n_repeats', '?')} repeats"
                  if v.get("type") == "RepeatedStratifiedKFold" else str(v.get("type", "Cross-validation")))
    targets, sd = {}, {}
    for task, fe in FRONTEND_TARGET.items():
        m = config["tasks"][task]["cv_metrics"]
        targets[fe] = {k: m[k]["mean"] for k in METRIC_KEYS if k in m}
        sd[fe] = {k: m[k]["sd"] for k in METRIC_KEYS if k in m}
        # A task deployed with a calibrated model (LAD) reports that model's own cross-validated
        # estimates, recorded in the config by scripts/promote_lad_calibration.py. The baseline
        # cv_metrics belong to the uncalibrated forest, so their fold SDs are dropped too.
        cal = (config["tasks"][task].get("calibration") or {}).get("oof_metrics")
        if cal:
            mapping = {"roc_auc": "roc_auc", "brier": "brier", "accuracy": "accuracy_at_0_5",
                       "precision": "precision_at_0_5", "recall": "sensitivity_at_0_5",
                       "f1": "f1_at_0_5", "specificity": "specificity_at_0_5"}
            for k, src in mapping.items():
                if src in cal:
                    targets[fe][k] = cal[src]
                    sd[fe].pop(k, None)
            for k in ("pr_auc", "balanced_accuracy"):      # not recorded for the calibrated model
                targets[fe].pop(k, None)
                sd[fe].pop(k, None)
    ranges = {name: [spec["training_min"], spec["training_max"]]
              for name, spec in config["input_schema"].items() if spec["kind"] == "numeric"}
    return {
        "validation": validation,
        "n_samples": n_samples,
        "targets": targets,
        "sd": sd,
        "threshold": 0.5,
        "models": {FRONTEND_TARGET[t]: {"name": config["tasks"][t]["model_name"],
                                         "type": config["tasks"][t]["model_type"],
                                         "n_input_features": len(config["tasks"][t]["input_features"])}
                   for t in FRONTEND_TARGET},
        "feature_ranges": ranges,
        "limitations": list(config.get("limitations", [])),
        "operating_points": operating_points or {},
        "operating_rule": rule,
    }
