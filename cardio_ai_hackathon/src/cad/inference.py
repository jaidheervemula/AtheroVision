"""
cad.inference
=============
Inference layer for the four selected Phase 1 pipelines (Cath, LAD, LCX, RCA).

EDUCATIONAL DECISION-SUPPORT PROTOTYPE (hackathon). NOT A DIAGNOSTIC TOOL.
The models were developed on 303 patients from one centre, with no external and no
clinical validation. Probabilities are model scores, not validated risk estimates.

What this module does
  * Reads config/model_selection.json (written by scripts/build_model_selection.py).
  * Loads the four saved scikit-learn pipelines from the paths in that config and
    checks each file's SHA-256 hash, classes and input feature list before use.
  * Validates patient input STRICTLY against the config's input schema. The saved
    pipelines would otherwise silently accept a missing number (median-imputed) or an
    unknown category (encoded as zeros), so validation happens BEFORE prediction.
  * Calls each pipeline's own predict_proba(); preprocessing is never rebuilt here.
    Column 1 of predict_proba is the positive class ("CAD" for Cath, "Stenotic" for
    LAD/LCX/RCA).
  * Returns all four probabilities plus 0/1 predictions at a threshold (default 0.5,
    NOT clinically optimised) and reports, rather than hides, disagreement between
    the independent models.

Typical use
    from cad.inference import CADPredictor
    predictor = CADPredictor()             # loads and checks the four models
    result = predictor.predict(patient)    # patient = dict of the 48 input fields
"""

import difflib
import hashlib
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model_selection.json"

TARGET_COLUMNS = ("Cath", "LAD", "LCX", "RCA")
TASK_ORDER = ("Cath", "LAD", "LCX", "RCA")
DISPLAY_NAMES = {
    "Cath": "Overall CAD",
    "LAD": "LAD stenosis",
    "LCX": "LCX stenosis",
    "RCA": "RCA stenosis",
}
DEFAULT_THRESHOLD = 0.5
THRESHOLD_NOTE = ("Binary predictions use a fixed threshold (default 0.5). It was not tuned "
                  "or clinically optimised; the probability is the primary output.")
DISCLAIMER = ("Educational decision-support prototype built for a hackathon. Not a diagnostic "
              "tool and not a substitute for clinical judgement or diagnostic imaging. "
              "Developed on 303 patients from one centre; no external or clinical validation.")


# ----------------------------------------------------------------------------- errors
class InputValidationError(ValueError):
    """Raised when patient input does not match the schema. `.errors` lists every problem."""

    def __init__(self, errors):
        self.errors = list(errors)
        message = "Invalid patient input ({} problem(s)):\n  - {}".format(
            len(self.errors), "\n  - ".join(self.errors))
        super().__init__(message)


class ArtifactError(RuntimeError):
    """Raised when a model file or the configuration is missing, changed or inconsistent."""


# ----------------------------------------------------------------------------- helpers
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _suggest(value, options):
    """Return a 'did you mean' hint for a mistyped name or category, or ''."""
    text = str(value)
    lowered = {str(o).strip().lower(): o for o in options}
    if text.strip().lower() in lowered:
        return f" Did you mean {lowered[text.strip().lower()]!r}? (values are case-sensitive)"
    close = difflib.get_close_matches(text, [str(o) for o in options], n=1, cutoff=0.6)
    return f" Did you mean {close[0]!r}?" if close else ""


def _is_number(value):
    # bool is a subclass of int in Python; True/False must not pass as numbers.
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_))


def load_config(config_path=DEFAULT_CONFIG):
    """Read the model-selection configuration (JSON) and check its basic structure."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ArtifactError(f"Configuration not found: {config_path}. Run "
                            "scripts/build_model_selection.py first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("tasks", "input_schema"):
        if key not in config:
            raise ArtifactError(f"Configuration {config_path} has no '{key}' section.")
    missing_tasks = [t for t in TASK_ORDER if t not in config["tasks"]]
    if missing_tasks:
        raise ArtifactError(f"Configuration is missing task(s): {missing_tasks}")
    leaked = [t for t in TARGET_COLUMNS if t in config["input_schema"]]
    if leaked:
        raise ArtifactError(f"Target column(s) {leaked} appear in the input schema.")
    return config


def synthetic_example(schema):
    """A schema-valid SYNTHETIC input (not a real patient) for tests and smoke checks.

    Numeric fields take the midpoint of the training range (rounded for integer
    fields); categorical fields take the first allowed value. The values are chosen
    mechanically and have no clinical meaning.
    """
    example = {}
    for name, spec in schema.items():
        if spec["kind"] == "numeric":
            mid = (spec["training_min"] + spec["training_max"]) / 2
            example[name] = int(round(mid)) if spec["dtype"] == "integer" else round(mid, 2)
        else:
            example[name] = spec["allowed_values"][0]
    return example


# ----------------------------------------------------------------------------- predictor
class CADPredictor:
    """Loads the four selected pipelines once and serves validated predictions."""

    def __init__(self, config_path=DEFAULT_CONFIG, verify_hashes=True):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.schema = self.config["input_schema"]
        self.tasks = {}
        for task in TASK_ORDER:
            self.tasks[task] = self._load_task(task, self.config["tasks"][task], verify_hashes)
        self._check_schema_covers_models()

    # ------------------------------------------------------------------ loading
    def _load_task(self, task, spec, verify_hash):
        path = PROJECT_ROOT / spec["artifact"]
        if not path.exists():
            raise ArtifactError(f"{task}: model file not found: {path}")
        if verify_hash and _sha256(path) != spec["artifact_sha256"]:
            raise ArtifactError(
                f"{task}: {path.name} does not match the SHA-256 recorded in the config. "
                "The file was changed or replaced; re-run scripts/build_model_selection.py "
                "only if that change was intended.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = joblib.load(path)
        if caught:
            raise ArtifactError(f"{task}: loading {path.name} produced warnings (often a "
                                f"library-version mismatch): {caught[0].message}")
        classes = [int(c) for c in pipeline.classes_]
        if classes != [0, 1]:
            raise ArtifactError(f"{task}: expected classes [0, 1], found {classes}")
        features = list(pipeline.feature_names_in_)
        if features != spec["input_features"]:
            raise ArtifactError(f"{task}: the pipeline's input features differ from the config.")
        leaked = [f for f in features if f in TARGET_COLUMNS]
        if leaked:
            raise ArtifactError(f"{task}: target column(s) {leaked} are model inputs.")
        return {
            "pipeline": pipeline,
            "features": features,
            "model_name": spec["model_name"],
            "model_type": spec["model_type"],
            "positive_label": spec["positive_label"],
            "negative_label": spec["negative_label"],
            "cv_metrics": spec["cv_metrics"],
            "artifact": spec["artifact"],
        }

    def _check_schema_covers_models(self):
        for task, info in self.tasks.items():
            missing = [f for f in info["features"] if f not in self.schema]
            if missing:
                raise ArtifactError(f"{task}: features {missing} have no entry in the input schema.")

    # ------------------------------------------------------------------ public info
    def required_fields(self):
        """Every input field the four models need together (one form = all four models)."""
        return list(self.schema.keys())

    def features_for(self, task):
        """The exact input features used by one task's pipeline."""
        return list(self.tasks[task]["features"])

    # ------------------------------------------------------------------ validation
    def validate_input(self, patient, allow_out_of_range=False):
        """Check one patient's input against the schema.

        Returns (clean_values, warnings). Raises InputValidationError listing every
        problem. Rules:
          * the input must be a dict containing every required field and nothing else;
          * target columns (Cath, LAD, LCX, RCA) are never accepted as inputs;
          * numbers must be real numbers (not text, not True/False, not missing/NaN),
            and whole numbers for integer fields;
          * categorical values must exactly match an allowed value (case-sensitive);
          * numbers outside the training range are rejected by default, because the
            models never saw such values. With allow_out_of_range=True they are
            accepted and reported as warnings instead.
        The input dict itself is never modified.
        """
        if not isinstance(patient, dict):
            raise InputValidationError(
                [f"Patient input must be a dict of field -> value, got {type(patient).__name__}."])
        errors, notes, clean = [], [], {}

        targets = [k for k in patient if k in TARGET_COLUMNS]
        for k in targets:
            errors.append(f"{k!r} is a prediction target and cannot be supplied as an input.")
        unknown = [k for k in patient if k not in self.schema and k not in TARGET_COLUMNS]
        for k in unknown:
            errors.append(f"Unknown field {k!r}." + _suggest(k, self.schema.keys()))
        missing = [k for k in self.schema if k not in patient]
        if missing:
            errors.append(f"Missing required field(s): {missing}")

        for name, spec in self.schema.items():
            if name not in patient:
                continue
            value = patient[name]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                errors.append(f"{name!r} is missing (None/NaN). Every field is required; "
                              "the models must not silently fill in values.")
                continue
            if spec["kind"] == "numeric":
                if not _is_number(value):
                    errors.append(f"{name!r} must be a number, got {value!r} "
                                  f"({type(value).__name__}).")
                    continue
                if not math.isfinite(float(value)):
                    errors.append(f"{name!r} must be a finite number, got {value!r}.")
                    continue
                if spec["dtype"] == "integer":
                    if float(value) != int(float(value)):
                        errors.append(f"{name!r} must be a whole number, got {value!r}.")
                        continue
                    value = int(float(value))
                else:
                    value = float(value)
                lo, hi = spec["training_min"], spec["training_max"]
                if value < lo or value > hi:
                    msg = (f"{name!r} = {value} is outside the training range [{lo}, {hi}]; "
                           "the models have not seen such values.")
                    if allow_out_of_range:
                        notes.append(msg + " Prediction is an extrapolation.")
                    else:
                        errors.append(msg + " Use a value inside the range, or pass "
                                      "allow_out_of_range=True to accept it with a warning.")
                        continue
                clean[name] = value
            else:
                allowed = spec["allowed_values"]
                numeric_codes = all(isinstance(a, int) for a in allowed)
                if numeric_codes:
                    ok = _is_number(value) and float(value) in [float(a) for a in allowed]
                    if ok:
                        clean[name] = int(float(value))
                    else:
                        errors.append(f"{name!r} must be one of {allowed} (as a number), got "
                                      f"{value!r} ({type(value).__name__}).")
                else:
                    if isinstance(value, str) and value in allowed:
                        clean[name] = value
                    else:
                        errors.append(f"{name!r} must be one of {allowed}, got {value!r}."
                                      + _suggest(value, allowed))
        if errors:
            raise InputValidationError(errors)
        return clean, notes

    # ------------------------------------------------------------------ prediction
    def _frame(self, clean, task):
        """One-row DataFrame with exactly the columns, in the order, this pipeline expects."""
        features = self.tasks[task]["features"]
        return pd.DataFrame([{f: clean[f] for f in features}], columns=features)

    def predict(self, patient, threshold=DEFAULT_THRESHOLD, allow_out_of_range=False):
        """Validate one patient and return all four model outputs.

        Returns a dict:
          disclaimer, threshold, threshold_note
          predictions: {task: {display_name, probability, predicted_positive,
                               predicted_label, positive_label, negative_label,
                               model_name, model_type, n_input_features,
                               input_features_used, cv_roc_auc}}
          agreement:   {cad_predicted_positive, vessels_predicted_positive,
                        models_disagree, note}
          warnings:    input warnings (e.g. accepted out-of-range values)
          limitations: copied from the configuration
        """
        if not _is_number(threshold) or not 0 < float(threshold) < 1:
            raise ValueError(f"threshold must be a number strictly between 0 and 1, got {threshold!r}")
        threshold = float(threshold)
        clean, notes = self.validate_input(patient, allow_out_of_range=allow_out_of_range)

        predictions = {}
        for task in TASK_ORDER:
            info = self.tasks[task]
            X = self._frame(clean, task)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                proba = info["pipeline"].predict_proba(X)
            if caught:
                # Validation should make this impossible; refuse to return a doubtful value.
                raise RuntimeError(f"{task}: the pipeline warned during prediction "
                                   f"({caught[0].message}); result withheld.")
            p = float(proba[0, 1])              # column 1 = positive class (classes_ == [0, 1])
            positive = p >= threshold
            predictions[task] = {
                "display_name": DISPLAY_NAMES[task],
                "probability": p,
                "predicted_positive": bool(positive),
                "predicted_label": info["positive_label"] if positive else info["negative_label"],
                "positive_label": info["positive_label"],
                "negative_label": info["negative_label"],
                "model_name": info["model_name"],
                "model_type": info["model_type"],
                "n_input_features": len(info["features"]),
                "input_features_used": list(info["features"]),
                "cv_roc_auc": info["cv_metrics"]["roc_auc"],
            }

        cad = predictions["Cath"]["predicted_positive"]
        vessels = [t for t in ("LAD", "LCX", "RCA") if predictions[t]["predicted_positive"]]
        disagree = (cad and not vessels) or (not cad and bool(vessels))
        if not disagree:
            note = "The overall CAD model and the vessel models agree at this threshold."
        elif cad:
            note = ("Overall CAD is predicted positive but no vessel model is positive. The "
                    "models are independent; this disagreement is shown, not corrected.")
        else:
            note = (f"Vessel model(s) {vessels} are positive while overall CAD is negative. The "
                    "models are independent; this disagreement is shown, not corrected.")
        return {
            "disclaimer": DISCLAIMER,
            "threshold": threshold,
            "threshold_note": THRESHOLD_NOTE,
            "predictions": predictions,
            "agreement": {"cad_predicted_positive": cad,
                          "vessels_predicted_positive": vessels,
                          "models_disagree": bool(disagree),
                          "note": note},
            "warnings": notes,
            "limitations": list(self.config.get("limitations", [])),
        }
