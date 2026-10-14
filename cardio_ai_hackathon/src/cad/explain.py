"""
cad.explain
===========
Per-patient explanations for the four selected pipelines.

EDUCATIONAL DECISION-SUPPORT PROTOTYPE (hackathon). NOT A DIAGNOSTIC TOOL.
Explanations describe how each MODEL's output responds to the inputs. They show
statistical associations learned from 303 patients at one centre; they are not
causes, medical reasons, or evidence of disease.

Two methods are used, chosen for the actual saved pipelines:

1. Logistic regression (Cath, LCX, RCA) -- EXACT ADDITIVE DECOMPOSITION IN LOG-ODDS
   The pipeline computes   log-odds(x) = b + sum_k w_k * z_k(x)
   where z(x) is the pipeline's own preprocessing output (scaled numbers, 0/1 one-hot
   columns), w the fitted coefficients and b the intercept. With z_ref = the average
   encoded value over the 303 training patients:
       contribution_k = w_k * (z_k(x) - z_ref_k)
       log-odds(x)    = reference_log_odds + sum_k contribution_k     (exact)
   Encoded columns are summed back to their original input feature (e.g. the VHD
   one-hot columns become one 'VHD' contribution). Units are LOG-ODDS, not
   percentage points of probability. This equals SHAP's linear explanation under a
   feature-independence assumption, computed directly without the shap package.

2. Random forest (LAD) -- APPROXIMATE SINGLE-FEATURE BACKGROUND REPLACEMENT
   For each input feature j, the patient's value is replaced in turn by the value
   of feature j from each of the 303 training patients (all other inputs kept), and
   the model's probability is averaged:
       impact_j = p(x) - mean_over_training_rows p(x with x_j replaced)
   Replacement values are real recorded values, so categories and ranges stay valid.
   Units are probability (0-1 scale). This is NOT additive: the impacts need not sum
   to p(x) minus the average prediction, and the gap is reported. Replacing one
   feature at a time ignores correlations between features, so some replaced rows
   are combinations unlikely in real patients.

The training dataset is read (never modified) to provide the reference/background
values; its SHA-256 must match the configuration.
"""

import hashlib
import math

import numpy as np
import pandas as pd

from .inference import (
    DISCLAIMER,
    DISPLAY_NAMES,
    PROJECT_ROOT,
    TARGET_COLUMNS,
    TASK_ORDER,
    ArtifactError,
    CADPredictor,
)

LR_METHOD = "exact_linear_decomposition_log_odds"
RF_METHOD = "single_feature_background_replacement"
ZERO_TOL = 1e-12

LR_LIMITATIONS = [
    "Contributions are in log-odds units relative to a reference (the training-average "
    "encoded input); they are not percentage-point changes in probability.",
    "The decomposition is exact for this model, but it treats features as independent; "
    "correlated inputs (e.g. BMI and weight-related flags) share influence arbitrarily.",
    "A contribution shows how the model's output responds to an input. It is a "
    "statistical association from 303 patients, not a cause or a medical reason.",
]
RF_LIMITATIONS = [
    "Approximate: impacts come from replacing one feature at a time with values seen in "
    "the training data; they are not an exact or additive decomposition.",
    "Impacts are in probability units (0-1). They do not sum to the difference between "
    "this prediction and the average prediction; the remaining gap is reported.",
    "Replacing one feature at a time ignores correlations between features, so some "
    "evaluated combinations may be unrealistic for real patients.",
    "The background is the training data the model was fitted on, so the reference "
    "(average model output over those patients) is in-sample and optimistic.",
    "An impact shows how the model's output responds to an input. It is a statistical "
    "association from 303 patients, not a cause or a medical reason.",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


def _direction(value, tol):
    if value > tol:
        return "raises model output"
    if value < -tol:
        return "lowers model output"
    return "no measurable effect"


def encoded_to_input_map(preprocessor):
    """List giving, for every encoded column, the original input feature it came from.

    Built from the fitted ColumnTransformer: one column per numeric feature, and for each
    categorical feature its one-hot categories minus the dropped one (if any).
    """
    owners = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if name == "num":
            owners.extend(columns)
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            drop_idx = getattr(onehot, "drop_idx_", None)
            for i, col in enumerate(columns):
                n_out = len(onehot.categories_[i])
                if drop_idx is not None and drop_idx[i] is not None:
                    n_out -= 1
                owners.extend([col] * n_out)
        else:
            raise ArtifactError(f"Unexpected preprocessing branch {name!r}; explanation "
                                "mapping is not defined for it.")
    return owners


class CADExplainer:
    """Explains the predictions of a CADPredictor's four models for one patient."""

    def __init__(self, predictor=None):
        self.predictor = predictor if predictor is not None else CADPredictor()
        self.background = self._load_background()
        self._lr_cache = {}

    # ------------------------------------------------------------------ setup
    def _load_background(self):
        config = self.predictor.config
        path = PROJECT_ROOT / "data" / "raw" / config["data_file"]
        if not path.exists():
            raise ArtifactError(f"Training data needed for explanation reference not found: {path}")
        if _sha256(path) != config["data_sha256"]:
            raise ArtifactError("Training data file does not match the SHA-256 in the config.")
        df = pd.read_excel(path, sheet_name=0)
        columns = list(self.predictor.schema)            # inputs only, never targets
        leaked = [c for c in columns if c in TARGET_COLUMNS]
        if leaked:
            raise ArtifactError(f"Target column(s) {leaked} in the explanation background.")
        return df[columns].reset_index(drop=True)

    def _lr_parts(self, task):
        """Fitted pieces of a logistic-regression pipeline, computed once per task."""
        if task not in self._lr_cache:
            info = self.predictor.tasks[task]
            pipe = info["pipeline"]
            pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
            owners = encoded_to_input_map(pre)
            z_ref = np.asarray(pre.transform(self.background[info["features"]]), dtype=float).mean(axis=0)
            if len(owners) != len(z_ref) or len(owners) != model.coef_.shape[1]:
                raise ArtifactError(f"{task}: encoded column mapping does not match the model.")
            self._lr_cache[task] = {
                "pre": pre, "model": model, "owners": owners, "z_ref": z_ref,
                "w": model.coef_[0].astype(float), "b": float(model.intercept_[0]),
            }
        return self._lr_cache[task]

    # ------------------------------------------------------------------ methods
    def _frame(self, clean, task):
        features = self.predictor.features_for(task)
        return pd.DataFrame([{f: clean[f] for f in features}], columns=features)

    def _explain_logistic(self, clean, task):
        parts = self._lr_parts(task)
        X = self._frame(clean, task)
        z = np.asarray(parts["pre"].transform(X), dtype=float)[0]
        log_odds = float(parts["model"].decision_function(z.reshape(1, -1))[0])
        reference = parts["b"] + float(parts["z_ref"] @ parts["w"])
        encoded_contrib = parts["w"] * (z - parts["z_ref"])
        per_feature = {}
        for owner, value in zip(parts["owners"], encoded_contrib):
            per_feature[owner] = per_feature.get(owner, 0.0) + float(value)
        total = sum(per_feature.values())
        error = abs(reference + total - log_odds)
        if error > 1e-8:
            raise RuntimeError(f"{task}: log-odds decomposition failed (error {error:.2e}).")
        return {
            "method": LR_METHOD,
            "method_description": "Exact additive decomposition of the logistic-regression "
                                  "log-odds relative to the training-average encoded input.",
            "additive": True,
            "units": "log-odds",
            "reference_log_odds": reference,
            "reference_probability": _sigmoid(reference),
            "reference_note": "Model output at the training-average encoded input; this is "
                              "not the average predicted probability.",
            "model_log_odds": log_odds,
            "sum_of_contributions": total,
            "reconstruction_error": error,
            "raw": per_feature,
            "tol": ZERO_TOL,
            "limitations": list(LR_LIMITATIONS),
        }

    def _explain_replacement(self, clean, task):
        info = self.predictor.tasks[task]
        features = info["features"]
        pipe = info["pipeline"]
        x = self._frame(clean, task)
        p = float(pipe.predict_proba(x)[0, 1])
        bg = self.background[features]
        n = len(bg)
        blocks = []
        for f in features:
            block = pd.concat([x] * n, ignore_index=True)
            block[f] = bg[f].to_numpy()
            blocks.append(block)
        probs = pipe.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1]
        per_feature = {f: p - float(probs[i * n:(i + 1) * n].mean()) for i, f in enumerate(features)}
        average_prediction = float(pipe.predict_proba(bg)[:, 1].mean())
        total = sum(per_feature.values())
        return {
            "method": RF_METHOD,
            "method_description": "Approximate local impact: change in predicted probability "
                                  "when one input at a time is replaced by values from the "
                                  "training data (all other inputs unchanged).",
            "additive": False,
            "units": "probability (0-1 scale)",
            "reference_probability": average_prediction,
            "reference_note": "Average model output over the 303 training patients (in-sample).",
            "sum_of_contributions": total,
            "non_additive_gap": (p - average_prediction) - total,
            "raw": per_feature,
            "tol": 1e-9,
            "limitations": list(RF_LIMITATIONS),
        }

    # ------------------------------------------------------------------ public
    def explain(self, patient, tasks=None, top_k=None, allow_out_of_range=False):
        """Explain one patient's predictions.

        patient            dict of input fields; validated by CADPredictor (same rules,
                           same errors: InputValidationError for invalid input)
        tasks              None for all four, or a list such as ["LAD"]
        top_k              keep only the k most influential features per task (None = all)
        allow_out_of_range passed to the predictor's validation

        Returns {"disclaimer", "warnings", "explanations": {task: {...}}}. Each task entry
        has the model, its positive-class probability (identical to CADPredictor.predict),
        the method and units, a reference value, and "contributions" ranked by absolute
        size, each with the patient's value and the direction of the effect.
        """
        tasks = list(TASK_ORDER) if tasks is None else list(tasks)
        unknown = [t for t in tasks if t not in TASK_ORDER]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown}; choose from {list(TASK_ORDER)}.")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
            raise ValueError("top_k must be a positive whole number or None.")

        prediction = self.predictor.predict(patient, allow_out_of_range=allow_out_of_range)
        clean, _ = self.predictor.validate_input(patient, allow_out_of_range=allow_out_of_range)

        explanations = {}
        for task in tasks:
            info = self.predictor.tasks[task]
            probability = prediction["predictions"][task]["probability"]
            if info["model_type"] == "LogisticRegression":
                result = self._explain_logistic(clean, task)
                if abs(_sigmoid(result["model_log_odds"]) - probability) > 1e-9:
                    raise RuntimeError(f"{task}: explanation does not match the prediction.")
            elif info["model_type"] == "RandomForestClassifier":
                result = self._explain_replacement(clean, task)
            else:
                raise ArtifactError(f"{task}: no explanation method for {info['model_type']}.")

            raw, tol = result.pop("raw"), result.pop("tol")
            ranked = sorted(raw.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
            contributions = [{"feature": f, "value": clean[f], "contribution": c,
                              "direction": _direction(c, tol)} for f, c in ranked]
            omitted = 0
            if top_k is not None and len(contributions) > top_k:
                omitted = len(contributions) - top_k
                contributions = contributions[:top_k]
            explanations[task] = {
                "task": task,
                "display_name": DISPLAY_NAMES[task],
                "model_name": info["model_name"],
                "model_type": info["model_type"],
                "positive_label": info["positive_label"],
                "probability": probability,
                **result,
                "contributions": contributions,
                "n_features_omitted": omitted,
                "features_not_used_by_model": [f for f in self.predictor.schema
                                               if f not in info["features"]],
            }
        return {"disclaimer": DISCLAIMER,
                "interpretation_note": "Contributions describe how the model's output responds "
                                       "to each input. They are associations, not causes, and "
                                       "do not establish a diagnosis.",
                "warnings": prediction["warnings"],
                "explanations": explanations}
