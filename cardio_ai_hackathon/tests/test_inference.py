"""
Tests for src/cad/inference.py, written with Python's built-in unittest (no pytest needed).

Run from the project folder:
    python -m unittest discover -s tests -v

These tests check loading, schema handling, input validation and output structure.
They deliberately do NOT assert that any example is "high" or "low" risk: the
example inputs are synthetic and have no clinical meaning.
"""

import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from cad.inference import (  # noqa: E402
    DEFAULT_CONFIG,
    TARGET_COLUMNS,
    TASK_ORDER,
    ArtifactError,
    CADPredictor,
    InputValidationError,
    synthetic_example,
)

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CADPredictor()           # loads once for all tests
        cls.schema = cls.predictor.schema
        cls.example = synthetic_example(cls.schema)

    # ------------------------------------------------------------------ loading
    def test_all_four_models_load(self):
        self.assertEqual(list(self.predictor.tasks), list(TASK_ORDER))
        for task, info in self.predictor.tasks.items():
            self.assertTrue(hasattr(info["pipeline"], "predict_proba"), task)
            self.assertEqual([int(c) for c in info["pipeline"].classes_], [0, 1], task)

    def test_positive_labels(self):
        expected = {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}
        for task, label in expected.items():
            self.assertEqual(self.predictor.tasks[task]["positive_label"], label)
            self.assertEqual(self.predictor.tasks[task]["negative_label"], "Normal")

    def test_each_model_uses_its_own_feature_list(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        for task in TASK_ORDER:
            pipe_features = list(self.predictor.tasks[task]["pipeline"].feature_names_in_)
            self.assertEqual(pipe_features, config["tasks"][task]["input_features"], task)
        # The selected models genuinely differ in their inputs.
        self.assertEqual(len(self.predictor.features_for("Cath")), 48)
        self.assertEqual(len(self.predictor.features_for("LCX")), 7)

    def test_schema_is_union_of_model_inputs(self):
        union = set()
        for task in TASK_ORDER:
            union |= set(self.predictor.features_for(task))
        self.assertEqual(union, set(self.schema))

    # ------------------------------------------------------------------ leakage
    def test_no_target_column_is_a_model_input(self):
        for task in TASK_ORDER:
            for target in TARGET_COLUMNS:
                self.assertNotIn(target, self.predictor.features_for(task), task)
        for target in TARGET_COLUMNS:
            self.assertNotIn(target, self.schema)

    def test_target_column_in_input_is_rejected(self):
        for target in TARGET_COLUMNS:
            bad = dict(self.example, **{target: "Normal"})
            with self.assertRaises(InputValidationError) as ctx:
                self.predictor.predict(bad)
            self.assertIn("prediction target", str(ctx.exception))

    # ------------------------------------------------------------------ valid input
    def test_valid_synthetic_example_output_structure(self):
        result = self.predictor.predict(self.example)
        for key in ("disclaimer", "threshold", "threshold_note", "predictions",
                    "agreement", "warnings", "limitations"):
            self.assertIn(key, result)
        self.assertEqual(result["threshold"], 0.5)
        self.assertIn("not a diagnostic tool", result["disclaimer"].lower())
        self.assertEqual(list(result["predictions"]), list(TASK_ORDER))
        for task, out in result["predictions"].items():
            p = out["probability"]
            self.assertIsInstance(p, float)
            self.assertTrue(0.0 <= p <= 1.0 and math.isfinite(p), task)
            self.assertEqual(out["predicted_positive"], p >= 0.5)
            expected_label = out["positive_label"] if p >= 0.5 else out["negative_label"]
            self.assertEqual(out["predicted_label"], expected_label)

    def test_probability_equals_direct_pipeline_call(self):
        """The layer must return exactly what the saved pipeline returns (column 1)."""
        result = self.predictor.predict(self.example)
        for task in TASK_ORDER:
            info = self.predictor.tasks[task]
            X = pd.DataFrame([{f: self.example[f] for f in info["features"]}])
            direct = float(info["pipeline"].predict_proba(X)[0, 1])
            self.assertAlmostEqual(result["predictions"][task]["probability"], direct, places=12)

    def test_threshold_changes_only_binary_output(self):
        """Probabilities never depend on the threshold; the 0/1 output follows p >= threshold."""
        base = self.predictor.predict(self.example)
        for threshold in (0.01, 0.3, 0.7, 0.99):
            result = self.predictor.predict(self.example, threshold=threshold)
            self.assertEqual(result["threshold"], threshold)
            for task in TASK_ORDER:
                p = result["predictions"][task]["probability"]
                self.assertAlmostEqual(
                p,
                base["predictions"][task]["probability"],
                places=12,
            )
                self.assertEqual(result["predictions"][task]["predicted_positive"], p >= threshold)

    def test_invalid_threshold_rejected(self):
        for bad in (0, 1, -0.2, 1.5, "0.5", None, True):
            with self.assertRaises(ValueError):
                self.predictor.predict(self.example, threshold=bad)

    def test_agreement_reports_disagreement_without_changing_outputs(self):
        result = self.predictor.predict(self.example)
        agree = result["agreement"]
        cad = result["predictions"]["Cath"]["predicted_positive"]
        vessels = [t for t in ("LAD", "LCX", "RCA") if result["predictions"][t]["predicted_positive"]]
        self.assertEqual(agree["cad_predicted_positive"], cad)
        self.assertEqual(agree["vessels_predicted_positive"], vessels)
        self.assertEqual(agree["models_disagree"], (cad and not vessels) or (not cad and bool(vessels)))

    def test_input_dict_is_not_modified(self):
        original = copy.deepcopy(self.example)
        self.predictor.predict(self.example)
        self.assertEqual(self.example, original)

    def test_every_training_row_passes_validation(self):
        """The schema must accept the real data the models were trained on (read-only)."""
        df = pd.read_excel(DATA_FILE)
        for i, row in df.iterrows():
            patient = {f: row[f].item() if hasattr(row[f], "item") else row[f] for f in self.schema}
            clean, notes = self.predictor.validate_input(patient)
            self.assertEqual(notes, [], f"row {i}")

    # ------------------------------------------------------------------ invalid input
    def assert_invalid(self, patient, *fragments):
        with self.assertRaises(InputValidationError) as ctx:
            self.predictor.predict(patient)
        text = str(ctx.exception)
        for fragment in fragments:
            self.assertIn(fragment, text)
        return ctx.exception

    def test_non_dict_rejected(self):
        self.assert_invalid([1, 2, 3], "must be a dict")

    def test_missing_required_field(self):
        bad = dict(self.example)
        del bad["Age"]
        self.assert_invalid(bad, "Missing required field", "Age")

    def test_all_fields_missing_lists_them(self):
        error = self.assert_invalid({}, "Missing required field")
        self.assertIn("EF-TTE", str(error))

    def test_none_and_nan_rejected(self):
        for value in (None, float("nan")):
            self.assert_invalid(dict(self.example, Age=value), "'Age' is missing")
            self.assert_invalid(dict(self.example, Sex=value), "'Sex' is missing")

    def test_unknown_categorical_value_rejected(self):
        self.assert_invalid(dict(self.example, Sex="Female"), "'Sex' must be one of")
        self.assert_invalid(dict(self.example, VHD="severe"), "Did you mean 'Severe'")
        self.assert_invalid(dict(self.example, DLP="y"), "Did you mean 'Y'")
        self.assert_invalid(dict(self.example, BBB="none"), "'BBB' must be one of")

    def test_binary_codes_must_be_numbers(self):
        self.assert_invalid(dict(self.example, DM="1"), "'DM' must be one of [0, 1]")
        self.assert_invalid(dict(self.example, DM=2), "'DM' must be one of [0, 1]")
        self.assert_invalid(dict(self.example, DM=True), "'DM' must be one of [0, 1]")

    def test_numeric_type_errors(self):
        self.assert_invalid(dict(self.example, Age="45"), "'Age' must be a number")
        self.assert_invalid(dict(self.example, Age=True), "'Age' must be a number")
        self.assert_invalid(dict(self.example, Age=45.5), "whole number")
        self.assert_invalid(dict(self.example, BMI=float("inf")), "finite")

    def test_unknown_field_rejected_with_hint(self):
        bad = dict(self.example)
        bad["age"] = bad.pop("Age")
        self.assert_invalid(bad, "Unknown field 'age'", "Did you mean 'Age'")

    def test_out_of_range_rejected_by_default_and_warned_when_allowed(self):
        bad = dict(self.example, Age=120)
        self.assert_invalid(bad, "outside the training range")
        result = self.predictor.predict(bad, allow_out_of_range=True)
        self.assertTrue(any("Age" in w for w in result["warnings"]))

    def test_all_errors_reported_together(self):
        bad = dict(self.example, Age="old", Sex="X", DM=5)
        error = self.assert_invalid(bad)
        self.assertGreaterEqual(len(error.errors), 3)

    # ------------------------------------------------------------------ artifact safety
    def test_changed_model_file_is_detected(self):
        """A config whose recorded hash does not match the file must refuse to load."""
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["tasks"]["RCA"]["artifact_sha256"] = "0" * 64
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "model_selection.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ArtifactError):
                CADPredictor(config_path=path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
