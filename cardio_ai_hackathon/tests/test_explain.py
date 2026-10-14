"""
Tests for src/cad/explain.py (built-in unittest; no pytest needed).

Run from the project folder:
    python -m unittest discover -s tests -v

The examples are synthetic or taken from the training file only to check software
behaviour. No test asserts that any feature or value means higher or lower risk.
"""

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from cad.explain import LR_METHOD, RF_METHOD, CADExplainer, encoded_to_input_map  # noqa: E402
from cad.inference import (  # noqa: E402
    TARGET_COLUMNS,
    TASK_ORDER,
    CADPredictor,
    InputValidationError,
    synthetic_example,
)

LR_TASKS = ("Cath", "LCX", "RCA")
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"


def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


class ExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CADPredictor()
        cls.explainer = CADExplainer(cls.predictor)
        cls.example = synthetic_example(cls.predictor.schema)
        cls.result = cls.explainer.explain(cls.example)

    # ------------------------------------------------------------------ structure
    def test_all_four_explanations_generated(self):
        self.assertEqual(list(self.result["explanations"]), list(TASK_ORDER))
        self.assertIn("not a diagnostic tool", self.result["disclaimer"].lower())
        for task, e in self.result["explanations"].items():
            for key in ("task", "model_name", "model_type", "probability", "method",
                        "units", "additive", "contributions", "limitations"):
                self.assertIn(key, e, f"{task} missing {key}")
            self.assertTrue(e["limitations"], task)

    def test_every_model_feature_explained_once(self):
        for task, e in self.result["explanations"].items():
            names = [c["feature"] for c in e["contributions"]]
            self.assertEqual(sorted(names), sorted(self.predictor.features_for(task)), task)
            self.assertEqual(len(names), len(set(names)), task)
            expected_unused = [f for f in self.predictor.schema
                               if f not in self.predictor.features_for(task)]
            self.assertEqual(e["features_not_used_by_model"], expected_unused, task)

    def test_contributions_ranked_by_absolute_size(self):
        for task, e in self.result["explanations"].items():
            sizes = [abs(c["contribution"]) for c in e["contributions"]]
            self.assertEqual(sizes, sorted(sizes, reverse=True), task)

    def test_direction_matches_sign(self):
        for task, e in self.result["explanations"].items():
            for c in e["contributions"]:
                if c["direction"] == "raises model output":
                    self.assertGreater(c["contribution"], 0)
                elif c["direction"] == "lowers model output":
                    self.assertLess(c["contribution"], 0)
                else:
                    self.assertEqual(c["direction"], "no measurable effect")
                    self.assertLess(abs(c["contribution"]), 1e-6)

    def test_values_are_the_patient_inputs(self):
        for e in self.result["explanations"].values():
            for c in e["contributions"]:
                self.assertEqual(c["value"], self.example[c["feature"]])

    def test_top_k_and_task_selection(self):
        r = self.explainer.explain(self.example, tasks=["LCX"], top_k=3)
        self.assertEqual(list(r["explanations"]), ["LCX"])
        e = r["explanations"]["LCX"]
        self.assertEqual(len(e["contributions"]), 3)
        self.assertEqual(e["n_features_omitted"], len(self.predictor.features_for("LCX")) - 3)
        with self.assertRaises(ValueError):
            self.explainer.explain(self.example, tasks=["Heart"])
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                self.explainer.explain(self.example, top_k=bad)

    # ------------------------------------------------------------------ agreement
    def test_probabilities_match_inference_layer(self):
        prediction = self.predictor.predict(self.example)
        for task in TASK_ORDER:
            self.assertAlmostEqual(self.result["explanations"][task]["probability"],
                                   prediction["predictions"][task]["probability"], places=12)

    # ------------------------------------------------------------------ logistic regression
    def test_logistic_contributions_reconstruct_log_odds(self):
        for task in LR_TASKS:
            e = self.result["explanations"][task]
            self.assertEqual(e["method"], LR_METHOD)
            self.assertTrue(e["additive"])
            self.assertEqual(e["units"], "log-odds")
            total = sum(c["contribution"] for c in e["contributions"])
            self.assertAlmostEqual(e["reference_log_odds"] + total, e["model_log_odds"], places=9)
            self.assertAlmostEqual(sigmoid(e["model_log_odds"]), e["probability"], places=9)
            pipe = self.predictor.tasks[task]["pipeline"]
            X = pd.DataFrame([{f: self.example[f] for f in self.predictor.features_for(task)}])
            self.assertAlmostEqual(float(pipe.decision_function(X)[0]), e["model_log_odds"], places=9)

    def test_logistic_reconstruction_on_real_training_rows(self):
        df = pd.read_excel(DATA_FILE)
        for i in range(0, len(df), 25):
            patient = {f: (df.loc[i, f].item() if hasattr(df.loc[i, f], "item") else df.loc[i, f])
                       for f in self.predictor.schema}
            r = self.explainer.explain(patient, tasks=list(LR_TASKS))
            for task, e in r["explanations"].items():
                total = sum(c["contribution"] for c in e["contributions"])
                self.assertAlmostEqual(e["reference_log_odds"] + total, e["model_log_odds"],
                                       places=9, msg=f"row {i} {task}")

    def test_encoded_mapping_covers_every_model_column(self):
        for task in LR_TASKS:
            pipe = self.predictor.tasks[task]["pipeline"]
            owners = encoded_to_input_map(pipe.named_steps["preprocess"])
            self.assertEqual(len(owners), pipe.named_steps["model"].coef_.shape[1], task)
            self.assertEqual(set(owners), set(self.predictor.features_for(task)), task)

    # ------------------------------------------------------------------ random forest
    def test_lad_method_is_declared_approximate(self):
        e = self.result["explanations"]["LAD"]
        self.assertEqual(e["model_type"], "RandomForestClassifier")
        self.assertEqual(e["method"], RF_METHOD)
        self.assertFalse(e["additive"])
        self.assertIn("probability", e["units"])
        self.assertIn("non_additive_gap", e)
        self.assertNotIn("reference_log_odds", e)
        self.assertTrue(any("approximate" in x.lower() for x in e["limitations"]))
        gap = (e["probability"] - e["reference_probability"]) - e["sum_of_contributions"]
        self.assertAlmostEqual(gap, e["non_additive_gap"], places=9)
        for c in e["contributions"]:
            self.assertLessEqual(abs(c["contribution"]), 1.0)

    # ------------------------------------------------------------------ determinism
    def test_repeated_explanations_are_deterministic(self):
        again = self.explainer.explain(self.example)
        for task in TASK_ORDER:
            first = self.result["explanations"][task]["contributions"]
            second = again["explanations"][task]["contributions"]
            self.assertEqual([c["feature"] for c in first], [c["feature"] for c in second], task)
            for a, b in zip(first, second):
                # Random-forest predictions may differ in the last bits between runs
                # (parallel summation), so compare to a tolerance.
                self.assertAlmostEqual(a["contribution"], b["contribution"], places=9)

    # ------------------------------------------------------------------ validation and leakage
    def test_invalid_inputs_rejected_by_existing_validator(self):
        bad_cases = [
            {k: v for k, v in self.example.items() if k != "Age"},   # missing field
            dict(self.example, Age=None),                            # missing value
            dict(self.example, VHD="severe"),                        # unknown category
            dict(self.example, LAD="Stenotic"),                      # target as input
        ]
        for bad in bad_cases:
            with self.assertRaises(InputValidationError):
                self.explainer.explain(bad)

    def test_no_target_column_used(self):
        for target in TARGET_COLUMNS:
            self.assertNotIn(target, self.explainer.background.columns)
        for e in self.result["explanations"].values():
            for c in e["contributions"]:
                self.assertNotIn(c["feature"], TARGET_COLUMNS)

    def test_input_not_modified(self):
        before = dict(self.example)
        self.explainer.explain(self.example)
        self.assertEqual(self.example, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
