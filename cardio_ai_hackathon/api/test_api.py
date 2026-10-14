"""
Tests for the model service.  Run from the project folder:

    python -m unittest api.test_api -v

They load the real models (same hash checks as src/cad) and use the example patients
exported from the AtheroVision frontend (api/frontend_examples.json), so they check exactly
what the browser sends and receives.
"""

import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from cad.inference import CADPredictor  # noqa: E402
from api.adapter import AdapterError, to_model_input  # noqa: E402
from api.main import app, predictor  # noqa: E402

EXAMPLES = json.loads((HERE / "frontend_examples.json").read_text(encoding="utf-8"))
client = TestClient(app)


def sigmoid(v):
    return 1 / (1 + math.exp(-v))


class AdapterTests(unittest.TestCase):
    def test_every_frontend_example_validates_against_the_model_schema(self):
        for name, features in EXAMPLES.items():
            patient, _ = to_model_input(features, predictor.schema)
            clean, _ = predictor.validate_input(patient, allow_out_of_range=True)
            self.assertEqual(set(clean), set(predictor.schema), name)

    def test_encodings(self):
        f = dict(EXAMPLES["typical"], Sex="Female", DLP=1, CRF=0)
        patient, _ = to_model_input(f, predictor.schema)
        self.assertEqual(patient["Sex"], "Fmale")          # dataset spelling
        self.assertEqual(patient["DLP"], "Y")              # Y/N text field
        self.assertEqual(patient["CRF"], "N")
        self.assertEqual(patient["DM"], 0)                 # 0/1 numeric field
        self.assertIsInstance(patient["Age"], int)         # integer field stays whole

    def test_targets_are_rejected_as_inputs(self):
        for t in ("Cath", "LAD", "LCX", "RCA"):
            with self.assertRaises(AdapterError) as ctx:
                to_model_input(dict(EXAMPLES["typical"], **{t: 1}), predictor.schema)
            self.assertTrue(any("prediction target" in e for e in ctx.exception.errors), t)

    def test_unused_inputs_listed_and_derivation_inputs_not_flagged(self):
        patient, unused = to_model_input(EXAMPLES["typical"], predictor.schema)
        self.assertEqual(set(patient), set(predictor.schema))
        self.assertNotIn("Weight", unused)
        self.assertIn("Neut", unused)

    def test_missing_field_is_reported(self):
        f = dict(EXAMPLES["typical"])
        del f["Age"]
        with self.assertRaises(AdapterError) as ctx:
            to_model_input(f, predictor.schema)
        self.assertIn("Age", ctx.exception.errors[0])


class EndpointTests(unittest.TestCase):
    def test_health(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_prediction_matches_the_ml_package_exactly(self):
        features = EXAMPLES["typical"]
        r = client.post("/predict", json={"features": features, "explain": False})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        patient, _ = to_model_input(features, predictor.schema)
        direct = CADPredictor.predict(predictor, patient, allow_out_of_range=True)
        for task, fe in (("Cath", "CAD"), ("LAD", "LAD"), ("LCX", "LCX"), ("RCA", "RCA")):
            self.assertAlmostEqual(body["predictions"][fe]["probability"],
                                   direct["predictions"][task]["probability"], places=12)
        self.assertFalse(body["explained"])
        self.assertNotIn("contributions", body["predictions"]["CAD"])

    def test_explanations_are_consistent_with_the_prediction(self):
        r = client.post("/predict", json={"features": EXAMPLES["multi"], "explain": True})
        self.assertEqual(r.status_code, 200)
        preds = r.json()["predictions"]
        for t in ("CAD", "LCX", "RCA"):          # logistic regression: exact, additive log-odds
            p = preds[t]
            self.assertEqual(p["units"], "log-odds")
            self.assertTrue(p["additive"])
            total = p["base_value"] + sum(c["shap"] for c in p["contributions"])
            self.assertAlmostEqual(sigmoid(total), p["probability"], places=9)
        lad = preds["LAD"]                        # random forest: approximate, probability units
        self.assertEqual(lad["units"], "probability")
        self.assertFalse(lad["additive"])
        self.assertEqual(len(lad["contributions"]), 48)
        self.assertEqual(len(preds["LCX"]["contributions"]), 7)
        self.assertEqual(len(preds["LCX"]["features_not_used"]), 41)
        # values are echoed in the frontend's own encoding
        sex = next(c for c in preds["CAD"]["contributions"] if c["feature"] == "Sex")
        self.assertEqual(sex["value"], "Male")

    def test_out_of_range_is_predicted_with_a_warning(self):
        f = dict(EXAMPLES["typical"], PR=40)
        r = client.post("/predict", json={"features": f, "explain": False})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(any("'PR'" in w for w in r.json()["warnings"]))

    def test_invalid_input_returns_422_with_every_problem(self):
        f = dict(EXAMPLES["typical"], Age=61.5, Sex="Unknown")
        r = client.post("/predict", json={"features": f, "explain": False})
        self.assertEqual(r.status_code, 422)
        errors = r.json()["detail"]["errors"]
        self.assertTrue(any("Age" in e for e in errors))
        self.assertTrue(any("Sex" in e for e in errors))

    def test_target_in_request_returns_422(self):
        r = client.post("/predict", json={"features": dict(EXAMPLES["typical"], Cath="CAD")})
        self.assertEqual(r.status_code, 422)
        self.assertTrue(any("prediction target" in e for e in r.json()["detail"]["errors"]))

    def test_unknown_category_returns_422_with_hint(self):
        r = client.post("/predict", json={"features": dict(EXAMPLES["typical"], BBB="lbbb"), "explain": False})
        self.assertEqual(r.status_code, 422)
        self.assertTrue(any("BBB" in e and "LBBB" in e for e in r.json()["detail"]["errors"]))

    def test_malformed_requests_return_422(self):
        self.assertEqual(client.post("/predict", json={}).status_code, 422)                       # no features
        self.assertEqual(client.post("/predict", json={"features": [1, 2]}).status_code, 422)     # not an object
        r = client.post("/predict", json={"features": dict(EXAMPLES["typical"], Age="old"), "explain": False})
        self.assertEqual(r.status_code, 422)
        self.assertTrue(any("Age" in e for e in r.json()["detail"]["errors"]))

    def test_explain_defaults_to_true(self):
        r = client.post("/predict", json={"features": EXAMPLES["low"]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["explained"])
        self.assertTrue(r.json()["predictions"]["CAD"]["contributions"])

    def test_dataset_row_274_matches_the_package(self):
        features = json.loads((HERE / "dataset_row_274.json").read_text(encoding="utf-8"))
        r = client.post("/predict", json={"features": features, "explain": False})
        self.assertEqual(r.status_code, 200)
        patient, _ = to_model_input(features, predictor.schema)
        direct = predictor.predict(patient, allow_out_of_range=True)["predictions"]
        for task, fe in (("Cath", "CAD"), ("LAD", "LAD"), ("LCX", "LCX"), ("RCA", "RCA")):
            self.assertAlmostEqual(r.json()["predictions"][fe]["probability"], direct[task]["probability"], places=12)

    def test_metrics(self):
        r = client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        m = r.json()
        self.assertEqual(m["n_samples"], 303)
        self.assertEqual(set(m["targets"]), {"CAD", "LAD", "LCX", "RCA"})
        for t in m["targets"].values():
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                self.assertTrue(0 <= t[k] <= 1)
        self.assertEqual(m["feature_ranges"]["Age"], [30, 86])

    def test_dataset_patient_matches_the_exported_row(self):
        r = client.get("/dataset/274")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        expected = json.loads((HERE / "dataset_row_274.json").read_text(encoding="utf-8"))
        self.assertEqual(body["features"], expected)
        # outcomes are returned separately and never inside the inputs
        for t in ("Cath", "LAD", "LCX", "RCA"):
            self.assertNotIn(t, body["features"])
        self.assertEqual(body["actual"], {"CAD": 1, "LAD": 1, "LCX": 0, "RCA": 1})
        for t in ("CAD", "LAD", "LCX", "RCA"):
            cv = body["cv"][t]
            self.assertEqual(cv["repeats"], 5)
            self.assertTrue(cv["min"] <= cv["probability"] <= cv["max"])

    def test_every_dataset_patient_is_accepted_by_predict(self):
        n = client.get("/dataset/0").json()["n_rows"]
        self.assertEqual(n, 303)
        for row in range(n):
            features = client.get(f"/dataset/{row}").json()["features"]
            patient, _ = to_model_input(features, predictor.schema)
            predictor.validate_input(patient)          # strict: inside training ranges, valid categories

    def test_label_conflict_is_flagged_only_for_row_93(self):
        self.assertIn("Label conflict", client.get("/dataset/93").json()["label_conflict"])
        flagged = [r for r in range(303) if client.get(f"/dataset/{r}").json()["label_conflict"]]
        self.assertEqual(flagged, [93])

    def test_operating_points_follow_the_rule_and_match_the_saved_report(self):
        from api.operating_points import DEFAULT_THRESHOLD, GRID, MIN_SENSITIVITY, compute_operating_points
        import csv
        ops = compute_operating_points(HERE.parent)
        self.assertEqual({t: o["threshold"] for t, o in ops.items()}, {"CAD": 0.5, "LAD": 0.5, "LCX": 0.3, "RCA": 0.3})
        for o in ops.values():
            self.assertGreaterEqual(o["cv"]["sensitivity"], MIN_SENSITIVITY)
            self.assertLessEqual(o["threshold"], DEFAULT_THRESHOLD)
        # the next grid step up would fall below the sensitivity floor (highest qualifying cut-off)
        from api.operating_points import _load_oof, metrics_at
        oof = _load_oof(HERE.parent)
        for t in ("LCX", "RCA"):
            nxt = GRID[GRID.index(ops[t]["threshold"]) + 1]
            self.assertLess(metrics_at(oof[t], nxt)["sensitivity"], MIN_SENSITIVITY)
        # independent check against the evaluation report written by the ML pipeline
        path = HERE.parent / "reports/final_ml_evaluation/threshold_tradeoffs.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["summary"] == "average_over_5_repeats"]
        for t in ("CAD", "LCX", "RCA"):
            task = "Cath" if t == "CAD" else t
            r = next(r for r in rows if r["task"] == task and abs(float(r["threshold"]) - ops[t]["threshold"]) < 1e-9)
            self.assertAlmostEqual(float(r["sensitivity"]), ops[t]["cv"]["sensitivity"], places=6)
            self.assertAlmostEqual(float(r["specificity"]), ops[t]["cv"]["specificity"], places=6)

    def test_flags_follow_each_targets_cut_off_and_probabilities_are_unchanged(self):
        body = client.post("/predict", json={"features": EXAMPLES["typical"], "explain": False}).json()
        patient, _ = to_model_input(EXAMPLES["typical"], predictor.schema)
        direct = predictor.predict(patient, allow_out_of_range=True)["predictions"]
        for task, fe in (("Cath", "CAD"), ("LAD", "LAD"), ("LCX", "LCX"), ("RCA", "RCA")):
            p = body["predictions"][fe]
            self.assertEqual(p["probability"], direct[task]["probability"])
            self.assertEqual(p["label"], int(p["probability"] >= p["threshold"]))
            self.assertEqual(p["default_threshold"], 0.5)
            self.assertEqual(p["default_label"], int(direct[task]["predicted_positive"]))
        self.assertEqual(body["predictions"]["LCX"]["threshold"], 0.3)
        self.assertEqual(body["predictions"]["LCX"]["operating_point"]["mode"], "screening")
        m = client.get("/metrics").json()
        self.assertEqual(m["operating_points"]["RCA"]["threshold"], 0.3)

    def test_reality_check_counts_similar_patients(self):
        from api.main import DATASET, REALITY
        from api.reality_check import wilson
        body = client.post("/predict", json={"features": EXAMPLES["typical"], "explain": False}).json()
        for t, p in body["predictions"].items():
            rc = p["reality_check"]
            self.assertEqual(rc["n"], 40)
            self.assertTrue(0 <= rc["positives"] <= 40)
            self.assertTrue(rc["ci_low"] <= rc["rate"] <= rc["ci_high"])
            self.assertLessEqual(rc["score_min"], rc["score_max"])
        # brute-force recount for one target
        p = body["predictions"]["LCX"]["probability"]
        rows = [DATASET.get(r) for r in range(len(DATASET))]
        pts = sorted(((d["cv"]["LCX"]["probability"], d["actual"]["LCX"]) for d in rows), key=lambda sp: (abs(sp[0] - p), sp[0]))[:40]
        self.assertEqual(body["predictions"]["LCX"]["reality_check"]["positives"], sum(y for _, y in pts))
        # Wilson interval, 12/40, worked by hand: centre 0.31753, half-width 0.13678
        lo, hi = wilson(12, 40)
        self.assertAlmostEqual(lo, 0.18075, places=4)
        self.assertAlmostEqual(hi, 0.45430, places=4)
        self.assertIsNotNone(REALITY.check("CAD", 0.5))

    def test_metrics_include_calibration_from_the_reports(self):
        cal = client.get("/metrics").json()["calibration"]
        self.assertEqual(set(cal), {"CAD", "LAD", "LCX", "RCA"})
        for t, c in cal.items():
            self.assertEqual(len(c["bins"]), 10, t)
            self.assertEqual(sum(b["n"] for b in c["bins"]), 303, t)
            for b in c["bins"]:
                self.assertTrue(b["ci_low"] <= b["observed"] <= b["ci_high"])
            self.assertIn("brier", c)
            self.assertIn("calibration_slope", c)
        self.assertAlmostEqual(cal["LAD"]["calibration_slope"]["estimate"], 1.001, places=2)   # deployed sigmoid model
        self.assertAlmostEqual(cal["CAD"]["brier"]["estimate"], 0.103, places=3)

    def test_lad_metrics_are_those_of_the_deployed_calibrated_model(self):
        m = client.get("/metrics").json()
        lad = m["targets"]["LAD"]
        self.assertEqual(lad["recall"], 0.862)          # calibrated forest, not the baseline 0.886
        self.assertEqual(lad["roc_auc"], 0.845)
        self.assertNotIn("recall", m["sd"]["LAD"])
        self.assertAlmostEqual(lad["recall"], m["operating_points"]["LAD"]["cv"]["sensitivity"], places=3)

    def test_reality_check_leaves_a_dataset_patient_out_of_their_own_comparison(self):
        from api.main import DATASET
        d = DATASET.get(274)
        with_self = client.post("/predict", json={"features": d["features"], "explain": False}).json()
        without = client.post("/predict", json={"features": d["features"], "explain": False, "dataset_row": 274}).json()
        for t in ("CAD", "LAD", "LCX", "RCA"):
            self.assertIsNone(with_self["predictions"][t]["reality_check"]["excluded_row"])
            self.assertEqual(without["predictions"][t]["reality_check"]["excluded_row"], 274)
            self.assertEqual(without["predictions"][t]["reality_check"]["n"], 40)
        # brute-force: the 40 nearest among the other 302 patients
        p = without["predictions"]["LCX"]["probability"]
        others = [DATASET.get(r) for r in range(len(DATASET)) if r != 274]
        pts = sorted(((o["cv"]["LCX"]["probability"], o["actual"]["LCX"]) for o in others), key=lambda sp: (abs(sp[0] - p), sp[0]))[:40]
        self.assertEqual(without["predictions"]["LCX"]["reality_check"]["positives"], sum(y for _, y in pts))

    def test_dataset_row_out_of_range_is_404(self):
        self.assertEqual(client.get("/dataset/303").status_code, 404)
        self.assertEqual(client.get("/dataset/-1").status_code, 404)

    def test_cors_allows_the_frontend(self):
        r = client.options("/predict", headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "POST"})
        self.assertEqual(r.headers.get("access-control-allow-origin"), "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
