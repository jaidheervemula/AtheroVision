"""
check_inference.py
==================
Command-line smoke test for the inference layer. It:
  1. loads the four selected pipelines (with hash and schema checks),
  2. prints each model's input features,
  3. runs ONE prediction on a SYNTHETIC input built mechanically from the schema,
  4. shows that an invalid input is rejected with a clear error.

The synthetic input is NOT a real patient and its outputs have no clinical meaning.

Run from the project folder:
    python scripts\\check_inference.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cad.inference import CADPredictor, InputValidationError, synthetic_example  # noqa: E402


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 72)
    print("INFERENCE SMOKE TEST (educational prototype, not a diagnostic tool)")
    print("=" * 72)

    predictor = CADPredictor()
    print("\n1. Loaded models (file hashes, classes and input features verified):")
    for task, info in predictor.tasks.items():
        print(f"   {task:4s} {info['model_name']:16s} {info['model_type']:22s} "
              f"{len(info['features']):2d} inputs  positive label = {info['positive_label']!r}")
    print(f"   Input form needs {len(predictor.required_fields())} fields (union of all four models).")

    print("\n2. Inputs used by the 7-feature models (LCX, RCA):")
    print("   " + ", ".join(predictor.features_for("LCX")))

    example = synthetic_example(predictor.schema)
    print("\n3. Prediction on a SYNTHETIC input (midpoints / first allowed values; NOT a patient):")
    result = predictor.predict(example)
    for task, out in result["predictions"].items():
        print(f"   {out['display_name']:14s} model score {out['probability']:.3f}  -> "
              f"{out['predicted_label']} at threshold {result['threshold']}")
    print(f"   Agreement: {result['agreement']['note']}")
    print(f"   Note: {result['threshold_note']}")

    print("\n4. Invalid input is rejected (expected):")
    bad = dict(example, Sex="Female", Age=None)
    try:
        predictor.predict(bad)
        print("   PROBLEM: invalid input was accepted.")
        sys.exit(1)
    except InputValidationError as error:
        for line in error.errors:
            print(f"   - {line}")

    print("\nDisclaimer:", result["disclaimer"])
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
