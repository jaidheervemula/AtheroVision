"""
check_explain.py
================
Verification script for the explanation layer. It loads the real selected models,
builds a SYNTHETIC input from the actual schema, explains all four predictions and
checks the numerical properties of each explanation.

All outputs below are ARTIFICIAL SOFTWARE-TEST EXAMPLES, not patient predictions.
Contributions describe the model's behaviour, not causes or medical reasons.

Run from the project folder:
    python scripts\\check_explain.py
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cad.explain import CADExplainer  # noqa: E402
from cad.inference import CADPredictor, synthetic_example  # noqa: E402


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    problems = []

    print("=" * 76)
    print("EXPLANATION CHECK - SYNTHETIC SOFTWARE-TEST INPUT, NOT A PATIENT")
    print("=" * 76)
    predictor = CADPredictor()
    explainer = CADExplainer(predictor)
    example = synthetic_example(predictor.schema)
    result = explainer.explain(example)
    prediction = predictor.predict(example)

    for task, e in result["explanations"].items():
        print(f"\n[{task}] {e['display_name']} | {e['model_name']} ({e['model_type']})")
        print(f"  Method: {e['method']}  | units: {e['units']}  | additive: {e['additive']}")
        print(f"  Model score (positive = {e['positive_label']!r}): {e['probability']:.3f}  "
              f"[artificial example]")
        if e["additive"]:
            err = abs(e["reference_log_odds"] + e["sum_of_contributions"] - e["model_log_odds"])
            print(f"  Reference log-odds {e['reference_log_odds']:+.3f} + contributions "
                  f"{e['sum_of_contributions']:+.3f} = model log-odds {e['model_log_odds']:+.3f} "
                  f"(error {err:.1e})")
            if err > 1e-9:
                problems.append(f"{task}: log-odds reconstruction error {err}")
            if abs(1 / (1 + math.exp(-e["model_log_odds"])) - e["probability"]) > 1e-9:
                problems.append(f"{task}: log-odds do not match the probability")
        else:
            print(f"  Average model output on training data {e['reference_probability']:.3f}; "
                  f"impacts sum to {e['sum_of_contributions']:+.3f}; non-additive gap "
                  f"{e['non_additive_gap']:+.3f} (approximation, see limitations)")
        if abs(e["probability"] - prediction["predictions"][task]["probability"]) > 1e-12:
            problems.append(f"{task}: probability differs from the inference layer")
        sizes = [abs(c["contribution"]) for c in e["contributions"]]
        if sizes != sorted(sizes, reverse=True):
            problems.append(f"{task}: contributions are not sorted")
        if len(e["contributions"]) != len(predictor.features_for(task)):
            problems.append(f"{task}: not every model input was explained")
        print(f"  Top 5 of {len(e['contributions'])} inputs ({e['units']}):")
        for c in e["contributions"][:5]:
            print(f"    {c['feature']:22s} value={c['value']!s:9s} {c['contribution']:+.4f}  "
                  f"{c['direction']}")
        if e["features_not_used_by_model"]:
            print(f"  {len(e['features_not_used_by_model'])} form inputs are not used by this model.")

    print("\n" + result["interpretation_note"])
    print("Disclaimer:", result["disclaimer"])
    if problems:
        print("\nCHECK FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nEXPLANATION CHECK PASSED")


if __name__ == "__main__":
    main()
