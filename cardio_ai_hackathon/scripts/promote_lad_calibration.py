"""
Promote the sigmoid-calibrated LAD Random Forest as a separate deployment artifact.

Run from the project root with the project's .venv active:
    python scripts\promote_lad_calibration.py

Safety:
- Never overwrites the original LAD RF artifact.
- Creates a one-time backup of config/model_selection.json.
- Verifies dataset hash and target leakage before training.
- Fits the calibration wrapper using the same 5-fold shuffled inner CV used in the
  calibration experiment; this is a deployment refit, NOT a new performance evaluation.
- Atomically updates the config only after the new artifact passes verification.
- Keeps original cv_metrics intact and stores the calibrated OOF estimates separately.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "model_selection.json"
BACKUP = ROOT / "config" / "model_selection.pre_lad_calibration.json"
ARTIFACT = ROOT / "models" / "deployment" / "LAD__rf_reduced_sigmoid.joblib"
DATA_DIR = ROOT / "data" / "raw"
FORBIDDEN = {"Cath", "LAD", "LCX", "RCA"}
EXPECTED_BASE_HASH = "04ee2c01869a50163ec3ae14b140915ddf6ccb321cd1a524f7709be630e042b9"
CALIBRATION_OOF_ESTIMATES = {
    "evaluation_type": "repeat-averaged outer-CV OOF estimate; internal only",
    "roc_auc": 0.845,
    "brier": 0.154,
    "log_loss": 0.476,
    "calibration_slope": 1.001,
    "calibration_intercept": -0.002,
    "sensitivity_at_0_5": 0.862,
    "specificity_at_0_5": 0.675,
    "precision_at_0_5": 0.788,
    "f1_at_0_5": 0.824,
    "accuracy_at_0_5": 0.784,
    "note": (
        "Rounded values copied from reports/lad_calibration. These are cross-validation "
        "estimates, not scores on the final all-data deployment artifact. No external or "
        "clinical validation. AUC is essentially unchanged; calibration and Brier/log-loss "
        "improved, with sensitivity lower and specificity higher at threshold 0.5."
    ),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

def main() -> None:
    if not CONFIG.is_file():
        raise SystemExit(f"Missing config: {CONFIG}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = config["tasks"]["LAD"]
    data_path = DATA_DIR / config["data_file"]
    original_artifact = ROOT / spec["artifact"]

    if sha256(data_path) != config["data_sha256"]:
        raise SystemExit("STOP: dataset SHA-256 differs from config; nothing changed.")
    if sha256(original_artifact) != EXPECTED_BASE_HASH or spec["artifact_sha256"] != EXPECTED_BASE_HASH:
        raise SystemExit("STOP: original LAD artifact hash differs from the expected frozen baseline.")
    if list(spec["input_features"]) != list(joblib.load(original_artifact).feature_names_in_):
        raise SystemExit("STOP: config features do not match original LAD artifact.")
    leaked = FORBIDDEN.intersection(spec["input_features"])
    if leaked:
        raise SystemExit(f"STOP: forbidden target columns among model inputs: {sorted(leaked)}")

    # Backup the exact pre-promotion config once; never overwrite an existing backup.
    if not BACKUP.exists():
        shutil.copy2(CONFIG, BACKUP)

    sys.path.insert(0, str(ROOT / "scripts"))
    import train_phase1 as tp

    metadata_path = ROOT / "models" / "phase1_candidates" / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing feature metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    df = pd.read_excel(data_path, sheet_name=0)
    features = list(spec["input_features"])
    X = df[features].copy()
    raw_y = df["LAD"].astype(str).str.strip()
    if sorted(raw_y.unique()) != sorted([spec["positive_label"], spec["negative_label"]]):
        raise SystemExit(f"Unexpected LAD target labels: {sorted(raw_y.unique())}")
    y = (raw_y == spec["positive_label"]).astype(int).to_numpy()
    numeric_set = set(metadata["numeric_features"])
    numeric = [c for c in features if c in numeric_set]
    categorical = [c for c in features if c not in numeric_set]

    baseline = joblib.load(original_artifact)
    rf = clone(baseline.named_steps["model"])
    base_pipeline = tp.make_pipeline(rf, numeric, categorical)
    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        ensemble=False,
    )
    calibrated.fit(X, y)

    # Verify before writing the artifact.
    if [int(c) for c in calibrated.classes_] != [0, 1]:
        raise SystemExit(f"STOP: unexpected calibrated classes {calibrated.classes_}.")
    if list(calibrated.feature_names_in_) != features:
        raise SystemExit("STOP: calibrated model feature schema differs from config.")
    p = calibrated.predict_proba(X.iloc[:3])
    if p.shape != (3, 2) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise SystemExit("STOP: calibrated artifact failed probability smoke test.")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    if ARTIFACT.exists():
        raise SystemExit(f"STOP: refusing to overwrite existing new artifact: {ARTIFACT}")
    joblib.dump(calibrated, ARTIFACT)
    artifact_hash = sha256(ARTIFACT)

    # Round-trip load from disk to ensure the serialized artifact behaves.
    loaded = joblib.load(ARTIFACT)
    if [int(c) for c in loaded.classes_] != [0, 1]:
        raise SystemExit("STOP: round-trip artifact classes check failed.")
    if list(loaded.feature_names_in_) != features:
        raise SystemExit("STOP: round-trip artifact feature check failed.")
    p2 = loaded.predict_proba(X.iloc[:3])
    if not np.allclose(p, p2, rtol=1e-12, atol=1e-12):
        raise SystemExit("STOP: round-trip predictions differ.")

    # Keep the baseline cv_metrics untouched. Add a clearly separate calibrated OOF record.
    spec["baseline_artifact"] = spec["artifact"]
    spec["baseline_artifact_sha256"] = spec["artifact_sha256"]
    spec["baseline_model_name"] = spec["model_name"]
    spec["baseline_model_type"] = spec["model_type"]
    spec["calibration"] = {
        "class": "CalibratedClassifierCV",
        "method": "sigmoid",
        "ensemble": False,
        "inner_cv": {"class": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "deployment_fit": "refit on all 303 rows; not a performance evaluation",
        "oof_metrics": CALIBRATION_OOF_ESTIMATES,
        "calibration_report": "reports/lad_calibration/calibration_report.md",
    }
    spec["model_name"] = "rf_reduced_sigmoid"
    # Preserve the base-model label used by the existing explanation path/tests.
    spec["model_type"] = "RandomForestClassifier"
    spec["artifact"] = ARTIFACT.relative_to(ROOT).as_posix()
    spec["artifact_sha256"] = artifact_hash
    spec["n_input_features"] = len(features)
    config["status"] = "provisional_with_calibrated_lad"
    config["deployment_update"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/promote_lad_calibration.py",
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "note": (
            "LAD points to a separately saved sigmoid-calibrated RF. Original baseline artifact "
            "and baseline cv_metrics are preserved; calibrated OOF estimates are in tasks.LAD.calibration."
        ),
    }
    atomic_json(CONFIG, config)

    # Verify the config and artifact hash the same way inference will.
    written = json.loads(CONFIG.read_text(encoding="utf-8"))
    lad = written["tasks"]["LAD"]
    if sha256(ROOT / lad["artifact"]) != lad["artifact_sha256"]:
        raise SystemExit("CONFIG WRITTEN BUT HASH VERIFICATION FAILED. Restore backup before continuing.")
    if sha256(original_artifact) != EXPECTED_BASE_HASH:
        raise SystemExit("Original LAD baseline unexpectedly changed.")
    print("PROMOTION SUCCEEDED")
    print(f"New artifact: {ARTIFACT.relative_to(ROOT)}")
    print(f"New SHA-256: {artifact_hash}")
    print(f"Config backup: {BACKUP.relative_to(ROOT)}")
    print(f"Original LAD artifact preserved: {original_artifact.relative_to(ROOT)}")
    print("No claim of improved AUC: repeat-averaged OOF AUC 0.847 -> 0.845.")
    print("Calibration OOF Brier 0.164 -> 0.154; log-loss 0.504 -> 0.476.")
    print("Threshold 0.5 trade-off: sensitivity 0.886 -> 0.862; specificity 0.643 -> 0.675.")
    print("Next: python -m unittest discover -s tests -v")
    print("Then: python scripts\\check_inference.py")
    print("Then: python scripts\\check_explain.py")

if __name__ == "__main__":
    main()
