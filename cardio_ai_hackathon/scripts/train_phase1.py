"""
train_phase1.py
===============
Phase 1 modeling for the Track A CAD project: baselines + classical models,
evaluated with repeated stratified cross-validation, for four separate tasks:

    Cath (overall CAD), LAD, LCX, RCA stenosis

Key safety rules built into this script
  * The original Excel file is only READ (its SHA-256 hash is checked before/after).
  * ALL FOUR target columns are removed from the inputs of EVERY model.
  * All learned preprocessing (imputation, one-hot encoding, scaling) lives inside a
    scikit-learn Pipeline, so it is re-fitted on the training part of every fold.
  * No hyperparameter tuning: every model uses fixed, pre-declared settings, so the
    cross-validation scores are not inflated by tuning on the same folds.
  * The inconsistent target row is KEPT in the primary analysis and examined in a
    separate sensitivity analysis (results with and without it).

Outputs
  reports/phase1_modeling/   metrics tables, predictions, log
  models/phase1_candidates/  one fitted pipeline per task x model (+ metadata.json)

Run from the project folder with the virtual environment active:
    python scripts\\train_phase1.py
"""

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------- settings
SEED = 42                 # one seed for CV splits and random forests -> reproducible
N_SPLITS = 5              # 5 folds: each test fold keeps >= ~17 minority-class patients
N_REPEATS = 5             # 5 different shuffles -> 25 scores per model, shows variation
THRESHOLD = 0.5           # fixed for now; threshold choice is a later, separate step
RARE_MIN_COUNT = 5        # binary features with < 5 patients in either level are excluded

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # works from any folder
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase1_modeling"
MODEL_DIR = PROJECT_ROOT / "models" / "phase1_candidates"

TARGETS = ["Cath", "LAD", "LCX", "RCA"]
POSITIVE_LABELS = {"cad", "stenotic"}     # positive class = disease present
NEGATIVE_LABELS = {"normal"}

# Simple clinical baseline: chosen BEFORE seeing any model results. These are the
# classic pre-test inputs (age, sex, chest-pain type) plus standard risk factors.
CLINICAL_BASELINE_FEATURES = ["Age", "Sex", "Typical Chest Pain", "DM", "HTN", "DLP",
                              "Current Smoker"]

# Redundancy check: BMI is computed from Weight and Length, Obesity is a BMI-based
# flag, and Neut is strongly (negatively) correlated with Lymph (rho ~ -0.91).
# The "reduced" feature set drops these four; we only adopt it if evidence supports it.
REDUNDANCY_DROP = ["Weight", "Length", "Obesity", "Neut"]

METRICS = ["roc_auc", "pr_auc", "brier", "accuracy", "balanced_accuracy",
           "precision", "recall", "specificity", "f1"]


# ----------------------------------------------------------------------------- helpers
class Report:
    """Prints to the console and also saves the same text to a log file."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, text=""):
        print(text)
        self.f.write(str(text) + "\n")
        self.f.flush()

    def section(self, title):
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def close(self):
        self.f.close()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(name):
    return " ".join(str(name).strip().lower().split())


def find_col(df, name):
    for c in df.columns:
        if norm(c) == norm(name):
            return c
    return None


def make_pipeline(estimator, numeric_cols, categorical_cols):
    """Preprocessing + model in ONE object, so every fold re-fits the preprocessing."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # no missing values today, but
        ("scale", StandardScaler()),                    # keeps inference robust later
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Y/N and two-level columns become ONE 0/1 column; multi-level ones become
        # several 0/1 columns. Unseen categories at prediction time become all zeros.
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                 sparse_output=False)),
    ])
    parts = []
    if numeric_cols:
        parts.append(("num", numeric, list(numeric_cols)))
    if categorical_cols:
        parts.append(("cat", categorical, list(categorical_cols)))
    pre = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("preprocess", pre), ("model", clone(estimator))])


def compute_metrics(y_true, proba):
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run_cv(X, y, feature_cols, numeric_set, estimator, splits):
    """Cross-validate one model on one task. Returns fold metrics and out-of-fold probabilities."""
    num_cols = [c for c in feature_cols if c in numeric_set]
    cat_cols = [c for c in feature_cols if c not in numeric_set]
    oof = np.full((len(y), N_REPEATS), np.nan)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        pipe = make_pipeline(estimator, num_cols, cat_cols)
        pipe.fit(X.iloc[train_idx][feature_cols], y[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx][feature_cols])[:, 1]
        repeat = fold_id // N_SPLITS
        oof[test_idx, repeat] = proba
        m = compute_metrics(y[test_idx], proba)
        m.update({"fold": fold_id, "repeat": repeat, "n_test": len(test_idx),
                  "n_test_positive": int(y[test_idx].sum())})
        rows.append(m)
    return rows, oof


# ----------------------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)

    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features_only = "--features-only" in sys.argv   # check encodings without training
    log_name = "feature_check_output.txt" if features_only else "phase1_output.txt"
    R = Report(REPORT_DIR / log_name)
    started = time.time()
    hash_before = sha256_of(DATA_PATH)

    R(f"train_phase1.py | Python {platform.python_version()} | scikit-learn {sklearn.__version__} "
      f"| pandas {pd.__version__} | numpy {np.__version__}")
    R(f"Dataset: {DATA_PATH}")
    R(f"SHA-256: {hash_before}")
    R(f"Validation: RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}, "
      f"random_state={SEED}); decision threshold {THRESHOLD}")

    df = pd.read_excel(DATA_PATH, sheet_name=0)
    R(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    # ------------------------------------------------------------------ 1. targets
    R.section("1. TARGETS (positive class = CAD / Stenotic)")
    target_cols, y_all = {}, {}
    for t in TARGETS:
        c = find_col(df, t)
        if c is None:
            R(f"ERROR: target column {t!r} not found.")
            sys.exit(1)
        v = df[c].astype(str).str.strip().str.lower()
        bad = sorted(set(v) - POSITIVE_LABELS - NEGATIVE_LABELS)
        if bad:
            R(f"ERROR: {t} has unexpected labels {bad}.")
            sys.exit(1)
        target_cols[t] = c
        y_all[t] = v.isin(POSITIVE_LABELS).astype(int).to_numpy()
        R(f"{t:4s}: positive={int(y_all[t].sum())}, negative={int((1 - y_all[t]).sum())}, "
          f"prevalence={y_all[t].mean():.3f}")

    any_vessel = (y_all["LAD"] + y_all["LCX"] + y_all["RCA"]) > 0
    inconsistent_idx = list(df.index[(y_all["Cath"] == 1) != any_vessel])
    R("")
    R(f"Rows where Cath disagrees with 'any vessel stenotic': {inconsistent_idx}")
    for i in inconsistent_idx:
        R("  index " + str(i) + ": " + ", ".join(f"{t}={df.loc[i, c]!r}"
                                                    for t, c in target_cols.items()))
    R("Primary analysis KEEPS these rows with labels exactly as recorded.")

    # ------------------------------------------------------------------ 2. features
    R.section("2. FEATURES")
    excluded = {}
    for t, c in target_cols.items():
        excluded[c] = "target column (excluded from every model)"
    candidates = [c for c in df.columns if c not in target_cols.values()]
    for c in candidates:
        counts = df[c].value_counts()
        if len(counts) <= 1:
            excluded[c] = "constant (one value in all rows)"
        elif len(counts) == 2 and counts.min() < RARE_MIN_COUNT:
            excluded[c] = (f"binary with only {int(counts.min())} patient(s) in one level "
                           f"(< {RARE_MIN_COUNT}); too rare to learn from")
    features = [c for c in candidates if c not in excluded]

    # Leakage guard: stop immediately if any target name slipped into the features.
    leaked = [c for c in features if norm(c) in {norm(t) for t in TARGETS}]
    if leaked:
        R(f"ERROR: target columns found among features: {leaked}")
        sys.exit(1)

    # Continuous/ordinal numbers are scaled. 0/1 indicator columns are NOT put here:
    # they go to the categorical branch, so every yes/no feature (0/1 or Y/N) is
    # encoded the same way, as one unscaled 0/1 column.
    numeric_set = {c for c in features
                   if pd.api.types.is_numeric_dtype(df[c])
                   and not set(df[c].dropna().unique()) <= {0, 1}}
    categorical = [c for c in features if c not in numeric_set]
    R(f"Excluded columns ({len(excluded)}):")
    for c, why in excluded.items():
        R(f"  - {c!r}: {why}")
    R(f"Features used: {len(features)} ({len(numeric_set)} numeric, {len(categorical)} text/categorical)")
    R("Text/categorical features and their levels (encoded inside each fold):")
    for c in categorical:
        vc = df[c].astype(str).value_counts()
        rare = [lv for lv, n in vc.items() if n < RARE_MIN_COUNT]
        note = f"   <- rare level(s) {rare}" if rare else ""
        R(f"  {c!r}: " + ", ".join(f"{lv!r}:{n}" for lv, n in vc.items()) + note)
    R("NOTE: 'Sex' is used exactly as recorded ('Male' / 'Fmale'). 'Fmale' is most likely a")
    R("      spelling of 'Female', but this is unconfirmed by the documentation; the label")
    R("      is not changed in the data and the issue is recorded in metadata.json.")
    R("NOTE: Function Class (0-3) and Region RWMA (0-4) are treated as ordinal numbers.")
    R("      Their exact definitions are undocumented; see metadata.json known_issues.")
    R("NOTE: 'Obesity' in this file equals BMI >= 25 (i.e. overweight), not BMI >= 30.")

    clinical = []
    for name in CLINICAL_BASELINE_FEATURES:
        c = find_col(df, name)
        if c is None or c not in features:
            R(f"WARNING: clinical baseline feature {name!r} not available; skipped.")
        else:
            clinical.append(c)
    reduced_drop = [find_col(df, n) for n in REDUNDANCY_DROP]
    reduced_drop = [c for c in reduced_drop if c is not None and c in features]
    reduced = [c for c in features if c not in reduced_drop]
    R(f"Clinical baseline features ({len(clinical)}): {clinical}")
    R(f"Reduced feature set = all features minus {reduced_drop} ({len(reduced)} features)")

    if features_only:
        # Fit the preprocessing ONCE on all rows purely to display the encoded columns.
        # No model is trained and nothing in models/ or the Phase 1 reports is changed.
        num_cols = [c for c in features if c in numeric_set]
        pre = make_pipeline(DummyClassifier(), num_cols, categorical).named_steps["preprocess"]
        encoded = pd.DataFrame(pre.fit_transform(df[features]),
                               columns=pre.get_feature_names_out())
        R.section("FEATURE CHECK (--features-only): encoded columns")
        R(f"{encoded.shape[1]} encoded columns from {len(features)} features")
        for col in encoded.columns:
            v = encoded[col]
            kind = "0/1 indicator" if set(v.unique()) <= {0.0, 1.0} else "scaled number"
            R(f"  {col:32s} {kind:14s} min={v.min():7.3f} max={v.max():7.3f}")
        source_names = [c.split("__", 1)[1] for c in encoded.columns]
        leaked = [n for n in source_names if any(n == t or n.startswith(t + "_") for t in TARGETS)]
        R(f"Target columns among encoded inputs: {leaked if leaked else 'none'}")
        R("Stopped after the feature check: no models trained.")
        R.close()
        return

    # ------------------------------------------------------------------ 3. models
    logreg = LogisticRegression(C=1.0, max_iter=5000)
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                    random_state=SEED, n_jobs=-1)
    experiments = {
        # name: (estimator, feature list, what it is for)
        "dummy_prior": (DummyClassifier(strategy="prior"), features,
                        "no-skill reference: always predicts the training prevalence"),
        "logreg_clinical": (logreg, clinical, "simple clinical baseline"),
        "logreg_all": (logreg, features, "L2 logistic regression, all features"),
        "rf_all": (forest, features, "random forest, all features"),
        "logreg_reduced": (logreg, reduced, "redundancy check for logistic regression"),
        "rf_reduced": (forest, reduced, "redundancy check for random forest"),
    }
    R.section("3. MODELS (fixed settings, no tuning)")
    for name, (est, cols, why) in experiments.items():
        R(f"  {name:16s} {len(cols):2d} features | {why}")

    # ------------------------------------------------------------------ 4. CV
    R.section("4. REPEATED STRATIFIED CROSS-VALIDATION (primary analysis, all rows)")
    X = df[features].copy()
    fold_rows, oof_frames, cm_rows = [], [], []
    for t in TARGETS:
        y = y_all[t]
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
        splits = list(cv.split(X, y))          # identical folds for every model of this task
        t0 = time.time()
        for name, (est, cols, _) in experiments.items():
            rows, oof = run_cv(X, y, cols, numeric_set, est, splits)
            for r in rows:
                r.update({"task": t, "model": name, "analysis": "primary"})
            fold_rows.extend(rows)
            mean_oof = np.nanmean(oof, axis=1)
            oof_frames.append(pd.DataFrame({
                "row_index": df.index, "task": t, "model": name, "y_true": y,
                "oof_proba_mean": mean_oof, "oof_proba_sd": np.nanstd(oof, axis=1),
            }))
            pred = (mean_oof >= THRESHOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            cm_rows.append({"task": t, "model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        R(f"  {t}: {len(experiments)} models x {len(splits)} folds done in {time.time() - t0:.1f}s")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(REPORT_DIR / "cv_fold_metrics.csv", index=False)
    pd.concat(oof_frames).to_csv(REPORT_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(REPORT_DIR / "confusion_matrices_oof.csv", index=False)

    summary = (folds.groupby(["task", "model"], sort=False)[METRICS]
               .agg(["mean", "std", "min", "max"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(REPORT_DIR / "cv_summary.csv", index=False)

    R.section("5. RESULTS: mean +/- SD across 25 test folds")
    R("(Precision/recall/F1/specificity use threshold 0.5. The dummy model has AUC 0.5 by")
    R(" design; its precision/recall depend only on which class is the majority.)")
    show = ["roc_auc", "pr_auc", "brier", "balanced_accuracy", "precision", "recall",
            "specificity", "f1", "accuracy"]
    for t in TARGETS:
        R("")
        R(f"--- {t} (prevalence {y_all[t].mean():.3f}) ---")
        sub = summary[summary.task == t]
        table = pd.DataFrame({"model": sub.model})
        for m in show:
            table[m] = [f"{a:.3f}±{b:.3f}" for a, b in zip(sub[f"{m}_mean"], sub[f"{m}_std"])]
        R(table.to_string(index=False))
        R("  Out-of-fold confusion matrix at 0.5 (mean probability over 5 repeats):")
        cm = pd.DataFrame([r for r in cm_rows if r["task"] == t]).drop(columns="task")
        R("  " + cm.to_string(index=False).replace("\n", "\n  "))

    # ------------------------------------------------------------------ 6. paired
    R.section("6. PAIRED COMPARISONS ON IDENTICAL FOLDS (ROC-AUC, A minus B)")
    R("Fold scores overlap (shared training data), so these are descriptive, not")
    R("formal significance tests.")
    pairs = [("logreg_clinical", "dummy_prior"), ("logreg_all", "logreg_clinical"),
             ("rf_all", "logreg_all"), ("logreg_reduced", "logreg_all"),
             ("rf_reduced", "rf_all")]
    pair_rows = []
    for t in TARGETS:
        ft = folds[folds.task == t]
        for a, b in pairs:
            da = ft[ft.model == a].sort_values("fold")["roc_auc"].to_numpy()
            db = ft[ft.model == b].sort_values("fold")["roc_auc"].to_numpy()
            d = da - db
            pair_rows.append({"task": t, "A": a, "B": b, "mean_diff": round(d.mean(), 4),
                              "sd_diff": round(d.std(ddof=1), 4),
                              "folds_A_better": int((d > 0).sum()), "n_folds": len(d)})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(REPORT_DIR / "paired_auc_comparison.csv", index=False)
    R(pair_df.to_string(index=False))

    # ------------------------------------------------------------------ 7. sensitivity
    R.section("7. SENSITIVITY ANALYSIS: inconsistent target row(s) removed")
    sens_rows = []
    if inconsistent_idx:
        keep = ~df.index.isin(inconsistent_idx)
        Xs = X.loc[keep].reset_index(drop=True)
        for t in TARGETS:
            ys = y_all[t][keep]
            cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
            splits = list(cv.split(Xs, ys))
            for name in ("logreg_all", "rf_all"):
                est, cols, _ = experiments[name]
                rows, _ = run_cv(Xs, ys, cols, numeric_set, est, splits)
                auc_s = np.mean([r["roc_auc"] for r in rows])
                auc_p = summary[(summary.task == t) & (summary.model == name)]["roc_auc_mean"].iloc[0]
                sens_rows.append({"task": t, "model": name, "n_rows": int(keep.sum()),
                                  "auc_primary": round(auc_p, 4),
                                  "auc_without_inconsistent": round(auc_s, 4),
                                  "difference": round(auc_s - auc_p, 4)})
        sens = pd.DataFrame(sens_rows)
        sens.to_csv(REPORT_DIR / "sensitivity_inconsistent_row.csv", index=False)
        R(sens.to_string(index=False))
        R("(Fold assignments differ once a row is removed, so small differences are expected")
        R(" from re-shuffling alone.)")
    else:
        R("No inconsistent rows found; sensitivity analysis not needed.")

    # ------------------------------------------------------------------ 8. save
    R.section("8. SAVING CANDIDATE MODELS (each refitted on all rows)")
    R("These are candidates only. The final model per task is chosen in the next step,")
    R("from the cross-validation results above, never from these refitted models.")
    saved = {}
    for t in TARGETS:
        for name, (est, cols, _) in experiments.items():
            if name == "dummy_prior":
                continue
            num_cols = [c for c in cols if c in numeric_set]
            cat_cols = [c for c in cols if c not in numeric_set]
            pipe = make_pipeline(est, num_cols, cat_cols)
            pipe.fit(X[cols], y_all[t])
            path = MODEL_DIR / f"{t}__{name}.joblib"
            joblib.dump(pipe, path)
            saved[f"{t}__{name}"] = {"file": path.name, "features": cols}
    R(f"Saved {len(saved)} pipelines to {MODEL_DIR}")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data_file": DATA_PATH.name, "data_sha256": hash_before,
        "seed": SEED, "cv": {"type": "RepeatedStratifiedKFold", "n_splits": N_SPLITS,
                             "n_repeats": N_REPEATS},
        "decision_threshold": THRESHOLD,
        "targets": {t: {"column": c, "positive": "CAD/Stenotic", "negative": "Normal"}
                    for t, c in target_cols.items()},
        "excluded_columns": excluded,
        "features_all": features, "numeric_features": sorted(numeric_set),
        "categorical_features": categorical,
        "feature_sets": {"clinical": clinical, "reduced": reduced, "reduced_dropped": reduced_drop},
        "inconsistent_target_rows": [int(i) for i in inconsistent_idx],
        "known_issues": [
            "Sex level 'Fmale' used as recorded; presumed 'Female' but unconfirmed.",
            "Inconsistent target row(s) kept in primary analysis; see sensitivity file.",
            "Units are undocumented by UCI; not needed for training, needed for the UI.",
            "Function Class (observed 0-3; level 1 has one patient) and Region RWMA (0-4) "
            "are undocumented; treated as ordinal. RWMA rises with lower EF and more Q "
            "waves, consistent with extent/severity, not a coded wall region.",
            "Obesity equals BMI >= 25 in this file; BMI equals Weight/(Length/100)^2.",
        ],
        "models": {name: {"estimator": str(est), "purpose": why}
                   for name, (est, _, why) in experiments.items()},
        "saved_pipelines": saved,
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    R.section("DONE")
    unchanged = sha256_of(DATA_PATH) == hash_before
    R(f"Original dataset unchanged: {'YES' if unchanged else 'NO - INVESTIGATE'}")
    R(f"Total time: {time.time() - started:.1f}s")
    R(f"Reports: {REPORT_DIR}")
    R(f"Models:  {MODEL_DIR}")
    R.close()


if __name__ == "__main__":
    main()