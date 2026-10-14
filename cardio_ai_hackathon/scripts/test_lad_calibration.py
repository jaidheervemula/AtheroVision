"""
test_lad_calibration.py
=======================
Experiment 1: does sigmoid (Platt) calibration make the LAD random forest's probabilities
more reliable, without losing discrimination?

EDUCATIONAL PROTOTYPE. Internal cross-validation on 303 patients from one centre; not
external or clinical validation. This script does NOT change the deployed model,
configuration, data, or any existing report. All outputs go to reports/lad_calibration/.

Compared on IDENTICAL outer splits (RepeatedStratifiedKFold, 5 folds x 5 repeats, seed 42):
  1. baseline   : the LAD rf_reduced pipeline (train_phase1.make_pipeline + estimator settings
                  cloned from models/phase1_candidates/LAD__rf_reduced.joblib), fitted on the
                  outer training fold.
  2. calibrated : CalibratedClassifierCV(estimator=<same unfitted pipeline>, method="sigmoid",
                  cv=StratifiedKFold(5, shuffle=True, random_state=42), ensemble=False),
                  fitted on the outer training fold ONLY.

How the calibrated model avoids leakage (scikit-learn 1.9.1 API)
  * Inside each outer training fold, cross_val_predict over the inner 5 folds gives
    predictions for every training patient from pipelines that did not see that patient;
    the sigmoid is fitted to those predictions and the training labels.
  * The pipeline (preprocessing + forest) is then refitted on the whole outer training fold.
  * The outer validation fold is used only for scoring; its labels and predictions are
    never seen by the forest, the preprocessing or the calibrator.

Outputs: reports/lad_calibration/
  calibration_report.md, calibration_comparison.csv, lad_calibration_predictions.csv,
  repeat_metrics.csv, fold_metrics.csv, calibration_bins.csv,
  calibration_LAD_baseline.svg, calibration_LAD_sigmoid.svg, run_config.json,
  lad_calibration_log.txt

Run from the project root (.venv active):
    python scripts\\test_lad_calibration.py
"""

import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import final_ml_evaluation as fe  # noqa: E402  (metric helpers and SVG plot only)
import train_phase1 as tp  # noqa: E402  (pipeline builder and CV settings only)

TASK = "LAD"
FORBIDDEN = ("Cath", "LAD", "LCX", "RCA")
INNER_SEED = 42
N_INNER = 5
N_BOOT = 2000
BOOT_SEED = 20261011
THRESHOLD = 0.5
OUT = PROJECT_ROOT / "reports" / "lad_calibration"
CONFIG = PROJECT_ROOT / "config" / "model_selection.json"
METADATA = PROJECT_ROOT / "models" / "phase1_candidates" / "metadata.json"
REF_OOF = PROJECT_ROOT / "reports" / "final_ml_evaluation" / "oof_per_repeat_selected.csv"
REF_FOLDS = PROJECT_ROOT / "reports" / "phase1_modeling" / "cv_fold_metrics.csv"
APPROACHES = ("baseline", "sigmoid")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, p):
    """All comparison metrics for one set of predictions (one per patient)."""
    t = fe.threshold_metrics(y, p, THRESHOLD)
    two = len(np.unique(y)) == 2
    return {
        # discrimination
        "roc_auc": roc_auc_score(y, p) if two else float("nan"),
        # probability accuracy / calibration
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "calibration_intercept": fe.calibration_intercept(y, p) if two else float("nan"),
        "calibration_slope": fe.calibration_slope(y, p) if two else float("nan"),
        "mean_predicted": float(np.mean(p)),
        # classification at threshold 0.5
        "sensitivity": t["sensitivity"], "specificity": t["specificity"],
        "precision": t["precision"], "recall": t["sensitivity"], "f1": t["f1"],
        "accuracy": t["accuracy"],
    }


METRIC_GROUPS = {
    "roc_auc": "discrimination",
    "brier": "probability accuracy", "log_loss": "probability accuracy",
    "calibration_intercept": "calibration", "calibration_slope": "calibration",
    "mean_predicted": "calibration",
    "sensitivity": "classification @0.5", "specificity": "classification @0.5",
    "precision": "classification @0.5", "recall": "classification @0.5",
    "f1": "classification @0.5", "accuracy": "classification @0.5",
}
METRICS = list(METRIC_GROUPS)


def repeat_avg(y, P):
    vals = [metrics(y, P[:, r]) for r in range(P.shape[1])]
    return {k: float(np.nanmean([v[k] for v in vals])) for k in METRICS}, vals


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log = fe.Log(OUT / "lad_calibration_log.txt")
    started = time.time()
    warnings_list = []

    # ------------------------------------------------------------------ inputs and checks
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    spec = config["tasks"][TASK]
    data_path = PROJECT_ROOT / "data" / "raw" / config["data_file"]
    artifact = PROJECT_ROOT / spec["artifact"]
    protected = [data_path, CONFIG, METADATA, REF_OOF, REF_FOLDS, artifact,
                 PROJECT_ROOT / "scripts" / "train_phase1.py",
                 PROJECT_ROOT / "scripts" / "final_ml_evaluation.py"]
    protected += sorted((PROJECT_ROOT / "models" / "phase1_candidates").glob("*.joblib"))
    hashes = {p: sha256_of(p) for p in protected}
    if hashes[data_path] != config["data_sha256"]:
        raise SystemExit("Dataset hash differs from config; stopping.")
    if hashes[artifact] != spec["artifact_sha256"]:
        raise SystemExit("LAD artifact hash differs from config; stopping.")

    df = pd.read_excel(data_path, sheet_name=0)
    deployed = joblib.load(artifact)
    features = list(spec["input_features"])
    if features != list(deployed.feature_names_in_):
        raise SystemExit("Config features differ from the deployed pipeline; stopping.")
    leaked = [f for f in features if f in FORBIDDEN]
    if leaked:
        raise SystemExit(f"Forbidden target columns among features: {leaked}")
    numeric_set = set(meta["numeric_features"])
    num = [c for c in features if c in numeric_set]
    cat = [c for c in features if c not in numeric_set]
    raw = df[TASK].astype(str).str.strip()
    if sorted(raw.unique()) != sorted([spec["positive_label"], spec["negative_label"]]):
        raise SystemExit(f"Unexpected LAD labels {sorted(raw.unique())}")
    y = (raw == spec["positive_label"]).to_numpy().astype(int)
    X = df[features].copy()
    rf_unfitted = clone(deployed.named_steps["model"])
    base_pipeline = tp.make_pipeline(rf_unfitted, num, cat)

    log("test_lad_calibration.py | educational prototype, internal CV only")
    log(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}")
    log(f"Features: {len(features)} ({len(num)} numeric, {len(cat)} categorical); forbidden "
        f"targets {list(FORBIDDEN)} absent: True")
    log(f"Labels: 1 = {spec['positive_label']!r} (n={y.sum()}), 0 = {spec['negative_label']!r} "
        f"(n={len(y) - y.sum()})")
    log(f"Estimator (cloned from deployment): {rf_unfitted}")

    # ------------------------------------------------------------------ outer CV
    outer = RepeatedStratifiedKFold(n_splits=tp.N_SPLITS, n_repeats=tp.N_REPEATS, random_state=tp.SEED)
    splits = list(outer.split(X, y))
    P = {a: np.full((len(y), tp.N_REPEATS), np.nan) for a in APPROACHES}
    fit_warnings = []

    def record_warnings(caught, approach, fold_id):
        for w in caught:
            fit_warnings.append({"approach": approach, "outer_fold": fold_id,
                                 "category": w.category.__name__, "message": str(w.message)})
    fold_of = np.full((len(y), tp.N_REPEATS), -1)
    fold_rows, sigmoid_params = [], []
    for fold_id, (tr, te) in enumerate(splits):
        r = fold_id // tp.N_SPLITS
        Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base = clone(base_pipeline).fit(Xtr, ytr)
            p_base = base.predict_proba(Xte)[:, 1]
        record_warnings(caught, "baseline", fold_id)

        inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=INNER_SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal = CalibratedClassifierCV(estimator=clone(base_pipeline), method="sigmoid",
                                         cv=inner, ensemble=False).fit(Xtr, ytr)
            p_cal = cal.predict_proba(Xte)[:, 1]
        record_warnings(caught, "sigmoid", fold_id)
        sig = cal.calibrated_classifiers_[0].calibrators[0]
        sigmoid_params.append({"fold": fold_id, "repeat": r, "a": float(sig.a_), "b": float(sig.b_)})

        P["baseline"][te, r], P["sigmoid"][te, r] = p_base, p_cal
        fold_of[te, r] = fold_id
        for a, p in (("baseline", p_base), ("sigmoid", p_cal)):
            m = metrics(y[te], p)
            fold_rows.append({"approach": a, "fold": fold_id, "repeat": r, "n_test": len(te),
                              "n_test_positive": int(y[te].sum()), **m})
    for a in APPROACHES:
        if np.isnan(P[a]).any():
            raise SystemExit(f"{a}: some patients lack an OOF prediction.")
    unknown_cat = [w for w in fit_warnings if "unknown categories" in w["message"]]
    other_warn = [w for w in fit_warnings if "unknown categories" not in w["message"]]
    if unknown_cat:
        import re
        names = sorted({cat[int(i)] for w in unknown_cat
                        for i in re.findall(r"\d+", w["message"].split("[")[1].split("]")[0])})
        msg = (f"{len(unknown_cat)} 'unknown category' warning(s) "
               f"({', '.join(sorted({w['approach'] for w in unknown_cat}))}; outer folds "
               f"{sorted({w['outer_fold'] for w in unknown_cat})}; feature(s) {names}): a rare "
               "category was absent from a training split and was encoded as all zeros for "
               "those predictions. This is not leakage, but it is a small-sample effect.")
        warnings_list.append(msg)
        log("WARNING: " + msg)
    for w in other_warn:
        warnings_list.append(f"{w['approach']} fold {w['outer_fold']}: {w['category']}: {w['message'][:150]}")
    pd.DataFrame(fit_warnings, columns=["approach", "outer_fold", "category", "message"]).to_csv(
        OUT / "fit_warnings.csv", index=False)
    log(f"Outer CV complete: {len(splits)} folds; fits = {len(splits)} baseline + "
        f"{len(splits) * (N_INNER + 1)} calibrated (inner {N_INNER} + 1 refit per fold)")

    # ------------------------------------------------------------------ baseline reproduction
    ref = pd.read_csv(REF_OOF)
    ref = ref[(ref.task == TASK) & (ref.model == spec["model_name"])]
    ref_P = ref.pivot(index="row_index", columns="repeat", values="oof_proba").sort_index().to_numpy()
    diff_oof = float(np.max(np.abs(ref_P - P["baseline"])))
    ref_f = pd.read_csv(REF_FOLDS)
    ref_auc = ref_f[(ref_f.task == TASK) & (ref_f.model == spec["model_name"])].sort_values("fold").roc_auc.to_numpy()
    my_auc = pd.DataFrame(fold_rows).query("approach == 'baseline'").sort_values("fold").roc_auc.to_numpy()
    diff_auc = float(np.max(np.abs(ref_auc - my_auc)))
    reproduced = diff_oof < 1e-6 and diff_auc < 1e-6
    log(f"Baseline reproduction: max |OOF - saved per-repeat OOF| = {diff_oof:.2e}; "
        f"max |fold AUC - Phase 1| = {diff_auc:.2e} -> {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")
    if not reproduced:
        warnings_list.append("Baseline did not reproduce the saved LAD evaluation; compare with care.")

    # ------------------------------------------------------------------ summaries
    rep_rows, summ = [], {}
    for a in APPROACHES:
        summ[a], per = repeat_avg(y, P[a])
        for r, m in enumerate(per):
            rep_rows.append({"approach": a, "repeat": r, **m})
    rep_df = pd.DataFrame(rep_rows)
    fold_df = pd.DataFrame(fold_rows)

    rng = np.random.default_rng(BOOT_SEED)
    boot = {a: {k: [] for k in METRICS} for a in APPROACHES}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))        # patients, with replacement
        for a in APPROACHES:
            est, _ = repeat_avg(y[idx], P[a][idx])
            for k in METRICS:
                boot[a][k].append(est[k])

    def ci(values):
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))) if len(v) else (
            float("nan"), float("nan"), 0)

    comp = []
    for k in METRICS:
        b, c = summ["baseline"][k], summ["sigmoid"][k]
        d = np.asarray(boot["sigmoid"][k]) - np.asarray(boot["baseline"][k])
        bl, bh, bn = ci(boot["baseline"][k])
        cl, chh, cn = ci(boot["sigmoid"][k])
        dl, dh, dn = ci(d)
        rb = rep_df[rep_df.approach == "baseline"][k]
        rc = rep_df[rep_df.approach == "sigmoid"][k]
        rdiff = rc.to_numpy() - rb.to_numpy()
        fb = fold_df[fold_df.approach == "baseline"].sort_values("fold")[k].to_numpy()
        fc = fold_df[fold_df.approach == "sigmoid"].sort_values("fold")[k].to_numpy()
        comp.append({
            "metric": k, "group": METRIC_GROUPS[k],
            "baseline": b, "sigmoid": c, "difference_sigmoid_minus_baseline": c - b,
            "baseline_ci95_low": bl, "baseline_ci95_high": bh,
            "sigmoid_ci95_low": cl, "sigmoid_ci95_high": chh,
            "difference_ci95_low": dl, "difference_ci95_high": dh,
            "bootstrap_valid_replicates": min(bn, cn, dn),
            "repeats_sigmoid_higher": int(np.sum(rdiff > 0)), "n_repeats": len(rdiff),
            "baseline_repeat_sd": float(rb.std(ddof=1)), "sigmoid_repeat_sd": float(rc.std(ddof=1)),
            "baseline_fold_mean": float(np.nanmean(fb)), "baseline_fold_sd": float(np.nanstd(fb, ddof=1)),
            "sigmoid_fold_mean": float(np.nanmean(fc)), "sigmoid_fold_sd": float(np.nanstd(fc, ddof=1)),
            "folds_sigmoid_higher": int(np.sum((fc - fb) > 0)), "n_folds": len(fb)})
    comp_df = pd.DataFrame(comp)

    # ------------------------------------------------------------------ files
    pred_rows = []
    for i in range(len(y)):
        for r in range(tp.N_REPEATS):
            pred_rows.append({"row_index": i, "repeat": r, "outer_fold": int(fold_of[i, r]),
                              "y_true": int(y[i]), "p_baseline": P["baseline"][i, r],
                              "p_sigmoid": P["sigmoid"][i, r]})
    pd.DataFrame(pred_rows).to_csv(OUT / "lad_calibration_predictions.csv", index=False)
    comp_df.to_csv(OUT / "calibration_comparison.csv", index=False)
    rep_df.to_csv(OUT / "repeat_metrics.csv", index=False)
    fold_df.merge(pd.DataFrame(sigmoid_params), on=["fold", "repeat"], how="left").to_csv(
        OUT / "fold_metrics.csv", index=False)

    bin_rows = []
    for a, label in (("baseline", "rf_reduced, uncalibrated"), ("sigmoid", "rf_reduced + sigmoid")):
        pm = P[a].mean(axis=1)
        order = np.argsort(pm, kind="mergesort")
        bins = []
        for chunk in np.array_split(order, fe.N_CAL_BINS):
            k_pos, n = int(y[chunk].sum()), len(chunk)
            lo, hi = fe.wilson(k_pos, n)
            bins.append({"approach": a, "bin": len(bins) + 1, "n_patients": n,
                         "mean_predicted": float(pm[chunk].mean()), "observed_rate": k_pos / n,
                         "ci_low": lo, "ci_high": hi})
        stats = {"brier": float(np.mean((pm - y) ** 2)),
                 "calibration_slope": fe.calibration_slope(y, pm),
                 "calibration_intercept": fe.calibration_intercept(y, pm)}
        fe.calibration_svg(TASK, label, bins, stats, OUT / f"calibration_LAD_{a}.svg")
        bin_rows += bins
    pd.DataFrame(bin_rows).to_csv(OUT / "calibration_bins.csv", index=False)

    run_config = {
        "script": "scripts/test_lad_calibration.py",
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "outer_cv": {"type": "RepeatedStratifiedKFold", "n_splits": tp.N_SPLITS,
                     "n_repeats": tp.N_REPEATS, "random_state": tp.SEED},
        "inner_cv": {"type": "StratifiedKFold", "n_splits": N_INNER, "shuffle": True,
                     "random_state": INNER_SEED},
        "calibration": {"class": "CalibratedClassifierCV", "method": "sigmoid", "ensemble": False},
        "estimator": str(rf_unfitted), "threshold": THRESHOLD,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "patient",
                      "statistic": "repeat-averaged OOF metric"},
        "features": features, "numeric_features": num, "categorical_features": cat,
        "forbidden_input_columns": list(FORBIDDEN), "forbidden_columns_in_features": leaked,
        "labels": {"positive": spec["positive_label"], "negative": spec["negative_label"]},
        "baseline_reproduced": reproduced,
        "fit_warnings_count": len(fit_warnings),
        "baseline_max_abs_oof_diff": diff_oof, "baseline_max_abs_fold_auc_diff": diff_auc,
        "input_sha256": {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): h for p, h in hashes.items()},
    }
    (OUT / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ report
    def row(k):
        return comp_df[comp_df.metric == k].iloc[0]

    fb_auc = fold_df[fold_df.approach == "baseline"].sort_values("fold").roc_auc.to_numpy()
    fs_auc = fold_df[fold_df.approach == "sigmoid"].sort_values("fold").roc_auc.to_numpy()
    fold_auc_gap = float(np.max(np.abs(fb_auc - fs_auc)))

    def fmt(k, a):
        r = row(k)
        return f"{r[a]:.3f} ({r[a + '_ci95_low']:.3f}–{r[a + '_ci95_high']:.3f})"

    def dfmt(k):
        r = row(k)
        return (f"{r.difference_sigmoid_minus_baseline:+.3f} ({r.difference_ci95_low:+.3f} to "
                f"{r.difference_ci95_high:+.3f})")

    def verdict(k, lower_better=False, target=None):
        r = row(k)
        lo, hi = r.difference_ci95_low, r.difference_ci95_high
        if target is not None:
            before, after = abs(r.baseline - target), abs(r.sigmoid - target)
            return "closer to ideal" if after < before else "further from ideal"
        if lo > 0:
            return "worse" if lower_better else "higher"
        if hi < 0:
            return "better" if lower_better else "lower"
        return "no clear difference"

    L = ["# LAD calibration experiment: uncalibrated vs sigmoid-calibrated random forest", "",
         "> **Educational prototype. Not a diagnostic tool.** Internal repeated cross-validation "
         "on 303 patients from one centre. This is not external or clinical validation, and this "
         "experiment does not change the deployed model.", "",
         "## Setup", "",
         f"- **Outer validation** (identical splits for both approaches): RepeatedStratifiedKFold, "
         f"{tp.N_SPLITS} folds × {tp.N_REPEATS} repeats, `random_state={tp.SEED}`, stratified on LAD.",
         f"- **Labels:** 1 = `{spec['positive_label']}` ({int(y.sum())}), 0 = `{spec['negative_label']}` "
         f"({int(len(y) - y.sum())}).",
         f"- **Features:** {len(features)} inputs, exactly the deployed `{spec['model_name']}` schema "
         f"({len(num)} numeric, {len(cat)} categorical). `Cath`, `LAD`, `LCX` and `RCA` are not inputs "
         "(checked by the script).",
         "- **Baseline:** `train_phase1.make_pipeline` with the random-forest settings cloned from "
         f"`{spec['artifact']}` (`{rf_unfitted}`). Preprocessing is fitted on each training fold.",
         f"- **Calibrated:** `CalibratedClassifierCV(method='sigmoid', ensemble=False, "
         f"cv=StratifiedKFold({N_INNER}, shuffle=True, random_state={INNER_SEED}))`, wrapped around the "
         f"same unfitted pipeline and fitted on the outer training fold only (scikit-learn "
         f"{sklearn.__version__}).",
         "  - Inner cross-validated predictions on the training fold are used to fit the sigmoid.",
         "  - The forest is then refitted on the whole training fold.",
         "  - The outer validation fold is used only for scoring.",
         "  - `ensemble=False` keeps it *the same forest plus a calibrator*, not an average of five "
         "forests trained on less data.",
         f"- **Fits:** {len(splits)} baseline and {len(splits) * (N_INNER + 1)} calibrated. The seeds are "
         f"the outer CV seed {tp.SEED}, the inner CV seed {INNER_SEED}, the forest seed (from the "
         f"deployed settings) and the bootstrap seed {BOOT_SEED}.",
         f"- **Baseline reproduction: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}.** The largest "
         f"difference from the saved LAD per-repeat out-of-fold predictions is {diff_oof:.1e}, and from "
         f"the Phase 1 fold AUCs {diff_auc:.1e}.", "",
         "## Results", "",
         "Each value is computed on one repeat's 303 out-of-fold predictions and averaged over the "
         "5 repeats. Brackets give 95% patient-level bootstrap intervals "
         f"({N_BOOT} resamples; each patient keeps all 5 predictions). For differences, the same "
         "resamples are used for both approaches (paired).", "",
         "| Group | Metric | Uncalibrated | Sigmoid | Difference (sigmoid − uncal.) | Repeats where sigmoid higher |",
         "|---|---|---|---|---|---|"]
    for k in METRICS:
        r = row(k)
        L.append(f"| {r.group} | {k} | {fmt(k, 'baseline')} | {fmt(k, 'sigmoid')} | {dfmt(k)} | "
                 f"{r.repeats_sigmoid_higher}/{r.n_repeats} |")
    L += ["", "Ideal values: calibration slope 1 and intercept 0. Lower is better for Brier and log "
          "loss. The 25-fold means and SDs, comparable to the Phase 1 tables, are in "
          "`calibration_comparison.csv` (`*_fold_mean`, `*_fold_sd`).", "",
          "## Reading the results (computed, not pre-judged)", "",
          f"- **Discrimination (ROC-AUC):** {dfmt('roc_auc')}: {verdict('roc_auc')} at repeat level. "
          f"Within each outer fold the ranking is unchanged: the largest per-fold ROC-AUC difference is "
          f"{fold_auc_gap:.1e}, because the calibrated pipeline contains the same forest and a monotone "
          "sigmoid. The repeat-level change arises because each fold has its own sigmoid, which "
          "shifts patients from different folds relative to each other when the 5 folds are pooled.",
          f"- **Brier score:** {dfmt('brier')}: {verdict('brier', lower_better=True)}.",
          f"- **Log loss:** {dfmt('log_loss')}: {verdict('log_loss', lower_better=True)}.",
          f"- **Calibration slope:** {row('calibration_slope').baseline:.2f} → "
          f"{row('calibration_slope').sigmoid:.2f}: {verdict('calibration_slope', target=1.0)} "
          f"(difference {dfmt('calibration_slope')}).",
          f"- **Calibration intercept:** {row('calibration_intercept').baseline:+.2f} → "
          f"{row('calibration_intercept').sigmoid:+.2f}. Both are essentially 0 (difference "
          f"{dfmt('calibration_intercept')}); calibration-in-the-large was already fine.",
          f"- **Threshold 0.5 classification:** sensitivity {dfmt('sensitivity')}, specificity "
          f"{dfmt('specificity')}, precision {dfmt('precision')}, F1 {dfmt('f1')}, accuracy "
          f"{dfmt('accuracy')}. These metrics change because calibration moves probabilities across "
          "0.5. That change is a different operating point, not better or worse discrimination.", "",
          "| Calibration plots (10 bins of about 30 patients; dashed line = ideal) | |",
          "|---|---|",
          "| ![uncalibrated](calibration_LAD_baseline.svg) | ![sigmoid](calibration_LAD_sigmoid.svg) |", "",
          "## Limitations", "",
          "- 303 patients from one centre; 177 Stenotic and 126 Normal. The intervals are wide, and "
          "about 30 patients per calibration bin makes each plotted point uncertain.",
          "- The intervals reflect only which patients were evaluated. They exclude retraining "
          "variability and the optimism from having selected `rf_reduced` among six candidates on "
          "these same folds. That selection happened before this experiment, so the baseline numbers "
          "are already somewhat optimistic.",
          "- The calibrator for each fold is learned from about 242 training patients via inner CV, "
          "which is a small amount of data for estimating a sigmoid.",
          "- The 0.5 threshold is not tuned or clinically optimised.",
          "- If calibration is adopted, a new deployment artifact would be needed. It would be fitted "
          "on all 303 patients, so its calibration could not itself be checked on independent data.",
          "- Nothing here shows clinical usefulness.", "",
          "## Files", "",
          "`calibration_comparison.csv` (all metrics, intervals and differences), "
          "`lad_calibration_predictions.csv` (patient × repeat out-of-fold probabilities with outer "
          "fold IDs), `repeat_metrics.csv`, `fold_metrics.csv` (with each fold's fitted sigmoid "
          "parameters a, b), `calibration_bins.csv`, the two SVG plots, `run_config.json` (seeds, "
          "versions, feature schema, input file hashes), `fit_warnings.csv` (every warning raised "
          "while fitting) and `lad_calibration_log.txt`.", "",
          "**No decision is made here.** Whether to adopt calibration is left to review of these results."]
    if warnings_list:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in warnings_list]
    (OUT / "calibration_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    changed = [str(p.relative_to(PROJECT_ROOT)) for p in protected if sha256_of(p) != hashes[p]]
    if changed:
        warnings_list.append(f"Protected files changed: {changed}")

    # ------------------------------------------------------------------ console summary
    log("\n=== SUMMARY (repeat-averaged OOF; 95% bootstrap CI of difference) ===")
    for k in ("roc_auc", "brier", "log_loss", "calibration_slope", "calibration_intercept",
              "sensitivity", "specificity", "precision", "f1", "accuracy"):
        r = row(k)
        log(f"  {k:22s} uncal {r.baseline:.3f} | sigmoid {r.sigmoid:.3f} | diff {dfmt(k)}")
    log(f"\nBaseline reproduced: {reproduced}")
    log(f"Protected files changed: {changed if changed else 'none'}")
    log(f"Outputs: {OUT}")
    log("Warnings: " + ("; ".join(warnings_list) if warnings_list else "none"))
    log(f"Total time {time.time() - started:.1f}s")
    log("LAD CALIBRATION EXPERIMENT COMPLETE (no decision made; deployed model unchanged)")
    log.close()


if __name__ == "__main__":
    main()
