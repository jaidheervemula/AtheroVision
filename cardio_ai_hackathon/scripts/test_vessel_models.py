"""
test_vessel_models.py
=====================
Experiment 2: can a small, PRE-DECLARED set of alternatives improve LCX and RCA
discrimination over the current clinical logistic-regression baseline?

EDUCATIONAL PROTOTYPE. Internal repeated cross-validation on 303 patients from one centre;
not external or clinical validation. This script changes NO existing file: no deployed
model, configuration, dataset, script or report. All outputs go to
reports/vessel_model_experiment/.

Candidates (fixed before any result was seen; identical for LCX and RCA)
  baseline                 LogisticRegression C=1.0 (cloned from the deployed pipeline), clinical (7)
  lr_balanced_clinical     LogisticRegression C=1.0, class_weight="balanced",            clinical (7)
  lr_strongreg_clinical    LogisticRegression C=0.1,                                      clinical (7)
  lr_strongreg_reduced     LogisticRegression C=0.1,                                      reduced (48)
  rf_conservative_reduced  RandomForest 500 trees, max_depth=5, min_samples_leaf=10,
                           max_features="sqrt", random_state=42,                          reduced (48)
Feature sets are the project's existing ones (metadata.json feature_sets); "reduced" is the
inference input schema. Preprocessing is train_phase1.make_pipeline (fitted in each training fold).

Validation
  Outer: RepeatedStratifiedKFold(5 folds, 5 repeats, random_state=42), created separately for
  each target and stratified on it; every candidate of a target uses identical splits.
  No tuning: every setting above is fixed. Metrics at threshold 0.5 where threshold-based.
  Uncertainty: 2000-replicate patient-level bootstrap (seed 20261012), each patient keeping all
  5 OOF predictions; differences vs baseline use the same resamples (paired).

Run from the project root (.venv active):
    python scripts\\test_vessel_models.py
"""

import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import final_ml_evaluation as fe  # noqa: E402  (threshold metrics, Log)
import train_phase1 as tp  # noqa: E402  (pipeline builder, CV settings)

TARGETS = ["LCX", "RCA"]
FORBIDDEN = ("Cath", "LAD", "LCX", "RCA")
THRESHOLD = 0.5
N_BOOT = 2000
BOOT_SEED = 20261012
OUT = PROJECT_ROOT / "reports" / "vessel_model_experiment"
CONFIG = PROJECT_ROOT / "config" / "model_selection.json"
METADATA = PROJECT_ROOT / "models" / "phase1_candidates" / "metadata.json"
REF_OOF = PROJECT_ROOT / "reports" / "final_ml_evaluation" / "oof_per_repeat_selected.csv"
REF_FOLDS = PROJECT_ROOT / "reports" / "phase1_modeling" / "cv_fold_metrics.csv"

CANDIDATES = {
    # name: (feature set, estimator or None = clone the deployed baseline, purpose)
    "baseline": ("clinical", None, "current deployed model (cloned settings)"),
    "lr_balanced_clinical": ("clinical", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
                             "class weighting"),
    "lr_strongreg_clinical": ("clinical", LogisticRegression(C=0.1, max_iter=5000),
                              "stronger L2 regularization"),
    "lr_strongreg_reduced": ("reduced", LogisticRegression(C=0.1, max_iter=5000),
                             "stronger L2 regularization on the 48-input set"),
    "rf_conservative_reduced": ("reduced", RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=10, max_features="sqrt",
        random_state=42, n_jobs=-1), "conservative random forest"),
}
METRICS = ["roc_auc", "pr_auc", "brier", "log_loss", "sensitivity", "specificity",
           "precision", "recall", "f1", "accuracy"]
LOWER_IS_BETTER = {"brier", "log_loss"}


# ----------------------------------------------------------------------------- metrics
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fast_auc(y, p):
    """ROC-AUC via the rank (Mann-Whitney) formula; ties get average ranks like sklearn."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_ap(y, p):
    """Average precision (step-wise, same definition as sklearn's average_precision_score)."""
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    distinct = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tp = np.cumsum(ys)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_logloss(y, p):
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def metric_set(y, p):
    t = fe.threshold_metrics(y, p, THRESHOLD)
    return {"roc_auc": fast_auc(y, p), "pr_auc": fast_ap(y, p),
            "brier": float(np.mean((p - y) ** 2)), "log_loss": fast_logloss(y, p),
            "sensitivity": t["sensitivity"], "specificity": t["specificity"],
            "precision": t["precision"], "recall": t["sensitivity"], "f1": t["f1"],
            "accuracy": t["accuracy"]}


def repeat_avg(y, P):
    per = [metric_set(y, P[:, r]) for r in range(P.shape[1])]
    out = {}
    for k in METRICS:
        v = np.array([d[k] for d in per], dtype=float)
        out[k] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
    return out, per


def pct_ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


# ----------------------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log = fe.Log(OUT / "vessel_experiment_log.txt")
    started = time.time()
    notes = []

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    data_path = PROJECT_ROOT / "data" / "raw" / config["data_file"]
    protected = [data_path, CONFIG, METADATA, REF_OOF, REF_FOLDS]
    protected += sorted((PROJECT_ROOT / "models" / "phase1_candidates").glob("*.joblib"))
    protected += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    protected += [p for p in sorted((PROJECT_ROOT / "reports").rglob("*"))
                  if p.is_file() and OUT not in p.parents]
    protected += sorted((PROJECT_ROOT / "src").rglob("*.py"))
    hashes = {p: sha256_of(p) for p in protected}
    if hashes[data_path] != config["data_sha256"]:
        raise SystemExit("Dataset hash differs from config; stopping.")

    df = pd.read_excel(data_path, sheet_name=0)
    numeric_set = set(meta["numeric_features"])
    feature_sets = {"clinical": list(meta["feature_sets"]["clinical"]),
                    "reduced": list(meta["feature_sets"]["reduced"])}
    for name, feats in feature_sets.items():
        bad = [f for f in feats if f in FORBIDDEN]
        if bad:
            raise SystemExit(f"Forbidden target column(s) {bad} in feature set {name}.")
    if sorted(feature_sets["reduced"]) != sorted(config["input_schema"]):
        notes.append("The 'reduced' set differs from the inference input schema.")

    log("test_vessel_models.py | educational prototype, internal CV only")
    log(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}")
    log(f"Feature sets: clinical {len(feature_sets['clinical'])}, reduced {len(feature_sets['reduced'])}; "
        f"forbidden {list(FORBIDDEN)} absent from both: True")

    # rare categories (whole dataset; informational, label-free)
    rare_rows = []
    for f in feature_sets["reduced"]:
        if f in numeric_set:
            continue
        for level, n in df[f].value_counts().items():
            if n < 15:
                rare_rows.append({"feature": f, "level": str(level), "n_patients": int(n),
                                  "in_clinical_set": f in feature_sets["clinical"]})
    pd.DataFrame(rare_rows).to_csv(OUT / "rare_categories.csv", index=False)
    log(f"Rare categorical levels (<15 patients) in the reduced set: {len(rare_rows)} "
        f"(none in the clinical set: {not any(r['in_clinical_set'] for r in rare_rows)})")

    ref_oof = pd.read_csv(REF_OOF)
    ref_folds = pd.read_csv(REF_FOLDS)
    oof_rows, fold_rows, rep_rows, comp_rows, diff_rows, warn_rows = [], [], [], [], [], []
    repro = {}
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(df), size=(N_BOOT, len(df)))     # same resamples everywhere

    for target in TARGETS:
        spec = config["tasks"][target]
        deployed = joblib.load(PROJECT_ROOT / spec["artifact"])
        if list(deployed.feature_names_in_) != feature_sets["clinical"]:
            raise SystemExit(f"{target}: deployed baseline does not use the clinical feature set.")
        raw = df[target].astype(str).str.strip()
        if sorted(raw.unique()) != sorted([spec["positive_label"], spec["negative_label"]]):
            raise SystemExit(f"{target}: unexpected labels {sorted(raw.unique())}")
        y = (raw == spec["positive_label"]).to_numpy().astype(int)
        log(f"\n=== {target}: 1 = {spec['positive_label']!r} (n={y.sum()}), "
            f"0 = {spec['negative_label']!r} (n={len(y) - y.sum()}) ===")

        cv = RepeatedStratifiedKFold(n_splits=tp.N_SPLITS, n_repeats=tp.N_REPEATS, random_state=tp.SEED)
        splits = list(cv.split(df, y))
        P = {}
        for cand, (fset, est, _) in CANDIDATES.items():
            feats = feature_sets[fset]
            estimator = clone(deployed.named_steps["model"]) if est is None else clone(est)
            num = [c for c in feats if c in numeric_set]
            cat = [c for c in feats if c not in numeric_set]
            X = df[feats]
            Pc = np.full((len(y), tp.N_REPEATS), np.nan)
            for fold_id, (tr, te) in enumerate(splits):
                r = fold_id // tp.N_SPLITS
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipe = tp.make_pipeline(estimator, num, cat).fit(X.iloc[tr], y[tr])
                    proba = pipe.predict_proba(X.iloc[te])[:, 1]
                for w in caught:
                    warn_rows.append({"target": target, "candidate": cand, "outer_fold": fold_id,
                                      "category": w.category.__name__, "message": str(w.message)})
                Pc[te, r] = proba
                for i, p_i in zip(te, proba):
                    oof_rows.append({"target": target, "candidate": cand, "row_index": int(i),
                                     "repeat": r, "outer_fold": fold_id, "y_true": int(y[i]),
                                     "oof_proba": float(p_i)})
                fm = metric_set(y[te], proba)
                fold_rows.append({"target": target, "candidate": cand, "fold": fold_id, "repeat": r,
                                  "n_test": len(te), "n_test_positive": int(y[te].sum()), **fm})
            if np.isnan(Pc).any():
                raise SystemExit(f"{target}/{cand}: missing OOF predictions.")
            P[cand] = Pc

        # cross-check the fast metric code against scikit-learn on the real predictions
        for cand, Pc in P.items():
            for r in range(tp.N_REPEATS):
                p = Pc[:, r]
                if (abs(fast_auc(y, p) - roc_auc_score(y, p)) > 1e-10
                        or abs(fast_ap(y, p) - average_precision_score(y, p)) > 1e-10
                        or abs(fast_logloss(y, p) - log_loss(y, p)) > 1e-10):
                    raise SystemExit(f"{target}/{cand}: fast metric mismatch with scikit-learn.")

        # baseline reproduction against saved Phase 1 / final evaluation outputs
        ref = ref_oof[(ref_oof.task == target) & (ref_oof.model == spec["model_name"])]
        ref_P = ref.pivot(index="row_index", columns="repeat", values="oof_proba").sort_index().to_numpy()
        d_oof = float(np.max(np.abs(ref_P - P["baseline"])))
        ref_auc = ref_folds[(ref_folds.task == target) & (ref_folds.model == spec["model_name"])
                            ].sort_values("fold").roc_auc.to_numpy()
        my_auc = np.array([r["roc_auc"] for r in fold_rows
                           if r["target"] == target and r["candidate"] == "baseline"])
        d_auc = float(np.max(np.abs(ref_auc - my_auc)))
        ok = d_oof < 1e-6 and d_auc < 1e-6
        repro[target] = {"max_abs_oof_diff": d_oof, "max_abs_fold_auc_diff": d_auc, "reproduced": ok}
        log(f"  Baseline reproduction: OOF diff {d_oof:.1e}, fold AUC diff {d_auc:.1e} -> "
            f"{'REPRODUCED' if ok else 'NOT REPRODUCED'}")
        if not ok:
            notes.append(f"{target}: baseline did not reproduce saved results.")

        # summaries, bootstrap and paired differences
        est, per = {}, {}
        for cand in CANDIDATES:
            est[cand], per[cand] = repeat_avg(y, P[cand])
            for r, m in enumerate(per[cand]):
                rep_rows.append({"target": target, "candidate": cand, "repeat": r, **m})
        boot = {cand: {k: np.empty(N_BOOT) for k in METRICS} for cand in CANDIDATES}
        for b in range(N_BOOT):
            idx = boot_idx[b]
            for cand in CANDIDATES:
                e, _ = repeat_avg(y[idx], P[cand][idx])
                for k in METRICS:
                    boot[cand][k][b] = e[k]
        for cand, (fset, est_obj, purpose) in CANDIDATES.items():
            row = {"target": target, "candidate": cand, "feature_set": fset,
                   "n_inputs": len(feature_sets[fset]), "purpose": purpose}
            for k in METRICS:
                lo, hi, n_ok = pct_ci(boot[cand][k])
                rep_vals = np.array([d[k] for d in per[cand]], dtype=float)
                row.update({k: est[cand][k], f"{k}_ci95_low": lo, f"{k}_ci95_high": hi,
                            f"{k}_repeat_sd": float(np.nanstd(rep_vals, ddof=1)),
                            f"{k}_valid_boot": n_ok})
            comp_rows.append(row)
            if cand == "baseline":
                continue
            for k in METRICS:
                d_boot = boot[cand][k] - boot["baseline"][k]
                lo, hi, n_ok = pct_ci(d_boot)
                d_rep = np.array([per[cand][r][k] - per["baseline"][r][k] for r in range(tp.N_REPEATS)])
                fb = np.array([f[k] for f in fold_rows if f["target"] == target and f["candidate"] == "baseline"])
                fc = np.array([f[k] for f in fold_rows if f["target"] == target and f["candidate"] == cand])
                d_fold = fc - fb
                if lo > 0:
                    verdict = "worse" if k in LOWER_IS_BETTER else "higher"
                elif hi < 0:
                    verdict = "better" if k in LOWER_IS_BETTER else "lower"
                else:
                    verdict = "no clear difference"
                diff_rows.append({
                    "target": target, "candidate": cand, "metric": k,
                    "baseline": est["baseline"][k], "candidate_value": est[cand][k],
                    "difference": est[cand][k] - est["baseline"][k],
                    "diff_ci95_low": lo, "diff_ci95_high": hi, "valid_bootstrap": n_ok,
                    "interval_verdict": verdict,
                    "repeats_candidate_higher": int(np.sum(d_rep > 0)), "n_repeats": len(d_rep),
                    "folds_candidate_higher": int(np.nansum(d_fold > 0)), "n_folds": len(d_fold),
                    "fold_diff_mean": float(np.nanmean(d_fold)), "fold_diff_sd": float(np.nanstd(d_fold, ddof=1))})
        for cand in CANDIDATES:
            log(f"  {cand:24s} ROC-AUC {est[cand]['roc_auc']:.3f} | PR-AUC {est[cand]['pr_auc']:.3f} | "
                f"Brier {est[cand]['brier']:.3f} | sens {est[cand]['sensitivity']:.2f} "
                f"spec {est[cand]['specificity']:.2f}")

    # ------------------------------------------------------------------ warnings
    warn_df = pd.DataFrame(warn_rows, columns=["target", "candidate", "outer_fold", "category", "message"])
    warn_df.to_csv(OUT / "fit_warnings.csv", index=False)
    n_conv = int((warn_df.category == ConvergenceWarning.__name__).sum()) if len(warn_df) else 0
    n_unknown = int(warn_df.message.str.contains("unknown categories").sum()) if len(warn_df) else 0
    n_other = len(warn_df) - n_conv - n_unknown
    if n_conv:
        notes.append(f"{n_conv} ConvergenceWarning(s); see fit_warnings.csv.")
    if n_unknown:
        g = warn_df[warn_df.message.str.contains("unknown categories")].groupby(["target", "candidate"]).size()
        notes.append(f"{n_unknown} 'unknown category' warning(s) (a rare level absent from a training "
                     f"fold, encoded as zeros; not leakage): " +
                     ", ".join(f"{t}/{c} x{n}" for (t, c), n in g.items()))
    if n_other:
        notes.append(f"{n_other} other warning(s); see fit_warnings.csv.")

    # ------------------------------------------------------------------ files
    pd.DataFrame(oof_rows).to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "repeat_metrics.csv", index=False)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "model_comparison.csv", index=False)
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(OUT / "paired_differences_vs_baseline.csv", index=False)
    run_config = {
        "script": "scripts/test_vessel_models.py", "python": platform.python_version(),
        "scikit_learn": sklearn.__version__, "targets": TARGETS,
        "outer_cv": {"type": "RepeatedStratifiedKFold", "n_splits": tp.N_SPLITS,
                     "n_repeats": tp.N_REPEATS, "random_state": tp.SEED, "stratified_on": "each target"},
        "threshold": THRESHOLD,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "patient (5 OOF predictions kept together)",
                      "statistic": "repeat-averaged pooled OOF metric", "paired": True},
        "candidates": {c: {"feature_set": f, "estimator": "cloned from deployed baseline" if e is None else " ".join(str(e).split()),
                           "purpose": p} for c, (f, e, p) in CANDIDATES.items()},
        "feature_sets": feature_sets, "forbidden_input_columns": list(FORBIDDEN),
        "labels": {t: {"positive": config["tasks"][t]["positive_label"],
                       "negative": config["tasks"][t]["negative_label"]} for t in TARGETS},
        "baseline_reproduction": repro,
        "warnings": {"convergence": n_conv, "unknown_category": n_unknown, "other": n_other},
        "input_sha256": {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): h for p, h in hashes.items()},
    }
    (OUT / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ report
    def ci_txt(r, k):
        return f"{r[k]:.3f} ({r[k + '_ci95_low']:.3f}–{r[k + '_ci95_high']:.3f})"

    L = ["# Experiment 2: LCX and RCA candidate models vs the clinical baseline", "",
         "> **Educational prototype. Not a diagnostic tool.** Internal repeated cross-validation on "
         "303 patients from one centre. This is not external or clinical validation, and nothing "
         "here changes the deployed models.", "",
         "## Protocol", "",
         f"- **Outer validation:** RepeatedStratifiedKFold, {tp.N_SPLITS} folds × {tp.N_REPEATS} "
         f"repeats, `random_state={tp.SEED}`. The splits are created separately for each target and "
         "stratified on it; within a target every candidate uses identical splits.",
         "- **Preprocessing:** `train_phase1.make_pipeline`, refitted inside every training fold. "
         "No hyperparameter was tuned; the candidates below were fixed before any result was seen.",
         f"- **Inputs:** the project's existing feature sets, `clinical` "
         f"({len(feature_sets['clinical'])}) and `reduced` ({len(feature_sets['reduced'])}, the "
         "inference schema). `Cath`, `LAD`, `LCX` and `RCA` are never inputs (checked by the script).",
         "- **Labels:** 1 = `Stenotic`, 0 = `Normal`. Threshold 0.5 for the threshold-based metrics.",
         f"- **Uncertainty:** {N_BOOT}-replicate patient-level bootstrap (seed {BOOT_SEED}). Each "
         "resampled patient keeps all 5 out-of-fold predictions together; differences from the "
         "baseline use the same resamples (paired).", "",
         "| Candidate | Inputs | Settings | Purpose |", "|---|---|---|---|"]
    for c, (f, e, p) in CANDIDATES.items():
        L.append(f"| `{c}` | {f} ({len(feature_sets[f])}) | "
                 f"{'cloned from deployed file: LogisticRegression(C=1.0, max_iter=5000)' if e is None else ' '.join(str(e).split())} | {p} |")
    L += ["", "**Baseline reproduction:** " + "; ".join(
        f"{t}: {'REPRODUCED' if v['reproduced'] else 'NOT REPRODUCED'} (OOF diff {v['max_abs_oof_diff']:.1e}, "
        f"fold AUC diff {v['max_abs_fold_auc_diff']:.1e})" for t, v in repro.items()), ""]
    for target in TARGETS:
        L += [f"## {target}", "",
              "Values are repeat-averaged out-of-fold metrics with 95% bootstrap intervals.", "",
              "| Candidate | ROC-AUC | PR-AUC | Brier | Log loss | Sens. | Spec. | Precision | F1 | Accuracy |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in comp[comp.target == target].to_dict("records"):
            L.append(f"| `{r['candidate']}` | {ci_txt(r, 'roc_auc')} | {ci_txt(r, 'pr_auc')} | "
                     f"{ci_txt(r, 'brier')} | {r['log_loss']:.3f} | {r['sensitivity']:.2f} | "
                     f"{r['specificity']:.2f} | {r['precision']:.2f} | {r['f1']:.2f} | {r['accuracy']:.2f} |")
        L += ["", "**Paired difference vs baseline** (candidate − baseline; 95% bootstrap CI; "
              "number of the 5 repeats where the candidate scored higher):", "",
              "| Candidate | ROC-AUC | PR-AUC | Brier (lower is better) | Log loss (lower is better) | Sensitivity | Specificity |",
              "|---|---|---|---|---|---|---|"]
        for cand in [c for c in CANDIDATES if c != "baseline"]:
            cells = []
            for k in ("roc_auc", "pr_auc", "brier", "log_loss", "sensitivity", "specificity"):
                d = diffs[(diffs.target == target) & (diffs.candidate == cand) & (diffs.metric == k)].iloc[0]
                cells.append(f"{d.difference:+.3f} ({d.diff_ci95_low:+.3f} to {d.diff_ci95_high:+.3f}); "
                             f"{d.repeats_candidate_higher}/5")
            L.append(f"| `{cand}` | " + " | ".join(cells) + " |")
        L.append("")
        auc_d = diffs[(diffs.target == target) & (diffs.metric == "roc_auc")]
        clear_up = auc_d[auc_d.diff_ci95_low > 0]
        clear_down = auc_d[auc_d.diff_ci95_high < 0]
        L.append("**ROC-AUC, read from the intervals:** " + (
            "candidate(s) with an interval entirely above 0: " +
            ", ".join(f"`{c}` ({d:+.3f})" for c, d in zip(clear_up.candidate, clear_up.difference))
            if len(clear_up) else "no candidate has an ROC-AUC difference interval entirely above 0.") + (
            " Interval entirely below 0 (worse discrimination): " +
            ", ".join(f"`{c}` ({d:+.3f})" for c, d in zip(clear_down.candidate, clear_down.difference)) + "."
            if len(clear_down) else ""))
        L.append("")
    L += ["## How to read this, and its limits", "",
          "- **Discrimination vs calibration vs threshold.** ROC-AUC and PR-AUC measure ranking. "
          "Brier and log loss measure probability accuracy. Sensitivity, specificity, precision, F1 "
          "and accuracy depend on the 0.5 threshold.",
          "- **Class weighting** (`lr_balanced_clinical`) deliberately pushes probabilities up for "
          "the minority (Stenotic) class. Expect more sensitivity, less specificity and worse "
          "calibration at 0.5, even with similar ROC-AUC. That's an operating-point change, not "
          "better discrimination.",
          "- **Selection optimism.** Four new candidates per target (eight comparisons) were scored "
          "on the same folds used to judge them, on top of the six Phase 1 candidates already "
          "compared on these folds. If you pick whichever looks best here, its estimate is biased "
          "upwards. A small positive difference with an interval crossing 0 is not evidence of "
          "improvement. Adopting a candidate would need confirmation, ideally by nested "
          "cross-validation.",
          "- **What the intervals capture.** They reflect only which patients were evaluated. They "
          "don't include retraining variability or the selection steps above, so treat them as "
          "lower bounds on the uncertainty. There are no corrections for multiple comparisons.",
          "- **Small sample.** LCX has 119 and RCA 114 positives. Several categorical levels in the "
          "48-input set occur in 5–15 patients (`rare_categories.csv`), so some training folds miss "
          "them entirely.",
          "- **Coherence.** These are independent vessel models; any change could alter agreement "
          "with the CAD model.", "",
          "## Warnings during fitting", ""]
    L += [f"- {n}" for n in notes] if notes else ["- None."]
    L += ["", "All warnings are listed in full in `fit_warnings.csv`; none were suppressed.", "",
          "## Files", "",
          "`model_comparison.csv` (all metrics with CIs and repeat SDs), "
          "`paired_differences_vs_baseline.csv` (every metric, with CI, repeat and fold win counts), "
          "`repeat_metrics.csv`, `fold_metrics.csv`, `oof_predictions.csv` (target × candidate × "
          "patient × repeat, with outer fold IDs), `rare_categories.csv`, `fit_warnings.csv`, "
          "`run_config.json` (seeds, settings, feature sets, input file hashes) and "
          "`vessel_experiment_log.txt`.", "",
          "**No candidate is adopted by this script.** The decision is left to review of these results."]
    (OUT / "vessel_experiment_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    changed = [str(p.relative_to(PROJECT_ROOT)) for p in protected if sha256_of(p) != hashes[p]]
    log("\n=== SUMMARY: ROC-AUC difference vs baseline (95% bootstrap CI) ===")
    for r in diffs[diffs.metric == "roc_auc"].itertuples():
        log(f"  {r.target} {r.candidate:24s} {r.difference:+.3f} ({r.diff_ci95_low:+.3f} to "
            f"{r.diff_ci95_high:+.3f}) -> {r.interval_verdict}")
    log(f"\nBaseline reproduced: {all(v['reproduced'] for v in repro.values())}")
    log(f"Warnings: {'; '.join(notes) if notes else 'none'}")
    log(f"Protected files checked: {len(protected)}; changed: {changed if changed else 'none'}")
    log(f"Outputs: {OUT}")
    log(f"Total time {time.time() - started:.1f}s")
    log("VESSEL MODEL EXPERIMENT COMPLETE (no candidate adopted; deployed models unchanged)")
    log.close()
    if changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
