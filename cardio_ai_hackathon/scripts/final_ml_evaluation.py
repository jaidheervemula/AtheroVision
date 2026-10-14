"""
final_ml_evaluation.py
======================
Final evaluation of the four SELECTED models (Cath, LAD, LCX, RCA):
  A. calibration report and threshold trade-offs,
  B. patient-level bootstrap 95% confidence intervals,
  C. row 93 sensitivity analysis.

EDUCATIONAL DECISION-SUPPORT PROTOTYPE. NOT A DIAGNOSTIC TOOL. All results are internal
cross-validation estimates on 303 patients from one centre; nothing here is external or
clinical validation.

Why some fitting is needed (documented, minimal)
  reports/phase1_modeling/oof_predictions.csv stores only each patient's MEAN and SD over
  the 5 CV repeats, not the 5 individual out-of-fold (OOF) predictions. Parts B and C need
  those per-repeat predictions, so this script re-runs the EXISTING Phase 1 protocol for
  the four selected models only:
    * same splits: RepeatedStratifiedKFold(5 folds, 5 repeats, seed 42) per task;
    * same preprocessing builder: train_phase1.make_pipeline (imported, not copied);
    * same estimator settings: cloned from each saved deployment pipeline.
  Before anything is used, the regenerated per-repeat predictions are checked against the
  saved OOF means and the saved per-fold ROC-AUCs. Models fitted here exist only in memory:
  nothing in models/ or reports/phase1_modeling/ is written or changed.
  Fits: 4 tasks x 25 folds (primary) + 4 tasks x 25 folds (row 93 excluded) = 200.

Outputs (new files in reports/final_ml_evaluation/)
  calibration_report.md, threshold_tradeoffs.csv, bootstrap_confidence_intervals.csv,
  calibration_summary.csv, row93_sensitivity.csv, row93_predictions.csv,
  oof_per_repeat_selected.csv, calibration_<task>.svg, final_ml_evaluation_log.txt

Run from the project folder (.venv active):
    python scripts\\final_ml_evaluation.py
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_phase1 as tp  # noqa: E402  (only its helpers and settings are used)

CONFIG = PROJECT_ROOT / "config" / "model_selection.json"
METADATA = PROJECT_ROOT / "models" / "phase1_candidates" / "metadata.json"
P1 = PROJECT_ROOT / "reports" / "phase1_modeling"
OUT = PROJECT_ROOT / "reports" / "final_ml_evaluation"

TASKS = ["Cath", "LAD", "LCX", "RCA"]
INCONSISTENT_ROW = 93
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
DEFAULT_THRESHOLD = 0.5
N_BOOT = 2000
BOOT_SEED = 20261010
N_CAL_BINS = 10


# ----------------------------------------------------------------------------- helpers
class Log:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, text=""):
        print(text)
        self.f.write(str(text) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def calibration_intercept(y, p):
    """Calibration-in-the-large: a in logit(P(y=1)) = a + logit(p), slope fixed at 1."""
    lp, a = logit(p), 0.0
    for _ in range(50):
        mu = sigmoid(a + lp)
        step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def calibration_slope(y, p):
    """Logistic recalibration logit(P(y=1)) = a + b*logit(p); returns b (ideal = 1)."""
    lp = logit(p)
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(100):
        mu = sigmoid(X @ beta)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return float("nan")
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[1]) if np.all(np.isfinite(beta)) else float("nan")


def threshold_metrics(y, p, t):
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")      # undefined if nobody flagged
    npv = tn / (tn + fn) if tn + fn else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (tp + fp) and (prec + sens) else (
        0.0 if (tp + fp) else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_flagged_positive": tp + fp,
            "sensitivity": sens, "specificity": spec, "precision": prec, "npv": npv, "f1": f1,
            "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else float("nan")}


def metric_set(y, p):
    """Metrics for ONE set of predictions (one per patient). NaN when undefined."""
    two = len(np.unique(y)) == 2
    m = threshold_metrics(y, p, DEFAULT_THRESHOLD)
    return {
        "roc_auc": roc_auc_score(y, p) if two else float("nan"),
        "pr_auc": average_precision_score(y, p) if two else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "sensitivity": m["sensitivity"], "specificity": m["specificity"],
        "precision": m["precision"], "f1": m["f1"], "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "calibration_intercept": calibration_intercept(y, p) if two else float("nan"),
        "calibration_slope": calibration_slope(y, p) if two else float("nan"),
    }


METRIC_NAMES = list(metric_set(np.array([0, 1, 0, 1]), np.array([.2, .8, .4, .6])).keys())


def repeat_averaged(y, P):
    """Primary estimand: compute each metric on one CV repeat's OOF predictions (one per
    patient), then average over the 5 repeats. Repeats with an undefined value are skipped."""
    per_repeat = [metric_set(y, P[:, r]) for r in range(P.shape[1])]
    return {k: (float(np.nanmean(v)) if not np.all(np.isnan(v := np.array([d[k] for d in per_repeat])))
                else float("nan")) for k in METRIC_NAMES}, per_repeat


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_svg(task, label, bins, stats, path):
    """Dependency-free SVG reliability plot with a labelled y = x reference line."""
    W, H, L, R, T, B = 460, 500, 64, 20, 54, 96
    pw, ph = W - L - R, H - T - B
    sx = lambda v: L + v * pw                       # noqa: E731
    sy = lambda v: T + (1 - v) * ph                 # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Arial, sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">'
             f'Calibration: {task} ({label})</text>',
             f'<text x="{W / 2}" y="38" text-anchor="middle" fill="#555">Out-of-fold, 303 patients, '
             f'{len(bins)} equal-count bins. Not a patient-level result.</text>']
    for v in np.linspace(0, 1, 6):
        parts.append(f'<line x1="{sx(v)}" y1="{sy(0)}" x2="{sx(v)}" y2="{sy(1)}" stroke="#eee"/>')
        parts.append(f'<line x1="{sx(0)}" y1="{sy(v)}" x2="{sx(1)}" y2="{sy(v)}" stroke="#eee"/>')
        parts.append(f'<text x="{sx(v)}" y="{sy(0) + 16}" text-anchor="middle">{v:.1f}</text>')
        parts.append(f'<text x="{sx(0) - 8}" y="{sy(v) + 4}" text-anchor="end">{v:.1f}</text>')
    parts.append(f'<rect x="{sx(0)}" y="{sy(1)}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>')
    parts.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="#888" '
                 f'stroke-dasharray="6,4" stroke-width="1.5"/>')
    lx, ly = sx(0.03), sy(0.97)
    parts.append(f'<rect x="{lx}" y="{ly}" width="190" height="42" fill="white" stroke="#ccc"/>')
    parts.append(f'<line x1="{lx + 8}" y1="{ly + 13}" x2="{lx + 34}" y2="{ly + 13}" stroke="#888" '
                 f'stroke-dasharray="6,4" stroke-width="1.5"/>')
    parts.append(f'<text x="{lx + 40}" y="{ly + 17}">Ideal calibration (y = x)</text>')
    parts.append(f'<circle cx="{lx + 21}" cy="{ly + 31}" r="4.5" fill="#1f5fa8"/>')
    parts.append(f'<text x="{lx + 40}" y="{ly + 35}">Bin, with 95% Wilson interval</text>')
    pts = []
    for b in bins:
        x, y0, lo, hi = b["mean_predicted"], b["observed_rate"], b["ci_low"], b["ci_high"]
        parts.append(f'<line x1="{sx(x)}" y1="{sy(lo)}" x2="{sx(x)}" y2="{sy(hi)}" stroke="#1f5fa8" stroke-width="1.5"/>')
        pts.append(f"{sx(x):.1f},{sy(y0):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#1f5fa8" stroke-width="1.5"/>')
    for b in bins:
        parts.append(f'<circle cx="{sx(b["mean_predicted"]):.1f}" cy="{sy(b["observed_rate"]):.1f}" '
                     f'r="4.5" fill="#1f5fa8"/>')
    parts.append(f'<text x="{W / 2}" y="{H - 34}" text-anchor="middle">Mean predicted probability '
                 f'(average of 5 OOF repeats per patient)</text>')
    parts.append(f'<text x="16" y="{T + ph / 2}" text-anchor="middle" '
                 f'transform="rotate(-90 16 {T + ph / 2})">Observed proportion positive</text>')
    parts.append(f'<text x="{W / 2}" y="{H - 14}" text-anchor="middle" fill="#333" font-size="11">'
                 f'Patient-mean probabilities: Brier {stats["brier"]:.3f}, slope '
                 f'{stats["calibration_slope"]:.2f}, intercept {stats["calibration_intercept"]:+.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run_cv(X, y, features, numeric_set, estimator):
    """Phase 1 CV protocol for one model; returns per-repeat OOF matrix and fold AUCs."""
    cv = RepeatedStratifiedKFold(n_splits=tp.N_SPLITS, n_repeats=tp.N_REPEATS, random_state=tp.SEED)
    num = [c for c in features if c in numeric_set]
    cat = [c for c in features if c not in numeric_set]
    P = np.full((len(y), tp.N_REPEATS), np.nan)
    fold_auc = []
    for fold_id, (tr, te) in enumerate(cv.split(X, y)):
        pipe = tp.make_pipeline(estimator, num, cat)
        pipe.fit(X.iloc[tr][features], y[tr])
        proba = pipe.predict_proba(X.iloc[te][features])[:, 1]
        P[te, fold_id // tp.N_SPLITS] = proba
        fold_auc.append(roc_auc_score(y[te], proba))
    if np.isnan(P).any():
        raise RuntimeError("Some patients did not receive an OOF prediction in every repeat.")
    return P, np.array(fold_auc)


# ----------------------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / "final_ml_evaluation_log.txt")
    started = time.time()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    data_path = PROJECT_ROOT / "data" / "raw" / config["data_file"]
    protected = [data_path, CONFIG, METADATA] + sorted(P1.glob("*")) + sorted(
        (PROJECT_ROOT / "models" / "phase1_candidates").glob("*.joblib"))
    hashes_before = {p: sha256_of(p) for p in protected}
    if hashes_before[data_path] != config["data_sha256"]:
        raise SystemExit("Dataset hash does not match the configuration; stopping.")

    df = pd.read_excel(data_path, sheet_name=0)
    oof_saved = pd.read_csv(P1 / "oof_predictions.csv")
    folds_saved = pd.read_csv(P1 / "cv_fold_metrics.csv")
    numeric_set = set(meta["numeric_features"])
    log("final_ml_evaluation.py | educational prototype, internal CV only")
    log(f"Protocol reused from train_phase1.py: RepeatedStratifiedKFold(n_splits={tp.N_SPLITS}, "
        f"n_repeats={tp.N_REPEATS}, random_state={tp.SEED}); threshold {DEFAULT_THRESHOLD}")
    log(f"Row {INCONSISTENT_ROW} as recorded (unchanged): "
        f"{df.loc[INCONSISTENT_ROW, TASKS].to_dict()}")

    per_repeat_rows, thr_rows, cal_rows, boot_rows, sens_rows, r93_rows = [], [], [], [], [], []
    report = {}
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(df), size=(N_BOOT, len(df)))   # same resamples for all tasks

    for task in TASKS:
        spec = config["tasks"][task]
        y = (df[task].astype(str).str.strip() == spec["positive_label"]).astype(int).to_numpy()
        features = spec["input_features"]
        deployed = joblib.load(PROJECT_ROOT / spec["artifact"])
        estimator = clone(deployed.named_steps["model"])           # identical settings, unfitted
        X = df[features].copy()
        log(f"\n=== {task}: {spec['model_name']} ({spec['model_type']}), {len(features)} inputs ===")

        # ---- regenerate per-repeat OOF and verify against saved Phase 1 outputs
        P, fold_auc = run_cv(X, y, features, numeric_set, estimator)
        saved = oof_saved[(oof_saved.task == task) & (oof_saved.model == spec["model_name"])]
        saved = saved.sort_values("row_index")
        diff_mean = float(np.max(np.abs(P.mean(axis=1) - saved.oof_proba_mean.to_numpy())))
        saved_auc = folds_saved[(folds_saved.task == task) & (folds_saved.model == spec["model_name"])
                                ].sort_values("fold").roc_auc.to_numpy()
        diff_auc = float(np.max(np.abs(fold_auc - saved_auc)))
        labels_ok = np.array_equal(saved.y_true.to_numpy(), y)
        log(f"  Reproduction check: max |regenerated mean OOF - saved| = {diff_mean:.2e}; "
            f"max |fold AUC - saved| = {diff_auc:.2e}; labels match = {labels_ok}")
        if diff_mean > 1e-6 or diff_auc > 1e-6 or not labels_ok:
            raise SystemExit(f"{task}: regenerated predictions do not reproduce Phase 1. Stopping.")
        for i in range(len(df)):
            for r in range(tp.N_REPEATS):
                per_repeat_rows.append({"row_index": i, "task": task, "model": spec["model_name"],
                                        "repeat": r, "y_true": int(y[i]), "oof_proba": P[i, r]})

        # ---- A. calibration (patient-mean probability; repeat-level summaries alongside)
        pmean = P.mean(axis=1)
        pooled = metric_set(y, pmean)
        rep_avg, per_rep = repeat_averaged(y, P)
        order = np.argsort(pmean, kind="mergesort")
        bins = []
        for chunk in np.array_split(order, N_CAL_BINS):
            k, n = int(y[chunk].sum()), len(chunk)
            lo, hi = wilson(k, n)
            bins.append({"task": task, "bin": len(bins) + 1, "n_patients": n,
                         "mean_predicted": float(pmean[chunk].mean()), "observed_rate": k / n,
                         "ci_low": lo, "ci_high": hi})
        calibration_svg(task, spec["model_name"], bins, pooled, OUT / f"calibration_{task}.svg")
        cal_rows.append({"task": task, "model": spec["model_name"], "prevalence": y.mean(),
                         "mean_predicted_patient_mean": pmean.mean(),
                         "brier_patient_mean": pooled["brier"],
                         "calibration_intercept_patient_mean": pooled["calibration_intercept"],
                         "calibration_slope_patient_mean": pooled["calibration_slope"],
                         "brier_repeat_avg": rep_avg["brier"],
                         "calibration_intercept_repeat_avg": rep_avg["calibration_intercept"],
                         "calibration_slope_repeat_avg": rep_avg["calibration_slope"],
                         "calibration_slope_repeat_min": min(d["calibration_slope"] for d in per_rep),
                         "calibration_slope_repeat_max": max(d["calibration_slope"] for d in per_rep)})

        for t in THRESHOLDS:
            m = threshold_metrics(y, pmean, t)
            thr_rows.append({"task": task, "model": spec["model_name"], "threshold": t,
                             "summary": "patient_mean_probability", **m})
            reps = [threshold_metrics(y, P[:, r], t) for r in range(tp.N_REPEATS)]
            avg = {k: float(np.nanmean([d[k] for d in reps])) if not all(
                math.isnan(d[k]) for d in reps) else float("nan") for k in reps[0]}
            thr_rows.append({"task": task, "model": spec["model_name"], "threshold": t,
                             "summary": "average_over_5_repeats", **avg})

        # ---- B. patient-level bootstrap (patients resampled; their 5 OOF predictions kept together)
        boot = {k: [] for k in METRIC_NAMES}
        for b in range(N_BOOT):
            idx = boot_idx[b]
            est, _ = repeat_averaged(y[idx], P[idx])
            for k in METRIC_NAMES:
                boot[k].append(est[k])
        for k in METRIC_NAMES:
            vals = np.array(boot[k])
            ok = vals[~np.isnan(vals)]
            boot_rows.append({
                "task": task, "model": spec["model_name"], "metric": k,
                "estimate": rep_avg[k],
                "ci95_low": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                "ci95_high": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
                "n_bootstrap": N_BOOT, "n_valid_replicates": int(len(ok)),
                "threshold": DEFAULT_THRESHOLD if k in ("sensitivity", "specificity", "precision",
                                                        "f1", "accuracy", "balanced_accuracy") else "",
                "estimand": "repeat-averaged pooled OOF metric (internal CV, fixed models)"})

        # ---- C. row 93
        keep = np.arange(len(df)) != INCONSISTENT_ROW
        Xs, ys = X.loc[keep].reset_index(drop=True), y[keep]
        Ps, _ = run_cv(Xs, ys, features, numeric_set, estimator)
        no93_eval_only, _ = repeat_averaged(y[keep], P[keep])
        no93_refit, _ = repeat_averaged(ys, Ps)
        rep_sd = {k: float(np.nanstd([d[k] for d in per_rep], ddof=1)) for k in METRIC_NAMES}
        for k in ("roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "calibration_slope"):
            sens_rows.append({
                "task": task, "model": spec["model_name"], "metric": k,
                "primary_all_303": rep_avg[k],
                "primary_preds_evaluated_without_row93": no93_eval_only[k],
                "refit_cv_without_row93": no93_refit[k],
                "difference_refit_minus_primary": no93_refit[k] - rep_avg[k],
                "sd_across_5_repeats_primary": rep_sd[k]})
        in_sample = float(deployed.predict_proba(df.loc[[INCONSISTENT_ROW], features])[0, 1])
        r93_rows.append({
            "task": task, "model": spec["model_name"],
            "recorded_label": df.loc[INCONSISTENT_ROW, task], "y_true": int(y[INCONSISTENT_ROW]),
            **{f"oof_repeat_{r}": P[INCONSISTENT_ROW, r] for r in range(tp.N_REPEATS)},
            "oof_mean": float(P[INCONSISTENT_ROW].mean()),
            "saved_oof_mean": float(saved.oof_proba_mean.to_numpy()[INCONSISTENT_ROW]),
            "oof_predicted_label_at_0_5": spec["positive_label"] if P[INCONSISTENT_ROW].mean() >= 0.5
            else spec["negative_label"],
            "abs_error_oof_mean": abs(int(y[INCONSISTENT_ROW]) - float(P[INCONSISTENT_ROW].mean())),
            "squared_error_oof_mean": (int(y[INCONSISTENT_ROW]) - float(P[INCONSISTENT_ROW].mean())) ** 2,
            "deployed_model_in_sample_probability_NOT_OOF": in_sample})
        report[task] = {"pooled": pooled, "rep_avg": rep_avg, "bins": bins, "spec": spec,
                        "prevalence": float(y.mean()), "n_pos": int(y.sum())}
        log(f"  Repeat-averaged: ROC-AUC {rep_avg['roc_auc']:.3f}, Brier {rep_avg['brier']:.3f}, "
            f"slope {rep_avg['calibration_slope']:.2f} | row-93-excluded refit ROC-AUC "
            f"{no93_refit['roc_auc']:.3f}")

    # ------------------------------------------------------------------ write tables
    pd.DataFrame(per_repeat_rows).to_csv(OUT / "oof_per_repeat_selected.csv", index=False)
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(OUT / "threshold_tradeoffs.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration_summary.csv", index=False)
    pd.DataFrame([b for t in TASKS for b in report[t]["bins"]]).to_csv(OUT / "calibration_bins.csv", index=False)
    ci = pd.DataFrame(boot_rows)
    ci.to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "row93_sensitivity.csv", index=False)
    r93 = pd.DataFrame(r93_rows)
    r93.to_csv(OUT / "row93_predictions.csv", index=False)

    # ------------------------------------------------------------------ write report
    def cell(task, metric):
        r = ci[(ci.task == task) & (ci.metric == metric)].iloc[0]
        return f"{r.estimate:.3f} ({r.ci95_low:.3f}–{r.ci95_high:.3f})"

    L = ["# Final ML evaluation: calibration, thresholds, bootstrap CIs, row 93", "",
         "> **Educational decision-support prototype. Not a diagnostic tool.** Every number "
         "below is an internal cross-validation estimate on 303 patients from one centre. "
         "Nothing here is external or clinical validation.", "",
         "Generated by `scripts/final_ml_evaluation.py`. Selected models: " +
         ", ".join(f"{t} `{report[t]['spec']['model_name']}`" for t in TASKS) + ".", "",
         "## How the predictions were obtained", "",
         "- `reports/phase1_modeling/oof_predictions.csv` stores only each patient's **mean and SD** "
         "of the 5 out-of-fold (OOF) predictions, not the 5 individual predictions.",
         "- To keep each patient's repeated predictions together, this script re-ran the **existing** "
         "Phase 1 CV protocol for the four selected models only (same splits, same "
         "`train_phase1.make_pipeline`, estimator settings cloned from each saved pipeline).",
         "- The models fitted for this were kept in memory only; no saved model or Phase 1 report was changed.",
         "- **Reproduction check passed for every task**: the regenerated predictions match the saved OOF "
         "means and the saved per-fold ROC-AUCs (see `final_ml_evaluation_log.txt`). They are saved in "
         "`oof_per_repeat_selected.csv` (303 patients x 5 repeats per task).",
         "- In total, 200 fits were run: 100 for the primary protocol and 100 for the row 93 exclusion.", "",
         "Two kinds of summary are used and are never mixed:",
         "- **Repeat-averaged** (primary): each metric is computed on one repeat's 303 OOF predictions "
         "(one per patient), then averaged over the 5 repeats.",
         "- **Patient-mean probability**: each patient's 5 predictions are averaged first, then metrics "
         "are computed once. This is used for the calibration plots, because it gives one point per patient.",
         "- In neither case are the 5 predictions treated as 5 patients.", "",
         "## A. Calibration", "",
         "| Task | Prevalence | Mean predicted | Brier | Calibration intercept | Calibration slope | Plot |",
         "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        c = [r for r in cal_rows if r["task"] == t][0]
        L.append(f"| {t} | {c['prevalence']:.3f} | {c['mean_predicted_patient_mean']:.3f} | "
                 f"{c['brier_repeat_avg']:.3f} | {c['calibration_intercept_repeat_avg']:+.2f} | "
                 f"{c['calibration_slope_repeat_avg']:.2f} (repeats {c['calibration_slope_repeat_min']:.2f}"
                 f"–{c['calibration_slope_repeat_max']:.2f}) | ![{t}](calibration_{t}.svg) |")
    L += ["", "Brier, intercept and slope are repeat-averaged. Intercept: 0 is ideal "
          "(calibration-in-the-large, slope fixed at 1). Slope: 1 is ideal. Below 1 means predictions "
          "are too extreme; above 1 means they are squeezed towards the middle. 95% intervals are in "
          "`bootstrap_confidence_intervals.csv`.", "",
          "Each plot sorts patients by mean predicted probability into 10 groups of about 30, and shows "
          "each group's observed proportion with a 95% Wilson interval against the dashed y = x line. "
          "With about 30 patients per group the intervals are wide, so small departures from the line "
          "are not meaningful.", ""]
    for t in TASKS:
        r = ci[(ci.task == t) & (ci.metric == "calibration_slope")].iloc[0]
        s, lo, hi = r.estimate, r.ci95_low, r.ci95_high
        rng_txt = f"slope {s:.2f} (95% CI {lo:.2f}–{hi:.2f})"
        if lo > 1:
            L.append(f"- **{t}**: {rng_txt}. The probabilities are clearly compressed towards the "
                     "middle (under-confident). Do not read them as risk estimates at face value.")
        elif hi < 1:
            L.append(f"- **{t}**: {rng_txt}. The probabilities are clearly too extreme (over-confident).")
        elif s < 0.9:
            L.append(f"- **{t}**: {rng_txt}. This leans towards over-confidence (predictions somewhat "
                     "too extreme), but the interval includes 1, so the data cannot confirm it.")
        elif s > 1.1:
            L.append(f"- **{t}**: {rng_txt}. This leans towards under-confidence, but the interval includes 1.")
        else:
            L.append(f"- **{t}**: {rng_txt}. Close to 1; no clear miscalibration at this sample size.")
    L.append("- Patient-mean probabilities (used in the plots) average 5 predictions, which pulls "
             "extreme values towards the middle. Their slope therefore differs slightly from the "
             "repeat-averaged slope in the table (see `calibration_summary.csv`).")
    L += ["", "The deployed models were refitted on all 303 patients, so their calibration cannot be "
          "measured on these data. The CV models' calibration shown here is the best available proxy.", "",
          "## Threshold trade-offs", "",
          "`threshold_tradeoffs.csv` has thresholds 0.05–0.95 in steps of 0.05, two rows per threshold:",
          "- `patient_mean_probability`: integer confusion-matrix counts;",
          "- `average_over_5_repeats`: counts are averages over repeats, so they can be fractional.",
          "",
          "Columns: TP, FP, FN, TN, number flagged positive, sensitivity (recall), specificity, precision, "
          "NPV, F1, accuracy and balanced accuracy. Selected rows (patient-mean probability):", "",
          "| Task | Threshold | Sensitivity | Specificity | Precision | F1 | Accuracy | TP/FP/FN/TN |",
          "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        for th in (0.3, 0.5, 0.7):
            r = thr[(thr.task == t) & (thr.threshold == th) & (thr.summary == "patient_mean_probability")].iloc[0]
            L.append(f"| {t} | {th} | {r.sensitivity:.2f} | {r.specificity:.2f} | {r.precision:.2f} | "
                     f"{r.f1:.2f} | {r.accuracy:.2f} | {int(r.tp)}/{int(r.fp)}/{int(r.fn)}/{int(r.tn)} |")
    L += ["", "**Limitations of choosing a threshold here**",
          "- These are the same OOF predictions used to judge the models, and the models were selected "
          "on them. Picking the threshold that looks best and reporting its performance on the same data "
          "is optimistic.",
          "- No threshold in this table is clinically optimal or validated. The app keeps the documented "
          "default of 0.5 and shows the probability.",
          "- Precision is undefined (blank) when no patient is flagged positive.", "",
          "## B. Bootstrap 95% confidence intervals", "",
          "| Task | ROC-AUC | PR-AUC | Brier | Sensitivity @0.5 | Specificity @0.5 | Precision @0.5 | F1 @0.5 |",
          "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        L.append(f"| {t} | {cell(t, 'roc_auc')} | {cell(t, 'pr_auc')} | {cell(t, 'brier')} | "
                 f"{cell(t, 'sensitivity')} | {cell(t, 'specificity')} | {cell(t, 'precision')} | "
                 f"{cell(t, 'f1')} |")
    invalid = ci[ci.n_valid_replicates < N_BOOT]
    L += ["", f"**Resampled:** patients, with replacement, {N_BOOT} times (seed {BOOT_SEED}; the same "
          "resamples for all four tasks). Each resampled patient keeps all 5 OOF predictions together. "
          "In each resample the repeat-averaged metric is recomputed; the interval is the 2.5th–97.5th "
          "percentile.", "",
          "**What the intervals describe:** uncertainty from *which patients* were evaluated, for the "
          "repeat-averaged OOF performance of these fixed model specifications in this dataset.", "",
          "**What they do NOT include:**",
          "- uncertainty from retraining (the CV fits were not repeated inside the bootstrap);",
          "- the optimism from selecting these models among six candidates on the same folds, which "
          "only nested CV could estimate;",
          "- any shift to other hospitals or populations.",
          "",
          "Read them as a lower bound on the true uncertainty.", "",
          "**Undefined values:** a metric can be undefined in a resample, for example AUC with one class "
          "only, or precision when nobody is flagged. Such repeats are skipped; a replicate with no valid "
          "repeat is dropped. Valid replicates out of " + str(N_BOOT) + ": " +
          ("all metrics had all replicates valid." if invalid.empty else
           "; ".join(f"{r.task} {r.metric} {r.n_valid_replicates}" for r in invalid.itertuples())) + "", "",
          "**Small-sample notes:** LCX and RCA have 119 and 114 positives, and Cath has only 87 "
          "negatives, so specificity for Cath and sensitivity/precision for LCX and RCA rest on few "
          "patients and have wide intervals.", "",
          "## C. Row 93 sensitivity analysis", "",
          "Row 93 is recorded as Cath = Normal, LAD = Stenotic, LCX = Normal, RCA = Normal. This "
          "contradicts the dataset's own rule that CAD means at least one stenotic vessel. "
          "**It is kept unchanged in the primary analysis.** It is not relabelled anywhere; it is only "
          "left out of the clearly labelled sensitivity run.", "",
          "**OOF predictions for row 93**",
          "",
          "Each value is the mean of 5 repeats, and each repeat comes from a model that never saw this "
          "patient. The 'deployed in-sample' column is NOT an OOF value: the deployed model was "
          "trained on this patient.", "",
          "| Task | Recorded label | OOF mean (5 repeats) | OOF repeat range | OOF label @0.5 | Abs. error | Deployed in-sample (not OOF) |",
          "|---|---|---|---|---|---|---|"]
    for r in r93_rows:
        reps = [r[f"oof_repeat_{i}"] for i in range(tp.N_REPEATS)]
        L.append(f"| {r['task']} | {r['recorded_label']} | {r['oof_mean']:.3f} | {min(reps):.3f}–{max(reps):.3f} | "
                 f"{r['oof_predicted_label_at_0_5']} | {r['abs_error_oof_mean']:.3f} | "
                 f"{r['deployed_model_in_sample_probability_NOT_OOF']:.3f} |")
    L += ["", "**Effect of excluding row 93** (repeat-averaged ROC-AUC and Brier):", "",
          "| Task | Primary (303) | Primary preds, row 93 not scored (302) | CV refit without row 93 (302) | Refit minus primary | SD across repeats (primary) |",
          "|---|---|---|---|---|---|"]
    for t in TASKS:
        for k in ("roc_auc", "brier"):
            r = sens[(sens.task == t) & (sens.metric == k)].iloc[0]
            L.append(f"| {t} {k} | {r.primary_all_303:.4f} | {r.primary_preds_evaluated_without_row93:.4f} | "
                     f"{r.refit_cv_without_row93:.4f} | {r.difference_refit_minus_primary:+.4f} | "
                     f"{r.sd_across_5_repeats_primary:.4f} |")
    L += ["", "The middle column only stops scoring row 93, without changing any model, which "
          "isolates its effect as a test case. The refit column also removes it from training, but "
          "with 302 patients the CV splits are reshuffled. Differences about the size of the "
          "repeat-to-repeat SD are therefore within ordinary CV noise. Full table: "
          "`row93_sensitivity.csv`.", "",
          "## Files", "",
          "`calibration_report.md` (this file), `threshold_tradeoffs.csv`, `bootstrap_confidence_intervals.csv`, "
          "`calibration_summary.csv`, `calibration_bins.csv`, `calibration_<task>.svg`, `row93_sensitivity.csv`, "
          "`row93_predictions.csv`, `oof_per_repeat_selected.csv`, `final_ml_evaluation_log.txt`."]
    (OUT / "calibration_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    changed = [str(p.relative_to(PROJECT_ROOT)) for p in protected if sha256_of(p) != hashes_before[p]]
    log(f"\nProtected files checked: {len(protected)}; changed: {changed if changed else 'none'}")
    log(f"Total time {time.time() - started:.1f}s. Outputs in {OUT}")
    log("FINAL ML EVALUATION COMPLETE" if not changed else "WARNING: protected files changed")
    log.close()


if __name__ == "__main__":
    main()
