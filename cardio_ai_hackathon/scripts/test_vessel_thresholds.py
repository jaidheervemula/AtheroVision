"""
test_vessel_thresholds.py
=========================
Final targeted experiment: does choosing the LCX/RCA decision threshold inside each
training fold change the sensitivity/specificity trade-off compared with the fixed 0.5?

EDUCATIONAL PROTOTYPE. Internal repeated cross-validation on 303 patients from one centre.
The selection objective below is an EXPERIMENTAL choice, not a clinically justified one, and
no threshold found here is clinically optimal. Nothing existing is changed: no deployed model,
config, dataset, script or report. Outputs go to reports/vessel_threshold_experiment/.

Pre-declared design (fixed before any result was seen)
  Models      : the deployed LCX / RCA pipelines' settings (LogisticRegression C=1.0, clinical
                7 inputs), rebuilt with train_phase1.make_pipeline and fitted in each fold.
  Outer CV    : RepeatedStratifiedKFold(5 folds, 5 repeats, random_state=42) per target.
  Inner CV    : StratifiedKFold(5, shuffle=True, random_state=42) INSIDE each outer training
                fold; cross_val_predict gives predictions for training patients only.
  Grid        : 0.20, 0.25, ..., 0.70.
  Objective   : maximise inner-CV sensitivity subject to inner-CV specificity >= 0.65;
                ties -> the higher threshold. If no grid value meets the constraint, use 0.5
                and flag it.
  Comparison  : the fixed 0.5 vs the selected threshold, applied to the SAME outer-fold
                probabilities (the outer model is the same pipeline fitted on the whole outer
                training fold). Outer validation labels/predictions are never used to choose.

Run from the project root (.venv active):
    python scripts\\test_vessel_thresholds.py
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import final_ml_evaluation as fe  # noqa: E402  (threshold_metrics, Log)
import train_phase1 as tp  # noqa: E402  (make_pipeline, CV settings)

TARGETS = ["LCX", "RCA"]
FORBIDDEN = ("Cath", "LAD", "LCX", "RCA")
GRID = [round(t, 2) for t in np.arange(0.20, 0.7001, 0.05)]
MIN_SPECIFICITY = 0.65
FALLBACK = 0.5
INNER_SPLITS, INNER_SEED = 5, 42
N_BOOT, BOOT_SEED = 2000, 20261014
OUT = PROJECT_ROOT / "reports" / "vessel_threshold_experiment"
CONFIG = PROJECT_ROOT / "config" / "model_selection.json"
METADATA = PROJECT_ROOT / "models" / "phase1_candidates" / "metadata.json"
REF_OOF = PROJECT_ROOT / "reports" / "final_ml_evaluation" / "oof_per_repeat_selected.csv"
REF_FOLDS = PROJECT_ROOT / "reports" / "vessel_model_experiment" / "oof_predictions.csv"
STRATEGIES = ("fixed_0.5", "selected")
CLASS_METRICS = ["sensitivity", "specificity", "precision", "recall", "f1", "accuracy"]
PROB_METRICS = ["roc_auc", "brier", "log_loss"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classification(y, pred):
    tp_ = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    sens = tp_ / (tp_ + fn) if tp_ + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp_ / (tp_ + fp) if tp_ + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (tp_ + fp) and (prec + sens) else (0.0 if tp_ + fp else float("nan"))
    return {"tp": tp_, "fp": fp, "fn": fn, "tn": tn, "sensitivity": sens, "specificity": spec,
            "precision": prec, "recall": sens, "f1": f1, "accuracy": (tp_ + tn) / len(y)}


def prob_metrics(y, p):
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return {"roc_auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))}


def select_threshold(y_inner, p_inner):
    """Pre-declared rule on INNER predictions only. Returns (threshold, met, rows)."""
    rows = []
    for t in GRID:
        m = classification(y_inner, (p_inner >= t).astype(int))
        rows.append({"threshold": t, "inner_sensitivity": m["sensitivity"], "inner_specificity": m["specificity"]})
    ok = [r for r in rows if r["inner_specificity"] >= MIN_SPECIFICITY]
    if not ok:
        return FALLBACK, False, rows
    best = max(ok, key=lambda r: (r["inner_sensitivity"], r["threshold"]))
    return best["threshold"], True, rows


def repeat_level(y, P, T):
    """Per-repeat metrics; T[i, r] is the threshold applied to patient i in repeat r."""
    per = []
    for r in range(P.shape[1]):
        m = classification(y, (P[:, r] >= T[:, r]).astype(int))
        m.update(prob_metrics(y, P[:, r]))
        per.append(m)
    keys = CLASS_METRICS + PROB_METRICS + ["tp", "fp", "fn", "tn"]
    avg = {k: float(np.nanmean([d[k] for d in per])) for k in keys}
    return avg, per


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log = fe.Log(OUT / "threshold_experiment_log.txt")
    started = time.time()
    notes = []

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    data_path = PROJECT_ROOT / "data" / "raw" / config["data_file"]
    protected = [p for d in ("data", "config", "models", "scripts", "src", "tests", "reports")
                 for p in sorted((PROJECT_ROOT / d).rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and OUT not in p.parents]
    hashes = {p: sha256_of(p) for p in protected}
    if hashes[data_path] != config["data_sha256"]:
        raise SystemExit("Dataset hash differs from config; stopping.")
    df = pd.read_excel(data_path, sheet_name=0)
    numeric_set = set(meta["numeric_features"])
    log("test_vessel_thresholds.py | educational prototype, internal CV only")
    log(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}")
    log(f"Grid {GRID}; objective: max inner sensitivity s.t. inner specificity >= {MIN_SPECIFICITY}; "
        f"ties -> higher threshold; fallback {FALLBACK}")

    ref_oof = pd.read_csv(REF_OOF)
    ref_folds = pd.read_csv(REF_FOLDS) if REF_FOLDS.exists() else None
    fold_rows, oof_rows, grid_rows, rep_rows, comp_rows, warn_rows = [], [], [], [], [], []
    repro = {}
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(df), size=(N_BOOT, len(df)))
    results = {}

    for target in TARGETS:
        spec = config["tasks"][target]
        artifact = PROJECT_ROOT / spec["artifact"]
        if hashes[artifact] != spec["artifact_sha256"]:
            raise SystemExit(f"{target}: artifact hash differs from config; stopping.")
        deployed = joblib.load(artifact)
        feats = list(spec["input_features"])
        if feats != list(deployed.feature_names_in_):
            raise SystemExit(f"{target}: config features differ from the deployed pipeline.")
        leaked = [f for f in feats if f in FORBIDDEN]
        if leaked:
            raise SystemExit(f"{target}: forbidden inputs {leaked}")
        raw = df[target].astype(str).str.strip()
        y = (raw == spec["positive_label"]).to_numpy().astype(int)
        X = df[feats]
        num = [c for c in feats if c in numeric_set]
        cat = [c for c in feats if c not in numeric_set]
        pipe0 = tp.make_pipeline(clone(deployed.named_steps["model"]), num, cat)
        log(f"\n=== {target}: {spec['model_name']}, {len(feats)} inputs {feats}; forbidden absent: True; "
            f"1 = {spec['positive_label']!r} (n={y.sum()}) ===")

        outer = RepeatedStratifiedKFold(n_splits=tp.N_SPLITS, n_repeats=tp.N_REPEATS, random_state=tp.SEED)
        P = np.full((len(y), tp.N_REPEATS), np.nan)
        T = np.full((len(y), tp.N_REPEATS), np.nan)
        F = np.full((len(y), tp.N_REPEATS), -1)
        for fold_id, (tr, te) in enumerate(outer.split(X, y)):
            r = fold_id // tp.N_SPLITS
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = clone(pipe0).fit(X.iloc[tr], y[tr])                       # outer model
                p_te = model.predict_proba(X.iloc[te])[:, 1]
                inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
                p_inner = cross_val_predict(clone(pipe0), X.iloc[tr], y[tr], cv=inner,
                                            method="predict_proba")[:, 1]          # training patients only
            for w in caught:
                warn_rows.append({"target": target, "outer_fold": fold_id, "category": w.category.__name__,
                                  "message": str(w.message)})
            thr, met, grid = select_threshold(y[tr], p_inner)
            for g in grid:
                grid_rows.append({"target": target, "outer_fold": fold_id, "repeat": r, **g,
                                  "meets_constraint": g["inner_specificity"] >= MIN_SPECIFICITY,
                                  "selected": g["threshold"] == thr and met})
            P[te, r], T[te, r], F[te, r] = p_te, thr, fold_id
            chosen = next(g for g in grid if g["threshold"] == thr) if met else {}
            m05 = classification(y[te], (p_te >= 0.5).astype(int))
            msel = classification(y[te], (p_te >= thr).astype(int))
            pm = prob_metrics(y[te], p_te)
            fold_rows.append({
                "target": target, "outer_fold": fold_id, "repeat": r, "n_train": len(tr), "n_validation": len(te),
                "n_validation_positive": int(y[te].sum()), "selected_threshold": thr, "constraint_met": met,
                "inner_sensitivity_at_selected": chosen.get("inner_sensitivity", float("nan")),
                "inner_specificity_at_selected": chosen.get("inner_specificity", float("nan")),
                **{f"val_{k}_at_0.5": m05[k] for k in CLASS_METRICS},
                **{f"val_{k}_at_selected": msel[k] for k in CLASS_METRICS},
                **{f"val_{k}": pm[k] for k in PROB_METRICS}})
            if not met:
                notes.append(f"{target} outer fold {fold_id}: no grid threshold met specificity >= "
                             f"{MIN_SPECIFICITY}; fallback {FALLBACK} used.")
        if np.isnan(P).any():
            raise SystemExit(f"{target}: missing OOF predictions.")

        # reproduction of the saved baseline OOF predictions and fold IDs
        ref = ref_oof[(ref_oof.task == target) & (ref_oof.model == spec["model_name"])]
        ref_P = ref.pivot(index="row_index", columns="repeat", values="oof_proba").sort_index().to_numpy()
        d_oof = float(np.max(np.abs(ref_P - P)))
        same_folds = None
        if ref_folds is not None:
            rf = ref_folds[(ref_folds.target == target) & (ref_folds.candidate == "baseline")]
            same_folds = bool(np.array_equal(rf.pivot(index="row_index", columns="repeat", values="outer_fold")
                                             .sort_index().to_numpy(), F))
        ok = d_oof < 1e-6 and same_folds is not False
        repro[target] = {"max_abs_oof_diff": d_oof, "fold_ids_match_saved": same_folds, "reproduced": ok}
        log(f"  Baseline reproduction: OOF diff {d_oof:.1e}; fold IDs match saved: {same_folds} -> "
            f"{'REPRODUCED' if ok else 'NOT REPRODUCED'}")
        if not ok:
            notes.append(f"{target}: baseline did not reproduce saved OOF predictions.")

        # metrics: identical probabilities; only thresholds differ
        T05 = np.full_like(P, 0.5)
        est = {}
        per = {}
        est["fixed_0.5"], per["fixed_0.5"] = repeat_level(y, P, T05)
        est["selected"], per["selected"] = repeat_level(y, P, T)
        for s in STRATEGIES:
            for r, m in enumerate(per[s]):
                rep_rows.append({"target": target, "strategy": s, "repeat": r, **m})
        boot = {s: {k: np.empty(N_BOOT) for k in CLASS_METRICS + PROB_METRICS} for s in STRATEGIES}
        for b in range(N_BOOT):
            idx = boot_idx[b]
            for s, TT in (("fixed_0.5", T05), ("selected", T)):
                e, _ = repeat_level(y[idx], P[idx], TT[idx])
                for k in boot[s]:
                    boot[s][k][b] = e[k]
        for k in CLASS_METRICS + PROB_METRICS:
            d = boot["selected"][k] - boot["fixed_0.5"][k]
            rd = np.array([per["selected"][r][k] - per["fixed_0.5"][r][k] for r in range(tp.N_REPEATS)])
            fb = np.array([f[f"val_{k}_at_0.5"] if k in CLASS_METRICS else f[f"val_{k}"]
                           for f in fold_rows if f["target"] == target])
            fs = np.array([f[f"val_{k}_at_selected"] if k in CLASS_METRICS else f[f"val_{k}"]
                           for f in fold_rows if f["target"] == target])

            def pct(v):
                v = v[~np.isnan(v)]
                return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) if len(v) else (np.nan, np.nan, 0)
            b05, bsel, bd = pct(boot["fixed_0.5"][k]), pct(boot["selected"][k]), pct(d)
            comp_rows.append({
                "target": target, "metric": k,
                "type": "probability (threshold-independent)" if k in PROB_METRICS else "classification",
                "fixed_0.5": est["fixed_0.5"][k], "fixed_0.5_ci95_low": b05[0], "fixed_0.5_ci95_high": b05[1],
                "selected": est["selected"][k], "selected_ci95_low": bsel[0], "selected_ci95_high": bsel[1],
                "difference": est["selected"][k] - est["fixed_0.5"][k],
                "difference_ci95_low": bd[0], "difference_ci95_high": bd[1], "valid_bootstrap": bd[2],
                "repeat_sd_fixed_0.5": float(np.nanstd([p_[k] for p_ in per["fixed_0.5"]], ddof=1)),
                "repeat_sd_selected": float(np.nanstd([p_[k] for p_ in per["selected"]], ddof=1)),
                "repeats_selected_higher": int(np.sum(rd > 1e-12)), "repeats_equal": int(np.sum(np.abs(rd) <= 1e-12)),
                "fold_mean_fixed_0.5": float(np.nanmean(fb)), "fold_sd_fixed_0.5": float(np.nanstd(fb, ddof=1)),
                "fold_mean_selected": float(np.nanmean(fs)), "fold_sd_selected": float(np.nanstd(fs, ddof=1))})
        for i in range(len(y)):
            for r in range(tp.N_REPEATS):
                oof_rows.append({"target": target, "row_index": i, "repeat": r, "outer_fold": int(F[i, r]),
                                 "y_true": int(y[i]), "oof_proba": P[i, r], "selected_threshold": T[i, r],
                                 "pred_at_0.5": int(P[i, r] >= 0.5), "pred_at_selected": int(P[i, r] >= T[i, r])})
        thr_list = [f["selected_threshold"] for f in fold_rows if f["target"] == target]
        results[target] = {"thr": thr_list, "est": est}
        log(f"  Selected thresholds over 25 folds: median {np.median(thr_list):.2f}, range "
            f"{min(thr_list):.2f}-{max(thr_list):.2f}; constraint met in "
            f"{sum(f['constraint_met'] for f in fold_rows if f['target'] == target)}/25 folds")
        log(f"  At 0.5      : sens {est['fixed_0.5']['sensitivity']:.3f}, spec {est['fixed_0.5']['specificity']:.3f}")
        log(f"  At selected : sens {est['selected']['sensitivity']:.3f}, spec {est['selected']['specificity']:.3f}")
        log(f"  ROC-AUC {est['fixed_0.5']['roc_auc']:.3f} (identical for both: same probabilities)")

    # ------------------------------------------------------------------ files
    warn_df = pd.DataFrame(warn_rows, columns=["target", "outer_fold", "category", "message"])
    warn_df.to_csv(OUT / "fit_warnings.csv", index=False)
    if len(warn_df):
        notes.append(f"{len(warn_df)} warning(s) during fitting; see fit_warnings.csv.")
    folds = pd.DataFrame(fold_rows); folds.to_csv(OUT / "per_fold_thresholds.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "inner_threshold_grid.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "repeat_metrics.csv", index=False)
    comp = pd.DataFrame(comp_rows); comp.to_csv(OUT / "threshold_comparison.csv", index=False)
    run_config = {
        "script": "scripts/test_vessel_thresholds.py", "python": platform.python_version(),
        "scikit_learn": sklearn.__version__, "targets": TARGETS,
        "outer_cv": {"type": "RepeatedStratifiedKFold", "n_splits": tp.N_SPLITS, "n_repeats": tp.N_REPEATS,
                     "random_state": tp.SEED},
        "inner_cv": {"type": "StratifiedKFold", "n_splits": INNER_SPLITS, "shuffle": True,
                     "random_state": INNER_SEED, "method": "cross_val_predict on outer training fold only"},
        "threshold_grid": GRID, "objective": f"maximise inner sensitivity subject to inner specificity >= {MIN_SPECIFICITY}",
        "tie_break": "higher threshold", "fallback_threshold": FALLBACK,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "patient (5 OOF predictions and their fold thresholds kept together)",
                      "statistic": "repeat-averaged metric", "paired": True},
        "models": {t: {"model_name": config["tasks"][t]["model_name"], "estimator": "cloned from deployed artifact",
                       "input_features": config["tasks"][t]["input_features"]} for t in TARGETS},
        "forbidden_input_columns": list(FORBIDDEN), "baseline_reproduction": repro,
        "input_sha256": {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): h for p, h in hashes.items()},
    }
    (OUT / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ report
    def cell(t, k, s):
        r = comp[(comp.target == t) & (comp.metric == k)].iloc[0]
        return f"{r[s]:.3f} ({r[s + '_ci95_low']:.3f}–{r[s + '_ci95_high']:.3f})"

    def dcell(t, k):
        r = comp[(comp.target == t) & (comp.metric == k)].iloc[0]
        return f"{r.difference:+.3f} ({r.difference_ci95_low:+.3f} to {r.difference_ci95_high:+.3f})"

    L = ["# LCX/RCA threshold experiment: fixed 0.5 vs threshold chosen inside each training fold", "",
         "> **Educational prototype. Not a diagnostic tool.** Internal repeated cross-validation on 303 "
         "patients from one centre. The selection objective is an experimental choice, and **no threshold "
         "here is clinically optimal or validated.** Nothing existing was changed: the deployed models, the "
         "config and its 0.5 threshold are untouched.", "",
         "## Design (fixed before running)", "",
         f"- **Models:** the deployed LCX and RCA settings (`LogisticRegression(C=1.0)`, 7 clinical inputs: "
         f"{', '.join(config['tasks']['LCX']['input_features'])}), rebuilt with `train_phase1.make_pipeline` and "
         "fitted in every fold. `Cath`, `LAD`, `LCX` and `RCA` are never inputs (checked).",
         f"- **Outer validation:** RepeatedStratifiedKFold, {tp.N_SPLITS} × {tp.N_REPEATS}, `random_state={tp.SEED}`, "
         "per target. These are the same splits as Phase 1, and the fold IDs were checked against saved ones.",
         f"- **Threshold choice, without leakage:** inside each outer training fold, "
         f"`StratifiedKFold({INNER_SPLITS}, shuffle=True, random_state={INNER_SEED})` and `cross_val_predict` give "
         "predictions for the training patients only.",
         f"  - On those, the rule picks from the grid {GRID[0]:.2f}–{GRID[-1]:.2f} (step 0.05) the threshold with the "
         f"**highest sensitivity among those with specificity ≥ {MIN_SPECIFICITY}**. Ties go to the higher threshold; "
         f"if none qualifies, {FALLBACK} is used and flagged.",
         "  - The outer validation fold is used only for scoring.",
         "- **Same probabilities for both strategies:** the outer model is the same pipeline fitted on the whole "
         "outer training fold, so only the cut-off differs. **ROC-AUC, Brier score and log loss are therefore "
         "identical by construction.**",
         f"- **Uncertainty:** {N_BOOT}-replicate patient-level bootstrap (seed {BOOT_SEED}). Each patient keeps all 5 "
         "predictions and their fold thresholds; differences are paired.", "",
         "**Baseline reproduction:** " + "; ".join(
             f"{t}: {'REPRODUCED' if v['reproduced'] else 'NOT REPRODUCED'} (OOF diff {v['max_abs_oof_diff']:.1e}, "
             f"fold IDs match saved: {v['fold_ids_match_saved']})" for t, v in repro.items()), ""]
    for t in TARGETS:
        f = folds[folds.target == t]
        vc = f.selected_threshold.value_counts().sort_index()
        L += [f"## {t}", "",
              f"**Selected thresholds (25 outer folds):** median {f.selected_threshold.median():.2f}, range "
              f"{f.selected_threshold.min():.2f}–{f.selected_threshold.max():.2f}; counts " +
              ", ".join(f"{k:.2f}×{v}" for k, v in vc.items()) +
              f". The constraint was met in {int(f.constraint_met.sum())}/25 folds. In the inner CV, the selected "
              f"thresholds averaged sensitivity {f.inner_sensitivity_at_selected.mean():.2f} and specificity "
              f"{f.inner_specificity_at_selected.mean():.2f}. Every fold is listed in `per_fold_thresholds.csv`.", "",
              "| Metric | Fixed 0.5 | Selected in training fold | Difference (95% CI) | Repeats higher |",
              "|---|---|---|---|---|"]
        for k in CLASS_METRICS + PROB_METRICS:
            r = comp[(comp.target == t) & (comp.metric == k)].iloc[0]
            L.append(f"| {k} | {cell(t, k, 'fixed_0.5')} | {cell(t, k, 'selected')} | {dcell(t, k)} | "
                     f"{r.repeats_selected_higher}/5{' (equal)' if r.repeats_equal == 5 else ''} |")
        L.append("")
    L += ["Values are repeat-averaged out-of-fold metrics; recall equals sensitivity. Per-fold and "
          "per-repeat values are in `per_fold_thresholds.csv` and `repeat_metrics.csv`.", "",
          "## Interpretation", "",
          "- **Changing the threshold does not improve ROC-AUC or the model's ability to rank patients.** "
          "Probabilities, ROC-AUC, Brier score and log loss are identical for both strategies. Only the "
          "operating point moves.",
          "- A lower threshold flags more patients as Stenotic, so sensitivity (recall) rises and "
          "specificity falls. Precision and accuracy usually fall too, because more Normal patients are "
          "flagged. Whether the trade-off is worth it is a clinical judgement this experiment can't make.",
          f"- The specificity floor of {MIN_SPECIFICITY} was set in the inner CV only. **On the outer validation "
          "folds, specificity at the selected threshold fell below it in " + "; ".join(
              f"{t}: {int((folds[folds.target == t].val_specificity_at_selected < MIN_SPECIFICITY).sum())}/25 folds "
              f"(lowest {folds[folds.target == t].val_specificity_at_selected.min():.2f})" for t in TARGETS) +
          "**. The constraint is therefore not guaranteed on new patients; the repeat-averaged specificity is the "
          "better guide.",
          "- If a selected threshold were ever deployed, it would have to be fixed once, for example as the "
          "median of the per-fold values or chosen on all 303 patients by the same rule, and then documented "
          "and reviewed. This experiment estimates the strategy's behaviour, not one final number.", "",
          "## Limitations", "",
          "- 303 patients from one centre, with 119 (LCX) and 114 (RCA) positives. Each outer fold holds about "
          "24 positives, so per-fold sensitivity moves in steps of about 0.04, and inner-CV estimates on about "
          "242 training patients are noisy. That's why the selected threshold varies between folds.",
          "- The objective (sensitivity subject to specificity ≥ 0.65) and the grid are experimental choices "
          "with no clinical basis. A different objective would give different thresholds.",
          "- The intervals cover patient sampling only, not retraining. These models were already selected on "
          "the same folds, so absolute performance is somewhat optimistic.",
          "- There is no external or clinical validation, and no threshold here is clinically optimal.", "",
          "## Warnings", ""]
    L += [f"- {n}" for n in notes] if notes else ["- None (no fitting warnings; the constraint was met in every fold)."]
    L += ["", "## Files", "",
          "`threshold_comparison.csv` (both strategies, CIs and paired differences), `per_fold_thresholds.csv` "
          "(threshold, constraint status, inner and outer metrics for each of the 25 folds per target), "
          "`inner_threshold_grid.csv` (inner-CV sensitivity and specificity at every grid value for every "
          "fold), `oof_predictions.csv` (patient × repeat: fold, probability, fold threshold, both predictions), "
          "`repeat_metrics.csv`, `fit_warnings.csv`, `run_config.json` (design, seeds, input file hashes) and "
          "`threshold_experiment_log.txt`.", "",
          "**No threshold is adopted.** The deployed config keeps 0.5."]
    (OUT / "threshold_experiment_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    changed = [str(p.relative_to(PROJECT_ROOT)) for p in protected if sha256_of(p) != hashes[p]]
    log("\n=== SUMMARY (repeat-averaged; 95% bootstrap CI of difference, selected - 0.5) ===")
    for t in TARGETS:
        for k in ("sensitivity", "specificity", "precision", "f1", "accuracy", "roc_auc"):
            r = comp[(comp.target == t) & (comp.metric == k)].iloc[0]
            log(f"  {t} {k:12s} 0.5: {r['fixed_0.5']:.3f} | selected: {r.selected:.3f} | diff {dcell(t, k)}")
    log(f"\nBaseline reproduced: {all(v['reproduced'] for v in repro.values())}")
    log(f"Warnings: {'; '.join(notes) if notes else 'none'}")
    log(f"Protected files checked: {len(protected)}; changed: {changed if changed else 'none'}")
    log(f"Outputs: {OUT}")
    log(f"Total time {time.time() - started:.1f}s")
    log("THRESHOLD EXPERIMENT COMPLETE (no threshold adopted; deployed models and config unchanged)")
    log.close()
    if changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
