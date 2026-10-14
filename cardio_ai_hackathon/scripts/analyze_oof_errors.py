"""
analyze_oof_errors.py
=====================
Experiment 3: out-of-fold (OOF) error analysis of the four selected models.

ANALYSIS ONLY. No model is fitted, no prediction is made by a deployed model, and no
existing file is changed. Outputs go to reports/error_analysis/.

EDUCATIONAL PROTOTYPE. 303 patients from one centre. Subgroup results are exploratory
descriptions, not clinical conclusions; the models are not clinically validated.

Inputs (read-only)
  reports/final_ml_evaluation/oof_per_repeat_selected.csv  per-patient, per-repeat OOF probabilities
  config/model_selection.json                               selected models, labels, threshold
  data/raw/extention of Z-Alizadeh sani dataset.xlsx         used ONLY to describe patients/subgroups
  reports/lad_calibration/lad_calibration_predictions.csv   saved fold IDs (LAD), cross-check
  reports/vessel_model_experiment/oof_predictions.csv       saved fold IDs (LCX/RCA), cross-check

Out-of-fold check: the Phase 1 splits (RepeatedStratifiedKFold, 5 x 5, seed 42, per target)
are regenerated (splitting only) to assign fold IDs and to verify that every patient is
predicted exactly once per repeat; where fold IDs were saved by later experiments they must
match. The script stops if any alignment check fails.

Run from the project root (.venv active):
    python scripts\\analyze_oof_errors.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import final_ml_evaluation as fe  # noqa: E402  (threshold_metrics, wilson, Log)
import train_phase1 as tp  # noqa: E402  (CV settings)

TASKS = ["Cath", "LAD", "LCX", "RCA"]
CONFIG = PROJECT_ROOT / "config" / "model_selection.json"
OOF = PROJECT_ROOT / "reports" / "final_ml_evaluation" / "oof_per_repeat_selected.csv"
LAD_FOLDS = PROJECT_ROOT / "reports" / "lad_calibration" / "lad_calibration_predictions.csv"
VESSEL_FOLDS = PROJECT_ROOT / "reports" / "vessel_model_experiment" / "oof_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "error_analysis"
N_BOOT = 1000
BOOT_SEED = 20261013
MIN_CLASS = 10            # subgroup metrics need >= 10 positives (sensitivity) / negatives (specificity)
INCONSISTENT_ROW = 93


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(y, p):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p, thr):
    t = fe.threshold_metrics(y, p, thr)
    return {"tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "tn": t["tn"],
            "sensitivity": t["sensitivity"], "specificity": t["specificity"],
            "precision": t["precision"], "recall": t["sensitivity"], "f1": t["f1"],
            "accuracy": t["accuracy"], "roc_auc": auc(y, p)}


KEYS = ["tp", "fp", "fn", "tn", "sensitivity", "specificity", "precision", "recall", "f1",
        "accuracy", "roc_auc"]


def repeat_average(y, P, thr):
    per = [metrics(y, P[:, r], thr) for r in range(P.shape[1])]
    avg = {}
    for k in KEYS:
        v = np.array([d[k] for d in per], dtype=float)
        avg[k] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
    return avg, per


def subgroup_definitions(df):
    """Label-free subgroups from the dataset (descriptive only; never model inputs here)."""
    age = pd.cut(df["Age"], [0, 49, 59, 69, 200], labels=["<50", "50-59", "60-69", "70+"])
    bmi = pd.cut(df["BMI"], [0, 24.999, 29.999, 100], labels=["<25", "25-30", ">=30"])
    return {
        "Age group": age.astype(str),
        "Sex (as recorded)": df["Sex"].astype(str),
        "Diabetes (DM)": df["DM"].map({0: "No", 1: "Yes"}),
        "Hypertension (HTN)": df["HTN"].map({0: "No", 1: "Yes"}),
        "Typical chest pain": df["Typical Chest Pain"].map({0: "No", 1: "Yes"}),
        "Current smoker": df["Current Smoker"].map({0: "No", 1: "Yes"}),
        "Dyslipidaemia (DLP)": df["DLP"].map({"N": "No", "Y": "Yes"}),
        "BMI group": bmi.astype(str),
        "EF-TTE": np.where(df["EF-TTE"] < 50, "<50%", ">=50%"),
        "Region RWMA": np.where(df["Region RWMA"] > 0, ">=1", "0"),
    }


PROFILE_FEATURES = ["Age", "BMI", "EF-TTE"]
PROFILE_BINARY = {"Male sex": ("Sex", "Male"), "DM": ("DM", 1), "HTN": ("HTN", 1),
                  "Typical chest pain": ("Typical Chest Pain", 1), "Current smoker": ("Current Smoker", 1),
                  "DLP": ("DLP", "Y"), "Region RWMA >= 1": ("Region RWMA", "pos")}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    started = time.time()

    # ------------------------------------------------------------------ inputs must exist
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else None
    data_path = PROJECT_ROOT / "data" / "raw" / config["data_file"] if config else None
    missing = [str(p) for p in (CONFIG, OOF, data_path) if p is None or not p.exists()]
    if missing:
        raise SystemExit(f"STOP: required input(s) missing: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    log = fe.Log(OUT / "analysis_log.txt")
    protected = [p for d in ("data", "config", "models", "scripts", "src", "tests", "reports")
                 for p in sorted((PROJECT_ROOT / d).rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and OUT not in p.parents]
    hashes = {p: sha256_of(p) for p in protected}

    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    log("analyze_oof_errors.py | analysis only, OOF predictions only")
    log("\n=== DATA AND ALIGNMENT CHECKS ===")
    df = pd.read_excel(data_path, sheet_name=0)
    oof = pd.read_csv(OOF)
    ok = check("dataset hash matches config", hashes[data_path] == config["data_sha256"])
    ok &= check("dataset has 303 rows with index 0..302", len(df) == 303 and list(df.index) == list(range(303)),
                f"{df.shape}")
    check("no exact duplicate rows", int(df.duplicated().sum()) == 0, f"{int(df.duplicated().sum())} duplicates")
    check("no missing values", int(df.isna().sum().sum()) == 0, f"{int(df.isna().sum().sum())} missing")
    ok &= check("OOF file has expected columns",
                set(["row_index", "task", "model", "repeat", "y_true", "oof_proba"]) <= set(oof.columns),
                str(list(oof.columns)))

    y_all, P_all, fold_all, thr = {}, {}, {}, {}
    for task in TASKS:
        spec = config["tasks"][task]
        thr[task] = float(spec["decision_threshold"])
        raw = df[task].astype(str).str.strip()
        y = (raw == spec["positive_label"]).to_numpy().astype(int)
        ok &= check(f"{task}: raw labels are exactly {{{spec['positive_label']}, {spec['negative_label']}}}",
                    sorted(raw.unique()) == sorted([spec["positive_label"], spec["negative_label"]]))
        d = oof[(oof.task == task) & (oof.model == spec["model_name"])]
        ok &= check(f"{task}: OOF rows for selected model {spec['model_name']}",
                    len(d) == 303 * tp.N_REPEATS and d.row_index.nunique() == 303
                    and sorted(d.repeat.unique()) == list(range(tp.N_REPEATS)), f"{len(d)} rows")
        P = d.pivot(index="row_index", columns="repeat", values="oof_proba").sort_index()
        ok &= check(f"{task}: probabilities complete and within [0, 1]",
                    P.shape == (303, tp.N_REPEATS) and not P.isna().any().any()
                    and float(P.min().min()) >= 0 and float(P.max().max()) <= 1)
        yt = d.groupby("row_index").y_true.agg(["min", "max"]).sort_index()
        ok &= check(f"{task}: y_true constant per patient and equal to dataset labels",
                    bool((yt["min"] == yt["max"]).all()) and np.array_equal(yt["min"].to_numpy(), y))
        splits = list(RepeatedStratifiedKFold(n_splits=tp.N_SPLITS, n_repeats=tp.N_REPEATS,
                                              random_state=tp.SEED).split(df, y))
        fold = np.full((303, tp.N_REPEATS), -1)
        for fid, (_, te) in enumerate(splits):
            fold[te, fid // tp.N_SPLITS] = fid
        ok &= check(f"{task}: every patient predicted exactly once per repeat (regenerated splits)",
                    bool((fold >= 0).all()) and all(
                        np.array_equal(np.sort(np.concatenate([splits[r * tp.N_SPLITS + k][1]
                                                               for k in range(tp.N_SPLITS)])), np.arange(303))
                        for r in range(tp.N_REPEATS)))
        saved = None
        if task == "LAD" and LAD_FOLDS.exists():
            s = pd.read_csv(LAD_FOLDS)
            saved = (s.pivot(index="row_index", columns="repeat", values="outer_fold").sort_index().to_numpy(),
                     s.pivot(index="row_index", columns="repeat", values="p_baseline").sort_index().to_numpy())
        if task in ("LCX", "RCA") and VESSEL_FOLDS.exists():
            s = pd.read_csv(VESSEL_FOLDS)
            s = s[(s.target == task) & (s.candidate == "baseline")]
            saved = (s.pivot(index="row_index", columns="repeat", values="outer_fold").sort_index().to_numpy(),
                     s.pivot(index="row_index", columns="repeat", values="oof_proba").sort_index().to_numpy())
        if saved is not None:
            ok &= check(f"{task}: regenerated fold IDs equal independently saved fold IDs",
                        np.array_equal(saved[0], fold))
            ok &= check(f"{task}: saved OOF probabilities agree across reports",
                        float(np.max(np.abs(saved[1] - P.to_numpy()))) < 1e-9,
                        f"max diff {float(np.max(np.abs(saved[1] - P.to_numpy()))):.1e}")
        else:
            check(f"{task}: no independently saved fold IDs to compare (regenerated only)", True)
        y_all[task], P_all[task], fold_all[task] = y, P.to_numpy(), fold
    any_vessel = (y_all["LAD"] + y_all["LCX"] + y_all["RCA"]) > 0
    inconsistent = [int(i) for i in np.where((y_all["Cath"] == 1) != any_vessel)[0]]
    check("inconsistent target rows identified (kept unchanged)", inconsistent == [INCONSISTENT_ROW],
          f"rows {inconsistent}")
    pd.DataFrame(checks).to_csv(OUT / "data_checks.csv", index=False)
    if not ok:
        log("\nSTOP: an alignment check failed; no analysis was produced. See data_checks.csv.")
        log.close()
        raise SystemExit(1)

    # ------------------------------------------------------------------ overall errors
    log("\n=== OVERALL ERRORS AT THE CONFIG THRESHOLD ===")
    cm_rows, rep_rows, summ_rows, pat_rows = [], [], [], []
    for task in TASKS:
        y, P, t = y_all[task], P_all[task], thr[task]
        avg, per = repeat_average(y, P, t)
        for r, m in enumerate(per):
            rep_rows.append({"task": task, "summary": f"repeat_{r}", **m})
            cm_rows.append({"task": task, "summary": f"repeat_{r}", "tp": m["tp"], "fp": m["fp"],
                            "fn": m["fn"], "tn": m["tn"]})
        cm_rows.append({"task": task, "summary": "average_over_5_repeats",
                        **{k: avg[k] for k in ("tp", "fp", "fn", "tn")}})
        pm = metrics(y, P.mean(axis=1), t)
        cm_rows.append({"task": task, "summary": "patient_mean_probability",
                        **{k: pm[k] for k in ("tp", "fp", "fn", "tn")}})
        rep_rows.append({"task": task, "summary": "average_over_5_repeats", **avg})
        rep_rows.append({"task": task, "summary": "patient_mean_probability", **pm})

        wrong = (P >= t).astype(int) != y[:, None]
        n_wrong = wrong.sum(axis=1)
        for i in range(303):
            kind = ("correct" if n_wrong[i] == 0 else
                    ("false negative" if y[i] == 1 else "false positive"))
            pat_rows.append({
                "row_index": i, "task": task, "y_true": int(y[i]),
                "recorded_label": df.loc[i, task], "n_repeats_misclassified": int(n_wrong[i]),
                "error_type": kind,
                "stability": "always correct" if n_wrong[i] == 0 else
                             ("misclassified in all 5 repeats" if n_wrong[i] == tp.N_REPEATS else "unstable (1-4 of 5)"),
                "mean_oof_proba": float(P[i].mean()), "min_oof_proba": float(P[i].min()),
                "max_oof_proba": float(P[i].max()),
                "outer_folds": ";".join(str(int(f)) for f in fold_all[task][i]),
                "inconsistent_target_row": i == INCONSISTENT_ROW})
        pos, neg = y == 1, y == 0
        summ_rows.append({
            "task": task, "model": config["tasks"][task]["model_name"], "threshold": t,
            "n_patients": 303, "n_positive": int(pos.sum()), "n_negative": int(neg.sum()),
            "avg_false_positives_per_repeat": avg["fp"], "avg_false_negatives_per_repeat": avg["fn"],
            **{f"{k}_repeat_avg": avg[k] for k in ("sensitivity", "specificity", "precision", "recall", "f1",
                                                   "accuracy", "roc_auc")},
            **{f"{k}_repeat_min": min(d[k] for d in per) for k in ("sensitivity", "specificity", "roc_auc")},
            **{f"{k}_repeat_max": max(d[k] for d in per) for k in ("sensitivity", "specificity", "roc_auc")},
            "patients_always_correct": int((n_wrong == 0).sum()),
            "patients_FN_all_5_repeats": int(((n_wrong == 5) & pos).sum()),
            "patients_FP_all_5_repeats": int(((n_wrong == 5) & neg).sum()),
            "patients_unstable_1_to_4": int(((n_wrong > 0) & (n_wrong < 5)).sum())})
        log(f"  {task}: per repeat FP {avg['fp']:.1f}, FN {avg['fn']:.1f} | sens {avg['sensitivity']:.3f} "
            f"spec {avg['specificity']:.3f} AUC {avg['roc_auc']:.3f} | persistent FN "
            f"{summ_rows[-1]['patients_FN_all_5_repeats']}, persistent FP {summ_rows[-1]['patients_FP_all_5_repeats']}, "
            f"unstable {summ_rows[-1]['patients_unstable_1_to_4']}")
    pat = pd.DataFrame(pat_rows)

    # ------------------------------------------------------------------ subgroups
    log("\n=== SUBGROUPS (exploratory) ===")
    groups = subgroup_definitions(df)
    rng = np.random.default_rng(BOOT_SEED)
    sub_rows = []
    for task in TASKS:
        y, P, t = y_all[task], P_all[task], thr[task]
        for gname, labels in groups.items():
            labels = pd.Series(labels, index=df.index).astype(str)
            order = {"<50": 0, "50-59": 1, "60-69": 2, "70+": 3, "<25": 0, "25-30": 1, ">=30": 2}
            for level in sorted(labels.unique(), key=lambda v: (order.get(v, 9), v)):
                idx = np.where(labels.to_numpy() == level)[0]
                yy, PP = y[idx], P[idx]
                n_pos, n_neg = int(yy.sum()), int(len(yy) - yy.sum())
                avg, _ = repeat_average(yy, PP, t)
                boot = {k: [] for k in ("sensitivity", "specificity", "accuracy", "roc_auc")}
                for _ in range(N_BOOT):
                    b = rng.integers(0, len(idx), len(idx))
                    e, _ = repeat_average(yy[b], PP[b], t)
                    for k in boot:
                        boot[k].append(e[k])
                row = {"task": task, "subgroup": gname, "level": level, "n_patients": len(idx),
                       "n_positive": n_pos, "n_negative": n_neg,
                       "avg_fp_per_repeat": avg["fp"], "avg_fn_per_repeat": avg["fn"]}
                for k, need in (("sensitivity", n_pos >= MIN_CLASS), ("specificity", n_neg >= MIN_CLASS),
                                ("accuracy", len(idx) >= 2 * MIN_CLASS),
                                ("roc_auc", n_pos >= MIN_CLASS and n_neg >= MIN_CLASS)):
                    v = np.array(boot[k], dtype=float)
                    v = v[~np.isnan(v)]
                    row[k] = avg[k] if need else float("nan")
                    row[f"{k}_ci95_low"] = float(np.percentile(v, 2.5)) if need and len(v) else float("nan")
                    row[f"{k}_ci95_high"] = float(np.percentile(v, 97.5)) if need and len(v) else float("nan")
                row["reliability"] = ("ok" if n_pos >= MIN_CLASS and n_neg >= MIN_CLASS else
                                      f"too few {'positives' if n_pos < MIN_CLASS else 'negatives'} "
                                      f"(<{MIN_CLASS}); some metrics withheld")
                sub_rows.append(row)
    sub = pd.DataFrame(sub_rows)
    log(f"  {len(sub)} task x subgroup-level rows; {int((sub.reliability != 'ok').sum())} flagged as too small")

    # ------------------------------------------------------------------ persistent-error profiles
    prof_rows = []
    for task in TASKS:
        p = pat[pat.task == task].set_index("row_index")
        for true_class, err_name in ((1, "FN all 5 repeats"), (0, "FP all 5 repeats")):
            in_class = p[p.y_true == true_class]
            groups_def = {err_name: in_class[in_class.n_repeats_misclassified == 5].index,
                          "always correct (same true class)": in_class[in_class.n_repeats_misclassified == 0].index}
            for gname, ids in groups_def.items():
                sub_df = df.loc[ids]
                row = {"task": task, "true_class": "positive" if true_class else "negative",
                       "group": gname, "n_patients": len(ids)}
                for f in PROFILE_FEATURES:
                    row[f"{f} median"] = float(sub_df[f].median()) if len(ids) else float("nan")
                for label, (col, val) in PROFILE_BINARY.items():
                    s = sub_df[col] > 0 if val == "pos" else sub_df[col] == val
                    row[f"{label} %"] = float(100 * s.mean()) if len(ids) else float("nan")
                prof_rows.append(row)
    prof = pd.DataFrame(prof_rows)

    # cross-task persistence
    persist = pat[pat.n_repeats_misclassified == 5].groupby("row_index").task.apply(list)
    multi = persist[persist.apply(len) >= 2]

    # ------------------------------------------------------------------ write CSVs
    pd.DataFrame(cm_rows).to_csv(OUT / "confusion_matrices.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "per_repeat_metrics.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(OUT / "error_summary.csv", index=False)
    sub.to_csv(OUT / "subgroup_metrics.csv", index=False)
    pat.to_csv(OUT / "patient_error_frequency.csv", index=False)
    prof.to_csv(OUT / "persistent_error_profiles.csv", index=False)

    # ------------------------------------------------------------------ report
    def f3(v):
        return "–" if pd.isna(v) else f"{v:.3f}"

    def ci(r, k):
        return "withheld" if pd.isna(r[k]) else f"{r[k]:.2f} ({r[k + '_ci95_low']:.2f}–{r[k + '_ci95_high']:.2f})"

    L = ["# Experiment 3: out-of-fold error analysis", "",
         "> **Educational prototype. Not a diagnostic tool.** Analysis of internal cross-validation "
         "predictions for 303 patients from one centre. Subgroup results are exploratory "
         "descriptions, not clinical conclusions. Nothing here is clinical validation.", "",
         "## What was analysed", "",
         "- Out-of-fold predictions only (`reports/final_ml_evaluation/oof_per_repeat_selected.csv`). "
         "Each patient has 5 predictions, one per CV repeat, each from a model that never saw them. "
         "No deployed-model (in-sample) prediction is used.",
         "- No model was fitted and nothing existing was changed. The dataset is used only to "
         "describe patients and define subgroups; `Cath`, `LAD`, `LCX` and `RCA` serve only as "
         "the true labels being compared.",
         f"- Selected models: " + ", ".join(f"{t} `{config['tasks'][t]['model_name']}`" for t in TASKS) +
         ". Positive labels: `CAD` (Cath) and `Stenotic` (vessels). Threshold "
         f"{thr['Cath']} from `config/model_selection.json`.",
         "- **Alignment checks:** every one passed (`data_checks.csv`).",
         "  - Labels match the dataset.",
         "  - Each patient is predicted exactly once per repeat, using the regenerated Phase 1 splits.",
         "  - For LAD, LCX and RCA, the regenerated fold IDs equal the fold IDs saved independently "
         "by the calibration and vessel experiments.",
         f"  - Row {INCONSISTENT_ROW} (Cath = Normal but LAD = Stenotic) is kept as recorded and "
         "flagged in `patient_error_frequency.csv`.", "",
         "**Two kinds of summary** (never mixed):",
         "- **Per-repeat / repeat-averaged:** metrics on one repeat's 303 predictions, then averaged.",
         "- **Patient-level:** how many of a patient's 5 repeats were misclassified.",
         "",
         "The 5 predictions for one patient are never counted as 5 patients.", "",
         "## Overall errors at threshold 0.5", "",
         "| Task | Positives / negatives | FP per repeat | FN per repeat | Sensitivity | Specificity | Precision | F1 | Accuracy | ROC-AUC |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {r.task} | {r.n_positive} / {r.n_negative} | {r.avg_false_positives_per_repeat:.1f} | "
                 f"{r.avg_false_negatives_per_repeat:.1f} | {r.sensitivity_repeat_avg:.3f} "
                 f"({r.sensitivity_repeat_min:.2f}–{r.sensitivity_repeat_max:.2f}) | "
                 f"{r.specificity_repeat_avg:.3f} ({r.specificity_repeat_min:.2f}–{r.specificity_repeat_max:.2f}) | "
                 f"{r.precision_repeat_avg:.3f} | {r.f1_repeat_avg:.3f} | {r.accuracy_repeat_avg:.3f} | "
                 f"{r.roc_auc_repeat_avg:.3f} |")
    L += ["", "Values are repeat-averaged; brackets give the lowest and highest of the 5 repeats. "
          "Recall equals sensitivity. Confusion matrices for each repeat, the repeat average and the "
          "patient-mean probability are in `confusion_matrices.csv`. Bootstrap CIs for the overall "
          "metrics are in `reports/final_ml_evaluation/bootstrap_confidence_intervals.csv`.", "",
          "## Patient-level error stability", "",
          "| Task | Always correct | Missed in all 5 repeats (persistent FN) | Falsely flagged in all 5 (persistent FP) | Unstable (wrong in 1–4 repeats) |",
          "|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {r.task} | {r.patients_always_correct} | {r.patients_FN_all_5_repeats} | "
                 f"{r.patients_FP_all_5_repeats} | {r.patients_unstable_1_to_4} |")
    L += ["", f"Patients misclassified in all 5 repeats for two or more tasks: **{len(multi)}**. "
          "Persistent errors mean the model consistently places these patients on the wrong side of "
          "0.5. Unstable patients sit close to the threshold, where small training differences flip the "
          "result. Row 93's per-task status is in `patient_error_frequency.csv` "
          "(`inconsistent_target_row = True`).", "",
          "**Descriptive profile of persistent errors** (`persistent_error_profiles.csv`). For each "
          "task, patients missed in all 5 repeats are compared with correctly classified patients of "
          "the same true class (medians and percentages). These groups are small, so read them as "
          "hints for further investigation, not explanations.", ""]
    for task in TASKS:
        fnp = prof[(prof.task == task) & (prof.true_class == "positive")]
        if len(fnp) == 2 and fnp.iloc[0].n_patients:
            a, b = fnp.iloc[0], fnp.iloc[1]
            L.append(f"- **{task}**: {int(a.n_patients)} persistent FN vs {int(b.n_patients)} correctly "
                     f"detected positives. Typical chest pain {a['Typical chest pain %']:.0f}% vs "
                     f"{b['Typical chest pain %']:.0f}%, Region RWMA ≥ 1 {a['Region RWMA >= 1 %']:.0f}% vs "
                     f"{b['Region RWMA >= 1 %']:.0f}%, male {a['Male sex %']:.0f}% vs {b['Male sex %']:.0f}%, "
                     f"DM {a['DM %']:.0f}% vs {b['DM %']:.0f}%, median age {a['Age median']:.0f} vs {b['Age median']:.0f}.")
    L += ["", "## Subgroups (exploratory)", "",
          f"Each subgroup's metrics are repeat-averaged, with 95% patient-level bootstrap intervals "
          f"({N_BOOT} resamples within the subgroup, seed {BOOT_SEED}). A metric is **withheld** when "
          f"its subgroup has fewer than {MIN_CLASS} patients of the relevant class. Subgroups overlap "
          "and many are compared, so some differences will occur by chance. Where intervals overlap "
          "substantially, don't read the gap as a real difference. Full table: `subgroup_metrics.csv`.", ""]
    for task in TASKS:
        L += [f"### {task}", "", "| Subgroup | Level | n (pos/neg) | Sensitivity | Specificity | ROC-AUC |",
              "|---|---|---|---|---|---|"]
        for r in sub[sub.task == task].to_dict("records"):
            L.append(f"| {r['subgroup']} | {r['level']} | {r['n_patients']} ({r['n_positive']}/{r['n_negative']}) | "
                     f"{ci(r, 'sensitivity')} | {ci(r, 'specificity')} | {ci(r, 'roc_auc')} |")
        L.append("")
    L += ["## Limitations", "",
          "- 303 patients from one centre. Subgroups of 50–100 patients give very wide intervals; "
          "no subgroup result here is a clinical finding.",
          "- The subgroup variables (age, sex, risk factors, EF, RWMA) are partly the models' own "
          "inputs, so performance differences can reflect how informative those inputs are, not bias "
          "in the usual sense.",
          "- The threshold 0.5 is not tuned. Error counts shift with any other threshold "
          "(`reports/final_ml_evaluation/threshold_tradeoffs.csv`).",
          "- No multiple-comparison correction. The selected models were chosen on these same folds, "
          "so overall performance is somewhat optimistic.",
          "- Persistent errors may include label noise (for example row 93); this analysis can't "
          "tell model error from label error.", "",
          "## Files", "",
          "`data_checks.csv`, `confusion_matrices.csv`, `per_repeat_metrics.csv`, `error_summary.csv`, "
          "`subgroup_metrics.csv`, `patient_error_frequency.csv` (303 patients × 4 tasks, with fold IDs), "
          "`persistent_error_profiles.csv` and `analysis_log.txt`."]
    (OUT / "error_analysis_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    changed = [str(p.relative_to(PROJECT_ROOT)) for p in protected if sha256_of(p) != hashes[p]]
    log("\n=== COMPLETION SUMMARY ===")
    log(f"Alignment checks: all passed ({len(checks)})")
    log(f"Patients misclassified in all 5 repeats for >= 2 tasks: {len(multi)}")
    log(f"Outputs: {OUT}")
    log(f"Protected files checked: {len(protected)}; changed: {changed if changed else 'none'}")
    log(f"Total time {time.time() - started:.1f}s")
    log("OOF ERROR ANALYSIS COMPLETE (analysis only; no model fitted; existing files unchanged)"
        if not changed else "WARNING: protected files changed")
    log.close()
    if changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
