"""
build_model_selection.py
========================
Documents the four provisionally selected Phase 1 pipelines and verifies the saved
artifacts. It does NOT train, tune, re-run cross-validation, or change any existing
file. Everything is read-only except three NEW output files:

    config/model_selection.json                       machine-readable selection config
    reports/model_selection/model_selection_report.md human-readable report
    reports/model_selection/artifact_verification.txt full log of every check

Inputs (read-only):
    data/raw/extention of Z-Alizadeh sani dataset.xlsx
    models/phase1_candidates/*.joblib and metadata.json
    reports/phase1_modeling/cv_summary.csv, cv_fold_metrics.csv,
    oof_predictions.csv, sensitivity_inconsistent_row.csv

Run from the project folder with the virtual environment active:
    python scripts\\build_model_selection.py
"""

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------- selections
# Your provisional choices. To change one, edit ONLY the model name here and re-run.
SELECTIONS = {
    "Cath": "logreg_reduced",
    "LAD": "rf_reduced",
    "LCX": "logreg_clinical",
    "RCA": "logreg_clinical",   # revised 2026-10-10 (was logreg_reduced); see REVISIONS
}

# Record of manual changes to SELECTIONS, written into the config and the report.
REVISIONS = [
    {"date": "2026-10-10", "task": "RCA", "from": "logreg_reduced", "to": "logreg_clinical",
     "reason": "Selection-rule review: logreg_clinical has higher mean CV ROC-AUC "
               "(about 0.740 vs 0.726) and logreg_reduced had a worse Brier score in "
               "19/25 paired folds. Decided after Phase 1 results were seen."},
]

# Exact raw labels in the Excel file. Pipelines were trained on 1 = positive, 0 = negative.
POSITIVE_RAW_LABEL = {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}
NEGATIVE_RAW_LABEL = "Normal"
TARGETS = ["Cath", "LAD", "LCX", "RCA"]

# Operational version of the selection rule stated before Phase 1 results were seen:
# "highest mean AUC wins; if a simpler model is about as good, prefer the simpler one".
# The numeric thresholds below were fixed AFTER the results were available, so they are
# documented as an operationalization, not as a pre-registered rule.
SIMPLICITY_ORDER = ["logreg_clinical", "logreg_reduced", "logreg_all", "rf_reduced", "rf_all"]
CLEARLY_WORSE_AUC = 0.01      # mean paired AUC deficit of at least this much ...
CLEARLY_WORSE_FOLDS = 18      # ... AND worse in at least this many of the 25 folds

REPORTED_METRICS = ["roc_auc", "pr_auc", "brier", "balanced_accuracy", "recall",
                    "specificity", "precision", "f1", "accuracy"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
CAND_DIR = ROOT / "models" / "phase1_candidates"
P1_DIR = ROOT / "reports" / "phase1_modeling"
CONFIG_OUT = ROOT / "config" / "model_selection.json"
REPORT_DIR = ROOT / "reports" / "model_selection"


# ----------------------------------------------------------------------------- helpers
class Log:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.failures = []

    def __call__(self, text=""):
        print(text)
        self.f.write(str(text) + "\n")
        self.f.flush()

    def check(self, ok, label, detail=""):
        self(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok

    def note(self, label):
        self(f"  [NOTE] {label}")

    def close(self):
        self.f.close()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(v):
    """Convert numpy scalars to plain Python values for JSON."""
    return v.item() if isinstance(v, np.generic) else v


def paired(folds, task, a, b, metric="roc_auc"):
    """Per-fold difference a - b on identical folds."""
    fa = folds[(folds.task == task) & (folds.model == a)].sort_values("fold")[metric].to_numpy()
    fb = folds[(folds.task == task) & (folds.model == b)].sort_values("fold")[metric].to_numpy()
    d = fa - fb
    return {"mean_diff": round(float(d.mean()), 4), "sd_diff": round(float(d.std(ddof=1)), 4),
            "folds_a_higher": int((d > 0).sum()), "n_folds": int(len(d))}


def rule_choice(folds, summary, task):
    """Simplest candidate that is not clearly worse than the top-AUC candidate."""
    sub = summary[(summary.task == task) & (summary.model.isin(SIMPLICITY_ORDER))]
    top = sub.sort_values("roc_auc_mean", ascending=False).model.iloc[0]
    for m in SIMPLICITY_ORDER:
        if m == top:
            return top, top
        p = paired(folds, task, top, m)
        clearly_worse = (p["mean_diff"] >= CLEARLY_WORSE_AUC
                         and p["folds_a_higher"] >= CLEARLY_WORSE_FOLDS)
        if not clearly_worse:
            return m, top
    return top, top


def fmt(summary_row, metric):
    return f"{summary_row[f'{metric}_mean']:.3f} ± {summary_row[f'{metric}_std']:.3f}"


# ----------------------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    log = Log(REPORT_DIR / "artifact_verification.txt")
    log(f"build_model_selection.py | Python {platform.python_version()} | scikit-learn "
        f"{sklearn.__version__} | pandas {pd.__version__} | numpy {np.__version__}")

    # ------------------------------------------------------------------ inputs
    log("\n=== 1. INPUT FILES (read-only) ===")
    needed = [DATA, CAND_DIR / "metadata.json", P1_DIR / "cv_summary.csv",
              P1_DIR / "cv_fold_metrics.csv", P1_DIR / "oof_predictions.csv",
              P1_DIR / "sensitivity_inconsistent_row.csv"]
    needed += [CAND_DIR / f"{t}__{m}.joblib" for t, m in SELECTIONS.items()]
    missing = [p for p in needed if not p.exists()]
    for p in needed:
        log.check(p.exists(), f"exists: {p.relative_to(ROOT)}")
    if missing:
        log("STOPPING: required files are missing.")
        log.close()
        sys.exit(1)
    hashes_before = {p: sha256_of(p) for p in needed}

    meta = json.loads((CAND_DIR / "metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(P1_DIR / "cv_summary.csv")
    folds = pd.read_csv(P1_DIR / "cv_fold_metrics.csv")
    oof = pd.read_csv(P1_DIR / "oof_predictions.csv")
    sens = pd.read_csv(P1_DIR / "sensitivity_inconsistent_row.csv")
    df = pd.read_excel(DATA, sheet_name=0)

    log("\n=== 2. ENVIRONMENT vs TRAINING ===")
    trained = meta["versions"]
    log.check(sklearn.__version__ == trained["scikit_learn"], "scikit-learn version matches training",
              f"now {sklearn.__version__}, trained {trained['scikit_learn']}")
    log.check(hashes_before[DATA] == meta["data_sha256"], "dataset hash matches training metadata")

    # ------------------------------------------------------------------ labels
    log("\n=== 3. TARGET LABELS (raw Excel values -> 1/0 used in training) ===")
    y_true = {}
    for t in TARGETS:
        raw = df[t].astype(str).str.strip()
        levels = sorted(raw.unique())
        expected = sorted({POSITIVE_RAW_LABEL[t], NEGATIVE_RAW_LABEL})
        log.check(levels == expected, f"{t} raw labels are exactly {expected}", f"found {levels}")
        y_true[t] = (raw == POSITIVE_RAW_LABEL[t]).astype(int).to_numpy()
        log(f"      {t}: 1 = {POSITIVE_RAW_LABEL[t]!r} (n={int(y_true[t].sum())}), "
            f"0 = {NEGATIVE_RAW_LABEL!r} (n={int((1 - y_true[t]).sum())})")
    inconsistent = meta.get("inconsistent_target_rows", [])
    log.check(inconsistent == [93], "inconsistent target row documented in metadata",
              f"rows {inconsistent}")
    r93 = df.loc[93, TARGETS].to_dict() if 93 in df.index else {}
    log(f"      row 93 as recorded (unchanged): {r93}")

    # ------------------------------------------------------------------ artifacts
    log("\n=== 4. SELECTED PIPELINES ===")
    artifacts, encoders = {}, {}
    for t, m in SELECTIONS.items():
        key = f"{t}__{m}"
        path = CAND_DIR / f"{key}.joblib"
        log(f"\n-- {key}  ({path.relative_to(ROOT)}, {path.stat().st_size:,} bytes)")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe = joblib.load(path)
        log.check(not w, "loads without warnings", "; ".join(str(x.message)[:80] for x in w))
        log.check(isinstance(pipe, Pipeline), "is a scikit-learn Pipeline", type(pipe).__name__)
        steps = [s for s, _ in pipe.steps]
        log.check(steps == ["preprocess", "model"], "steps are preprocess -> model", str(steps))
        model = pipe.named_steps["model"]
        classes = [plain(c) for c in pipe.classes_]
        log.check(classes == [0, 1], "classes_ == [0, 1] (column 1 = positive)", str(classes))
        inputs = list(pipe.feature_names_in_)
        log.check(inputs == meta["saved_pipelines"][key]["features"],
                  "input feature list matches metadata.json", f"{len(inputs)} features")
        leaked = [c for c in inputs if c in TARGETS]
        log.check(not leaked, "no target column among inputs", str(leaked) if leaked else "")

        X = df[inputs]
        proba = pipe.predict_proba(X)
        ok_shape = proba.shape == (len(df), 2)
        ok_range = bool(np.all((proba >= 0) & (proba <= 1)))
        ok_sum = bool(np.allclose(proba.sum(axis=1), 1.0))
        log.check(ok_shape and ok_range and ok_sum, "predict_proba gives (n, 2), values in [0,1], rows sum to 1")
        # Orientation check only. These are the training patients, so this AUC is
        # optimistic and must never be reported as performance.
        auc_in = roc_auc_score(y_true[t], proba[:, 1])
        log.check(auc_in > 0.5, "column 1 rises with the positive label (orientation check)",
                  f"in-sample AUC {auc_in:.3f} -- NOT a performance estimate")
        p_cv = summary[(summary.task == t) & (summary.model == m)].iloc[0]
        log(f"      CV ROC-AUC {fmt(p_cv, 'roc_auc')} (25 folds) | in-sample probability range "
            f"{proba[:, 1].min():.3f}-{proba[:, 1].max():.3f}")

        # Input-contract probes (behaviour the dashboard must guard against).
        row = X.iloc[[0]].copy()
        base = pipe.predict_proba(row)[0, 1]
        try:
            pipe.predict_proba(row.drop(columns=[inputs[0]]))
            log.check(False, "missing input column raises an error")
        except ValueError:
            log.check(True, "missing input column raises an error")
        pre = pipe.named_steps["preprocess"]
        cat_cols = [c for n, _, cs in pre.transformers_ if n == "cat" for c in cs]
        num_cols = [c for n, _, cs in pre.transformers_ if n == "num" for c in cs]
        if cat_cols:
            bad = row.copy().astype({cat_cols[0]: object})
            bad.loc[bad.index[0], cat_cols[0]] = "not-a-valid-value"
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                changed = pipe.predict_proba(bad)[0, 1]
            log.note(f"invalid category in {cat_cols[0]!r} is NOT rejected: silently encoded as "
                     f"'none of the known values' (probability {base:.3f} -> {changed:.3f}). "
                     f"The dashboard must validate categories before calling the pipeline.")
        if num_cols:
            gap = row.copy().astype({num_cols[0]: float})
            gap.loc[gap.index[0], num_cols[0]] = np.nan
            filled = pipe.predict_proba(gap)[0, 1]
            log.note(f"missing value in {num_cols[0]!r} is NOT rejected: silently replaced by the "
                     f"training median (probability {base:.3f} -> {filled:.3f}). The dashboard "
                     f"must reject or clearly flag missing values.")
        if cat_cols:
            onehot = pre.named_transformers_["cat"].named_steps["onehot"]
            for c, cats in zip(cat_cols, onehot.categories_):
                encoders[c] = [plain(v) for v in cats]
        artifacts[t] = {"pipe": pipe, "key": key, "path": path, "inputs": inputs,
                        "model_type": type(model).__name__, "num_cols": num_cols,
                        "cat_cols": cat_cols}

    # ------------------------------------------------------------------ selection rule
    log("\n=== 5. SELECTION RULE CHECK ===")
    log(f"Rule: simplest of {SIMPLICITY_ORDER} that is not clearly worse than the top-AUC")
    log(f"candidate (clearly worse = mean paired AUC deficit >= {CLEARLY_WORSE_AUC} AND worse in")
    log(f">= {CLEARLY_WORSE_FOLDS}/25 folds). Thresholds set after results were seen.")
    rule = {}
    for t, m in SELECTIONS.items():
        chosen_by_rule, top = rule_choice(folds, summary, t)
        agrees = chosen_by_rule == m
        rule[t] = {"top_auc_model": top, "rule_choice": chosen_by_rule, "agrees": agrees}
        if agrees:
            log.check(True, f"{t}: selection {m!r} agrees with rule", f"top AUC: {top}")
        else:
            p = paired(folds, t, m, chosen_by_rule)
            pb = paired(folds, t, m, chosen_by_rule, "brier")
            log(f"  [REVIEW] {t}: selection {m!r} differs from rule choice {chosen_by_rule!r}. "
                f"{m} minus {chosen_by_rule}: AUC {p['mean_diff']:+.4f} "
                f"({p['folds_a_higher']}/25 folds higher), Brier {pb['mean_diff']:+.4f} "
                f"(higher Brier = worse; higher in {pb['folds_a_higher']}/25 folds).")

    # ------------------------------------------------------------------ coherence
    log("\n=== 6. CROSS-TASK COHERENCE (out-of-fold, threshold 0.5) ===")
    wide = {}
    for t, m in SELECTIONS.items():
        s = oof[(oof.task == t) & (oof.model == m)].set_index("row_index")["oof_proba_mean"]
        wide[t] = s
    wide = pd.DataFrame(wide)
    any_vessel = (wide[["LAD", "LCX", "RCA"]] >= 0.5).any(axis=1)
    cad_pos = wide["Cath"] >= 0.5
    n_vessel_no_cad = int((any_vessel & ~cad_pos).sum())
    n_cad_no_vessel = int((cad_pos & ~any_vessel).sum())
    log(f"  Patients with a vessel predicted stenotic but CAD predicted negative: {n_vessel_no_cad}")
    log(f"  Patients with CAD predicted positive but no vessel predicted stenotic: {n_cad_no_vessel}")
    log("  (The four models are independent, so such disagreements are possible and must be")
    log("   shown honestly in the dashboard rather than hidden.)")

    # ------------------------------------------------------------------ input schema
    union = []
    for t in TARGETS:
        for c in artifacts[t]["inputs"]:
            if c not in union:
                union.append(c)
    schema = {}
    for c in union:
        used_by = [t for t in TARGETS if c in artifacts[t]["inputs"]]
        if c in encoders:
            schema[c] = {"kind": "categorical", "allowed_values": encoders[c], "used_by": used_by}
        else:
            s = df[c]
            schema[c] = {"kind": "numeric",
                         "dtype": "integer" if pd.api.types.is_integer_dtype(s) else "float",
                         "training_min": plain(s.min()), "training_max": plain(s.max()),
                         "used_by": used_by}

    # ------------------------------------------------------------------ write config
    rationale = {
        "Cath": "Top mean AUC is rf_reduced, but logreg_reduced is within noise of it and "
                "has a lower (better) Brier score. Logistic regression is simpler and its "
                "explanations are exact, so it is preferred.",
        "LAD": "rf_reduced has the highest mean AUC and beats logistic regression on most "
               "folds, which suggests real non-linear signal, so the more complex model is "
               "justified for this task.",
        "LCX": "logreg_clinical (7 inputs) has the highest mean AUC; the 52-feature logistic "
               "regression does worse, consistent with overfitting on this weakest task.",
        "RCA": "logreg_clinical (7 inputs) has the highest mean AUC and the lowest Brier "
               "score; the larger logistic regressions do not improve on it."
               if SELECTIONS["RCA"] == "logreg_clinical" else
               "Provisional user choice. See rule check: logreg_clinical has a higher mean AUC "
               "and lower Brier score on the same folds.",
    }
    tasks_cfg = {}
    for t, m in SELECTIONS.items():
        a = artifacts[t]
        row = summary[(summary.task == t) & (summary.model == m)].iloc[0]
        alts = summary[(summary.task == t) & (summary.model != "dummy_prior")]
        tasks_cfg[t] = {
            "model_name": m,
            "artifact": str(a["path"].relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": hashes_before[a["path"]],
            "model_type": a["model_type"],
            "input_features": a["inputs"],
            "n_input_features": len(a["inputs"]),
            "target_column": t,
            "positive_label": POSITIVE_RAW_LABEL[t],
            "negative_label": NEGATIVE_RAW_LABEL,
            "probability_column": 1,
            "decision_threshold": meta["decision_threshold"],
            "threshold_status": "default 0.5, not tuned",
            "cv_metrics": {k: {"mean": round(float(row[f"{k}_mean"]), 4),
                               "sd": round(float(row[f"{k}_std"]), 4)} for k in REPORTED_METRICS},
            "rationale": rationale[t],
            "rule_check": rule[t],
            "alternatives_roc_auc": {r.model: round(float(r.roc_auc_mean), 4)
                                     for r in alts.itertuples()},
        }
    limitations = [
        "Performance comes from repeated stratified 5-fold CV (5 repeats) on 303 patients "
        "from one centre. There is no external validation; the models are not clinically "
        "validated and are for education/decision-support demonstration only.",
        "Each selection was made by comparing six candidates on the same folds, so the "
        "reported means are likely slightly optimistic.",
        "Threshold-based metrics use an untuned 0.5 threshold.",
        "Calibration was only summarised by the Brier score; probabilities are model "
        "scores, not validated risk estimates.",
        "Saved pipelines were refitted on all 303 patients. Predictions for those same "
        "patients are in-sample and optimistic; dataset patients shown in a demo should "
        "use the stored out-of-fold probabilities or be labelled as training patients.",
        "Row 93 (Cath=Normal, LAD=Stenotic) contradicts the dataset definition of CAD. It is "
        "kept as recorded; removing it changed CV AUC by at most "
        f"{sens['difference'].abs().max():.3f}.",
        f"The four models are independent: out-of-fold, {n_vessel_no_cad} patient(s) had a "
        f"vessel predicted stenotic while CAD was predicted negative, and {n_cad_no_vessel} "
        "had CAD predicted with no vessel predicted.",
        "The LCX model uses only 7 clinical inputs; ECG, echo and laboratory values do not "
        "affect its output.",
        "The dataset has no lesion location data, so vessel colouring shows whole-vessel "
        "model scores, never a lesion position or severity. Region RWMA is not a location.",
        "Sex level 'Fmale' is used as recorded (presumed 'Female', unconfirmed). 'Obesity' "
        "means BMI >= 25. Function Class and Region RWMA definitions are undocumented.",
        "The pipelines silently accept unknown categories and missing numeric values "
        "(encoded as zeros / median-imputed). Inputs must be validated before prediction.",
    ]
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "provisional",
        "created_by": "scripts/build_model_selection.py",
        "data_file": DATA.name, "data_sha256": hashes_before[DATA],
        "training_versions": trained,
        "validation": meta["cv"],
        "selection_rule": {"simplicity_order": SIMPLICITY_ORDER,
                           "clearly_worse_auc": CLEARLY_WORSE_AUC,
                           "clearly_worse_folds": CLEARLY_WORSE_FOLDS,
                           "note": "Thresholds set after Phase 1 results were available."},
        "tasks": tasks_cfg,
        "revisions": REVISIONS,
        "input_schema": schema,
        "inconsistent_target_rows": inconsistent,
        "limitations": limitations,
        "disclaimer": "Educational decision-support prototype. Not a diagnostic device and "
                      "not a substitute for clinical judgement or diagnostic imaging.",
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ write report
    lines = ["# Model selection report (provisional)", "",
             f"Generated {config['created_utc']} by `scripts/build_model_selection.py`. "
             "Nothing was retrained; all numbers come from the saved Phase 1 reports.", "",
             "> Educational decision-support prototype. Cross-validation on 303 patients from "
             "one centre does not establish clinical validity or diagnostic accuracy.", "",
             "## Selected pipelines", "",
             "| Task | Model | Inputs | Positive label | ROC-AUC | PR-AUC | Brier | Bal. acc. (0.5) | Rule check |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t, c in tasks_cfg.items():
        row = summary[(summary.task == t) & (summary.model == c["model_name"])].iloc[0]
        check = "agrees" if c["rule_check"]["agrees"] else f"**review: rule picks {c['rule_check']['rule_choice']}**"
        lines.append(f"| {t} | `{c['model_name']}` ({c['model_type']}) | {c['n_input_features']} | "
                     f"{c['positive_label']} | {fmt(row, 'roc_auc')} | {fmt(row, 'pr_auc')} | "
                     f"{fmt(row, 'brier')} | {fmt(row, 'balanced_accuracy')} | {check} |")
    lines += ["", "Values are mean ± SD over 25 test folds. Brier: lower is better.", ""]
    for t, c in tasks_cfg.items():
        lines += [f"## {t}", "", f"**Selected:** `{c['model_name']}` -> `{c['artifact']}`", "",
                  f"**Rationale:** {c['rationale']}", "", "| Candidate | ROC-AUC | Brier |", "|---|---|---|"]
        for r in summary[(summary.task == t)].itertuples():
            mark = " (selected)" if r.model == c["model_name"] else ""
            lines.append(f"| {r.model}{mark} | {r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f} | "
                         f"{r.brier_mean:.3f} ± {r.brier_std:.3f} |")
        lines.append("")
    lines += ["## Verified artifact contract", "",
              "- Each artifact is a scikit-learn `Pipeline` (`preprocess` -> `model`) saved with joblib.",
              "- Input: a pandas DataFrame whose column names match `input_features` exactly "
              "(extra columns are ignored; a missing column raises an error).",
              "- Output: `predict_proba(X)[:, 1]` is the probability of the positive label "
              "(`CAD` for Cath, `Stenotic` for LAD/LCX/RCA); `classes_` is `[0, 1]`.",
              "- Preprocessing (imputation, encoding, scaling) is inside the pipeline and must not "
              "be rebuilt separately.", "",
              "## Revision history", ""]
    lines += [f"- {r['date']}: {r['task']} changed from `{r['from']}` to `{r['to']}`. "
              f"{r['reason']}" for r in REVISIONS] or ["- none"]
    lines += ["", "## Known limitations", ""]
    lines += [f"- {x}" for x in limitations]
    (REPORT_DIR / "model_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ finish
    log("\n=== 7. READ-ONLY CHECK ===")
    changed = [p for p in needed if sha256_of(p) != hashes_before[p]]
    log.check(not changed, "no input file or artifact was modified",
              ", ".join(str(p.relative_to(ROOT)) for p in changed))
    log("\nWrote:")
    log(f"  {CONFIG_OUT.relative_to(ROOT)}")
    log(f"  {(REPORT_DIR / 'model_selection_report.md').relative_to(ROOT)}")
    log(f"  {(REPORT_DIR / 'artifact_verification.txt').relative_to(ROOT)}")
    log(f"\nFAILED CHECKS: {log.failures if log.failures else 'none'}")
    review = [t for t in TARGETS if not rule[t]["agrees"]]
    log(f"SELECTIONS NEEDING REVIEW: {review if review else 'none'}")
    log.close()


if __name__ == "__main__":
    main()
