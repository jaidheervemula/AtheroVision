"""
audit_step2.py
==============
Targeted, READ-ONLY audit of the "Extension of Z-Alizadeh Sani" dataset (UCI id 411).

Sections
  0. File and column overview
  1. Target consistency: Cath vs LAD/LCX/RCA
  2. Categorical audit
  3. Numeric audit
  4. Draft data dictionary
  5. Redundant / derived features
  6. Leakage screening
  7. Validation feasibility

The original data file is never written to.
All output is printed and saved to reports/audit_step2/.
"""

import argparse
import hashlib
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_DATA = Path("data") / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
OUT_DIR = Path("reports") / "audit_step2"

TARGET_NAMES = ["Cath", "LAD", "LCX", "RCA"]
VESSELS = ["LAD", "LCX", "RCA"]
POSITIVE_LABELS = {"cad", "stenotic"}
NEGATIVE_LABELS = {"normal"}

# These are review thresholds, not official clinical rules.
# Values outside them are flagged, never automatically deleted.
PLAUSIBLE_BOUNDS = {
    "age": (18, 100),
    "weight": (30, 200),
    "length": (120, 210),
    "bmi": (12, 60),
    "bp": (60, 260),
    "pr": (30, 200),
    "fbs": (40, 600),
    "cr": (0.2, 15),
    "tg": (20, 2000),
    "ldl": (10, 400),
    "hdl": (10, 150),
    "bun": (2, 150),
    "esr": (0, 150),
    "hb": (5, 22),
    "k": (2.0, 7.5),
    "na": (115, 165),
    "wbc": (1000, 50000),
    "lymph": (0, 100),
    "neut": (0, 100),
    "plt": (20, 1000),
    "ef-tte": (5, 80),
}

EXPECTED_CODED_SETS = {
    "region rwma": {0, 1, 2, 3, 4},
    "function class": {0, 1, 2, 3, 4},
}


# ------------------------------- Helpers ------------------------------------

class Report:
    """Prints to the console and writes the same text to a log file."""

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
    """Lower-case, trimmed, single-spaced version of a name."""
    return " ".join(str(name).strip().lower().split())


def find_col(df, name):
    for c in df.columns:
        if norm(c) == norm(name):
            return c
    return None


def as_binary(s):
    """Return a 0/1 series for Y/N or 0/1 columns, otherwise None."""
    if pd.api.types.is_numeric_dtype(s):
        values = set(s.dropna().unique())
        return s.astype(float) if values <= {0, 1} else None

    v = s.astype(str).str.strip().str.upper()
    if set(v.unique()) <= {"Y", "N"}:
        return v.map({"Y": 1.0, "N": 0.0})

    return None


def auc_safe(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return np.nan

    return roc_auc_score(y, x)


def hanley_mcneil_halfwidth(auc, n_pos, n_neg):
    """Approximate 95% CI half-width for an AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)

    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc ** 2)
        + (n_neg - 1) * (q2 - auc ** 2)
    ) / (n_pos * n_neg)

    return 1.96 * math.sqrt(var)


def save_csv(df, name, written):
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    written.append(path)


def build_encoded(df, feature_cols):
    """Numeric feature encoding for audit statistics only, not modeling."""
    enc, groups, skipped, notes = {}, {}, [], []

    for c in feature_cols:
        s = df[c]

        if s.nunique(dropna=False) <= 1:
            skipped.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            enc[c] = s.astype(float)
            groups[c] = c
            continue

        v = s.astype(str).str.strip()
        levels = sorted(v.unique())

        if norm(c) == "vhd":
            m = v.str.lower().map({
                "n": 0,
                "mild": 1,
                "moderate": 2,
                "severe": 3,
            })

            if m.notna().all():
                enc[c] = m.astype(float)
                groups[c] = c
                notes.append(
                    f"{c}: ordinal N=0, mild=1, Moderate=2, Severe=3"
                )
                continue

        if len(levels) == 2:
            pos = "Y" if "Y" in levels else levels[-1]
            neg = [lv for lv in levels if lv != pos][0]

            enc[c] = (v == pos).astype(float)
            groups[c] = c
            notes.append(f"{c}: 1 = {pos!r}, 0 = {neg!r}")

        else:
            for lv in levels:
                key = f"{c}={lv}"
                enc[key] = (v == lv).astype(float)
                groups[key] = c

    return pd.DataFrame(enc, index=df.index), groups, skipped, notes
# -------------------------- Section 0: Overview -----------------------------

def section0_overview(df, data_path, R):
    R.section("SECTION 0 — FILE AND COLUMN OVERVIEW")

    R(f"File: {data_path}")
    R(f"Rows: {df.shape[0]}")
    R(f"Columns: {df.shape[1]}")
    R(f"Column names: {list(df.columns)}")

    R("\nData types:")
    R(df.dtypes.to_string())

    R("\nMissing values by column:")
    missing = df.isna().sum()
    R(missing[missing > 0].to_string() if missing.any() else "No missing values.")

    R("\nExact duplicate rows:")
    R(int(df.duplicated().sum()))

    R("\nConstant columns:")
    constant = [
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]
    R(constant if constant else "None")

    R("\nNumeric summary:")
    R(df.describe(include=[np.number]).T.to_string())


# ---------------------- Section 1: Target consistency -----------------------

def section1_targets(df, R, written):
    R.section("SECTION 1 — TARGET LABEL CONSISTENCY")

    tcols = {}
    for name in TARGET_NAMES:
        col = find_col(df, name)
        if col is None:
            raise ValueError(
                f"Required target column {name!r} was not found. "
                f"Available columns: {list(df.columns)}"
            )
        tcols[name] = col

    labels = pd.DataFrame(index=df.index)

    for name, col in tcols.items():
        s = df[col].astype(str).str.strip().str.lower()

        if name == "Cath":
            labels[name] = s.map({
                "cad": 1,
                "normal": 0,
            })
        else:
            labels[name] = s.map({
                "stenotic": 1,
                "normal": 0,
            })

        unknown = labels[name].isna()
        if unknown.any():
            R(
                f"WARNING: {name} contains "
                f"{int(unknown.sum())} unrecognized labels."
            )
            R(f"Observed labels: {sorted(s.unique().tolist())}")

    if labels.isna().any().any():
        R(
            "Target consistency comparison skipped because at least one "
            "target contains unrecognized labels."
        )
        return labels, tcols

    vessel_positive = labels[VESSELS].eq(1).any(axis=1)
    cad_positive = labels["Cath"].eq(1)

    cad_with_stenosis = int((cad_positive & vessel_positive).sum())
    cad_all_normal = int((cad_positive & ~vessel_positive).sum())
    normal_with_stenosis = int((~cad_positive & vessel_positive).sum())
    normal_all_normal = int((~cad_positive & ~vessel_positive).sum())

    R("\nTarget counts:")
    for name in TARGET_NAMES:
        R(f"\n{name}:")
        R(labels[name].map({1: "Positive", 0: "Negative"}).value_counts().to_string())

    R("\nCross-check: Cath CAD versus any stenotic vessel")
    R(f"Cath CAD AND at least one stenotic vessel: {cad_with_stenosis}")
    R(f"Cath CAD but all three vessels normal: {cad_all_normal}")
    R(f"Cath Normal but at least one vessel stenotic: {normal_with_stenosis}")
    R(f"Cath Normal AND all three vessels normal: {normal_all_normal}")

    inconsistent = (
        (cad_positive & ~vessel_positive)
        | (~cad_positive & vessel_positive)
    )

    R("\nRows with disagreement between Cath and vessel labels:")
    bad = df.loc[inconsistent, list(tcols.values())]
    R(bad.to_string() if len(bad) else "None")

    if len(bad):
        save_csv(
            df.loc[inconsistent].copy(),
            "target_inconsistent_rows.csv",
            written,
        )

    R(
        "\nNOTE: This is a consistency check only. It does not establish "
        "that Cath must equal the combination of the three vessel labels."
    )

    return labels, tcols


# ----------------------- Section 2: Categories ------------------------------

def section2_categorical(df, tcols, R, written):
    R.section("SECTION 2 — CATEGORICAL VALUES AND RARE LEVELS")

    cat_rows = []

    for c in df.columns:
        if (
            pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        ):
            s = df[c].astype("string")
            counts = s.value_counts(dropna=False)

            R(f"\nColumn: {c}")
            R(f"Unique values: {s.nunique(dropna=False)}")
            R(counts.to_string())

            for value, count in counts.items():
                cat_rows.append({
                    "column": c,
                    "value": str(value),
                    "count": int(count),
                    "proportion": round(count / len(df), 4),
                    "rare_level_under_5": count < 5,
                    "has_leading_or_trailing_whitespace": (
                        isinstance(value, str)
                        and value != value.strip()
                    ),
                })

    if cat_rows:
        save_csv(pd.DataFrame(cat_rows), "categorical_audit.csv", written)

    R("\nPotential spelling / whitespace issues:")
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            vals = df[c].dropna().astype(str)
            whitespace = vals[vals != vals.str.strip()]
            if len(whitespace):
                R(f"{c}: whitespace values {whitespace.unique().tolist()}")

    R(
        "\nReview category spelling differences manually. "
        "Do not automatically merge categories without checking the source."
    )
    # ------------------------- Section 3: Numeric audit -------------------------

def section3_numeric(df, tcols, R, written):
    R.section("SECTION 3 — NUMERIC RANGES AND EXTREME VALUES")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary_rows = []
    flagged_rows = []

    for c in numeric_cols:
        s = df[c].dropna()

        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr

        summary_rows.append({
            "column": c,
            "count": len(s),
            "min": s.min(),
            "q1": q1,
            "median": s.median(),
            "q3": q3,
            "max": s.max(),
            "iqr_outlier_count": int(
                ((s < lower_iqr) | (s > upper_iqr)).sum()
            ),
        })

        R(
            f"\n{c}: min={s.min()}, median={s.median()}, "
            f"max={s.max()}, IQR outliers="
            f"{int(((s < lower_iqr) | (s > upper_iqr)).sum())}"
        )

        key = norm(c)
        bounds = PLAUSIBLE_BOUNDS.get(key)

        if bounds:
            low, high = bounds
            bad = df[c].notna() & (
                (df[c] < low) | (df[c] > high)
            )

            for idx in df.index[bad]:
                flagged_rows.append({
                    "row_index": idx,
                    "column": c,
                    "value": df.loc[idx, c],
                    "review_lower_bound": low,
                    "review_upper_bound": high,
                    "note": "Review threshold only; not an automatic error",
                })

    if summary_rows:
        save_csv(
            pd.DataFrame(summary_rows),
            "numeric_summary.csv",
            written,
        )

    if flagged_rows:
        flagged = pd.DataFrame(flagged_rows)
        save_csv(flagged, "numeric_values_to_review.csv", written)
        R("\nValues outside review thresholds:")
        R(flagged.to_string(index=False))
    else:
        R("\nNo values outside the configured review thresholds.")

    R(
        "\nIMPORTANT: IQR outliers and review-threshold flags are not "
        "proof of incorrect data. Do not automatically delete them."
    )


# ----------------------- Section 4: Data dictionary -------------------------

def section4_dictionary(df, tcols, R, written):
    R.section("SECTION 4 — DRAFT DATA DICTIONARY")

    rows = []

    for c in df.columns:
        s = df[c]
        examples = s.dropna().unique()[:8]

        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "non_missing_count": int(s.notna().sum()),
            "missing_count": int(s.isna().sum()),
            "unique_count": int(s.nunique(dropna=True)),
            "example_values": " | ".join(map(str, examples)),
            "is_target": c in tcols.values(),
            "units_or_definition": "Needs verification from source documentation",
        })

    dictionary = pd.DataFrame(rows)
    save_csv(dictionary, "draft_data_dictionary.csv", written)

    R(dictionary.to_string(index=False))
    R(
        "\nThe dictionary records observed data properties. "
        "It does not establish units or clinical meanings."
    )


# ---------------------- Section 5: Redundancy audit -------------------------

def section5_redundancy(df, enc, groups, R, written):
    R.section("SECTION 5 — REDUNDANCY AND DERIVED FEATURES")

    constant_cols = [
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    R(f"Constant columns: {constant_cols}")

    for c in ["BMI", "Weight", "Length", "Obesity",
              "Typical Chest Pain", "Atypical", "Nonanginal"]:
        actual = find_col(df, c)
        if actual is not None:
            R(f"\n{actual}:")
            R(f"  dtype: {df[actual].dtype}")
            R(f"  unique values: {df[actual].nunique(dropna=False)}")
            R(f"  sample values: {df[actual].dropna().unique()[:12].tolist()}")

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr(method="spearman")
        pairs = []

        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1:]:
                value = corr.loc[c1, c2]

                if pd.notna(value) and abs(value) >= 0.85:
                    pairs.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "spearman_correlation": value,
                    })

        if pairs:
            pair_df = pd.DataFrame(pairs)
            pair_df["abs_correlation"] = pair_df[
                "spearman_correlation"
            ].abs()
            pair_df = pair_df.sort_values(
                "abs_correlation", ascending=False
            )

            save_csv(pair_df, "high_numeric_correlations.csv", written)
            R("\nNumeric pairs with absolute Spearman correlation >= 0.85:")
            R(pair_df.to_string(index=False))
        else:
            R("\nNo numeric pairs crossed the 0.85 correlation threshold.")

    R(
        "\nReview redundancy before modeling. Correlation alone does not "
        "prove leakage, and correlated features need not always be removed."
    )
    # ---------------------- Section 6: Leakage screening ------------------------

def section6_leakage(enc, y, skipped, notes, R, written):
    R.section("SECTION 6 — UNIVARIATE LEAKAGE SCREENING")

    R("This is an in-sample screening exercise, NOT model performance.")
    R("All four target columns are excluded from candidate features.")

    R("\nEncoding notes:")
    for note in notes:
        R(note)

    R("\nSkipped constant features:")
    R(skipped if skipped else "None")

    rows = []

    for target in TARGET_NAMES:
        if target not in y.columns:
            continue

        target_values = y[target].to_numpy(dtype=float)

        for feature in enc.columns:
            x = enc[feature].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(target_values)

            if valid.sum() < 10:
                continue

            yt = target_values[valid]
            xt = x[valid]

            if len(np.unique(yt)) != 2 or len(np.unique(xt)) < 2:
                continue

            try:
                auc = roc_auc_score(yt, xt)
                auc = max(auc, 1 - auc)
            except ValueError:
                continue

            rows.append({
                "target": target,
                "feature": feature,
                "original_feature_group": groups.get(feature, feature),
                "n": int(valid.sum()),
                "univariate_auc_absolute_direction": auc,
                "review_if_auc_above_0_85": auc > 0.85,
            })

    if rows:
        results = pd.DataFrame(rows).sort_values(
            ["target", "univariate_auc_absolute_direction"],
            ascending=[True, False],
        )

        save_csv(results, "univariate_auc_screen.csv", written)

        for target in TARGET_NAMES:
            R(f"\nTop univariate associations for {target}:")
            subset = results[results["target"] == target].head(15)
            R(subset.to_string(index=False))

        flagged = results[
            results["review_if_auc_above_0_85"]
        ]

        R("\nFeatures requiring manual review (AUC > 0.85):")
        R(flagged.to_string(index=False) if len(flagged) else "None")
    else:
        R("No eligible feature-target pairs for AUC screening.")

    R(
        "\nHigh univariate AUC is a review flag, not proof of leakage. "
        "Check timing, clinical availability and feature definitions."
    )


# ---------------------- Section 7: Validation audit -------------------------

def section7_validation(y, R, written):
    R.section("SECTION 7 — VALIDATION FEASIBILITY")

    count_rows = []

    for target in TARGET_NAMES:
        if target not in y.columns:
            continue

        values = y[target].dropna().astype(int)
        counts = values.value_counts().sort_index()

        n_negative = int((values == 0).sum())
        n_positive = int((values == 1).sum())
        n = len(values)

        R(f"\nTarget: {target}")
        R(f"Total usable rows: {n}")
        R(f"Negative class (0): {n_negative}")
        R(f"Positive class (1): {n_positive}")

        for folds in [3, 5, 10]:
            smallest = min(n_negative, n_positive)
            feasible = smallest >= folds

            R(
                f"{folds}-fold stratification: "
                f"{'class counts permit it' if feasible else 'not feasible'} "
                f"(minority count={smallest})"
            )

        count_rows.append({
            "target": target,
            "usable_rows": n,
            "negative_count": n_negative,
            "positive_count": n_positive,
            "minority_count": min(n_negative, n_positive),
        })

    if count_rows:
        save_csv(
            pd.DataFrame(count_rows),
            "validation_class_counts.csv",
            written,
        )

    # Joint patterns of the three vessel labels.
    if all(v in y.columns for v in VESSELS):
        patterns = y[VESSELS].dropna().astype(int).copy()
        pattern_names = {
            (0, 0, 0): "All vessels normal",
            (1, 0, 0): "LAD only",
            (0, 1, 0): "LCX only",
            (0, 0, 1): "RCA only",
            (1, 1, 0): "LAD and LCX",
            (1, 0, 1): "LAD and RCA",
            (0, 1, 1): "LCX and RCA",
            (1, 1, 1): "All three vessels",
        }

        tuples = [tuple(row) for row in patterns.to_numpy()]
        pattern_counts = pd.Series(tuples).value_counts()

        pattern_rows = []
        R("\nJoint vessel-label patterns:")

        for pattern, count in pattern_counts.items():
            label = pattern_names.get(pattern, str(pattern))
            R(f"{label}: {count}")

            pattern_rows.append({
                "pattern": label,
                "LAD": pattern[0],
                "LCX": pattern[1],
                "RCA": pattern[2],
                "count": int(count),
            })

        save_csv(
            pd.DataFrame(pattern_rows),
            "joint_vessel_patterns.csv",
            written,
        )

        R(
            "\nVery rare joint patterns can make joint stratification "
            "impractical. Do not merge patterns without a justified reason."
        )

    R("\nApproximate 95% ROC-AUC confidence-interval half-widths:")
    ci_rows = []

    for target in TARGET_NAMES:
        if target not in y.columns:
            continue

        values = y[target].dropna().astype(int)
        pos = int((values == 1).sum())
        neg = int((values == 0).sum())

        for evaluation, fraction in [
            ("20% holdout", 0.2),
            ("All rows pooled out-of-fold", 1.0),
        ]:
            n_pos = max(int(round(pos * fraction)), 2)
            n_neg = max(int(round(neg * fraction)), 2)

            for assumed_auc in [0.70, 0.80]:
                halfwidth = hanley_mcneil_halfwidth(
                    assumed_auc, n_pos, n_neg
                )

                ci_rows.append({
                    "target": target,
                    "evaluation": evaluation,
                    "positive_count_approx": n_pos,
                    "negative_count_approx": n_neg,
                    "assumed_auc": assumed_auc,
                    "approx_ci_halfwidth": round(halfwidth, 3),
                })

    ci_df = pd.DataFrame(ci_rows)
    R(ci_df.to_string(index=False))
    save_csv(ci_df, "auc_ci_width.csv", written)


# ---------------------- Section 6: Leakage screening ------------------------

def section6_leakage(enc, y, groups, skipped, notes, R, written):
    R.section("SECTION 6 — UNIVARIATE LEAKAGE SCREENING")

    R("This is an in-sample screening exercise, NOT model performance.")
    R("All four target columns are excluded from candidate features.")

    R("\nEncoding notes:")
    for note in notes:
        R(note)

    R("\nSkipped constant features:")
    R(skipped if skipped else "None")

    rows = []

    for target in TARGET_NAMES:
        if target not in y.columns:
            continue

        target_values = y[target].to_numpy(dtype=float)

        for feature in enc.columns:
            x = enc[feature].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(target_values)

            if valid.sum() < 10:
                continue

            yt = target_values[valid]
            xt = x[valid]

            if len(np.unique(yt)) != 2 or len(np.unique(xt)) < 2:
                continue

            try:
                auc = roc_auc_score(yt, xt)
                auc = max(auc, 1 - auc)
            except ValueError:
                continue

            rows.append({
                "target": target,
                "feature": feature,
                "original_feature_group": groups.get(feature, feature),
                "n": int(valid.sum()),
                "univariate_auc_absolute_direction": auc,
                "review_if_auc_above_0_85": auc > 0.85,
            })

    if rows:
        results = pd.DataFrame(rows).sort_values(
            ["target", "univariate_auc_absolute_direction"],
            ascending=[True, False],
        )

        save_csv(results, "univariate_auc_screen.csv", written)

        for target in TARGET_NAMES:
            R(f"\nTop univariate associations for {target}:")
            subset = results[results["target"] == target].head(15)
            R(subset.to_string(index=False))

        flagged = results[
            results["review_if_auc_above_0_85"]
        ]

        R("\nFeatures requiring manual review (AUC > 0.85):")
        R(flagged.to_string(index=False) if len(flagged) else "None")
    else:
        R("No eligible feature-target pairs for AUC screening.")

    R(
        "\nHigh univariate AUC is a review flag, not proof of leakage. "
        "Check timing, clinical availability and feature definitions."
    )


# ---------------------- Section 7: Validation audit -------------------------

def section7_validation(y, R, written):
    R.section("SECTION 7 — VALIDATION FEASIBILITY")

    count_rows = []

    for target in TARGET_NAMES:
        if target not in y.columns:
            continue

        values = y[target].dropna().astype(int)
        counts = values.value_counts().sort_index()

        n_negative = int((values == 0).sum())
        n_positive = int((values == 1).sum())
        n = len(values)

        R(f"\nTarget: {target}")
        R(f"Total usable rows: {n}")
        R(f"Negative class (0): {n_negative}")
        R(f"Positive class (1): {n_positive}")

        for folds in [3, 5, 10]:
            smallest = min(n_negative, n_positive)
            feasible = smallest >= folds

            R(
                f"{folds}-fold stratification: "
                f"{'class counts permit it' if feasible else 'not feasible'} "
                f"(minority count={smallest})"
            )

        count_rows.append({
            "target": target,
            "usable_rows": n,
            "negative_count": n_negative,
            "positive_count": n_positive,
            "minority_count": min(n_negative, n_positive),
        })

    if count_rows:
        save_csv(
            pd.DataFrame(count_rows),
            "validation_class_counts.csv",
            written,
        )

    # Joint patterns of the three vessel labels.
    if all(v in y.columns for v in VESSELS):
        patterns = y[VESSELS].dropna().astype(int).copy()
        pattern_names = {
            (0, 0, 0): "All vessels normal",
            (1, 0, 0): "LAD only",
            (0, 1, 0): "LCX only",
            (0, 0, 1): "RCA only",
            (1, 1, 0): "LAD and LCX",
            (1, 0, 1): "LAD and RCA",
            (0, 1, 1): "LCX and RCA",
            (1, 1, 1): "All three vessels",
        }

        tuples = [tuple(row) for row in patterns.to_numpy()]
        pattern_counts = pd.Series(tuples).value_counts()

        pattern_rows = []
        R("\nJoint vessel-label patterns:")

        for pattern, count in pattern_counts.items():
            label = pattern_names.get(pattern, str(pattern))
            R(f"{label}: {count}")

            pattern_rows.append({
                "pattern": label,
                "LAD": pattern[0],
                "LCX": pattern[1],
                "RCA": pattern[2],
                "count": int(count),
            })

        save_csv(
            pd.DataFrame(pattern_rows),
            "joint_vessel_patterns.csv",
            written,
        )

        R(
            "\nVery rare joint patterns can make joint stratification "
            "impractical. Do not merge patterns without a justified reason."
        )

    R("\nApproximate 95% ROC-AUC confidence-interval half-widths:")
    ci_rows = []

    for target in TARGET_NAMES:
        if target not in y.columns:
            continue

        values = y[target].dropna().astype(int)
        pos = int((values == 1).sum())
        neg = int((values == 0).sum())

        for evaluation, fraction in [
            ("20% holdout", 0.2),
            ("All rows pooled out-of-fold", 1.0),
        ]:
            n_pos = max(int(round(pos * fraction)), 2)
            n_neg = max(int(round(neg * fraction)), 2)

            for assumed_auc in [0.70, 0.80]:
                halfwidth = hanley_mcneil_halfwidth(
                    assumed_auc, n_pos, n_neg
                )

                ci_rows.append({
                    "target": target,
                    "evaluation": evaluation,
                    "positive_count_approx": n_pos,
                    "negative_count_approx": n_neg,
                    "assumed_auc": assumed_auc,
                    "approx_ci_halfwidth": round(halfwidth, 3),
                })

    ci_df = pd.DataFrame(ci_rows)
    R(ci_df.to_string(index=False))
    save_csv(ci_df, "auc_ci_width.csv", written)


# -------------------------------- Main ---------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    parser = argparse.ArgumentParser(
        description="Read-only targeted dataset audit."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the original .xlsx or .csv dataset",
    )
    args = parser.parse_args()
    data_path = args.data

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path.resolve()}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    R = Report(OUT_DIR / "audit_step2_output.txt")
    written = [R.path]

    try:
        hash_before = sha256_of(data_path)

        suffix = data_path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            with pd.ExcelFile(data_path) as xf:
                sheets = xf.sheet_names
                df = xf.parse(sheets[0])

            sheet_note = (
                f"Excel sheets: {sheets} "
                f"(audited: {sheets[0]!r})"
            )

        elif suffix == ".csv":
            df = pd.read_csv(data_path)
            sheet_note = "CSV file"

        else:
            raise ValueError(f"Unsupported file type: {suffix!r}")

        R(
            f"audit_step2.py | pandas {pd.__version__} "
            f"| numpy {np.__version__}"
        )
        R(f"SHA-256 of data file: {hash_before}")
        R(sheet_note)

        section0_overview(df, data_path, R)
        y, tcols = section1_targets(df, R, written)

        # Exclude every target column from the candidate feature set.
        feature_cols = [
            c for c in df.columns
            if c not in tcols.values()
        ]

        enc, groups, skipped, notes = build_encoded(
            df, feature_cols
        )

        steps = [
            ("2", lambda: section2_categorical(
                df, tcols, R, written
            )),
            ("3", lambda: section3_numeric(
                df, tcols, R, written
            )),
            ("4", lambda: section4_dictionary(
                df, tcols, R, written
            )),
            ("5", lambda: section5_redundancy(
                df, enc, groups, R, written
            )),
            ("6", lambda: section6_leakage(
                enc, y, groups,skipped, notes, R, written
            )),
            ("7", lambda: section7_validation(
                y, R, written
            )),
        ]

        failed = []

        for name, fn in steps:
            try:
                fn()
            except Exception:
                failed.append(name)
                R(
                    f"\n!!! Section {name} failed. "
                    "Traceback:"
                )
                R(traceback.format_exc())

        R.section("DONE")

        hash_after = sha256_of(data_path)
        R(
            "Data file unchanged: "
            f"{'YES' if hash_after == hash_before else 'NO - INVESTIGATE'}"
        )
        R(
            f"Sections that failed: "
            f"{failed if failed else 'none'}"
        )

        R("Files written:")
        for path in written:
            R(f"  {path}")

    except Exception:
        R("\nFATAL ERROR:")
        R(traceback.format_exc())
        raise

    finally:
        R.close()


if __name__ == "__main__":
    main()