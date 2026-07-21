"""
02_clean.py

Purpose:
    Take the filtered BC/AB CFSDS subset produced by 01_filter_bc_ab.py
    and prepare it for modeling:
      1. Inspect and handle missing values (with a saved report)
      2. Check for and remove duplicate rows
      3. Validate physically plausible ranges for key variables
      4. Verify the peatland-related features: use 'peatprop' (proportion
         of the burn day in true peatland classes) as the primary feature,
         and correctly interpret 'peattype' (codes 1-4 = peatland,
         5-9 = other valid land-cover classes, NOT peatland) if kept
      5. Encode categorical variables (ecozone) as category dtype (actual
         one-hot/ordinal encoding happens in 03_feature_engineering.py,
         not here)
      6. Check multicollinearity among FWI System indices
         (Spearman, with Pearson reported alongside for comparison)
      7. Save a clean, model-ready CSV plus supporting report files

Input:
    processed/cfsds_bc_ab_2002_2024.csv   (output of 01_filter_bc_ab.py)

Output:
    processed/cfsds_bc_ab_clean.csv
    processed/missing_report.csv
    processed/fwi_correlation_matrix.csv
    processed/fwi_high_corr_pairs.csv

Requirements:
    pip install pandas numpy scikit-learn
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — paths anchored to project root, independent of where you run from
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, "processed", "cfsds_bc_ab_2002_2024.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "cfsds_bc_ab_clean.csv")
MISSING_REPORT_FILE = os.path.join(OUTPUT_DIR, "missing_report.csv")
CORR_FILE = os.path.join(OUTPUT_DIR, "fwi_correlation_matrix.csv")
HIGH_CORR_FILE = os.path.join(OUTPUT_DIR, "fwi_high_corr_pairs.csv")

# Your chosen target variable (see proposal, Section 3)
TARGET_COL = "sprdistm"

# Columns that should NEVER be median-imputed:
#   - TARGET_COL: rows with a missing target are dropped, not imputed
#   - lat/lon: imputing coordinates with a median produces a physically
#     meaningless "average location" and can silently corrupt any
#     downstream spatial logic (e.g. province/ecozone joins)
#   - year: a categorical/temporal identifier, not a continuous quantity
IMPUTE_EXCLUDE_COLS = [TARGET_COL, "lat", "lon", "year"]

# Categorical columns to encode. Adjust names if your actual columns differ
# (run the script once, check the printed column list if it errors out).
CATEGORICAL_COLS = ["ecozone"]

# CFSDS peatland-related columns (see Barber et al. 2024, Table 3):
#   - peattype: pixel land-cover class as predicted by Pontone et al. (2023),
#     mode of values for the burn day. Classes 1-4 are true peatland types
#     (Bog, Rich Fen, Poor Fen, Peatland Permafrost Complex); classes 5-9
#     are NOT peatland (Mineral Wetlands, Water, Uplands, Agriculture,
#     Urban). A missing value here does NOT mean "non-peatland" — it means
#     no land-cover class was assigned at all, so it must not be used to
#     derive an is_peatland flag via notna().
#   - peatprop: proportion of a burn day's area that fell in the true
#     peatland classes (1-4) combined. This is already the correct,
#     continuous feature for "how much of this fire-day was in peatland",
#     and (unlike peattype) is reported with little to no missingness —
#     so we use peatprop directly rather than deriving anything from
#     peattype.
PEATPROP_COL = "peatprop"
PEATTYPE_COL = "peattype"
PEATLAND_CLASS_CODES = [1, 2, 3, 4]  # Bog, Rich Fen, Poor Fen, Peatland Permafrost Complex

# FWI System indices — used for the multicollinearity check (H1 in proposal
# relies on comparing these against topographic/anthropogenic covariates,
# so it's worth knowing up front how correlated they are with each other).
FWI_COLS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]

# Plausible physical ranges for sanity-checking key variables.
# (column, min, max) — max=None means "no upper bound to check"
RANGE_CHECKS = [
    ("ffmc", 0, 101),      # Fine Fuel Moisture Code is defined on 0-101
    ("isi", 0, None),      # Initial Spread Index cannot be negative
    ("bui", 0, None),      # Buildup Index cannot be negative
    ("fwi", 0, None),      # Fire Weather Index cannot be negative
    (TARGET_COL, 0, None), # Spread distance cannot be negative
    ("peatprop", 0, 1),    # Proportion of burn day in peatland classes
    ("nonfuel1k", 0, 1),   # Proportion of landscape that is nonfuel (1km)
    ("nonfuel2k", 0, 1),   # Proportion of landscape that is nonfuel (2km)
    ("nonfuel5k", 0, 1),   # Proportion of landscape that is nonfuel (5km)
    ("nonfuel10k", 0, 1),  # Proportion of landscape that is nonfuel (10km)
]

# Missing-value strategy per column type
NUMERIC_MISSING_STRATEGY = "median"   # "median", "mean", or "drop"

# Drop columns missing more than this fraction of values.
# 0.4 (40%) is a commonly used rule of thumb: columns missing more than
# this are usually too sparse to impute reliably without introducing bias,
# while columns below this threshold still carry enough real information
# to be worth keeping and imputing.
MAX_MISSING_FRACTION = 0.4


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Did you run 01_filter_bc_ab.py first?"
        )
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def report_missingness(df, out_path):
    missing_counts = df.isna().sum()
    missing_pct = df.isna().mean() * 100

    report = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct": missing_pct,
    }).sort_values("missing_pct", ascending=False)
    report.to_csv(out_path)

    n_total_missing = df.isna().sum().sum()
    n_cols_with_na = (missing_counts > 0).sum()
    n_rows_with_na = df.isna().any(axis=1).sum()

    print("\n--- Missingness summary ---")
    print(f"Total missing values : {n_total_missing:,}")
    print(f"Columns with NA      : {n_cols_with_na} / {df.shape[1]}")
    print(f"Rows with NA         : {n_rows_with_na:,} / {len(df):,}")

    nonzero = report[report["missing_pct"] > 0]
    if not nonzero.empty:
        print("\nTop columns by missing %:")
        print(nonzero.head(15).round(2).to_string())
    print(f"\nSaved full missing-value report to {out_path}")

    return report


def check_duplicates(df):
    n_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dupes}")
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"Dropped {n_dupes} duplicate row(s). New shape: {df.shape}")
    return df


def check_value_ranges(df, checks):
    print("\n--- Range validation ---")
    for col, min_val, max_val in checks:
        if col not in df.columns:
            print(f"[NOTE] '{col}' not found, skipping range check.")
            continue

        below = (df[col] < min_val).sum() if min_val is not None else 0
        above = (df[col] > max_val).sum() if max_val is not None else 0

        if below or above:
            range_desc = f"[{min_val}, {max_val if max_val is not None else '∞'}]"
            print(
                f"[WARNING] '{col}' expected range {range_desc}: "
                f"{below} value(s) below min, {above} value(s) above max"
            )
        else:
            print(f"'{col}': all values within expected range.")


def handle_peatland_features(df, peatprop_col, peattype_col, peatland_codes):
    """Check that peatprop (the correct, continuous peatland-proportion
    feature) is present and well-populated. Also correctly interprets
    peattype if kept: only codes in `peatland_codes` (1-4) are true
    peatland classes — codes 5-9 (Mineral Wetlands, Water, Uplands,
    Agriculture, Urban) are valid land-cover classes but NOT peatland,
    so peattype must never be treated as "present = peatland"."""
    if peatprop_col in df.columns:
        missing_pct = df[peatprop_col].isna().mean() * 100
        print(
            f"\n'{peatprop_col}' present — mean burn-day peatland proportion: "
            f"{df[peatprop_col].mean():.3f} ({missing_pct:.2f}% missing). "
            f"This is the recommended feature for peatland influence on spread."
        )
    else:
        print(f"\n[NOTE] '{peatprop_col}' not found in data.")

    if peattype_col in df.columns:
        is_true_peatland = df[peattype_col].isin(peatland_codes)
        print(
            f"'{peattype_col}' present — {is_true_peatland.sum()} rows "
            f"({is_true_peatland.mean():.1%}) have a dominant true-peatland "
            f"class (codes {peatland_codes}); remaining non-null rows fall "
            f"in other land-cover classes (water/uplands/agriculture/urban/"
            f"mineral wetlands), not peatland."
        )

    return df


def drop_high_missing_columns(df, missing_report, threshold):
    to_drop = missing_report[missing_report["missing_pct"] / 100 > threshold].index.tolist()
    if to_drop:
        print(
            f"\nDropping {len(to_drop)} column(s) with >{threshold:.0%} "
            f"missing values: {to_drop}"
        )
        df = df.drop(columns=to_drop)
    return df


def drop_missing_target(df, target_col):
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    print(f"\nDropped {dropped} rows with missing target ('{target_col}').")
    return df


def impute_numeric(df, strategy, exclude_cols):
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    n_before = df[numeric_cols].isna().sum().sum()

    if strategy == "drop":
        df = df.dropna(subset=numeric_cols)
    else:
        fill_values = (
            df[numeric_cols].median()
            if strategy == "median"
            else df[numeric_cols].mean()
        )
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)

    n_after = df[numeric_cols].isna().sum().sum()
    print(
        f"\nNumeric imputation ({strategy}, excluding {exclude_cols}): "
        f"{n_before} missing values -> {n_after} remaining."
    )
    return df


def encode_categoricals(df, cat_cols):
    """Store as pandas 'category' dtype. Actual encoding for modeling
    (one-hot / ordinal) is deliberately deferred to 03_feature_engineering.py,
    so this step stays reversible and doesn't blow up the file with dummy
    columns before we know which model needs which encoding."""
    present = [c for c in cat_cols if c in df.columns]
    missing_cols = [c for c in cat_cols if c not in df.columns]
    if missing_cols:
        print(f"\n[NOTE] Categorical columns not found, skipping: {missing_cols}")

    for col in present:
        df[col] = df[col].astype("category")
        if df[col].isna().any():
            df[col] = df[col].cat.add_categories(["unknown"]).fillna("unknown")

    print(f"\nEncoded as categorical dtype: {present}")
    return df, present


def check_fwi_multicollinearity(df, fwi_cols, corr_out_path, high_corr_out_path):
    present = [c for c in fwi_cols if c in df.columns]
    missing_cols = [c for c in fwi_cols if c not in df.columns]
    if missing_cols:
        print(f"\n[NOTE] FWI columns not found, skipping: {missing_cols}")
    if len(present) < 2:
        print("Not enough FWI columns present to compute correlations.")
        return

    spearman_corr = df[present].corr(method="spearman")
    pearson_corr = df[present].corr(method="pearson")

    print("\nFWI index correlation matrix (Spearman, primary):")
    print(spearman_corr.round(2).to_string())
    print("\nFWI index correlation matrix (Pearson, for comparison):")
    print(pearson_corr.round(2).to_string())

    high_corr_pairs = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            r_spearman = spearman_corr.iloc[i, j]
            if abs(r_spearman) > 0.85:
                high_corr_pairs.append({
                    "var_1": present[i],
                    "var_2": present[j],
                    "spearman_r": round(r_spearman, 3),
                    "pearson_r": round(pearson_corr.iloc[i, j], 3),
                })

    if high_corr_pairs:
        print(
            "\n[WARNING] Highly correlated FWI pairs (Spearman |r| > 0.85) — "
            "consider dropping one of each pair, or using regularization "
            "(e.g. Ridge) for the linear baseline model:"
        )
        for pair in high_corr_pairs:
            print(f"  {pair['var_1']} <-> {pair['var_2']}: "
                  f"spearman r = {pair['spearman_r']}, pearson r = {pair['pearson_r']}")
        pd.DataFrame(high_corr_pairs).to_csv(high_corr_out_path, index=False)
        print(f"Saved high-correlation pairs to {high_corr_out_path}")
    else:
        print("\nNo FWI pairs exceed the 0.85 Spearman correlation threshold.")

    spearman_corr.to_csv(corr_out_path)
    print(f"\nSaved Spearman correlation matrix to {corr_out_path}")


def print_final_summary(df):
    print("\n=== Final cleaned dataset ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn dtypes:")
    print(df.dtypes.value_counts().to_string())
    print("\nNumeric summary (first 5 columns shown):")
    print(df.select_dtypes(include=[np.number]).describe().T.head().to_string())


def main():
    df = load_data(INPUT_FILE)

    missing_report = report_missingness(df, MISSING_REPORT_FILE)
    df = check_duplicates(df)
    check_value_ranges(df, RANGE_CHECKS)

    df = handle_peatland_features(df, PEATPROP_COL, PEATTYPE_COL, PEATLAND_CLASS_CODES)
    df = drop_high_missing_columns(df, missing_report, MAX_MISSING_FRACTION)
    df = drop_missing_target(df, TARGET_COL)
    df = impute_numeric(df, NUMERIC_MISSING_STRATEGY, IMPUTE_EXCLUDE_COLS)
    df, encoded_cols = encode_categoricals(df, CATEGORICAL_COLS)
    check_fwi_multicollinearity(df, FWI_COLS, CORR_FILE, HIGH_CORR_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved clean dataset to {CLEAN_FILE}")

    print_final_summary(df)


if __name__ == "__main__":
    main()