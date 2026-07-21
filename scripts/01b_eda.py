"""
01b_eda.py

Purpose:
    Exploratory Data Analysis on the cleaned CFSDS BC/AB dataset
    (output of 02_clean.py). Produces summary statistics and saves
    a set of plots to eda_outputs/ for use in the report/presentation.

    This is meant to run AFTER 02_clean.py, on already-cleaned data —
    the goal here is understanding and communicating patterns, not
    fixing data quality issues (that happened in cleaning).

Input:
    processed/cfsds_bc_ab_clean.csv

Output:
    eda_outputs/
        summary_stats.csv
        missing_values.png
        sprdistm_distribution.png
        sprdistm_by_province.png
        fires_by_year.png
        fire_locations_map.png
        top_correlations_with_target.png
        correlation_heatmap.png
        scatter_matrix.png

Requirements:
    pip install pandas numpy matplotlib seaborn
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIG — paths anchored to project root, independent of where you run from
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CLEAN_FILE = os.path.join(PROJECT_ROOT, "processed", "cfsds_bc_ab_clean.csv")
EDA_DIR = os.path.join(PROJECT_ROOT, "eda_outputs")

TARGET_COL = "sprdistm"
YEAR_COL = "year"          # adjust if your actual column is named differently
PROVINCE_COL = "province"
LAT_CANDIDATES = ["lat", "latitude", "Latitude", "Lat", "y", "Y"]
LON_CANDIDATES = ["lon", "long", "longitude", "Longitude", "Lon", "x", "X"]

# Columns used for the scatter matrix — target plus a handful of covariates.
# Adjust these to match whichever columns matter most for your analysis;
# keep the list short (4-6 columns), otherwise the matrix becomes unreadable.
SCATTER_MATRIX_COLS = [TARGET_COL, "isi", "bui", "fwi", "ws"]

sns.set_style("whitegrid")


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Input file not found: {path}\nDid you run 02_clean.py first?"
        )
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df   

def save_summary_stats(df, out_dir):
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().T
    summary["missing_pct"] = df[numeric_df.columns].isna().mean() * 100
    path = os.path.join(out_dir, "summary_stats.csv")
    summary.to_csv(path)
    print(f"\nSaved summary statistics to {path}")
    print("\nQuick look at the target variable:")
    print(numeric_df[TARGET_COL].describe().to_string())


def plot_missing_values(df, out_dir):
    """Bar chart of missing-value percentage per column (top 20)."""
    missing = df.isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]

    if missing.empty:
        print("\nNo missing values remain in the cleaned dataset — skipping missing-values plot.")
        return

    top = missing.head(20)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top.index[::-1], top.values[::-1], color="slategray")
    ax.set_title("Missing values by column (%)")
    ax.set_xlabel("% missing")
    plt.tight_layout()
    path = os.path.join(out_dir, "missing_values.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    print("\nMissing values by column (%):")
    print(top.round(2).to_string())


def plot_target_distribution(df, out_dir):
    median = df[TARGET_COL].median()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(df[TARGET_COL], bins=50, ax=axes[0], color="firebrick")
    axes[0].axvline(median, color="red", linestyle="--", label=f"median = {median:.1f}")
    axes[0].set_title(f"{TARGET_COL} distribution (linear scale)")
    axes[0].set_xlabel("Spread distance (m/day)")
    axes[0].legend()

    # Log scale helps if the distribution is heavily right-skewed, which is
    # common for spread distance (many small/moderate days, a few extreme ones)
    positive = df[df[TARGET_COL] > 0][TARGET_COL]
    log_median = np.log1p(median) if median > 0 else np.log1p(positive.median())
    sns.histplot(np.log1p(positive), bins=50, ax=axes[1], color="darkorange")
    axes[1].axvline(log_median, color="red", linestyle="--", label=f"median = {log_median:.2f}")
    axes[1].set_title(f"log(1 + {TARGET_COL}) distribution")
    axes[1].set_xlabel("log(1 + spread distance)")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "sprdistm_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_by_province(df, out_dir, province_col):
    if province_col not in df.columns:
        print(f"[NOTE] '{province_col}' column not found, skipping province plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x=province_col, y=TARGET_COL, ax=ax, showfliers=False)
    ax.axhline(df[TARGET_COL].median(), color="red", linestyle="--", label="overall median")
    ax.set_title(f"{TARGET_COL} by province (outliers hidden for readability)")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "sprdistm_by_province.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_fires_by_year(df, out_dir, year_col):
    """Bar chart of fire-day record counts per year, via value_counts()."""
    if year_col not in df.columns:
        print(f"[NOTE] '{year_col}' column not found, skipping year plot.")
        return

    counts = df[year_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title("Number of fire-day records by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, "fires_by_year.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    print("\nFire-day record counts by year:")
    print(counts.to_string())


def plot_fire_locations(df, out_dir):
    lat_col = find_column(df, LAT_CANDIDATES)
    lon_col = find_column(df, LON_CANDIDATES)
    if lat_col is None or lon_col is None:
        print("[NOTE] Lat/lon columns not found, skipping location map.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    hue_col = PROVINCE_COL if PROVINCE_COL in df.columns else None
    sns.scatterplot(
        data=df, x=lon_col, y=lat_col, hue=hue_col, s=6, alpha=0.4, ax=ax
    )
    ax.set_title("Fire-day locations (BC & AB)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    path = os.path.join(out_dir, "fire_locations_map.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_top_correlations(df, out_dir, target_col, top_n=15):
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        print(f"[NOTE] '{target_col}' not numeric or missing, skipping correlation plot.")
        return

    corr = numeric_df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    top = corr.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["firebrick" if v < 0 else "steelblue" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} covariates correlated with {target_col}")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    path = os.path.join(out_dir, "top_correlations_with_target.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    print(f"\nTop {top_n} correlations with {target_col}:")
    print(top.round(3).to_string())


def plot_correlation_heatmap(df, out_dir, max_cols=25):
    """Full correlation heatmap. If there are many numeric columns, keep
    only the ones most correlated with the target so the plot stays readable."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > max_cols and TARGET_COL in numeric_df.columns:
        top_cols = (
            numeric_df.corr()[TARGET_COL]
            .abs()
            .sort_values(ascending=False)
            .head(max_cols)
            .index
        )
        numeric_df = numeric_df[top_cols]
        print(
            f"\n[NOTE] Too many numeric columns for a readable heatmap — "
            f"showing only the top {max_cols} most correlated with {TARGET_COL}."
        )

    #corr = numeric_df.corr()
    corr = numeric_df.corr(method="spearman")

    corr = corr.loc[
        corr[TARGET_COL].abs().sort_values(ascending=False).index[:15],
        corr[TARGET_COL].abs().sort_values(ascending=False).index[:15],
    ]

    fig, ax = plt.subplots(figsize=(0.5 * len(corr.columns) + 3, 0.5 * len(corr.columns) + 2))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, annot=False, square=True,
        cbar_kws={"shrink": 0.7}, ax=ax
    )
    ax.set_title("Correlation heatmap")
    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_scatter_matrix(df, out_dir, cols):
    present = [c for c in cols if c in df.columns]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"\n[NOTE] Scatter matrix columns not found, skipping them: {missing_cols}")
    if len(present) < 2:
        print("[NOTE] Not enough valid columns for a scatter matrix, skipping.")
        return

    hue_col = PROVINCE_COL if PROVINCE_COL in df.columns else None
    plot_cols = present + ([hue_col] if hue_col else [])
    sample = df[plot_cols]

    # Subsample for speed/readability if the dataset is large
    if len(sample) > 3000:
        sample = sample.sample(3000, random_state=42)
        print("[NOTE] Subsampled to 3000 rows for the scatter matrix (performance).")

    g = sns.pairplot(sample, vars=present, hue=hue_col, diag_kind="hist", plot_kws={"alpha": 0.4, "s": 10})
    g.fig.suptitle("Scatter matrix: target vs. key covariates", y=1.02)
    path = os.path.join(out_dir, "scatter_matrix.png")
    g.savefig(path, dpi=150)
    plt.close("all")
    print(f"Saved {path}")


def main():
    os.makedirs(EDA_DIR, exist_ok=True)
    df = load_data(CLEAN_FILE)

    save_summary_stats(df, EDA_DIR)
    plot_missing_values(df, EDA_DIR)
    plot_target_distribution(df, EDA_DIR)
    plot_by_province(df, EDA_DIR, PROVINCE_COL)
    plot_fires_by_year(df, EDA_DIR, YEAR_COL)
    plot_fire_locations(df, EDA_DIR)
    plot_top_correlations(df, EDA_DIR, TARGET_COL)
    plot_correlation_heatmap(df, EDA_DIR)
    plot_scatter_matrix(df, EDA_DIR, SCATTER_MATRIX_COLS)

    print(f"\nAll EDA outputs saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()