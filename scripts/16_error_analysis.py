"""
Task 3b (diagnostic) — False negative and false positive error analysis
16_error_analysis.py
"""
import pandas as pd
import numpy as np
import joblib
from scipy import stats

CHOSEN_THRESHOLD = 0.212

X_test  = pd.read_csv("processed/X_test_tree.csv")
y_test_clf = pd.read_csv("processed/y_test_clf.csv").iloc[:, 0]

X_test["ecozone"] = X_test["ecozone"].astype("category")

lgbm_clf = joblib.load("models/lgbm_classifier.pkl")
y_proba = lgbm_clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= CHOSEN_THRESHOLD).astype(int)

analysis_df = X_test.copy()
analysis_df["y_true"] = y_test_clf.values
analysis_df["y_pred"] = y_pred
analysis_df["proba"] = y_proba

fn_mask = (analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 0)
tp_mask = (analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 1)
fp_mask = (analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 1)
tn_mask = (analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 0)

false_negatives = analysis_df[fn_mask]
true_positives  = analysis_df[tp_mask]
false_positives = analysis_df[fp_mask]
true_negatives  = analysis_df[tn_mask]

print("False negatives (missed high-spread days):", len(false_negatives))
print("True positives (correctly caught):", len(true_positives))
print("False positives (false alarms):", len(false_positives))
print("True negatives (correctly rejected):", len(true_negatives))

print("\n-- False negatives - proba stats --")
print(false_negatives["proba"].describe())
print("\n-- True positives - proba stats --")
print(true_positives["proba"].describe())
print("\n-- False positives - proba stats --")
print(false_positives["proba"].describe())

eco_cols = [c for c in analysis_df.columns if c.startswith("eco_")]
if eco_cols:
    print("\n-- False negative rate by ecozone --")
    for col in eco_cols:
        sub = analysis_df[analysis_df[col] == 1]
        n_pos = (sub["y_true"] == 1).sum()
        if n_pos > 0:
            fn_rate = ((sub["y_true"] == 1) & (sub["y_pred"] == 0)).sum() / n_pos
            print(f"{col}: FN rate = {fn_rate:.2%} (n_true_positive_days={n_pos})")

    print("\n-- False positive rate by ecozone --")
    for col in eco_cols:
        sub = analysis_df[analysis_df[col] == 1]
        n_neg = (sub["y_true"] == 0).sum()
        if n_neg > 0:
            fp_rate = ((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum() / n_neg
            print(f"{col}: FP rate = {fp_rate:.2%} (n_true_negative_days={n_neg})")
elif "ecozone" in analysis_df.columns:
    print("\n-- False negative rate by ecozone --")
    grp = analysis_df[analysis_df["y_true"] == 1].groupby("ecozone", observed=True).apply(
        lambda d: (d["y_pred"] == 0).mean(), include_groups=False
    )
    print(grp.sort_values(ascending=False))

    print("\n-- False positive rate by ecozone --")
    grp_fp = analysis_df[analysis_df["y_true"] == 0].groupby("ecozone", observed=True).apply(
        lambda d: (d["y_pred"] == 1).mean(), include_groups=False
    )
    print(grp_fp.sort_values(ascending=False))

if "fireday_sin" in analysis_df.columns and "fireday_cos" in analysis_df.columns:
    analysis_df["fireday_approx"] = (
        np.arctan2(analysis_df["fireday_sin"], analysis_df["fireday_cos"]) / (2 * np.pi) * 365
    ) % 365
    analysis_df["month_bin"] = pd.cut(analysis_df["fireday_approx"], bins=12, labels=False)

    print("\n-- False negative rate by approx fire-day (binned into months) --")
    grp = analysis_df[analysis_df["y_true"] == 1].groupby("month_bin", observed=True).apply(
        lambda d: (d["y_pred"] == 0).mean(), include_groups=False
    )
    print(grp)

    print("\n-- False positive rate by approx fire-day (binned into months) --")
    grp_fp = analysis_df[analysis_df["y_true"] == 0].groupby("month_bin", observed=True).apply(
        lambda d: (d["y_pred"] == 1).mean(), include_groups=False
    )
    print(grp_fp)

numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in
                ["y_true", "y_pred", "proba", "fireday_approx", "month_bin"]]

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.std()**2 + (n2 - 1) * group2.std()**2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0.0

def benjamini_hochberg(p_values, alpha=0.05):
    """Returns adjusted p-values and a significance mask at the given FDR level."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value down
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    p_adj = np.empty(n)
    p_adj[order] = adjusted
    return p_adj, p_adj < alpha

def significance_table(group_a, group_b, cols):
    rows = []
    for col in cols:
        a, b = group_a[col].dropna(), group_b[col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)
        rows.append({
            "feature": col,
            "mean_group_a": a.mean(),
            "mean_group_b": b.mean(),
            "cohens_d": d,
            "abs_cohens_d": abs(d),
            "p_value": p_val,
        })
    df = pd.DataFrame(rows)
    df["p_adj"], df["significant_fdr_05"] = benjamini_hochberg(df["p_value"].values)
    return df.sort_values("abs_cohens_d", ascending=False)

fn_sig = significance_table(false_negatives, true_positives, numeric_cols)
fn_sig = fn_sig.rename(columns={"mean_group_a": "false_negative_mean", "mean_group_b": "true_positive_mean"})
print("\n-- Feature comparison: false negatives vs true positives --")
print("Ranked by standardized effect size (Cohen's d), significance FDR-corrected across all features tested")
print(fn_sig.head(15).to_string(index=False))
print(f"\nFeatures surviving FDR correction (p_adj < 0.05): {fn_sig['significant_fdr_05'].sum()} of {len(fn_sig)}")

fp_sig = significance_table(false_positives, true_negatives, numeric_cols)
fp_sig = fp_sig.rename(columns={"mean_group_a": "false_positive_mean", "mean_group_b": "true_negative_mean"})
print("\n-- Feature comparison: false positives vs true negatives --")
print("Ranked by standardized effect size (Cohen's d), significance FDR-corrected across all features tested")
print(fp_sig.head(15).to_string(index=False))
print(f"\nFeatures surviving FDR correction (p_adj < 0.05): {fp_sig['significant_fdr_05'].sum()} of {len(fp_sig)}")
