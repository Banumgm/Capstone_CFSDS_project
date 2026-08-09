"""
Task 3b (diagnostic) — False negative and false positive error analysis
16_error_analysis.py
"""
import pandas as pd
import numpy as np
import joblib

CHOSEN_THRESHOLD = 0.027

X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_test_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

X_test["ecozone"] = X_test["ecozone"].astype("category")

lgbm_clf = joblib.load("/Workspace/Capstone_Group1/models/lgbm_classifier.pkl")
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
    grp = analysis_df[analysis_df["y_true"] == 1].groupby("ecozone").apply(
        lambda d: (d["y_pred"] == 0).mean()
    )
    print(grp.sort_values(ascending=False))

    print("\n-- False positive rate by ecozone --")
    grp_fp = analysis_df[analysis_df["y_true"] == 0].groupby("ecozone").apply(
        lambda d: (d["y_pred"] == 1).mean()
    )
    print(grp_fp.sort_values(ascending=False))
else:
    print("\nNo ecozone columns found — skipping ecozone breakdown.")

if "fireday_sin" in analysis_df.columns and "fireday_cos" in analysis_df.columns:
    analysis_df["fireday_approx"] = (
        np.arctan2(analysis_df["fireday_sin"], analysis_df["fireday_cos"]) / (2 * np.pi) * 365
    ) % 365
    analysis_df["month_bin"] = pd.cut(analysis_df["fireday_approx"], bins=12, labels=False)

    print("\n-- False negative rate by approx fire-day (binned into months) --")
    grp = analysis_df[analysis_df["y_true"] == 1].groupby("month_bin").apply(
        lambda d: (d["y_pred"] == 0).mean()
    )
    print(grp)

    print("\n-- False positive rate by approx fire-day (binned into months) --")
    grp_fp = analysis_df[analysis_df["y_true"] == 0].groupby("month_bin").apply(
        lambda d: (d["y_pred"] == 1).mean()
    )
    print(grp_fp)
else:
    print("\nNo fireday_sin/cos columns found — skipping seasonal breakdown.")

numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in
                ["y_true", "y_pred", "proba", "fireday_approx", "month_bin"]]

comparison = pd.DataFrame({
    "false_negative_mean": false_negatives[numeric_cols].mean(),
    "true_positive_mean": true_positives[numeric_cols].mean(),
})
comparison["diff"] = comparison["true_positive_mean"] - comparison["false_negative_mean"]
comparison["abs_diff"] = comparison["diff"].abs()
comparison = comparison.sort_values("abs_diff", ascending=False)

print("\n-- Feature means: false negatives vs true positives (top 15 by difference) --")
print(comparison.head(15).to_string())

# What's actually driving the false alarms — compares the days the model
# wrongly flagged against the low-spread days it correctly left alone.
comparison_fp = pd.DataFrame({
    "false_positive_mean": false_positives[numeric_cols].mean(),
    "true_negative_mean": true_negatives[numeric_cols].mean(),
})
comparison_fp["diff"] = comparison_fp["false_positive_mean"] - comparison_fp["true_negative_mean"]
comparison_fp["abs_diff"] = comparison_fp["diff"].abs()
comparison_fp = comparison_fp.sort_values("abs_diff", ascending=False)

print("\n-- Feature means: false positives vs true negatives (top 15 by difference) --")
print("(explains what makes the model over-alert on a day that turned out low-spread)")
print(comparison_fp.head(15).to_string())