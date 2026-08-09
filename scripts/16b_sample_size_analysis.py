"""
Task 3b (diagnostic) — False negative and false positive error analysis
16b_sample_size_analysis.py
Check — sample size behind ecozone false-negative and false-positive rates
"""
import pandas as pd

eco_cols = [c for c in analysis_df.columns if c.startswith("eco_")]
if eco_cols:
    for col in eco_cols:
        sub = analysis_df[analysis_df[col] == 1]
        n_true_pos_days = (sub["y_true"] == 1).sum()
        n_fn = ((sub["y_true"] == 1) & (sub["y_pred"] == 0)).sum()
        n_true_neg_days = (sub["y_true"] == 0).sum()
        n_fp = ((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum()
        print(f"{col}: n_true_high_spread_days={n_true_pos_days}, n_missed={n_fn}, "
              f"n_true_low_spread_days={n_true_neg_days}, n_false_alarms={n_fp}")
elif "ecozone" in analysis_df.columns:
    grp_pos = analysis_df[analysis_df["y_true"] == 1].groupby("ecozone").size()
    grp_neg = analysis_df[analysis_df["y_true"] == 0].groupby("ecozone").size()
    print("True high-spread day counts by ecozone:")
    print(grp_pos)
    print("\nTrue low-spread day counts by ecozone:")
    print(grp_neg)