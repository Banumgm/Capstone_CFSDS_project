"""
Task 3a (diagnostic) — Sample size check for province threshold sensitivity
22b_province_threshold_sample_check.py
Confirms whether the province-level threshold difference found in
11c is backed by sufficient sample size, or could be a small-sample
artifact akin to the ecozone false-negative rates in the error analysis.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

train_df = pd.read_csv(f"{BASE_DIR}/train_temporal.csv")
TARGET = "sprdistm"

print("--- Sample size by province (train) ---")
counts = train_df["province"].value_counts()
print(counts.to_string())
print(f"\nTotal train rows: {len(train_df):,}")
print(f"Split: {(counts / len(train_df) * 100).round(1).to_string()}")

# Bootstrap the 90th percentile per province to check estimate stability
print("\n--- Bootstrap stability check (1,000 resamples per province) ---")
rng = np.random.default_rng(42)
for prov in train_df["province"].unique():
    vals = train_df.loc[train_df["province"] == prov, TARGET].values
    n = len(vals)
    boot_qs = [
        np.quantile(rng.choice(vals, size=n, replace=True), 0.90)
        for _ in range(1000)
    ]
    point_est = np.quantile(vals, 0.90)
    ci_low, ci_high = np.percentile(boot_qs, [2.5, 97.5])
    print(f"{prov} (n={n:,}): 90th pct = {point_est:,.2f}, "
          f"95% bootstrap CI = [{ci_low:,.2f}, {ci_high:,.2f}]")