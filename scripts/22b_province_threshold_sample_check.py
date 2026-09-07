"""
Task 3a (diagnostic) — Province threshold bootstrap stability check
22b_province_threshold_sample_check.py

Bootstraps the per-province 90th-percentile threshold to check whether
the difference between Alberta and British Columbia is stable, or could
be sampling noise. Resampling is done at the fire level, not the row
level: a single fire contributes multiple correlated burn-day rows, so
resampling individual rows would treat correlated observations as
independent and understate the true uncertainty. Each bootstrap
iteration resamples fire IDs with replacement and takes all rows
belonging to the resampled fires.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

train_df = pd.read_csv(f"{BASE_DIR}/train_temporal.csv")
TARGET = "sprdistm"
PERCENTILE = 0.90

print("--- Sample size by province (train) ---")
print(train_df["province"].value_counts().to_string())

rng = np.random.default_rng(42)
N_BOOTSTRAP = 1000

print(f"\n--- Fire-block bootstrap ({N_BOOTSTRAP} resamples per province) ---")
for prov in train_df["province"].unique():
    prov_df = train_df[train_df["province"] == prov]
    fire_ids = prov_df["ID"].unique()
    n_fires = len(fire_ids)

    boot_qs = []
    for _ in range(N_BOOTSTRAP):
        sampled_fire_ids = rng.choice(fire_ids, size=n_fires, replace=True)
        # take all rows belonging to each resampled fire (a fire sampled
        # twice contributes its rows twice)
        resampled_rows = prov_df[prov_df["ID"].isin(sampled_fire_ids)]
        # weight by how many times each fire was drawn, to preserve the
        # bootstrap's intended resampling distribution
        counts = pd.Series(sampled_fire_ids).value_counts()
        weighted_values = np.repeat(
            resampled_rows[TARGET].values,
            resampled_rows["ID"].map(counts).fillna(0).astype(int).values
        )
        if len(weighted_values) > 0:
            boot_qs.append(np.quantile(weighted_values, PERCENTILE))

    point_est = prov_df[TARGET].quantile(PERCENTILE)
    ci_low, ci_high = np.percentile(boot_qs, [2.5, 97.5])
    print(f"{prov} (n_fires={n_fires}, n_rows={len(prov_df)}): "
          f"90th pct = {point_est:,.2f}, "
          f"95% fire-block bootstrap CI = [{ci_low:,.2f}, {ci_high:,.2f}]")