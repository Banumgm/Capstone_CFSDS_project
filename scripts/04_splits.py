"""
04_splits.py
Create temporal and spatial holdout splits from the common feature set,
split at the fire (ID) level to prevent leakage, and apply the confirmed
d_tmax unit-mismatch correction before saving.
"""
import os
import pandas as pd

# Auto-detect environment: Databricks vs local. Databricks always has
# /Workspace mounted; a local machine (VS Code, etc.) does not. Using the
# same file everywhere on both Databricks and GitHub avoids maintaining
# two divergent copies of the same script.
if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

COMMON_OUT_FILE = f"{BASE_DIR}/cfsds_features_common.csv"
df_common = pd.read_csv(COMMON_OUT_FILE)

FIRE_ID_COL = "ID"

# --- Normalize province strings (defensive) ---
df_common["province"] = df_common["province"].astype(str).str.strip()
BC_LABEL = "British Columbia"
AB_LABEL = "Alberta"

# --- Temporal holdout, split at the fire level ---
fire_year = df_common.groupby(FIRE_ID_COL)["year"].min()
train_fire_ids_temporal = fire_year[fire_year <= 2018].index
test_fire_ids_temporal  = fire_year[fire_year >= 2019].index

train_temporal = df_common[df_common[FIRE_ID_COL].isin(train_fire_ids_temporal)].copy()
test_temporal  = df_common[df_common[FIRE_ID_COL].isin(test_fire_ids_temporal)].copy()

overlap_temporal = set(train_temporal[FIRE_ID_COL]) & set(test_temporal[FIRE_ID_COL])
assert len(overlap_temporal) == 0, f"[ERROR] {len(overlap_temporal)} fire IDs leaked across temporal train/test"
print(f"Temporal split (fire-level) -> train: {train_temporal.shape}, test: {test_temporal.shape}")

# --- Spatial holdout, split at the fire level, excluding border-crossing fires ---
province_per_fire = df_common.groupby(FIRE_ID_COL)["province"].nunique()
multi_province_fire_ids = province_per_fire[province_per_fire > 1].index
print(f"[CHECK] Fires spanning multiple provinces: {len(multi_province_fire_ids)}")

df_spatial_eligible = df_common[~df_common[FIRE_ID_COL].isin(multi_province_fire_ids)]
fire_province = df_spatial_eligible.groupby(FIRE_ID_COL)["province"].first()

train_fire_ids_spatial = fire_province[fire_province == BC_LABEL].index
test_fire_ids_spatial  = fire_province[fire_province == AB_LABEL].index

train_spatial = df_spatial_eligible[df_spatial_eligible[FIRE_ID_COL].isin(train_fire_ids_spatial)].copy()
test_spatial  = df_spatial_eligible[df_spatial_eligible[FIRE_ID_COL].isin(test_fire_ids_spatial)].copy()

overlap_spatial = set(train_spatial[FIRE_ID_COL]) & set(test_spatial[FIRE_ID_COL])
assert len(overlap_spatial) == 0, f"[ERROR] {len(overlap_spatial)} fire IDs leaked across spatial train/test"
print(f"Spatial split (fire-level, border-crossing fires excluded) -> train: {train_spatial.shape}, test: {test_spatial.shape}")

# --- Apply confirmed d_tmax Celsius/Kelvin unit-mismatch correction ---
def fix_dtmax_kelvin_bug(df, threshold=30):
    """Fix rows where |d_tmax| > threshold, consistent with a Celsius/Kelvin
    mismatch (~273.15 offset) in the previous-day tmax lookup (CFSDS v1.1 beta, 2022+)."""
    mask = df["d_tmax"].abs() > threshold
    n_fixed = mask.sum()
    tmax_prev_implied = df.loc[mask, "d_tmax"] + df.loc[mask, "tmax"]
    tmax_prev_corrected = tmax_prev_implied - 273.15
    df.loc[mask, "d_tmax"] = df.loc[mask, "tmax"] - tmax_prev_corrected
    print(f"Fixed {n_fixed} rows with Kelvin/Celsius mismatch in d_tmax.")
    return df

train_temporal = fix_dtmax_kelvin_bug(train_temporal)
test_temporal  = fix_dtmax_kelvin_bug(test_temporal)
train_spatial  = fix_dtmax_kelvin_bug(train_spatial)
test_spatial   = fix_dtmax_kelvin_bug(test_spatial)

print("\nd_tmax by year after fix (temporal test):")
print(test_temporal.groupby("year")["d_tmax"].agg(["min", "max", "mean"]))

# --- Save ---
TRAIN_FILE = f"{BASE_DIR}/train_temporal.csv"
TEST_FILE  = f"{BASE_DIR}/test_temporal.csv"
train_temporal.to_csv(TRAIN_FILE, index=False)
test_temporal.to_csv(TEST_FILE, index=False)
train_spatial.to_csv(f"{BASE_DIR}/train_spatial.csv", index=False)
test_spatial.to_csv(f"{BASE_DIR}/test_spatial.csv", index=False)

print(f"\nSaved: {TRAIN_FILE}, {TEST_FILE}")
print("Saved: train_spatial.csv, test_spatial.csv")