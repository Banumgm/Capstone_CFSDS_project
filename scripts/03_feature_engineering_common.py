"""
03_feature_engineering_common.py
Create common derived features (shared step before branching into linear/tree)

Notes:
  - ws_x_vpd interaction not created: vpd is excluded due to an unresolved
    scale/unit mismatch in CFSDS v1.1 beta data starting 2022 (see
    DATA_QUALITY_RISK_COLS below).
  - vpd and d_vpd flagged here as data-quality risks (actually dropped in
    Task 3/4 -- 05_feature_engineering_linear.py, 06_feature_engineering_tree.py).
  - prevgrow flagged here as a missing-data encoding artifact, not a
    leakage issue (see decisions_log.md for the full investigation).
"""
import os
import numpy as np
import pandas as pd

# Auto-detect environment: Databricks vs local. Databricks always has
# /Workspace mounted; a local machine (VS Code, etc.) does not. Using the
# same file everywhere on both Databricks and GitHub avoids maintaining
# two divergent copies of the same script.
if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

CLEAN_FILE = f"{BASE_DIR}/cfsds_bc_ab_clean.csv"
COMMON_OUT_FILE = f"{BASE_DIR}/cfsds_features_common.csv"

df = pd.read_csv(CLEAN_FILE)

# ecozone: restore integer category codes (was cast to float on CSV save/reload)
df["ecozone"] = df["ecozone"].astype(int).astype("category")

# --- Cyclical encoding ---
df["fireday_sin"] = np.sin(2 * np.pi * df["fireday"] / 365)
df["fireday_cos"] = np.cos(2 * np.pi * df["fireday"] / 365)

df["aspect_sin"] = np.sin(2 * np.pi * df["aspect"] / 360)
df["aspect_cos"] = np.cos(2 * np.pi * df["aspect"] / 360)

# --- Interaction terms ---
# ws x slope: well-supported by fire-behavior literature (wind-slope alignment
# drives flame attachment / spread acceleration). Keep by default.
df["ws_x_slope"] = df["ws"] * df["slope"]

# ws x vpd: not created -- vpd excluded from modeling (see below).

# aspect x fireday: south-facing slopes dry out faster, effect is seasonal.
df["aspect_x_season"] = df["aspect_cos"] * df["fireday_cos"]

# --- Flag potential data-leakage columns ---
LEAKAGE_RISK_COLS = ["cumuarea", "pctgrowth"]
print(f"[NOTE] Leakage-risk columns (closely tied to target={df['sprdistm'].name} by definition): {LEAKAGE_RISK_COLS}")
print("-> Excluding from model inputs until source documentation (Barber et al. 2024) confirms timing.")

# --- Flag data-quality issue: prevgrow missing-data encoding artifact ---
# Initially kept as a legitimate lagged predictor (VIF = 1.09, no
# multicollinearity -- see decisions_log.md), but empirically restoring it
# degraded Tweedie CV RMSE (721 -> 793). Root cause: prevgrow = 0 for 100%
# of fireday==1 records (no true "previous day" on a fire's first day),
# conflating "genuinely no prior growth" with "value unavailable" under
# the same encoded value. Re-excluded; not a leakage or collinearity issue.
DATA_QUALITY_RISK_COLS_PREVGROW = ["prevgrow"]
print(f"[DATA QUALITY WARNING] {DATA_QUALITY_RISK_COLS_PREVGROW}: missing-data encoding "
      f"artifact (0 for 100% of fireday==1 records). Excluding from model inputs.")

# --- Flag data-quality issue: vpd scale mismatch from 2022 onward ---
# vpd drops from a historical mean of ~16 (2002-2021) to ~1.3-1.6 (2022-2024),
# a ~10-15x scale reduction, while its correlation structure with tmax/rh stays
# consistent -- suggesting a unit/formula change in the CFSDS v1.1 beta pipeline
# rather than a genuine climate shift. No single correction factor could be
# confirmed. d_tmax has a separate, already-corrected Celsius/Kelvin bug
# (see 04_splits.py, fix_dtmax_kelvin_bug function) and is NOT included here.
DATA_QUALITY_RISK_COLS = ["vpd", "d_vpd"]
print(f"[DATA QUALITY WARNING] {DATA_QUALITY_RISK_COLS}: unresolved scale/unit mismatch "
      f"starting 2022 (CFSDS v1.1 beta). Excluding from model inputs.")

print("\n[NOTE] Interaction term confidence levels:")
print("  ws_x_slope       : well-supported by fire-behavior literature, keep by default")
print("  aspect_x_season  : exploratory, validate via ablation before treating as confirmed")

df.to_csv(COMMON_OUT_FILE, index=False)
print(f"\nSaved: {COMMON_OUT_FILE}, shape={df.shape}")