"""
06_feature_engineering_tree.py
Tree branch: ecozone stays as native category dtype, no scaling applied.
DROP_COLS reflects the final feature set: firearea, FWI delta columns,
vpd/d_vpd, and prevgrow all excluded (see decisions_log.md for rationale
on each).
"""
import os
import pandas as pd

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

TRAIN_FILE = f"{BASE_DIR}/train_temporal.csv"
TEST_FILE  = f"{BASE_DIR}/test_temporal.csv"

train_df_tree = pd.read_csv(TRAIN_FILE)
test_df_tree  = pd.read_csv(TEST_FILE)

train_df_tree["ecozone"] = train_df_tree["ecozone"].astype(int).astype("category")
test_df_tree["ecozone"]  = test_df_tree["ecozone"].astype(int).astype("category")

TARGET = "sprdistm"
DROP_COLS = ["ID", "source_file", "fireday", "aspect", "province", TARGET,
             "cumuarea", "pctgrowth", "prevgrow",  # prevgrow re-excluded:
             # missing-data encoding artifact (0 for 100% of fireday==1
             # records), not leakage — see decisions_log.md
             "firearea",
             "vpd", "d_vpd",
             "d_fwi", "d_isi", "d_ffmc", "d_dmc", "d_dc", "d_bui"]

feature_cols_tree = [c for c in train_df_tree.columns if c not in DROP_COLS]

X_train_tree, y_train_tree = train_df_tree[feature_cols_tree], train_df_tree[TARGET]
X_test_tree,  y_test_tree  = test_df_tree[feature_cols_tree],  test_df_tree[TARGET]

X_train_tree.to_csv(f"{BASE_DIR}/X_train_tree.csv", index=False)
X_test_tree.to_csv(f"{BASE_DIR}/X_test_tree.csv", index=False)

print(f"X_train_tree: {X_train_tree.shape}, X_test_tree: {X_test_tree.shape}")
print(f"ecozone dtype: {X_train_tree['ecozone'].dtype}, number of categories: {X_train_tree['ecozone'].nunique()}")