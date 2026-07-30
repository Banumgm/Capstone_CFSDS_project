"""
05_feature_engineering_linear.py
Linear/GLM branch: one-hot encode ecozone + StandardScaler (fit on train only).
DROP_COLS reflects the final feature set: firearea, FWI delta columns,
vpd/d_vpd, and prevgrow all excluded (see decisions_log.md for rationale
on each).
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

TRAIN_FILE = f"{BASE_DIR}/train_temporal.csv"
TEST_FILE  = f"{BASE_DIR}/test_temporal.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

train_df = pd.get_dummies(train_df, columns=["ecozone"], prefix="eco", drop_first=True)
test_df  = pd.get_dummies(test_df,  columns=["ecozone"], prefix="eco", drop_first=True)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

dummy_cols_created = [c for c in train_df.columns if c.startswith("eco_")]
train_df[dummy_cols_created] = train_df[dummy_cols_created].astype(int)
test_df[dummy_cols_created]  = test_df[dummy_cols_created].astype(int)

TARGET = "sprdistm"
DROP_COLS = ["ID", "source_file", "fireday", "aspect", "province", TARGET,
             "cumuarea", "pctgrowth", "prevgrow",  # prevgrow re-excluded:
             # missing-data encoding artifact (0 for 100% of fireday==1
             # records), not leakage — see decisions_log.md
             "firearea",
             "vpd", "d_vpd",
             "d_fwi", "d_isi", "d_ffmc", "d_dmc", "d_dc", "d_bui"]

feature_cols_linear = [c for c in train_df.columns if c not in DROP_COLS]
dummy_cols = [c for c in feature_cols_linear if c.startswith("eco_")]
continuous_cols = [c for c in feature_cols_linear if c not in dummy_cols]

X_train_lin, y_train = train_df[feature_cols_linear], train_df[TARGET]
X_test_lin,  y_test  = test_df[feature_cols_linear],  test_df[TARGET]

scaler = StandardScaler()
X_train_scaled = X_train_lin.copy()
X_test_scaled  = X_test_lin.copy()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train_lin[continuous_cols])
X_test_scaled[continuous_cols]  = scaler.transform(X_test_lin[continuous_cols])

X_train_scaled = X_train_scaled.astype(float)
X_test_scaled  = X_test_scaled.astype(float)

print(f"Scaled {len(continuous_cols)} continuous columns, left {len(dummy_cols)} dummy columns untouched.")
print(f"Dtype check -> {X_train_scaled.dtypes.unique()}")

X_train_scaled.to_csv(f"{BASE_DIR}/X_train_linear.csv", index=False)
X_test_scaled.to_csv(f"{BASE_DIR}/X_test_linear.csv", index=False)
y_train.to_csv(f"{BASE_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{BASE_DIR}/y_test.csv", index=False)

print(f"\nX_train_linear: {X_train_scaled.shape}, X_test_linear: {X_test_scaled.shape}")