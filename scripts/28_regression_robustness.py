"""
Phase 4, Step 4 — Regression Robustness by Region & Year
28_regression_robustness.py

Evaluates the stability of the final LightGBM regression model across 
geographical provinces (Alberta, British Columbia) and individual years 
within the test set.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"
MODEL_DIR = "/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models"

print("1. Loading model, features, and target data...")
# Load feature data and fix categorical datatype
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_test["ecozone"] = X_test["ecozone"].astype("category")

# Load actual target values 
y_test_df = pd.read_csv(f"{BASE_DIR}/y_test.csv")
y_true = y_test_df.iloc[:, 0].values

print("2. Generating predictions...")
# Load the saved regression model and predict
lgbm_reg = joblib.load(f"{MODEL_DIR}/lgbm_regressor.pkl")
y_pred = lgbm_reg.predict(X_test)

# Reconstruct the eval_df_reg dataframe so the rest of the script works
eval_df_reg = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred
})

print("3. Loading temporal metadata...")
test_raw = pd.read_csv(f"{BASE_DIR}/test_temporal.csv")

eval_df_reg["province"] = test_raw["province"].values
year_col = next((col for col in ["year", "fire_year", "fireyear", "Year", "FIRE_YEAR"] if col in test_raw.columns), None)
if year_col: 
    eval_df_reg["year"] = test_raw[year_col].values

def score_reg(df):
    if len(df) == 0: return 0, 0
    return np.sqrt(mean_squared_error(df["y_true"], df["y_pred"])), mean_absolute_error(df["y_true"], df["y_pred"])

print("\n" + "="*50)
print("9.1.3 REGRESSION ROBUSTNESS BY PROVINCE")
print("="*50)
res_prov = [{"Province": p, "N": len(d), "RMSE": round(score_reg(d)[0], 2), "MAE": round(score_reg(d)[1], 2)} for p, d in eval_df_reg.groupby("province")]
print(pd.DataFrame(res_prov).to_string(index=False))

if year_col:
    print("\n" + "="*50)
    print("9.1.3 REGRESSION ROBUSTNESS BY YEAR")
    print("="*50)
    res_yr = [{"Year": y, "N": len(d), "RMSE": round(score_reg(d)[0], 2), "MAE": round(score_reg(d)[1], 2)} for y, d in eval_df_reg.groupby("year")]
    print(pd.DataFrame(res_yr).to_string(index=False))