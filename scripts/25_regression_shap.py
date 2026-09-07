"""
Phase 4, Step 1 — Regression SHAP Diagnostics
25_regression_shap.py

Executes TreeSHAP diagnostics on the continuous regression model to evaluate the absolute 
marginal contribution of individual features toward predicting baseline fire spread. 
This provides the primary evaluation for Hypothesis 1 in a continuous magnitude context.
"""
import os
import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb

BASE_DIR = "processed"

print("1. Loading data and fitting final LightGBM Regressor for SHAP extraction...")
X_train = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
y_train_reg = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]

X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"] = X_test["ecozone"].astype("category")

# Fast Re-fit
lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
lgb_param_cols = [c for c in lgb_search.index if c != "cv_rmse"]
lgb_int_cols = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
BEST_PARAMS_LGB = {c: (int(lgb_search[c]) if c in lgb_int_cols else float(lgb_search[c])) for c in lgb_param_cols}

final_lgb_reg = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
final_lgb_reg.fit(X_train, y_train_reg, categorical_feature=["ecozone"])
print("Saving LightGBM Regressor to models folder...")
joblib.dump(final_lgb_reg, "models/lgbm_regressor.pkl")

print("2. Calculating SHAP values for the Regressor (Sample N=2000)...")
X_test_sample = X_test.sample(n=2000, random_state=42)
explainer_reg = shap.TreeExplainer(final_lgb_reg)
shap_values_reg = explainer_reg.shap_values(X_test_sample)

mean_abs_shap_reg = np.abs(shap_values_reg).mean(axis=0)
shap_imp_reg = pd.Series(mean_abs_shap_reg, index=X_test_sample.columns).sort_values(ascending=False)

GROUP_VARS = {
    "Seasonal Dynamics": ["fireday_sin", "fireday_cos", "DOB", "aspect_x_season"],
    "Fuel & Hydrology": ["nonfuel1k", "hydrodens2k", "hydrodens5k", "Biomass"],
    "FWI Indices": ["fwi", "isi", "bui", "ffmc", "dmc", "dc"],
    "Topographic": ["aspect_cos", "twi", "slope", "dem", "aspect_sin"],
    "Anthropogenic": ["roaddens2k", "roaddist", "roaddens5k", "roaddens10k", "roaddens25k"],
}

print("\n" + "="*60)
print("9.1.1 REGRESSION SHAP DIAGNOSTICS & H1 TEST")
print("="*60)
print("\nTop 15 Features by Mean Absolute SHAP (Regression):")
print(shap_imp_reg.head(15).round(3))

print("\n-- H1 Test: Maximum SHAP impact by Feature Group --")
for group_name, features in GROUP_VARS.items():
    available = [f for f in features if f in shap_imp_reg.index]
    if available:
        print(f"{group_name:<20}: Max Impact = {shap_imp_reg[available].max():>6.3f} (driven by {shap_imp_reg[available].idxmax()})")