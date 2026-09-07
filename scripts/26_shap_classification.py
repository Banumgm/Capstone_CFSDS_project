"""
Phase 4, Step 2 — SHAP Diagnostics for LightGBM Classification (H1 Validation)
26_shap_classification.py

Loads the finalized LightGBM classifier, calculates TreeSHAP values on a 
representative test sample, and outputs the mean absolute SHAP values grouped 
by the five updated categories to cross-validate Hypothesis 1 and engineered features.
"""
import os
import pandas as pd
import numpy as np
import joblib
import shap

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"
MODEL_DIR = "/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models"

print("1. Loading classification model and test data...")
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_test["ecozone"] = X_test["ecozone"].astype("category")

lgbm_clf = joblib.load(f"{MODEL_DIR}/lgbm_classifier.pkl")

print("2. Calculating SHAP values for the Classifier (Sample N=2000)...")
X_test_sample = X_test.sample(n=2000, random_state=42)
explainer_clf = shap.TreeExplainer(lgbm_clf)
shap_values_clf = explainer_clf.shap_values(X_test_sample)

# Handle LightGBM binary classification list output
if isinstance(shap_values_clf, list):
    shap_values_clf = shap_values_clf[1]

mean_abs_shap_clf = np.abs(shap_values_clf).mean(axis=0)
shap_imp_clf = pd.Series(mean_abs_shap_clf, index=X_test_sample.columns).sort_values(ascending=False)

# UPDATED GROUP_VARS (Consistent with Regression Step 1)
GROUP_VARS = {
    "Seasonal Dynamics": ["fireday_sin", "fireday_cos", "DOB", "aspect_x_season"],
    "Fuel & Hydrology": ["nonfuel1k", "hydrodens2k", "hydrodens5k", "Biomass"],
    "FWI Indices": ["fwi", "isi", "bui", "ffmc", "dmc", "dc"],
    "Topographic": ["aspect_cos", "twi", "slope", "dem", "aspect_sin"],
    "Anthropogenic": ["roaddens2k", "roaddist", "roaddens5k", "roaddens10k", "roaddens25k"],
}

print("\n" + "="*60)
print("9.2.1 CLASSIFICATION SHAP DIAGNOSTICS & H1 CROSS-VALIDATION")
print("="*60)
print("\nTop 15 Features by Mean Absolute SHAP (Classification):")
print(shap_imp_clf.head(15).round(3))

print("\n-- H1 Cross-Validation: Maximum SHAP impact by Feature Group --")
for group_name, features in GROUP_VARS.items():
    available = [f for f in features if f in shap_imp_clf.index]
    if available:
        print(f"{group_name:<20}: Max Impact = {shap_imp_clf[available].max():>6.3f} (driven by {shap_imp_clf[available].idxmax()})")