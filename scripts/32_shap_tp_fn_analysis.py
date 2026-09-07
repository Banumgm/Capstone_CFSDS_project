"""
Phase 4, Step 8 — Classification TP/FN Local SHAP Analysis
32_shap_tp_fn_analysis.py

Extracts local SHAP explanations specifically for True Positives and False Negatives.
Compares the mean directional SHAP values to identify exactly which environmental 
features are "tricking" the model into missing 90th-percentile extreme events.
"""
import os
import pandas as pd
import numpy as np
import joblib
import shap

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"
MODEL_DIR = "/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models"

print("1. Loading classification data and model...")
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_test["ecozone"] = X_test["ecozone"].astype("category")
y_test_clf = pd.read_csv(f"{BASE_DIR}/y_test_clf.csv").iloc[:, 0]
lgbm_clf = joblib.load(f"{MODEL_DIR}/lgbm_classifier.pkl")

# FIXED: Updated to Simran's new Phase 4 Handoff Threshold (0.212)
print("2. Isolating True Positives (TP) and False Negatives (FN) at 0.212 threshold...")
OPTIMAL_THRESHOLD = 0.212
y_proba = lgbm_clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

# Masks for TP and FN (Only looking at ACTUAL extreme events)
tp_mask = (y_test_clf == 1) & (y_pred == 1)
fn_mask = (y_test_clf == 1) & (y_pred == 0)

X_tp = X_test[tp_mask]
X_fn = X_test[fn_mask]

print(f"   Found {len(X_tp)} TPs and {len(X_fn)} FNs.")

print("3. Calculating directional SHAP values...")
explainer = shap.TreeExplainer(lgbm_clf)

# Extract SHAP values (index 1 for positive class probabilities)
shap_tp = explainer.shap_values(X_tp)
shap_fn = explainer.shap_values(X_fn)
if isinstance(shap_tp, list):
    shap_tp = shap_tp[1]
    shap_fn = shap_fn[1]

# Calculate mean DIRECTIONAL SHAP 
mean_shap_tp = shap_tp.mean(axis=0)
mean_shap_fn = shap_fn.mean(axis=0)

shap_compare_df = pd.DataFrame({
    'Feature': X_test.columns,
    'TP_Mean_SHAP': mean_shap_tp,
    'FN_Mean_SHAP': mean_shap_fn
})

# Calculate the divergence
shap_compare_df['SHAP_Divergence'] = shap_compare_df['TP_Mean_SHAP'] - shap_compare_df['FN_Mean_SHAP']

# Sort by the features pulling the FN predictions down the hardest
top_fn_drivers = shap_compare_df.sort_values(by='FN_Mean_SHAP', ascending=True).head(10)
top_divergences = shap_compare_df.sort_values(by='SHAP_Divergence', ascending=False).head(10)

print("\n" + "="*70)
print("9.2.5 EXTREME EVENT MISCLASSIFICATION: LOCAL SHAP ANALYSIS")
print("="*70)
print("\nTop Features Pulling False Negatives DOWN (Tricking the model):")
print("(Negative values indicate the feature is pushing the prediction towards 0)")
print(top_fn_drivers[['Feature', 'FN_Mean_SHAP', 'TP_Mean_SHAP']].round(3).to_string(index=False))

print("\nTop Features with the Biggest SHAP Divergence (TP vs FN):")
print("(Large positive divergence means the feature behaves vastly differently in TPs vs FNs)")
print(top_divergences[['Feature', 'SHAP_Divergence', 'TP_Mean_SHAP', 'FN_Mean_SHAP']].round(3).to_string(index=False))