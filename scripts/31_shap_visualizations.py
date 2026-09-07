"""
Phase 4, Step 7 — SHAP Visual Diagnostics
31_shap_visualizations.py

Generates global and local interpretability visualizations for both
the regression and classification models, including Beeswarm (Summary), 
Dependence (Interaction), and Waterfall (Local Decomposition) plots.
"""
import os
import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"
MODEL_DIR = "/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models"

print("1. Loading Data and Models for SHAP Visualization...")
X_train = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"] = X_test["ecozone"].astype("category")
y_train_reg = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]

# Load Classification Model
lgbm_clf = joblib.load(f"{MODEL_DIR}/lgbm_classifier.pkl")

# Fast Re-fit of Regression Model (to ensure it's in memory)
lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
lgb_param_cols = [c for c in lgb_search.index if c != "cv_rmse"]
lgb_int_cols = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
BEST_PARAMS_LGB = {c: (int(lgb_search[c]) if c in lgb_int_cols else float(lgb_search[c])) for c in lgb_param_cols}

final_lgb_reg = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
final_lgb_reg.fit(X_train, y_train_reg, categorical_feature=["ecozone"])

# Sample for SHAP calculation
print("2. Calculating SHAP values (Sample N=2000)...")
X_test_sample = X_test.sample(n=2000, random_state=42)

# --- Regression SHAP Objects ---
explainer_reg = shap.TreeExplainer(final_lgb_reg)
shap_values_reg_array = explainer_reg.shap_values(X_test_sample)
# Generate Explanation object (Required for Waterfall plots)
explanation_reg = explainer_reg(X_test_sample)

# --- Classification SHAP Objects ---
explainer_clf = shap.TreeExplainer(lgbm_clf)
shap_values_clf_array = explainer_clf.shap_values(X_test_sample)
if isinstance(shap_values_clf_array, list):
    shap_values_clf_array = shap_values_clf_array[1]

# Generate Explanation object (Required for Waterfall plots)
explanation_clf = explainer_clf(X_test_sample)
# Handle dimensions for binary classification (extracting the positive class)
if len(explanation_clf.shape) == 3:
    explanation_clf = explanation_clf[:, :, 1]


# =====================================================================
# VISUALIZATION 1: BEESWARM (GLOBAL SUMMARY)
# =====================================================================
print("3. Generating SHAP Beeswarm Plots (Global Importance)...")

plt.figure(figsize=(10, 6))
plt.title("Regression Model: SHAP Summary (Impact on Continuous Spread)", fontsize=14, fontweight='bold', pad=20)
shap.summary_plot(shap_values_reg_array, X_test_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Classification Model: SHAP Summary (Impact on Extreme Event Probability)", fontsize=14, fontweight='bold', pad=20)
shap.summary_plot(shap_values_clf_array, X_test_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()


# =====================================================================
# VISUALIZATION 2: DEPENDENCE (NON-LINEAR & INTERACTION EFFECTS)
# =====================================================================
print("4. Generating SHAP Dependence Plots (Interaction/Thresholds)...")

fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot("fwi", shap_values_reg_array, X_test_sample, ax=ax, show=False)
ax.set_title("Regression Dependence: Fire Weather Index (FWI)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot("fwi", shap_values_clf_array, X_test_sample, ax=ax, show=False)
ax.set_title("Classification Dependence: Fire Weather Index (FWI)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# =====================================================================
# VISUALIZATION 3: WATERFALL (LOCAL INDIVIDUAL DECOMPOSITION)
# =====================================================================
print("5. Generating SHAP Waterfall Plots (Local Individual Decomposition)...")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation_reg[0], max_display=10, show=False)
plt.title("Regression Local Decomposition: Sample Row 0", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation_clf[0], max_display=10, show=False)
plt.title("Classification Local Decomposition: Sample Row 0", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# =====================================================================
# VISUALIZATION 4: FORCE PLOT (LOCAL INDIVIDUAL DECOMPOSITION, SAME ROW AS WATERFALL)
# =====================================================================
print("6. Generating SHAP Force Plots (Local Individual Decomposition)...")

# --- Regression: expected_value is a single float for LGBMRegressor ---
expected_value_reg = explainer_reg.expected_value

plt.figure(figsize=(20, 3))
shap.force_plot(
    expected_value_reg,
    shap_values_reg_array[0],
    X_test_sample.iloc[0],
    matplotlib=True,
    show=False
)
plt.title("Regression Force Plot: Sample Row 0", fontsize=14, fontweight='bold', pad=30)
plt.tight_layout()
plt.show()

# --- Classification: expected_value may come back as a list (per-class) ---
expected_value_clf = explainer_clf.expected_value
if isinstance(expected_value_clf, (list, np.ndarray)) and np.ndim(expected_value_clf) > 0:
    expected_value_clf = expected_value_clf[1]  # positive class, consistent with shap_values_clf_array above

plt.figure(figsize=(20, 3))
shap.force_plot(
    expected_value_clf,
    shap_values_clf_array[0],
    X_test_sample.iloc[0],
    matplotlib=True,
    show=False
)
plt.title("Classification Force Plot: Sample Row 0", fontsize=14, fontweight='bold', pad=30)
plt.tight_layout()
plt.show()