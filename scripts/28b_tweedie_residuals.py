"""
Phase 4, Step 4b — Tweedie Residual Visual Diagnostics
28b_tweedie_residuals.py

Calculates Tweedie deviance residuals and generates 4 diagnostic plots:
1. Deviance Residual Distribution (Histogram)
2. Residuals vs. Fitted Values (Scatter)
3. Residuals by Province (Boxplot)
4. Residuals by Year (Boxplot)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

print("Loading data for Residual Analysis...")
X_train = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"] = X_test["ecozone"].astype("category")

y_train_reg = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]
y_test_reg = pd.read_csv(f"{BASE_DIR}/y_test.csv").iloc[:, 0]
test_raw = pd.read_csv(f"{BASE_DIR}/test_temporal.csv")

# 1. Fast Model Refit to guarantee continuous predictions are in memory
print("Generating continuous predictions...")
lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
lgb_param_cols = [c for c in lgb_search.index if c != "cv_rmse"]
lgb_int_cols = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
BEST_PARAMS_LGB = {c: (int(lgb_search[c]) if c in lgb_int_cols else float(lgb_search[c])) for c in lgb_param_cols}

final_lgb_reg = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
final_lgb_reg.fit(X_train, y_train_reg, categorical_feature=["ecozone"])
y_pred_reg = final_lgb_reg.predict(X_test)

# 2. Mathematical Definition of Tweedie Deviance Residuals (p=1.3)
def calculate_tweedie_deviance_residuals(y, mu, p=1.3):
    dev = np.zeros_like(y, dtype=float)
    mu = np.maximum(mu, 1e-10) # Prevent log/zero errors
    
    # y == 0 formula
    mask_zero = (y == 0)
    dev[mask_zero] = 2 * (mu[mask_zero]**(2-p)) / (2-p)
    
    # y > 0 formula
    mask_pos = (y > 0)
    y_pos, mu_pos = y[mask_pos], mu[mask_pos]
    term1 = (y_pos**(2-p)) / ((1-p)*(2-p))
    term2 = (y_pos * mu_pos**(1-p)) / (1-p)
    term3 = (mu_pos**(2-p)) / (2-p)
    dev[mask_pos] = 2 * (term1 - term2 + term3)
    
    dev = np.maximum(dev, 0) # Ensure strictly positive before sqrt
    return np.sign(y - mu) * np.sqrt(dev)

print("Calculating Tweedie Deviance Residuals...")
dev_residuals = calculate_tweedie_deviance_residuals(y_test_reg.values, y_pred_reg, p=1.3)

# Build plotting DataFrame
res_df = pd.DataFrame({
    'y_true': y_test_reg.values,
    'y_pred': y_pred_reg,
    'deviance_residual': dev_residuals,
    'province': test_raw['province'].values
})
year_col = next((col for col in ["year", "fire_year", "fireyear", "Year", "FIRE_YEAR"] if col in test_raw.columns), None)
if year_col:
    res_df['year'] = test_raw[year_col].values

# 3. Generate the 4-Panel Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Tweedie Regression Residual Diagnostics (p=1.3)', fontsize=16, fontweight='bold', y=0.95)

# Plot A: Residual Distribution
sns.histplot(res_df['deviance_residual'], bins=50, kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('A: Deviance Residual Distribution')
axes[0, 0].set_xlabel('Tweedie Deviance Residual')
axes[0, 0].set_ylabel('Frequency')

# Plot B: Residuals vs Fitted
sns.scatterplot(x=res_df['y_pred'], y=res_df['deviance_residual'], alpha=0.3, ax=axes[0, 1], color='darkred')
axes[0, 1].axhline(0, color='black', linestyle='--')
axes[0, 1].set_title('B: Residuals vs. Fitted Values')
axes[0, 1].set_xlabel('Predicted Spread Distance (m/day)')
axes[0, 1].set_ylabel('Deviance Residual')

# Plot C: Residuals by Province (Updated for Seaborn v0.14.0 compatibility)
sns.boxplot(x='province', y='deviance_residual', data=res_df, ax=axes[1, 0], hue='province', palette='muted', legend=False)
axes[1, 0].axhline(0, color='black', linestyle='--')
axes[1, 0].set_title('C: Deviance Residuals by Province')
axes[1, 0].set_xlabel('Province')
axes[1, 0].set_ylabel('Deviance Residual')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot D: Residuals by Year (Updated for Seaborn v0.14.0 compatibility)
if year_col:
    sns.boxplot(x='year', y='deviance_residual', data=res_df, ax=axes[1, 1], hue='year', palette='viridis', legend=False)
    axes[1, 1].axhline(0, color='black', linestyle='--')
    axes[1, 1].set_title('D: Deviance Residuals by Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Deviance Residual')
else:
    axes[1, 1].text(0.5, 0.5, 'Year data not available', ha='center', va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
print("Plots generated successfully. Review the outputs for heteroscedasticity.")