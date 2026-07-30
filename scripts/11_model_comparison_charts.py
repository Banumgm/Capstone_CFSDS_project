"""
11_model_comparison_charts.py
Phase 2 visualization suite: model comparison, temporal vs. spatial
validation, preliminary group importance, and spatial error map.
Reads all values from saved result CSVs (no manually typed numbers).
Saves each chart as a PNG to model_charts/ (mirrors the eda_outputs/
pattern used in Phase 1). Uses the non-interactive Agg backend locally
so charts save directly without popping up windows.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
if not os.path.exists("/Workspace"):
    matplotlib.use("Agg")  # local runs only: save without popping up windows
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
    OUTPUT_DIR = "/Workspace/Capstone_Group1/model_charts"
else:
    BASE_DIR = "processed"
    OUTPUT_DIR = "model_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"OUTPUT_DIR (relative): {OUTPUT_DIR}")
print(f"OUTPUT_DIR (absolute): {os.path.abspath(OUTPUT_DIR)}")
print(f"Folder exists after makedirs: {os.path.exists(OUTPUT_DIR)}")

# =========================================================
# Chart 1: Model comparison (RMSE/MAE bar chart)
# =========================================================
comparison_df = pd.read_csv(f"{BASE_DIR}/model_comparison_final.csv")

LABELS = {
    "tweedie_baseline": "Tweedie\n(baseline)",
    "tweedie_tuned": "Tweedie\n(tuned)",
    "random_forest": "Random\nForest",
    "lightgbm_optuna": "LightGBM",
    "xgboost_optuna": "XGBoost",
}
comparison_df["model"] = comparison_df["model"].map(LABELS)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

best_rmse_idx = comparison_df["rmse"].idxmin()
best_mae_idx = comparison_df["mae"].idxmin()

bar_colors_rmse = ["#2563eb" if i == best_rmse_idx else "#94a3b8" for i in range(len(comparison_df))]
axes[0].bar(comparison_df["model"], comparison_df["rmse"], color=bar_colors_rmse)
axes[0].set_title("RMSE by Model (lower is better)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("RMSE (m)")
for i, v in enumerate(comparison_df["rmse"]):
    axes[0].text(i, v + 8, f"{v:,.1f}", ha="center", fontsize=10)

bar_colors_mae = ["#2563eb" if i == best_mae_idx else "#94a3b8" for i in range(len(comparison_df))]
axes[1].bar(comparison_df["model"], comparison_df["mae"], color=bar_colors_mae)
axes[1].set_title("MAE by Model (lower is better)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("MAE (m)")
for i, v in enumerate(comparison_df["mae"]):
    axes[1].text(i, v + 3, f"{v:,.1f}", ha="center", fontsize=10)

plt.suptitle("Phase 2 Model Comparison — Temporal Holdout (Test Set)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_rmse_mae.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/model_comparison_rmse_mae.png")
plt.show()
plt.close(fig)

# =========================================================
# Chart 2: Temporal vs. spatial validation (grouped bar chart)
# =========================================================
temporal_df = pd.read_csv(f"{BASE_DIR}/model_comparison_final.csv")
spatial_df = pd.read_csv(f"{BASE_DIR}/spatial_cross_model_comparison.csv")

MODEL_MAP = {
    "lightgbm_optuna": "lightgbm",
    "xgboost_optuna": "xgboost",
    "random_forest": "random_forest",
}
DISPLAY_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random Forest"}

validation_data = {}
for temporal_key, spatial_key in MODEL_MAP.items():
    temporal_rmse = temporal_df.loc[temporal_df["model"] == temporal_key, "rmse"].values[0]
    bc_ab_rmse = spatial_df.loc[(spatial_df["model"] == spatial_key) & (spatial_df["direction"] == "BC -> AB"), "rmse"].values[0]
    ab_bc_rmse = spatial_df.loc[(spatial_df["model"] == spatial_key) & (spatial_df["direction"] == "AB -> BC"), "rmse"].values[0]
    validation_data[DISPLAY_LABELS[spatial_key]] = {
        "Temporal": temporal_rmse, "BC → AB": bc_ab_rmse, "AB → BC": ab_bc_rmse
    }

models = list(validation_data.keys())
scenarios = ["Temporal", "BC → AB", "AB → BC"]
x = np.arange(len(scenarios))
width = 0.25
colors_bar = {"LightGBM": "#2563eb", "XGBoost": "#94a3b8", "Random Forest": "#c7ccd1"}

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(models):
    values = [validation_data[model][s] for s in scenarios]
    offset = width * (i - (len(models) - 1) / 2)
    bars = ax.bar(x + offset, values, width, label=model, color=colors_bar.get(model, "#94a3b8"))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10, f"{v:,.0f}",
                ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("RMSE (m)")
ax.set_title("Temporal vs. Spatial Holdout Validation\n(asymmetric generalization: BC→AB worse than AB→BC, replicated across all models)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.axhline(y=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/temporal_vs_spatial_validation.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/temporal_vs_spatial_validation.png")
plt.show()
plt.close(fig)

# =========================================================
# Chart 3: Preliminary feature-group importance (LightGBM, gain-based)
# =========================================================
importance_lgb = pd.read_csv(f"{BASE_DIR}/lightgbm_feature_importance.csv", index_col=0)["importance"]

GROUP_VARS = {
    "FWI Indices": ["fwi", "isi", "bui"],
    "Topographic": ["aspect_cos", "twi", "slope", "dem", "aspect_sin"],
    "Anthropogenic": ["roaddens2k", "roaddist", "roaddens5k", "roaddens10k", "roaddens25k"],
}
groups = {group: importance_lgb[vars_].to_dict() for group, vars_ in GROUP_VARS.items()}

fig, ax = plt.subplots(figsize=(8, 5))
group_maxes = [max(v.values()) for v in groups.values()]
group_names = list(groups.keys())
colors_group = ["#f59e0b", "#2563eb", "#10b981"]

bars = ax.bar(group_names, group_maxes, color=colors_group)
for bar, v in zip(bars, group_maxes):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Max feature importance (gain-based)")
ax.set_title("Preliminary Group Comparison — LightGBM\n(FWI vs. Topographic vs. Anthropogenic max importance)",
             fontsize=12, fontweight="bold")
plt.figtext(0.5, -0.05, "Note: Preliminary observation only — formal H1 test deferred to Phase 4 (SHAP, PR #24)",
            ha="center", fontsize=9, style="italic", color="gray")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/h1_preliminary_group_importance.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/h1_preliminary_group_importance.png")
plt.show()
plt.close(fig)

# =========================================================
# Chart 4: Geographic map of spatial-holdout prediction errors (LightGBM)
# Re-fits the LightGBM spatial models here (hyperparameters read from CSV,
# a few seconds to run) so this chart is independently runnable without
# depending on variables left over from 10_spatial_validation.py.
# =========================================================
train_spatial_raw = pd.read_csv(f"{BASE_DIR}/train_spatial.csv")
test_spatial_raw  = pd.read_csv(f"{BASE_DIR}/test_spatial.csv")

TARGET = "sprdistm"
DROP_COLS = ["ID", "source_file", "fireday", "aspect", "province", TARGET,
             "cumuarea", "pctgrowth", "prevgrow",
             "firearea", "vpd", "d_vpd",
             "d_fwi", "d_isi", "d_ffmc", "d_dmc", "d_dc", "d_bui"]

all_ecozone_codes = sorted(
    set(train_spatial_raw["ecozone"].dropna().astype(int).unique()) |
    set(test_spatial_raw["ecozone"].dropna().astype(int).unique())
)
ecozone_dtype = pd.CategoricalDtype(categories=all_ecozone_codes)
train_spatial_raw["ecozone"] = train_spatial_raw["ecozone"].astype(int).astype(ecozone_dtype)
test_spatial_raw["ecozone"]  = test_spatial_raw["ecozone"].astype(int).astype(ecozone_dtype)

feature_cols = [c for c in train_spatial_raw.columns if c not in DROP_COLS]
X_bc, y_bc = train_spatial_raw[feature_cols], train_spatial_raw[TARGET]
X_ab, y_ab = test_spatial_raw[feature_cols], test_spatial_raw[TARGET]

lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
lgb_param_cols = [c for c in lgb_search.index if c != "cv_rmse"]
lgb_int_cols = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
BEST_PARAMS_LGB = {c: (int(lgb_search[c]) if c in lgb_int_cols else float(lgb_search[c])) for c in lgb_param_cols}

model_bc_train_lgb = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
model_bc_train_lgb.fit(X_bc, y_bc, categorical_feature=["ecozone"])

model_ab_train_lgb = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
model_ab_train_lgb.fit(X_ab, y_ab, categorical_feature=["ecozone"])

preds_bc_to_ab = model_bc_train_lgb.predict(X_ab)
preds_ab_to_bc = model_ab_train_lgb.predict(X_bc)

error_bc_to_ab = np.abs(y_ab.values - preds_bc_to_ab)
error_ab_to_bc = np.abs(y_bc.values - preds_ab_to_bc)

VMAX_FIXED = 1500

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

sc1 = axes[0].scatter(test_spatial_raw["lon"], test_spatial_raw["lat"],
                       c=error_bc_to_ab, cmap="Reds", s=15, alpha=0.6,
                       vmin=0, vmax=VMAX_FIXED)
axes[0].set_title("BC-trained model → AB fires\n(worse generalization)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
plt.colorbar(sc1, ax=axes[0], label="Absolute error (m)")

sc2 = axes[1].scatter(train_spatial_raw["lon"], train_spatial_raw["lat"],
                       c=error_ab_to_bc, cmap="Reds", s=15, alpha=0.6,
                       vmin=0, vmax=VMAX_FIXED)
axes[1].set_title("AB-trained model → BC fires\n(better generalization)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
plt.colorbar(sc2, ax=axes[1], label="Absolute error (m)")

plt.suptitle("Spatial Generalization Error by Fire Location (LightGBM)\nColor scale fixed at 1,500m across both panels",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/spatial_error_map.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/spatial_error_map.png")
plt.show()
plt.close(fig)

print("\nAll charts saved successfully.")