"""
10_spatial_validation.py
Spatial holdout validation (BC<->AB) for the three tree-based models
(LightGBM -- final model; XGBoost and Random Forest -- cross-checks), plus
a spatial feature-importance comparison (BC-trained vs. AB-trained,
LightGBM) to help explain the asymmetric generalization pattern.
H1/feature-importance comparison limited to reporting top features per
direction -- not framed as a formal H1 test (deferred to Phase 4 SHAP,
PR #24). All hyperparameters are READ from their saved search-result CSVs
(no manual re-entry).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

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

# =========================================================
# Read tuned hyperparameters from saved search results (no manual entry)
# =========================================================
lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
lgb_param_cols = [c for c in lgb_search.index if c != "cv_rmse"]
lgb_int_cols = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
BEST_PARAMS_LGB = {c: (int(lgb_search[c]) if c in lgb_int_cols else float(lgb_search[c])) for c in lgb_param_cols}
print("LightGBM params read from lightgbm_optuna_search.csv:", BEST_PARAMS_LGB)

xgb_search = pd.read_csv(f"{BASE_DIR}/xgboost_optuna_search.csv").iloc[0]
xgb_param_cols = [c for c in xgb_search.index if c != "cv_rmse"]
xgb_int_cols = ["n_estimators", "max_depth", "min_child_weight"]
BEST_PARAMS_XGB = {c: (int(xgb_search[c]) if c in xgb_int_cols else float(xgb_search[c])) for c in xgb_param_cols}
print("XGBoost params read from xgboost_optuna_search.csv:", BEST_PARAMS_XGB)

rf_search = pd.read_csv(f"{BASE_DIR}/rf_hyperparameter_search.csv").iloc[0]
rf_param_cols = [c for c in rf_search.index if c != "cv_rmse"]

def _cast_rf_param(name, value):
    if name == "max_depth" and pd.isna(value):
        return None
    if name in ["n_estimators", "min_samples_split", "min_samples_leaf", "max_depth"]:
        return int(value)
    if name == "max_features":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value

RF_BEST_PARAMS = {c: _cast_rf_param(c, rf_search[c]) for c in rf_param_cols}
print("Random Forest params read from rf_hyperparameter_search.csv:", RF_BEST_PARAMS)

# =========================================================
# PART 1: LightGBM spatial validation (final model)
# =========================================================
def fit_and_evaluate_spatial_lgb(X_tr, y_tr, X_te, y_te, label):
    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.3,
        random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
    )
    model.fit(X_tr, y_tr, categorical_feature=["ecozone"])
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae = mean_absolute_error(y_te, preds)
    print(f"{label}: RMSE={rmse:,.2f}, MAE={mae:,.2f}")
    return model, rmse, mae

print("\n=== Spatial holdout validation (LightGBM, final model) ===")
model_bc_train_lgb, rmse_bc_ab_lgb, mae_bc_ab_lgb = fit_and_evaluate_spatial_lgb(X_bc, y_bc, X_ab, y_ab, "Train BC -> Test AB")
model_ab_train_lgb, rmse_ab_bc_lgb, mae_ab_bc_lgb = fit_and_evaluate_spatial_lgb(X_ab, y_ab, X_bc, y_bc, "Train AB -> Test BC")

# =========================================================
# PART 2: XGBoost spatial validation (cross-check)
# =========================================================
def fit_and_evaluate_spatial_xgb(X_tr, y_tr, X_te, y_te, label):
    model = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.3,
        enable_categorical=True, tree_method="hist",
        random_state=42, n_jobs=-1, **BEST_PARAMS_XGB
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae = mean_absolute_error(y_te, preds)
    print(f"{label}: RMSE={rmse:,.2f}, MAE={mae:,.2f}")
    return model, rmse, mae

print("\n=== Spatial holdout validation (XGBoost, cross-check) ===")
model_bc_train_xgb, rmse_bc_ab_xgb, mae_bc_ab_xgb = fit_and_evaluate_spatial_xgb(X_bc, y_bc, X_ab, y_ab, "Train BC -> Test AB")
model_ab_train_xgb, rmse_ab_bc_xgb, mae_ab_bc_xgb = fit_and_evaluate_spatial_xgb(X_ab, y_ab, X_bc, y_bc, "Train AB -> Test BC")

# =========================================================
# PART 3: Random Forest spatial validation (cross-check)
# =========================================================
X_bc_rf = pd.get_dummies(X_bc, columns=["ecozone"], prefix="eco")
X_ab_rf = pd.get_dummies(X_ab, columns=["ecozone"], prefix="eco")
X_bc_rf, X_ab_rf = X_bc_rf.align(X_ab_rf, join="outer", axis=1, fill_value=0)
dummy_cols_spatial = [c for c in X_bc_rf.columns if c.startswith("eco_")]
X_bc_rf[dummy_cols_spatial] = X_bc_rf[dummy_cols_spatial].astype(int)
X_ab_rf[dummy_cols_spatial] = X_ab_rf[dummy_cols_spatial].astype(int)

def fit_and_evaluate_spatial_rf(X_tr, y_tr, X_te, y_te, label):
    model = RandomForestRegressor(**RF_BEST_PARAMS, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae = mean_absolute_error(y_te, preds)
    print(f"{label}: RMSE={rmse:,.2f}, MAE={mae:,.2f}")
    return rmse, mae

print("\n=== Spatial holdout validation (Random Forest, cross-check) ===")
rmse_rf_bc_ab, mae_rf_bc_ab = fit_and_evaluate_spatial_rf(X_bc_rf, y_bc, X_ab_rf, y_ab, "RF: Train BC -> Test AB")
rmse_rf_ab_bc, mae_rf_ab_bc = fit_and_evaluate_spatial_rf(X_ab_rf, y_ab, X_bc_rf, y_bc, "RF: Train AB -> Test BC")

# =========================================================
# PART 4: Cross-model comparison and replication check
# =========================================================
print(f"\n=== Cross-model spatial comparison ===")
print(f"LightGBM: BC->AB RMSE {rmse_bc_ab_lgb:,.2f} | AB->BC RMSE {rmse_ab_bc_lgb:,.2f}")
print(f"XGBoost:  BC->AB RMSE {rmse_bc_ab_xgb:,.2f} | AB->BC RMSE {rmse_ab_bc_xgb:,.2f}")
print(f"RF:       BC->AB RMSE {rmse_rf_bc_ab:,.2f} | AB->BC RMSE {rmse_rf_ab_bc:,.2f}")

pattern_lgb = rmse_bc_ab_lgb > rmse_ab_bc_lgb
pattern_xgb = rmse_bc_ab_xgb > rmse_ab_bc_xgb
pattern_rf = rmse_rf_bc_ab > rmse_rf_ab_bc

if pattern_lgb == pattern_xgb == pattern_rf:
    print("\n[FINDING] Asymmetric generalization pattern REPLICATED across all three model types")
else:
    print("\n[FINDING] Pattern differs by model")

spatial_cross_model = pd.DataFrame([
    {"model": "lightgbm", "direction": "BC -> AB", "rmse": rmse_bc_ab_lgb, "mae": mae_bc_ab_lgb},
    {"model": "lightgbm", "direction": "AB -> BC", "rmse": rmse_ab_bc_lgb, "mae": mae_ab_bc_lgb},
    {"model": "xgboost", "direction": "BC -> AB", "rmse": rmse_bc_ab_xgb, "mae": mae_bc_ab_xgb},
    {"model": "xgboost", "direction": "AB -> BC", "rmse": rmse_ab_bc_xgb, "mae": mae_ab_bc_xgb},
    {"model": "random_forest", "direction": "BC -> AB", "rmse": rmse_rf_bc_ab, "mae": mae_rf_bc_ab},
    {"model": "random_forest", "direction": "AB -> BC", "rmse": rmse_rf_ab_bc, "mae": mae_rf_ab_bc},
])
spatial_cross_model.to_csv(f"{BASE_DIR}/spatial_cross_model_comparison.csv", index=False)
print("\nSaved: spatial_cross_model_comparison.csv")

# =========================================================
# PART 5: Spatial feature-importance comparison (BC-trained vs. AB-trained, LightGBM)
# Supports the "feature weighting shifts by region" explanation for the
# asymmetric generalization finding above.
# =========================================================
model_bc_train_lgb_gain = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    importance_type="gain", random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
model_bc_train_lgb_gain.fit(X_bc, y_bc, categorical_feature=["ecozone"])

model_ab_train_lgb_gain = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    importance_type="gain", random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
model_ab_train_lgb_gain.fit(X_ab, y_ab, categorical_feature=["ecozone"])

importance_bc = pd.Series(model_bc_train_lgb_gain.feature_importances_, index=feature_cols).sort_values(ascending=False)
importance_ab = pd.Series(model_ab_train_lgb_gain.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("Top 10 features, BC-trained model (LightGBM, gain-based):")
print(importance_bc.head(10))
print("\nTop 10 features, AB-trained model (LightGBM, gain-based):")
print(importance_ab.head(10))