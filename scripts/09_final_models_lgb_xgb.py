"""
09_final_models_lgb_xgb.py
Full-budget Optuna Bayesian hyperparameter search (N_TRIALS=100, 5-fold
GroupKFold CV, grouped by fire ID, early stopping, seed fixed for
reproducibility) combined with the final LightGBM/XGBoost fits.
BEST_PARAMS_LGB / BEST_PARAMS_XGB are READ from the Optuna study results
(no manual re-entry). Baseline/RF numbers for the comparison table are
READ from their saved result CSVs. Best model determined automatically
from the comparison table (no hardcoded "best" label). H1 comparison
blocks omitted (deferred to Phase 4 SHAP, PR #24).

Saves: lightgbm_optuna_search.csv, xgboost_optuna_search.csv,
       lightgbm_feature_importance.csv, xgboost_feature_importance.csv,
       model_comparison_final.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

X_train_tree = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test_tree  = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
y_train = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]
y_test  = pd.read_csv(f"{BASE_DIR}/y_test.csv").iloc[:, 0]

X_train_tree["ecozone"] = X_train_tree["ecozone"].astype("category")
X_test_tree["ecozone"]  = X_test_tree["ecozone"].astype("category")

TRAIN_FILE = f"{BASE_DIR}/train_temporal.csv"
train_df_raw = pd.read_csv(TRAIN_FILE)
groups = train_df_raw["ID"].values

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train_tree, y_train, groups=groups))
N_TRIALS = 100

# =========================================================
# PART 1: Optuna hyperparameter search (full budget)
# =========================================================
def objective_lgb(trial):
    params = {
        "objective": "tweedie", "tweedie_variance_power": 1.3,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        "importance_type": "gain", "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    fold_rmses = []
    for train_idx, val_idx in cv_splits:
        X_tr, X_val = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, categorical_feature=["ecozone"],
                  eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val, num_iteration=model.best_iteration_)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(fold_rmses)

study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)
print("LightGBM best CV RMSE:", study_lgb.best_value)
print("LightGBM best params:", study_lgb.best_params)
pd.DataFrame([{**study_lgb.best_params, "cv_rmse": study_lgb.best_value}]
             ).to_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv", index=False)

def objective_xgb(trial):
    params = {
        "objective": "reg:tweedie", "tweedie_variance_power": 1.3,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "enable_categorical": True, "tree_method": "hist", "random_state": 42, "n_jobs": -1,
    }
    fold_rmses = []
    for train_idx, val_idx in cv_splits:
        X_tr, X_val = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(fold_rmses)

study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
print("\nXGBoost best CV RMSE:", study_xgb.best_value)
print("XGBoost best params:", study_xgb.best_params)
pd.DataFrame([{**study_xgb.best_params, "cv_rmse": study_xgb.best_value}]
             ).to_csv(f"{BASE_DIR}/xgboost_optuna_search.csv", index=False)

# =========================================================
# PART 2: Final fits using searched BEST_PARAMS (read, not re-typed)
# =========================================================
BEST_PARAMS_LGB = study_lgb.best_params
final_lgb = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3,
    importance_type="gain", random_state=42, n_jobs=-1, verbose=-1,
    **BEST_PARAMS_LGB
)
final_lgb.fit(X_train_tree, y_train, categorical_feature=["ecozone"])
preds_lgb = final_lgb.predict(X_test_tree)
rmse_lgb = np.sqrt(mean_squared_error(y_test, preds_lgb))
mae_lgb = mean_absolute_error(y_test, preds_lgb)
print(f"\n=== LightGBM (Optuna-tuned) test performance ===")
print(f"RMSE: {rmse_lgb:,.2f}  MAE: {mae_lgb:,.2f}")

BEST_PARAMS_XGB = study_xgb.best_params
final_xgb = xgb.XGBRegressor(
    objective="reg:tweedie", tweedie_variance_power=1.3,
    enable_categorical=True, tree_method="hist",
    random_state=42, n_jobs=-1,
    **BEST_PARAMS_XGB
)
final_xgb.fit(X_train_tree, y_train)
preds_xgb = final_xgb.predict(X_test_tree)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
mae_xgb = mean_absolute_error(y_test, preds_xgb)
print(f"\n=== XGBoost (Optuna-tuned) test performance ===")
print(f"RMSE: {rmse_xgb:,.2f}  MAE: {mae_xgb:,.2f}")

importance_lgb = pd.Series(final_lgb.feature_importances_, index=X_train_tree.columns).sort_values(ascending=False)
importance_xgb = pd.Series(final_xgb.feature_importances_, index=X_train_tree.columns).sort_values(ascending=False)
print("\nTop 15 feature importances (LightGBM):\n", importance_lgb.head(15))
print("\nTop 15 feature importances (XGBoost):\n", importance_xgb.head(15))

importance_lgb.to_csv(f"{BASE_DIR}/lightgbm_feature_importance.csv", header=["importance"])
importance_xgb.to_csv(f"{BASE_DIR}/xgboost_feature_importance.csv", header=["importance"])

# --- Read Tweedie baseline/tuned and RF numbers from their saved result CSVs ---
tweedie_results = pd.read_csv(f"{BASE_DIR}/tweedie_baseline_vs_tuned_final.csv")
tweedie_baseline_row = tweedie_results[tweedie_results["label"].str.contains("baseline")].iloc[0]
tweedie_tuned_row = tweedie_results[tweedie_results["label"].str.contains("tuned")].iloc[0]

rf_results = pd.read_csv(f"{BASE_DIR}/rf_v3_results.csv").iloc[0]

comparison_out = pd.DataFrame([
    {"model": "tweedie_baseline", "rmse": tweedie_baseline_row["rmse"], "mae": tweedie_baseline_row["mae"]},
    {"model": "tweedie_tuned", "rmse": tweedie_tuned_row["rmse"], "mae": tweedie_tuned_row["mae"]},
    {"model": "random_forest", "rmse": rf_results["rmse"], "mae": rf_results["mae"]},
    {"model": "lightgbm_optuna", "rmse": rmse_lgb, "mae": mae_lgb},
    {"model": "xgboost_optuna", "rmse": rmse_xgb, "mae": mae_xgb},
])
comparison_out.to_csv(f"{BASE_DIR}/model_comparison_final.csv", index=False)

print("\n=== Final model comparison (read from result CSVs) ===")
print(comparison_out.to_string(index=False))

# --- Determine best model automatically instead of hardcoding the label ---
best_by_rmse = comparison_out.loc[comparison_out["rmse"].idxmin()]
best_by_mae = comparison_out.loc[comparison_out["mae"].idxmin()]
print(f"\nBest model by RMSE: {best_by_rmse['model']} (RMSE={best_by_rmse['rmse']:,.2f})")
print(f"Best model by MAE : {best_by_mae['model']} (MAE={best_by_mae['mae']:,.2f})")

print("\nSaved: lightgbm_optuna_search.csv, xgboost_optuna_search.csv, "
      "lightgbm_feature_importance.csv, xgboost_feature_importance.csv, "
      "model_comparison_final.csv")