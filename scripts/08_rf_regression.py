"""
08_rf_regression.py
Combines the RandomizedSearchCV hyperparameter search (GroupKFold, grouped
by fire ID) with the final Random Forest fit. BEST_PARAMS is READ from the
search results (no manual re-entry). H1 comparison block omitted (deferred
to Phase 4 SHAP, PR #24); top-15 feature importance retained.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

X_train_tree = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test_tree  = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
y_train = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]
y_test  = pd.read_csv(f"{BASE_DIR}/y_test.csv").iloc[:, 0]

X_train_rf = pd.get_dummies(X_train_tree, columns=["ecozone"], prefix="eco", drop_first=True)
X_test_rf  = pd.get_dummies(X_test_tree,  columns=["ecozone"], prefix="eco", drop_first=True)
X_test_rf  = X_test_rf.reindex(columns=X_train_rf.columns, fill_value=0)

dummy_cols = [c for c in X_train_rf.columns if c.startswith("eco_")]
X_train_rf[dummy_cols] = X_train_rf[dummy_cols].astype(int)
X_test_rf[dummy_cols]  = X_test_rf[dummy_cols].astype(int)

TRAIN_FILE = f"{BASE_DIR}/train_temporal.csv"
train_df_raw = pd.read_csv(TRAIN_FILE)
groups = train_df_raw["ID"].values

# =========================================================
# PART 1: Hyperparameter search (RandomizedSearchCV)
# =========================================================
param_dist = {
    "n_estimators": [200, 300],
    "max_depth": [10, 15, 25, None],
    "min_samples_leaf": [1, 2, 5],
    "max_features": [0.3, 0.5, "sqrt"],
    "min_samples_split": [2, 5, 10],
}

gkf = GroupKFold(n_splits=3)
rf = RandomForestRegressor(random_state=42, n_jobs=1)

random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=15, cv=gkf.split(X_train_rf, y_train, groups=groups),
    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1, random_state=42
)
random_search.fit(X_train_rf, y_train)

print("Best params:", random_search.best_params_)
print(f"Best CV RMSE: {-random_search.best_score_:,.2f}")
pd.DataFrame([{**random_search.best_params_, "cv_rmse": -random_search.best_score_}]
             ).to_csv(f"{BASE_DIR}/rf_hyperparameter_search.csv", index=False)

# =========================================================
# PART 2: Final fit using searched BEST_PARAMS (read, not re-typed)
# =========================================================
BEST_PARAMS = random_search.best_params_

rf_final = RandomForestRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1)
rf_final.fit(X_train_rf, y_train)

preds = rf_final.predict(X_test_rf)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print(f"\n=== Random Forest test performance ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE : {mae:,.2f}")

importance = pd.Series(rf_final.feature_importances_, index=X_train_rf.columns).sort_values(ascending=False)
print("\nTop 15 feature importances:")
print(importance.head(15))

RF_OUT = f"{BASE_DIR}/rf_v3_results.csv"
pd.DataFrame([{"model": "random_forest_v3", "rmse": rmse, "mae": mae}]).to_csv(RF_OUT, index=False)
importance.to_csv(f"{BASE_DIR}/rf_v3_feature_importance.csv")
print(f"\nSaved: rf_hyperparameter_search.csv, {RF_OUT}, rf_v3_feature_importance.csv")