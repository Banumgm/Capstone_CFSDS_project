"""
36_build_dashboard_exports.py
Rebuilds the final regression/classification models and produces all four
Power BI export tables:
  - wildfire_predictions_powerbi.csv
  - wildfire_shap_powerbi.csv
  - global_shap_importance_powerbi.csv
  - fire_conditions_powerbi.csv
"""

import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap

BASE_DIR = "processed" if os.path.exists("/Workspace/Capstone_Group1/processed") else "processed"
MODEL_DIR = "models" if os.path.exists("/Workspace/Capstone_Group1/models") else "models"

def display(df):
    print(df.to_string() if hasattr(df, "to_string") else df)

print("=" * 70)
print("PATHS")
print("=" * 70)
print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)

# ============================================================
# SECTION 1 — Rebuild / load models (shared by predictions + SHAP)
# ============================================================
print("\n" + "=" * 70)
print("SECTION 1 — REBUILD / LOAD MODELS")
print("=" * 70)

lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
lgb_param_cols = [c for c in lgb_search.index if c != "cv_rmse"]
lgb_int_cols = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
BEST_PARAMS_LGB = {c: (int(lgb_search[c]) if c in lgb_int_cols else float(lgb_search[c])) for c in lgb_param_cols}

X_train = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
y_train = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]
if "ecozone" in X_train.columns:
    X_train["ecozone"] = X_train["ecozone"].astype("category")

final_lgb = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.3, importance_type="gain",
    random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS_LGB
)
final_lgb.fit(X_train, y_train, categorical_feature=["ecozone"])
lgbm_clf = joblib.load(f"{MODEL_DIR}/lgbm_classifier.pkl")
print("Regression model : final_lgb OK")
print("Classification model : lgbm_clf OK")

# ============================================================
# SECTION 2 — Shared test data: X_test, y_test, test_raw (+ date/FireDayKey)
# Loaded once, reused by predictions, SHAP, and fire_conditions tables.
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2 — LOAD SHARED TEST DATA")
print("=" * 70)

X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv").reset_index(drop=True)
y_test = pd.read_csv(f"{BASE_DIR}/y_test.csv").iloc[:, 0].reset_index(drop=True)
test_raw = pd.read_csv(f"{BASE_DIR}/test_temporal.csv").reset_index(drop=True)
if "ecozone" in X_test.columns:
    X_test["ecozone"] = X_test["ecozone"].astype("category")

if not (len(X_test) == len(y_test) == len(test_raw)):
    raise ValueError("Row count mismatch between X_test, y_test, and test_raw.")

required_cols = {"ID", "year", "DOB", "fireday", "province", "lat", "lon"}
missing = required_cols - set(test_raw.columns)
if missing:
    raise ValueError(f"Missing required metadata columns: {missing}")

# True calendar date = Jan 1 of year + (DOB - 1)
test_raw["date"] = pd.to_datetime(test_raw["year"].astype(int).astype(str), format="%Y") + \
    pd.to_timedelta(test_raw["DOB"].astype(int) - 1, unit="D")

# FireDayKey = fire + year + fire-day
test_raw["FireDayKey"] = test_raw["ID"].astype(str) + "|" + test_raw["year"].astype(int).astype(str) + "|" + test_raw["fireday"].astype(int).astype(str)

n_rows, n_unique_keys = len(test_raw), test_raw["FireDayKey"].nunique()
print("Total rows:", n_rows, "| Unique FireDayKey:", n_unique_keys)
if n_rows != n_unique_keys:
    dup = test_raw[test_raw["FireDayKey"].duplicated(keep=False)].sort_values("FireDayKey")
    print("\nWARNING: Duplicate FireDayKey found.")
    display(dup[["ID", "year", "fireday", "DOB", "FireDayKey"]].head(50))
    raise ValueError("FireDayKey is not unique. Stop and investigate duplicates.")

# ============================================================
# SECTION 3 — Prediction table -> wildfire_predictions_powerbi.csv
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3 — PREDICTION TABLE")
print("=" * 70)

CLASSIFICATION_THRESHOLD = 0.212
SPREAD_THRESHOLD = 984.44

regression_predictions = final_lgb.predict(X_test)
classification_probabilities = lgbm_clf.predict_proba(X_test)[:, 1]

prediction_df = pd.DataFrame({
    "FireDayKey": test_raw["FireDayKey"].to_numpy(),
    "fire_id": test_raw["ID"].astype(str).to_numpy(),
    "date": test_raw["date"].to_numpy(),
    "province": test_raw["province"].to_numpy(),
    "latitude": test_raw["lat"].to_numpy(),
    "longitude": test_raw["lon"].to_numpy(),
    "actual_spread": y_test.to_numpy(),
    "predicted_spread": regression_predictions,
    "high_spread_probability": classification_probabilities,
})
prediction_df["high_spread_prediction"] = (prediction_df["high_spread_probability"] >= CLASSIFICATION_THRESHOLD).astype(int)
prediction_df["risk_level"] = np.where(prediction_df["predicted_spread"] >= SPREAD_THRESHOLD, "High", "Normal")
prediction_df["prediction_error"] = prediction_df["actual_spread"] - prediction_df["predicted_spread"]
prediction_df["absolute_error"] = prediction_df["prediction_error"].abs()

print("Rows:", len(prediction_df), "| Unique FireDayKey:", prediction_df["FireDayKey"].nunique())
print("Missing values:\n", prediction_df.isna().sum())
print("Date range:", prediction_df["date"].min(), "to", prediction_df["date"].max())
print("Probability range:", prediction_df["high_spread_probability"].min(), "to", prediction_df["high_spread_probability"].max())
print("Risk levels:\n", prediction_df["risk_level"].value_counts())

check_2019_101 = prediction_df[prediction_df["fire_id"] == "2019_101"].copy()
display(check_2019_101[["FireDayKey", "fire_id", "date", "province", "actual_spread", "predicted_spread", "high_spread_probability", "high_spread_prediction", "risk_level"]].sort_values("date"))
print("Years found for 2019_101:", check_2019_101["date"].dt.year.unique())

prediction_df = prediction_df.sort_values(["date", "predicted_spread"], ascending=[True, False]).reset_index(drop=True)
prediction_df = prediction_df[["FireDayKey", "fire_id", "date", "province", "latitude", "longitude", "actual_spread", "predicted_spread", "high_spread_probability", "high_spread_prediction", "risk_level", "prediction_error", "absolute_error"]]

pred_output_path = f"{BASE_DIR}/wildfire_predictions_powerbi.csv"
prediction_df.to_csv(pred_output_path, index=False)
print(f"Saved: {pred_output_path} ({len(prediction_df):,} rows)")

# ============================================================
# SECTION 4 — SHAP table -> wildfire_shap_powerbi.csv
# Also produces reg_shap_values / clf_shap_values / feature_names,
# reused by Section 5 (global importance).
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4 — SHAP TABLE")
print("=" * 70)

shap_metadata = pd.DataFrame({"fire_id": test_raw["ID"].astype(str).to_numpy(), "date": test_raw["date"].to_numpy()})
if len(shap_metadata) != len(X_test):
    raise ValueError("SHAP metadata and X_test row counts do not match.")

print("Calculating Regression SHAP...")
reg_shap_values = np.asarray(shap.TreeExplainer(final_lgb).shap_values(X_test))
print("Regression SHAP shape:", reg_shap_values.shape)

print("Calculating Classification SHAP...")
clf_shap_values = shap.TreeExplainer(lgbm_clf).shap_values(X_test)
if isinstance(clf_shap_values, list):
    clf_shap_values = clf_shap_values[1]  # positive class
else:
    clf_shap_values = np.asarray(clf_shap_values)
    if clf_shap_values.ndim == 3:
        clf_shap_values = clf_shap_values[:, :, 1]
print("Classification SHAP shape:", clf_shap_values.shape)

feature_names = X_test.columns.tolist()
print("Number of features:", len(feature_names))

if reg_shap_values.shape[0] != len(shap_metadata) or clf_shap_values.shape[0] != len(shap_metadata):
    raise ValueError("SHAP rows do not match metadata rows.")
if reg_shap_values.shape[1] != len(feature_names) or clf_shap_values.shape[1] != len(feature_names):
    raise ValueError("SHAP feature count mismatch.")

def create_shap_long_table(shap_values, model_name, metadata, X_data, feature_names, top_n=15):
    records = []
    shap_values = np.asarray(shap_values)
    for i in range(shap_values.shape[0]):
        abs_shap = np.abs(shap_values[i])
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        for idx in top_indices:
            feature = feature_names[idx]
            records.append({
                "fire_id": metadata.iloc[i]["fire_id"], "date": metadata.iloc[i]["date"],
                "feature": feature, "feature_value": X_data.iloc[i][feature],
                "shap_value": shap_values[i, idx], "abs_shap_value": abs_shap[idx],
                "model_type": model_name,
            })
    return pd.DataFrame(records)

reg_shap_df = create_shap_long_table(reg_shap_values, "Regression", shap_metadata, X_test, feature_names)
clf_shap_df = create_shap_long_table(clf_shap_values, "Classification", shap_metadata, X_test, feature_names)
print("Regression SHAP rows:", len(reg_shap_df), "| Classification SHAP rows:", len(clf_shap_df))

shap_powerbi = pd.concat([reg_shap_df, clf_shap_df], ignore_index=True)
shap_powerbi["feature_value"] = pd.to_numeric(shap_powerbi["feature_value"], errors="coerce")
shap_powerbi = shap_powerbi.sort_values(["fire_id", "date", "model_type", "abs_shap_value"], ascending=[True, True, True, False]).reset_index(drop=True)

expected_rows = len(test_raw) * 15 * 2
actual_rows = len(shap_powerbi)
print("Expected rows:", f"{expected_rows:,}", "| Actual rows:", f"{actual_rows:,}")
print("Missing values:\n", shap_powerbi.isna().sum())
print("Model counts:\n", shap_powerbi["model_type"].value_counts())
if actual_rows != expected_rows:
    raise ValueError("Unexpected SHAP row count.")

shap_output_path = f"{BASE_DIR}/wildfire_shap_powerbi.csv"
shap_powerbi.to_csv(shap_output_path, index=False)
print(f"Saved: {shap_output_path} ({len(shap_powerbi):,} rows)")

# ============================================================
# SECTION 5 — Global SHAP importance table -> global_shap_importance_powerbi.csv
# ============================================================
print("\n" + "=" * 70)
print("SECTION 5 — GLOBAL SHAP IMPORTANCE TABLE")
print("=" * 70)

mean_abs_reg, mean_abs_clf = np.abs(reg_shap_values).mean(axis=0), np.abs(clf_shap_values).mean(axis=0)
mean_signed_reg, mean_signed_clf = reg_shap_values.mean(axis=0), clf_shap_values.mean(axis=0)

global_importance_long = pd.concat([
    pd.DataFrame({"feature": feature_names, "model_type": "Regression", "mean_abs_shap": mean_abs_reg, "mean_signed_shap": mean_signed_reg}),
    pd.DataFrame({"feature": feature_names, "model_type": "Classification", "mean_abs_shap": mean_abs_clf, "mean_signed_shap": mean_signed_clf}),
], ignore_index=True).sort_values(["model_type", "mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)

expected_gi_rows = len(feature_names) * 2
if len(global_importance_long) != expected_gi_rows:
    raise ValueError("Unexpected row count in global_importance_long.")
print("Rows:", len(global_importance_long))
print("Missing values:\n", global_importance_long.isna().sum())
print("Top 10 Regression drivers:\n", global_importance_long[global_importance_long["model_type"] == "Regression"].head(10).to_string(index=False))

global_shap_output_path = f"{BASE_DIR}/global_shap_importance_powerbi.csv"
global_importance_long.to_csv(global_shap_output_path, index=False)
print(f"Saved: {global_shap_output_path} ({len(global_importance_long):,} rows)")

# ============================================================
# SECTION 6 — Fire conditions table -> fire_conditions_powerbi.csv
# ============================================================
print("\n" + "=" * 70)
print("SECTION 6 — FIRE CONDITIONS TABLE")
print("=" * 70)

page3_columns = [
    "FireDayKey", "ID", "year", "DOB", "fireday", "date", "province", "lat", "lon",
    "fwi", "ffmc", "isi", "bui", "tmax", "rh", "ws",
    "Biomass", "nonfuel1k", "slope", "dem", "aspect", "twi", "hydrodens2k", "roaddist",
    "ecozone", "peatprop",
    "fireday_sin", "fireday_cos",
    "ws_x_slope", "aspect_x_season",
]
missing_page3 = [c for c in page3_columns if c not in test_raw.columns]
if missing_page3:
    print("\n[WARNING] Missing Page 3 columns:", missing_page3)
    related_terms = ["fwi", "ffmc", "isi", "bui", "tmax", "rh", "ws", "biomass", "fuel", "nonfuel", "slope", "dem", "aspect", "twi", "hydro", "road", "peat"]
    related_columns = sorted([c for c in test_raw.columns if any(t in c.lower() for t in related_terms)])
    print("Available related columns:", related_columns)
    raise ValueError("One or more Page 3 columns are missing. Review column names above.")

fire_conditions_powerbi = test_raw[page3_columns].copy().rename(columns={
    "ID": "fire_id", "lat": "latitude", "lon": "longitude", "dem": "elevation",
    "tmax": "temperature", "rh": "humidity", "ws": "wind_speed", "Biomass": "biomass",
    "nonfuel1k": "fuel_continuity_1km", "hydrodens2k": "hydrological_density_2km", "roaddist": "distance_to_road",
})

numeric_columns = ["year", "DOB", "fireday", "latitude", "longitude", "fwi", "ffmc", "isi", "bui",
    "temperature", "humidity", "wind_speed", "biomass", "fuel_continuity_1km", "slope", "dem",
    "aspect", "twi", "hydrological_density_2km", "distance_to_road", "peatprop",
    "fireday_sin", "fireday_cos", "ws_x_slope", "aspect_x_season"]
for col in numeric_columns:
    if col in fire_conditions_powerbi.columns:
        fire_conditions_powerbi[col] = pd.to_numeric(fire_conditions_powerbi[col], errors="coerce")

missing_summary = fire_conditions_powerbi.isna().sum().sort_values(ascending=False)
print("Missing values:\n", missing_summary[missing_summary > 0])
print("Date range:", fire_conditions_powerbi["date"].min(), "to", fire_conditions_powerbi["date"].max())

check_2019_101_fc = fire_conditions_powerbi[fire_conditions_powerbi["fire_id"] == "2019_101"].copy().sort_values("date")
display(check_2019_101_fc[["FireDayKey", "fire_id", "date", "province", "fwi", "ffmc", "isi", "bui", "temperature", "humidity", "wind_speed", "fuel_continuity_1km", "hydrological_density_2km", "distance_to_road", "slope", "elevation", "aspect", "twi"]])
print("Years found:", check_2019_101_fc["date"].dt.year.unique())

fire_conditions_powerbi = fire_conditions_powerbi.sort_values(["date", "fire_id", "fireday"]).reset_index(drop=True)
print("Rows:", f"{len(fire_conditions_powerbi):,}", "| Unique FireDayKey:", f"{fire_conditions_powerbi['FireDayKey'].nunique():,}")
print("Columns:", fire_conditions_powerbi.columns.tolist())

conditions_output_path = f"{BASE_DIR}/fire_conditions_powerbi.csv"
fire_conditions_powerbi.to_csv(conditions_output_path, index=False)
print(f"Saved: {conditions_output_path} ({len(fire_conditions_powerbi):,} rows)")

# ============================================================
# DONE — list output files
# ============================================================
print("\n" + "=" * 70)
print("ALL DASHBOARD EXPORT TABLES CREATED SUCCESSFULLY")
print("=" * 70)
for f in sorted(os.listdir(BASE_DIR)):
    if f.endswith("_powerbi.csv"):
        print(" -", f)