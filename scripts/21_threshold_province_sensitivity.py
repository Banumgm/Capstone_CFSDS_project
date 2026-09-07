"""
Task 3b — Sensitivity analysis: province-specific threshold target
21_threshold_province_sensitivity.py

Tests whether defining "high-spread day" using each row's own province
threshold (computed on TRAIN only, per province) changes classification
performance versus the combined 90th-percentile target. Same feature set
(X_train_tree/X_test_tree), same LightGBM tuning protocol as the primary
model, for a fair comparison. Cross-validation and threshold selection
are both grouped by fire ID, consistent with the primary model.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import os
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report, precision_score, recall_score
)

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

# --- Load feature sets (same as primary classifier) ---
X_train = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test  = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")

# --- Load raw temporal splits to get province + target (province was
#     dropped from the feature set, so pull it back in by row order --
#     row alignment between X_train_tree.csv and train_temporal.csv was
#     verified independently before relying on this) ---
train_raw = pd.read_csv(f"{BASE_DIR}/train_temporal.csv")
test_raw  = pd.read_csv(f"{BASE_DIR}/test_temporal.csv")

TARGET = "sprdistm"
PERCENTILE = 0.90

# --- Per-province thresholds, computed on TRAIN only (no leakage) ---
province_thresholds = train_raw.groupby("province")[TARGET].quantile(PERCENTILE)
print("Per-province 90th percentile thresholds (train only):")
print(province_thresholds.to_string())

# --- Build province-specific binary labels ---
y_train_clf_prov = (
    train_raw[TARGET] >= train_raw["province"].map(province_thresholds)
).astype(int)
y_test_clf_prov = (
    test_raw[TARGET] >= test_raw["province"].map(province_thresholds)
).astype(int)

print(f"\nPositive rate (train, province-specific target): {y_train_clf_prov.mean():.1%}")
print(f"Positive rate (test,  province-specific target): {y_test_clf_prov.mean():.1%}")

# --- Same categorical handling as primary classifier ---
X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"]  = X_test["ecozone"].astype("category")
cat_cols = X_train.select_dtypes(include="category").columns.tolist()
fire_ids = train_raw["ID"]

def objective(trial):
    scale_pos_weight = (y_train_clf_prov == 0).sum() / (y_train_clf_prov == 1).sum()
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    model = lgb.LGBMClassifier(**params, random_state=42)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train_clf_prov, groups=fire_ids, cv=cv,
        scoring="average_precision",
        params={"categorical_feature": cat_cols} if cat_cols else None
    )
    return scores.mean()

sampler = optuna.samplers.TPESampler(seed=42)
study_prov = optuna.create_study(direction="maximize", sampler=sampler)
study_prov.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest CV PR-AUC (province-specific threshold): {study_prov.best_value:.4f}")
print(f"Best params: {study_prov.best_params}")

SCALE_POS_WEIGHT_PROV_FULL = (y_train_clf_prov == 0).sum() / (y_train_clf_prov == 1).sum()

best_lgbm_prov = lgb.LGBMClassifier(
    **study_prov.best_params, objective="binary",
    scale_pos_weight=SCALE_POS_WEIGHT_PROV_FULL, random_state=42, verbosity=-1
)
best_lgbm_prov.fit(X_train, y_train_clf_prov, categorical_feature=cat_cols)

y_proba_prov_test = best_lgbm_prov.predict_proba(X_test)[:, 1]
print("\n--- Test performance @ 0.5 threshold ---")
print("ROC-AUC:", round(roc_auc_score(y_test_clf_prov, y_proba_prov_test), 4))
print("PR-AUC: ", round(average_precision_score(y_test_clf_prov, y_proba_prov_test), 4))

# --- Validation split carved out of TRAIN only, grouped by fire ID ---
group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fit_idx, val_idx = next(group_kfold.split(X_train, y_train_clf_prov, groups=fire_ids))
X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
y_fit, y_val = y_train_clf_prov.iloc[fit_idx], y_train_clf_prov.iloc[val_idx]

scale_pos_weight_fit = (y_fit == 0).sum() / (y_fit == 1).sum()

val_model_prov = lgb.LGBMClassifier(
    **study_prov.best_params, objective="binary",
    scale_pos_weight=scale_pos_weight_fit, random_state=42, verbosity=-1
)
val_model_prov.fit(X_fit, y_fit, categorical_feature=cat_cols)
y_proba_prov_val = val_model_prov.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_prov_val)
precisions, recalls = precisions[:-1], recalls[:-1]
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
best_f2_idx = np.argmax(f2_scores)
f2_threshold_prov = thresholds[best_f2_idx]

print("\n-- F2-optimal threshold (province-specific, selected on VALIDATION) --")
print(f"Threshold: {f2_threshold_prov:.3f}")
print(f"Validation precision: {precisions[best_f2_idx]:.3f}")
print(f"Validation recall:    {recalls[best_f2_idx]:.3f}")
print(f"Validation F2-score:  {f2_scores[best_f2_idx]:.3f}")

# --- Test touched exactly once, at the fixed threshold ---
y_pred_test = (y_proba_prov_test >= f2_threshold_prov).astype(int)
test_precision = precision_score(y_test_clf_prov, y_pred_test, zero_division=0)
test_recall = recall_score(y_test_clf_prov, y_pred_test, zero_division=0)
test_f2 = (5 * test_precision * test_recall) / (4 * test_precision + test_recall + 1e-10)

print(f"\n-- Test-set performance @ validation-selected threshold ({f2_threshold_prov:.3f}) --")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F2-score:  {test_f2:.3f}")
print(f"\nConfusion matrix @ threshold={f2_threshold_prov:.3f}:")
print(confusion_matrix(y_test_clf_prov, y_pred_test))
print(classification_report(y_test_clf_prov, y_pred_test))

print("\n-- Performance by province (province-specific target model) --")
eval_df = pd.DataFrame({
    "province": test_raw["province"].values,
    "y_true": y_test_clf_prov.values,
    "y_pred": y_pred_test,
})
for prov in eval_df["province"].unique():
    sub = eval_df[eval_df["province"] == prov]
    if sub["y_true"].sum() > 0:
        prec = precision_score(sub["y_true"], sub["y_pred"], zero_division=0)
        rec = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)
        print(f"{prov}: n={len(sub)}, precision={prec:.3f}, recall={rec:.3f}, "
              f"positive_rate_true={sub['y_true'].mean():.1%}")

os.makedirs("/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models", exist_ok=True)
joblib.dump(best_lgbm_prov,
    (f"/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models") + "/lgbm_classifier_province.pkl")
print("\nSaved: lgbm_classifier_province.pkl")