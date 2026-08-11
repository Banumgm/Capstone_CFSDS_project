"""
Task 3b — Sensitivity analysis: 85th percentile target threshold
19_threshold_85pct_sensitivity.py

Tests an alternative target definition (85th percentile instead of 90th)
on the same feature set, to check how sensitive classification performance
is to the choice of "high-spread" cutoff. 90th percentile remains the
primary target definition; this serves as a documented robustness check.

The F2-optimal threshold is selected on a validation split carved out of
TRAIN only, not on test. Test is touched exactly once, at the end, purely
to report performance at the already-fixed threshold.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report, precision_score, recall_score
)

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"]  = X_test["ecozone"].astype("category")

y_train_reg = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train.csv").iloc[:, 0]
y_test_reg  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test.csv").iloc[:, 0]

THRESHOLD_85 = y_train_reg.quantile(0.85)
print(f"85th percentile threshold: {THRESHOLD_85:.2f} m/day (90th percentile was 984.44)")

y_train_clf_85 = (y_train_reg >= THRESHOLD_85).astype(int)
y_test_clf_85  = (y_test_reg  >= THRESHOLD_85).astype(int)

print(f"\nTrain positives: {y_train_clf_85.sum()}")
print(f"Test positives:  {y_test_clf_85.sum()}")

cat_cols = X_train.select_dtypes(include="category").columns.tolist()

def objective(trial):
    scale_pos_weight = (y_train_clf_85 == 0).sum() / (y_train_clf_85 == 1).sum()
    params = {
        "objective": "binary", "metric": "average_precision", "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    model = lgb.LGBMClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train_clf_85, cv=cv,
        scoring="average_precision",
        params={"categorical_feature": cat_cols} if cat_cols else None
    )
    return scores.mean()

sampler_85 = optuna.samplers.TPESampler(seed=42)
study_85 = optuna.create_study(direction="maximize", sampler=sampler_85)
study_85.optimize(objective, n_trials=40, show_progress_bar=True)

print("\nBest CV PR-AUC (85th percentile):", round(study_85.best_value, 4))
print("Best params:", study_85.best_params)

SCALE_POS_WEIGHT_85_FULL = (y_train_clf_85 == 0).sum() / (y_train_clf_85 == 1).sum()

# --- Deployed 85th-percentile model, fit on 100% of train ---
best_lgbm_85 = lgb.LGBMClassifier(
    **study_85.best_params, objective="binary",
    scale_pos_weight=SCALE_POS_WEIGHT_85_FULL, random_state=42, verbosity=-1
)
best_lgbm_85.fit(X_train, y_train_clf_85, categorical_feature=cat_cols)

y_proba_85_test = best_lgbm_85.predict_proba(X_test)[:, 1]
print("\n--- Test performance @ 0.5 threshold ---")
print("ROC-AUC:", round(roc_auc_score(y_test_clf_85, y_proba_85_test), 4))
print("PR-AUC: ", round(average_precision_score(y_test_clf_85, y_proba_85_test), 4))

# --- Validation split carved out of TRAIN only, for threshold selection ---
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train_clf_85, test_size=0.2, stratify=y_train_clf_85, random_state=42
)
scale_pos_weight_fit = (y_fit == 0).sum() / (y_fit == 1).sum()

val_model_85 = lgb.LGBMClassifier(
    **study_85.best_params, objective="binary",
    scale_pos_weight=scale_pos_weight_fit, random_state=42, verbosity=-1
)
val_model_85.fit(X_fit, y_fit, categorical_feature=cat_cols)
y_proba_85_val = val_model_85.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_85_val)
precisions, recalls = precisions[:-1], recalls[:-1]
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
best_f2_idx = np.argmax(f2_scores)
f2_threshold_85 = thresholds[best_f2_idx]

print("\n-- F2-optimal threshold (85th percentile, selected on VALIDATION) --")
print(f"Threshold: {f2_threshold_85:.3f}")
print(f"Validation precision: {precisions[best_f2_idx]:.3f}")
print(f"Validation recall:    {recalls[best_f2_idx]:.3f}")
print(f"Validation F2-score:  {f2_scores[best_f2_idx]:.3f}")

# --- Test touched exactly once, at the fixed threshold, using the deployed model ---
y_pred_test = (y_proba_85_test >= f2_threshold_85).astype(int)
test_precision = precision_score(y_test_clf_85, y_pred_test, zero_division=0)
test_recall = recall_score(y_test_clf_85, y_pred_test, zero_division=0)
test_f2 = (5 * test_precision * test_recall) / (4 * test_precision + test_recall + 1e-10)

print(f"\n-- Test-set performance @ validation-selected threshold ({f2_threshold_85:.3f}) --")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F2-score:  {test_f2:.3f}")
print(f"\nConfusion matrix @ threshold={f2_threshold_85:.3f}:")
print(confusion_matrix(y_test_clf_85, y_pred_test))
print(classification_report(y_test_clf_85, y_pred_test))

os.makedirs("/Workspace/Capstone_Group1/models", exist_ok=True)
joblib.dump(best_lgbm_85, "/Workspace/Capstone_Group1/models/lgbm_classifier_85pct.pkl")
print("\nSaved: lgbm_classifier_85pct.pkl")