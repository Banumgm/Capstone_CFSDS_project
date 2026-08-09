"""
Task 3b — LightGBM classifier: training + Optuna tuning
13_lgbm_classifier.py
Primary classification model for high-spread vs low-spread days.
Uses scale_pos_weight for imbalance handling (no SMOTE).
Tuning objective: PR-AUC (average precision), not accuracy/log-loss.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]
y_test_clf  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"]  = X_test["ecozone"].astype("category")

SCALE_POS_WEIGHT = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()
cat_cols = X_train.select_dtypes(include="category").columns.tolist()
print(f"Categorical columns (native): {cat_cols}")
print(f"Feature count: {X_train.shape[1]}")

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    model = lgb.LGBMClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train_clf, cv=cv,
        scoring="average_precision",
        params={"categorical_feature": cat_cols} if cat_cols else None
    )
    return scores.mean()

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest CV PR-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_lgbm_clf = lgb.LGBMClassifier(
    **study.best_params,
    objective="binary",
    scale_pos_weight=SCALE_POS_WEIGHT,
    random_state=42,
    verbosity=-1
)
best_lgbm_clf.fit(X_train, y_train_clf, categorical_feature=cat_cols)

y_pred = best_lgbm_clf.predict(X_test)
y_proba = best_lgbm_clf.predict_proba(X_test)[:, 1]

print("\n--- Test set performance (threshold = 0.5) ---")
print(f"Precision: {precision_score(y_test_clf, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test_clf, y_pred):.4f}")
print(f"F1:        {f1_score(y_test_clf, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_clf, y_proba):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test_clf, y_proba):.4f}")
print(f"\nConfusion matrix:\n{confusion_matrix(y_test_clf, y_pred)}")
print(f"\n{classification_report(y_test_clf, y_pred)}")

import joblib
import os
os.makedirs("/Workspace/Capstone_Group1/models", exist_ok=True)
joblib.dump(best_lgbm_clf, "/Workspace/Capstone_Group1/models/lgbm_classifier.pkl")
print("\nSaved: lgbm_classifier.pkl")