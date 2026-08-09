"""
Task 3b — Random Forest classifier: training + Optuna tuning
14_rf_classifier.py
Comparison baseline for the LightGBM classifier above.
Uses class_weight='balanced' for imbalance handling (no SMOTE).
Tuning objective: PR-AUC, same protocol as LightGBM for fair comparison.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
import optuna

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]
y_test_clf  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

cat_cols = ["ecozone"]
X_train_rf = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_rf  = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_train_rf, X_test_rf = X_train_rf.align(X_test_rf, join="left", axis=1, fill_value=0)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_rf, y_train_clf, cv=cv, scoring="average_precision")
    return scores.mean()

sampler_rf = optuna.samplers.TPESampler(seed=42)
study_rf = optuna.create_study(direction="maximize", sampler=sampler_rf)
study_rf.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best CV PR-AUC:", round(study_rf.best_value, 4))
print("Best params:", study_rf.best_params)

best_rf_clf = RandomForestClassifier(
    **study_rf.best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
best_rf_clf.fit(X_train_rf, y_train_clf)

y_pred = best_rf_clf.predict(X_test_rf)
y_proba = best_rf_clf.predict_proba(X_test_rf)[:, 1]

print("--- Test set performance (threshold = 0.5) ---")
print("Precision:", round(precision_score(y_test_clf, y_pred), 4))
print("Recall:   ", round(recall_score(y_test_clf, y_pred), 4))
print("F1:       ", round(f1_score(y_test_clf, y_pred), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test_clf, y_proba), 4))
print("PR-AUC:   ", round(average_precision_score(y_test_clf, y_proba), 4))

cm = confusion_matrix(y_test_clf, y_pred)
print("Confusion matrix:")
print(cm)

report = classification_report(y_test_clf, y_pred)
print(report)

os.makedirs("/Workspace/Capstone_Group1/models", exist_ok=True)
joblib.dump(best_rf_clf, "/Workspace/Capstone_Group1/models/rf_classifier.pkl")
print("Saved: rf_classifier.pkl")