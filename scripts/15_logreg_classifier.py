"""
Task 3b — Logistic Regression: interpretability baseline
15_logreg_classifier.py
Uses the linear-branch features (one-hot + StandardScaler, same as Tweedie
GLM). Included for H1 interpretability via standardized coefficients, not
expected to compete with tree models on raw performance.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
import optuna

X_train_lin = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_linear.csv")
X_test_lin  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_linear.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]
y_test_clf  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

def objective(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 10, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 2000,
    }
    model = LogisticRegression(**params, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_lin, y_train_clf, cv=cv, scoring="average_precision")
    return scores.mean()

sampler_lr = optuna.samplers.TPESampler(seed=42)
study_lr = optuna.create_study(direction="maximize", sampler=sampler_lr)
study_lr.optimize(objective, n_trials=25, show_progress_bar=True)

print("Best CV PR-AUC:", round(study_lr.best_value, 4))
print("Best params:", study_lr.best_params)

best_logreg = LogisticRegression(
    **study_lr.best_params,
    class_weight="balanced",
    solver="liblinear",
    max_iter=2000,
    random_state=42
)
best_logreg.fit(X_train_lin, y_train_clf)

y_pred = best_logreg.predict(X_test_lin)
y_proba = best_logreg.predict_proba(X_test_lin)[:, 1]

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

# Standardized coefficients — ranked by absolute value, for H1
coef_df = pd.DataFrame({
    "feature": X_train_lin.columns,
    "coefficient": best_logreg.coef_[0]
}).assign(abs_coef=lambda d: d["coefficient"].abs()).sort_values("abs_coef", ascending=False)

print("Top 15 standardized coefficients (by magnitude):")
print(coef_df.head(15).to_string(index=False))

os.makedirs("/Workspace/Capstone_Group1/models", exist_ok=True)
joblib.dump(best_logreg, "/Workspace/Capstone_Group1/models/logreg_classifier.pkl")
print("Saved: logreg_classifier.pkl")