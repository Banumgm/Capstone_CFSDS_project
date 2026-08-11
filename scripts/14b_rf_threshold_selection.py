"""
Task 3b — Threshold selection, Random Forest
14b_rf_threshold_selection.py

Selects the F2-optimal decision threshold for the Random Forest classifier.
The threshold is chosen on a validation split carved out of TRAIN only,
never on the test set. Test-set probabilities are touched exactly once,
at the end, purely to report performance at the already-fixed threshold.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, classification_report,
    precision_score, recall_score
)

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]
y_test_clf  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

# --- One-hot encode ecozone, aligned across train/test (same as 14_rf_classifier.py) ---
cat_cols = ["ecozone"]
X_train_rf = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_rf  = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_train_rf, X_test_rf = X_train_rf.align(X_test_rf, join="left", axis=1, fill_value=0)

# --- Deployed model (fit on 100% of train) -- used only for the final,
#     single, end-of-cell evaluation on test. Not used for threshold search. ---
rf_clf = joblib.load("/Workspace/Capstone_Group1/models/rf_classifier.pkl")

# --- Validation split carved out of TRAIN only (already one-hot encoded,
#     so fit/val share the same columns as X_test_rf) ---
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_rf, y_train_clf, test_size=0.2, stratify=y_train_clf, random_state=42
)

BEST_PARAMS = {"n_estimators": 543, "max_depth": 20, "min_samples_leaf": 5}

val_model = RandomForestClassifier(
    **BEST_PARAMS, class_weight="balanced", random_state=42, n_jobs=-1
)
val_model.fit(X_fit, y_fit)

y_proba_val = val_model.predict_proba(X_val)[:, 1]

# --- Threshold search happens ONLY on the validation set ---
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)
precisions, recalls = precisions[:-1], recalls[:-1]

print("-- Thresholds to hit target recall levels (diagnostic, on VALIDATION) --")
for target in (0.60, 0.70, 0.80):
    idx = np.where(recalls >= target)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(precisions[idx])]
        print(f"Recall >= {target:.0%}: threshold={thresholds[best_idx]:.3f}, "
              f"precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}")
    else:
        print(f"Recall >= {target:.0%}: not achievable")

f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
best_f2_idx = np.argmax(f2_scores)
chosen_threshold = thresholds[best_f2_idx]

print("\n-- F2-optimal threshold (selected on VALIDATION) --")
print(f"Threshold: {chosen_threshold:.3f}")
print(f"Validation precision: {precisions[best_f2_idx]:.3f}")
print(f"Validation recall:    {recalls[best_f2_idx]:.3f}")
print(f"Validation F2-score:  {f2_scores[best_f2_idx]:.3f}")

# --- Test set touched exactly once: apply the already-fixed threshold,
#     using the deployed (100%-train) model's probabilities, purely to report. ---
y_proba_test = rf_clf.predict_proba(X_test_rf)[:, 1]
y_pred_test = (y_proba_test >= chosen_threshold).astype(int)

test_precision = precision_score(y_test_clf, y_pred_test, zero_division=0)
test_recall = recall_score(y_test_clf, y_pred_test, zero_division=0)
test_f2 = (5 * test_precision * test_recall) / (4 * test_precision + test_recall + 1e-10)

print(f"\n-- Test-set performance @ validation-selected threshold ({chosen_threshold:.3f}) --")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F2-score:  {test_f2:.3f}")
print(f"\nConfusion matrix @ threshold={chosen_threshold:.3f}:")
print(confusion_matrix(y_test_clf, y_pred_test))
print(classification_report(y_test_clf, y_pred_test))
