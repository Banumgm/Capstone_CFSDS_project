"""
Task 3b — Threshold selection
13b_lgbm_threshold_selection.py

Selects the F2-optimal decision threshold for the LightGBM classifier.
The threshold is chosen on a validation split carved out of TRAIN only
(same 80/20 pattern used in 17_calibration_check.py), never on the test
set. Test-set probabilities are touched exactly once, at the end, purely
to report performance at the already-fixed threshold.

The deployed model (lgbm_classifier.pkl) is unchanged -- it is still
fit on 100% of train. A second model, fit on 80% of train with the same
tuned hyperparameters, is used only to generate validation-set
probabilities for threshold selection.
"""
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, classification_report,
    precision_score, recall_score
)

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]
y_test_clf  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"]  = X_test["ecozone"].astype("category")
cat_cols = X_train.select_dtypes(include="category").columns.tolist()

# --- Deployed model (fit on 100% of train) -- used only for the final,
#     single, end-of-cell evaluation on test. Not used for threshold search. ---
lgbm_clf = joblib.load("/Workspace/Capstone_Group1/models/lgbm_classifier.pkl")

# --- Validation split carved out of TRAIN only, same 80/20 pattern as the
#     calibration check. A second model, with the same tuned hyperparameters,
#     is fit on the 80% portion purely to generate validation probabilities. ---
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train_clf, test_size=0.2, stratify=y_train_clf, random_state=42
)

BEST_PARAMS = {
    "num_leaves": 73, "max_depth": 10, "learning_rate": 0.06155574273677577,
    "n_estimators": 495, "min_child_samples": 74
}
scale_pos_weight_fit = (y_fit == 0).sum() / (y_fit == 1).sum()

val_model = lgb.LGBMClassifier(
    **BEST_PARAMS, objective="binary", scale_pos_weight=scale_pos_weight_fit,
    random_state=42, verbosity=-1
)
val_model.fit(X_fit, y_fit, categorical_feature=cat_cols)

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
y_proba_test = lgbm_clf.predict_proba(X_test)[:, 1]
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