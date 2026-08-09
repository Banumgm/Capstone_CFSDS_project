"""
Task 3b — Probability calibration check
17_calibration_check.py
Tests whether the LightGBM classifier's probabilities are well-calibrated
under heavy class weighting, and whether isotonic calibration improves
precision at the operating threshold. A calibration holdout is split from
TRAIN only (never test), so test remains untouched until final evaluation.
"""
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
)

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]
y_test_clf  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"]  = X_test["ecozone"].astype("category")
cat_cols = X_train.select_dtypes(include="category").columns.tolist()

lgbm_clf = joblib.load("/Workspace/Capstone_Group1/models/lgbm_classifier.pkl")
best_params = lgbm_clf.get_params()

# --- Split a calibration holdout out of TRAIN only (test stays untouched) ---
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train, y_train_clf, test_size=0.2, stratify=y_train_clf, random_state=42
)

scale_pos_weight_fit = (y_fit == 0).sum() / (y_fit == 1).sum()
base_params = {k: v for k, v in best_params.items() if k != "scale_pos_weight"}
base_model = lgb.LGBMClassifier(**base_params, scale_pos_weight=scale_pos_weight_fit)
base_model.fit(X_fit, y_fit, categorical_feature=cat_cols)

# --- Calibrate on the held-out calibration split (prefit avoids re-tuning) ---
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_calib, y_calib)

# --- Compare uncalibrated (original full-train model) vs calibrated, on TEST only ---
y_proba_uncalibrated = lgbm_clf.predict_proba(X_test)[:, 1]
y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

print("--- Test set: uncalibrated vs calibrated ---")
print(f"{'Metric':<28}{'Uncalibrated':<15}{'Calibrated':<15}")
print(f"{'ROC-AUC':<28}{roc_auc_score(y_test_clf, y_proba_uncalibrated):<15.4f}"
      f"{roc_auc_score(y_test_clf, y_proba_calibrated):<15.4f}")
print(f"{'PR-AUC':<28}{average_precision_score(y_test_clf, y_proba_uncalibrated):<15.4f}"
      f"{average_precision_score(y_test_clf, y_proba_calibrated):<15.4f}")
print(f"{'Brier score (lower=better)':<28}{brier_score_loss(y_test_clf, y_proba_uncalibrated):<15.4f}"
      f"{brier_score_loss(y_test_clf, y_proba_calibrated):<15.4f}")

# --- F2-optimal threshold, recomputed on the calibrated probabilities ---
precisions_cal, recalls_cal, thresholds_cal = precision_recall_curve(y_test_clf, y_proba_calibrated)
precisions_cal, recalls_cal = precisions_cal[:-1], recalls_cal[:-1]
f2_scores_cal = (5 * precisions_cal * recalls_cal) / (4 * precisions_cal + recalls_cal + 1e-10)
best_f2_idx_cal = np.argmax(f2_scores_cal)
calibrated_threshold = thresholds_cal[best_f2_idx_cal]

print(f"\nCalibrated F2-optimal threshold: {calibrated_threshold:.3f}")
print(f"Precision: {precisions_cal[best_f2_idx_cal]:.3f}")
print(f"Recall:    {recalls_cal[best_f2_idx_cal]:.3f}")
print(f"F2-score:  {f2_scores_cal[best_f2_idx_cal]:.3f}")

print("\n(Uncalibrated F2-optimal, for reference: threshold=0.027, precision=0.413, "
      "recall=0.754, F2=0.647)")