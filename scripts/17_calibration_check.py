"""
Task 3b — Probability calibration check
17_calibration_check.py

Tests whether the LightGBM classifier's probabilities are well-calibrated
under heavy class weighting, and whether isotonic calibration improves
precision at the operating threshold. F2-optimal thresholds for both the
uncalibrated and calibrated models are selected on a calibration holdout
carved out of TRAIN only (X_calib/y_calib). Test is touched exactly once,
at the end, to report performance at both already-fixed thresholds.
"""
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, precision_score, recall_score
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

# --- Split a calibration/validation holdout out of TRAIN only (test stays untouched) ---
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train, y_train_clf, test_size=0.2, stratify=y_train_clf, random_state=42
)

scale_pos_weight_fit = (y_fit == 0).sum() / (y_fit == 1).sum()
base_params = {k: v for k, v in best_params.items() if k != "scale_pos_weight"}
base_model = lgb.LGBMClassifier(**base_params, scale_pos_weight=scale_pos_weight_fit)
base_model.fit(X_fit, y_fit, categorical_feature=cat_cols)

calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_calib, y_calib)

# --- Threshold search for BOTH models happens on the calibration/validation holdout ---
def f2_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f2 = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    idx = np.argmax(f2)
    return thresholds[idx], precisions[idx], recalls[idx], f2[idx]

y_proba_uncal_val = base_model.predict_proba(X_calib)[:, 1]
y_proba_cal_val = calibrated_model.predict_proba(X_calib)[:, 1]

thr_uncal, p_uncal_val, r_uncal_val, f2_uncal_val = f2_optimal_threshold(y_calib, y_proba_uncal_val)
thr_cal, p_cal_val, r_cal_val, f2_cal_val = f2_optimal_threshold(y_calib, y_proba_cal_val)

print("-- F2-optimal thresholds selected on VALIDATION --")
print(f"Uncalibrated: threshold={thr_uncal:.3f}, precision={p_uncal_val:.3f}, "
      f"recall={r_uncal_val:.3f}, F2={f2_uncal_val:.3f}")
print(f"Calibrated:   threshold={thr_cal:.3f}, precision={p_cal_val:.3f}, "
      f"recall={r_cal_val:.3f}, F2={f2_cal_val:.3f}")

# --- Test set touched exactly once: apply both already-fixed thresholds, purely to report ---
y_proba_uncal_test = lgbm_clf.predict_proba(X_test)[:, 1]
y_proba_cal_test = calibrated_model.predict_proba(X_test)[:, 1]

y_pred_uncal_test = (y_proba_uncal_test >= thr_uncal).astype(int)
y_pred_cal_test = (y_proba_cal_test >= thr_cal).astype(int)

p_uncal_test = precision_score(y_test_clf, y_pred_uncal_test, zero_division=0)
r_uncal_test = recall_score(y_test_clf, y_pred_uncal_test, zero_division=0)
f2_uncal_test = (5 * p_uncal_test * r_uncal_test) / (4 * p_uncal_test + r_uncal_test + 1e-10)

p_cal_test = precision_score(y_test_clf, y_pred_cal_test, zero_division=0)
r_cal_test = recall_score(y_test_clf, y_pred_cal_test, zero_division=0)
f2_cal_test = (5 * p_cal_test * r_cal_test) / (4 * p_cal_test + r_cal_test + 1e-10)

print("\n--- Test set: uncalibrated vs calibrated, at validation-selected thresholds ---")
print(f"{'Metric':<28}{'Uncalibrated':<15}{'Calibrated':<15}")
print(f"{'Threshold':<28}{thr_uncal:<15.3f}{thr_cal:<15.3f}")
print(f"{'ROC-AUC':<28}{roc_auc_score(y_test_clf, y_proba_uncal_test):<15.4f}"
      f"{roc_auc_score(y_test_clf, y_proba_cal_test):<15.4f}")
print(f"{'PR-AUC':<28}{average_precision_score(y_test_clf, y_proba_uncal_test):<15.4f}"
      f"{average_precision_score(y_test_clf, y_proba_cal_test):<15.4f}")
print(f"{'Brier score (lower=better)':<28}{brier_score_loss(y_test_clf, y_proba_uncal_test):<15.4f}"
      f"{brier_score_loss(y_test_clf, y_proba_cal_test):<15.4f}")
print(f"{'Precision':<28}{p_uncal_test:<15.3f}{p_cal_test:<15.3f}")
print(f"{'Recall':<28}{r_uncal_test:<15.3f}{r_cal_test:<15.3f}")
print(f"{'F2-score':<28}{f2_uncal_test:<15.3f}{f2_cal_test:<15.3f}")