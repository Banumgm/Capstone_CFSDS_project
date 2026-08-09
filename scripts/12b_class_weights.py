"""
Task 3a (cont.) — Class weighting setup
12b_class_weights.py
Computes scale_pos_weight (boosting) from TRAIN class counts only.
sklearn's class_weight='balanced' is computed automatically at fit time,
so no manual value is needed for RF / Logistic Regression.
"""
import pandas as pd

y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]

n_neg = (y_train_clf == 0).sum()
n_pos = (y_train_clf == 1).sum()
SCALE_POS_WEIGHT = n_neg / n_pos

print(f"Train negatives (low-spread):  {n_neg}")
print(f"Train positives (high-spread): {n_pos}")
print(f"scale_pos_weight (for LightGBM/XGBoost): {SCALE_POS_WEIGHT:.3f}")