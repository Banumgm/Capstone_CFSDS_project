"""
Task 3b — Threshold selection
13b_lgbm_threshold_selection.py
Uses a precision-recall curve on the LightGBM classifier's test-set
probabilities to identify an operating threshold, rather than defaulting
to the standard 0.5 cutoff. Reports thresholds for 60/70/80% recall
targets as diagnostics, then selects the threshold that maximizes the
F2-score — which formally weights recall twice as heavily as precision,
consistent with the early-warning framing where missing a high-spread
day is costlier than a false alarm.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

X_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/X_test_tree.csv")
y_test_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv").iloc[:, 0]

X_test["ecozone"] = X_test["ecozone"].astype("category")

lgbm_clf = joblib.load("/Workspace/Capstone_Group1/models/lgbm_classifier.pkl")
y_proba = lgbm_clf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test_clf, y_proba)
precisions, recalls = precisions[:-1], recalls[:-1]

print("-- Thresholds to hit target recall levels (diagnostic) --")
for target in (0.60, 0.70, 0.80):
    idx = np.where(recalls >= target)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(precisions[idx])]
        print(f"Recall >= {target:.0%}: threshold={thresholds[best_idx]:.3f}, "
              f"precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}")
    else:
        print(f"Recall >= {target:.0%}: not achievable")

# F2-score: weights recall twice as heavily as precision, matching the
# early-warning framing (missing a high-spread day costs more than a
# false alarm) — same objective used for the 85th-percentile sensitivity model.
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
best_f2_idx = np.argmax(f2_scores)
chosen_threshold = thresholds[best_f2_idx]

print("\n-- F2-optimal threshold --")
print(f"Threshold: {chosen_threshold:.3f}")
print(f"Precision: {precisions[best_f2_idx]:.3f}")
print(f"Recall:    {recalls[best_f2_idx]:.3f}")
print(f"F2-score:  {f2_scores[best_f2_idx]:.3f}")

y_pred_chosen = (y_proba >= chosen_threshold).astype(int)
print(f"\nConfusion matrix @ threshold={chosen_threshold:.3f}:")
print(confusion_matrix(y_test_clf, y_pred_chosen))
print(classification_report(y_test_clf, y_pred_chosen))