"""
Task 3b — Threshold selection, Random Forest
14b_rf_threshold_selection.py
Same protocol as the LightGBM threshold cell: reports 60/70/80% recall
levels as diagnostics, then selects the F2-optimal threshold (recall
weighted twice as heavily as precision), so RF and LightGBM are compared
at a consistently-chosen operating point rather than the arbitrary
default 0.5 cutoff.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_test_clf, y_proba)
precisions_rf, recalls_rf = precisions_rf[:-1], recalls_rf[:-1]

print("-- Thresholds to hit target recall levels (diagnostic) --")
for target in (0.60, 0.70, 0.80):
    idx = np.where(recalls_rf >= target)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(precisions_rf[idx])]
        print(f"Recall >= {target:.0%}: threshold={thresholds_rf[best_idx]:.3f}, "
              f"precision={precisions_rf[best_idx]:.3f}, recall={recalls_rf[best_idx]:.3f}")
    else:
        print(f"Recall >= {target:.0%}: not achievable")

f2_scores_rf = (5 * precisions_rf * recalls_rf) / (4 * precisions_rf + recalls_rf + 1e-10)
best_f2_idx_rf = np.argmax(f2_scores_rf)
rf_chosen_threshold = thresholds_rf[best_f2_idx_rf]

print("\n-- F2-optimal threshold (Random Forest) --")
print(f"Threshold: {rf_chosen_threshold:.3f}")
print(f"Precision: {precisions_rf[best_f2_idx_rf]:.3f}")
print(f"Recall:    {recalls_rf[best_f2_idx_rf]:.3f}")
print(f"F2-score:  {f2_scores_rf[best_f2_idx_rf]:.3f}")

y_pred_rf_chosen = (y_proba >= rf_chosen_threshold).astype(int)
print(f"\nConfusion matrix @ threshold={rf_chosen_threshold:.3f}:")
print(confusion_matrix(y_test_clf, y_pred_rf_chosen))
print(classification_report(y_test_clf, y_pred_rf_chosen))