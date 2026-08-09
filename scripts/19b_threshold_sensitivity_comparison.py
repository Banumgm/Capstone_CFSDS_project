"""
Task 3b — Sensitivity analysis comparison: 90th vs 85th percentile
19b_threshold_sensitivity_comparison.py
"""
import pandas as pd

comparison_thresholds = pd.DataFrame([
    {
        "target_definition": "90th percentile",
        "value_m_per_day": 984.44,
        "positive_rate_test": "6.6%",
        "cv_pr_auc": 0.7075,
        "test_roc_auc": 0.9396,
        "test_pr_auc": 0.5654,
        "f2_threshold": 0.027,
        "precision_at_f2": 0.413,
        "recall_at_f2": 0.754,
        "f2_score": 0.647,
    },
    {
        "target_definition": "85th percentile",
        "value_m_per_day": 589.33,
        "positive_rate_test": "10.4%",
        "cv_pr_auc": 0.7857,
        "test_roc_auc": 0.9277,
        "test_pr_auc": 0.6534,
        "f2_threshold": 0.001,
        "precision_at_f2": 0.404,
        "recall_at_f2": 0.851,
        "f2_score": 0.697,
    },
])

print(comparison_thresholds.to_string(index=False))

comparison_thresholds.to_csv(
    "/Workspace/Capstone_Group1/processed/threshold_sensitivity_comparison.csv", index=False
)
print("\nSaved: threshold_sensitivity_comparison.csv")

print("""
Note: the 85th percentile model's F2-optimal threshold (0.001) is
extremely low, indicating a flatter, less-discriminating probability
distribution near the decision boundary compared to the 90th percentile
model's threshold (0.027) -- even though its overall PR-AUC is higher.
This is a property of the model, not an error in threshold selection.
""")