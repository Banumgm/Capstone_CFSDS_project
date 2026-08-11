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
        "f2_threshold": 0.120,      
        "precision_at_f2": 0.513,
        "recall_at_f2": 0.601,
        "f2_score": 0.581,
    },
    {
    "target_definition": "85th percentile",
    "value_m_per_day": 589.33,
    "positive_rate_test": "10.4%",
    "cv_pr_auc": 0.7857,
    "test_roc_auc": 0.9277,
    "test_pr_auc": 0.6534,
    "f2_threshold": 0.003,
    "precision_at_f2": 0.460,
    "recall_at_f2": 0.778,
    "f2_score": 0.683,
    },
])

print(comparison_thresholds.to_string(index=False))
comparison_thresholds.to_csv(
    "/Workspace/Capstone_Group1/processed/threshold_sensitivity_comparison.csv", index=False
)
print("\nSaved: threshold_sensitivity_comparison.csv")