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
        "cv_pr_auc": 0.6674,
        "test_roc_auc": 0.9404,
        "test_pr_auc": 0.5719,
        "f2_threshold": 0.212,
        "precision_at_f2": 0.468,
        "recall_at_f2": 0.700,
        "f2_score": 0.637,
    },
    {
        "target_definition": "85th percentile",
        "value_m_per_day": 589.33,
        "positive_rate_test": "10.4%",
        "cv_pr_auc": 0.7341,
        "test_roc_auc": 0.9307,
        "test_pr_auc": 0.6551,
        "f2_threshold": 0.393,
        "precision_at_f2": 0.519,
        "recall_at_f2": 0.702,
        "f2_score": 0.656,
    },
])

print(comparison_thresholds.to_string(index=False))

comparison_thresholds.to_csv(
    "/Workspace/Capstone_Group1/processed/threshold_sensitivity_comparison.csv", index=False
)
print("\nSaved: threshold_sensitivity_comparison.csv")

print("""
Note: the 85th percentile target definition outperforms the 90th
percentile (primary) definition on cross-validated PR-AUC, test PR-AUC,
precision, and F2-score at each definition's own validation-selected
threshold, with essentially equivalent recall. The 90th percentile is
retained as the primary target definition on operational grounds -- a
rarer, more genuinely extreme "high-spread" category is judged a more
trustworthy and actionable early-warning signal than the top 15% of days
more broadly -- rather than because of any performance advantage; this
is a deliberate tradeoff, not a statistically-driven choice.
""")