"""
Task 3b — Chosen model and threshold configuration
20_classifier_config.py
"""
import json

CHOSEN_MODEL = "lgbm_classifier.pkl"
CHOSEN_THRESHOLD = 0.212

config = {
    "chosen_model": CHOSEN_MODEL,
    "chosen_threshold": CHOSEN_THRESHOLD,
    "threshold_selection_method": "F2-optimal (recall weighted 2x precision), "
                                   "selected on a validation split carved out "
                                   "of TRAIN only, grouped by fire ID",
    "cross_validation_method": "StratifiedGroupKFold, grouped by fire ID, "
                                "5 folds",
    "target_definition": {"percentile": 0.90, "value_m_per_day": 984.44},
    "feature_set": "49 features, ecozone native categorical",
    "test_performance": {
        "cv_pr_auc": 0.6674,
        "test_roc_auc": 0.9404,
        "test_pr_auc": 0.5719,
        "precision_at_threshold": 0.468,
        "recall_at_threshold": 0.700,
        "f2_score": 0.637
    },
    "rationale": "90th percentile retained as the primary target definition: "
                 "consistent basis for comparison across LightGBM, Random Forest, "
                 "and Logistic Regression, and reflects a rare, operationally "
                 "meaningful 'high-spread' alert category for an early-warning "
                 "use case. F2-optimal threshold chosen over a fixed recall "
                 "target, since it formally weights recall twice as heavily as "
                 "precision -- consistent with prioritizing detection of "
                 "high-spread days over minimizing false alarms, rather than "
                 "picking an arbitrary round recall number. Cross-validation "
                 "and validation-split threshold selection are both grouped by "
                 "fire ID, since a single fire contributes multiple burn-day "
                 "rows; this ensures every evaluation is against fires the "
                 "model has not seen any day of, not just unseen days. "
                 "LightGBM chosen over Random Forest and Logistic Regression: "
                 "it achieves the best cross-validated PR-AUC, test ROC-AUC, "
                 "test PR-AUC, and the best F2-score at its own operating "
                 "threshold among all three models, with no metric on which "
                 "it trails either alternative.",
    "sensitivity_analysis": {
        "target_percentile_85": {
            "target_definition": {"percentile": 0.85, "value_m_per_day": 589.33},
            "cv_pr_auc": 0.7341,
            "test_roc_auc": 0.9307,
            "test_pr_auc": 0.6551,
            "f2_threshold": 0.393,
            "precision_at_f2": 0.519,
            "recall_at_f2": 0.702,
            "f2_score": 0.656,
            "note": "Tested as an alternative target definition, evaluated at "
                    "its own validation-selected F2-optimal point. Outperforms "
                    "the 90th-percentile (primary) definition on CV PR-AUC, "
                    "test PR-AUC, precision, and F2-score, with essentially "
                    "equivalent recall; only ROC-AUC modestly favors the 90th "
                    "percentile. Not adopted as primary despite this: a rarer, "
                    "more genuinely extreme 'high-spread' category is judged a "
                    "more trustworthy and actionable early-warning signal than "
                    "the top 15% of days more broadly. This is a deliberate "
                    "operational tradeoff against a measured performance cost, "
                    "not a statistically-driven choice."
        },
        "province_specific_90th_percentile": {
            "target_definition": {
                "method": "per-province 90th percentile, computed on TRAIN only",
                "alberta_value_m_per_day": 1291.39,
                "british_columbia_value_m_per_day": 843.30
            },
            "cv_pr_auc": 0.6437,
            "test_roc_auc": 0.9385,
            "test_pr_auc": 0.5498,
            "f2_threshold": 0.225,
            "precision_at_f2": 0.370,
            "recall_at_f2": 0.779,
            "f2_score": 0.638,
            "performance_by_province": {
                "british_columbia": {"n": 12605, "precision": 0.356, "recall": 0.764, "positive_rate_true": 0.065},
                "alberta": {"n": 6771, "precision": 0.396, "recall": 0.805, "positive_rate_true": 0.070}
            },
            "note": "Tested using separate 90th-percentile thresholds per "
                    "province (train only) instead of a single combined "
                    "threshold, since Alberta's 90th percentile is "
                    "approximately 31% higher than British Columbia's on this "
                    "data. F2-score at its own operating point (0.638) is "
                    "effectively tied with the combined-threshold primary "
                    "model (0.637), while trailing on every aggregate metric "
                    "(CV PR-AUC, test ROC-AUC, test PR-AUC). Critically, a "
                    "province-specific threshold does not equalize performance "
                    "across provinces: Alberta still outperforms British "
                    "Columbia on both precision (0.396 vs 0.356) and recall "
                    "(0.805 vs 0.764) even under its own tailored threshold. "
                    "Not adopted as primary: it adds deployment complexity "
                    "without resolving the province-level performance gap it "
                    "was intended to address, indicating the gap reflects a "
                    "genuine difference in learnability between provinces "
                    "rather than an artifact of the shared target definition."
        }
    }
}

with open("/Workspace/Capstone_Group1/models/classifier_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved: classifier_config.json")
print(json.dumps(config, indent=2))