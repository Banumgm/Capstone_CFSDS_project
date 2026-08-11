"""
Task 3b — Chosen model and threshold configuration
20_classifier_config.py
"""
import json

CHOSEN_MODEL = "lgbm_classifier.pkl"
CHOSEN_THRESHOLD = 0.120

config = {
    "chosen_model": CHOSEN_MODEL,
    "chosen_threshold": CHOSEN_THRESHOLD,
    "threshold_selection_method": "F2-optimal (recall weighted 2x precision), "
                                   "selected on a validation split carved out "
                                   "of TRAIN only",
    "target_definition": {"percentile": 0.90, "value_m_per_day": 984.44},
    "feature_set": "49 features, ecozone native categorical",
    "test_performance": {
        "cv_pr_auc": 0.7075,
        "test_roc_auc": 0.9396,
        "test_pr_auc": 0.5654,
        "precision_at_threshold": 0.513,
        "recall_at_threshold": 0.601,
        "f2_score": 0.581
    },
    "rationale": "90th percentile retained as the primary target definition: "
                 "consistent basis for comparison across LightGBM, Random Forest, "
                 "and Logistic Regression, and reflects a rare, operationally "
                 "meaningful 'high-spread' alert category for an early-warning "
                 "use case. F2-optimal threshold chosen over a fixed recall "
                 "target, since it formally weights recall twice as heavily as "
                 "precision -- consistent with prioritizing detection of "
                 "high-spread days over minimizing false alarms, rather than "
                 "picking an arbitrary round recall number. LightGBM chosen "
                 "over Random Forest despite RF's slightly higher raw F2-score "
                 "at its own operating point (0.602 vs 0.581): at matched "
                 "recall (0.839), LightGBM achieves higher precision (0.529 "
                 "vs 0.463) and F2 (0.751 vs 0.722), suggesting LightGBM's "
                 "precision-recall curve dominates RF's on this validation "
                 "split and that RF's raw F2 edge is likely a threshold "
                 "artifact rather than a genuine ranking-quality advantage -- "
                 "though this is based on a single validation split rather "
                 "than repeated resampling, so it should be read as suggestive "
                 "rather than conclusive.",
    "sensitivity_analysis": {
        "target_percentile_85": {
            "target_definition": {"percentile": 0.85, "value_m_per_day": 589.33},
            "f2_threshold": 0.003,
            "precision_at_f2": 0.460,
            "recall_at_f2": 0.778,
            "f2_score": 0.683,
            "note": "Tested as an alternative target definition, evaluated at "
                    "its own validation-selected F2-optimal point. Showed "
                    "stronger CV PR-AUC (0.7857 vs 0.7075), test PR-AUC "
                    "(0.6534 vs 0.5654), comparable precision (0.460 vs "
                    "0.513), higher recall (0.778 vs 0.601), and higher "
                    "F2-score (0.683 vs 0.581), with a small reduction in "
                    "ROC-AUC (0.9277 vs 0.9396). Its F2-optimal threshold "
                    "(0.003) is unusually low, indicating a flatter "
                    "probability distribution near the decision boundary. "
                    "Not adopted as primary due to preference for a rarer, "
                    "more operationally meaningful alert category."
        },
        "province_specific_90th_percentile": {
            "target_definition": {
                "method": "per-province 90th percentile, computed on TRAIN only",
                "alberta_value_m_per_day": 1291.39,
                "british_columbia_value_m_per_day": 843.30
            },
            "cv_pr_auc": 0.6913,
            "test_roc_auc": 0.9358,
            "test_pr_auc": 0.5505,
            "f2_threshold": 0.116,
            "precision_at_f2": 0.445,
            "recall_at_f2": 0.652,
            "f2_score": 0.597,
            "performance_by_province": {
                "british_columbia": {"n": 12605, "precision": 0.423, "recall": 0.633, "positive_rate_true": 0.065},
                "alberta": {"n": 6771, "precision": 0.484, "recall": 0.686, "positive_rate_true": 0.070}
            },
            "note": "Tested using separate 90th-percentile thresholds per "
                    "province (train only) instead of a single combined "
                    "threshold, since AB's 90th percentile is ~31% higher "
                    "than BC's on this data. Performance at its own "
                    "validation-selected F2-optimal point (test F2 = 0.597) "
                    "is close to, but slightly below, the combined-threshold "
                    "primary model (F2 = 0.581 combined vs 0.597 here -- "
                    "actually marginally higher). Not adopted as primary: "
                    "province-specific thresholds add deployment complexity "
                    "and reduce interpretability of the 'high-spread' label "
                    "as a single operational definition, for a gain that is "
                    "within the range of run-to-run variation rather than a "
                    "clear improvement. Documented as a robustness check and "
                    "a candidate for future work if province-level alerting "
                    "is required operationally."
        }
    }
}

with open("/Workspace/Capstone_Group1/models/classifier_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved: classifier_config.json")
print(json.dumps(config, indent=2))