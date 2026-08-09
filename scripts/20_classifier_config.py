"""
Task 3b — Chosen model and threshold configuration
20_classifier_config.py
"""
import json

CHOSEN_MODEL = "lgbm_classifier.pkl"
CHOSEN_THRESHOLD = 0.027

config = {
    "chosen_model": CHOSEN_MODEL,
    "chosen_threshold": CHOSEN_THRESHOLD,
    "threshold_selection_method": "F2-optimal (recall weighted 2x precision)",
    "target_definition": {"percentile": 0.90, "value_m_per_day": 984.44},
    "feature_set": "49 features, ecozone native categorical",
    "test_performance": {
        "cv_pr_auc": 0.7075,
        "test_roc_auc": 0.9396,
        "test_pr_auc": 0.5654,
        "precision_at_threshold": 0.413,
        "recall_at_threshold": 0.754,
        "f2_score": 0.647
    },
    "rationale": "90th percentile retained as the primary target definition: "
                 "consistent basis for comparison across LightGBM, Random Forest, "
                 "and Logistic Regression, and reflects a genuinely rare, "
                 "operationally meaningful 'high-spread' alert category for an "
                 "early-warning use case. F2-optimal threshold chosen over a "
                 "fixed recall target, since it formally weights recall twice "
                 "as heavily as precision -- consistent with prioritizing "
                 "detection of high-spread days over minimizing false alarms, "
                 "rather than picking an arbitrary round recall number.",
    "sensitivity_analysis": {
        "target_definition": {"percentile": 0.85, "value_m_per_day": 589.33},
        "f2_threshold": 0.001,
        "precision_at_f2": 0.404,
        "recall_at_f2": 0.851,
        "f2_score": 0.697,
        "note": "Tested as an alternative target definition, evaluated at its "
                "own F2-optimal point. Showed stronger CV PR-AUC (0.7857 vs "
                "0.7075), test PR-AUC (0.6534 vs 0.5654), recall (0.851 vs "
                "0.754), and F2-score (0.697 vs 0.647), with a small reduction "
                "in ROC-AUC (0.9277 vs 0.9396) and roughly comparable precision "
                "(0.404 vs 0.413). Its F2-optimal threshold (0.001) is unusually "
                "low, indicating a flatter probability distribution near the "
                "decision boundary. Not adopted as primary due to preference "
                "for a rarer, more operationally meaningful alert category."
    }
}

with open("/Workspace/Capstone_Group1/models/classifier_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved: classifier_config.json")
print(json.dumps(config, indent=2))