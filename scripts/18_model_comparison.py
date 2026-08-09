"""
Task 3b — Classification model comparison table
18_model_comparison.py
Consolidates LightGBM, Random Forest, and Logistic Regression results
for model selection writeup.
"""
import pandas as pd

comparison = pd.DataFrame([
    {
        "model": "LightGBM",
        "cv_pr_auc": 0.7075,
        "test_roc_auc": 0.9396,
        "test_pr_auc": 0.5654,
        "f2_threshold": 0.027,
        "precision_at_f2": 0.413,
        "recall_at_f2": 0.754,
        "f2_score": 0.647,
    },
    {
        "model": "Random Forest",
        "cv_pr_auc": 0.6521,
        "test_roc_auc": 0.9283,
        "test_pr_auc": 0.5332,
        "f2_threshold": 0.214,
        "precision_at_f2": 0.349,
        "recall_at_f2": 0.781,
        "f2_score": 0.626,
    },
    {
        "model": "Logistic Regression",
        "cv_pr_auc": 0.5412,
        "test_roc_auc": 0.8918,
        "test_pr_auc": 0.3810,
        "f2_threshold": None,  # interpretability baseline, not tuned for an operating point
        "precision_at_f2": None,
        "recall_at_f2": None,
        "f2_score": None,
    },
])

print(comparison.to_string(index=False))

comparison.to_csv("/Workspace/Capstone_Group1/processed/classification_model_comparison.csv", index=False)
print("\nSaved: classification_model_comparison.csv")

print("""
Rationale: LightGBM achieved the best CV PR-AUC (0.7075 vs 0.6521), test
ROC-AUC (0.9396 vs 0.9283), and test PR-AUC (0.5654 vs 0.5332) compared to
Random Forest. At each model's own F2-optimal operating point, Random
Forest actually edges out LightGBM on recall (0.781 vs 0.754), but
LightGBM has meaningfully higher precision (0.413 vs 0.349) and a higher
overall F2-score (0.647 vs 0.626) -- so LightGBM remains the better-balanced
choice under the recall-weighted objective this project is optimizing for,
rather than winning on every individual metric.

Logistic Regression trails both tree-based models as expected (a linear
decision boundary cannot capture nonlinear fire-behavior interactions),
but is retained as an interpretability baseline: its standardized
coefficients also support H1 (bui, dmc, and ffmc -- all FWI-related terms
-- outweigh the largest topographic term, aspect_cos), mirroring the role
Tweedie GLM plays for the regression models in Phase 2.

Chosen model: LightGBM, F2-optimal threshold = 0.027.
""")