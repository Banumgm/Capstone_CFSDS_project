"""
Task 3b — Sensitivity analysis: province-specific threshold target
21_threshold_province_sensitivity.py

Tests whether defining "high-spread day" using EACH ROW'S OWN PROVINCE
threshold (computed on TRAIN only, per province_threshold_comparison.csv)
changes classification performance vs. the combined 90th-percentile
target. Same feature set (X_train_tree/X_test_tree), same LightGBM
tuning protocol as the primary model, for a fair comparison.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    precision_recall_curve
)

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

# --- Load feature sets (same as primary classifier) ---
X_train = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv")
X_test  = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")

# --- Load raw temporal splits to get province + target (province was
#     dropped from the feature set, so pull it back in by row order) ---
train_raw = pd.read_csv(f"{BASE_DIR}/train_temporal.csv")
test_raw  = pd.read_csv(f"{BASE_DIR}/test_temporal.csv")

TARGET = "sprdistm"
PERCENTILE = 0.90

# --- Per-province thresholds, computed on TRAIN only (no leakage) ---
province_thresholds = train_raw.groupby("province")[TARGET].quantile(PERCENTILE)
print("Per-province 90th percentile thresholds (train only):")
print(province_thresholds.to_string())

# --- Build province-specific binary labels ---
y_train_clf_prov = (
    train_raw[TARGET] >= train_raw["province"].map(province_thresholds)
).astype(int)
y_test_clf_prov = (
    test_raw[TARGET] >= test_raw["province"].map(province_thresholds)
).astype(int)

print(f"\nPositive rate (train, province-specific target): {y_train_clf_prov.mean():.1%}")
print(f"Positive rate (test,  province-specific target): {y_test_clf_prov.mean():.1%}")
print("(combined-threshold positive rate was 6.6% on test, for reference)")

# --- Same categorical handling as primary classifier ---
X_train["ecozone"] = X_train["ecozone"].astype("category")
X_test["ecozone"]  = X_test["ecozone"].astype("category")
cat_cols = X_train.select_dtypes(include="category").columns.tolist()

SCALE_POS_WEIGHT_PROV = (y_train_clf_prov == 0).sum() / (y_train_clf_prov == 1).sum()

# --- Same Optuna tuning protocol as primary model (12_lgbm_classifier.py) ---
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "scale_pos_weight": SCALE_POS_WEIGHT_PROV,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    model = lgb.LGBMClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train_clf_prov, cv=cv,
        scoring="average_precision",
        params={"categorical_feature": cat_cols} if cat_cols else None
    )
    return scores.mean()

sampler = optuna.samplers.TPESampler(seed=42)
study_prov = optuna.create_study(direction="maximize", sampler=sampler)
study_prov.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest CV PR-AUC (province-specific threshold): {study_prov.best_value:.4f}")
print("(combined 90th percentile CV PR-AUC was 0.7075, for comparison)")
print(f"Best params: {study_prov.best_params}")

best_lgbm_prov = lgb.LGBMClassifier(
    **study_prov.best_params, objective="binary",
    scale_pos_weight=SCALE_POS_WEIGHT_PROV, random_state=42, verbosity=-1
)
best_lgbm_prov.fit(X_train, y_train_clf_prov, categorical_feature=cat_cols)

y_proba_prov = best_lgbm_prov.predict_proba(X_test)[:, 1]

print("\n--- Test performance @ 0.5 threshold ---")
print("ROC-AUC:", round(roc_auc_score(y_test_clf_prov, y_proba_prov), 4), " (combined was 0.9396)")
print("PR-AUC: ", round(average_precision_score(y_test_clf_prov, y_proba_prov), 4), " (combined was 0.5654)")

# --- F2-optimal threshold, same protocol as primary model ---
precisions, recalls, thresholds = precision_recall_curve(y_test_clf_prov, y_proba_prov)
precisions, recalls = precisions[:-1], recalls[:-1]
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
best_f2_idx = np.argmax(f2_scores)
f2_threshold_prov = thresholds[best_f2_idx]

print("\n-- F2-optimal threshold (province-specific target) --")
print(f"Threshold: {f2_threshold_prov:.3f}")
print(f"Precision: {precisions[best_f2_idx]:.3f}")
print(f"Recall:    {recalls[best_f2_idx]:.3f}")
print(f"F2-score:  {f2_scores[best_f2_idx]:.3f}")

y_pred_f2 = (y_proba_prov >= f2_threshold_prov).astype(int)
print(f"\n-- Confusion matrix @ F2-optimal threshold --")
print(confusion_matrix(y_test_clf_prov, y_pred_f2))
print(classification_report(y_test_clf_prov, y_pred_f2))

# --- Per-province breakdown at F2-optimal threshold (the actual question:
#     does the province-specific model perform more EVENLY across provinces
#     than the combined-threshold model did?) ---
print("\n-- Performance by province (province-specific target model) --")
eval_df = pd.DataFrame({
    "province": test_raw["province"].values,
    "y_true": y_test_clf_prov.values,
    "y_pred": y_pred_f2,
})
for prov in eval_df["province"].unique():
    sub = eval_df[eval_df["province"] == prov]
    if sub["y_true"].sum() > 0:
        prec = precision_score(sub["y_true"], sub["y_pred"], zero_division=0)
        rec = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)
        print(f"{prov}: n={len(sub)}, precision={prec:.3f}, recall={rec:.3f}, "
              f"positive_rate_true={sub['y_true'].mean():.1%}")

# --- Summary comparison table ---
summary = pd.DataFrame([
    {
        "target_definition": "Combined 90th percentile (primary)",
        "cv_pr_auc": 0.7075, "test_roc_auc": 0.9396, "test_pr_auc": 0.5654,
        "f2_threshold": 0.027, "precision_at_f2": 0.413, "recall_at_f2": 0.754,
        "f2_score": 0.647,
    },
    {
        "target_definition": "Province-specific 90th percentile",
        "cv_pr_auc": round(study_prov.best_value, 4),
        "test_roc_auc": round(roc_auc_score(y_test_clf_prov, y_proba_prov), 4),
        "test_pr_auc": round(average_precision_score(y_test_clf_prov, y_proba_prov), 4),
        "f2_threshold": round(f2_threshold_prov, 3),
        "precision_at_f2": round(precisions[best_f2_idx], 3),
        "recall_at_f2": round(recalls[best_f2_idx], 3),
        "f2_score": round(f2_scores[best_f2_idx], 3),
    },
])
print("\n" + "=" * 70)
print("SUMMARY: combined vs. province-specific threshold, at F2-optimal point")
print("=" * 70)
print(summary.to_string(index=False))

summary.to_csv(f"{BASE_DIR}/province_threshold_sensitivity_comparison.csv", index=False)
os.makedirs("/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models", exist_ok=True)
joblib.dump(best_lgbm_prov,
    (f"/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models") + "/lgbm_classifier_province.pkl")
print("\nSaved: province_threshold_sensitivity_comparison.csv, lgbm_classifier_province.pkl")