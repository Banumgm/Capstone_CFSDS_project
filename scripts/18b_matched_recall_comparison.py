"""
Task 3b — Matched-recall comparison: LightGBM vs Random Forest
18b_matched_recall_comparison.py

Compares LightGBM and Random Forest precision at the SAME recall level
(the recall RF achieves at its own F2-optimal point), using validation
data only (same 80/20 split carved from TRAIN as 13b/14b). This checks
whether LightGBM's better ranking (higher PR-AUC) also means better
precision at RF's operating recall -- i.e. whether RF's F2 win is a real
model-quality edge or just an artifact of where its threshold happened
to land on its own PR curve.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

X_train = pd.read_csv("/Workspace/Capstone_Group1/processed/X_train_tree.csv")
y_train_clf = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv").iloc[:, 0]

X_train["ecozone"] = X_train["ecozone"].astype("category")
cat_cols = X_train.select_dtypes(include="category").columns.tolist()

# --- LightGBM: same train/val split as 13b ---
X_fit_lgbm, X_val_lgbm, y_fit_lgbm, y_val_lgbm = train_test_split(
    X_train, y_train_clf, test_size=0.2, stratify=y_train_clf, random_state=42
)
LGBM_BEST_PARAMS = {
    "num_leaves": 73, "max_depth": 10, "learning_rate": 0.06155574273677577,
    "n_estimators": 495, "min_child_samples": 74
}
scale_pos_weight_lgbm = (y_fit_lgbm == 0).sum() / (y_fit_lgbm == 1).sum()
lgbm_val_model = lgb.LGBMClassifier(
    **LGBM_BEST_PARAMS, objective="binary",
    scale_pos_weight=scale_pos_weight_lgbm, random_state=42, verbosity=-1
)
lgbm_val_model.fit(X_fit_lgbm, y_fit_lgbm, categorical_feature=cat_cols)
y_proba_val_lgbm = lgbm_val_model.predict_proba(X_val_lgbm)[:, 1]

# --- Random Forest: same train/val split as 14b (one-hot encoded) ---
X_train_rf = pd.get_dummies(X_train, columns=["ecozone"], drop_first=True)
X_fit_rf, X_val_rf, y_fit_rf, y_val_rf = train_test_split(
    X_train_rf, y_train_clf, test_size=0.2, stratify=y_train_clf, random_state=42
)
RF_BEST_PARAMS = {"n_estimators": 543, "max_depth": 20, "min_samples_leaf": 5}
rf_val_model = RandomForestClassifier(
    **RF_BEST_PARAMS, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_val_model.fit(X_fit_rf, y_fit_rf)
y_proba_val_rf = rf_val_model.predict_proba(X_val_rf)[:, 1]

# --- PR curves on validation for both models ---
precisions_lgbm, recalls_lgbm, thresholds_lgbm = precision_recall_curve(y_val_lgbm, y_proba_val_lgbm)
precisions_lgbm, recalls_lgbm = precisions_lgbm[:-1], recalls_lgbm[:-1]

precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_val_rf, y_proba_val_rf)
precisions_rf, recalls_rf = precisions_rf[:-1], recalls_rf[:-1]

# --- RF's own F2-optimal recall (the target we're matching LightGBM to) ---
f2_rf = (5 * precisions_rf * recalls_rf) / (4 * precisions_rf + recalls_rf + 1e-10)
rf_f2_idx = np.argmax(f2_rf)
target_recall = recalls_rf[rf_f2_idx]

print(f"RF's own F2-optimal point (validation): recall={target_recall:.3f}, "
      f"precision={precisions_rf[rf_f2_idx]:.3f}, F2={f2_rf[rf_f2_idx]:.3f}")

# --- Find LightGBM's precision at that same (or nearest-above) recall level ---
idx_match = np.where(recalls_lgbm >= target_recall)[0]
if len(idx_match) > 0:
    # among points achieving at least target_recall, take the one with highest precision
    best_match_idx = idx_match[np.argmax(precisions_lgbm[idx_match])]
    matched_threshold = thresholds_lgbm[best_match_idx]
    matched_precision = precisions_lgbm[best_match_idx]
    matched_recall = recalls_lgbm[best_match_idx]
    matched_f2 = (5 * matched_precision * matched_recall) / (4 * matched_precision + matched_recall + 1e-10)

    print(f"\nLightGBM at matched recall (validation):")
    print(f"Threshold: {matched_threshold:.3f}")
    print(f"Precision: {matched_precision:.3f}")
    print(f"Recall:    {matched_recall:.3f}")
    print(f"F2:        {matched_f2:.3f}")

    print(f"\n--- Comparison at recall ~= {target_recall:.3f} ---")
    print(f"{'Model':<15}{'Precision':<12}{'Recall':<12}{'F2':<10}")
    print(f"{'LightGBM':<15}{matched_precision:<12.3f}{matched_recall:<12.3f}{matched_f2:<10.3f}")
    print(f"{'Random Forest':<15}{precisions_rf[rf_f2_idx]:<12.3f}{target_recall:<12.3f}{f2_rf[rf_f2_idx]:<10.3f}")

    if matched_precision > precisions_rf[rf_f2_idx]:
        print("\n=> At matched recall, LightGBM has higher precision (and F2) than RF's "
              "F2-optimal point -- RF's earlier F2 'win' was a threshold artifact, "
              "not a real ranking-quality advantage. LightGBM is the stronger model.")
    else:
        print("\n=> At matched recall, RF still has higher (or equal) precision than "
              "LightGBM -- RF's F2 advantage reflects a genuine ranking-quality edge "
              "at this operating point, not just where the threshold landed.")
else:
    print(f"\nLightGBM cannot reach recall >= {target_recall:.3f} on this validation split.")