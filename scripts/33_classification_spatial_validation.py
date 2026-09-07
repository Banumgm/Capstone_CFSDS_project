"""
Phase 4, Step 9 — Classification Spatial Holdout (BC↔AB)
33_classification_spatial_validation.py

True cross-provincial spatial holdout for the classifier: trains exclusively 
on one province and tests exclusively on the other, using the same 
train_spatial.csv / test_spatial.csv files as the regression spatial holdout 
(Section 7.3), for direct methodological comparability. Evaluates both 
directions (BC→AB, AB→BC) at the fixed global threshold (0.212) and at a 
locally re-optimized F2-optimal threshold, following the leakage-safe 
validation-split protocol from Section 8.4.3.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score, precision_recall_curve

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

FIXED_THRESHOLD = 0.212
LABEL_CUTOFF = 984.44  # fixed primary threshold defined on the original temporal-training set

print("1. Loading spatial holdout files (train_spatial.csv / test_spatial.csv)...")
train_spatial = pd.read_csv(f"{BASE_DIR}/train_spatial.csv")
test_spatial = pd.read_csv(f"{BASE_DIR}/test_spatial.csv")

# Reuse the same tuned LightGBM hyperparameters as the deployed classifier —
# no re-tuning per split, consistent with threshold-reuse discipline (Section 8.7)
lgb_search = pd.read_csv(f"{BASE_DIR}/lightgbm_optuna_search.csv").iloc[0]
BEST_PARAMS = {
    "num_leaves": int(lgb_search["num_leaves"]),
    "max_depth": int(lgb_search["max_depth"]),
    "learning_rate": float(lgb_search["learning_rate"]),
    "n_estimators": int(lgb_search["n_estimators"]),
    "min_child_samples": int(lgb_search["min_child_samples"]),
}

X_train_reference = pd.read_csv(f"{BASE_DIR}/X_train_tree.csv", nrows=1)
FEATURE_COLS = X_train_reference.columns.tolist()

missing_in_spatial = [c for c in FEATURE_COLS if c not in train_spatial.columns]
if missing_in_spatial:
    raise ValueError(f"train_spatial.csv is missing expected feature columns: {missing_in_spatial}")
def find_f2_optimal_threshold(clf, X_val, y_val):
    """F2-optimal threshold search on a held-out validation split, mirroring 8.4.2/8.4.3."""
    proba_val = clf.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    precision, recall = precision[:-1], recall[:-1]
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-12)
    best_idx = np.nanargmax(f2)
    return thresholds[best_idx]


def run_spatial_direction(train_df, test_df, train_label, test_label):
    print(f"\n{'='*70}\nSPATIAL HOLDOUT: TRAINED ON {train_label} -> TESTED ON {test_label}\n{'='*70}")

    X_train_full = train_df[FEATURE_COLS].copy()
    X_train_full["ecozone"] = X_train_full["ecozone"].astype("category")
    y_train_full = (train_df["sprdistm"] >= LABEL_CUTOFF).astype(int)

    X_test = test_df[FEATURE_COLS].copy()
    X_test["ecozone"] = X_test["ecozone"].astype("category")
    y_test = (test_df["sprdistm"] >= LABEL_CUTOFF).astype(int)

    scale_pos_weight = (y_train_full == 0).sum() / (y_train_full == 1).sum()

    # --- 80/20 fire-grouped validation split, carved from train only (Section 8.4.3) ---
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit_idx, val_idx = next(sgkf.split(X_train_full, y_train_full, groups=train_df["ID"]))
    X_fit, X_val = X_train_full.iloc[fit_idx], X_train_full.iloc[val_idx]
    y_fit, y_val = y_train_full.iloc[fit_idx], y_train_full.iloc[val_idx]

    clf_fit = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1,
                                  scale_pos_weight=scale_pos_weight, **BEST_PARAMS)
    clf_fit.fit(X_fit, y_fit, categorical_feature=["ecozone"])
    local_threshold = find_f2_optimal_threshold(clf_fit, X_val, y_val)

    # --- Final model refit on the full province-train set, evaluated on the other province ---
    clf_final = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1,
                                    scale_pos_weight=scale_pos_weight, **BEST_PARAMS)
    clf_final.fit(X_train_full, y_train_full, categorical_feature=["ecozone"])
    proba_test = clf_final.predict_proba(X_test)[:, 1]

    results = {}
    for label, thresh in [("Fixed (0.212)", FIXED_THRESHOLD),
                           (f"Local F2-optimal ({local_threshold:.3f})", local_threshold)]:
        preds = (proba_test >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
        results[label] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                           "Precision": round(precision, 3), "Recall": round(recall, 3), "F2": round(f2, 3)}
        print(f"\n-- {label} --")
        print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"Precision={precision:.3f}  Recall={recall:.3f}  F2={f2:.3f}")

    return results

print(f"BC rows (train_spatial): {len(train_spatial)}")
print(f"AB rows (test_spatial): {len(test_spatial)}")

# --- Direction 1: BC -> AB (files already match this direction as-is) ---
results_bc_ab = run_spatial_direction(train_spatial, test_spatial, "BC", "AB")

# --- Direction 2: AB -> BC (swap the files, no province filtering needed —
#     each spatial file is already single-province by construction) ---
results_ab_bc = run_spatial_direction(test_spatial, train_spatial, "AB", "BC")


print(f"\n{'='*70}\nSUMMARY: CLASSIFICATION SPATIAL HOLDOUT, BOTH DIRECTIONS\n{'='*70}")
summary_rows = []
for direction, res in [("BC → AB", results_bc_ab), ("AB → BC", results_ab_bc)]:
    for thresh_label, m in res.items():
        summary_rows.append({"Direction": direction, "Threshold": thresh_label, **m})
print(pd.DataFrame(summary_rows).to_string(index=False))