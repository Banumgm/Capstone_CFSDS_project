"""
Phase 4, Step 6 — Classification Robustness by Region & Year
30_classification_robustness.py

Evaluates the stability of the final LightGBM classification model (at the leak-free 
0.212 F2 threshold) across geographical provinces and individual years within the test set.
Outputs Recall, Precision, and raw confusion metrics per subset.
"""
import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"
MODEL_DIR = "/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models"

# 1. Load data for standalone testing
X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_test["ecozone"] = X_test["ecozone"].astype("category") # <--- THE FIX

y_test_clf = pd.read_csv(f"{BASE_DIR}/y_test_clf.csv").iloc[:, 0]
test_raw = pd.read_csv(f"{BASE_DIR}/test_temporal.csv")
lgbm_clf = joblib.load(f"{MODEL_DIR}/lgbm_classifier.pkl")

# 2. Run predictions at 0.212 threshold
OPTIMAL_THRESHOLD = 0.212
y_proba_clf = lgbm_clf.predict_proba(X_test)[:, 1]
y_pred_clf = (y_proba_clf >= OPTIMAL_THRESHOLD).astype(int)

eval_df_clf = pd.DataFrame({
    "y_true": y_test_clf,
    "y_pred": y_pred_clf,
    "province": test_raw["province"].values
})

year_col = next((col for col in ["year", "fire_year", "fireyear", "Year", "FIRE_YEAR"] if col in test_raw.columns), None)
if year_col: 
    eval_df_clf["year"] = test_raw[year_col].values

def calculate_clf_metrics(df):
    tp = ((df["y_true"] == 1) & (df["y_pred"] == 1)).sum()
    fn = ((df["y_true"] == 1) & (df["y_pred"] == 0)).sum()
    fp = ((df["y_true"] == 0) & (df["y_pred"] == 1)).sum()
    tn = ((df["y_true"] == 0) & (df["y_pred"] == 0)).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return len(df), tp, fp, fn, round(recall, 3), round(precision, 3)

print("\n" + "="*75)
print("9.2.3 CLASSIFICATION ROBUSTNESS BY PROVINCE")
print("="*75)
prov_results = []
for prov in eval_df_clf["province"].unique():
    sub = eval_df_clf[eval_df_clf["province"] == prov]
    if len(sub) > 0:
        n, tp, fp, fn, rec, prec = calculate_clf_metrics(sub)
        prov_results.append({
            "Province": prov, "N": n, "True_Pos": tp, 
            "False_Pos": fp, "Missed (FN)": fn, 
            "Recall": rec, "Precision": prec
        })
print(pd.DataFrame(prov_results).to_string(index=False))

if year_col:
    print("\n" + "="*75)
    print("9.2.3 CLASSIFICATION ROBUSTNESS BY YEAR")
    print("="*75)
    year_results = []
    for yr in sorted(eval_df_clf["year"].unique()):
        sub = eval_df_clf[eval_df_clf["year"] == yr]
        if len(sub) > 0:
            n, tp, fp, fn, rec, prec = calculate_clf_metrics(sub)
            year_results.append({
                "Year": yr, "N": n, "True_Pos": tp, 
                "False_Pos": fp, "Missed (FN)": fn, 
                "Recall": rec, "Precision": prec
            })
    print(pd.DataFrame(year_results).to_string(index=False))