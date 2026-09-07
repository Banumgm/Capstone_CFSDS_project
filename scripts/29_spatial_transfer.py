"""
Phase 4, Step 5 — Classification Error Analysis
29_classification_error_analysis.py

Conducts a detailed error analysis of the binary classification model by isolating
True Positives, False Positives, True Negatives, and False Negatives. Evaluates the
mean feature distributions across these cohorts to identify environmental or spatial
patterns driving model misclassifications at the operational 0.212 F2 threshold.
"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Setup and Load Data/Model (Included for standalone testing)
BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"
MODEL_DIR = "/Workspace/Capstone_Group1/models" if os.path.exists("/Workspace") else "models"

X_test = pd.read_csv(f"{BASE_DIR}/X_test_tree.csv")
X_test["ecozone"] = X_test["ecozone"].astype("category")
lgbm_clf = joblib.load(f"{MODEL_DIR}/lgbm_classifier.pkl")

# 2. Execute Step 5 Diagnostics
print("Loading classification targets...")
y_test_clf = pd.read_csv(f"{BASE_DIR}/y_test_clf.csv").iloc[:, 0]
OPTIMAL_THRESHOLD = 0.212

y_proba_clf = lgbm_clf.predict_proba(X_test)[:, 1]
y_pred_clf = (y_proba_clf >= OPTIMAL_THRESHOLD).astype(int)

error_df = X_test.copy()
error_df["y_true"] = y_test_clf
error_df["y_pred"] = y_pred_clf

conditions = [
    (error_df['y_true'] == 1) & (error_df['y_pred'] == 1),
    (error_df['y_true'] == 0) & (error_df['y_pred'] == 1),
    (error_df['y_true'] == 0) & (error_df['y_pred'] == 0),
    (error_df['y_true'] == 1) & (error_df['y_pred'] == 0)
]
choices = ['1_True_Positive', '2_False_Positive', '3_True_Negative', '4_False_Negative']
error_df['Prediction_Cohort'] = np.select(conditions, choices, default='Unknown')

print("\n" + "="*70)
print("9.2.2 CLASSIFICATION ERROR ANALYSIS - COHORT DISTRIBUTIONS")
print("="*70)

print("Cohort Counts:")
print(error_df['Prediction_Cohort'].value_counts().sort_index().to_string())

analysis_features = ['fwi', 'twi', 'roaddist', 'hydrodens2k', 'nonfuel1k']
print("\nMean Feature Values by Cohort:")
print(error_df.groupby('Prediction_Cohort')[analysis_features].mean().round(3).to_string())

# 3. Generate Confusion Matrix Heatmap
print("\nGenerating Confusion Matrix Heatmap...")
cm = confusion_matrix(error_df['y_true'], error_df['y_pred'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low-Spread (0)', 'High-Spread (1)'], 
            yticklabels=['Low-Spread (0)', 'High-Spread (1)'],
            cbar_kws={'label': 'Number of Observations'})
plt.title(f'Classification Confusion Matrix\n(Threshold = {OPTIMAL_THRESHOLD})', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()