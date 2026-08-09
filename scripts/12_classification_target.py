"""
Task 3a — Target definition: high-spread vs low-spread classification label
12_classification_target.py
Computes the 90th-percentile threshold for sprdistm on the temporal-split
TRAIN set only, and applies that fixed threshold to both train and test
to create a binary high_spread label. Threshold is never recomputed on
test data, to avoid leakage into the label definition itself.
"""
import pandas as pd

y_train = pd.read_csv("/Workspace/Capstone_Group1/processed/y_train.csv").iloc[:, 0]
y_test  = pd.read_csv("/Workspace/Capstone_Group1/processed/y_test.csv").iloc[:, 0]

THRESHOLD = y_train.quantile(0.90)
print(f"90th percentile threshold (train only): {THRESHOLD:.2f} m/day")

y_train_clf = (y_train >= THRESHOLD).astype(int)
y_test_clf  = (y_test  >= THRESHOLD).astype(int)

print(f"\nTrain class balance:\n{y_train_clf.value_counts(normalize=True)}")
print(f"\nTest class balance:\n{y_test_clf.value_counts(normalize=True)}")

y_train_clf.to_csv("/Workspace/Capstone_Group1/processed/y_train_clf.csv", index=False)
y_test_clf.to_csv("/Workspace/Capstone_Group1/processed/y_test_clf.csv", index=False)

print("\nSaved: y_train_clf.csv, y_test_clf.csv")