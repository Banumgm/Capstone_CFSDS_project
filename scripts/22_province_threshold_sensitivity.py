"""
Task 3a — Per-province threshold sensitivity check
22_province_threshold_sensitivity.py
Quick diagnostic: is the 90th-percentile "high-spread day" threshold
meaningfully different between BC and AB? Computed on TRAIN only,
same convention as the combined threshold, to avoid leakage.
Does NOT change the primary target definition — informational only,
to support the team's discussion on whether province-specific
thresholds are worth pursuing (or should go in Limitations).
"""
import pandas as pd
import os

BASE_DIR = "/Workspace/Capstone_Group1/processed" if os.path.exists("/Workspace") else "processed"

train_df = pd.read_csv(f"{BASE_DIR}/train_temporal.csv")

TARGET = "sprdistm"
PERCENTILE = 0.90

# --- Combined threshold (current primary approach, for reference) ---
combined_threshold = train_df[TARGET].quantile(PERCENTILE)

# --- Per-province thresholds ---
province_thresholds = (
    train_df.groupby("province")[TARGET]
    .quantile(PERCENTILE)
    .rename("threshold_90th")
)

print(f"Combined 90th percentile threshold (train, both provinces): {combined_threshold:,.2f} m/day\n")
print("Per-province 90th percentile thresholds (train only):")
print(province_thresholds.to_string())

# Relative difference vs. combined threshold
pct_diff = ((province_thresholds - combined_threshold) / combined_threshold * 100).round(1)
print("\nDifference from combined threshold:")
for prov, diff in pct_diff.items():
    print(f"  {prov}: {diff:+.1f}%")

# --- How many extra/fewer "high-spread" days would each province get
#     if it used its OWN threshold instead of the combined one? ---
print("\n--- Effect on positive rate if using province-specific thresholds ---")
results = []
for prov in train_df["province"].unique():
    sub = train_df[train_df["province"] == prov][TARGET]
    own_thresh = province_thresholds[prov]
    pos_rate_combined = (sub >= combined_threshold).mean()
    pos_rate_own = (sub >= own_thresh).mean()
    print(f"{prov}: positive rate with combined threshold = {pos_rate_combined:.1%}, "
          f"with own threshold = {pos_rate_own:.1%}")
    results.append({
        "province": prov,
        "threshold_90th": own_thresh,
        "pct_diff_from_combined": pct_diff[prov],
        "positive_rate_combined_threshold": pos_rate_combined,
        "positive_rate_own_threshold": pos_rate_own,
    })

# Save full comparison (not just raw thresholds) for the report / limitations discussion
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(f"{BASE_DIR}/province_threshold_comparison.csv", index=False)
print(f"\nSaved: province_threshold_comparison.csv")