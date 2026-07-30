"""
07_baseline_tweedie.py
Combines the hyperparameter search (GroupKFold CV grid search over power/alpha,
grouped by fire ID) with the final baseline-vs-tuned fit. Tuned config is READ
from the search results (no manual re-entry); baseline is the a priori
proposal-specified config (power=1.5, alpha=10), not a search result.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_tweedie_deviance

if os.path.exists("/Workspace"):
    BASE_DIR = "/Workspace/Capstone_Group1/processed"
else:
    BASE_DIR = "processed"

X_train_scaled = pd.read_csv(f"{BASE_DIR}/X_train_linear.csv")
X_test_scaled  = pd.read_csv(f"{BASE_DIR}/X_test_linear.csv")
y_train = pd.read_csv(f"{BASE_DIR}/y_train.csv").iloc[:, 0]
y_test  = pd.read_csv(f"{BASE_DIR}/y_test.csv").iloc[:, 0]

TRAIN_FILE = f"{BASE_DIR}/train_temporal.csv"
train_df_raw = pd.read_csv(TRAIN_FILE)
groups = train_df_raw["ID"].values

LINEAR_PRED_CLIP = np.log(y_train.max() * 3)

# =========================================================
# PART 1: Hyperparameter search (GroupKFold CV grid search)
# =========================================================
param_grid = {"power": [1.1, 1.3, 1.5, 1.7], "alpha": [0.5, 1, 5, 10, 25, 50]}
gkf = GroupKFold(n_splits=5)
search_results = []

for power in param_grid["power"]:
    for alpha in param_grid["alpha"]:
        fold_rmses = []
        for train_idx, val_idx in gkf.split(X_train_scaled, y_train, groups=groups):
            X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = TweedieRegressor(power=power, alpha=alpha, max_iter=5000, tol=1e-6)
            model.fit(X_tr.to_numpy(dtype=np.float64), y_tr)
            X_val_arr = X_val.to_numpy(dtype=np.float64)
            lp = X_val_arr @ model.coef_ + model.intercept_
            preds = np.exp(np.clip(lp, None, LINEAR_PRED_CLIP))
            fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        mean_rmse = np.mean(fold_rmses)
        search_results.append({"power": power, "alpha": alpha, "cv_rmse": mean_rmse})
        print(f"power={power}, alpha={alpha:>4}: CV RMSE = {mean_rmse:,.2f}")

search_df = pd.DataFrame(search_results).sort_values("cv_rmse")
print("\n=== Top 5 hyperparameter combinations ===")
print(search_df.head(5).to_string(index=False))
search_df.to_csv(f"{BASE_DIR}/tweedie_hyperparameter_search.csv", index=False)

# =========================================================
# PART 2: Final baseline-vs-tuned fit
# =========================================================
# Baseline: pre-specified in the proposal (Section 5), not a search result --
# stays as an explicit constant rather than being "read" from anywhere.
PROPOSAL_BASELINE_POWER = 1.5
PROPOSAL_BASELINE_ALPHA = 10.0

# Tuned: read the best combination from the search results above (no manual re-entry)
best_row = search_df.iloc[0]
TUNED_POWER = float(best_row["power"])
TUNED_ALPHA = float(best_row["alpha"])
print(f"\nTuned config read from search results: power={TUNED_POWER}, alpha={TUNED_ALPHA} "
      f"(CV RMSE={best_row['cv_rmse']:,.2f})")

MODEL_CONFIGS = {
    f"baseline (power={PROPOSAL_BASELINE_POWER}, alpha={PROPOSAL_BASELINE_ALPHA:.0f})":
        {"power": PROPOSAL_BASELINE_POWER, "alpha": PROPOSAL_BASELINE_ALPHA},
    f"tuned (power={TUNED_POWER}, alpha={TUNED_ALPHA:.0f})":
        {"power": TUNED_POWER, "alpha": TUNED_ALPHA},
}

def fit_and_evaluate(power, alpha, label):
    model = TweedieRegressor(power=power, alpha=alpha, max_iter=5000, tol=1e-6)
    model.fit(X_train_scaled.to_numpy(dtype=np.float64), y_train)
    X_test_arr = X_test_scaled.to_numpy(dtype=np.float64)
    linear_pred = X_test_arr @ model.coef_ + model.intercept_
    n_clipped = (linear_pred > LINEAR_PRED_CLIP).sum()
    preds = np.exp(np.clip(linear_pred, None, LINEAR_PRED_CLIP))
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    deviance = mean_tweedie_deviance(y_test, preds.clip(min=1e-6), power=power)
    print(f"\n--- {label} ---")
    print(f"Rows clipped: {n_clipped}/{len(y_test)} ({n_clipped/len(y_test):.1%})")
    print(f"RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  Deviance: {deviance:,.4f}")
    return {"label": label, "power": power, "alpha": alpha, "rmse": rmse, "mae": mae,
            "tweedie_deviance": deviance, "model": model}

results = {label: fit_and_evaluate(cfg["power"], cfg["alpha"], label) for label, cfg in MODEL_CONFIGS.items()}

summary = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results.values()])
print("\n=== Baseline vs Tuned summary ===")
print(summary.to_string(index=False))

# NOTE (team review comment): Tweedie deviance is not directly comparable
# across models with different `power` values, since the deviance formula
# itself depends on power. Baseline and tuned use different power values
# and are therefore NOT comparable on the deviance column below -- only
# RMSE/MAE are valid for this comparison. This caveat must also appear
# below the results table in the final report.
print(f"\n[NOTE] Tweedie deviance values above use different `power` settings "
      f"({PROPOSAL_BASELINE_POWER} vs {TUNED_POWER}) and are therefore NOT "
      f"directly comparable to each other. Deviance is only meaningful when "
      f"comparing models that share the same power. Use RMSE/MAE for the "
      f"baseline-vs-tuned comparison.")

summary.to_csv(f"{BASE_DIR}/tweedie_baseline_vs_tuned_final.csv", index=False)

# --- H1 test: use whichever config is labeled "tuned" ---
tuned_label = [k for k in results if k.startswith("tuned")][0]
final_model = results[tuned_label]["model"]
coef_summary = pd.Series(final_model.coef_, index=X_train_scaled.columns)

h1_fwi_vars = ["isi", "bui", "fwi"]
h1_topo_vars = ["slope", "aspect_cos", "aspect_sin", "twi", "dem"]
h1_anthro_vars = ["roaddens2k", "roaddens5k", "roaddens10k", "roaddens25k", "roaddist"]

print("\n=== H1 test: standardized coefficients (tuned model, |coef|) ===")
print("FWI indices:\n", coef_summary[h1_fwi_vars].abs().sort_values(ascending=False))
print("\nTopographic:\n", coef_summary[h1_topo_vars].abs().sort_values(ascending=False))
print("\nAnthropogenic:\n", coef_summary[h1_anthro_vars].abs().sort_values(ascending=False))

print("\nSaved: tweedie_hyperparameter_search.csv, tweedie_baseline_vs_tuned_final.csv")