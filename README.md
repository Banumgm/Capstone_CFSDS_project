# Wildfire Spread Prediction — British Columbia & Alberta

**DAMO 699 Capstone Project | Master of Data Analytics**

A dual-model machine learning system predicting daily wildfire spread distance and identifying extreme-spread ("high-spread") days across British Columbia and Alberta, built on the Canadian Forest Service Daily Spread dataset (CFSDS v1.1 beta, 2002–2024). https://osf.io/f48ry/overview

**Team:** Gulbanu Mukhanbetkali · Kyungsun Choi · Rafael Gavidia · Simran Kaur
---
## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Pipeline / How to Reproduce](#pipeline--how-to-reproduce)
- [Dashboard](#dashboard)
- [Key Methodological Decisions & Data Quality Issues](#key-methodological-decisions--data-quality-issues)
- [Limitations](#limitations)
- [Setup](#setup)

---

## Project Overview

Wildfire spread is difficult to predict because the target variable — daily spread distance (`sprdistm`) — is **zero-inflated and heavily right-skewed**: most fire-days show little to no growth, while a small fraction of extreme days drive the majority of burned area. A single regression model optimized for average performance systematically underpredicts exactly the events that matter most operationally.

This project addresses that with a **dual-model architecture**:

1. **Regression model** (LightGBM, Tweedie-motivated target handling) — predicts continuous daily spread distance for every fire-day.
2. **Classification model** (LightGBM) — predicts the probability that a fire-day is a "high-spread" event (≥ 90th percentile of the training distribution, **984.44 m/day**), acting as an early-warning layer the regression model cannot reliably provide on its own.

The two models are evaluated independently, cross-validated using fire-level grouping to prevent leakage, and interpreted using TreeSHAP to identify — and formally test — which categories of predictors (fire weather, topography, fuel/hydrology, human infrastructure, or engineered seasonal features) actually drive predictions.

**Research question:** Do Fire Weather Index (FWI) system indices dominate spread prediction relative to topographic and anthropogenic covariates (Hypothesis 1), or do other factors — fuel continuity, hydrological barriers, seasonal timing — play an equal or larger role?

---

## Dataset

- **Source:** Canadian Forest Service Daily Spread dataset (CFSDS v1.1 beta), OSF
- **Scope:** Fire-day records for British Columbia and Alberta, 2002–2024, filtered via spatial join with provincial boundaries
- **Split strategy:**
  - **Temporal holdout:** train on 2002–2018 (≈18,942 records, 49.4%), test on 2019–2024 (≈19,376 records, 50.6%)
  - **Spatial holdout:** bidirectional BC↔AB train/test, used as an independent robustness check (not the primary split)
  - All splits are performed **at the fire-ID level**, never at the row level, to prevent a single fire's days from leaking across train/test
- **Cross-validation:** `StratifiedGroupKFold`, grouped by fire ID, 5 folds

---

## Methodology

### Regression (continuous spread distance)

| Step | Approach |
|---|---|
| Baseline | Tweedie GLM (`sklearn.TweedieRegressor`), power ≈ 1.1–1.5 — chosen over OLS/log1p because `sprdistm` is zero-inflated + right-skewed, which Tweedie's compound Poisson-Gamma distribution models natively |
| Comparison models | Random Forest, XGBoost (Optuna-tuned) |
| Final model | **LightGBM** (Optuna-tuned, 100 trials, 5-fold GroupKFold CV) |
| Diagnostics | Tweedie deviance residuals, zero-separated evaluation, regional/temporal robustness, spatial holdout (BC↔AB) |

### Classification (high-spread day, binary)

| Step | Approach |
|---|---|
| Target definition | Binary label = 1 if `sprdistm` ≥ 90th-percentile threshold **computed on TRAIN only** (984.44 m/day), applied unchanged to test |
| Imbalance strategy | Class-weighting (`scale_pos_weight` / `class_weight='balanced'`) — **not SMOTE**, because the feature set includes a native categorical (`ecozone`) and cyclical-encoded pairs that SMOTE's nearest-neighbor interpolation isn't suited to |
| Models compared | LightGBM (final), Random Forest, Logistic Regression (interpretability baseline) |
| Threshold selection | F2-optimal (recall weighted 2× precision) — extreme-event misses are treated as operationally more costly than false alarms — selected on a validation split carved out of TRAIN only |
| Fixed operating threshold | **0.212** (used consistently across all evaluations, never recomputed per subgroup or per province) |

### Interpretability

- **TreeSHAP** applied separately to the regression and classification models (N = 2,000 test sample each)
- Features grouped into 5 categories to formally test Hypothesis 1: FWI Indices, Topographic, Anthropogenic, Fuel & Hydrology, Seasonal Dynamics (engineered)

---

## Key Results

### Regression — model comparison (temporal holdout, test set)

| Model | RMSE | MAE |
|---|---|---|
| Tweedie GLM (baseline, power=1.5/alpha=10) | 729.81 | 235.95 |
| Tweedie GLM (tuned, power=1.1/alpha=5) | 725.99 | 223.40 |
| Random Forest | 636.06 | 219.29 |
| **LightGBM (final model)** | **634.31** | **174.57** |
| XGBoost | 656.26 | 179.11 |

**Zero-separated evaluation** (LightGBM): the global metrics above hide a large gap between routine and extreme conditions —

| Subset | N | RMSE | MAE |
|---|---|---|---|
| Zero-spread days | 9,853 | — | 19.85 |
| Extreme events (≥ 90th pct., 984.44 m/day) | 1,272 | 2,388.37 | 1,614.84 |

This confirms the regression model is highly precise on ordinary days but **systematically underpredicts extreme spread** — the core justification for the dual-model architecture.

**Spatial holdout (BC↔AB), regression:**

| Direction | Model | RMSE | MAE |
|---|---|---|---|
| BC → AB | LightGBM | 921.63 | 275.96 |
| AB → BC | LightGBM | 533.45 | 182.66 |
| BC → AB | XGBoost | 969.11 | 291.72 |
| AB → BC | XGBoost | 534.37 | 180.09 |
| BC → AB | Random Forest | 856.57 | 301.26 |
| AB → BC | Random Forest | 617.76 | 306.56 |

Generalization is **asymmetric and replicates across all three model families**: AB-trained models transfer to BC noticeably better than the reverse — attributed to BC's greater ecological heterogeneity (coastal, montane, boreal cordillera) versus Alberta's more homogeneous boreal-prairie transition, rather than any single algorithm's quirk.

### Classification — model comparison (test set)

| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Threshold | Precision | Recall | F2 |
|---|---|---|---|---|---|---|---|
| **LightGBM (final model)** | 0.6674 | 0.9404 | 0.5719 | 0.212 | 0.468 | **0.700** | **0.637** |
| Random Forest | 0.6197 | 0.9284 | 0.5327 | 0.255 | 0.386 | 0.728 | 0.619 |
| Logistic Regression | 0.5225 | 0.8894 | 0.3751 | — | — | — | — |

**Spatial holdout (BC↔AB), classification** — model trained exclusively on one province, tested on the other, fixed threshold 0.212:

| Direction | Recall | Precision | F2 |
|---|---|---|---|
| BC → AB | 0.600 | 0.654 | 0.610 |
| AB → BC | 0.596 | 0.419 | 0.550 |

Recall transfers consistently in both directions (~60%), but precision degrades substantially when transferring AB→BC — the classifier produces more false alarms in that direction. Notably, this spatial asymmetry runs **opposite** to the regression asymmetry (regression: BC→AB is the weaker direction; classification: AB→BC is the weaker direction), meaning the two tasks are affected by spatial heterogeneity in different ways.

### SHAP — Hypothesis 1 (max mean |SHAP| by feature group)

| Group | Regression | Classification |
|---|---|---|
| Seasonal Dynamics (engineered) | **0.408** (`fireday_sin`) | **0.806** (`fireday_cos`) |
| Fuel & Hydrology | 0.332 (`nonfuel1k`) | 0.678 (`hydrodens2k`) |
| FWI Indices | 0.272 (`fwi`) | 0.249 (`fwi`) |
| Anthropogenic | 0.272 (`roaddist`) | 0.254 (`roaddist`) |
| Topographic | 0.176 (`twi`) | 0.316 (`twi`) |

**Conclusion:** H1 (FWI dominance) is only **partially supported** for regression (FWI ties with road distance, both exceeded by seasonal/fuel features) and is **formally rejected** for classification, where extreme events are gated more by seasonal timing and hydrological/fuel barriers than by daily fire weather.

---

## Pipeline / How to Reproduce

All code lives in a single flat `scripts/` folder (not split into `src/phase*` subfolders). Scripts are numbered in execution order and designed to be run sequentially; each reads the outputs of the previous step from `processed/`. Suffixed scripts (`01b`, `12b`, `13b`, etc.) are follow-on diagnostic/sensitivity steps for the script they extend, not a separate phase.

Before running anything, download the raw CFSDS annual CSVs from OSF (https://osf.io/f48ry/overview) and place them under `raw_data/` (not committed to this repository due to size — see `.gitignore`).

```bash
# Phase 1 — Setup & data prep
python scripts/01_filter_bc_ab.py
python scripts/01b_eda.py
python scripts/02_clean.py

# Phase 2 — Feature engineering + regression
python scripts/03_feature_engineering_common.py
python scripts/04_splits.py
python scripts/05_feature_engineering_linear.py
python scripts/06_feature_engineering_tree.py
python scripts/07_baseline_tweedie.py
python scripts/08_rf_regression.py
python scripts/09_final_models_lgb_xgb.py
python scripts/10_spatial_validation.py
python scripts/11_model_comparison_charts.py

# Phase 3 — Classification
python scripts/12_classification_target.py
python scripts/12b_class_weights.py
python scripts/13_lgbm_classifier.py
python scripts/13b_lgbm_threshold_selection.py
python scripts/14_rf_classifier.py
python scripts/14b_rf_threshold_selection.py
python scripts/15_logreg_classifier.py
python scripts/16_error_analysis.py
python scripts/16b_sample_size_analysis.py
python scripts/17_calibration_check.py
python scripts/18_model_comparison.py
python scripts/19_threshold_85pct_sensitivity.py
python scripts/19b_threshold_sensitivity_comparison.py
python scripts/20_classifier_config.py
python scripts/21_threshold_province_sensitivity.py
python scripts/22_province_threshold_sensitivity.py
python scripts/22b_province_threshold_sample_check.py

# Phase 4 — Diagnostics & SHAP
python scripts/25_regression_shap.py
python scripts/26_shap_classification.py
python scripts/27_zero_separated_evaluation.py
python scripts/28_regression_robustness.py
python scripts/28b_tweedie_residuals.py
python scripts/29_spatial_transfer.py
python scripts/30_classification_robustness.py
python scripts/31_shap_visualizations.py
python scripts/32_shap_tp_fn_analysis.py
python scripts/33_classification_spatial_validation.py
python scripts/36_build_dashboard_exports.py

# Dashboard exports (run last, after all models are finalized)
python scripts/36_build_dashboard_exports.py

```

All scripts were originally developed and run on **Databricks** (paths default to `/Workspace/Capstone_Group1/processed`); most fall back to a local `processed/` directory if that path doesn't exist, for local reproduction. Alternatively, run `python run_all.py` to execute the full pipeline in one step.

See [`scripts/decisions_log.md`](scripts/decisions_log.md) for the full record of methodological decisions and their rationale.
---

## Dashboard

A 4-page Power BI dashboard consumes four CSVs produced by the export scripts above:

| Table | Grain | Purpose |
|---|---|---|
| `wildfire_predictions_powerbi.csv` | 1 row / fire-day | Predicted/actual spread, high-spread probability, risk level |
| `wildfire_shap_powerbi.csv` | Top-15 features / fire-day / model | Local SHAP explanation for the selected fire |
| `global_shap_importance_powerbi.csv` | 1 row / feature / model | Unbiased global feature importance (full dataset, not top-N truncated) |
| `fire_conditions_powerbi.csv` | 1 row / fire-day | Environmental (FWI/FFMC/ISI/BUI/temp/humidity/wind), fuel & landscape variables |

**Story flow:** *What is happening? → Where & how fast? → Why? → Can we trust it, and what should we do?*

- **Page 1 — Situation Overview:** KPI cards, risk map, top SHAP drivers for the highest-priority fire, operational alerts
- **Page 2 — How Fast is it Spreading:** global SHAP importance (regression/classification toggle), local SHAP for the selected fire, environmental conditions, fuel & landscape
- **Page 3 — What Drives Fire Spread:** spread trend over time, spread distribution vs. the 984.44 m/day threshold, top-10 fires, seasonal pattern
- **Page 4 — How Reliable are the predictions?:** classification performance (precision/recall/F2), confusion matrix, predicted vs. actual scatter, BC vs. AB comparison, operational workflow

**Modeling notes carried into the data model:**
- `global_shap_importance_powerbi` and `EnvThresholds` (environmental LOW/MODERATE/HIGH cutoffs, p33/p66 on TRAIN) are loaded **without relationships** to any fact table, so Fire ID/date slicers can never accidentally filter them.
- The fixed classification threshold (0.212) and regression label threshold (984.44 m/day) are constants carried from the modeling pipeline — never recalculated inside Power BI or per-province.

---

## Key Methodological Decisions & Data Quality Issues

Several issues were identified independently during EDA/modeling rather than relying solely on provider documentation (CFSDS v1.1 beta):

| Issue | Resolution |
|---|---|
| `firearea` — confirmed target leakage (RF importance 67%, R²=1.0 reconstructs `sprdistm`) | Dropped from all models |
| `prevgrow` — missing-data encoding artifact (0 for 100% of `fireday==1`, conflating "no prior growth" with "not available") | Re-excluded after empirical test degraded CV RMSE; flagged for an `is_first_day` fix in future work |
| FWI indices — severe multicollinearity (VIF up to 1,000–2,000 on delta versions) | Dropped `d_fwi`/`d_isi`/`d_bui` etc.; kept raw `fwi`/`isi`/`bui` (needed for H1) |
| `d_tmax` — unit mismatch (Celsius/Kelvin mix, 2022+ records) | Corrected via offset formula |
| `vpd`/`d_vpd` — scale collapse (2022+, no reliable correction factor) | Excluded from modeling (substitutes: `rh`, `ffmc`) |
| Fire rows spanning train/test | Split by fire ID, not row; verified zero overlap |
| Tweedie log-link divergence on extreme values | Data-driven prediction clipping based on `y_train` scale |

---

## Limitations

- CFSDS v1.1 is a **beta** dataset; some known data quality issues (`d_tmax`, `vpd`) may be resolved in a future official release
- The classification target (90th percentile) is fixed for methodological consistency; an **85th-percentile sensitivity check** showed comparable/slightly better PR-AUC (0.6551 vs. 0.5719) but was not adopted as primary, to preserve a rarer, more operationally meaningful "extreme event" definition
- Regression systematically underpredicts extreme events (MAE 1,614.84 m/day on the top decile) — the model should be used as a **decision-support layer**, not a standalone predictor, particularly for catastrophic-scale events
- Spatial generalization is asymmetric for both tasks, in **opposite directions** — deploying the model in a province not represented in training data warrants caution
- SHAP explains model behavior, not causation — dashboard and report language deliberately avoids "causes" in favor of "model drivers" / "contributing factors"

---

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `optuna`, `shap`, `geopandas`, `matplotlib`, `seaborn`, `joblib`

Raw CFSDS annual CSVs are not committed to this repository (size); download from OSF and place under `data/raw/` before running `01_filter_bc_ab.py`. See `decisions_log.md` for the full record of methodological decisions and their rationale.
