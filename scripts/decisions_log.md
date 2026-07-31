# Decisions Log — Wildfire Spread Prediction (BC & AB)

Running record of methodological decisions made during the project, with
short justifications. Add a new entry whenever the team makes a choice
that a reader of the final report would otherwise have to guess at.
Keep entries short — one paragraph max. Newest entries at the top.

---
## Data Collection / Scope

**Downloaded and included the full 2002–2024 period** (23 yearly
files). Dataset totals 122,851 records across Canada; after spatial filtering
to British Columbia and Alberta, this yields 38,318 fire-day records
(British Columbia: 24,724, 64.5%; Alberta: 13,594, 35.5%), split
almost evenly across the intended temporal holdout — 18,942 records
(49.4%) in 2002–2018, 19,376 records (50.6%) in 2019–2024.

**Restricted analysis to British Columbia and Alberta**, filtered via
spatial join against Statistics Canada provincial boundary files (with an
ecozone-code fallback: 12/13/14 for BC, 9/10 for AB, if no boundary file
is available). Rationale: both provinces had some of the largest and most
economically damaging fires in the CFSDS period; both were included in
the original CFSDS agency-perimeter validation (Sørensen-Dice 0.72-0.73);
and together they represent contrasting fire regimes (coastal/mountainous
vs. boreal-prairie transition), which motivates the spatial holdout
(train on one province, test on the other).

**Used the fire-day aggregated table (`Firegrowth_groups`), not the
pixel-level table (`Firegrowth_pts`) or the DOY rasters.** The research
question is defined at the level of a single fire's daily spread event;
the aggregated unit also reduces spatial autocorrelation among adjacent
pixels within the same fire-day. Rasters and pixel-level data were not
downloaded at all — not needed for this scope.

## Data Cleaning

**Used `peatprop` instead of deriving a feature from `peattype`.**
`peattype` is a 9-class land-cover classification (Pontone et al., 2023),
where only codes 1-4 are true peatland types (Bog, Rich Fen, Poor Fen,
Peatland Permafrost Complex); codes 5-9 are other valid land-cover classes
(Mineral Wetlands, Water, Uplands, Agriculture, Urban) — NOT peatland. A
missing value in `peattype` does not mean "non-peatland." `peatprop`
(proportion of the burn day's area in classes 1-4 combined) is the
correct, continuous, near-complete feature for peatland influence
(0.00% missing on both the partial and full dataset; mean = 0.156 on the
full 2002–2024 dataset), so it was used directly instead of a flag
derived from `peattype` presence/absence.

**Dropped `peattype` (55.1% missing on the full dataset) rather than
imputing it.** Given the missingness level and that `peatprop` already
captures the relevant peatland signal, imputing a 9-class categorical
with over half its values missing was judged not worth the added
noise/complexity.

**Excluded `sprdistm` (target), `lat`, `lon`, and `year` from median
imputation.** Imputing coordinates with a median produces a physically
meaningless "average location" and could silently corrupt downstream
spatial logic. The target is dropped when missing, not imputed. `year`
is a temporal identifier, not a continuous quantity to average.

**Used Spearman correlation (not Pearson) as the primary method for
checking FWI index multicollinearity.** FWI System indices are related
through a hierarchical, non-linear formula (e.g., ISI and BUI combine
into FWI), so a rank-based (monotonic) correlation measure is more
appropriate than a purely linear one. Pearson is still reported alongside
for comparison since it's the conventional basis for linear-regression
diagnostics (e.g., VIF). On the full 2002–2024 dataset, the strongest
pairs remain DMC–BUI (Spearman r = 0.99) and FFMC–ISI (r = 0.98),
consistent with the partial-dataset result — this will be addressed via
variable selection or regularization (e.g., Ridge) in the linear
baseline model.

**Missing-column drop threshold set at 40%.** Columns missing more than
this fraction were judged too sparse to impute reliably without
introducing bias; columns below this threshold still carry enough real
signal to be worth keeping and imputing (median, for numeric columns).

**Categorical columns (`ecozone`) kept as pandas `category` dtype in
cleaning, not one-hot/ordinal encoded.** Actual encoding for modeling is
deferred to `03_feature_engineering.py`, so the cleaning step stays
reversible and doesn't inflate the file with dummy columns before it's
clear which model needs which encoding scheme.
---

## FOR Feature Engineering
**Exclude `firearea`, `cumuarea`, and `pctgrowth` from the model feature
set.** These variables are derived from the same circular-growth
approximation, for the same burn day, as the target variable `sprdistm`
(see Barber et al., 2024, Eq. 1) — they are not independent environmental
predictors but alternate representations of the same measured spread
event. `firearea` shows the strongest raw correlation with the target
(r = 0.645 on the full 2002–2024 dataset) precisely because of this
shared derivation, not because it reflects weather, fuel, or topography.
Including it would let the model largely "predict" `sprdistm` from a
mathematically related quantity rather than from genuine environmental
covariates, undermining the test of H1 and the practical value of the
model (at prediction time, tomorrow's `firearea` is not actually known in
advance either). `prevgrow` (previous day's growth) was kept, since using
a lagged value of a related variable to predict today's spread is a
legitimate autoregressive predictor, not same-event leakage.

## Phase 2 — Feature Engineering & Regression Models

**Split at the fire (ID) level, not row level, for both temporal and
spatial holdouts.** A single fire's daily rows could otherwise span both
train and test, leaking information about that fire's behavior across the
split. Verified zero fire-ID overlap after splitting (assert checks). 10
fires spanning both provinces excluded from the spatial split only
(retained in the temporal split, which doesn't depend on province).

**`firearea` identified as target leakage and excluded from all models.**
Discovered via Random Forest feature importance (67% importance on the
first run, an implausibly high concentration). Combined with `cumuarea`,
`firearea` exactly reconstructs `sprdistm` via Eq. 1 (Barber et al., 2024)
— R² = 1.0000 on reconstruction. This is a stronger and more direct form
of leakage than `cumuarea`/`pctgrowth` (already excluded per the earlier
Phase 1 decision); all models were re-trained and re-tuned after removal.

**FWI delta columns (`d_fwi`, `d_isi`, `d_ffmc`, `d_dmc`, `d_dc`, `d_bui`)
dropped for severe multicollinearity; raw FWI indices retained for H1.**
VIF on the raw FWI family reached 1,000–2,000 with the delta columns
included. Dropping only the deltas reduced VIF to 40–80 for `fwi`/`isi`/
`bui`, with no measurable loss in CV performance (721.40 → 722.27) —
indicating the deltas carried little independent signal. Raw `fwi`,
`isi`, `bui` were kept despite still-elevated VIF because H1 specifically
requires testing their individual standardized coefficients / feature
importance against topographic and anthropogenic variables; dropping
them would make H1 untestable.

**`prevgrow` re-tested and re-excluded, reversing the earlier Phase 1
"keep" decision.** The original rationale (legitimate lagged
autoregressive predictor, not same-event leakage) still holds
theoretically, and VIF confirms no multicollinearity (VIF = 1.09).
However, empirically restoring `prevgrow` degraded Tweedie CV RMSE
(721 → 793, even after re-tuning power/alpha). Root cause: `prevgrow` = 0
for 100% of `fireday==1` records (no true "previous day" exists on a
fire's first day), conflating "genuinely no prior growth" with "value
unavailable" under the same encoded value — a missing-data encoding
artifact, not leakage or collinearity. Re-excluded from the final feature
set. A cleaner fix (e.g., an `is_first_day` flag, or leaving `prevgrow`
as NaN for `fireday==1`) was not pursued given time constraints, but is
noted for a future iteration.

**`d_tmax` unit-mismatch bug (Celsius/Kelvin) confirmed and corrected.**
Records from 2022 onward showed `d_tmax` values around +273 in magnitude
(e.g., 277 instead of ~2), consistent with a previous-day `tmax` lookup
stored in Kelvin rather than Celsius. Verified by reconstructing the
implied "previous day tmax" (clustered at 254–322, a plausible Kelvin
range) and confirming that subtracting 273.15 yields a plausible Celsius
range and a corrected `d_tmax` matching the historical (pre-2022)
distribution. Corrected via offset formula rather than dropped, since the
underlying temperature-change signal is real and useful once fixed.

**`vpd`/`d_vpd` excluded due to an unresolved scale/unit mismatch, not a
simple additive bug like `d_tmax`.** From 2022 onward, `vpd` drops from a
historical mean of ~16 to ~1.3–1.6 (a ~10–15x scale reduction), while its
correlation structure with `tmax`/`rh` stays consistent — suggesting a
unit or formula change in the data pipeline rather than a genuine climate
shift. Unlike `d_tmax`, no single correction factor could be confirmed
(candidates tested: 7–15x, best fit ~13x, not a standard unit-conversion
constant), and VPD is a nonlinear function of temperature, so a constant
offset cannot recover it. Excluded from modeling; `rh` and `ffmc` serve
as substitute atmospheric-dryness signals. Both this issue and the
`d_tmax` bug trace to the CFSDS v1.1 beta release (June 12, 2025), which
first added 2022–2024 data and is self-reported by the provider as beta
with known errors — not documented in the original CFSDS methodology
paper (Barber et al., 2024), which covers only 2002–2021.

**Tiered model comparison: Tweedie GLM (baseline) → Random Forest →
LightGBM/XGBoost (Optuna-tuned).** Tweedie chosen over plain OLS or
polynomial regression given `sprdistm`'s zero-inflated (median = 0),
right-skewed (max ~20,000m) distribution. All hyperparameter searches
(GroupKFold grid search for Tweedie, RandomizedSearchCV for RF, Optuna
for LightGBM/XGBoost) grouped CV folds by fire ID to prevent the same
fire-level leakage addressed in the train/test split.

**Final model: LightGBM (Optuna-tuned), RMSE 634.31, MAE 174.57 on
temporal holdout.** Selected over XGBoost after repeated tuning runs
showed XGBoost's test RMSE varying considerably (629–656) despite similar
or better CV RMSE, indicating less stable tuned configurations at this
data scale; LightGBM's test RMSE stayed consistently near 634 across
multiple independent runs. Random Forest (RMSE 636.06) performed close to
LightGBM/XGBoost on RMSE but noticeably worse on MAE (219 vs. 175),
indicating comparatively larger errors on the many low/zero-spread days
that dominate the target's distribution.

**Spatial holdout (BC↔AB) reveals asymmetric generalization, replicated
across all three tree-based models (LightGBM, XGBoost, Random Forest).**
Models trained on BC generalize substantially worse to Alberta
(RMSE 857–969) than the reverse, models trained on Alberta generalizing
to BC (RMSE 521–618). Consistent with BC's greater ecological
heterogeneity (coastal, montane, boreal cordillera ecozones) vs.
Alberta's more homogeneous boreal-prairie transition: a model trained on
BC's varied conditions appears to rely on region-specific composite
indices that don't transfer, while a model trained on Alberta's simpler
regime learns more broadly applicable rules. A geographic error map
(color scale fixed at 1,500m for fair comparison) shows BC-trained model
errors clustering in Alberta's northern boreal-plains transition zone.

**H1 (FWI indices more important than topographic/anthropogenic
variables) — preliminary, model-dependent result; formal test deferred
to Phase 4 (SHAP, per Task Plan PR #24).** Random Forest, XGBoost, and
LightGBM feature importance all show FWI indices' max importance
exceeding topographic/anthropogenic maxima. Tweedie GLM's standardized
coefficients show the opposite (`aspect_cos` largest). This 3-of-4-model
pattern should not be treated as conclusive — the discrepancy across
model families is itself a notable finding to explore further in Phase 4.

**Reproducibility: hyperparameter search results vary slightly across
independent runs (different machines, or the same machine at different
times) despite fixed random seeds.** Random Forest/LightGBM/XGBoost use
parallel processing (`n_jobs=-1`); floating-point operation order can
differ slightly by hardware/thread count, and for Optuna specifically,
this compounds because the TPE sampler chooses each new trial based on
all previous trials' results. Differences are small (a few RMSE points,
e.g., 634 vs. 643) and do not change the core conclusions (LightGBM/
XGBoost outperform Tweedie/RF; the BC↔AB asymmetric generalization
pattern), but the exact "best" hyperparameters and third-decimal metrics
can shift. The Databricks run is the source of truth for all numbers
reported in the final report; local (VS Code) runs are for verifying the
pipeline executes correctly, not for generating new official results.


<!-- Add new entries above this line -->
