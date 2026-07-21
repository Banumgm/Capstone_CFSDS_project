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


<!-- Add new entries above this line -->
