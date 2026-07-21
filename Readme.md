# Capstone: Wildfire Spread Prediction (BC & AB)

## Setup
pip install -r requirements.txt

## Data
Raw CFSDS files are NOT in this repo (too large). Download from OSF
(https://doi.org/10.17605/OSF.IO/F48RY) and place `Firegrowth_groups_v1_1_*.csv`
files in `raw_data/`, or grab them from our shared Google Drive folder: [ссылка]

## Pipeline
- python scripts/01_filter_bc_ab.py
- python scripts/02_clean.py
- python scripts/01b_eda.py
