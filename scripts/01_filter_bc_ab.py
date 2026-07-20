"""
01_filter_bc_ab.py

Purpose:
    Load all yearly CFSDS "Firegrowth_groups" CSVs from raw_data/,
    combine them, and filter down to fires located in British Columbia
    and Alberta only. Saves a single, lightweight processed CSV that
    can safely be committed to GitHub.

Expected input:
    raw_data/
        Firegrowth_groups_v1_1_2020.csv
        Firegrowth_groups_v1_1_2021.csv
        Firegrowth_groups_v1_1_2022.csv
        Firegrowth_groups_v1_1_2023.csv
        Firegrowth_groups_v1_1_2024.csv

Output:
    processed/cfsds_bc_ab_2020_2024.csv

Requirements:
    pip install pandas geopandas shapely --break-system-packages
"""

import glob
import os
import sys

import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# CONFIG — adjust these paths/names if your setup differs
# ---------------------------------------------------------------------------
RAW_DATA_DIR = "raw_data"
FILE_PATTERN = "Firegrowth_groups_v1_1_*.csv"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cfsds_bc_ab_2020_2024.csv")

# Path to a provincial boundary shapefile/geojson (Statistics Canada
# cartographic boundary file, or any equivalent). Download once from:
# https://www150.statcan.gc.ca/n1/en/catalogue/92-160-X
# and point this at the .shp (or .geojson) file.
PROVINCE_BOUNDARY_PATH = "province_boundaries/lpr_000b21a_e.shp"

TARGET_PROVINCES = ["British Columbia", "Alberta"]

# Candidate column names — CFSDS column naming can vary slightly between
# versions, so we check a few likely options automatically.
LAT_CANDIDATES = ["lat", "latitude", "Latitude", "Lat", "y", "Y"]
LON_CANDIDATES = ["lon", "long", "longitude", "Longitude", "Lon", "x", "X"]

# Fallback: known ecozone codes for BC / AB (from Barber et al. 2024, Table 2)
# Used only as a sanity check / fallback if no boundary file is available.
BC_ECOZONES = [12, 13, 14]   # Boreal Cordillera, Pacific Maritime, Montane Cordillera
AB_ECOZONES = [9, 10]        # Boreal Plains, Prairies


def find_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a {label} column. Tried: {candidates}. "
        f"Actual columns are: {list(df.columns)}"
    )


def load_all_years(raw_dir, pattern):
    paths = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not paths:
        sys.exit(
            f"No files found matching {pattern} in {raw_dir}/. "
            f"Check RAW_DATA_DIR and FILE_PATTERN."
        )
    print(f"Found {len(paths)} yearly files:")
    for p in paths:
        print(f"  - {p}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # Track source year/file for traceability
        df["source_file"] = os.path.basename(p)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape (all years, all provinces): {combined.shape}")
    return combined


def filter_by_boundary(df, lat_col, lon_col):
    """Spatial join against a provincial boundary shapefile."""
    if not os.path.exists(PROVINCE_BOUNDARY_PATH):
        print(
            f"\n[WARNING] Boundary file not found at {PROVINCE_BOUNDARY_PATH}.\n"
            f"Falling back to ecozone-based filtering instead (less precise "
            f"at province edges, but requires no extra download)."
        )
        return None

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4269",  # CFSDS coordinate system
    )

    provinces = gpd.read_file(PROVINCE_BOUNDARY_PATH).to_crs("EPSG:4269")

    # Statistics Canada boundary files typically use "PRENAME" for province
    # name in English. Adjust if your shapefile uses a different field.
    name_col = "PRENAME" if "PRENAME" in provinces.columns else provinces.columns[0]

    provinces = provinces[provinces[name_col].isin(TARGET_PROVINCES)]

    joined = gpd.sjoin(gdf, provinces, predicate="within", how="inner")
    joined = joined.rename(columns={name_col: "province"})
    return pd.DataFrame(joined.drop(columns="geometry"))


def filter_by_ecozone(df):
    """Fallback filter using the ecozone column already present in CFSDS."""
    if "ecozone" not in df.columns:
        sys.exit(
            "No boundary file available AND no 'ecozone' column found — "
            "cannot filter to BC/AB. Please provide a boundary shapefile."
        )
    bc = df[df["ecozone"].isin(BC_ECOZONES)].copy()
    ab = df[df["ecozone"].isin(AB_ECOZONES)].copy()
    bc["province"] = "British Columbia (approx., via ecozone)"
    ab["province"] = "Alberta (approx., via ecozone)"
    return pd.concat([bc, ab], ignore_index=True)


def main():
    combined = load_all_years(RAW_DATA_DIR, FILE_PATTERN)

    lat_col = find_column(combined, LAT_CANDIDATES, "latitude")
    lon_col = find_column(combined, LON_CANDIDATES, "longitude")
    print(f"\nUsing lat column: '{lat_col}', lon column: '{lon_col}'")

    filtered = filter_by_boundary(combined, lat_col, lon_col)
    if filtered is None:
        filtered = filter_by_ecozone(combined)

    print(f"\nFiltered shape (BC + AB only): {filtered.shape}")
    print(filtered["province"].value_counts())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved processed subset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()