"""
01_filter_bc_ab.py

Purpose
-------
Merge all annual CFSDS Firegrowth datasets (2002–2024),
retain only fires located in British Columbia and Alberta
using a spatial join with provincial boundaries, and save
the filtered dataset.

Input
-----
raw_data/
    Firegrowth_groups_v1_1_2002.csv
 ...
    Firegrowth_groups_v1_1_2024.csv

province_boundaries/
    lpr_000b21a_e.shp
    ...

Output
------
processed/cfsds_bc_ab_2002_2024.csv

Requirements
------------
pip install pandas geopandas shapely pyogrio
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("processed")
BOUNDARY_FILE = Path("raw_data/canada_provinces") / "lpr_000b21a_e.shp"

OUTPUT_FILE = OUTPUT_DIR / "cfsds_bc_ab_2002_2024.csv"

TARGET_PROVINCES = [
    "British Columbia",
    "Alberta",
]

LAT_COLUMNS = [
    "LATITUDE",
    "Latitude",
    "latitude",
    "LAT",
    "lat",
    "Y",
]

LON_COLUMNS = [
    "LONGITUDE",
    "Longitude",
    "longitude",
    "LONG",
    "Lon",
    "lon",
    "X",
]

PROVINCE_COLUMNS = [
    "PRENAME",
    "PRENAMEEN",
    "PRNAME",
    "NAME",
    "name",
]


# =============================================================================
# Helper functions
# =============================================================================

def find_column(columns, candidates, description):
    """Return the first matching column name."""

    for col in candidates:
        if col in columns:
            return col

    raise ValueError(
        f"Unable to locate {description} column.\n"
        f"Available columns:\n{list(columns)}"
    )


def load_fire_data():
    """Load and merge all annual wildfire datasets."""

    files = sorted(RAW_DATA_DIR.glob("Firegrowth_groups_v1_1_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No Firegrowth_groups_v1_1_*.csv files found in {RAW_DATA_DIR}"
        )

    print(f"Found {len(files)} yearly datasets.\n")

    frames = []

    for file in files:
        print(f"Loading {file.name}")

        df = pd.read_csv(file, low_memory=False)

        df["source_file"] = file.name

        try:
            df["year"] = int(file.stem[-4:])
        except ValueError:
            pass

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nCombined records: {len(combined):,}")

    return combined


def create_geodataframe(df):
    """Convert DataFrame to GeoDataFrame."""

    lat_col = find_column(df.columns, LAT_COLUMNS, "latitude")
    lon_col = find_column(df.columns, LON_COLUMNS, "longitude")

    print(f"Latitude column : {lat_col}")
    print(f"Longitude column: {lon_col}")

    df = df.dropna(subset=[lat_col, lon_col])

    df = df[
        df[lat_col].between(-90, 90)
        & df[lon_col].between(-180, 180)
    ].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df[lon_col],
            df[lat_col],
        ),
        crs="EPSG:4269",
    )

    return gdf


def load_provinces(target_crs):
    """Load provincial boundaries."""

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found:\n{BOUNDARY_FILE}"
        )

    provinces = gpd.read_file(BOUNDARY_FILE)

    if provinces.crs != target_crs:
        provinces = provinces.to_crs(target_crs)

    province_col = find_column(
        provinces.columns,
        PROVINCE_COLUMNS,
        "province name",
    )

    provinces = provinces[
        provinces[province_col].isin(TARGET_PROVINCES)
    ].copy()

    provinces = provinces.rename(
        columns={province_col: "province"}
    )

    return provinces


# =============================================================================
# Main
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    fires = load_fire_data()

    fires_gdf = create_geodataframe(fires)

    provinces = load_provinces(fires_gdf.crs)

    print("\nRunning spatial join...")

    filtered = gpd.sjoin(
        fires_gdf,
        provinces[["province", "geometry"]],
        predicate="within",
        how="inner",
    )

    filtered = filtered.drop(
        columns=[
            "geometry",
            "index_right",
        ],
        errors="ignore",
    )

    filtered.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nDone.")
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Final records: {len(filtered):,}")

    print("\nProvince counts:")
    print(filtered["province"].value_counts())


if __name__ == "__main__":
    main()