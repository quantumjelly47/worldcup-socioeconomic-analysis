from pathlib import Path

import pandas as pd

from scripts.utils.filter_countries import check_country_coverage
from scripts.utils.socio_helpers import (
    BASE_DIR,
    filter_wc,
    interpolate_by_group,
    load_csv,
    save_world_cup,
)

OUTPUT_DIR = BASE_DIR / "data" / "created_datasets" / "socioeconomic"


def clean_gdp(path="data/raw_data_files/Socioeconomic Data/GDP per Capita Data.csv", verbose=True):
    """
    Load, clean, and subset the GDP per capita dataset (1960–2024), compute a normalized metric,
    and produce World Cup vs. non–World Cup splits.

    Args:
        path (str): Path to the raw GDP CSV.
        verbose (bool): Whether to print summary stats and coverage.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (world_cup_gdp, non_world_cup_gdp, full_long)
    """

    # Load dataset
    gdp = load_csv(path, skiprows=4)

    # Keep only relevant columns
    year_cols = [c for c in gdp.columns if c.isdigit()]
    gdp = gdp[["Country Name", "Country Code"] + year_cols]

    # Reshape from wide to long format
    gdp_long = gdp.melt(
        id_vars=["Country Name", "Country Code"],
        var_name="year",
        value_name="gdp_per_capita"
    )

    # Standardize column names
    gdp_long.rename(columns={
        "Country Name": "country",
        "Country Code": "iso3"
    }, inplace=True)

    # Convert data types
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_per_capita"] = pd.to_numeric(gdp_long["gdp_per_capita"], errors="coerce")

    # Interpolate missing GDP values within each country
    gdp_long = interpolate_by_group(gdp_long, "gdp_per_capita", group_col="country")

    # Remove regional, income, and aggregate entities
    non_country_keywords = [
        "World", "income", "IBRD", "OECD", "Euro area", "European Union",
        "Caribbean", "Sub-Saharan Africa", "Europe & Central Asia",
        "East Asia & Pacific", "Middle East", "North America", "Latin America",
        "Arab World", "South Asia", "Fragile", "Small states", "Pre-demographic",
        "Post-demographic", "demographic dividend", "Not classified",
        "Heavily indebted", "Least developed", "Low & middle",
        "High income", "Lower middle", "Upper middle", "IDA total",
        "IDA blend", "IDA only", "IDA & IBRD"
    ]

    mask = gdp_long["country"].apply(
        lambda x: not any(keyword.lower() in x.lower() for keyword in non_country_keywords)
    )
    gdp_long = gdp_long[mask].copy()

    # Normalize GDP per capita using global mean/std BEFORE World Cup filtering
    gdp_mean = gdp_long["gdp_per_capita"].mean()
    gdp_std = gdp_long["gdp_per_capita"].std()
    gdp_long["norm_gdp_per_capita"] = (gdp_long["gdp_per_capita"] - gdp_mean) / gdp_std

    # Keep observations from 1990 onward
    gdp_long = gdp_long[gdp_long["year"] >= 1990].reset_index(drop=True)

    world_cup_gdp, non_world_cup_gdp = filter_wc(
        gdp_long, "country", include_non=True
    )

    world_cup_path = save_world_cup(world_cup_gdp, "gdp", OUTPUT_DIR)

    if verbose:
        print("\nWorld Cup GDP overview:")
        print(
            f" - Rows: {world_cup_gdp.shape[0]} | Countries: {world_cup_gdp['country'].nunique()} "
            f"| Years: {world_cup_gdp['year'].min()}-{world_cup_gdp['year'].max()}"
        )
        print(" - gdp_per_capita summary:")
        print(world_cup_gdp["gdp_per_capita"].describe())
        print(" - norm_gdp_per_capita summary:")
        print(world_cup_gdp["norm_gdp_per_capita"].describe())
        print(f" - Saved CSV: {world_cup_path}")
        check_country_coverage(world_cup_gdp)

    return world_cup_gdp, non_world_cup_gdp, gdp_long


if __name__ == "__main__":
    world_cup_gdp, non_world_cup_gdp, gdp_long = clean_gdp()
