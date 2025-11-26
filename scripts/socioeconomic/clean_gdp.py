from pathlib import Path

import pandas as pd
from scripts.utils.filter_countries import filter_world_cup_countries, check_country_coverage

OUTPUT_DIR = Path("data/created_datasets/socioeconomic")


def clean_gdp(path="data/raw_data_files/Socioeconomic Data/GDP per Capita Data.csv", verbose=True):
    """Load, clean, and subset the GDP per Capita dataset (1960-2024)."""

    # Load dataset
    gdp = pd.read_csv(path, encoding="latin-1", skiprows=4)

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
    def interpolate_gdp(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    gdp_long["gdp_per_capita"] = (
        gdp_long.groupby("country", group_keys=False)["gdp_per_capita"]
        .apply(interpolate_gdp)
    )

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

    # Normalize GDP per capita using global mean/std (before World Cup filtering)
    gdp_mean = gdp_long["gdp_per_capita"].mean()
    gdp_std = gdp_long["gdp_per_capita"].std()
    gdp_long["norm_gdp_per_capita"] = (gdp_long["gdp_per_capita"] - gdp_mean) / gdp_std

    # Keep observations from 1990 onward
    gdp_long = gdp_long[gdp_long["year"] >= 1990].reset_index(drop=True)

    world_cup_gdp, non_world_cup_gdp = filter_world_cup_countries(
        gdp_long, column="country", include_non_world_cup=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    world_cup_path = OUTPUT_DIR / "gdp_world_cup.csv"
    world_cup_gdp.to_csv(world_cup_path, index=False)

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
