from pathlib import Path

import pandas as pd
from scripts.utils.filter_countries import filter_world_cup_countries, check_country_coverage

OUTPUT_DIR = Path("data/created_datasets/socioeconomic")


def clean_life_expectancy(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """Load, clean, and subset the UNDP Life Expectancy dataset (1990-2023)."""

    # Load dataset
    undp = pd.read_csv(path, encoding="latin-1")
    undp.columns = undp.columns.str.strip()

    # Assign missing regions for developed countries
    developed_countries = [
        'Andorra', 'Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'Croatia',
        'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
        'Greece', 'Hong Kong, China (SAR)', 'Hungary', 'Iceland', 'Ireland', 'Israel',
        'Italy', 'Japan', 'Korea (Republic of)', 'Latvia', 'Liechtenstein', 'Lithuania',
        'Luxembourg', 'Malta', 'Netherlands', 'New Zealand', 'Norway', 'Poland',
        'Portugal', 'Romania', 'Russian Federation', 'San Marino', 'Slovakia',
        'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'United States'
    ]
    undp.loc[undp["country"].isin(developed_countries), "region"] = "Developed"

    # Select relevant columns for life expectancy
    le_cols = ['iso3', 'country', 'hdicode', 'region', 'hdi_rank_2023'] + [
        c for c in undp.columns if c.startswith('le_') and not c.startswith(('le_f', 'le_m'))
    ]
    le_subset = undp[le_cols].loc[:, ~undp[le_cols].columns.duplicated()]

    # Reshape from wide to long format
    le_long = le_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="life_expectancy"
    )
    le_long["year"] = le_long["year"].str.replace("le_", "").astype(int)
    le_long["life_expectancy"] = pd.to_numeric(le_long["life_expectancy"], errors="coerce")

    # Remove regional/aggregate rows
    non_countries = [
        "Very high human development", "High human development",
        "Medium human development", "Low human development",
        "Arab States", "East Asia and the Pacific", "Europe and Central Asia",
        "Latin America and the Caribbean", "South Asia",
        "Sub-Saharan Africa", "World"
    ]
    le_long = le_long[~le_long["country"].isin(non_countries)].copy()

    # Interpolation helper
    def interpolate_safely(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    # Interpolate missing life expectancy by country
    le_long["life_expectancy"] = (
        le_long.groupby("country", group_keys=False)["life_expectancy"]
        .apply(interpolate_safely)
    )

    # Fill missing life expectancy with regional averages
    le_long["life_expectancy"] = (
        le_long.groupby(["region", "year"])["life_expectancy"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Final fallback for any remaining missing values
    le_long["region"] = le_long["region"].fillna("GLOBAL")
    le_long["life_expectancy"] = (
        le_long.groupby(["region", "year"])["life_expectancy"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Keep observations from 1990 onward
    le_long = le_long[le_long["year"] >= 1990].reset_index(drop=True)

    # Split into World Cup and non-World Cup countries
    world_cup_life, non_world_cup_life = filter_world_cup_countries(
        le_long, column="country", include_non_world_cup=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    world_cup_path = OUTPUT_DIR / "life_expectancy_world_cup.csv"
    world_cup_life.to_csv(world_cup_path, index=False)

    if verbose:
        missing = le_long[le_long["life_expectancy"].isna()]["country"].unique()
        print(f"\nRemaining missing Life Expectancy values: {le_long['life_expectancy'].isna().sum()}")
        if len(missing) > 0:
            print("Countries still missing data:", missing)
        else:
            print("All Life Expectancy values filled successfully.")
        print("\nWorld Cup Life Expectancy overview:")
        print(
            f" - Rows: {world_cup_life.shape[0]} | Countries: {world_cup_life['country'].nunique()} "
            f"| Years: {world_cup_life['year'].min()}-{world_cup_life['year'].max()}"
        )
        print(" - life_expectancy summary:")
        print(world_cup_life["life_expectancy"].describe())
        print(f" - Saved CSV: {world_cup_path}")
        check_country_coverage(world_cup_life)

    return world_cup_life, non_world_cup_life, le_long


if __name__ == "__main__":
    world_cup_life, non_world_cup_life, le_long = clean_life_expectancy()
    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample Life Expectancy values:")
    print(le_long[le_long["country"].isin(sample)].head(10))
