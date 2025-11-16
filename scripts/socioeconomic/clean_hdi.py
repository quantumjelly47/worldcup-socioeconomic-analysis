from pathlib import Path

import pandas as pd
from scripts.utils.filter_countries import filter_world_cup_countries, check_country_coverage

OUTPUT_DIR = Path("data/created_datasets/socioeconomic")


def clean_hdi(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """Load, clean, and reshape the UNDP HDI dataset (1990–2023)."""

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

    # Select relevant columns
    hdi_cols = ['iso3', 'country', 'hdicode', 'region', 'hdi_rank_2023'] + [
        c for c in undp.columns if c.startswith('hdi_') and not c.startswith(('hdi_f', 'hdi_m'))
    ]
    hdi_subset = undp[hdi_cols].loc[:, ~undp[hdi_cols].columns.duplicated()]

    # Reshape from wide to long format
    hdi_long = hdi_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="hdi"
    )
    hdi_long["year"] = hdi_long["year"].str.replace("hdi_", "").astype(int)
    hdi_long["hdi"] = pd.to_numeric(hdi_long["hdi"], errors="coerce")

    # Remove regional or summary rows
    non_countries = [
        "Very high human development", "High human development",
        "Medium human development", "Low human development",
        "Arab States", "East Asia and the Pacific", "Europe and Central Asia",
        "Latin America and the Caribbean", "South Asia",
        "Sub-Saharan Africa", "World"
    ]
    hdi_long = hdi_long[~hdi_long["country"].isin(non_countries)].copy()

    # Interpolation helper
    def interpolate_safely(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    # Interpolate missing HDI by country
    hdi_long["hdi"] = (
        hdi_long.groupby("country", group_keys=False)["hdi"]
        .apply(interpolate_safely)
    )

    # Fill missing HDI with regional means
    hdi_long["hdi"] = (
        hdi_long.groupby(["region", "year"])["hdi"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Final fallback for any remaining missing values
    hdi_long["region"] = hdi_long["region"].fillna("GLOBAL")
    hdi_long["hdi"] = (
        hdi_long.groupby(["region", "year"])["hdi"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Keep observations from 1990 onward
    hdi_long = hdi_long[hdi_long["year"] >= 1990].reset_index(drop=True)

    # Filter to World Cup countries (and optionally keep non–World Cup data)
    world_cup_hdi, non_world_cup_hdi = filter_world_cup_countries(hdi_long, "country", include_non_world_cup=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    world_cup_path = OUTPUT_DIR / "hdi_world_cup.csv"
    world_cup_hdi.to_csv(world_cup_path, index=False)

    if verbose:
        print("\nWorld Cup HDI overview:")
        print(
            f" - Rows: {world_cup_hdi.shape[0]} | Countries: {world_cup_hdi['country'].nunique()} "
            f"| Years: {world_cup_hdi['year'].min()}-{world_cup_hdi['year'].max()}"
        )
        print(" - HDI summary stats:")
        print(world_cup_hdi["hdi"].describe())
        print(f" - Saved CSV: {world_cup_path}")
        check_country_coverage(world_cup_hdi)

    return world_cup_hdi, non_world_cup_hdi, hdi_long


if __name__ == "__main__":
    world_cup_hdi, non_world_cup_hdi, hdi_long = clean_hdi()

    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample HDI values:")
    print(hdi_long[hdi_long["country"].isin(sample)].sort_values(["country", "year"]))
