from pathlib import Path

import pandas as pd

from scripts.utils.filter_countries import check_country_coverage
from scripts.utils.socio_helpers import (
    BASE_DIR,
    assign_developed,
    drop_aggregates,
    fill_by_region,
    filter_wc,
    interpolate_by_group,
    load_csv,
    save_world_cup,
)

OUTPUT_DIR = BASE_DIR / "data" / "created_datasets" / "socioeconomic"


def clean_life_expectancy(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """
    Load, clean, and subset the UNDP Life Expectancy dataset (1990–2023), interpolate/fill missing values,
    and produce World Cup vs. non–World Cup splits.

    Args:
        path (str): Path to the raw HDI CSV (life expectancy columns included).
        verbose (bool): Whether to print summary stats and coverage.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (world_cup_life, non_world_cup_life, full_long)
    """

    undp = load_csv(path)
    undp.columns = undp.columns.str.strip()
    undp = assign_developed(undp)

    le_cols = ["iso3", "country", "hdicode", "region", "hdi_rank_2023"] + [
        c for c in undp.columns if c.startswith("le_") and not c.startswith(("le_f", "le_m"))
    ]
    le_subset = undp[le_cols].loc[:, ~undp[le_cols].columns.duplicated()]

    le_long = le_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="life_expectancy",
    )
    le_long["year"] = le_long["year"].str.replace("le_", "").astype(int)
    le_long["life_expectancy"] = pd.to_numeric(le_long["life_expectancy"], errors="coerce")

    le_long = drop_aggregates(le_long)
    le_long = interpolate_by_group(le_long, "life_expectancy")
    le_long = fill_by_region(le_long, "life_expectancy")

    le_long["region"] = le_long["region"].fillna("GLOBAL")
    le_long = fill_by_region(le_long, "life_expectancy")

    # Normalize life expectancy globally (mean/std over all countries before WC filter)
    life_mean = le_long["life_expectancy"].mean()
    life_std = le_long["life_expectancy"].std()
    le_long["norm_life_expectancy"] = (le_long["life_expectancy"] - life_mean) / life_std

    le_long = le_long[le_long["year"] >= 1990].reset_index(drop=True)

    world_cup_life, non_world_cup_life = filter_wc(
        le_long, "country", include_non=True
    )

    world_cup_path = save_world_cup(world_cup_life, "life_expectancy", OUTPUT_DIR)

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
        print(" - norm_life_expectancy summary:")
        print(world_cup_life["norm_life_expectancy"].describe())
        print(f" - Saved CSV: {world_cup_path}")
        check_country_coverage(world_cup_life)

    return world_cup_life, non_world_cup_life, le_long


if __name__ == "__main__":
    world_cup_life, non_world_cup_life, le_long = clean_life_expectancy()
    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample Life Expectancy values:")
    print(le_long[le_long["country"].isin(sample)].head(10))
