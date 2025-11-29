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


def clean_schooling(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """
    Load, clean, and subset the UNDP Mean Years of Schooling dataset (1990–2023), interpolate/fill missing values,
    and produce World Cup vs. non–World Cup splits.

    Args:
        path (str): Path to the raw HDI CSV (schooling columns included).
        verbose (bool): Whether to print summary stats and coverage.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (world_cup_school, non_world_cup_school, full_long)
    """

    undp = load_csv(path)
    undp.columns = undp.columns.str.strip()
    undp = assign_developed(undp)
    undp["region"] = undp["region"].replace("Other Countries or Territories", "Developed")

    mys_cols = ["iso3", "country", "hdicode", "region", "hdi_rank_2023"] + [
        c for c in undp.columns if c.startswith("mys_") and not c.startswith(("mys_f", "mys_m"))
    ]
    mys_subset = undp[mys_cols].loc[:, ~undp[mys_cols].columns.duplicated()]

    mys_long = mys_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="mean_school_years",
    )
    mys_long["year"] = mys_long["year"].str.replace("mys_", "").astype(int)
    mys_long["mean_school_years"] = pd.to_numeric(mys_long["mean_school_years"], errors="coerce")

    mys_long["country"] = mys_long["country"].str.strip()
    mys_long = drop_aggregates(mys_long)
    mys_long = interpolate_by_group(mys_long, "mean_school_years")
    mys_long = fill_by_region(mys_long, "mean_school_years")

    mys_long["region"] = mys_long["region"].fillna("GLOBAL")
    mys_long = fill_by_region(mys_long, "mean_school_years")

    # Normalize schooling globally (mean/std over all countries before WC filter)
    school_mean = mys_long["mean_school_years"].mean()
    school_std = mys_long["mean_school_years"].std()
    mys_long["norm_mean_school_years"] = (mys_long["mean_school_years"] - school_mean) / school_std

    mys_long = mys_long[mys_long["year"] >= 1990].reset_index(drop=True)

    world_cup_school, non_world_cup_school = filter_wc(
        mys_long, "country", include_non=True
    )

    world_cup_path = save_world_cup(world_cup_school, "schooling", OUTPUT_DIR)

    if verbose:
        missing = mys_long[mys_long["mean_school_years"].isna()]["country"].unique()
        print(f"\nRemaining missing MYS values: {mys_long['mean_school_years'].isna().sum()}")
        if len(missing) > 0:
            print("Countries still missing data:", missing)
        else:
            print("All MYS values filled successfully.")
        print("\nWorld Cup Schooling overview:")
        print(
            f" - Rows: {world_cup_school.shape[0]} | Countries: {world_cup_school['country'].nunique()} "
            f"| Years: {world_cup_school['year'].min()}-{world_cup_school['year'].max()}"
        )
        print(" - mean_school_years summary:")
        print(world_cup_school["mean_school_years"].describe())
        print(" - norm_mean_school_years summary:")
        print(world_cup_school["norm_mean_school_years"].describe())
        print(f" - Saved CSV: {world_cup_path}")
        check_country_coverage(world_cup_school)

    return world_cup_school, non_world_cup_school, mys_long


if __name__ == "__main__":
    world_cup_school, non_world_cup_school, mys_long = clean_schooling()
    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample Mean Years of Schooling values:")
    print(mys_long[mys_long["country"].isin(sample)].head(10))
