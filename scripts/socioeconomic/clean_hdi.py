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


def clean_hdi(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """
    Load, clean, and reshape the UNDP HDI panel (1990–2023), interpolate/fill missing values,
    and produce World Cup vs. non–World Cup splits.

    Args:
        path (str): Path to the raw HDI CSV.
        verbose (bool): Whether to print summary stats and coverage.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (world_cup_hdi, non_world_cup_hdi, full_long)
    """

    undp = load_csv(path)
    undp.columns = undp.columns.str.strip()
    undp = assign_developed(undp)

    hdi_cols = ["iso3", "country", "hdicode", "region", "hdi_rank_2023"] + [
        c for c in undp.columns if c.startswith("hdi_") and not c.startswith(("hdi_f", "hdi_m"))
    ]
    hdi_subset = undp[hdi_cols].loc[:, ~undp[hdi_cols].columns.duplicated()]

    hdi_long = hdi_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="hdi",
    )
    hdi_long["year"] = hdi_long["year"].str.replace("hdi_", "").astype(int)
    hdi_long["hdi"] = pd.to_numeric(hdi_long["hdi"], errors="coerce")

    hdi_long = drop_aggregates(hdi_long)
    hdi_long = interpolate_by_group(hdi_long, "hdi")
    hdi_long = fill_by_region(hdi_long, "hdi")

    hdi_long["region"] = hdi_long["region"].fillna("GLOBAL")
    hdi_long = fill_by_region(hdi_long, "hdi")

    # Normalize HDI globally (mean/std over all countries before WC filter)
    hdi_mean = hdi_long["hdi"].mean()
    hdi_std = hdi_long["hdi"].std()
    hdi_long["norm_hdi"] = (hdi_long["hdi"] - hdi_mean) / hdi_std

    hdi_long = hdi_long[hdi_long["year"] >= 1990].reset_index(drop=True)

    world_cup_hdi, non_world_cup_hdi = filter_wc(
        hdi_long, "country", include_non=True
    )

    world_cup_path = save_world_cup(world_cup_hdi, "hdi", OUTPUT_DIR)

    if verbose:
        print("\nWorld Cup HDI overview:")
        print(
            f" - Rows: {world_cup_hdi.shape[0]} | Countries: {world_cup_hdi['country'].nunique()} "
            f"| Years: {world_cup_hdi['year'].min()}-{world_cup_hdi['year'].max()}"
        )
        print(" - HDI summary stats:")
        print(world_cup_hdi["hdi"].describe())
        print(" - norm_hdi summary stats:")
        print(world_cup_hdi["norm_hdi"].describe())
        print(f" - Saved CSV: {world_cup_path}")
        check_country_coverage(world_cup_hdi)

    return world_cup_hdi, non_world_cup_hdi, hdi_long


if __name__ == "__main__":
    world_cup_hdi, non_world_cup_hdi, hdi_long = clean_hdi()

    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample HDI values:")
    print(hdi_long[hdi_long["country"].isin(sample)].sort_values(["country", "year"]))
