"""
Merge World Cup performance data with top-5 league share and FIFA rankings,
adding 2022 performance and cleaning historical country records.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]


def compare_datasets(A: pd.DataFrame, B: pd.DataFrame, key):
    """
    Compare two DataFrames on a given key and report differences and key-only rows.

    Args:
        A (pd.DataFrame): First dataset.
        B (pd.DataFrame): Second dataset.
        key (str | list): Column name(s) to join on.

    Returns:
        tuple:
            differences (pd.DataFrame): Cell-level differences for matching keys.
            only_A_rows (pd.DataFrame): Rows whose keys appear only in A.
            only_B_rows (pd.DataFrame): Rows whose keys appear only in B.
    """
    # 1) Identify key sets
    A_keys = A[key].drop_duplicates()
    B_keys = B[key].drop_duplicates()

    # Keys only in A
    only_A_keys = A_keys.merge(B_keys, on=key, how="left", indicator=True)
    only_A_keys = only_A_keys[only_A_keys["_merge"] == "left_only"].drop(columns="_merge")

    # Keys only in B
    only_B_keys = B_keys.merge(A_keys, on=key, how="left", indicator=True)
    only_B_keys = only_B_keys[only_B_keys["_merge"] == "left_only"].drop(columns="_merge")

    # Keys in both (intersection)
    common_keys = A_keys.merge(B_keys, on=key, how="inner").drop_duplicates()

    # 2) Subset both datasets to only common keys
    A_common = A.merge(common_keys, on=key).sort_values(key).reset_index(drop=True)
    B_common = B.merge(common_keys, on=key).sort_values(key).reset_index(drop=True)

    # 3) Align columns
    A_common = A_common[sorted(A_common.columns)]
    B_common = B_common[sorted(B_common.columns)]

    # 4) Compare only the common rows
    differences = A_common.compare(B_common)

    # 5) Extract full rows for keys only in A or B
    only_A_rows = A.merge(only_A_keys, on=key, how="inner")
    only_B_rows = B.merge(only_B_keys, on=key, how="inner")

    return differences, only_A_rows, only_B_rows


def main():
    # Load baseline inputs
    performance_wo_2022 = pd.read_csv(
        BASE_DIR / "data/created_datasets/world_cup/performance_by_world_cup_and_team.csv"
    )
    performance_w_2022 = pd.read_csv(
        BASE_DIR / "data/created_datasets/world_cup/perf_by_wc_team_incl2022.csv"
    )
    share_of_top5 = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/prop_big5.csv")
    rankings = pd.read_csv(BASE_DIR / "data/raw_data_files/World Cup Data/Rankings/fifa_ranking-2024-06-20.csv")
    match_level_data = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/match_level_data.csv")

    # Quick sanity checks on 2022 add-in
    print(performance_wo_2022.columns == performance_w_2022.columns)
    years_w = set(performance_w_2022["world_cup_year"])
    years_wo = set(performance_wo_2022["world_cup_year"])
    print(years_w - years_wo)  # should show {2022}

    differences, A_only, B_only = compare_datasets(
        performance_wo_2022,
        performance_w_2022,
        key=["team", "world_cup_year"],
    )
    print("SS rows not in EN:", A_only.shape[0])
    print("EG rows not in SS:", B_only.shape[0])
    print("Mismatched common rows:", differences.shape[0])

    # Build 2022 performance cleaned against pre-2022 stage labels
    stage_mapping_pre_2022 = performance_wo_2022[["max_stage", "max_stage_numeric"]].drop_duplicates()
    perf_2022 = (
        performance_w_2022[
            (performance_w_2022["world_cup_year"] == 2022)
            & (performance_w_2022["max_stage_numeric"] > 0)
        ]
        .drop(columns="max_stage")
        .merge(stage_mapping_pre_2022, on="max_stage_numeric", how="left")
        .assign(team=lambda df: df["team"].replace({"IR Iran": "Iran", "Korea Republic": "South Korea"}))
    )

    row_keys = ["team", "world_cup_year"]

    # Combine and backfill non-qualifiers across all team/year combos
    concat_init = (
        pd.concat([performance_wo_2022, perf_2022], axis=0)
        .sort_values(by=row_keys)
    )

    all_team_years = (
        pd.DataFrame({"world_cup_year": concat_init["world_cup_year"].unique()})
        .merge(pd.DataFrame({"team": concat_init["team"].unique()}), how="cross")
    )

    fill_dict = {
        "max_stage": "Did not qualify",
        "max_stage_numeric": 0,
        "matches_played": 0,
        "matches_won": 0,
        "goals_for": 0,
        "goals_against": 0,
    }

    stacked_data_1986_2022 = (
        all_team_years.merge(concat_init, on=["world_cup_year", "team"], how="left")
        .fillna(fill_dict)
        .sort_values(["world_cup_year", "team"])
    )

    # Clean historical country records (Czechoslovakia / Yugoslavia splits)
    df = stacked_data_1986_2022
    cz_successors = ["Czech Republic", "Slovakia"]
    yugo_successors = [
        "Croatia",
        "Slovenia",
        "Bosnia and Herzegovina",
        "Serbia",
        "Montenegro",
        "Serbia and Montenegro",
        "North Macedonia",
    ]

    drop_cz_early = df["team"].isin(cz_successors) & (df["world_cup_year"] <= 1990)
    drop_cz_late = (df["team"] == "Czechoslovakia") & (df["world_cup_year"] >= 1994)
    drop_yugo_early = df["team"].isin(yugo_successors) & (df["world_cup_year"] <= 1990)
    drop_yugo_late = (df["team"] == "Yugoslavia") & (df["world_cup_year"] >= 1994)
    drop_mask = drop_cz_early | drop_cz_late | drop_yugo_early | drop_yugo_late

    performance_data_1986_2022 = df[~drop_mask].copy()

    # Keep only countries that qualified at least once from 1994–2022
    qualified_teams = (
        performance_data_1986_2022[
            (performance_data_1986_2022["world_cup_year"] >= 1994)
            & (performance_data_1986_2022["max_stage_numeric"] > 0)
        ]["team"]
        .drop_duplicates()
    )
    performance_data_1986_2022 = performance_data_1986_2022[
        performance_data_1986_2022["team"].isin(qualified_teams)
    ].copy()

    # Merge top-5 league share
    team_year_performance_and_share_of_top5 = performance_data_1986_2022.merge(
        share_of_top5[row_keys + ["Big5_flag"]], on=row_keys, how="left"
    )

    # Merge rankings: clean names, find closest ranking date before WC start
    wc_countries = np.sort(team_year_performance_and_share_of_top5["team"].unique())
    # Normalize country names so rankings table aligns with WC dataset
    country_mapping = {
        "Czechia": "Czech Republic",
        "IR Iran": "Iran",
        "CA'te d'Ivoire": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "USA": "United States",
        "China PR": "China",
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
    }
    rankings_clean_country = rankings.assign(
        clean_country=rankings["country_full"].replace(country_mapping)
    )
    print(
        "After cleaning missing countries in rankings:",
        set(wc_countries) - set(rankings_clean_country["clean_country"].unique()),
    )

    world_cup_start_dates = (
        match_level_data.assign(date=lambda df: pd.to_datetime(df["Date"]).dt.date)
        .groupby("Year", as_index=False)
        .agg(start_date=("Date", "min"))
        .rename(columns={"Year": "world_cup_year"})
    )
    # Add 2022 start date
    world_cup_start_dates = (
        pd.concat(
            [
                world_cup_start_dates,
                pd.DataFrame({"world_cup_year": [2022], "start_date": ["2022-11-20"]}).assign(
                    start_date=lambda df: pd.to_datetime(df["start_date"]).dt.date
                ),
            ]
        )
        .reset_index(drop=True)
    )

    merge_world_cup_start_dates = (
        team_year_performance_and_share_of_top5.merge(world_cup_start_dates, on="world_cup_year", how="left")
        .loc[lambda df: df["world_cup_year"] >= 1994]
    )

    rankings_clean_country["rank_date"] = pd.to_datetime(rankings_clean_country["rank_date"])
    merge_world_cup_start_dates["start_date"] = pd.to_datetime(merge_world_cup_start_dates["start_date"])
    rankings_clean_country = rankings_clean_country.sort_values("rank_date")
    merge_world_cup_start_dates = merge_world_cup_start_dates.sort_values("start_date")

    closest_rank_date_by_wc = pd.merge_asof(
        merge_world_cup_start_dates,
        rankings_clean_country[["rank_date"]],
        left_on="start_date",
        right_on="rank_date",
        direction="backward",
    )

    merge_country_ranks_before_wc = (
        pd.merge(
            closest_rank_date_by_wc,
            rankings_clean_country[["rank_date", "clean_country", "rank"]],
            left_on=["rank_date", "team"],
            right_on=["rank_date", "clean_country"],
            how="left",
        )
        .sort_values(by=["team", "world_cup_year"])
    )

    # Export
    OUTPUT_DIR = BASE_DIR / "data/created_datasets/world_cup"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    world_cup_path = OUTPUT_DIR / "merge_country_ranks_before_wc.csv"
    merge_country_ranks_before_wc.to_csv(world_cup_path, index=False)
    print(f"Saved merged rankings: {world_cup_path} ({merge_country_ranks_before_wc.shape})")


if __name__ == "__main__":
    main()
