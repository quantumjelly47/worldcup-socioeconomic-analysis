from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw_data_files" / "World Cup Data" / "World+Cup"
OUTPUT_DIR = BASE_DIR / "data" / "created_datasets" / "world_cup"

MIN_YEAR = 1994
MAX_YEAR = 2018

STAGE_ORDER = [
    "Group stage",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Third place",
    "Final",
]
STAGE_MAPPING = {stage: i + 1 for i, stage in enumerate(STAGE_ORDER)}

POSITIONS = [
    "Group stage",
    "Round of 16",
    "Quarter-finals",
    "Fourth place",
    "Third place",
    "Runners-up",
    "Winner",
]
POSITIONS_MAPPING = {stage: i + 1 for i, stage in enumerate(POSITIONS)}


def read_csv(name: str) -> pd.DataFrame:
    """Load a CSV from the World Cup raw data directory."""
    return pd.read_csv(RAW_DIR / name)


def load_sources():
    """Load hosts, historical matches (1986–2018), and 2022 matches (kept for future use)."""
    hosts = read_csv("world_cups.csv")
    matches_1986_2018 = read_csv("world_cup_matches.csv")
    matches_2022 = read_csv("2022_world_cup_matches.csv")  # kept for future if needed
    return hosts, matches_1986_2018, matches_2022


def build_match_level(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level data to team-level rows with wins, goals for/against, and penalty outcomes.
    Filters to years within MIN_YEAR..MAX_YEAR.
    """
    # Filter years and basic columns
    wide = (
        matches.query("Year >= @MIN_YEAR and Year <= @MAX_YEAR")[
            [
                "ID",
                "Year",
                "Date",
                "Stage",
                "Home Team",
                "Away Team",
                "Home Goals",
                "Away Goals",
                "Win Conditions",
            ]
        ]
        .assign(
            clean_stage=lambda df: df["Stage"].map(STAGE_MAPPING),
            penalty_flag=lambda df: df["Win Conditions"].str.lower().str.contains("penalt"),
            penalty_winner=lambda df: np.where(
                df["penalty_flag"], df["Win Conditions"].str.split().str[0], pd.NA
            ),
        )
        .sort_values(
            by=["Year", "clean_stage", "Date", "ID"],
            ascending=[True, False, True, True],
        )
    )

    # Melt and reshape to team-level rows
    long_init = wide.melt(
        id_vars=["ID", "Year", "Date", "Stage", "clean_stage", "penalty_winner"],
        value_vars=["Home Team", "Away Team", "Home Goals", "Away Goals"],
        var_name="col_name",
        value_name="col_value",
    )
    long_init["home_or_away"] = long_init["col_name"].str.extract(r"(Home|Away)", expand=False)
    long_init["team_or_goals"] = long_init["col_name"].str.extract(r"(Team|Goals)", expand=False)

    match_long = (
        long_init.pivot(
            index=["ID", "Year", "Date", "Stage", "clean_stage", "home_or_away", "penalty_winner"],
            columns="team_or_goals",
            values="col_value",
        )
        .reset_index()
        [["ID", "Year", "Date", "Stage", "clean_stage", "home_or_away", "Team", "Goals", "penalty_winner"]]
        .sort_values(by=["Year", "clean_stage", "ID", "home_or_away"], ascending=[True, False, True, True])
        .assign(goals_against=lambda df: df.groupby(["Year", "ID"])["Goals"].transform("sum") - df["Goals"])
        .assign(
            match_win=lambda df: (df["Goals"] > df["goals_against"]).astype(int),
            penalty_win=lambda df: (df["Team"] == df["penalty_winner"]).astype(int),
        )
        .assign(win=lambda df: df[["match_win", "penalty_win"]].max(axis=1))
        .reset_index(drop=True)
    )
    return match_long


def summarize_by_team(match_long: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize by (Year, Team): max stage reached (numeric and label),
    matches played/won, goals for/against.
    """
    # Max stage reached
    max_stage_indices = match_long.groupby(["Year", "Team"])["clean_stage"].idxmax()
    stage_by_team_wc = (
        match_long.loc[max_stage_indices]
        .rename(columns={"Stage": "max_stage"})
        .assign(
            max_stage=lambda df: np.select(
                condlist=[
                    (df["max_stage"] == "Final") & (df["win"] == 1),
                    (df["max_stage"] == "Final") & (df["win"] == 0),
                    (df["max_stage"] == "Third place") & (df["win"] == 1),
                    (df["max_stage"] == "Third place") & (df["win"] == 0),
                ],
                choicelist=["Winner", "Runners-up", "Third place", "Fourth place"],
                default=df["max_stage"],
            )
        )[["Year", "Team", "max_stage"]]
        .assign(max_stage_numeric=lambda df: df["max_stage"].map(POSITIONS_MAPPING))
        .sort_values(by=["Year", "max_stage_numeric"], ascending=[False, False])
    )

    # Basic stats
    stats_by_team_wc = (
        match_long.groupby(["Year", "Team"], as_index=False).agg(
            matches_played=("ID", "count"),
            matches_won=("win", "sum"),
            goals_for=("Goals", "sum"),
            goals_against=("goals_against", "sum"),
        )
    )

    merged = (
        stage_by_team_wc.merge(stats_by_team_wc, on=["Year", "Team"], how="outer")
        .assign(Team = lambda df: np.where((df['Team'] == 'Yugoslavia') & (df['Year'] >= 1998),
                                           "Serbia",
                                           df['Team']))
        .sort_values(by=["Year", "max_stage_numeric"], ascending=[False, False])
    )
    
    
    return merged


def expand_all_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Expand to all (Year, Team) combinations present in the summary, filling non-qualifiers
    with defaults (Did not qualify, zeros for stats).
    """
    all_years = summary["Year"].unique()
    all_teams = summary["Team"].unique()
    all_year_team = pd.MultiIndex.from_product([all_years, all_teams], names=["Year", "Team"])
    all_combos = pd.DataFrame(index=all_year_team).reset_index()

    expanded = (
        all_combos.merge(summary, on=["Year", "Team"], how="left")
        .fillna(
            {
                "max_stage": "Did not qualify",
                "max_stage_numeric": 0,
                "matches_played": 0,
                "matches_won": 0,
                "goals_for": 0,
                "goals_against": 0,
            }
        )
        .sort_values(by=["Team", "Year"], ascending=[True, True])
    )
    return expanded


def add_hosts(expanded: pd.DataFrame, hosts: pd.DataFrame) -> pd.DataFrame:
    """Attach host country info and rename columns for the final performance dataset."""
    return (
        expanded.merge(hosts[["Year", "Host Country"]], on="Year", how="left")
        .rename(
            columns={
                "Host Country": "host_country",
                "Year": "world_cup_year",
                "Team": "team",
            }
        )[
            [
                "world_cup_year",
                "host_country",
                "team",
                "max_stage",
                "max_stage_numeric",
                "matches_played",
                "matches_won",
                "goals_for",
                "goals_against",
            ]
        ]
    )


def main():
    """Build performance_by_world_cup_and_team.csv and match_level_data.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hosts, matches_1986_2018, _matches_2022 = load_sources()

    match_long = build_match_level(matches_1986_2018)
    summary = summarize_by_team(match_long)
    expanded = expand_all_pairs(summary)
    expanded_with_host = add_hosts(expanded, hosts)

    # Save outputs
    expanded_with_host.to_csv(OUTPUT_DIR / "performance_by_world_cup_and_team.csv", index=False)
    match_long.to_csv(OUTPUT_DIR / "match_level_data.csv", index=False)

    print(f"Saved performance_by_world_cup_and_team.csv ({expanded_with_host.shape})")
    print(f"Saved match_level_data.csv ({match_long.shape})")


if __name__ == "__main__":
    main()
