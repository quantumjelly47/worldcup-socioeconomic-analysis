from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw_data_files" / "World Cup Data" / "World+Cup"
OUTPUT_DIR = BASE_DIR / "data" / "created_datasets" / "world_cup"

# ============================================================
#                       GLOBAL VARIABLES
# ============================================================


MIN_YEAR = 1994
MAX_YEAR = 2022

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

# ============================================================
#                       IMPORT FUNCTIONS
# ============================================================

def read_csv(name: str) -> pd.DataFrame:
    """Load a CSV from the World Cup raw data directory."""
    return pd.read_csv(RAW_DIR / name)


def load_sources():
    """Load hosts, historical matches (1986–2018), and 2022 matches."""
    hosts = read_csv("world_cups.csv")
    matches_1986_2018 = read_csv("world_cup_matches.csv")
    matches_2022 = read_csv("matches_1930_2022.csv")
    return hosts, matches_1986_2018, matches_2022



# ============================================================
#                       PRE-2022 DATA PROCESSING
# ============================================================
def build_match_level_pre_2022(matches: pd.DataFrame) -> pd.DataFrame:
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


def summarize_by_team_pre_2022(match_long: pd.DataFrame) -> pd.DataFrame:
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
        .assign(team = lambda df: np.where((df['Team'] == 'Yugoslavia') & (df['Year'] >= 1998),
                                           "Serbia",
                                           df['Team']))
        .filter(['Year', 'team', 'max_stage', 'max_stage_numeric', 'matches_played', 'matches_won', 'goals_for', 'goals_against'])
        .sort_values(by=["Year", "max_stage_numeric"], ascending=[False, False])
    )
    
    
    return merged

# ============================================================
#                       2022 DATA PROCESSING
# ============================================================

def summarize_by_team_2022(df:  pd.DataFrame) -> pd.DataFrame:
    """
    Summarize World Cup performance for a given year by team.
    Returns max stage (numeric and label), matches played/won, goals for/against.
    """
    # Pull all matches for the target year from the archive
    matches = df[df["Year"] == 2022]

    # Keep match fields we need, parse shootout info, and standardize column names
    matches = (
        matches[
            [
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "Round",
                "Date",
                "Score",
                "Host",
                "Year",
                "home_penalty_shootout_goal_long",
                "away_penalty_shootout_goal_long",
            ]
        ]
        .assign(
            home_penalty_shootout_goal_long=lambda x: x["home_penalty_shootout_goal_long"]
            .str.split(",")
            .str.len(),
            away_penalty_shootout_goal_long=lambda x: x["away_penalty_shootout_goal_long"]
            .str.split(",")
            .str.len(),
        )
        .rename(
            columns={
                "Round": "Stage",
                "Host": "Host Team",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "home_score": "Home Goals",
                "away_score": "Away Goals",
                "home_penalty_shootout_goal_long": "home_shootout",
                "away_penalty_shootout_goal_long": "away_shootout",
            }
        )
    )

    # Build home-side rows
    home_df = matches.assign(
        team=matches["Home Team"],
        home_away="home",
        opponent=matches["Away Team"],
        goals_scored=matches["Home Goals"],
        goals_conceded=matches["Away Goals"],
        shootout_status=lambda d: np.where(
            d["home_shootout"].isna() & d["away_shootout"].isna(),
            np.nan,
            (d["home_shootout"].fillna(0) > d["away_shootout"].fillna(0)).astype(int),
        ),
    )[
        [
            "team",
            "home_away",
            "opponent",
            "Stage",
            "Date",
            "goals_scored",
            "goals_conceded",
            "shootout_status",
        ]
    ]

    # Build away-side rows
    away_df = matches.assign(
        team=matches["Away Team"],
        home_away="away",
        opponent=matches["Home Team"],
        goals_scored=matches["Away Goals"],
        goals_conceded=matches["Home Goals"],
        shootout_status=lambda d: np.where(
            d["home_shootout"].isna() & d["away_shootout"].isna(),
            np.nan,
            (d["away_shootout"].fillna(0) > d["home_shootout"].fillna(0)).astype(int),
        ),
    )[
        [
            "team",
            "home_away",
            "opponent",
            "Stage",
            "Date",
            "goals_scored",
            "goals_conceded",
            "shootout_status",
        ]
    ]

    # Combine home/away rows into one team-level frame
    reshaped_df = pd.concat([home_df, away_df], ignore_index=True).reset_index(drop=True)

    # Match outcome: win=1 if ahead in goals or wins shootout
    reshaped_df["win"] = np.where(
        (reshaped_df["goals_scored"] > reshaped_df["goals_conceded"])
        | (reshaped_df["shootout_status"] == 1),
        1,
        np.where(
            (reshaped_df["goals_scored"] < reshaped_df["goals_conceded"])
            | (reshaped_df["shootout_status"] == 0),
            0,
            0,
        ),
    )

    # Stage numeric: handle Final/Third-place win/loss explicitly, otherwise map standard stages
    stage_order = {"Group stage": 1, "Round of 16": 2, "Quarter-finals": 3}
    reshaped_df["stage_numeric"] = np.where(
        (reshaped_df["Stage"] == "Final") & (reshaped_df["win"] == 1),
        7,
        np.where(
            (reshaped_df["Stage"] == "Final") & (reshaped_df["win"] == 0),
            6,
            np.where(
                (reshaped_df["Stage"] == "Third-place match") & (reshaped_df["win"] == 1),
                5,
                np.where(
                    (reshaped_df["Stage"] == "Third-place match") & (reshaped_df["win"] == 0),
                    4,
                    reshaped_df["Stage"].map(stage_order),
                ),
            ),
        ),
    )

    # Team-level aggregates
    df_by_year = (
        reshaped_df.groupby("team")
        .agg(
            max_stage_numeric=("stage_numeric", "max"),
            matches_played=("opponent", "count"),
            matches_won=("win", "sum"),
            goals_for=("goals_scored", "sum"),
            goals_against=("goals_conceded", "sum"),
        )
        .reset_index()
    )

    # Pull the representative row (stage) to recover stage label later
    df_by_year = df_by_year.merge(
        reshaped_df[["team", "stage_numeric", "win"]],
        left_on=["team", "max_stage_numeric"],
        right_on=["team", "stage_numeric"],
    )

    # Map numeric stage back to text label
    df_by_year["max_stage"] = np.where(
        df_by_year["max_stage_numeric"] == 7,
        "Winner",
        np.where(
            df_by_year["max_stage_numeric"] == 6,
            "Runners-up",
            np.where(
                df_by_year["max_stage_numeric"] == 5,
                "Third place",
                np.where(
                    df_by_year["max_stage_numeric"] == 4,
                    "Fourth place",
                    np.where(
                        df_by_year["max_stage_numeric"] == 3,
                        "Quarter-finals",
                        np.where(
                            df_by_year["max_stage_numeric"] == 2, "Round of 16", "Group stage"
                        ),
                    ),
                ),
            ),
        ),
    )

    df_by_year = df_by_year.drop(columns=["win", "stage_numeric"]).drop_duplicates().reset_index(drop=True)
    
    out_df = (
        df_by_year
        .assign(Year = 2022,
                team = lambda df: df["team"].replace({"IR Iran": "Iran", "Korea Republic": "South Korea"}),
                max_stage_numeric = lambda df: df['max_stage_numeric'].astype(int))
        .filter(['Year', 'team', 'max_stage', 'max_stage_numeric', 'matches_played', 'matches_won', 'goals_for', 'goals_against'])
        .sort_values(by=["Year", "max_stage_numeric"], ascending=[False, False])
)
    
    return out_df

# ============================================================
#                       CONSOLIDATE
# ============================================================


def stack_pre_2022_and_2022(pre_2022_df: pd.DataFrame, df_2022: pd.DataFrame):
    """
    Stacks pre 2022 and 2022 data into a single dataframe, sorted by team and year.
    """
    stacked = (
        pd.concat([pre_2022_df, df_2022], axis=0)
        .sort_values(by=['team', 'Year'])
    )
    
    return stacked


def expand_all_pairs(stacked: pd.DataFrame) -> pd.DataFrame:
    """
    Expand to all (Year, Team) combinations present in the summary, filling non-qualifiers
    with defaults (Did not qualify, zeros for stats).
    """
    all_years = stacked["Year"].unique()
    all_teams = stacked["team"].unique()
    all_year_team = pd.MultiIndex.from_product([all_years, all_teams], names=["Year", "team"])
    all_combos = pd.DataFrame(index=all_year_team).reset_index()

    expanded = (
        all_combos.merge(stacked, on=["Year", "team"], how="left")
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
        .sort_values(by=["team", "Year"], ascending=[True, True])
    )
    return expanded


def add_hosts(expanded: pd.DataFrame, hosts: pd.DataFrame) -> pd.DataFrame:
    """Attach host country info and rename columns for the final performance dataset."""
    expanded_w_host = (
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
    return expanded_w_host

def add_wc_start_dates(expanded_w_host: pd.DataFrame, match_long: pd.DataFrame):
    """Attach world cup start date to use for merging rankings later"""
    world_cup_start_dates = (
        match_long.assign(date=lambda df: pd.to_datetime(df["Date"]).dt.date)
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
    
    expanded_w_host_and_start_date = (
        expanded_w_host.merge(world_cup_start_dates, on="world_cup_year", how="left")
        .loc[lambda df: df["world_cup_year"] >= 1994]
    )
    return expanded_w_host_and_start_date

def main():
    """Build performance_by_world_cup_and_team.csv and match_level_data.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hosts, matches_1986_2018, _matches_2022 = load_sources()
    match_long = build_match_level_pre_2022(matches_1986_2018)
    summary = summarize_by_team_pre_2022(match_long)
    summary_2022 = summarize_by_team_2022(_matches_2022)
    stacked = stack_pre_2022_and_2022(summary, summary_2022)
    expanded = expand_all_pairs(stacked)
    expanded_with_host = add_hosts(expanded, hosts)
    expanded_with_host_and_start_date = add_wc_start_dates(expanded_with_host, match_long)

    # Save outputs
    expanded_with_host_and_start_date.to_csv(OUTPUT_DIR / "performance_by_world_cup_and_team.csv", index=False)
    match_long.to_csv(OUTPUT_DIR / "match_level_data.csv", index=False)

    print(f"Saved performance_by_world_cup_and_team.csv ({expanded_with_host_and_start_date.shape})")
    print(f"Saved match_level_data.csv ({match_long.shape})")


if __name__ == "__main__":
    main()
