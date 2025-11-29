from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_ARCHIVE = BASE_DIR / "data" / "raw_data_files" / "World Cup Data" / "archive"
OUTPUT_DIR = BASE_DIR / "data" / "created_datasets" / "world_cup"


def preprocess_year_data(year: int) -> pd.DataFrame:
    """
    Summarize World Cup performance for a given year by team.
    Returns max stage (numeric and label), matches played/won, goals for/against.
    """
    # Pull all matches for the target year from the archive
    matches = pd.read_csv(RAW_ARCHIVE / "matches_1930_2022.csv")
    matches = matches[matches["Year"] == year]

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
    return df_by_year


def main():
    # 2022 performance
    df_2022 = preprocess_year_data(2022)

    # All countries in Oct 2022 FIFA ranking (for non-qualifiers)
    all_countries_ao2022 = pd.read_csv(RAW_ARCHIVE / "fifa_ranking_2022-10-06.csv")
    all_countries_ao2022["team"] = all_countries_ao2022["team"].replace({"USA": "United States"})

    # Keep countries that did not qualify in 2022 so we can append "Did not qualify" rows
    non_qualified = all_countries_ao2022[
        ~all_countries_ao2022["team"].isin(set(df_2022["team"].unique()))
    ][["team"]]

    # Combine qualifiers + non-qualifiers and fill missing performance with DNQ/zeros
    final_2022_df = pd.concat([df_2022, non_qualified], ignore_index=True)
    final_2022_df["max_stage"] = final_2022_df["max_stage"].fillna("Did not qualify")
    final_2022_df = final_2022_df.fillna(0)
    final_2022_df["world_cup_year"] = 2022
    final_2022_df["host_country"] = "Qatar"

    # Base performance data (1986–2018)
    perf_df = pd.read_csv(OUTPUT_DIR / "performance_by_world_cup_and_team.csv")

    # Append 2022 to historical performance
    final_df = pd.concat([perf_df, final_2022_df], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "perf_by_wc_team_incl2022.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Saved combined performance data including 2022: {out_path} ({final_df.shape})")


if __name__ == "__main__":
    main()
