"""
Merge World Cup performance data with top-5 league share and FIFA rankings,
adding 2022 performance and cleaning historical country records.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data/created_datasets/world_cup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
merge_keys = ['world_cup_year', 'team']

def load_datasets():
    performance = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/performance_by_world_cup_and_team.csv")
    share_of_top5 = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/prop_big5.csv")
    rankings = pd.read_csv(BASE_DIR / "data/raw_data_files/World Cup Data/Rankings/fifa_ranking-2024-06-20.csv")
    return performance, share_of_top5, rankings

def merge_share_of_top5(performance: pd.DataFrame, share_of_top5: pd.DataFrame):
    performance_w_share_of_top5 = performance.merge(
        share_of_top5[merge_keys + ["Big5_flag"]], on=merge_keys, how="left"
    )
    return performance_w_share_of_top5

def merge_rankings(performance_w_share_of_top5: pd.DataFrame, rankings: pd.DataFrame):
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
        "Yugoslavia": "Serbia", # since we are looking post 1994
        "Serbia and Montenegro": "Serbia"
    }
    rankings_clean_country = rankings.assign(
        clean_country=rankings["country_full"].replace(country_mapping)
    )
    
    rankings_clean_country["rank_date"] = pd.to_datetime(rankings_clean_country["rank_date"])
    performance_w_share_of_top5["start_date"] = pd.to_datetime(performance_w_share_of_top5["start_date"])
    rankings_clean_country = rankings_clean_country.sort_values("rank_date")
    performance_w_share_of_top5 = performance_w_share_of_top5.sort_values("start_date")

    closest_rank_date_by_wc = pd.merge_asof(
        performance_w_share_of_top5,
        rankings_clean_country[["rank_date"]],
        left_on="start_date",
        right_on="rank_date",
        direction="backward",
    )

    performance_w_share_of_top5_and_rankings = (
        pd.merge(
            closest_rank_date_by_wc,
            rankings_clean_country[["rank_date", "clean_country", "rank"]],
            left_on=["rank_date", "team"],
            right_on=["rank_date", "clean_country"],
            how="left",
        )
        .sort_values(by=["team", "world_cup_year"])
    )
    
    return performance_w_share_of_top5_and_rankings


def main():
    performance, share_of_top5, rankings = load_datasets()
    performance_w_share_of_top5 = merge_share_of_top5(performance, share_of_top5)
    performance_w_share_of_top5_and_rankings = merge_rankings(performance_w_share_of_top5, rankings)
    
    # Export
    world_cup_path = OUTPUT_DIR / "merge_country_ranks_before_wc.csv"
    performance_w_share_of_top5_and_rankings.to_csv(world_cup_path, index=False)
    print(f"Saved merged rankings: {world_cup_path} ({performance_w_share_of_top5_and_rankings.shape})")


if __name__ == "__main__":
    main()
