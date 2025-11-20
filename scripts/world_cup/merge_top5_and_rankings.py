#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:16:49 2025

@author: ssastri1
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]

################ Comparison function
def compare_datasets(A, B, key):
    # ---- 1️⃣ Identify key sets ----
    A_keys = A[key].drop_duplicates()
    B_keys = B[key].drop_duplicates()

    # keys only in A
    only_A_keys = A_keys.merge(B_keys, on=key, how='left', indicator=True)
    only_A_keys = only_A_keys[only_A_keys['_merge'] == 'left_only'].drop(columns='_merge')

    # keys only in B
    only_B_keys = B_keys.merge(A_keys, on=key, how='left', indicator=True)
    only_B_keys = only_B_keys[only_B_keys['_merge'] == 'left_only'].drop(columns='_merge')

    # keys in both (intersection)
    common_keys = A_keys.merge(B_keys, on=key, how='inner').drop_duplicates()

    # ---- 2️⃣ Subset both datasets to only common keys ----
    A_common = A.merge(common_keys, on=key).sort_values(key).reset_index(drop=True)
    B_common = B.merge(common_keys, on=key).sort_values(key).reset_index(drop=True)

    # ---- 3️⃣ Align columns ----
    A_common = A_common[sorted(A_common.columns)]
    B_common = B_common[sorted(B_common.columns)]

    # ---- 4️⃣ Compare only the common rows ----
    differences = A_common.compare(B_common)

    # ---- 5️⃣ Extract full rows for keys only in A or B ----
    only_A_rows = A.merge(only_A_keys, on=key, how='inner')
    only_B_rows = B.merge(only_B_keys, on=key, how='inner')

    return differences, only_A_rows, only_B_rows

################ Imports
##Baseline performance by (team, world cup)
performance_wo_2022 = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/performance_by_world_cup_and_team.csv") # sachin
performance_w_2022 = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/perf_by_wc_team_incl2022.csv") # estee
# Share of top 5 league players in each (team, world cup)
share_of_top5 = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/prop_big5.csv")
# Rankings data (raw)
rankings = pd.read_csv(BASE_DIR / "data/raw_data_files/World Cup Data/Rankings/fifa_ranking-2024-06-20.csv")
# Match level data
match_level_data = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/match_level_data.csv") 

############### SANITY CHECKING

# Compare wo and w 2022 data
col_check = print(performance_wo_2022.columns == performance_w_2022.columns)

# Confirm 2022 is only addl year in estee dataset
years_w = set(performance_w_2022['world_cup_year'])
years_wo = set(performance_wo_2022['world_cup_year'])

anti_years = years_w - years_wo
print(anti_years)

# Do a full check of the two datasets pre 2022
differences, A_only, B_only = compare_datasets(
    performance_wo_2022,
    performance_w_2022,
    key=['team', 'world_cup_year']
)

print("SS rows not in EN:",  A_only.shape[0])
print("EG rows not in SS:", B_only.shape[0])
print("Mismatched common rows:", differences.shape[0])


########## Combine 2018 and 2022 data
## From 2022 data:
## 1) drop countries that never qualified for any world cup between 1986 and 2022, for consistency
## 2) clean the "max_stage" variable based on pre-2022 -- appears to always be "Did Not Qualify"
## 3) Handle Czechoslovakia and Yugoslavia
stage_mapping_pre_2022 = performance_wo_2022[["max_stage", "max_stage_numeric"]].drop_duplicates()
perf_2022 = (performance_w_2022[(performance_w_2022['world_cup_year'] == 2022) & (performance_w_2022['max_stage_numeric'] > 0)]
             .drop(columns = 'max_stage')
             .merge(stage_mapping_pre_2022, on = 'max_stage_numeric', how = 'left')
             .assign(team = lambda df: df['team'].replace({
            'IR Iran': 'Iran',
            'Korea Republic': 'South Korea'
        }))
)

row_keys = ['team', 'world_cup_year']

concat_init = (
    pd.concat([performance_wo_2022, perf_2022], axis = 0)
    .sort_values(by = row_keys)
    )

all_team_years = (
    pd.DataFrame({'world_cup_year': concat_init['world_cup_year'].unique()})
    .merge(
        pd.DataFrame({'team': concat_init['team'].unique()}),
        how='cross'
    )
)

fill_dict = {
    'max_stage': 'Did not qualify',
    'max_stage_numeric': 0,
    'matches_played': 0,
    'matches_won': 0,
    'goals_for': 0,
    'goals_against': 0
}

stacked_data_1986_2022 = (
    all_team_years.merge(concat_init, on=['world_cup_year', 'team'], how='left')
        .fillna(fill_dict)
        .sort_values(['world_cup_year', 'team'])
)

############ Clean countries
df = stacked_data_1986_2022 

# ---- Define groups of country names ----
cz_successors = ['Czech Republic', 'Slovakia']
yugo_successors = [
    'Croatia',
    'Slovenia',
    'Bosnia and Herzegovina',
    'Serbia',
    'Montenegro',
    'Serbia and Montenegro',
    'North Macedonia',
]

# ---- Build masks for rows you want to DROP ----

# 1) Czechoslovakia block:
#    - drop Czech/Slovak rows in early years when Czechoslovakia is the union
drop_cz_early = (
    df['team'].isin(cz_successors)
    & (df['world_cup_year'] <= 1990)
)

#    - drop Czechoslovakia rows in later years after dissolution
drop_cz_late = (
    (df['team'] == 'Czechoslovakia')
    & (df['world_cup_year'] >= 1994)
)

# 2) Yugoslavia block:
#    - drop successor states in early years when Yugoslavia is the union
drop_yugo_early = (
    df['team'].isin(yugo_successors)
    & (df['world_cup_year'] <= 1990)
)

#    - drop Yugoslavia rows in later years
drop_yugo_late = (
    (df['team'] == 'Yugoslavia')
    & (df['world_cup_year'] >= 1994)
)

# Combine everything you want to drop
drop_mask = drop_cz_early | drop_cz_late | drop_yugo_early | drop_yugo_late

# Filter
performance_data_1986_2022 = df[~drop_mask].copy()



# ############## SANITY CHECK
## Should be 10
num_obs_by_country = (
    performance_data_1986_2022.
    groupby('team').
    size().
    unique()
    )

## Should be the total number of countries that ever qualified for a wc in all years
num_obs_by_wc = (
    performance_data_1986_2022.
    groupby('world_cup_year').
    size().
    unique()
    )

## Check years for czechoslovakia, yugoslavia
czech_yugo_check = (
    performance_data_1986_2022[performance_data_1986_2022['team'].isin(cz_successors + yugo_successors + ['Czechoslovakia', 'Yugoslavia'])]
    .groupby('team')['world_cup_year']
    .agg(min_year = 'min',
         max_year = 'max')
    )

# ############################## Merge share of top 5 league
team_year_performance_and_share_of_top5 = (
    performance_data_1986_2022
    .merge(share_of_top5[row_keys + ['Big5_flag']], on=row_keys, how='left')
)

na_big5_by_wc_year = (
    team_year_performance_and_share_of_top5
    .assign(qualified=lambda df: np.where(df['max_stage_numeric'] > 0, 1, 0))
    .groupby(['world_cup_year', 'qualified'])['Big5_flag']
    .agg(
        non_missing=lambda x: x.notna().sum(),
        missing=lambda x: x.isna().sum()
    )
)
# #Missing for 1986, 1990, and 2022.  Populated for all qualified teams in all other years


############################ Merge rankings

## Investigations
ranking_dates = (
    rankings
    .groupby('rank_date', as_index = False)
    .agg(num_rows = ('rank', 'count'))
    .sort_values(by = ['rank_date'])
    )

ranking_dates['rank_date'] = pd.to_datetime(ranking_dates['rank_date'])

# compute lag in days
ranking_dates['lag_days'] = (
    ranking_dates['rank_date'] - ranking_dates['rank_date'].shift(1)
).dt.days

ranking_countries = (
    rankings
    .groupby('country_full')
    .size())

wc_countries = np.sort(team_year_performance_and_share_of_top5['team'].unique())
ranking_countries = np.sort(rankings['country_full'].unique())
countries_in_wc_not_rankings = set(wc_countries) - set(ranking_countries)
print("Before cleaning:", countries_in_wc_not_rankings)

country_mapping = {'Czechia': 'Czech Republic',
                   'IR Iran': 'Iran',
                   "Côte d'Ivoire": 'Ivory Coast',
                   'USA': 'United States',
                   'China PR': 'China',
                   'Korea Republic': 'South Korea',
                   'Korea DPR': 'North Korea'}

rankings_clean_country = rankings.assign(
    clean_country = rankings['country_full'].replace(country_mapping)
)

updated_countries_in_wc_not_rankings = set(wc_countries) - set(rankings_clean_country['clean_country'].unique())
print("After cleaning:", updated_countries_in_wc_not_rankings)


# For each world cup, identify the closest ranking date
world_cup_start_dates = (
    match_level_data
    .assign(date = lambda df: pd.to_datetime(df['Date']).dt.date)
    .groupby('Year', as_index = False)
    .agg(start_date = ('Date', 'min'))
    .rename(columns = {'Year': 'world_cup_year'})
    )

world_cup_start_dates = (pd.concat([world_cup_start_dates,
                                  pd.DataFrame({'world_cup_year': [2022],
                                                'start_date': ['2022-11-20']})
                                  .assign(start_date = lambda df: pd.to_datetime(df['start_date']).dt.date)])
                         .reset_index(drop = True)
)

merge_world_cup_start_dates = (team_year_performance_and_share_of_top5
                        .merge(world_cup_start_dates, on = 'world_cup_year', how = 'left')
                        [lambda df: df['world_cup_year'] >= 1994]
 )

# Identify closest rank date beforeeach world cup start date
rankings_clean_country['rank_date'] = pd.to_datetime(rankings_clean_country['rank_date'])
merge_world_cup_start_dates['start_date'] = pd.to_datetime(merge_world_cup_start_dates['start_date'])
rankings_clean_country = rankings_clean_country.sort_values('rank_date')
merge_world_cup_start_dates = merge_world_cup_start_dates.sort_values('start_date')

closest_rank_date_by_wc = pd.merge_asof(
    merge_world_cup_start_dates,
    rankings_clean_country[['rank_date']],
    left_on='start_date',
    right_on='rank_date',
    direction='backward'
)

merge_country_ranks_before_wc = (pd.merge(closest_rank_date_by_wc,
                                         rankings_clean_country[['rank_date', 'clean_country', 'rank']],
                                         left_on = ['rank_date', 'team'],
                                         right_on = ['rank_date', 'clean_country'],
                                         how = 'left')
                                 .sort_values(by = ['team', 'world_cup_year']))

merge_failures = (
    merge_country_ranks_before_wc[merge_country_ranks_before_wc['clean_country'].isna()]
    )


OUTPUT_DIR = BASE_DIR / "data/created_datasets/world_cup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
world_cup_path = OUTPUT_DIR / "merge_country_ranks_before_wc.csv"
merge_country_ranks_before_wc.to_csv(world_cup_path, index=False)

