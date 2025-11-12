%reset -f
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:33:59 2025

@author: ssastri28
"""
import numpy as np
import pandas as pd
import os

data_folder = "../../data/raw_data_files/World Cup Data/World+Cup"
files = os.listdir(data_folder)

def read_file(file_name):
    out_df = pd.read_csv(os.path.join(data_folder, file_name))
    return(out_df)

# Import
world_cup_hosts_import = read_file("world_cups.csv")
world_cup_matches_1930_2018_import = read_file("world_cup_matches.csv")
world_cup_matches_2022_import = read_file("2022_world_cup_matches.csv")

# Global variables to filter to world cups before/after a certain year
min_year = 1986
max_year = 2018

# Ordered list of stages a team can reach (applies to post 1986 world cups)
stage_order = [
    'Group stage',
    'Round of 16',
    'Quarter-finals',
    'Semi-finals',
    'Third place',
    'Final'
]
stage_mapping = {stage: i+1 for i, stage in enumerate(stage_order)}

# Filter to relevant world cups
# Create a clean_stage variable to use later in identifying how far a team made it in a given world cup
# Flag matches that went to penalties and who won the shootout
world_cup_matches_wide = (
    world_cup_matches_1930_2018_import.query('Year >= @min_year and Year <= @max_year')
    [['ID', 'Year', 'Date', 'Stage', 
      'Home Team', 'Away Team', 
      'Home Goals', 'Away Goals',
      'Win Conditions']].
    assign(clean_stage = lambda df: df['Stage'].map(stage_mapping),
           penalty_flag = lambda df: df['Win Conditions'].str.lower().str.contains('penalt'),
           penalty_winner = lambda df: np.where(df['penalty_flag'], 
                                                df['Win Conditions'].str.split().str[0], 
                                                pd.NA)).
    sort_values(by = ['Year', 'clean_stage', 'Date','ID'],
                ascending = [True, False, True, True])
    )

# Reshape data so that data is (match, team) level rather than match level
## Pivot home goals, away goals, home team, away team long
long_init = world_cup_matches_wide.melt(
    id_vars=['ID', 'Year', 'Stage', 'clean_stage', 'penalty_winner'],
    value_vars=['Home Team', 'Away Team', 'Home Goals', 'Away Goals'],
    var_name='col_name',
    value_name='col_value'
)

## Track whether a given row is looking at home or away team and at goals or team name
long_init['home_or_away'] = long_init['col_name'].str.extract(r'(Home|Away)', expand=False)
long_init['team_or_goals'] = long_init['col_name'].str.extract(r'(Team|Goals)', expand=False)

world_cup_matches_long = (
    long_init
    .pivot(index=['ID', 'Year', 'Stage', 'clean_stage', 'home_or_away', 'penalty_winner'], 
           columns='team_or_goals', 
           values='col_value')
    .reset_index()
    [['ID', 'Year', 'Stage', 'clean_stage', 'home_or_away', 'Team', 'Goals', 'penalty_winner']]
    .sort_values(by = ['Year', 'clean_stage', 'ID', 'home_or_away'],
                 ascending = [True, False, True, True])
    .assign(goals_against = lambda df: df.groupby(['Year','ID'])['Goals'].transform('sum')-df['Goals'])
    .assign(match_win = lambda df: (df['Goals'] > df['goals_against']).astype(int),
            penalty_win = lambda df: (df['Team'] == df['penalty_winner']).astype(int),
            win = lambda df: df[['match_win', 'penalty_win']].max(axis = 1))
    .reset_index(drop = True)
)


# Summarize max stage reached by team in each world cup
## Store indices of rows corresponding to the latest game played by each team in each world cup
max_stage_indices = world_cup_matches_long.groupby(['Year','Team'])['clean_stage'].idxmax()

## Final position rankings
positions = [
    'Group stage',
    'Round of 16',
    'Quarter-finals',
    'Fourth place',
    'Third place',
    'Runners-up',
    'Winner'
]
positions_mapping = {stage: i+1 for i, stage in enumerate(positions)}

## Filter data to max_stage_indices and then distinguish between winners/losers in final and third place match
stage_by_team_wc = (
    world_cup_matches_long
    .loc[max_stage_indices]
    .rename(columns={'Stage': 'max_stage'})
    .assign(max_stage = lambda df: np.select(
        condlist = [
            (df['max_stage'] == 'Final') & (df['win'] == 1),
            (df['max_stage'] == 'Final') & (df['win'] == 0),
            (df['max_stage'] == 'Third place') & (df['win'] == 1),
            (df['max_stage'] == 'Third place') & (df['win'] == 0)
            ], 
        choicelist = ['Winner', 'Runners-up', 'Third place', 'Fourth place'],
        default = df['max_stage']))
    [['Year', 'Team', 'max_stage']]
    .assign(max_stage_numeric = lambda df: df['max_stage'].map(positions_mapping))
    .sort_values(by = ['Year', 'max_stage_numeric'],
                 ascending = [False, False])
    )

## sanity check that number of teams by stage make sense
num_teams_by_stage = (stage_by_team_wc.
                      groupby(['max_stage', 'max_stage_numeric', 'Year'])['Team']
                      .count()
                      .reset_index(name = 'num_teams')
                      .sort_values(by = ['max_stage_numeric', 'Year'],
                                   ascending = [False, False]))

# Summarize games played, games won, goals scored, goals conceded by (team, world cup)
stats_by_team_wc = (
    world_cup_matches_long
    .groupby(['Year', 'Team'], as_index=False)
    .agg(matches_played = ('ID', 'count'), 
         matches_won = ('win', 'sum'),
         goals_for = ('Goals', 'sum'), 
         goals_against = ('goals_against', 'sum'))
    )

# Merge max stage and stats
merged_summary_by_team_wc = (
    pd.merge(
    stage_by_team_wc,
    stats_by_team_wc,           
    on=['Year', 'Team'],        # merge keys
    how='outer')                # keep all teams in either df
    .sort_values(by = ['Year', 'max_stage_numeric'],
                 ascending = [False, False])
)

# Expand dataset to include all possible (Year, Team) combinations

## Get all unique years and teams
all_years = merged_summary_by_team_wc['Year'].unique()
all_teams = merged_summary_by_team_wc['Team'].unique()

## Create a DataFrame with every combination of Year × Team
full_index = pd.MultiIndex.from_product([all_years, all_teams], names=['Year', 'Team'])
all_year_team_combinations = pd.DataFrame(index=full_index).reset_index()

## Merge the existing summary into the full Year × Team combinations
expanded_summary = (
    pd.merge(
        all_year_team_combinations,
        merged_summary_by_team_wc,
        on=['Year', 'Team'],
        how='left'
    )
    # Fill missing values for non-qualifying teams
    .fillna({
        'max_stage': 'Did not qualify',
        'max_stage_numeric': 0,
        'matches_played': 0,
        'matches_won': 0,
        'goals_for': 0,
        'goals_against': 0
    })
    .sort_values(by = ['Team', 'Year'],
                 ascending = [True, True])
)

expanded_summary_w_host = (
    pd.merge(
        expanded_summary,
        world_cup_hosts_import[['Year', 'Host Country']],
        on = ['Year'],
        how = 'left')
    .rename(columns = {'Host Country': 'host_country', 'Year': 'world_cup_year', 'Team': 'team'})
    [['world_cup_year', 'host_country', 'team', 'max_stage', 'max_stage_numeric',
      'matches_played', 'matches_won', 'goals_for', 'goals_against']]
    )

dataset_folder = "../../data/created_datasets/world_cup"
expanded_summary_w_host.to_csv(os.path.join(dataset_folder, 
                                            "performance_by_world_cup_and_team.csv"), 
                               index=False)

