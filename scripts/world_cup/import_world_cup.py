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

world_cups = read_file("world_cups.csv")
# Has host by year

world_cup_matches = read_file("world_cup_matches.csv")

################# Exploration
## Duplicate matches
world_cup_matches_dups = (
    world_cup_matches.
    groupby(by = ['Year', 'Stage', 'Home Team', 'Away Team']).
    filter(lambda x: len(x) > 1).
    sort_values(by = ['Year', 'Stage', 'Home Team', 'Away Team'])
    )

stages_by_tournament = (
    world_cup_matches.
    groupby(by = ['Year', 'Stage']).
    size().
    reset_index(name = 'count').
    sort_values(by = ['Year', 'Stage'])
    )

unique_stages = np.sort(stages_by_tournament['Stage'].unique())

################ CClean 1986-2018 data
stage_order = [
    'Group stage',
    'Round of 16',
    'Quarter-finals',
    'Semi-finals',
    'Third place',
    'Final'
]
stage_mapping = {stage: i+1 for i, stage in enumerate(stage_order)}

world_cup_1986_2018 = (
    world_cup_matches.query('Year >= 1986')
    [['Year', 'Date', 'Stage', 'Home Team', 'Away Team', 'Home Goals', 'Away Goals']].
    assign(clean_stage = lambda df: df['Stage'].map(stage_mapping),
           winner=lambda df: np.where(
                               df['Home Goals'] > df['Away Goals'],
                               df['Home Team'],
                               np.where(
                                   df['Home Goals'] < df['Away Goals'],
                                   df['Away Team'],
                                   'Draw')
                               )).
    sort_values(by = ['Year', 'clean_stage', 'Date','Home Team'],
                ascending = [True, False, True, True])
    )

performance_by_team_1986_2018 = (
    world_cup_1986_2018.
    groupby
    )
