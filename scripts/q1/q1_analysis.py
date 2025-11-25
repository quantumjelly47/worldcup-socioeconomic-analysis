#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:16:49 2025

@author: ssastri1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
wc_socio_merged = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/merge_wc_with_socioeconomic.csv")

#### TO REMOVE:  check that all countries are associated with all world cup years - both arrays below should have exactly one value
print("Number of observations by country:", wc_socio_merged.groupby('team').size().unique())
print("Number of observations by world_cup_year:", wc_socio_merged.groupby('world_cup_year').size().unique())


########### Data preparation
# Set wc year and team as index
data_for_analysis = (
    wc_socio_merged.set_index(['world_cup_year','team'])
    )

# Normalize data
def normalize(df):
    '''
    Normalizes all columns in the input dataframe by subtracting their respective means and dividing by their respective standard deviations
    Args:
    df:  A dataframe with only numeric columns
    Returns:
    Normalized version of df
    '''
    return (df - df.mean())/df.std()

socio_cols_pattern = r'^(gdp|hdi|life_expectancy|mean_school_years)'
normalized_socio_data_each_wc = (
    data_for_analysis
    .filter(regex=socio_cols_pattern)
    .groupby(level = 'world_cup_year')
    .transform(normalize)
    )

# check normalization
print(normalized_socio_data_each_wc.groupby(level = 'world_cup_year').agg(['mean', 'std'])) # means should be close to 0, sd 1

# Dataset with unadjusted performance metrics and socio metrics normalized across countries by year
wc_perf_and_normalized_socio = pd.merge(data_for_analysis.drop(columns = normalized_socio_data_each_wc.columns),
                                        normalized_socio_data_each_wc,
                                        left_index = True,
                                        right_index = True)

################# Scatterplots of normalized socioeconomic indicators by wc stage in each world cup
metrics = normalized_socio_data_each_wc.columns[normalized_socio_data_each_wc.columns.str.contains('0')]
metric_mapping = {'gdp_per_capita': 'GDP Per Capita', 'hdi': 'HDI', 
                  'life_expectancy': 'Life Expectancy', 'mean_school_years': 'Mean School Years'}

df = wc_perf_and_normalized_socio.reset_index()
stage_order = (
    df[['max_stage_numeric', 'max_stage']]
    .drop_duplicates()
    .sort_values('max_stage_numeric')['max_stage']
    .tolist()
)

df['max_stage'] = pd.Categorical(df['max_stage'], categories=stage_order, ordered=True)
df['world_cup_year'] = df['world_cup_year'].astype('category')


n = len(metrics)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10), sharey=True)
axes = axes.flatten()

# Build ordered stage list
stage_order_df = (
    df[['max_stage_numeric', 'max_stage']]
    .drop_duplicates()
    .sort_values('max_stage_numeric')
)
ticks = stage_order_df['max_stage_numeric'].tolist()
ticklabels = stage_order_df['max_stage'].tolist()

for ax, metric in zip(axes, metrics):
    
    sns.scatterplot(
        data=df,
        x=metric,
        y='max_stage_numeric',        
        hue='world_cup_year',
        palette='tab10',
        alpha=0.75,
        ax=ax
    )
    
    # Replace numeric ticks with stage labels
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    ax.set_xlabel(metric_mapping[metric.replace("_tminus0", "")])
    ax.set_ylabel("")   
    ax.set_title(metric_mapping[metric.replace("_tminus0", "")])
    ax.tick_params(axis='x')

# Hide empty axes
for j in range(len(metrics), len(axes)):
    axes[j].set_visible(False)

plt.suptitle(
    "Normalized Socioeconomic Indicators vs Max Stage (All World Cups)",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



########################## Box plot of socio economic indicators by stage reached
n = len(metrics)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10), sharey=False)
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    
    sns.boxplot(
        data=df,
        x='max_stage',
        y=metric,
        ax=ax
    )
    
    ax.set_xlabel("Stage Reached")
    ax.set_ylabel("")   # we add custom labels below
    ax.set_title(metric_mapping[metric.replace("_tminus0", "")])
    ax.tick_params(axis='x', rotation=45)

# Hide any unused axes (if metrics count is odd)
for j in range(len(metrics), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Normalized Socioeconomic Indicators by World Cup Stage, Across All World Cups", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

####################### Stage probability curve
df_in_wc = df[df["max_stage_numeric"] > 0].copy()
# Tournament stages you want to model
stage_thresholds = {
    
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Final": 5
}

# --- Setup the subplot grid ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# --- Loop through socioeconomic indicators ---
for ax, metric in zip(axes, metrics):

    # Bin into 20 quantile bins
    df_in_wc["_bin"] = pd.qcut(df_in_wc[metric], 20, duplicates="drop")

    # Prepare probability DataFrame
    prob_df = (
        df_in_wc.groupby("_bin")["max_stage_numeric"]
                .apply(list)
                .reset_index()
    )
    prob_df["x"] = prob_df["_bin"].apply(lambda b: b.mid)

    # Compute probability for each stage threshold
    for stage_name, threshold in stage_thresholds.items():
        prob_df[stage_name] = prob_df["max_stage_numeric"].apply(
            lambda lst: np.mean([x >= threshold for x in lst])
        )

    # Plot probability curves
    for stage_name in stage_thresholds.keys():
        ax.plot(
            prob_df["x"],
            prob_df[stage_name],
            label=f"P(Reach {stage_name})",
            linewidth=2.0
        )
        
    clean_metric = metric_mapping[metric.replace("_tminus0", "")]
    ax.set_title(clean_metric, fontsize=14)
    ax.set_xlabel(f"Normalized {clean_metric} (t=0)")
    ax.set_ylabel("Probability of Reaching Stage")
    ax.grid(True, alpha=0.25)

# Legend only on final plot
axes[-1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), title="Stage")

plt.suptitle("How Socioeconomic Indicators 'Raise the Floor' of World Cup Performance", fontsize=18)
plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()



