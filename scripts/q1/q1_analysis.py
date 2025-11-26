#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:16:49 2025

@author: ssastri1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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

################# Scatterplot/box plot/stage probability of normalized socioeconomic indicators by wc stage
# ============================================================
#                   SETUP AND HELPERS
# ============================================================

df = wc_perf_and_normalized_socio.reset_index()

# Mapping for nicer labels
metric_labels = {
    "gdp_per_capita": "GDP Per Capita",
    "hdi": "HDI",
    "life_expectancy": "Life Expectancy",
    "mean_school_years": "Mean School Years"
}

xmin = min(df[m + "_tminus0"].min() for m in metric_labels.keys())
xmax = max(df[m + "_tminus0"].max() for m in metric_labels.keys())


# Helper to convert column names → human-readable text
def pretty_metric(metric):
    base = metric.replace("_tminus0", "")
    return metric_labels.get(base, base.replace("_", " ").title())

# Generate ordered stage ticks (numeric → string labels)
stage_order_df = (
    df[['max_stage_numeric', 'max_stage']]
    .drop_duplicates()
    .sort_values('max_stage_numeric')
)

stage_ticks = stage_order_df['max_stage_numeric'].tolist()
stage_labels = stage_order_df['max_stage'].tolist()

# Helper to quickly create subplot grids
def make_grid(nrows=2, ncols=2, figsize=(16, 12), sharey=False):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, sharey=sharey)
    return fig, axes.flatten()

# Metrics to plot
metrics = [
    "gdp_per_capita_tminus0",
    "hdi_tminus0",
    "life_expectancy_tminus0",
    "mean_school_years_tminus0"
]

SAVE_DIR = BASE_DIR / "plots" / "q1_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig, filename):
    if filename is None:
        return
    out_path = SAVE_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to: {out_path}")

# ============================================================
#               1. SCATTERPLOT PANEL
# ============================================================

def plot_scatter_panel(df, metrics, filename=None):
    fig, axes = make_grid(2, 2, figsize=(14, 10), sharey=True)

    for ax, metric in zip(axes, metrics):
        sns.scatterplot(
            data=df,
            x=metric,
            y="max_stage_numeric",
            hue="world_cup_year",
            palette="tab10",
            alpha=0.75,
            ax=ax
        )

        ax.set_yticks(stage_ticks)
        ax.set_yticklabels(stage_labels)

        clean = pretty_metric(metric)
        ax.set_title(clean)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(clean)
        ax.set_ylabel("")
        ax.legend_.remove()

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5),
                    title="World Cup Year")

    fig.suptitle("Normalized Socioeconomic Indicators vs Max Stage (All World Cups)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # SAVE
    save_figure(fig, filename)

    plt.show()


# ============================================================
#               2. BOXPLOT PANEL
# ============================================================

def plot_box_panel(df, metrics, filename=None):

    # Ensure consistent stage order

    df = df.copy()
    df["max_stage"] = pd.Categorical(df["max_stage"],
                                     categories=stage_labels,
                                     ordered=True)

    fig, axes = make_grid(2, 2, figsize=(14, 10))
    
    box_color = "#B0B0B0"   # medium gray
    dot_color = "#2A9D8F"  # teal/green

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x="max_stage", y=metric, ax=ax, color = box_color)
        
        sns.pointplot( # plot mean as data point
            data=df,
            x="max_stage",
            y=metric,
            estimator=np.mean,
            color=dot_color,
            markers="o",
            scale=1.2,
            errwidth=0,      # turn off confidence interval
            linestyles="",   # no connecting line
            ax=ax
)
        
        clean = pretty_metric(metric)
        ax.set_title(clean)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel("Stage Reached")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        
        

    mean_handle = mlines.Line2D([], [], color=dot_color, marker='o',
                                linestyle='None', markersize=8, label='Mean')

    fig.legend(handles=[mean_handle],
               loc='upper right',
               bbox_to_anchor=(0.98, 0.98))
    
    fig.suptitle("Normalized Socioeconomic Indicators by World Cup Stage", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, filename)
    plt.show()



# ============================================================
#       3. STAGE PROBABILITY PANEL (“RAISES THE FLOOR”)
# ============================================================

def plot_stage_probability_panel(df, metrics, filename=None):

    df_in_wc = df[df["max_stage_numeric"] > 0].copy()

    stage_thresholds = {
        "Round of 16": 2,
        "Quarter-finals": 3,
        "Semi-finals": 4,
        "Final": 5
    }

    fig, axes = make_grid(2, 2, figsize=(16, 12))

    for ax, metric in zip(axes, metrics):

        tmp = df_in_wc.copy()
        bins = np.linspace(df_in_wc[metric].min(), df_in_wc[metric].max(), 5)
        tmp["_bin"] = pd.cut(tmp[metric], bins=bins, include_lowest=True)

        prob_df = (
            tmp.groupby("_bin")["max_stage_numeric"]
               .apply(list)
               .reset_index()
        )
        prob_df["x"] = prob_df["_bin"].apply(lambda b: b.mid)

        for stage_name, thresh in stage_thresholds.items():
            prob_df[stage_name] = prob_df["max_stage_numeric"].apply(
                lambda lst: np.mean([v >= thresh for v in lst])
            )
            ax.plot(prob_df["x"], prob_df[stage_name], linewidth=2,
                    label=f"P(Reach {stage_name})")

        clean = pretty_metric(metric)
        ax.set_title(clean)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(f"Normalized {clean} (t=0)")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.25)

    axes[-1].legend(title="Stage", loc="center left", bbox_to_anchor=(1.05, 0.5))

    fig.suptitle("How Socioeconomic Indicators 'Raise the Floor' of World Cup Performance", fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    # IMPORTANT: show before saving
    plt.show()




# ============================================================
#                   RUN ALL THREE PANELS
# ============================================================

plot_scatter_panel(df, metrics, 'scatter.png')
plot_box_panel(df, metrics, 'box.png')
plot_stage_probability_panel(df, metrics)
