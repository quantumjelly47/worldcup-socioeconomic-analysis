#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:16:49 2025

@author: ssastri1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

################ Imports
wc_socio_merged = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/merge_wc_with_socioeconomic.csv")

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

# Dataset with unadjusted performance metrics and socio metrics normalized across countries by year
wc_perf_and_normalized_socio = (pd.merge(data_for_analysis.drop(columns = normalized_socio_data_each_wc.columns),
                                        normalized_socio_data_each_wc,
                                        left_index = True,
                                        right_index = True)
                                .assign(gdp_per_capita_growth_3yr = lambda df: df['gdp_per_capita_tminus0'] - df['gdp_per_capita_tminus3'],
                                        hdi_growth_3yr = lambda df: df['hdi_tminus0'] - df['hdi_tminus3'],
                                        life_expectancy_growth_3yr = lambda df: df['life_expectancy_tminus0'] - df['life_expectancy_tminus3'],
                                        mean_school_years_growth_3yr = lambda df: df['mean_school_years_tminus0'] - df['mean_school_years_tminus3']))

################# Scatterplot/box plot/stacked bar chart of normalized socioeconomic indicators by wc stage
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


# Helper to convert column names → human-readable text
def pretty_metric(metric):
    base = metric.replace("_tminus0", "").replace("_growth_3yr", "")
    return metric_labels.get(base)

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


SAVE_DIR = BASE_DIR / "plots" / "q1_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig, filename):
    if filename is None:
        return
    out_path = SAVE_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


# ============================================================
#               1. BOXPLOT
# ============================================================

def plot_box(df, metrics, growth_flag = False, filename=None):

    # Ensure consistent stage order

    df = df.copy()
    df["max_stage"] = pd.Categorical(df["max_stage"],
                                     categories=stage_labels,
                                     ordered=True)

    fig, axes = make_grid(2, 2, figsize=(14, 10))
    
    box_color = "#B0B0B0"   # medium gray
    dot_color = "#2A9D8F"  # teal/green
    
    xmin = min(df[m].min() for m in metrics)
    xmax = max(df[m].max() for m in metrics)

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x="max_stage", y=metric, ax=ax, color = box_color)
        
        
        sns.pointplot(
            data=df,
            x="max_stage",
            y=metric,
            estimator=np.mean,
            errorbar=None,                   # remove CIs
            markers="o",
            markersize=8,                    # replaces scale=1.3
            linestyles="",                   # no connecting line
            err_kws={"linewidth": 0},        # replaces errwidth=0
            ax=ax)

        
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
    
    if growth_flag:
        title = "Distribution of Past 3 Years Socieconomic Change Across Teams at Each Stage"
    else:
        title = "Normalized Socioeconomic Indicators by World Cup Stage"
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, filename)
    plt.show()



# ============================================================
#       2. STACKED BAR CHART
# ============================================================

from matplotlib.colors import LinearSegmentedColormap


# === Option A Strong Contrast Green Palette ===
quartile_cmap = LinearSegmentedColormap.from_list(
    "ses_green_strong",
    ["#e6f5d0", "#a1d76a", "#4daf4a", "#006837"]   # pale → lime → green → deep forest
)


def plot_metric_stage_stacked(df, metrics, q=4, filename=None):

    df_v = df[df.max_stage_numeric >= 1].copy()   # Keep Group Stage +

    # Create SES percentile bucket labels
    bucket_labels = [f"{int(100*i/q)}-{int(100*(i+1)/q)}%" for i in range(q)]

    # Assign buckets for each socioeconomic metric
    for m in metrics:
        df_v[f"{m}_bucket"] = pd.qcut(
            df_v[m], q, labels=bucket_labels, duplicates="drop"
        )

    # Stage ordering (ensures Winner is last)
    stage_order = (
        df_v[["max_stage_numeric","max_stage"]]
        .drop_duplicates()
        .sort_values("max_stage_numeric")["max_stage"]
        .tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):

        # Count and convert to share of teams within stage
        result = (
            df_v.groupby(["max_stage", f"{metric}_bucket"], observed=True)
                .size()
                .reset_index(name="count")
        )
        result["proportion"] = (
            result.groupby("max_stage")["count"]
                  .transform(lambda x: x / x.sum())
        )

        # Wide format for stacking
        wide = result.pivot(
            index="max_stage",
            columns=f"{metric}_bucket",
            values="proportion"
        ).reindex(stage_order)

        clean = pretty_metric(metric.replace("_tminus0",""))

        # === Stacked bars with new palette ===
        wide.plot(
            kind="bar",
            stacked=True,
            colormap=quartile_cmap,
            ax=ax
        )

        # Axis + label cleanup
        ax.set_title(f"{clean} Percentile Composition by Stage", fontsize=12.5)
        ax.set_xlabel("Stage Reached")
        ax.set_ylabel("Share of Teams")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.30)

        # Rotate stage labels 45°
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Clean, human legend title
        ax.legend(
            title=f"{clean} Percentile",
            bbox_to_anchor=(1.05,1),
            loc="upper left"
        )

    fig.suptitle("Percentage of Teams In Each Socioeconomic Quartile By Stage", fontsize=18, x = 0.5)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96]) 

    if filename:
        save_figure(fig, filename)

    plt.show()

# Metrics to plot
metrics = [
    "gdp_per_capita_tminus0",
    "hdi_tminus0",
    "life_expectancy_tminus0",
    "mean_school_years_tminus0"
]

growth_metrics = [
    "gdp_per_capita_growth_3yr",
    "hdi_growth_3yr",
    "life_expectancy_growth_3yr",
    "mean_school_years_growth_3yr"
]


plot_box(df, metrics, False, 'box.png')
plot_box(df, growth_metrics, True, 'growth_box.png')
plot_metric_stage_stacked(df, metrics, q = 4, filename = "stacked_bar_chart.png")

######################### Ordinal regression

##### Check correlation between socioeconomic indicators for multicollinearity
fig, ax = plt.subplots(figsize=(8,6))  # <-- capture figure first

sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation Between Socioeconomic Predictors")

save_figure(fig, "socio_corr_heatmap.png")   # <-- save here

plt.show()


#### Run ordered model on a socio score that is just mean of normalized scores across the four indicators
from statsmodels.miscmodels.ordinal_model import OrderedModel

df_reg = df[metrics+['max_stage_numeric']].dropna()
df_reg["socio_index"] = df[metrics].mean(axis=1)
model = OrderedModel(df_reg['max_stage_numeric'], df_reg[['socio_index']],   # one variable at a time
        distr='logit').fit(method = 'bfgs', disp = False)
print(model.summary())
odds_ratio = np.exp(0.2913)

def plot_milestone_prob_curves(model, df_reg, filename=None):

    x_vals = np.linspace(df_reg["socio_index"].min(), df_reg["socio_index"].max(), 200)
    exog = pd.DataFrame({"socio_index": x_vals})

    # This returns a pandas DataFrame — let's keep it that way
    pred = model.predict(exog=exog, which="prob")  # shape: (N rows, 7–8 outcome classes)

    # Probability of reaching AT LEAST a given milestone
    p_knockout = pred.iloc[:, 2:].sum(axis=1)   # stage ≥ Round of 16
    p_qf       = pred.iloc[:, 3:].sum(axis=1)   # stage ≥ QF
    p_sf       = pred.iloc[:, 4:].sum(axis=1)   # stage ≥ SF
    p_final    = pred.iloc[:, 6:].sum(axis=1)   # stage ≥ Final (Final or Winner)
    

    # === Plot === #
    plt.figure(figsize=(12,7))

    plt.plot(x_vals, p_knockout, linewidth=3, label="Reach Knockouts (R16+)", color="#1f77b4")
    plt.plot(x_vals, p_qf,       linewidth=3, label="Reach Quarter-finals", color="#2ca02c")
    plt.plot(x_vals, p_sf,       linewidth=3, label="Reach Semi-finals", color="#d62728")
    plt.plot(x_vals, p_final,    linewidth=3, label="Reach Final/Win", color="#9467bd")

    plt.title("Probability of Reaching World Cup Milestones\nas Socioeconomic Index Increases", fontsize=18)
    plt.xlabel("Socioeconomic Index (Composite of GDP, HDI, Life Expectancy, Education)")
    plt.ylabel("Predicted Probability")
    plt.ylim(0,1)
    plt.grid(alpha=0.25)
    plt.legend(title="Milestone", fontsize=11)

    if filename:
        save_figure(plt.gcf(), filename)

plot_milestone_prob_curves(model, df_reg, "milestone_prob_vs_composite_score.png")


########################################### Changes from t-3 to t
df["gdp_growth_3yr"] = df["gdp_per_capita_tminus0"] - df["gdp_per_capita_tminus3"]



