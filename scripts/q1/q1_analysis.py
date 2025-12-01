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

# ============================================================
#                       DATA PREPARATION
# ============================================================

# Import input file
wc_socio_merged = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/merge_wc_with_socioeconomic.csv")

data_for_analysis = (
    wc_socio_merged
    .assign(qualified = lambda df: np.where(df['max_stage_numeric'] == 0, 
                                            'Did Not Qualify', 
                                            'Qualified'))
    .assign(stage_num_for_expansion = lambda df: np.select(
        condlist = [df['max_stage_numeric'] <= 3,
                    df['max_stage_numeric'] <= 5,
                    df['max_stage_numeric'] <= 7], 
        choicelist = [df['max_stage_numeric'],
                      4,
                      df['max_stage_numeric']-1]))
    .assign(gdp_per_capita_growth_3yr = lambda df: (df['gdp_per_capita_tminus0'] - df['gdp_per_capita_tminus3'])/df['gdp_per_capita_tminus3'],
            hdi_growth_3yr = lambda df: (df['hdi_tminus0'] - df['hdi_tminus3'])/df['hdi_tminus3'],
            life_expectancy_growth_3yr = lambda df: (df['life_expectancy_tminus0'] - df['life_expectancy_tminus3'])/df['life_expectancy_tminus3'],
            mean_school_years_growth_3yr = lambda df: (df['mean_school_years_tminus0'] - df['mean_school_years_tminus3'])/df['mean_school_years_tminus3'])
)

data_for_analysis['qualified'] = pd.Categorical(
    data_for_analysis['qualified'], 
    categories=['Did Not Qualify','Qualified'], 
    ordered=True
)


################ Charting data for did not qualify vs qualified charts
socio_cols_pattern = r'(gdp|hdi|life_expectancy|mean_school_years)'

all_teams_charting_data = (
    data_for_analysis
    .loc[:, ['world_cup_year', 'team', 'qualified'] + data_for_analysis.filter(regex = socio_cols_pattern).columns.tolist()]
    .sort_values(by = ['world_cup_year', 'team'])
)

################# Charting data for qualified teams only
stage_ticks = list(range(1,7))
stage_labels = ["Group Stage","Round of 16","Quarterfinal","Semifinal","Final","Winner"]
stage_map    = dict(zip(stage_ticks,stage_labels))

expanded_rows = []
for _, row in data_for_analysis.iterrows():
    for stage_num in range(1, int(row.stage_num_for_expansion) + 1):

        r = row.copy()
        r["stage_expanded_num"] = stage_num                     
        r["stage_expanded"] = stage_map[stage_num]
        expanded_rows.append(r)

qualified_teams_charting_data = pd.DataFrame(expanded_rows)

# Make stage_expanded an ordered category for consistent plotting
qualified_teams_charting_data["stage_expanded"] = pd.Categorical(
    qualified_teams_charting_data["stage_expanded"],
    categories=stage_labels,
    ordered=True
)

qualified_teams_charting_data = (
    qualified_teams_charting_data
    .loc[:, ['world_cup_year', 'team', 'max_stage', 'stage_expanded_num', 'stage_expanded'] + qualified_teams_charting_data.filter(regex = socio_cols_pattern).columns.tolist()]
    .sort_values(by = ['world_cup_year', 'team', 'stage_expanded'])
)

## Make sure that number of rows per team in a given year makes sense given the stage they reached
sanity_check = (
    qualified_teams_charting_data
    .groupby(['world_cup_year', 'team', 'max_stage'])
    .size().reset_index(name = 'num_rows')
    .groupby(['world_cup_year', 'max_stage', 'num_rows'])
    .size().reset_index(name = 'instances')
    .sort_values(by = ['world_cup_year', 'num_rows'])
    
)

# ============================================================
#                   SETUP AND HELPERS
# ============================================================
# Mapping for cleaner labels
metric_labels = {
    "gdp_per_capita": "GDP Per Capita",
    "hdi": "HDI",
    "life_expectancy": "Life Expectancy",
    "mean_school_years": "Mean School Years"
}


# Helper to convert column names → human-readable text
def pretty_metric(metric):
    base = metric.replace("_tminus0", "").replace("norm_","").replace("_growth_3yr", "")
    return metric_labels.get(base)


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

# Metrics to plot
non_growth_metrics = [
    "norm_gdp_per_capita_tminus0",
    "norm_hdi_tminus0",
    "norm_life_expectancy_tminus0",
    "norm_mean_school_years_tminus0"
]

growth_metrics = [
    "gdp_per_capita_growth_3yr",
    "hdi_growth_3yr",
    "life_expectancy_growth_3yr",
    "mean_school_years_growth_3yr"
]


# ############################################################
# #  1. BOXPLOTS
# ############################################################

# def plot_box(df, growth_flag=False, filename=None):

#     df = df.copy()

#     fig, axes = make_grid(2, 2, figsize=(14, 10))

#     box_color = "#B0B0B0"
#     dot_color = "#2A9D8F"

#     if growth_flag:
#         metrics = growth_metrics
#         shared_ylim = None # Allow variation in y-axis across subplots when looking at growth               
#     else:
#         metrics = non_growth_metrics
#         shared_ylim = (-3, 6) # Limits based on min and max normalized score across all four indicators, rounded to nearest integer            

#     for ax, metric in zip(axes, metrics):

#         sns.boxplot(data=df, x="max_stage", y=metric, ax=ax, color=box_color)

#         sns.pointplot(
#             data=df,
#             x="max_stage",
#             y=metric,
#             estimator=np.mean,
#             errorbar=None,
#             markers="o",
#             markersize=8,
#             linestyles="",
#             err_kws={"linewidth": 0},
#             ax=ax,
#             color=dot_color
#         )

#         clean = pretty_metric(metric)
#         ax.set_title(clean)
#         ax.set_xlabel("Stage Reached")
#         ax.set_ylabel("")
#         ax.tick_params(axis="x", rotation=45)

#         if shared_ylim:
#             ax.set_ylim(shared_ylim)
#         else:
#             ymin, ymax = df[metric].min(), df[metric].max()
#             pad = (ymax - ymin) * 0.1
#             ax.set_ylim(ymin - pad, ymax + pad)

#     mean_handle = mlines.Line2D([], [], color=dot_color, marker='o',
#                                 linestyle='None', markersize=8, label='Mean')
#     fig.legend(handles=[mean_handle],
#                loc='upper right',
#                bbox_to_anchor=(0.98, 0.98))

#     title = ("Distribution of Past 3-Year Socioeconomic Change by World Cup Stage"
#              if growth_flag else
#              "Normalized Socioeconomic Indicators by World Cup Stage")

#     fig.suptitle(title, fontsize=16)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])

#     save_figure(fig, filename)
#     plt.show()





# # ============================================================
# #       2. STACKED BAR CHART
# # ============================================================

# from matplotlib.colors import LinearSegmentedColormap

# # --- SES Strong Green Palette (Contrast-Optimized) ---
# quartile_cmap = LinearSegmentedColormap.from_list(
#     "ses_green_strong",
#     ["#e6f5d0", "#a1d76a", "#4daf4a", "#006837"]   # pale → lime → deep green
# )

# def plot_stacked_bar(df, growth_flag=False, filename=None):

#     df = df.copy()

#     # Set metric and title based on growth_flag
#     if growth_flag:
#         metrics = growth_metrics
#         title = "Distribution of 3-Year Socioeconomic Growth by Stage"
#     else:
#         metrics = non_growth_metrics
#         title = "Percentage of Teams in Each Socioeconomic Quartile by Stage"

#     # Create percentile buckets
#     q = 4
#     bucket_labels = [f"{int(100*i/q)}-{int(100*(i+1)/q)}%" for i in range(q)]
#     for m in metrics:
#         df[f"{m}_bucket"] = pd.qcut(
#             df[m], q, labels=bucket_labels, duplicates="drop"
#         )

#     fig, axes = make_grid(2, 2, figsize=(16, 12))
#     legend_handles = None   # capture legend once globally

#     for ax, metric in zip(axes, metrics):

#         # Proportion of each SES bucket within each stage
#         counts = (
#             df.groupby(["max_stage", f"{metric}_bucket"], observed=True)
#               .size()
#               .reset_index(name="count")
#         )
#         counts["proportion"] = (
#             counts.groupby("max_stage")["count"].transform(lambda x: x/x.sum())
#         )

#         wide = counts.pivot(
#             index="max_stage",
#             columns=f"{metric}_bucket",
#             values="proportion"
#         ).reindex(stage_labels)

#         clean = pretty_metric(metric.replace("_tminus0","").replace("_growth_3yr",""))

#         # Plot stacked bars
#         barplot = wide.plot(
#             kind="bar",
#             stacked=True,
#             colormap=quartile_cmap,
#             width=0.90,
#             ax=ax
#         )

#         # Store handles once to build single global legend later
#         if legend_handles is None:
#             legend_handles = barplot.get_legend_handles_labels()

#         # ─── Formatting ──────────────────────────────────────────────
#         ax.set_title(f"{clean} Quartile Composition by Stage", fontsize=13)
#         ax.set_ylim(0, 1)
#         ax.grid(axis='y', alpha=0.3)
#         ax.set_xlabel("Stage Reached")
#         ax.set_ylabel("Share of Teams")
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

#         ax.legend().remove()  # suppress subplot legends

#     # --- One global legend ---
#     fig.legend(*legend_handles,
#                title="Percentile Bucket",
#                loc="upper right",
#                bbox_to_anchor=(0.98, 0.97))

#     fig.tight_layout(rect=[0,0,0.92,0.92])
#     fig.suptitle(title, fontsize=18, y=0.99)


#     if filename:
#         save_figure(fig, filename)

#     plt.show()


# plot_box(data_for_analysis, False, 'box.png')
# plot_box(data_for_analysis, True, 'growth_box.png')
# plot_stacked_bar(data_for_analysis, False, "stacked_bar_chart.png")
# plot_stacked_bar(data_for_analysis, True, "growth_stacked_bar_chart.png")

# # ######################### Ordinal regression

# # ##### Check correlation between socioeconomic indicators for multicollinearity
# # fig, ax = plt.subplots(figsize=(8,6))  # <-- capture figure first

# # sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
# # ax.set_title("Correlation Between Socioeconomic Predictors")

# # save_figure(fig, "socio_corr_heatmap.png")   # <-- save here

# # plt.show()


# # #### Run ordered model on a socio score that is just mean of normalized scores across the four indicators
# # from statsmodels.miscmodels.ordinal_model import OrderedModel

# # df_reg = df[metrics+['max_stage_numeric']].dropna()
# # df_reg["socio_index"] = df[metrics].mean(axis=1)
# # model = OrderedModel(df_reg['max_stage_numeric'], df_reg[['socio_index']],   # one variable at a time
# #         distr='logit').fit(method = 'bfgs', disp = False)
# # print(model.summary())
# # odds_ratio = np.exp(0.2913)

# # def plot_milestone_prob_curves(model, df_reg, filename=None):

# #     x_vals = np.linspace(df_reg["socio_index"].min(), df_reg["socio_index"].max(), 200)
# #     exog = pd.DataFrame({"socio_index": x_vals})

# #     # This returns a pandas DataFrame — let's keep it that way
# #     pred = model.predict(exog=exog, which="prob")  # shape: (N rows, 7–8 outcome classes)

# #     # Probability of reaching AT LEAST a given milestone
# #     p_knockout = pred.iloc[:, 2:].sum(axis=1)   # stage ≥ Round of 16
# #     p_qf       = pred.iloc[:, 3:].sum(axis=1)   # stage ≥ QF
# #     p_sf       = pred.iloc[:, 4:].sum(axis=1)   # stage ≥ SF
# #     p_final    = pred.iloc[:, 6:].sum(axis=1)   # stage ≥ Final (Final or Winner)
    

# #     # === Plot === #
# #     plt.figure(figsize=(12,7))

# #     plt.plot(x_vals, p_knockout, linewidth=3, label="Reach Knockouts (R16+)", color="#1f77b4")
# #     plt.plot(x_vals, p_qf,       linewidth=3, label="Reach Quarter-finals", color="#2ca02c")
# #     plt.plot(x_vals, p_sf,       linewidth=3, label="Reach Semi-finals", color="#d62728")
# #     plt.plot(x_vals, p_final,    linewidth=3, label="Reach Final/Win", color="#9467bd")

# #     plt.title("Probability of Reaching World Cup Milestones\nas Socioeconomic Index Increases", fontsize=18)
# #     plt.xlabel("Socioeconomic Index (Composite of GDP, HDI, Life Expectancy, Education)")
# #     plt.ylabel("Predicted Probability")
# #     plt.ylim(0,1)
# #     plt.grid(alpha=0.25)
# #     plt.legend(title="Milestone", fontsize=11)

# #     if filename:
# #         save_figure(plt.gcf(), filename)

# # plot_milestone_prob_curves(model, df_reg, "milestone_prob_vs_composite_score.png")


# # ########################################### Changes from t-3 to t
# # df["gdp_growth_3yr"] = df["gdp_per_capita_tminus0"] - df["gdp_per_capita_tminus3"]



