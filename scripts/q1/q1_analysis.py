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
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter
from statsmodels.miscmodels.ordinal_model import OrderedModel

BASE_DIR = Path(__file__).resolve().parents[2] # Base directory
SAVE_DIR = BASE_DIR / "plots" / "q1_analysis" # Directory to save plots
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
#                       DATA PREPARATION
# ============================================================

# Import input file
wc_socio_merged = pd.read_csv(BASE_DIR / "data/created_datasets/world_cup/merge_wc_with_socioeconomic.csv")

# Create a qualified vs did not qualify flag and create a variable indicating the stage that each team reached (slightly different from max_stage, used in qualified_teams_charting_data)
# Create 3-year delta for each socieconomic indicator - tells you whether a country moved up or down the global socioeconomic ladder
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
    .assign(gdp_per_capita_growth_3yr = lambda df: df['norm_gdp_per_capita_tminus0'] - df['norm_gdp_per_capita_tminus3'],
            hdi_growth_3yr = lambda df: df['norm_hdi_tminus0'] - df['norm_hdi_tminus3'],
            life_expectancy_growth_3yr = lambda df: df['norm_life_expectancy_tminus0'] - df['norm_life_expectancy_tminus3'],
            mean_school_years_growth_3yr = lambda df: df['norm_mean_school_years_tminus0'] - df['norm_mean_school_years_tminus3'])
)

# Make qualified an ordered categorical variable
data_for_analysis['qualified'] = pd.Categorical(
    data_for_analysis['qualified'], 
    categories=['Did Not Qualify','Qualified'], 
    ordered=True
)


################ Charting data for did not qualify vs qualified charts
socio_cols_pattern = r'^(norm_|.*growth_3yr$)'

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

qualified_teams_charting_data = pd.DataFrame(expanded_rows).reset_index(drop = True)

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
    """
    Convert a raw metric column name into a cleaner human-readable label to use in charts.

    This function removes suffixes like `_tminus0`, `norm_`, and `_growth_3yr`,
    then maps the cleaned term to a descriptive label defined in `metric_labels`.

    Args:
        metric (str): Raw column name from one of the charting datasets

    Returns:
        str or None: A cleaned, human-readable metric label. Returns None
            if the cleaned name is not found in `metric_labels`.
    """
    base = metric.replace("_tminus0", "").replace("norm_","").replace("_growth_3yr", "")
    return metric_labels.get(base)


# Helper to quickly create subplot grids
def make_grid(nrows=2, ncols=2, figsize=(16, 12), sharey=False):
    """
    Create a grid of subplots and return the figure and a flattened axes array.

    Args:
        nrows (int, optional): Number of subplot rows. Defaults to 2.
        ncols (int, optional): Number of subplot columns. Defaults to 2.
        figsize (tuple, optional): Figure size in inches (width, height).
            Defaults to (16, 12).
        sharey (bool, optional): Whether to share the y-axis across subplots.

    Returns:
        tuple:
            fig (matplotlib.figure.Figure): The created figure.
            axes (np.ndarray): Flattened array of Axes objects.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, sharey=sharey)
    return fig, axes.flatten()

def save_figure(fig, filename):
    """
    Save a Matplotlib figure to the project's standard plot directory.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        filename (str or None): Output filename (without directory).  
            If None, the function performs no action.

    Returns:
        None
    """

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
# #  1. BOX PLOTS
# ############################################################
def plot_box(df, x_axis_var, growth_flag=False, filename=None):
    """
    Generate a 2×2 grid of boxplots comparing socioeconomic variables
    across qualification categories or World Cup stages.

    The figure also overlays mean markers and applies dynamic y-axis labels.
    When `growth_flag=True`, the function uses 3-year socioeconomic deltas.

    Args:
        df (pd.DataFrame): Input dataset containing socioeconomic variables
            and either a 'qualified' or 'stage_expanded' column.
        x_axis_var (str): Column used for the x-axis. Must be either
            'qualified' or 'stage_expanded'.
        growth_flag (bool, optional): If True, plots growth_metrics; if False,
            plots non_growth_metrics. Defaults to False.
        filename (str or None): If provided, the figure is saved under this
            filename using `save_figure`.

    Returns:
        None
    """

    df = df.copy()

    fig, axes = make_grid(2, 2, figsize=(14, 10))

    box_color = "#5B8CC0"
    dot_color = "#0A3D91"

    if growth_flag:
        metrics = growth_metrics
        base_ylabel = "Past 3-Year Normalized Score Change"
        suptitle_txt_1 = "Distribution of Past 3-Year Normalized Socioeconomic Change"
    else:
        metrics = non_growth_metrics
        base_ylabel = ""
        suptitle_txt_1 = "Distribution of Normalized Socioeconomic Indicators"

    if x_axis_var == "qualified":
        x_label = "Qualification Status"
        suptitle_txt_2 = "Among Teams that Qualified vs Teams that Did Not Qualify"
    elif x_axis_var == "stage_expanded":
        x_label = "Stage"
        suptitle_txt_2 = "Among Teams At Each World Cup Stage"

    for ax, metric in zip(axes, metrics):

        sns.boxplot(data=df, x=x_axis_var, y=metric, ax=ax, color=box_color)

        sns.pointplot(
            data=df,
            x=x_axis_var,
            y=metric,
            estimator=np.mean,
            errorbar=None,
            markers="o",
            markersize=8,
            linestyles="",
            err_kws={"linewidth": 0},
            ax=ax,
            color=dot_color
        )

        clean = pretty_metric(metric)
        ax.set_title(clean)

        # Local Y-label logic (fix)
        if base_ylabel == "":
            local_ylabel = f"Normalized {clean}"
        else:
            local_ylabel = base_ylabel
        ax.set_ylabel(local_ylabel)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", rotation=45)

        ymin, ymax = df[metric].min(), df[metric].max()
        pad = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    mean_handle = mlines.Line2D([], [], color=dot_color, marker='o',
                                linestyle='None', markersize=8, label='Mean')

    fig.legend(handles=[mean_handle], loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # Better title formatting
    title = f"{suptitle_txt_1}\n{suptitle_txt_2}"

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_figure(fig, filename)
    plt.show()

plot_box(all_teams_charting_data, 'qualified', False, 'qualifed_vs_not_level_box.png')
plot_box(all_teams_charting_data, 'qualified', True, 'qualifed_vs_not_growth_box.png')
plot_box(qualified_teams_charting_data, 'stage_expanded', False, 'by_stage_level_box.png')
plot_box(qualified_teams_charting_data, 'stage_expanded', True, 'by_stage_growth_box.png')

# # ============================================================
# #       2. STACKED BAR CHART
# # ============================================================
# Strong contrast SES green palette
quartile_colors = [
    "#DCE6F2",
    "#A9C2E3",
    "#5B8CC0",
    "#1F4F82"
]
quartile_cmap = ListedColormap(quartile_colors)

def plot_stacked_bar(df, x_axis_var, filename=None):
    """
    Plot 2×2 stacked bar charts showing the distribution of teams across
    socioeconomic quartiles for each metric.  Only considers levels (i.e. non_growth_metrics).

    Quartiles are computed independently for each metric using `pd.qcut`.
    Bars are grouped by qualification or World Cup stage.

    Args:
        df (pd.DataFrame): Input dataset containing socioeconomic variables
            and either 'qualified' or 'stage_expanded'.
        x_axis_var (str): Categorical variable on the x-axis. Must be either
            'qualified' or 'stage_expanded'.
        filename (str or None): If provided, the final figure is saved under
            this name in the standard plot directory.

    Raises:
        ValueError: If `x_axis_var` is not one of the accepted values.

    Returns:
        None
    """

    df = df.copy()

    metrics = non_growth_metrics
    suptitle_txt_1 = "Proportion of Teams in Each Socioeconomic Quartile"

    # Handle x-axis type (qualification vs stage)
    if x_axis_var == "qualified":
        x_label = "Qualification Status"
        suptitle_txt_2 = "Among Teams that Qualified vs Teams that Did Not Qualify"
        x_order = ['Did Not Qualify', 'Qualified']
    elif x_axis_var == "stage_expanded":
        x_label = "Stage"
        suptitle_txt_2 = "Among Teams At Each World Cup Stage"
        x_order = stage_labels
    else:
        raise ValueError("x_axis_var must be 'qualified' or 'stage_expanded'")

    # Ensure x-axis is categorical in correct order
    df[x_axis_var] = pd.Categorical(df[x_axis_var], categories=x_order, ordered=True)

    # Build percentile buckets
    q = 4
    bucket_labels = [f"{int(100*i/q)}-{int(100*(i+1)/q)}%" for i in range(q)]

    # Create subplots
    fig, axes = make_grid(2, 2, figsize=(16, 12))

    legend_handles = None

    for ax, metric in zip(axes, metrics):

        # Assign quartiles for each metric
        df[f"{metric}_bucket"] = pd.qcut(
            df[metric], q, labels=bucket_labels, duplicates="drop"
        )

        # Compute proportion inside each x-axis group
        temp = (
            df.groupby([x_axis_var, f"{metric}_bucket"], observed=True)
              .size()
              .reset_index(name="count")
        )

        temp["proportion"] = (
            temp.groupby(x_axis_var)["count"].transform(lambda x: x / x.sum())
        )

        # Wide format for stacked bars
        wide = (
            temp.pivot(index=x_axis_var,
                       columns=f"{metric}_bucket",
                       values="proportion")
                .reindex(x_order)
        )

        # Human-readable label
        clean = pretty_metric(metric.replace("_tminus0","").replace("_growth_3yr",""))

        # Plot stacked bars
        barplot = wide.plot(
            kind="bar",
            stacked=True,
            colormap=quartile_cmap,
            width=0.90,
            ax=ax
        )

        # Capture legend only once
        if legend_handles is None:
            legend_handles = barplot.get_legend_handles_labels()

        # Formatting
        ax.set_title(f"{clean} Quartile Composition", fontsize=13)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Share of Teams")
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        ax.legend().remove()  # remove subplot legends

    # Global legend
    fig.legend(*legend_handles,
               title="Percentile Bucket",
               loc="upper right",
               bbox_to_anchor=(0.98, 0.97))

    # Combined title
    fig.suptitle(f"{suptitle_txt_1}\n{suptitle_txt_2}", fontsize=18)

    fig.tight_layout(rect=[0, 0, 0.92, 0.90])

    save_figure(fig, filename)
    plt.show()

plot_stacked_bar(all_teams_charting_data, "qualified", "qualified_vs_not_level_stacked.png")
plot_stacked_bar(qualified_teams_charting_data, "stage_expanded", "by_stage_level_stacked.png")



# ############################################################
# #  CORRELATION MATRIX FOR NORMALIZED SOCIECONOMIC VARIABLES
# ############################################################

# # ##### Check correlation between socioeconomic indicators for multicollinearity
fig, ax = plt.subplots(figsize=(8, 6))

# 1. Compute correlation matrix
corr = qualified_teams_charting_data[non_growth_metrics].corr()

# 2. Apply pretty_metric() to row/column labels
pretty_names = ["Normailzed " + pretty_metric(col) for col in corr.columns]
corr.index = pretty_names
corr.columns = pretty_names

# 3. Plot heatmap
sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    vmin=-1,
    vmax=1,
    ax=ax,
    fmt=".2f"
)

ax.set_title("Correlation Between Socioeconomic Predictors")
plt.tight_layout()
save_figure(fig, "socio_corr_heatmap.png")   # <-- save here
plt.show()

# ############################################################
# #  ORDINAL LOGISTIC REGRESSION
# ############################################################

# Keep only the row corresponding to the latest stage reached for each (world_cup_year, team)
df_reg = (
    qualified_teams_charting_data[['world_cup_year', 'team', 'stage_expanded_num', 'stage_expanded'] + non_growth_metrics]
    .dropna()
    .sort_values(["world_cup_year", "team", "stage_expanded"])
    .groupby(["world_cup_year", "team"], as_index=False)
    .tail(1)   # keep the row with the max stage_expanded_num
)

# Create a composite score that is the average of the normalized values across all four indicators
df_reg["socio_index"] = df_reg[non_growth_metrics].mean(axis=1)

# Run ordered logistic model
model = OrderedModel(df_reg['stage_expanded'], df_reg[['socio_index']],
                     distr='probit').fit(method = 'bfgs', disp = False)

# Print out model results
coef = model.params['socio_index']
pval = model.pvalues['socio_index']
print("\n=== SES Coefficient ===")
print(f"Coefficient (β): {coef:.4f}")
print(f"P-value: {pval:.4f}")
delta_names = [name for name in model.params.index if '/' in name]
deltas = model.params[delta_names].values
taus = [None for i in range(len(deltas))]
for i in range(len(deltas)):
    if i == 0:
        taus[i] = deltas[i]
    else:
        taus[i] = taus[i-1] + np.exp(deltas[i])
print("\n=== Estimated Thresholds (τ) ===")
for i, t in enumerate(taus, start=1):
    print(f"τ{i}: {t:.4f}")


# Create a series of socio_economic index values
x_vals = np.linspace(df_reg["socio_index"].min(), df_reg["socio_index"].max(), 300)
exog = pd.DataFrame({"socio_index": x_vals})
# predict the probability of reaching each stage (same number of rows as x_val, one column for each stage)
pred = model.predict(exog=exog, which="prob")

# Probability of reaching AT LEAST a given milestone
p_knockout = pred.iloc[:, 1:].sum(axis=1)   # stage ≥ Round of 16
p_qf = pred.iloc[:, 2:].sum(axis=1)   # stage ≥ QF
p_sf = pred.iloc[:, 3:].sum(axis=1)   # stage ≥ SF
p_final = pred.iloc[:, 4:].sum(axis=1)   # stage ≥ Final
p_winner = pred.iloc[:, 5:] # Stage = winner


# === Plot === #
fig = plt.figure(figsize=(12,7))

plt.plot(x_vals, p_knockout, linewidth=3, label="Reach Knockouts (R16+)", color="#1f77b4")
plt.plot(x_vals, p_qf, linewidth=3, label="Reach Quarter-finals", color="#2ca02c")
plt.plot(x_vals, p_sf, linewidth=3, label="Reach Semi-finals", color="#d62728")
plt.plot(x_vals, p_final,linewidth=3, label="Reach Final", color="#9467bd")
plt.plot(x_vals, p_winner, linewidth = 3, label = "Win", color = "#b58900")

plt.title("Probability of Reaching World Cup Milestones\nas Normalized Socioeconomic Index Increases (Probit Model)", fontsize=18)
plt.xlabel("Normalized Socioeconomic Index (Composite of GDP, HDI, Life Expectancy, Education)")
plt.ylabel("Predicted Probability (Probit)")
plt.ylim(0,1)
plt.grid(alpha=0.25)
plt.legend(title="Milestone", fontsize=11)

plt.show()

save_figure(fig, "milestone_prob_vs_composite_score.png")
