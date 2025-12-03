from pathlib import Path

import pandas as pd

# Paths to inputs/outputs (anchored at repo root)
BASE_DIR = Path(__file__).resolve().parents[2]
WORLD_CUP_DIR = BASE_DIR / "data" / "created_datasets" / "world_cup"
SOCIO_DIR = BASE_DIR / "data" / "created_datasets" / "socioeconomic"

WC_PATH = WORLD_CUP_DIR / "merge_country_ranks_before_wc.csv"
GDP_PATH = SOCIO_DIR / "gdp_world_cup.csv"
HDI_PATH = SOCIO_DIR / "hdi_world_cup.csv"
LIFE_PATH = SOCIO_DIR / "life_expectancy_world_cup.csv"
SCHOOL_PATH = SOCIO_DIR / "schooling_world_cup.csv"
POP_PATH = SOCIO_DIR / "world_cup_pop_long.csv"
OUTPUT_PATH = WORLD_CUP_DIR / "merge_wc_with_socioeconomic.csv"


def load_data():
    """Load WC performance with rankings plus each socioeconomic panel.

    Returns:
        tuple[pd.DataFrame, ...]: wc, gdp, hdi, life, school, pop DataFrames.
    """
    wc = pd.read_csv(WC_PATH)
    gdp = pd.read_csv(GDP_PATH)
    hdi = pd.read_csv(HDI_PATH)
    life = pd.read_csv(LIFE_PATH)
    school = pd.read_csv(SCHOOL_PATH)
    pop = pd.read_csv(POP_PATH)
    return wc, gdp, hdi, life, school, pop


def attach_metric(
    base_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Add metric columns for WC year and prior 3 years.

    Args:
        base_df: DataFrame with columns ['team', 'world_cup_year'].
        metric_df: DataFrame with columns ['country', 'year', value_col].
        value_col: Column name in metric_df to pull.
        prefix: Prefix to use for new columns.

    Returns:
        pd.DataFrame: base_df with new columns f\"{prefix}_tminus0..3\".
    """
    lookup = (
        metric_df[["country", "year", value_col]]
        .set_index(["country", "year"])
        .squeeze()
    )

    result = base_df.copy()
    for lag in range(0, 4):  # 0=current WC year, 1=prev year, etc.
        idx = pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "country": result["team"],
                    "year": result["world_cup_year"] - lag,
                }
            )
        )
        result[f"{prefix}_tminus{lag}"] = lookup.reindex(idx).to_numpy()

    return result


def main():
    """
    Merge WC performance/rankings with socioeconomic metrics (GDP, normalized GDP,
    HDI, life expectancy, schooling, population) at WC year and t-1/2/3, and save to CSV.
    """
    wc_df, gdp_df, hdi_df, life_df, school_df, pop_df = load_data()

    merged = wc_df.copy()
    merged = attach_metric(merged, gdp_df, "gdp_per_capita", "gdp_per_capita")
    merged = attach_metric(merged, gdp_df, "norm_gdp_per_capita", "norm_gdp_per_capita")
    merged = attach_metric(merged, hdi_df, "hdi", "hdi")
    merged = attach_metric(merged, hdi_df, "norm_hdi", "norm_hdi")
    merged = attach_metric(merged, life_df, "life_expectancy", "life_expectancy")
    merged = attach_metric(merged, life_df, "norm_life_expectancy", "norm_life_expectancy")
    merged = attach_metric(
        merged, school_df, "mean_school_years", "mean_school_years"
    )
    merged = attach_metric(
        merged, school_df, "norm_mean_school_years", "norm_mean_school_years"
    )
    pop_metrics = (
        pop_df.rename(
            columns={
                "Country": "country",
                "Year": "year",
                "Median Age": "median_age",
            }
        )
        .drop_duplicates(subset=["country", "year"])
    )
    merged = attach_metric(merged, pop_metrics, "median_age", "median_age")
    merged = attach_metric(merged, pop_metrics, "norm_median_age", "norm_median_age")
    merged = attach_metric(merged, pop_metrics, "skew", "skew")
    merged = attach_metric(merged, pop_metrics, "norm_skew", "norm_skew")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Merged shape: {merged.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
