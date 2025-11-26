from pathlib import Path

import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
WORLD_CUP_DIR = BASE_DIR / "data" / "created_datasets" / "world_cup"
SOCIO_DIR = BASE_DIR / "data" / "created_datasets" / "socioeconomic"

WC_PATH = WORLD_CUP_DIR / "merge_country_ranks_before_wc.csv"
GDP_PATH = SOCIO_DIR / "gdp_world_cup.csv"
HDI_PATH = SOCIO_DIR / "hdi_world_cup.csv"
LIFE_PATH = SOCIO_DIR / "life_expectancy_world_cup.csv"
SCHOOL_PATH = SOCIO_DIR / "schooling_world_cup.csv"
OUTPUT_PATH = WORLD_CUP_DIR / "merge_wc_with_socioeconomic.csv"


def load_data():
    """Load rankings/world cup data and socioeconomic panels."""
    wc = pd.read_csv(WC_PATH)
    gdp = pd.read_csv(GDP_PATH)
    hdi = pd.read_csv(HDI_PATH)
    life = pd.read_csv(LIFE_PATH)
    school = pd.read_csv(SCHOOL_PATH)
    return wc, gdp, hdi, life, school


def attach_metric(
    base_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Add metric for WC year and prior 3 years."""
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
    wc_df, gdp_df, hdi_df, life_df, school_df = load_data()

    merged = wc_df.copy()
    merged = attach_metric(merged, gdp_df, "gdp_per_capita", "gdp_per_capita")
    merged = attach_metric(merged, gdp_df, "norm_gdp_per_capita", "norm_gdp_per_capita")
    merged = attach_metric(merged, hdi_df, "hdi", "hdi")
    merged = attach_metric(merged, life_df, "life_expectancy", "life_expectancy")
    merged = attach_metric(
        merged, school_df, "mean_school_years", "mean_school_years"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Merged shape: {merged.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
