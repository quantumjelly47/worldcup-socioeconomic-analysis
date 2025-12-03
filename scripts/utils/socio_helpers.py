from pathlib import Path
import pandas as pd

from scripts.utils.filter_countries import filter_world_cup_countries
from scripts.utils.socio_constants import DEVELOPED_COUNTRIES, NON_COUNTRIES

BASE_DIR = Path(__file__).resolve().parents[2]


def load_csv(path, encoding="latin-1", skiprows=None):
    """Wrapper around pandas.read_csv with defaults for encoding/skiprows."""
    return pd.read_csv(path, encoding=encoding, skiprows=skiprows)


def assign_developed(df, country_col="country"):
    """Mark developed countries' region as 'Developed'.

    Args:
        df: Input DataFrame.
        country_col: Column containing country names.

    Returns:
        pd.DataFrame: Updated with region set to Developed where applicable.
    """
    df = df.copy()
    df.loc[df[country_col].isin(DEVELOPED_COUNTRIES), "region"] = "Developed"
    return df


def drop_aggregates(df, country_col="country"):
    """Drop aggregate/region rows based on NON_COUNTRIES list."""
    return df[~df[country_col].isin(NON_COUNTRIES)].copy()


def interpolate_by_group(df, value_col, group_col="country"):
    """Interpolate a value column within groups when at least two points exist.

    Args:
        df: Input DataFrame.
        value_col: Column to interpolate.
        group_col: Grouping column (default country).

    Returns:
        pd.DataFrame: With interpolated values.
    """
    def _interp(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    df = df.copy()
    df[value_col] = df.groupby(group_col, group_keys=False)[value_col].apply(_interp)
    return df


def fill_by_region(df, value_col):
    """Fill missing values by region-year mean."""
    df = df.copy()
    df[value_col] = df.groupby(["region", "year"])[value_col].transform(lambda x: x.fillna(x.mean()))
    return df


def filter_wc(df, country_col="country", include_non=False):
    """Filter DataFrame to World Cup countries (optionally return non-WC subset)."""
    return filter_world_cup_countries(df, country_col, include_non_world_cup=include_non)


def save_world_cup(df, name, output_dir=None):
    """
    Save a World Cup subset to the socioeconomic created_datasets directory.
    If output_dir is None, it defaults to <repo_root>/data/created_datasets/socioeconomic.

    Args:
        df: DataFrame to save.
        name: Base filename prefix.
        output_dir: Optional output directory.

    Returns:
        Path: Path to the saved CSV.
    """
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "created_datasets" / "socioeconomic"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_world_cup.csv"
    df.to_csv(path, index=False)
    return path
