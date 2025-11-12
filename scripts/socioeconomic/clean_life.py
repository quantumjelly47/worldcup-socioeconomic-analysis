import pandas as pd

def clean_life_expectancy(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """Load, clean, and reshape the UNDP Life Expectancy dataset (1990–2023)."""

    # Load dataset
    undp = pd.read_csv(path, encoding="latin-1")
    undp.columns = undp.columns.str.strip()

    # Assign missing regions for developed countries
    developed_countries = [
        'Andorra', 'Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'Croatia',
        'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
        'Greece', 'Hong Kong, China (SAR)', 'Hungary', 'Iceland', 'Ireland', 'Israel',
        'Italy', 'Japan', 'Korea (Republic of)', 'Latvia', 'Liechtenstein', 'Lithuania',
        'Luxembourg', 'Malta', 'Netherlands', 'New Zealand', 'Norway', 'Poland',
        'Portugal', 'Romania', 'Russian Federation', 'San Marino', 'Slovakia',
        'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'United States'
    ]
    undp.loc[undp["country"].isin(developed_countries), "region"] = "Developed"

    # Select relevant columns for life expectancy
    le_cols = ['iso3', 'country', 'hdicode', 'region', 'hdi_rank_2023'] + [
        c for c in undp.columns if c.startswith('le_') and not c.startswith(('le_f', 'le_m'))
    ]
    le_subset = undp[le_cols].loc[:, ~undp[le_cols].columns.duplicated()]

    # Reshape from wide to long format
    le_long = le_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="life_expectancy"
    )
    le_long["year"] = le_long["year"].str.replace("le_", "").astype(int)
    le_long["life_expectancy"] = pd.to_numeric(le_long["life_expectancy"], errors="coerce")

    # Remove regional/aggregate rows
    non_countries = [
        "Very high human development", "High human development",
        "Medium human development", "Low human development",
        "Arab States", "East Asia and the Pacific", "Europe and Central Asia",
        "Latin America and the Caribbean", "South Asia",
        "Sub-Saharan Africa", "World"
    ]
    le_long = le_long[~le_long["country"].isin(non_countries)].copy()

    # Interpolation helper
    def interpolate_safely(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    # Interpolate missing life expectancy by country
    le_long["life_expectancy"] = (
        le_long.groupby("country", group_keys=False)["life_expectancy"]
        .apply(interpolate_safely)
    )

    # Fill missing life expectancy with regional averages
    le_long["life_expectancy"] = (
        le_long.groupby(["region", "year"])["life_expectancy"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Final fallback for any remaining missing values
    le_long["region"] = le_long["region"].fillna("GLOBAL")
    le_long["life_expectancy"] = (
        le_long.groupby(["region", "year"])["life_expectancy"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Optional diagnostics
    if verbose:
        missing = le_long[le_long["life_expectancy"].isna()]["country"].unique()
        print(f"\nRemaining missing Life Expectancy values: {le_long['life_expectancy'].isna().sum()}")
        if len(missing) > 0:
            print("Countries still missing data:", missing)
        else:
            print("All Life Expectancy values filled successfully.")
        print(f"Final dataset shape: {le_long.shape}")

    return le_long


if __name__ == "__main__":
    le_long = clean_life_expectancy()
    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample Life Expectancy values:")
    print(le_long[le_long["country"].isin(sample)].head(10))
