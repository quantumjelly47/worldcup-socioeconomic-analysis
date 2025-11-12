import pandas as pd

def clean_schooling(path="data/raw_data_files/Socioeconomic Data/HDI 1990-2023.csv", verbose=True):
    """Load, clean, and reshape the UNDP Mean Years of Schooling dataset (1990–2023)."""

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
    undp["region"] = undp["region"].replace("Other Countries or Territories", "Developed")

    # Select relevant columns for mean years of schooling
    mys_cols = ['iso3', 'country', 'hdicode', 'region', 'hdi_rank_2023'] + [
        c for c in undp.columns if c.startswith('mys_') and not c.startswith(('mys_f', 'mys_m'))
    ]
    mys_subset = undp[mys_cols].loc[:, ~undp[mys_cols].columns.duplicated()]

    # Reshape from wide to long format
    mys_long = mys_subset.melt(
        id_vars=["iso3", "country", "hdicode", "region", "hdi_rank_2023"],
        var_name="year",
        value_name="mean_school_years"
    )
    mys_long["year"] = mys_long["year"].str.replace("mys_", "").astype(int)
    mys_long["mean_school_years"] = pd.to_numeric(mys_long["mean_school_years"], errors="coerce")

    # Clean up country names
    mys_long["country"] = mys_long["country"].str.strip()

    # Remove regional/aggregate rows
    non_countries = [
        "Very high human development", "High human development",
        "Medium human development", "Low human development",
        "Arab States", "East Asia and the Pacific", "Europe and Central Asia",
        "Latin America and the Caribbean", "South Asia",
        "Sub-Saharan Africa", "World"
    ]
    mys_long = mys_long[~mys_long["country"].isin(non_countries)].copy()

    # Interpolation helper
    def interpolate_safely(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    # Interpolate missing MYS by country
    mys_long["mean_school_years"] = (
        mys_long.groupby("country", group_keys=False)["mean_school_years"]
        .apply(interpolate_safely)
    )

    # Fill missing MYS with regional averages
    mys_long["mean_school_years"] = (
        mys_long.groupby(["region", "year"])["mean_school_years"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Final fallback for any remaining missing values
    mys_long["region"] = mys_long["region"].fillna("GLOBAL")
    mys_long["mean_school_years"] = (
        mys_long.groupby(["region", "year"])["mean_school_years"]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Diagnostics
    if verbose:
        missing = mys_long[mys_long["mean_school_years"].isna()]["country"].unique()
        print(f"\nRemaining missing MYS values: {mys_long['mean_school_years'].isna().sum()}")
        if len(missing) > 0:
            print("Countries still missing data:", missing)
        else:
            print("All MYS values filled successfully.")
        print(f"Final dataset shape: {mys_long.shape}")

    return mys_long


if __name__ == "__main__":
    mys_long = clean_schooling()
    sample = ["United States", "France", "India", "Brazil", "Nigeria"]
    print("\nSample Mean Years of Schooling values:")
    print(mys_long[mys_long["country"].isin(sample)].head(10))
