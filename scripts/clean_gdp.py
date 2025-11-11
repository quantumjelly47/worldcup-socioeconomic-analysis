import pandas as pd

def clean_gdp(path="data/raw_data_files/Socioeconomic Data/GDP per Capita Data.csv", verbose=True):
    """Load, clean, and reshape the GDP per Capita dataset (1960–2024)."""

    # Load dataset
    gdp = pd.read_csv(path, encoding="latin-1", skiprows=4)

    # Keep only relevant columns
    year_cols = [c for c in gdp.columns if c.isdigit()]
    gdp = gdp[["Country Name", "Country Code"] + year_cols]

    # Reshape from wide to long format
    gdp_long = gdp.melt(
        id_vars=["Country Name", "Country Code"],
        var_name="year",
        value_name="gdp_per_capita"
    )

    # Standardize column names
    gdp_long.rename(columns={
        "Country Name": "country",
        "Country Code": "iso3"
    }, inplace=True)

    # Convert data types
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_per_capita"] = pd.to_numeric(gdp_long["gdp_per_capita"], errors="coerce")

    # Interpolate missing GDP values within each country
    def interpolate_gdp(series):
        if series.notna().sum() > 1:
            return series.interpolate(limit_direction="both")
        return series

    gdp_long["gdp_per_capita"] = (
        gdp_long.groupby("country", group_keys=False)["gdp_per_capita"]
        .apply(interpolate_gdp)
    )

    # Drop countries or years still missing GDP after interpolation
    gdp_long = gdp_long.dropna(subset=["gdp_per_capita"])

    # Remove regional, income, and aggregate entities
    non_country_keywords = [
        "World", "income", "IDA", "IBRD", "OECD", "Euro area", "European Union",
        "Caribbean", "Sub-Saharan Africa", "Europe & Central Asia",
        "East Asia & Pacific", "Middle East", "North America", "Latin America",
        "Arab World", "South Asia", "Fragile", "Small states", "Pre-demographic",
        "Post-demographic", "demographic dividend", "Not classified",
        "Heavily indebted", "Least developed", "Low & middle",
        "High income", "Lower middle", "Upper middle"
    ]

    mask = gdp_long["country"].apply(
        lambda x: not any(keyword.lower() in x.lower() for keyword in non_country_keywords)
    )
    gdp_long = gdp_long[mask].copy()

    if verbose:
        print("\nSample cleaned rows:")
        print(gdp_long.head(10))
        print("Final shape:", gdp_long.shape)

    return gdp_long


if __name__ == "__main__":
    clean_gdp()
