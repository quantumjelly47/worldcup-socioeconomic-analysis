import pandas as pd


def normalize_country_names(df, column="country"):
    """
    Standardize country names so they align with World Cup naming conventions.
    """
    replacements = {
        "Bahamas, The": "Bahamas",
        "Bolivia (Plurinational State of)": "Bolivia",
        "China PR": "China",
        "China PR": "China PR",
        "CA'te d'Ivoire": "Ivory Coast",
        "C�te d'Ivoire": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Czechia": "Czech Republic",
        "Egypt, Arab Rep.": "Egypt",
        "Eswatini (Kingdom of)": "Eswatini",
        "Hong Kong SAR, China": "Hong Kong, China (SAR)",
        "Iran (Islamic Republic of)": "Iran",
        "Iran, Islamic Rep.": "Iran",
        "Ireland": "Republic of Ireland",
        "Korea (Democratic People's Rep. of)": "North Korea",
        "Korea, Dem. Rep.": "North Korea",
        "Korea, Dem. People's Rep.": "North Korea",
        "Korea (Republic of)": "South Korea",
        "Korea, Rep.": "South Korea",
        "Lao PDR": "Lao People's Democratic Republic",
        "Micronesia, Fed. Sts.": "Micronesia (Federated States of)",
        "Moldova": "Moldova (Republic of)",
        "Slovak Republic": "Slovakia",
        "Russian Federation": "Russia",
        "Sao Tome and Principe": "Sao Tome and Principe",
        "SA�o TomAc and PrA-ncipe": "Sao Tome and Principe",
        "Somalia, Fed. Rep.": "Somalia",
        "Tanzania": "Tanzania (United Republic of)",
        "TA�rkiye": "Turkey",
        "Turkiye": "Turkey",
        "Türkiye": "Turkey",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Venezuela, RB": "Venezuela",
        "Viet Nam": "Vietnam",
    }

    df[column] = df[column].replace(replacements)
    return df


def expand_united_kingdom(df, column="country"):
    """
    Replace 'United Kingdom' with England, Scotland, and Wales copies.
    """
    if "United Kingdom" not in df[column].values:
        return df

    uk_rows = df[df[column] == "United Kingdom"].copy()
    subteams = ["England", "Scotland", "Wales"]

    expanded_rows = []
    for sub in subteams:
        temp = uk_rows.copy()
        temp[column] = sub
        expanded_rows.append(temp)

    df = pd.concat([df, *expanded_rows], ignore_index=True)
    return df[df[column] != "United Kingdom"].reset_index(drop=True)


def expand_legacy_countries(df, column="country"):
    """
    Duplicate modern teams for historical entrants lacking standalone records.
    """
    legacy_map = {
        "Serbia": ["FR Yugoslavia", "Serbia and Montenegro"],
    }

    new_rows = []
    for source, targets in legacy_map.items():
        if source not in df[column].values:
            continue

        source_rows = df[df[column] == source]
        for target in targets:
            if target in df[column].values:
                continue
            temp = source_rows.copy()
            temp[column] = target
            new_rows.append(temp)

    if new_rows:
        df = pd.concat([df, *new_rows], ignore_index=True)

    return df


WORLD_CUP_COUNTRIES = [
    'Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Belgium', 'Bolivia',
    'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cameroon', 'Canada', 'Chile',
    'China', 'Colombia', 'Costa Rica', 'Croatia', 'Czech Republic', 'Denmark',
    'Ecuador', 'Egypt', 'England', 'FR Yugoslavia', 'France', 'Germany', 'Ghana',
    'Greece', 'Honduras', 'Iceland', 'Iran', 'Italy', 'Ivory Coast', 'Jamaica',
    'Japan', 'Mexico', 'Morocco', 'Netherlands', 'New Zealand', 'Nigeria',
    'North Korea', 'Norway', 'Panama', 'Paraguay', 'Peru', 'Poland', 'Portugal',
    'Qatar', 'Republic of Ireland', 'Romania', 'Russia', 'Saudi Arabia', 'Scotland',
    'Senegal', 'Serbia', 'Serbia and Montenegro', 'Slovakia', 'Slovenia',
    'South Africa', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Togo',
    'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Ukraine', 'United States',
    'Uruguay', 'Wales'
]


def filter_world_cup_countries(df, column="country", include_non_world_cup=False):
    """
    Filter dataset to keep only World Cup countries (optionally return the rest).
    """
    normalized_df = normalize_country_names(df, column)
    expanded_df = expand_united_kingdom(normalized_df, column)
    expanded_df = expand_legacy_countries(expanded_df, column)

    mask = expanded_df[column].isin(WORLD_CUP_COUNTRIES)
    world_cup_df = expanded_df[mask].reset_index(drop=True)

    if include_non_world_cup:
        non_world_cup_df = expanded_df[~mask].reset_index(drop=True)
        return world_cup_df, non_world_cup_df

    return world_cup_df


def check_country_coverage(df, column="country"):
    """
    Print which World Cup entrants are missing from the provided dataset.
    """
    df_countries = set(df[column].unique())
    missing = [c for c in WORLD_CUP_COUNTRIES if c not in df_countries]

    if not missing:
        print("All World Cup countries present.")
    else:
        print(f"Missing {len(missing)} countries:\n", missing)
