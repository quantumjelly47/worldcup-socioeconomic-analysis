## **World Cup Performance & Socioeconomic Analysis (1994–2022)**

DS-GA 1007 Group Project · NYU Center for Data Science · Dhairya Dhamani, Estee Ng, George Buck, Sachin Sastri

---

## **Project Overview**

This project examines how **economic development**, **demographic structure**, and **world-class football talent** help explain national team performance at FIFA World Cups from **1994 to 2022**.

Our pipeline produces a unified dataset combining:

* Match-level & team-level World Cup performance
* FIFA pre-tournament rankings (aligned via `merge_asof`)
* Share of players in Europe’s Top-5 leagues
* Socioeconomic indicators:
  GDP per capita, HDI, life expectancy, mean schooling years
* Demographic metrics:
  population median age & age skew
* Normalized (z-score) metrics
* Lagged socioeconomic values at t, t–1, t–2, t–3

The final merged file supports all regression, simulation, and exploratory analyses.

---

## **Repository Layout**

A full project skeleton is provided below. Key directories include:

* **data/raw_data_files** – Original World Cup, FIFA ranking, squad, and socioeconomic files
* **data/created_datasets** – All cleaned, standardized, merged datasets
* **scripts/world_cup** – World Cup cleaning, feature engineering, ranking alignment
* **scripts/socioeconomic** – GDP/HDI/life/schooling/pop cleaning + normalization
* **scripts/utils** – Shared helpers for name cleaning, interpolation, normalization
* **scripts/q1**, **scripts/q2**, **scripts/monte_carlo** – Analysis notebooks

---

# **Repository Structure**

```
.
├── README.md
├── .gitignore
│
├── data/
│   ├── created_datasets/
│   │   ├── socioeconomic/
│   │   │   ├── gdp_world_cup.csv
│   │   │   ├── hdi_world_cup.csv
│   │   │   ├── life_expectancy_world_cup.csv
│   │   │   ├── schooling_world_cup.csv
│   │   │   ├── world_cup_pop_long.csv
│   │   │
│   │   ├── world_cup/
│   │       ├── big5_leagues.xlsx
│   │       ├── match_level_data.csv
│   │       ├── performance_by_world_cup_and_team.csv
│   │       ├── prop_big5.csv
│   │       ├── merge_country_ranks_before_wc.csv
│   │       ├── merge_wc_with_socioeconomic.csv
│
│   ├── raw_data_files/
│       ├── Socioeconomic Data/
│       │   ├── GDP per Capita Data.csv
│       │   ├── HDI 1990-2023.csv
│       │   ├── Population Data.xlsx
│       │
│       ├── World Cup Data/
│           ├── Rankings/
│           │   ├── fifa_mens_rank.csv
│           │   ├── fifa_ranking-2024-06-20.csv
│           │
│           ├── Wiki Squads/
│           │   ├── WC1994/
│           │   ├── WC1998/
│           │   ├── WC2002/
│           │   ├── WC2006/
│           │   ├── WC2010/
│           │   ├── WC2014/
│           │   ├── WC2018/
│           │   ├── WC2022/
│           │   └── respective squad CSVs
│           │
│           ├── World+Cup/
│               ├── 2022_world_cup_groups.csv
│               ├── *_matches.csv
│               ├── *_squads.csv
│               ├── data_dictionary.csv
│               ├── international_matches.csv
│               ├── matches_1930_2022.csv
│               ├── world_cups.csv
│               └── world_cup_matches.csv
│
├── scripts/
│   ├── monte_carlo/
│   │   └── monte_carlo_final.ipynb
│   │
│   ├── q1/
│   │   └── q1_analysis.ipynb
│   │
│   ├── q2/
│   │   └── eda-rq2.ipynb
│   │
│   ├── socioeconomic/
│   │   ├── clean_gdp.py
│   │   ├── clean_hdi.py
│   │   ├── clean_life.py
│   │   ├── clean_school.py
│   │   ├── pop_data_cleaning.ipynb
│   │   └── socioeconomic_eda.ipynb
│   │
│   ├── utils/
│   │   ├── filter_countries.py
│   │   ├── socio_constants.py
│   │   └── socio_helpers.py
│   │
│   ├── world_cup/
│       ├── import_world_cup.py
│       ├── merge_top5_and_rankings.py
│       ├── merge_world_cup_socioeconomic.py
│       ├── scrape_wiki.ipynb
│       └── squad_data_prep.ipynb
```


## **Data Processing Pipeline**

### **1. World Cup Processing**

Handled in `scripts/world_cup/`:

* Convert match-level data → team-level
* Compute wins, goals for/against, shootout results, max stage
* Clean/standardize country names (legacy splits, UK expansion)
* Merge Top-5 league share (`prop_big5.csv`)
* Align rankings with tournament start dates using `merge_asof`
* Output key files:

  * `match_level_data.csv`
  * `performance_by_world_cup_and_team.csv`
  * `merge_country_ranks_before_wc.csv`

### **2. Socioeconomic Processing**

From `scripts/socioeconomic/`:

* Reshape long datasets (GDP, HDI, life, schooling)
* Standardize names with shared country-mapping utilities
* Drop aggregate regions
* Fill missing values:

  * within-country interpolation
  * regional backfill
  * global backfill (region = “GLOBAL”)
* Z-score normalization for all metrics
* Filter to year ≥ 1990
* Output files such as:

  * `gdp_world_cup.csv`
  * `hdi_world_cup.csv`
  * `world_cup_pop_long.csv`

### **3. Merging Everything**

`merge_world_cup_socioeconomic.py` does the final join:

* Attach socioeconomic + population metrics to each WC team
* Add normalized and lagged versions
* Save the main analysis dataset:
  **`merge_wc_with_socioeconomic.csv`**

---

## **How to Reproduce**

Run from the **repo root**.

### **Step 1 — Build World Cup Data**

```bash
python scripts/world_cup/import_world_cup.py
scrape_wiki.ipynb
squad_data_prep.ipynb
python scripts/world_cup/merge_top5_and_rankings.py
```

### **Step 2 — Build Socioeconomic Panels**

```bash
py -B -m scripts.socioeconomic.clean_gdp
py -B -m scripts.socioeconomic.clean_hdi
py -B -m scripts.socioeconomic.clean_life
py -B -m scripts.socioeconomic.clean_school
pop_data_cleaning.ipynb
```

### **Step 3 — Final Merge**

```bash
python scripts/world_cup/merge_world_cup_socioeconomic.py
```

### **Step 4 — Optional Analysis**

Run:

* Monte Carlo simulation: `scripts/monte_carlo/monte_carlo_final.ipynb`
* RQ1 analysis: `scripts/q1/q1_analysis.ipynb`
* RQ2 EDA: `scripts/q2/eda-rq2.ipynb`
* Interactive socioeconomic map: `scripts/socioeconomic/socioeconomic_eda.ipynb`

---

## **Dependencies**

Main packages:

* pandas, numpy, pathlib
* plotly, ipywidgets, altair (visualization)
* BeautifulSoup, requests (for squad scraping)
* jupyter / ipykernel

---

## **Important Assumptions**

* Many countries and tournament formats changed before **1994**
  → analysis restricted to *modern WC era* (1994–2022)
* HDI and several indicators don’t exist before **1990**
* England, Wales, Scotland **inherit UK socioeconomic values**
* Socioeconomic normalization is performed **globally before filtering**
* Rankings aligned using nearest pre-tournament release
  (`merge_asof` on WC start dates)
