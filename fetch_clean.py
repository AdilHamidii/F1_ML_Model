import pandas as pd
import fastf1
import logging

logging.getLogger("fastf1").setLevel(logging.ERROR)
fastf1.Cache.enable_cache("fastf1_cache")

SEASON_ROUNDS = {
    2022: 22,
    2023: 22,
    2024: 24,
    2025: 9
}

rows = []

for year, max_round in SEASON_ROUNDS.items():
    for rnd in range(1, max_round + 1):
        try:
            quali = fastf1.get_session(year, rnd, "Q")
            race  = fastf1.get_session(year, rnd, "R")

            quali.load()
            race.load()

            if quali.results is None or race.results is None:
                continue

            q = quali.results[[
                "DriverNumber", "Abbreviation", "TeamName",
                "Position", "Q1", "Q2", "Q3"
            ]].copy()

            r = race.results[[
                "DriverNumber", "Position"
            ]].copy()

            q = q.rename(columns={"Position": "QualiPosition"})
            r = r.rename(columns={"Position": "RacePosition"})

            df = pd.merge(q, r, on="DriverNumber", how="left")

            df["Year"] = year
            df["Round"] = rnd

            rows.append(df)

            print(f"Loaded {year} round {rnd}")

        except Exception as e:
            print(f"Skipping {year} round {rnd}: {e}")

if not rows:
    raise RuntimeError("No data could be loaded")

df = pd.concat(rows, ignore_index=True)


for col in ["Q1", "Q2", "Q3"]:
    df[col] = df[col].dt.total_seconds()

df.to_csv("training_data.csv", index=False)
