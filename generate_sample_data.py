"""Generate a synthetic MLS-style dataset so the pipeline runs without real data.

Replace data/matches.csv with real MLS data (same columns) when you have it.
"""
import numpy as np
from pathlib import Path
import pandas as pd

TEAMS = [
    "Atlanta United", "Austin FC", "CF Montreal", "Charlotte FC", "Chicago Fire",
    "FC Cincinnati", "Colorado Rapids", "Columbus Crew", "D.C. United", "FC Dallas",
    "Houston Dynamo", "Inter Miami", "LA Galaxy", "LAFC", "Minnesota United",
    "Nashville SC", "New England Revolution", "NYCFC", "New York Red Bulls",
    "Orlando City", "Philadelphia Union", "Portland Timbers", "Real Salt Lake",
    "San Jose Earthquakes", "Seattle Sounders", "Sporting KC", "St. Louis City",
    "Toronto FC", "Vancouver Whitecaps",
]


def generate(seasons=(2019, 2020, 2021, 2022, 2023, 2024), seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 0.3) for t in TEAMS}
    rows = []
    for season in seasons:
        # strengths drift a little each season
        for t in TEAMS:
            strength[t] += rng.normal(0, 0.1)
        date = pd.Timestamp(f"{season}-03-01")
        for _ in range(34):  # 34 matchdays
            order = list(rng.permutation(TEAMS))
            if len(order) % 2:
                order.pop()
            for i in range(0, len(order), 2):
                home, away = order[i], order[i + 1]
                diff = strength[home] - strength[away]
                hg = rng.poisson(max(0.2, np.exp(0.25 + 0.2 + diff / 2)))
                ag = rng.poisson(max(0.2, np.exp(0.25 - diff / 2)))
                poss_h = float(np.clip(50 + 8 * diff + rng.normal(0, 5), 30, 70))
                rows.append({
                    "date": date,
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_shots": max(0, int(rng.normal(12 + 3 * diff + hg, 3))),
                    "away_shots": max(0, int(rng.normal(10 - 3 * diff + ag, 3))),
                    "home_possession": round(poss_h, 1),
                    "away_possession": round(100 - poss_h, 1),
                })
            date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/matches.csv", index=False)
    print(f"Wrote {len(df)} matches to data/matches.csv")
