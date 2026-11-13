"""Feature engineering. Every feature for a match uses ONLY games played before it."""
import numpy as np
import pandas as pd

RESULT_MAP = {"L": 0, "D": 1, "W": 2}   # from the HOME team's perspective
RESULT_NAMES = {0: "Away win", 1: "Draw", 2: "Home win"}
STATS = ["gf", "ga", "shots", "shots_against", "poss", "pts"]
FEATURE_COLS = (
    [f"home_{s}_form" for s in STATS]
    + [f"away_{s}_form" for s in STATS]
    + [f"diff_{s}_form" for s in STATS]
    + ["home_games_played", "away_games_played", "home_advantage"]
)


def _result(hg, ag):
    return np.where(hg > ag, "W", np.where(hg == ag, "D", "L"))


def to_team_games(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, so rolling stats are easy to compute."""
    home = pd.DataFrame({
        "match_id": matches.index, "date": matches["date"], "team": matches["home_team"],
        "gf": matches["home_goals"], "ga": matches["away_goals"],
        "shots": matches["home_shots"], "shots_against": matches["away_shots"],
        "poss": matches["home_possession"],
    })
    away = pd.DataFrame({
        "match_id": matches.index, "date": matches["date"], "team": matches["away_team"],
        "gf": matches["away_goals"], "ga": matches["home_goals"],
        "shots": matches["away_shots"], "shots_against": matches["home_shots"],
        "poss": matches["away_possession"],
    })
    tg = pd.concat([home, away]).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    tg["pts"] = np.select([tg.gf > tg.ga, tg.gf == tg.ga], [3, 1], default=0)
    return tg


def add_rolling_form(tg: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling mean of the previous `window` games (shift(1) prevents leakage)."""
    g = tg.groupby("team")
    for s in STATS:
        tg[f"{s}_form"] = g[s].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    tg["games_played"] = g.cumcount()
    return tg


def build_dataset(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Return matches with features (and a `result` target for played games)."""
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    tg = add_rolling_form(to_team_games(matches), window)
    form_cols = [f"{s}_form" for s in STATS] + ["games_played"]

    home = tg.merge(matches[["home_team"]].assign(match_id=matches.index),
                    left_on=["match_id", "team"], right_on=["match_id", "home_team"])
    away = tg.merge(matches[["away_team"]].assign(match_id=matches.index),
                    left_on=["match_id", "team"], right_on=["match_id", "away_team"])

    df = matches.copy()
    for s in form_cols:
        df[f"home_{s}"] = home.set_index("match_id")[s]
        df[f"away_{s}"] = away.set_index("match_id")[s]
    for s in STATS:
        df[f"diff_{s}_form"] = df[f"home_{s}_form"] - df[f"away_{s}_form"]
    df["home_advantage"] = 1  # constant for now; replace with e.g. travel distance later

    df["result"] = pd.Series(_result(df.home_goals, df.away_goals)).map(RESULT_MAP)
    # Drop each team's first game (no history yet)
    return df[(df.home_games_played > 0) & (df.away_games_played > 0)].reset_index(drop=True)


def latest_team_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Each team's form going INTO its next game (uses all played matches)."""
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    tg = to_team_games(matches.sort_values("date").reset_index(drop=True))
    rows = {}
    for team, g in tg.groupby("team"):
        last = g.tail(window)
        rows[team] = {**{f"{s}_form": last[s].mean() for s in STATS}, "games_played": len(g)}
    return pd.DataFrame(rows).T


def build_upcoming(fixtures: pd.DataFrame, history: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Features for fixtures that haven't been played (columns: date, home_team, away_team)."""
    form = latest_team_form(history, window)
    df = fixtures.copy()
    for s in STATS:
        df[f"home_{s}_form"] = df.home_team.map(form[f"{s}_form"])
        df[f"away_{s}_form"] = df.away_team.map(form[f"{s}_form"])
        df[f"diff_{s}_form"] = df[f"home_{s}_form"] - df[f"away_{s}_form"]
    df["home_games_played"] = df.home_team.map(form["games_played"])
    df["away_games_played"] = df.away_team.map(form["games_played"])
    df["home_advantage"] = 1
    unknown = df[df[FEATURE_COLS].isna().any(axis=1)]
    if len(unknown):
        raise ValueError(f"No history for teams in: {unknown[['home_team','away_team']].values.tolist()}")
    return df
