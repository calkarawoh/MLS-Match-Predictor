"""Dark Sports Analytics dashboard for the MLS Match Predictor.

Run from the MLS_Match_Predictor folder:
    py dashboard_old.py

Requires: customtkinter, pandas, joblib and existing features.py.
The supplied matches.csv contains SYNTHETIC demonstration results.
"""

from pathlib import Path
import tkinter.messagebox as messagebox

import customtkinter as ctk
import joblib
import pandas as pd

from features import FEATURE_COLS, build_upcoming, latest_team_form, to_team_games

# Paths are relative to this file, not to your current terminal directory.
PROJECT = Path(__file__).resolve().parent
HISTORY_FILE = PROJECT / "data" / "matches.csv"
MODEL_FILE = PROJECT / "outputs" / "model.joblib"

# Color palette
BG = "#090F17"
PANEL = "#141E2B"
PANEL_ALT = "#1D2A39"
BORDER = "#2B3C4D"
WHITE = "#F3F7FA"
MUTED = "#93A6B8"
GREEN = "#36D39A"
YELLOW = "#FFC857"
RED = "#FF6B78"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


def get_prediction(bundle, history, home, away):
    """Return probabilities by class ID; 0=away, 1=draw, 2=home."""
    if home == away:
        raise ValueError("The home and away teams must be different.")

    fixture = pd.DataFrame([{
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "home_team": home,
        "away_team": away,
    }])
    # Match the feature order and rolling window used when training.
    frame = build_upcoming(fixture, history, window=bundle["window"])
    model = bundle["model"]
    probabilities = model.predict_proba(frame[FEATURE_COLS])[0]
    return {int(cls): float(p) for cls, p in zip(model.classes_, probabilities)}


def recent_results(team_games, team, n=5):
    """The team's last n recorded results, newest first."""
    games = team_games[team_games["team"] == team].tail(n).iloc[::-1]
    out = []
    for _, game in games.iterrows():
        if game["gf"] > game["ga"]:
            out.append("W")
        elif game["gf"] == game["ga"]:
            out.append("D")
        else:
            out.append("L")
    return out


class MLSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MLS Predictor | Dark Sports Analytics")
        self.geometry("1060x780")
        self.minsize(790, 620)
        self.configure(fg_color=BG)

        if not MODEL_FILE.exists() or not HISTORY_FILE.exists():
            messagebox.showerror(
                "Missing project files",
                "Could not find outputs/model.joblib or data/matches.csv.\n\n"
                "Run 'py train.py --test-season 2024' and try again.",
            )
            self.after(0, self.destroy)
            return

        try:
            self.bundle = joblib.load(MODEL_FILE)  # Load only your OWN local model.
            self.history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
            self.history = self.history.sort_values("date").reset_index(drop=True)
            self.teams = sorted(
                set(self.history["home_team"]) | set(self.history["away_team"])
            )
            self.team_form = latest_team_form(
                self.history, window=self.bundle["window"]
            )