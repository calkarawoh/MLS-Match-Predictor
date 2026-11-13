"""Dark Sports Analytics dashboard for the MLS Match Predictor.

Run from the MLS_Match_Predictor folder:
    py dashboard.py

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
BG = "#0a0a0a"
PANEL = "#16021c"
PANEL_ALT = "#1D2A39"
BORDER = "#8cfaad"
WHITE = "#F3F7FA"
MUTED = "#93A6B8"
GREEN = "#9cf000"
YELLOW = "#ffdd00"
RED = "#ff0000"

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
        self.title("Cal's MLS Predictor | Dark Sports Analytics")
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
            self.team_games = to_team_games(self.history)
        except Exception as error:
            messagebox.showerror("Could not load data", str(error))
            self.after(0, self.destroy)
            return

        self.home_var = ctk.StringVar(
            value="Seattle Sounders" if "Seattle Sounders" in self.teams else self.teams[0]
        )
        self.away_var = ctk.StringVar(
            value="Inter Miami" if "Inter Miami" in self.teams else self.teams[-1]
        )
        self.model_name = self.bundle.get("name", "Trained model")
        self.window = int(self.bundle["window"])
        self.probability_widgets = {}

        self.build_ui()
        self.update_form_panels()

    def panel(self, parent, **kwargs):
        return ctk.CTkFrame(
            parent, fg_color=PANEL, corner_radius=17,
            border_width=1, border_color=BORDER, **kwargs
        )

    def heading(self, parent, text, size=18, color=WHITE):
        return ctk.CTkLabel(
            parent, text=text, text_color=color,
            font=ctk.CTkFont(family="Segoe UI", size=size, weight="bold")
        )

    def build_ui(self):
        # Scroll if window is small so information never disappears below it.
        content = ctk.CTkScrollableFrame(self, fg_color=BG)
        content.pack(fill="both", expand=True, padx=22, pady=16)

        title_row = ctk.CTkFrame(content, fg_color="transparent")
        title_row.pack(fill="x", pady=(0, 12))
        self.heading(title_row, "⚽  MLS MATCH PREDICTOR", 28).pack(side="left")
        ctk.CTkLabel(
            title_row, text="DARK SPORTS ANALYTICS",
            font=("Segoe UI", 12, "bold"),
            text_color=GREEN, fg_color=PANEL_ALT, corner_radius=12,
            padx=16, pady=9,
        ).pack(side="right")

        ctk.CTkLabel(
            content,
            text=f"{self.model_name}  •  Last {self.window} recorded matches  •  Sample data: synthetic (not live MLS)",
            text_color=MUTED, font=("Segoe UI", 12)
        ).pack(anchor="w", pady=(0, 18))

        selector = self.panel(content)
        selector.pack(fill="x", pady=(0, 16))
        self.heading(selector, "MATCHUP", 17).pack(anchor="w", padx=20, pady=(16, 8))

        selector_row = ctk.CTkFrame(selector, fg_color="transparent")
        selector_row.pack(fill="x", padx=20, pady=(0, 14))
        selector_row.grid_columnconfigure(0, weight=1)
        selector_row.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(selector_row, text="HOME TEAM", text_color=MUTED).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(selector_row, text="AWAY TEAM", text_color=MUTED).grid(row=0, column=2, sticky="w")

        option_style = dict(
            values=self.teams, fg_color=PANEL_ALT,
            button_color="#314657", button_hover_color="#456176",
            text_color=WHITE, dropdown_fg_color=PANEL_ALT,
            dropdown_text_color=WHITE, dropdown_hover_color="#34495C",
            height=43, font=("Segoe UI", 14),
        )
        ctk.CTkOptionMenu(
            selector_row, variable=self.home_var, command=self.on_team_changed,
            **option_style,
        ).grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.heading(selector_row, "VS", 18, GREEN).grid(row=1, column=1, padx=18)
        ctk.CTkOptionMenu(
            selector_row, variable=self.away_var, command=self.on_team_changed,
            **option_style,
        ).grid(row=1, column=2, sticky="ew", pady=(5, 0))

        ctk.CTkButton(
            selector, text="PREDICT MATCH  →", command=self.predict,
            fg_color=GREEN, hover_color="#28AD80", text_color="#082019",
            font=("Segoe UI", 15, "bold"), height=48, corner_radius=10,
        ).pack(fill="x", padx=20, pady=(0, 19))

        results = self.panel(content)
        results.pack(fill="x", pady=(0, 16))
        self.heading(results, "OUTCOME PROBABILITIES", 17).pack(anchor="w", padx=20, pady=(16, 4))
        self.prediction_label = ctk.CTkLabel(
            results, text="Choose two teams and click Predict Match",
            font=("Segoe UI", 13), text_color=MUTED,
        )
        self.prediction_label.pack(anchor="w", padx=20, pady=(0, 11))

        for code, name, color in [
            (2, "HOME WIN", GREEN),
            (1, "DRAW", YELLOW),
            (0, "AWAY WIN", RED),
        ]:
            row = ctk.CTkFrame(results, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                row, text=name, width=115, anchor="w", text_color=WHITE,
                font=("Segoe UI", 12, "bold"),
            ).grid(row=0, column=0, sticky="w")
            bar = ctk.CTkProgressBar(
                row, height=15, progress_color=color,
                fg_color="#314152", corner_radius=8,
            )
            bar.grid(row=0, column=1, sticky="ew", padx=(8, 15))
            bar.set(0)
            percent = ctk.CTkLabel(
                row, text="—", width=72, anchor="e", text_color=color,
                font=("Segoe UI", 16, "bold"),
            )
            percent.grid(row=0, column=2)
            self.probability_widgets[code] = (bar, percent)

        ctk.CTkLabel(
            results,
            text="These are model estimates, not guaranteed outcomes or betting odds.",
            text_color=MUTED, font=("Segoe UI", 11),
        ).pack(anchor="w", padx=20, pady=(12, 17))

        self.heading(content, f"RECENT FORM  •  LAST {self.window} MATCHES", 17).pack(
            anchor="w", pady=(4, 12)
        )
        form_row = ctk.CTkFrame(content, fg_color="transparent")
        form_row.pack(fill="x", pady=(0, 15))
        form_row.grid_columnconfigure(0, weight=1)
        form_row.grid_columnconfigure(1, weight=1)
        self.home_panel = self.panel(form_row)
        self.home_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.away_panel = self.panel(form_row)
        self.away_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(
            content,
            text="DATA NOTICE: bundled matches.csv is synthetic (2019–2024). "
                 "Real forecasts require real historical data and model retraining. "
                 "Lineups, injuries and live fixtures are not included.",
            wraplength=930, justify="left", text_color=YELLOW,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(0, 12))

    def on_team_changed(self, _selection=None):
        self.prediction_label.configure(text="Click Predict Match to calculate new probabilities.", text_color=MUTED)
        for bar, label in self.probability_widgets.values():
            bar.set(0)
            label.configure(text="—")
        self.update_form_panels()

    def update_form_panels(self):
        for frame, team in [
            (self.home_panel, self.home_var.get()),
            (self.away_panel, self.away_var.get()),
        ]:
            for child in frame.winfo_children():
                child.destroy()

            self.heading(frame, team, 16).pack(anchor="w", padx=17, pady=(16, 4))
            ctk.CTkLabel(
                frame, text="Latest results (newest first)",
                text_color=MUTED, font=("Segoe UI", 11),
            ).pack(anchor="w", padx=17)

            form = ctk.CTkFrame(frame, fg_color="transparent")
            form.pack(anchor="w", padx=17, pady=(9, 12))
            for outcome in recent_results(self.team_games, team, self.window):
                color = {"W": GREEN, "D": YELLOW, "L": RED}[outcome]
                ctk.CTkLabel(
                    form, text=outcome, text_color=BG, fg_color=color,
                    width=30, height=30, corner_radius=15,
                    font=("Segoe UI", 13, "bold"),
                ).pack(side="left", padx=(0, 7))

            stats = self.team_form.loc[team]
            for title, key, fmt in [
                ("Goals scored / match", "gf_form", ".2f"),
                ("Goals conceded / match", "ga_form", ".2f"),
                ("Shots / match", "shots_form", ".1f"),
                ("Possession", "poss_form", ".1f"),
                ("Points / match", "pts_form", ".2f"),
            ]:
                line = ctk.CTkFrame(frame, fg_color="transparent")
                line.pack(fill="x", padx=17, pady=2)
                ctk.CTkLabel(line, text=title, text_color=MUTED).pack(side="left")
                number = f"{float(stats[key]):{fmt}}" + ("%" if key == "poss_form" else "")
                ctk.CTkLabel(
                    line, text=number, text_color=WHITE,
                    font=("Segoe UI", 12, "bold"),
                ).pack(side="right")
            ctk.CTkLabel(frame, text="", height=10).pack()

    def predict(self):
        home, away = self.home_var.get(), self.away_var.get()
        try:
            probabilities = get_prediction(self.bundle, self.history, home, away)
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error), parent=self)
            return

        names = {2: f"{home} win", 1: "Draw", 0: f"{away} win"}
        best = max(probabilities, key=probabilities.get)
        self.prediction_label.configure(
            text=f"Top prediction: {names[best]}  ({100 * probabilities[best]:.1f}%)",
            text_color=GREEN,
        )
        for code, (bar, label) in self.probability_widgets.items():
            probability = probabilities.get(code, 0.0)
            bar.set(probability)  # CTkProgressBar takes values from 0.0 to 1.0.
            label.configure(text=f"{100 * probability:.1f}%")


if __name__ == "__main__":
    app = MLSApp()
    app.mainloop()
