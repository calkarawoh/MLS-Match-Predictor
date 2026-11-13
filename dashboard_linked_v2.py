r"""Dark Sports Analytics: original dashboard design connected to real MLS data.

Place this file beside real_model.py and mls_live_data.py.
Required local files:
  data/mls_real_matches.csv
  outputs/real_mls_model.joblib

Run:
  .\.venv\Scripts\python.exe dashboard_linked.py
"""
from pathlib import Path
import tkinter.messagebox as messagebox
from datetime import datetime

import customtkinter as ctk
import pandas as pd
import joblib

import real_model
import mls_live_data

PROJECT = Path(__file__).resolve().parent
HISTORY_FILE = PROJECT / "data" / "mls_real_matches.csv"
MODEL_FILE = PROJECT / "outputs" / "real_mls_model.joblib"
SNAPSHOT_FILE = PROJECT / "data" / "mls_live_snapshot.json"

# User's exact Dark Sports Analytics palette
BG = "#0a0a0a"
PANEL = "#16021c"
PANEL_ALT = "#1D2A39"
BORDER = "#8cfaad"
WHITE = "#F3F7FA"
MUTED = "#93A6B8"
GREEN = "#bec2b8"
YELLOW = "#ffdd00"
RED = "#ff0000"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


def get_recent(history, team, n=5):
    rows = []
    for game in history.to_dict("records"):
        if game["home_team"] == team:
            gf, ga = int(game["home_goals"]), int(game["away_goals"])
        elif game["away_team"] == team:
            gf, ga = int(game["away_goals"]), int(game["home_goals"])
        else:
            continue
        result = "W" if gf > ga else "D" if gf == ga else "L"
        rows.append((pd.to_datetime(game["date"], utc=True), result, gf, ga))
    rows.sort(key=lambda x: x[0])
    return list(reversed(rows[-n:]))


def team_summary(history, team, n=5):
    recent = get_recent(history, team, n)
    if not recent:
        raise ValueError(f"No completed match history found for {team}.")
    gf = sum(x[2] for x in recent) / len(recent)
    ga = sum(x[3] for x in recent) / len(recent)
    points = sum(3 if x[1] == "W" else 1 if x[1] == "D" else 0 for x in recent) / len(recent)
    return recent, gf, ga, points


class MLSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dark Sports Analytics | MLS Predictor")
        self.geometry("1060x850")
        self.minsize(820, 680)
        self.configure(fg_color=BG)

        try:
            if not HISTORY_FILE.exists():
                raise FileNotFoundError(f"Missing {HISTORY_FILE}. Run mls_live_data.py sync --start 2024")
            if not MODEL_FILE.exists():
                raise FileNotFoundError(f"Missing {MODEL_FILE}. Run real_model.py train")
            self.history = real_model.load_matches()
            self.bundle = joblib.load(MODEL_FILE)
            self.histories = real_model.team_histories(self.history)
            self.teams = sorted(self.histories)
            if len(self.teams) < 2:
                raise RuntimeError("The real MLS dataset does not contain enough teams.")
        except Exception as exc:
            messagebox.showerror("Could not load real MLS model/data", str(exc))
            self.after(100, self.destroy)
            return

        home_default = next((t for t in self.teams if "Seattle" in t), self.teams[0])
        away_default = next((t for t in self.teams if "Inter Miami" in t), self.teams[-1])
        self.home_var = ctk.StringVar(value=home_default)
        self.away_var = ctk.StringVar(value=away_default)
        self.probability_widgets = {}
        self.build_ui()
        self.update_form_panels()
        self.load_fixtures()

    def panel(self, parent, **kwargs):
        return ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=17,
                            border_width=1, border_color=BORDER, **kwargs)

    def heading(self, parent, text, size=18, color=WHITE):
        return ctk.CTkLabel(parent, text=text, text_color=color,
                            font=ctk.CTkFont(family="Segoe UI", size=size, weight="bold"))

    def build_ui(self):
        content = ctk.CTkScrollableFrame(self, fg_color=BG)
        content.pack(fill="both", expand=True, padx=22, pady=16)

        title_row = ctk.CTkFrame(content, fg_color="transparent")
        title_row.pack(fill="x", pady=(0, 12))
        self.heading(title_row, "⚽  MLS MATCH PREDICTOR", 28).pack(side="left")
        ctk.CTkLabel(title_row, text="DARK SPORTS ANALYTICS",
                     font=("Segoe UI", 12, "bold"), text_color=GREEN,
                     fg_color=PANEL_ALT, corner_radius=12, padx=16, pady=9).pack(side="right")

        ctk.CTkLabel(
            content,
            text=f"REAL MLS RESULTS  •  {len(self.history):,} completed matches  •  Random Forest",
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
        option_style = dict(values=self.teams, fg_color=PANEL_ALT,
                            button_color="#314657", button_hover_color="#456176",
                            text_color=WHITE, dropdown_fg_color=PANEL_ALT,
                            dropdown_text_color=WHITE, dropdown_hover_color="#34495C",
                            height=43, font=("Segoe UI", 14))
        ctk.CTkOptionMenu(selector_row, variable=self.home_var, command=self.on_team_changed,
                          **option_style).grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.heading(selector_row, "VS", 18, GREEN).grid(row=1, column=1, padx=18)
        ctk.CTkOptionMenu(selector_row, variable=self.away_var, command=self.on_team_changed,
                          **option_style).grid(row=1, column=2, sticky="ew", pady=(5, 0))
        ctk.CTkButton(selector, text="PREDICT MATCH  →", command=self.predict,
                      fg_color=GREEN, hover_color="#28AD80", text_color="#082019",
                      font=("Segoe UI", 15, "bold"), height=48, corner_radius=10
                      ).pack(fill="x", padx=20, pady=(0, 19))

        results = self.panel(content)
        results.pack(fill="x", pady=(0, 16))
        self.heading(results, "OUTCOME PROBABILITIES", 17).pack(anchor="w", padx=20, pady=(16, 4))
        self.prediction_label = ctk.CTkLabel(results, text="Choose two teams and click Predict Match",
                                             font=("Segoe UI", 13), text_color=MUTED)
        self.prediction_label.pack(anchor="w", padx=20, pady=(0, 11))
        for code, name, color in [(2, "HOME WIN", GREEN), (1, "DRAW", YELLOW), (0, "AWAY WIN", RED)]:
            row = ctk.CTkFrame(results, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=name, width=115, anchor="w", text_color=WHITE,
                         font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
            bar = ctk.CTkProgressBar(row, height=15, progress_color=color,
                                     fg_color="#314152", corner_radius=8)
            bar.grid(row=0, column=1, sticky="ew", padx=(8, 15))
            bar.set(0)
            label = ctk.CTkLabel(row, text="—", width=72, anchor="e", text_color=color,
                                 font=("Segoe UI", 16, "bold"))
            label.grid(row=0, column=2)
            self.probability_widgets[code] = (bar, label)
        ctk.CTkLabel(results, text="Model estimates, not guaranteed outcomes or betting odds.",
                     text_color=MUTED, font=("Segoe UI", 11)).pack(anchor="w", padx=20, pady=(12, 17))

        self.heading(content, "RECENT FORM  •  LAST 5 MATCHES", 17).pack(anchor="w", pady=(4, 12))
        form_row = ctk.CTkFrame(content, fg_color="transparent")
        form_row.pack(fill="x", pady=(0, 15))
        form_row.grid_columnconfigure(0, weight=1)
        form_row.grid_columnconfigure(1, weight=1)
        self.home_panel = self.panel(form_row)
        self.home_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.away_panel = self.panel(form_row)
        self.away_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        fixtures = self.panel(content)
        fixtures.pack(fill="x", pady=(0, 16))
        fixture_header = ctk.CTkFrame(fixtures, fg_color="transparent")
        fixture_header.pack(fill="x", padx=17, pady=(13, 7))
        self.heading(fixture_header, "MLS FIXTURES & RECENT SCORES", 16).pack(side="left")
        ctk.CTkButton(fixture_header, text="REFRESH", width=100, height=30,
                      fg_color=GREEN, hover_color="#28AD80", text_color=BG,
                      command=self.refresh_fixtures).pack(side="right")
        self.fixture_text = ctk.CTkTextbox(fixtures, height=145, fg_color=PANEL_ALT,
                                           text_color=WHITE, border_color=BORDER,
                                           border_width=1, corner_radius=10)
        self.fixture_text.pack(fill="x", padx=17, pady=(0, 15))
        self.fixture_text.configure(state="disabled")

        ctk.CTkLabel(content,
                     text="DATA: Real completed MLS scores from the project's ESPN feed. Shots, possession, lineups and injuries are not included in this goals-only model.",
                     wraplength=930, justify="left", text_color=YELLOW,
                     font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 12))

    def on_team_changed(self, _selection=None):
        self.prediction_label.configure(text="Click Predict Match to calculate new probabilities.", text_color=MUTED)
        for bar, label in self.probability_widgets.values():
            bar.set(0)
            label.configure(text="—")
        self.update_form_panels()

    def update_form_panels(self):
        for frame, team in [(self.home_panel, self.home_var.get()), (self.away_panel, self.away_var.get())]:
            for child in frame.winfo_children():
                child.destroy()
            self.heading(frame, team, 16).pack(anchor="w", padx=17, pady=(16, 4))
            ctk.CTkLabel(frame, text="Latest results (newest first)", text_color=MUTED,
                         font=("Segoe UI", 11)).pack(anchor="w", padx=17)
            form = ctk.CTkFrame(frame, fg_color="transparent")
            form.pack(anchor="w", padx=17, pady=(9, 12))
            recent, gf, ga, points = team_summary(self.history, team, 5)
            for _, outcome, _, _ in recent:
                color = {"W": GREEN, "D": YELLOW, "L": RED}[outcome]
                ctk.CTkLabel(form, text=outcome, text_color=BG, fg_color=color,
                             width=30, height=30, corner_radius=15,
                             font=("Segoe UI", 13, "bold")).pack(side="left", padx=(0, 7))
            for title, value, fmt in [
                ("Goals scored / match", gf, ".2f"),
                ("Goals conceded / match", ga, ".2f"),
                ("Points / match", points, ".2f"),
            ]:
                line = ctk.CTkFrame(frame, fg_color="transparent")
                line.pack(fill="x", padx=17, pady=2)
                ctk.CTkLabel(line, text=title, text_color=MUTED).pack(side="left")
                ctk.CTkLabel(line, text=f"{value:{fmt}}", text_color=WHITE,
                             font=("Segoe UI", 12, "bold")).pack(side="right")
            ctk.CTkLabel(frame, text="", height=10).pack()

    def predict(self):
        home, away = self.home_var.get(), self.away_var.get()
        try:
            probabilities = real_model.predict(home, away, self.bundle)
            # Model class IDs: 0 away win, 1 draw, 2 home win.
            names = {2: f"{home} win", 1: "Draw", 0: f"{away} win"}
            best = max(probabilities, key=probabilities.get)
            self.prediction_label.configure(
                text=f"Top prediction: {names[best]}  ({100 * probabilities[best]:.1f}%)",
                text_color=GREEN)
            for code, (bar, label) in self.probability_widgets.items():
                p = probabilities.get(code, 0.0)
                bar.set(p)
                label.configure(text=f"{100 * p:.1f}%")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc), parent=self)

    def load_fixtures(self):
        snapshot = mls_live_data.load_snapshot()
        events = snapshot.get("events", [])
        lines = []
        for event in events[:18]:
            state = event.get("state", "scheduled")
            home, away = event.get("home_team", "?"), event.get("away_team", "?")
            if state == "scheduled":
                score = "VS"
            elif event.get("home_goals") is not None and event.get("away_goals") is not None:
                score = f'{event["home_goals"]} - {event["away_goals"]}'
            else:
                score = "LIVE"
            date = str(event.get("date", ""))[:16].replace("T", " ")
            lines.append(f'{date}   {home}   {score}   {away}   [{state.upper()}]')
        if not lines:
            lines = ["No saved fixture snapshot yet. Click REFRESH to fetch current fixtures."]
        self.fixture_text.configure(state="normal")
        self.fixture_text.delete("1.0", "end")
        self.fixture_text.insert("1.0", "\n".join(lines))
        self.fixture_text.configure(state="disabled")

    def refresh_fixtures(self):
        try:
            events = mls_live_data.refresh_live()
            self.load_fixtures()
            messagebox.showinfo("Fixtures refreshed", f"Retrieved {len(events)} MLS fixtures/results.", parent=self)
        except Exception as exc:
            messagebox.showerror("Fixture refresh failed", str(exc), parent=self)


if __name__ == "__main__":
    app = MLSApp()
    app.mainloop()
