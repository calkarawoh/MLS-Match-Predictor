"""Dark Sports Analytics: original UI linked to real MLS matches, model and fixtures.

Run from the project folder with your virtual environment: dashboard_linked.py
Requires existing mls_live_data.py, real_model.py, trained real model, and real data CSV.
"""

from pathlib import Path
import threading
from datetime import datetime, timezone
import tkinter.messagebox as messagebox

import customtkinter as ctk
import joblib

from mls_live_data import load_snapshot, refresh_live
from real_model import MODEL, CSV, load_matches, predict as real_predict, team_histories, team_stats

# Paths are relative to this file, not to your current terminal directory.
PROJECT = Path(__file__).resolve().parent
HISTORY_FILE = CSV
MODEL_FILE = MODEL

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


def recent_results(history, team, n=5):
    """Real results, most recent first; never fabricate missing games."""
    return [g["form"] for g in history.get(team, [])[-n:][::-1]]


class MLSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dark Sports Analytics | Real MLS")
        self.geometry("1060x870")
        self.minsize(790, 620)
        self.configure(fg_color=BG)

        if not MODEL_FILE.exists() or not HISTORY_FILE.exists():
            messagebox.showerror(
                "Missing project files",
                "Real MLS data/model not found.\n\nRun these in PowerShell:\n"
                ".\\.venv\\Scripts\\python.exe mls_live_data.py sync --start 2024\n"
                ".\\.venv\\Scripts\\python.exe real_model.py train",
            )
            self.after(0, self.destroy)
            return

        try:
            self.bundle = joblib.load(MODEL_FILE)  # Load only your own locally trained model.
            self.matches = load_matches()
            self.history = team_histories(self.matches)
            self.teams = sorted(team for team, games in self.history.items() if len(games) >= 3)
            if len(self.teams) < 2:
                raise ValueError("Not enough teams in the real MLS dataset")
        except Exception as error:
            messagebox.showerror("Could not load data", str(error))
            self.after(0, self.destroy)
            return

        self.home_var = ctk.StringVar(
            value="Seattle Sounders FC" if "Seattle Sounders FC" in self.teams else self.teams[0]
        )
        self.away_var = ctk.StringVar(
            value="Inter Miami CF" if "Inter Miami CF" in self.teams else self.teams[-1]
        )
        if self.home_var.get() == self.away_var.get():
            self.away_var.set(next(t for t in self.teams if t != self.home_var.get()))
        self.model_name = "Random Forest · real completed MLS scores"
        self.window = 5
        self.refreshing = False
        self.probability_widgets = {}

        self.build_ui()
        self.update_form_panels()
        self.render_fixture_snapshot()
        self.after(300, self.refresh_fixtures)
        self.after(120000, self.auto_refresh)

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
        self.heading(title_row, "⚽  DARK SPORTS ANALYTICS", 28).pack(side="left")
        ctk.CTkLabel(
            title_row, text="REAL MLS • LIVE FIXTURES",
            font=("Segoe UI", 12, "bold"),
            text_color=GREEN, fg_color=PANEL_ALT, corner_radius=12,
            padx=16, pady=9,
        ).pack(side="right")

        ctk.CTkLabel(
            content,
            text=f"{self.model_name}  •  Last {self.window} real completed matches  •  Pre-match predictions",
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
            button_color=GREEN, button_hover_color="#baff4a",
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
            fg_color=GREEN, hover_color="#baff4a", text_color=BG,
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
                fg_color=PANEL_ALT, corner_radius=8,
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

        fixtures = self.panel(content)
        fixtures.pack(fill="x", pady=(0, 14))
        fixture_top = ctk.CTkFrame(fixtures, fg_color="transparent")
        fixture_top.pack(fill="x", padx=17, pady=(13, 4))
        self.heading(fixture_top, "MLS LIVE SCORES & UPCOMING FIXTURES", 16).pack(side="left")
        self.refresh_button = ctk.CTkButton(
            fixture_top, text="REFRESH", width=100, command=self.refresh_fixtures,
            fg_color=PANEL_ALT, hover_color="#304356", text_color=GREEN,
        )
        self.refresh_button.pack(side="right")
        self.fixture_status = ctk.CTkLabel(fixtures, text="Checking scores...", text_color=MUTED)
        self.fixture_status.pack(anchor="w", padx=17, pady=(0, 5))
        self.fixture_list = ctk.CTkTextbox(
            fixtures, height=130, fg_color=PANEL_ALT, text_color=WHITE,
            font=("Consolas", 11), wrap="word",
        )
        self.fixture_list.pack(fill="x", padx=17, pady=(0, 16))
        self.fixture_list.configure(state="disabled")
        ctk.CTkLabel(
            content,
            text="DATA: real MLS completed scores via unofficial ESPN feed. "
                 "No verified shots, possession, injuries, or lineups. "
                 "Live scoreboard refresh does not automatically retrain the prediction model.",
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
            for outcome in recent_results(self.history, team, self.window):
                color = {"W": GREEN, "D": YELLOW, "L": RED}[outcome]
                ctk.CTkLabel(
                    form, text=outcome, text_color=BG, fg_color=color,
                    width=30, height=30, corner_radius=15,
                    font=("Segoe UI", 13, "bold"),
                ).pack(side="left", padx=(0, 7))

            stats = team_stats(self.history.get(team, []), self.window)
            for title, key, fmt in [
                ("Goals scored / match", "gf", ".2f"),
                ("Goals conceded / match", "ga", ".2f"),
                ("Points / match", "points", ".2f"),
            ]:
                line = ctk.CTkFrame(frame, fg_color="transparent")
                line.pack(fill="x", padx=17, pady=2)
                ctk.CTkLabel(line, text=title, text_color=MUTED).pack(side="left")
                number = f"{float(stats[key]):{fmt}}"
                ctk.CTkLabel(
                    line, text=number, text_color=WHITE,
                    font=("Segoe UI", 12, "bold"),
                ).pack(side="right")
            ctk.CTkLabel(frame, text="", height=10).pack()

    def predict(self):
        home, away = self.home_var.get(), self.away_var.get()
        try:
            probabilities = real_predict(home, away, self.bundle)
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

    def render_fixture_snapshot(self):
        """Display only scores for games genuinely live or completed."""
        try:
            snapshot = load_snapshot()
            updated = snapshot.get("updated_utc") or "never"
            self.fixture_status.configure(text=f"Last scoreboard check (UTC): {updated}", text_color=MUTED)
            lines = []
            for event in snapshot.get("events", [])[:25]:
                date = str(event.get("date", ""))[:16].replace("T", " ")
                state = event.get("state", "scheduled")
                home, away = event.get("home_team", "?"), event.get("away_team", "?")
                hg, ag = event.get("home_goals"), event.get("away_goals")
                score = f"{hg}–{ag}" if state in ("live", "final") and hg is not None and ag is not None else "VS"
                lines.append(f"{date} UTC [{state.upper()}]  {home}  {score}  {away}")
            self.fixture_list.configure(state="normal")
            self.fixture_list.delete("1.0", "end")
            self.fixture_list.insert("1.0", "\n".join(lines) if lines else "No fixtures available in the current feed window.")
            self.fixture_list.configure(state="disabled")
        except Exception as exc:
            self.fixture_status.configure(text=f"Fixture cache unavailable: {exc}", text_color=RED)

    def refresh_fixtures(self):
        if self.refreshing:
            return
        self.refreshing = True
        self.refresh_button.configure(state="disabled")
        self.fixture_status.configure(text="Checking the MLS fixture feed...", text_color=MUTED)

        def worker():
            try:
                refresh_live()
                self.after(0, lambda: self.finish_refresh(None))
            except Exception as exc:
                error = str(exc)
                self.after(0, lambda e=error: self.finish_refresh(e))

        threading.Thread(target=worker, daemon=True).start()

    def finish_refresh(self, error):
        self.refreshing = False
        self.refresh_button.configure(state="normal")
        if error:
            self.fixture_status.configure(text=f"Scoreboard refresh failed: {error}", text_color=RED)
        else:
            # New completed games update form, but the model remains the previously trained version.
            try:
                self.history = team_histories(load_matches())
                self.update_form_panels()
            except Exception:
                pass
            self.render_fixture_snapshot()

    def auto_refresh(self):
        self.refresh_fixtures()
        self.after(120000, self.auto_refresh)


if __name__ == "__main__":
    app = MLSApp()
    app.mainloop()
