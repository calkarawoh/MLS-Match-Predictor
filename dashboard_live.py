"""Neon MLS dashboard: REAL match history, model predictions, refreshing fixtures.
Run: .venv\\Scripts\\python.exe dashboard_live.py
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter.messagebox as messagebox

import customtkinter as ctk
import joblib

from mls_live_data import load_snapshot, refresh_live, HISTORY
from real_model import MODEL, load_matches, predict, team_histories, team_stats

BG, CARD, RAISED = '#070B0D', '#10191A', '#182B2A'
NEON, CYAN, YELLOW, PINK = '#39FF14', '#00E5FF', '#FFD60A', '#FF3864'
TEXT, MUTED = '#E8FFF4', '#93AAA6'
ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('green')


class LiveMLS(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('MLS • NEON LIVE ANALYTICS')
        self.geometry('1080x820')
        self.minsize(850, 630)
        self.configure(fg_color=BG)
        self.bundle = None
        self.history = {}
        self.snapshot = load_snapshot()
        self.busy = False
        self.home = ctk.StringVar(value='')
        self.away = ctk.StringVar(value='')
        self._load_model()
        self._build()
        self._render_data()
        # Refresh the score feed automatically every 120 seconds while the app is open.
        self.after(1000, self.refresh)
        self.after(120000, self.auto_refresh)

    def _load_model(self):
        if MODEL.exists() and HISTORY.exists():
            try:
                self.bundle = joblib.load(MODEL)  # Trust only your locally trained file.
                self.history = team_histories(load_matches())
            except Exception as exc:
                messagebox.showwarning('Real MLS data problem', str(exc))
        teams = sorted(t for t, games in self.history.items() if len(games) >= 3)
        if teams:
            self.home.set('Seattle Sounders FC' if 'Seattle Sounders FC' in teams else teams[0])
            self.away.set('Inter Miami CF' if 'Inter Miami CF' in teams else teams[-1])
            if self.home.get() == self.away.get() and len(teams) > 1:
                self.away.set(teams[1])

    def label(self, parent, value, size=14, color=TEXT, bold=False):
        return ctk.CTkLabel(parent, text=value, font=('Segoe UI', size, 'bold' if bold else 'normal'), text_color=color)

    def frame(self, parent):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=1, border_color='#25443B')

    def _build(self):
        body = ctk.CTkScrollableFrame(self, fg_color=BG)
        body.pack(fill='both', expand=True, padx=22, pady=17)
        top = ctk.CTkFrame(body, fg_color='transparent')
        top.pack(fill='x', pady=(0, 12))
        self.label(top, '⚽  MLS NEON LIVE ANALYTICS', 26, NEON, True).pack(side='left')
        ctk.CTkButton(top, text='⟳  REFRESH SCORES', command=self.refresh, fg_color=RAISED,
                      hover_color='#245C36', text_color=NEON, height=37).pack(side='right')
        self.status = self.label(body, 'Initializing MLS feed...', 12, MUTED)
        self.status.pack(anchor='w', pady=(0, 15))

        fixtures = self.frame(body)
        fixtures.pack(fill='x', pady=(0, 16))
        self.label(fixtures, 'LIVE SCORES & UPCOMING MLS FIXTURES', 17, TEXT, True).pack(anchor='w', padx=18, pady=(14, 5))
        self.fixture_text = ctk.CTkTextbox(fixtures, height=145, fg_color=RAISED,
                                           text_color=TEXT, font=('Consolas', 12), wrap='word')
        self.fixture_text.pack(fill='x', padx=18, pady=(3, 16))
        self.fixture_text.configure(state='disabled')

        match = self.frame(body)
        match.pack(fill='x', pady=(0, 16))
        self.label(match, 'PREDICT AN UPCOMING MATCH', 17, TEXT, True).pack(anchor='w', padx=18, pady=(14, 8))
        pick = ctk.CTkFrame(match, fg_color='transparent')
        pick.pack(fill='x', padx=18, pady=(0, 13))
        pick.grid_columnconfigure((0, 2), weight=1)
        self.label(pick, 'HOME', 11, MUTED).grid(row=0, column=0, sticky='w')
        self.label(pick, 'AWAY', 11, MUTED).grid(row=0, column=2, sticky='w')
        style = dict(fg_color=RAISED, button_color='#215E34', button_hover_color='#2E7F45',
                     dropdown_fg_color=RAISED, dropdown_hover_color='#245C36',
                     text_color=TEXT, dropdown_text_color=TEXT, height=40)
        self.home_menu = ctk.CTkOptionMenu(pick, variable=self.home, values=['Load historical data'], **style)
        self.home_menu.grid(row=1, column=0, sticky='ew')
        self.label(pick, 'VS', 17, CYAN, True).grid(row=1, column=1, padx=18)
        self.away_menu = ctk.CTkOptionMenu(pick, variable=self.away, values=['Load historical data'], **style)
        self.away_menu.grid(row=1, column=2, sticky='ew')
        self.predict_btn = ctk.CTkButton(match, text='PREDICT MATCH  →', command=self.do_predict,
                                         fg_color=NEON, hover_color='#2CCB13', text_color=BG,
                                         height=45, font=('Segoe UI', 14, 'bold'))
        self.predict_btn.pack(fill='x', padx=18, pady=(0, 14))
        self.summary = self.label(match, 'Predictions require a model trained on real MLS results.', 12, YELLOW)
        self.summary.pack(anchor='w', padx=18, pady=(0, 15))

        outcome = self.frame(body)
        outcome.pack(fill='x', pady=(0, 16))
        self.label(outcome, 'MODEL PROBABILITIES (PRE-MATCH)', 17, TEXT, True).pack(anchor='w', padx=18, pady=(14, 12))
        self.bars = {}
        for label, result, color in [('HOME WIN', 2, NEON), ('DRAW', 1, YELLOW), ('AWAY WIN', 0, PINK)]:
            r = ctk.CTkFrame(outcome, fg_color='transparent')
            r.pack(fill='x', padx=18, pady=6)
            r.grid_columnconfigure(1, weight=1)
            self.label(r, label, 12, TEXT, True).grid(row=0, column=0, sticky='w', padx=(0, 12))
            bar = ctk.CTkProgressBar(r, fg_color=RAISED, progress_color=color, height=13)
            bar.grid(row=0, column=1, sticky='ew', padx=(0, 12))
            bar.set(0)
            percentage = self.label(r, '—', 14, color, True)
            percentage.grid(row=0, column=2)
            self.bars[result] = (bar, percentage)
        self.label(outcome, 'Estimates from completed-game results; not live win probabilities or betting odds.', 11, MUTED).pack(anchor='w', padx=18, pady=(13, 16))

        forms = self.frame(body)
        forms.pack(fill='x', pady=(0, 12))
        self.label(forms, 'RECENT FORM • LAST FIVE REAL MATCHES', 17, TEXT, True).pack(anchor='w', padx=18, pady=(14, 10))
        self.form_label = self.label(forms, 'Load real match history to see recent form.', 13, MUTED)
        self.form_label.configure(wraplength=950, justify='left')
        self.form_label.pack(anchor='w', padx=18, pady=(0, 18))
        self.label(body, 'DATA: ESPN unofficial public feed. Coverage and refresh times are not guaranteed. '
                         'No verified shots, possession, player injuries or lineups are integrated.',
                   11, YELLOW).pack(anchor='w', pady=8)

    def _render_data(self):
        self.snapshot = load_snapshot()
        updated = self.snapshot.get('updated_utc') or 'never'
        self.status.configure(text=f'Scoreboard last checked (UTC): {updated}  •  '
                                   f'Historical matches: {sum(len(v) for v in self.history.values()) // 2}')
        lines = []
        for e in self.snapshot.get('events', [])[:30]:
            date = e['date'][:16].replace('T', ' ')
            home, away = e['home_team'], e['away_team']
            score = f"{e['home_goals']}–{e['away_goals']}" if e['home_goals'] is not None and e['away_goals'] is not None else 'vs'
            lines.append(f"{date} UTC  [{e['state'].upper()}]  {home}  {score}  {away}")
        self.fixture_text.configure(state='normal')
        self.fixture_text.delete('1.0', 'end')
        self.fixture_text.insert('1.0', '\n'.join(lines) if lines else 'No upcoming/live MLS events were returned for the checked dates.')
        self.fixture_text.configure(state='disabled')
        teams = sorted(t for t, matches in self.history.items() if len(matches) >= 3)
        if teams:
            self.home_menu.configure(values=teams)
            self.away_menu.configure(values=teams)
            if self.home.get() not in teams:
                self.home.set(teams[0])
            if self.away.get() not in teams:
                self.away.set(teams[-1])
            self.show_form()
        self.predict_btn.configure(state='normal' if self.bundle and teams else 'disabled')

    def show_form(self):
        chunks = []
        for team in (self.home.get(), self.away.get()):
            games = self.history.get(team, [])
            stats = team_stats(games)
            results = '  '.join(g['form'] for g in games[-5:][::-1]) or '—'
            chunks.append(f'{team}:  {results}  |  Goals {stats["gf"]:.2f}/game  |  Conceded {stats["ga"]:.2f}/game  |  Points {stats["points"]:.2f}/game')
        self.form_label.configure(text='\n'.join(chunks))

    def do_predict(self):
        try:
            prob = predict(self.home.get(), self.away.get(), self.bundle)
            self.summary.configure(text='Model prediction from REAL MLS historical match results', text_color=NEON)
            for code, (bar, label) in self.bars.items():
                p = prob.get(code, 0.)
                bar.set(p)
                label.configure(text=f'{p:.1%}')
            self.show_form()
        except Exception as exc:
            messagebox.showerror('Prediction unavailable', str(exc), parent=self)

    def refresh(self):
        if self.busy:
            return
        self.busy = True
        self.status.configure(text='Checking the ESPN MLS feed...')
        def work():
            try:
                refresh_live()
                self.after(0, lambda: self.refresh_done(None))
            except Exception as exc:
                self.after(0, lambda e=str(exc): self.refresh_done(e))
        threading.Thread(target=work, daemon=True).start()

    def refresh_done(self, error):
        self.busy = False
        if error:
            self.status.configure(text=f'Unable to refresh: {error}', text_color=PINK)
            return
        # Keep model and form in sync with newly completed matches; do not silently
        # claim the model has been retrained. Use train_real_model after new games.
        if HISTORY.exists():
            try:
                self.history = team_histories(load_matches())
            except Exception:
                pass
        self._render_data()

    def auto_refresh(self):
        self.refresh()
        self.after(120000, self.auto_refresh)


if __name__ == '__main__':
    app = LiveMLS()
    app.mainloop()
