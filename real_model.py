"""Honest goals-only MLS model trained on *real completed* match results.

No fabricated shots or possession. Chronological out-of-season evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'data' / 'mls_real_matches.csv'
MODEL = ROOT / 'outputs' / 'real_mls_model.joblib'
WINDOW = 5
METRICS = ('gf', 'ga', 'points')
FEATURES = [f'{side}_{field}' for side in ('home', 'away')
            for field in ('gf', 'ga', 'points', 'games')]
FEATURES += ['diff_gf', 'diff_ga', 'diff_points']
LABELS = {0: 'Away win', 1: 'Draw', 2: 'Home win'}


def team_stats(games: list[dict], window: int = WINDOW) -> dict:
    recent = games[-window:]
    if not recent:
        return {'gf': 0., 'ga': 0., 'points': 0., 'games': 0.}
    return {key: float(np.mean([g[key] for g in recent])) for key in METRICS} | {'games': float(len(games))}


def make_row(home: dict, away: dict) -> dict:
    result = {f'home_{k}': home[k] for k in ('gf', 'ga', 'points', 'games')}
    result.update({f'away_{k}': away[k] for k in ('gf', 'ga', 'points', 'games')})
    for key in METRICS:
        result[f'diff_{key}'] = home[key] - away[key]
    return result


def load_matches() -> pd.DataFrame:
    if not CSV.exists():
        raise FileNotFoundError(f'No real data at {CSV}. Run: python mls_live_data.py sync')
    df = pd.read_csv(CSV, dtype={'event_id': str})
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])
    df = df.drop_duplicates('event_id').sort_values(['date', 'event_id']).reset_index(drop=True)
    return df


def build_training(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    history = defaultdict(list)
    rows = []
    # All features are calculated BEFORE adding the match being predicted.
    for game in df.to_dict('records'):
        home, away = game['home_team'], game['away_team']
        hs, as_ = team_stats(history[home], window), team_stats(history[away], window)
        if hs['games'] >= 3 and as_['games'] >= 3:
            label = 2 if game['home_goals'] > game['away_goals'] else 0 if game['home_goals'] < game['away_goals'] else 1
            rows.append(make_row(hs, as_) | {'result': label, 'season': int(game['season'])})
        h, a = int(game['home_goals']), int(game['away_goals'])
        history[home].append({'gf': h, 'ga': a, 'points': 3 if h > a else 1 if h == a else 0})
        history[away].append({'gf': a, 'ga': h, 'points': 3 if a > h else 1 if a == h else 0})
    return pd.DataFrame(rows)


def train() -> None:
    df = load_matches()
    frame = build_training(df)
    if len(frame) < 200 or frame.result.nunique() != 3 or frame.season.nunique() < 2:
        raise RuntimeError('Need at least 200 real training examples, three outcomes, and two seasons. Sync more history.')
    test_year = int(frame.season.max())
    train_frame = frame[frame.season < test_year]
    test_frame = frame[frame.season == test_year]
    if len(train_frame) < 100 or len(test_frame) < 30:
        raise RuntimeError('Insufficient train/test examples by season. Add more historical years.')
    model = RandomForestClassifier(n_estimators=350, max_depth=7, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1)
    model.fit(train_frame[FEATURES], train_frame['result'])
    preds = model.predict(test_frame[FEATURES])
    baseline = DummyClassifier(strategy='most_frequent').fit(train_frame[FEATURES], train_frame['result'])
    print(f'REAL MLS evaluation | training seasons < {test_year} ({len(train_frame)} games) | test season {test_year} ({len(test_frame)} games)')
    print(f'Baseline accuracy: {accuracy_score(test_frame.result, baseline.predict(test_frame[FEATURES])):.1%}')
    print(f'Random Forest accuracy: {accuracy_score(test_frame.result, preds):.1%}')
    print(classification_report(test_frame.result, preds, labels=[0, 1, 2], target_names=['Away win', 'Draw', 'Home win'], zero_division=0))
    # Model for operational prediction is then fitted on all completed games.
    model.fit(frame[FEATURES], frame.result)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'window': WINDOW, 'source': 'real MLS completed scores (unofficial ESPN)', 'test_year': test_year}, MODEL)
    print(f'Saved trained real-data model to {MODEL}')


def team_histories(df: pd.DataFrame) -> dict:
    history = defaultdict(list)
    for game in df.to_dict('records'):
        home, away = game['home_team'], game['away_team']
        h, a = int(game['home_goals']), int(game['away_goals'])
        history[home].append({'gf': h, 'ga': a, 'points': 3 if h > a else 1 if h == a else 0, 'form': 'W' if h > a else 'D' if h == a else 'L'})
        history[away].append({'gf': a, 'ga': h, 'points': 3 if a > h else 1 if h == a else 0, 'form': 'W' if a > h else 'D' if h == a else 'L'})
    return history


def predict(home: str, away: str, bundle: dict | None = None) -> dict[int, float]:
    if home == away:
        raise ValueError('Choose two different teams')
    if bundle is None:
        bundle = joblib.load(MODEL)  # Only load a model generated locally by this project.
    history = team_histories(load_matches())
    if len(history.get(home, [])) < 3 or len(history.get(away, [])) < 3:
        raise ValueError('Not enough completed games for one or both teams')
    row = pd.DataFrame([make_row(team_stats(history[home], bundle['window']), team_stats(history[away], bundle['window']))])
    probabilities = bundle['model'].predict_proba(row[bundle['features']])[0]
    return {int(label): float(value) for label, value in zip(bundle['model'].classes_, probabilities)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train'])
    parser.parse_args()
    train()
