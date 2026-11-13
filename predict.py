"""Predict upcoming fixtures.

Usage:  python predict.py --fixtures data/upcoming.csv
fixtures CSV columns: date, home_team, away_team
"""
import argparse
import joblib
import pandas as pd
from features import FEATURE_COLS, RESULT_NAMES, build_upcoming


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/matches.csv")
    ap.add_argument("--fixtures", default="data/upcoming.csv")
    ap.add_argument("--model", default="outputs/model.joblib")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    history = pd.read_csv(args.history)
    fixtures = pd.read_csv(args.fixtures)
    X = build_upcoming(fixtures, history, bundle["window"])

    proba = bundle["model"].predict_proba(X[FEATURE_COLS])
    out = fixtures.copy()
    out["prediction"] = [RESULT_NAMES[i] for i in proba.argmax(axis=1)]
    out["P(away)"], out["P(draw)"], out["P(home)"] = proba[:, 0], proba[:, 1], proba[:, 2]
    print(f"Model: {bundle['name']}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
