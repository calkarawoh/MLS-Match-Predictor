# MLS Match Outcome Predictor — PyCharm project

A Python machine-learning learning project that predicts **Away win / Draw / Home win** from team form in prior matches.

> **IMPORTANT:** `data/matches.csv` is **synthetic, generated example data** (2019–2024), **not real MLS history**. `data/upcoming.csv` contains example fixtures from **March 2025**, not current upcoming matches. The included plots show example results on synthetic data. Do not treat the predictions as reliable advice or real-world forecast accuracy.

## Open and run in PyCharm

1. Extract the ZIP and open the `MLS_Match_Predictor` folder with **File → Open**.
2. In PyCharm, choose **Settings → Project → Python Interpreter → Add Interpreter**, then create a **virtual environment** using Python 3.10+.
3. Open the **Terminal** tab, and enter:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run model training:
   ```bash
   python train.py --test-season 2024
   ```
   The script compares Random Forest and XGBoost against a most-frequent-class baseline, generates plots in `outputs/`, and saves a newly trained model to `outputs/model.joblib`.
5. Predict sample fixture outcomes:
   ```bash
   python predict.py --fixtures data/upcoming.csv
   ```
   You'll see the predicted outcome and estimated probability for each of the three outcomes.

If you are on Windows, run the commands from PyCharm's Terminal while the project folder is open. If `python` is not recognized, try `py`.

## Project layout

```text
MLS_Match_Predictor/
  data/
    matches.csv           # SYNTHETIC match history
    upcoming.csv          # SAMPLE historical-dated fixtures
  outputs/
    accuracy_comparison.png
    confusion_random.png
    importance_random.png
    model.joblib           # generated when you run train.py
  features.py             # rolling statistics from *prior* matches
  generate_sample_data.py # optional example-data generator
  train.py                # model training and evaluation
  predict.py              # fixture predictions
  requirements.txt
  README.md
  .gitignore
```

## Required real-data schema

Replace `data/matches.csv` with real results **only after obtaining them from a permitted source** and ensuring these columns exist:

```
date,season,home_team,away_team,home_goals,away_goals,home_shots,away_shots,home_possession,away_possession
```

Fixture input requires `date,home_team,away_team`. Team names must match history exactly. To predict matches after the latest known result, the history file must contain only matches played **before each target fixture**. This implementation computes form from the entire supplied history, so do **not** include results dated after the fixture being predicted.

## Machine learning logic

- The `features.py` module makes rolling averages (last five completed games) for goals scored/conceded, shots, possession and points.
- Using `shift(1)` prevents each training match from using its own result as a feature.
- Train/test split is time based: earlier years for training, selected test season for evaluation.
- The original project's `home_advantage` column is constant (`1`) for all rows, so it **does not actually teach the model a varying home advantage**. The home/away form columns still give it the home/away perspective.
- Verified local run on the supplied synthetic 2024 data: **39.7% baseline, 44.5% Random Forest, and 45.0% XGBoost**. Both trained models had trouble identifying draws. Results on real data will differ.

## Roadmap — adding real lineups

Lineups are **not currently included**. A later version could add confirmed starting XI, injuries, days of rest, player ratings, Elo rankings, expected goals (xG), transfers and travel distance. Lineups must be collected from reliable licensed sources and must be available **before kickoff** so that test data never leaks later information.

## Security note

Saved `.joblib` files are pickle-based. **Never load unknown/untrusted models**. This ZIP intentionally does **not** include the model binary from the uploaded files; the training script creates your own model locally.
