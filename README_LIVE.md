# Real MLS + live scoreboard upgrade (CustomTkinter)

Copy **these three Python files** into your existing `MLS_Match_Predictor` folder (beside `train.py` and `features.py`). Your old files and synthetic model stay untouched.

## Windows PowerShell commands

```powershell
.\.venv\Scripts\python.exe -m pip install requests customtkinter pandas scikit-learn joblib
.\.venv\Scripts\python.exe mls_live_data.py sync --start 2024
.\.venv\Scripts\python.exe real_model.py train
.\.venv\Scripts\python.exe dashboard_live.py
```

- **sync** downloads completed real MLS scores into `data/mls_real_matches.csv`. It may make several dozen requests; if an endpoint is unavailable, it fails rather than fabricating matches.
- **train** runs chronological holdout testing on the most recent included season and creates `outputs/real_mls_model.joblib`. Inspect accuracy and especially draw performance; score-only results do not guarantee good forecasts.
- **dashboard_live** shows neon-green real-data pre-match predictions, recent results, scores, and upcoming fixtures. It polls the unofficial ESPN scoreboard every 2 minutes while open, checking a window of 2 days before and 10 days after today. This is **near-real-time polling**, not guaranteed second-by-second updates.
- The dashboard does **not** auto-retrain the ML model after new games. Run `real_model.py train` periodically after the history CSV receives new results, then restart the dashboard.
- Times shown are **UTC**, clearly labeled. No shots, possession, xG, injuries, or lineups are claimed; these require separate data acquisition and a redesigned feature set.
- ESPN's publicly reachable endpoints are **unofficial, undocumented, rate-limited at their discretion**, and may change. Do not treat the feed as an official licensed commercial data integration. Check terms and attribution before public hosting.
- Historical and live availability depend on provider coverage and your network connection. The provided scripts have passed local syntax/simulated-data tests; live network calls must be tested on your PC.
