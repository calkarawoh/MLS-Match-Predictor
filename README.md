# ⚽ MLS Match Predictor

**A Python-based machine learning application for predicting Major League Soccer match outcomes.**

The MLS Match Predictor analyzes historical match data, team performance, and recent form to estimate the probability of three possible outcomes:

- Home Win
- Draw
- Away Win

The project combines machine learning, football statistics, data processing, and a graphical user interface to explore how predictive analytics can be applied to professional soccer.

---

## Project Overview

The MLS Match Predictor is a machine learning project developed in Python using historical match statistics and team performance indicators.

The system explores how factors such as goals scored, goals conceded, shots, possession, recent form, and home/away performance can contribute to predicting match outcomes.

The project includes a desktop prediction interface, data-processing modules, model-training scripts, and live-data integration.

**Important:** Not all datasets in this repository represent verified real MLS results. The original `data/matches.csv` contains synthetic example data. Predictions and accuracy measurements based on synthetic data should not be interpreted as real-world forecasting performance.

## Features

- **Match Outcome Prediction:** Estimates Home Win, Draw, and Away Win probabilities.
- **Machine Learning:** Uses Random Forest and XGBoost algorithms.
- **Team Performance Analysis:** Examines recent goals, shots, possession, and points.
- **Live Data Integration:** Includes modules for refreshing MLS fixtures and match results.
- **Desktop GUI:** Provides a graphical interface for interacting with the prediction system.
- **Model Evaluation:** Generates accuracy comparisons, confusion matrices, and feature-importance visualizations.
- **Prediction Analytics:** Explores how historical performance influences predicted outcomes.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting |
| Matplotlib | Data visualization |
| CustomTkinter / Tkinter | Desktop graphical interface |
| Joblib | Model serialization |

## Project Structure

```text
MLS_Match_Predictor/
│
├── data/
│   ├── matches.csv
│   ├── mls_real_matches.csv
│   ├── mls_live_snapshot.json
│   └── upcoming.csv
│
├── outputs/
│   ├── accuracy_comparison.png
│   ├── confusion_random.png
│   ├── confusion_xgboost.png
│   ├── importance_random.png
│   └── importance_xgboost.png
│
├── dashboard.py
├── dashboard_fixed.py
├── dashboard_linked.py
├── dashboard_linked_v2.py
├── dashboard_live.py
├── dashboard_old.py
│
├── features.py
├── gui.py
├── mls_live_data.py
├── generate_sample_data.py
├── predict.py
├── real_model.py
├── train.py
│
├── requirements.txt
├── README.md
└── README_LIVE.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/calkarawoh/MLS-Match-Predictor.git
```

### 2. Open the project

Open the `MLS-Match-Predictor` folder in PyCharm.

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the desktop application

The current development dashboard can be launched with:

```bash
python dashboard_linked_v2.py
```

The live-data dashboard is also available:

```bash
python dashboard_live.py
```

## Machine Learning Models

The project explores two machine learning algorithms:

**Random Forest**

Uses multiple decision trees to identify patterns in historical match statistics and estimate match outcomes.

**XGBoost**

Uses gradient boosting to learn patterns from match statistics and improve predictive performance.

### Prediction Features

The model explores:

- Goals scored and conceded
- Shots taken
- Possession percentages
- Recent team performance
- Home and away form
- Points from previous matches

Rolling averages are calculated using previous matches, with a five-match form window in the original training pipeline.

### Model Evaluation

The original synthetic-data experiment produced the following results:

| Model | Accuracy |
|---|---:|
| Most-frequent-class baseline | 39.7% |
| Random Forest | 44.5% |
| XGBoost | 45.0% |

These figures are from the supplied synthetic 2024 test data, not verified real MLS match predictions.

## Data Sources and Limitations

The repository contains multiple datasets with different purposes:

- `matches.csv`: Synthetic example match history.
- `upcoming.csv`: Example fixtures dated March 2025.
- `mls_real_matches.csv`: Separate historical match dataset requiring source and provenance verification.
- `mls_live_snapshot.json`: Stored MLS fixture/result snapshot, not a guarantee of continuously live information.

Prediction quality depends on the accuracy, completeness, and timing of the underlying data.

## Future Development Roadmap

The long-term goal is to develop the MLS Match Predictor into a publicly accessible, mobile-friendly sports analytics platform.

Planned improvements include:

1. Modern mobile-responsive web interface.
2. Live MLS fixtures and match results.
3. Improved machine learning models.
4. Confirmed starting lineups and player availability.
5. Injury and suspension information.
6. Expected goals (xG) statistics.
7. Team rankings and Elo ratings.
8. Match history and prediction tracking.
9. Model performance monitoring.

## Developer

**Calvin Karau**

Computer Science Student | Aspiring Software Engineer

GitHub: [@calkarawoh](https://github.com/calkarawoh)

---

*This project is developed for educational and experimental purposes. Predictions are statistical estimates, not guaranteed match outcomes or betting advice.*