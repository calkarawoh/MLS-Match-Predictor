"""Desktop interface for the MLS Match Outcome Predictor.

Run from the project directory: py gui.py
The bundled dataset is synthetic; displayed probabilities are demonstrations,
not validated forecasts of real MLS matches.
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd

from features import FEATURE_COLS, RESULT_NAMES, build_upcoming

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "matches.csv"


def predict_match(model_bundle, history, home_team, away_team):
    """Return {class_id: probability} using the model's actual class ordering."""
    if home_team == away_team:
        raise ValueError("Choose two different teams.")

    fixtures = pd.DataFrame([{
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "home_team": home_team,
        "away_team": away_team,
    }])
    features = build_upcoming(fixtures, history, window=model_bundle["window"])
    model = model_bundle["model"]
    values = model.predict_proba(features[FEATURE_COLS])[0]
    return {int(label): float(prob) for label, prob in zip(model.classes_, values)}


def main():
    root = tk.Tk()
    root.title("MLS Match Predictor")
    root.geometry("640x550")
    root.minsize(560, 510)
    root.configure(bg="#102033")

    try:
        bundle = joblib.load(MODEL_PATH)
        history = pd.read_csv(DATA_PATH)
        teams = sorted(set(history["home_team"]) | set(history["away_team"]))
    except (OSError, KeyError, ValueError, ImportError) as error:
        messagebox.showerror(
            "Cannot open predictor",
            f"Could not load the trained model or match data:\n{error}\n\n"
            "Run 'py train.py --test-season 2024' first.",
            parent=root,
        )
        root.destroy()
        return

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Team.TCombobox", padding=7, font=("Segoe UI", 11))
    style.configure("Result.Horizontal.TProgressbar", troughcolor="#26384d", background="#3fd0a1")

    frame = tk.Frame(root, bg="#102033", padx=28, pady=22)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="MLS MATCH PREDICTOR", bg="#102033", fg="#f8fafc",
             font=("Segoe UI", 22, "bold")).pack(pady=(0, 4))
    tk.Label(frame, text=f"Model: {bundle['name']}  •  Historical form (last {bundle['window']} games)",
             bg="#102033", fg="#a7b8c9", font=("Segoe UI", 10)).pack(pady=(0, 20))

    home = tk.StringVar(value="Seattle Sounders" if "Seattle Sounders" in teams else teams[0])
    away = tk.StringVar(value="Inter Miami" if "Inter Miami" in teams else teams[1])

    tk.Label(frame, text="HOME TEAM", bg="#102033", fg="#f8fafc",
             font=("Segoe UI", 10, "bold")).pack(anchor="w")
    home_box = ttk.Combobox(frame, textvariable=home, values=teams, state="readonly",
                            style="Team.TCombobox")
    home_box.pack(fill="x", pady=(4, 12))

    tk.Label(frame, text="AWAY TEAM", bg="#102033", fg="#f8fafc",
             font=("Segoe UI", 10, "bold")).pack(anchor="w")
    away_box = ttk.Combobox(frame, textvariable=away, values=teams, state="readonly",
                            style="Team.TCombobox")
    away_box.pack(fill="x", pady=(4, 15))

    result = tk.StringVar(value="Select two teams, then click Predict Match")
    label = tk.Label(frame, textvariable=result, bg="#102033", fg="#f8fafc",
                     font=("Segoe UI", 12, "bold"), wraplength=550)

    values = {}
    bars = {}
    for outcome in (2, 1, 0):
        row = tk.Frame(frame, bg="#102033")
        row.pack(fill="x", pady=5)
        tk.Label(row, text=RESULT_NAMES[outcome], bg="#102033", fg="#d9e5f0",
                 width=13, anchor="w", font=("Segoe UI", 10)).pack(side="left")
        bar = ttk.Progressbar(row, maximum=100, mode="determinate", style="Result.Horizontal.TProgressbar")
        bar.pack(side="left", fill="x", expand=True, padx=(4, 12))
        val = tk.StringVar(value="--")
        tk.Label(row, textvariable=val, bg="#102033", fg="#f8fafc",
                 width=7, anchor="e", font=("Segoe UI", 11, "bold")).pack(side="right")
        values[outcome], bars[outcome] = val, bar

    def on_predict():
        try:
            probs = predict_match(bundle, history, home.get(), away.get())
        except (ValueError, KeyError) as error:
            messagebox.showwarning("Prediction unavailable", str(error), parent=root)
            return
        winner = max(probs, key=probs.get)
        result.set(f"Prediction: {RESULT_NAMES[winner]}")
        for outcome in (2, 1, 0):
            percentage = 100 * probs.get(outcome, 0.0)
            values[outcome].set(f"{percentage:.1f}%")
            bars[outcome]["value"] = percentage

    tk.Button(frame, text="PREDICT MATCH", command=on_predict, bg="#3fd0a1",
              fg="#102033", activebackground="#62e4b7", relief="flat",
              font=("Segoe UI", 12, "bold"), padx=18, pady=10).pack(fill="x", pady=(0, 15))
    label.pack(pady=(5, 14))

    tk.Label(frame, text="DEMO ONLY: sample training matches are synthetic, not real MLS results.",
             bg="#102033", fg="#ffcf87", font=("Segoe UI", 9), wraplength=560).pack(side="bottom")
    root.mainloop()


if __name__ == "__main__":
    main()