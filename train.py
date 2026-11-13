"""Train + evaluate Random Forest and XGBoost on MLS match outcomes.

Usage:  python train.py [--data data/matches.csv] [--test-season 2024] [--window 5]
"""
import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from features import FEATURE_COLS, RESULT_NAMES, build_dataset

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_models(seed=42):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed)
    else:
        print("! xgboost not installed -> skipping it (pip install xgboost)")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matches.csv")
    ap.add_argument("--test-season", type=int, default=None, help="default: latest season")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    df = build_dataset(pd.read_csv(args.data), args.window)
    test_season = args.test_season or df.season.max()
    train, test = df[df.season < test_season], df[df.season == test_season]
    if train.empty or test.empty:
        raise ValueError("Training/test split is empty. Check --test-season and the CSV.")
    print(f"Train: {len(train)} matches (seasons < {test_season}) | Test: {len(test)} matches ({test_season})")

    X_tr, y_tr, X_te, y_te = train[FEATURE_COLS], train.result, test[FEATURE_COLS], test.result

    # Baseline: always guess the most common result
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    results = {"Baseline (most common)": accuracy_score(y_te, base.predict(X_te))}
    fitted, preds = {}, {}

    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        preds[name] = model.predict(X_te)
        fitted[name] = model
        results[name] = accuracy_score(y_te, preds[name])
        print(f"\n=== {name} ===")
        print(classification_report(y_te, preds[name], labels=[0, 1, 2],
                                    target_names=list(RESULT_NAMES.values()), zero_division=0))

    print("\nAccuracy summary")
    for k, v in results.items():
        print(f"  {k:<24}{v:.1%}")

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(results.keys(), results.values(), color=["#999", "#2a9d8f", "#e76f51"][:len(results)])
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in results.values()])
    ax.set_ylabel("Accuracy"); ax.set_title(f"Test accuracy ({test_season} season)")
    plt.xticks(rotation=15); plt.tight_layout()
    fig.savefig("outputs/accuracy_comparison.png", dpi=150); plt.close(fig)

    for name, model in fitted.items():
        imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).nlargest(12)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot.barh(ax=ax, color="#264653"); ax.set_title(f"{name}: top features")
        plt.tight_layout()
        fig.savefig(f"outputs/importance_{name.split()[0].lower()}.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_te, preds[name], labels=[0, 1, 2], display_labels=["Away", "Draw", "Home"],
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}: confusion matrix"); plt.tight_layout()
        fig.savefig(f"outputs/confusion_{name.split()[0].lower()}.png", dpi=150); plt.close(fig)

    # Save the best model, retrained on ALL data for real predictions
    best = max((k for k in fitted), key=lambda k: results[k])
    final = get_models()[best].fit(df[FEATURE_COLS], df.result)
    joblib.dump({"model": final, "name": best, "window": args.window}, "outputs/model.joblib")
    print(f"\nSaved {best} (retrained on all data) to outputs/model.joblib. Plots are in outputs/.")


if __name__ == "__main__":
    main()
