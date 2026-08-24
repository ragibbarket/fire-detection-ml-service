"""
train_model.py
----------------------------------------------------------------
Trains the fire / false_alarm classifier used by app.py.

Two data sources feed training, combined together:

  1. SYNTHETIC data (generate_synthetic_data) - always included, since
     it's what makes the model usable from day one, before any real
     verified readings exist.
  2. REAL, HUMAN-VERIFIED data - fetched from the backend's
     GET /api/alerts/training-data endpoint (see retrain.py). Only
     alerts a human actually verified (confirmed real fire, or marked
     false alarm) are used as ground truth - NOT the model's own past
     predictions, which would just reinforce whatever it already
     believes instead of actually learning.

Run this directly to train on synthetic data only:
    python train_model.py

For automatic retraining that also pulls in verified data from the
backend, see retrain.py (used by app.py's background scheduler).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SAMPLES_PER_CLASS = 1000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fire_model.joblib")


def generate_synthetic_data(n_per_class=N_SAMPLES_PER_CLASS):
    """
    Generates a synthetic dataset for the two classes the ML service
    needs to distinguish. Only called for readings that already crossed
    the basic suspicion threshold in the backend (see sensorRoutes.js),
    so both classes here have elevated smoke/temperature - the model's
    job is to separate real fire from a false alarm within that range.

    The ranges below deliberately OVERLAP, and each sample gets extra
    gaussian noise, so the model doesn't end up overconfident (always
    ~100%) the way cleanly-separated synthetic classes would cause.
    """
    rows = []

    for _ in range(n_per_class):
        flame = 1 if np.random.rand() < 0.85 else 0
        smoke = np.random.uniform(380, 800) + np.random.normal(0, 25)
        temp = np.random.uniform(40, 70) + np.random.normal(0, 2)
        humidity = np.random.uniform(15, 45) + np.random.normal(0, 3)
        rows.append([flame, smoke, temp, humidity, "fire"])

    for _ in range(n_per_class):
        flame = 1 if np.random.rand() < 0.10 else 0
        smoke = np.random.uniform(280, 560) + np.random.normal(0, 25)
        temp = np.random.uniform(30, 50) + np.random.normal(0, 2)
        humidity = np.random.uniform(30, 65) + np.random.normal(0, 3)
        rows.append([flame, smoke, temp, humidity, "false_alarm"])

    df = pd.DataFrame(rows, columns=["flameDetected", "smokeValue", "temperature", "humidity", "label"])
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def train_and_save(df, model_path=MODEL_PATH, verbose=True):
    """
    Trains a RandomForestClassifier on the given labeled dataframe
    (columns: flameDetected, smokeValue, temperature, humidity, label)
    and saves it to model_path. Shared by the manual script (below) and
    the automatic retraining scheduler in retrain.py.
    """
    X = df[["flameDetected", "smokeValue", "temperature", "humidity"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=5,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    if verbose:
        y_pred = model.predict(X_test)
        print("=== Classification report ===")
        print(classification_report(y_test, y_pred))
        print("=== Confusion matrix (rows=actual, cols=predicted) ===")
        print("classes:", model.classes_)
        print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    if verbose:
        print(f"\nSaved trained model to {model_path}")

    return model


def main():
    df = generate_synthetic_data()
    train_and_save(df)


if __name__ == "__main__":
    main()
