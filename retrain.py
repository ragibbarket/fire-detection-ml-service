"""
retrain.py
----------------------------------------------------------------
Automatic retraining: pulls HUMAN-VERIFIED alerts from the Node backend
(GET /api/alerts/training-data) and retrains the model on synthetic data
+ that real, human-labeled data.

Only human-verified alerts count as training data on purpose - see the
docstring in train_model.py for why using the model's own past
predictions as labels would be a mistake (it would just reinforce
whatever the model already believes, not teach it anything new).

Used by:
  - app.py's background scheduler (automatic periodic retraining)
  - POST /retrain on the Flask app (manual trigger, e.g. right after a
    demo verification, for testing)
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from train_model import generate_synthetic_data, train_and_save, MODEL_PATH

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")


def fetch_verified_data():
    """
    Returns a dataframe of human-verified real readings, or None if the
    backend is unreachable or there's nothing verified yet.
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/api/alerts/training-data", timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("rows", [])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df[["flameDetected", "smokeValue", "temperature", "humidity", "label"]]
        df["flameDetected"] = df["flameDetected"].astype(int)
        return df
    except Exception as e:
        print(f"[retrain] Could not fetch verified data from backend ({BACKEND_URL}): {e}")
        return None


def retrain():
    """
    Retrains and overwrites the saved model. Returns the number of real
    verified samples that were included (0 if none/unreachable).
    """
    synthetic_df = generate_synthetic_data()
    real_df = fetch_verified_data()

    if real_df is not None and len(real_df) > 0:
        combined_df = pd.concat([synthetic_df, real_df], ignore_index=True)
        print(f"[retrain] Training on {len(synthetic_df)} synthetic + {len(real_df)} human-verified real samples")
    else:
        combined_df = synthetic_df
        print(f"[retrain] No verified real data available yet - training on {len(synthetic_df)} synthetic samples only")

    train_and_save(combined_df, model_path=MODEL_PATH, verbose=True)
    return len(real_df) if real_df is not None else 0


if __name__ == "__main__":
    retrain()
