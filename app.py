"""
app.py
----------------------------------------------------------------
Flask microservice that the Node.js backend calls to classify a
sensor reading as "fire" or "false_alarm", with a plain-language
reasoning string explaining the decision.

Contract (matches services/mlService.js in the backend):

  POST /predict
  Body: { "flameDetected": bool, "smokeValue": number,
          "temperature": number, "humidity": number }
  Response: { "prediction": "fire" | "false_alarm",
              "confidence": 0.0-1.0, "reasoning": "..." }

This service also AUTOMATICALLY RETRAINS itself in the background,
periodically pulling newly human-verified/labeled readings from the
backend (see retrain.py) and re-fitting the model - without needing to
restart the service. Configure the interval with RETRAIN_INTERVAL_MINUTES.

Run locally:
  python train_model.py   # once, to create model/fire_model.joblib
  python app.py            # starts the dev server on port 8000 (or $PORT)

Run in production (e.g. Render):
  gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT --timeout 120
  (--workers 1 matters: the retrain scheduler is stateful, running it in
  more than one worker process would retrain/reload redundantly)
"""

from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import joblib
import numpy as np
import os
import atexit

load_dotenv()

from retrain import retrain
from train_model import MODEL_PATH, generate_synthetic_data, train_and_save

app = Flask(__name__)

model = None
last_verified_count = 0  # avoids pointless retraining when nothing new was verified
RETRAIN_INTERVAL_MINUTES = int(os.environ.get("RETRAIN_INTERVAL_MINUTES", 30))


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        # first deploy on a fresh host (e.g. Render) - no model file yet,
        # train once on synthetic data so the service can start
        print("[startup] No saved model found - training an initial one on synthetic data")
        train_and_save(generate_synthetic_data(), model_path=MODEL_PATH, verbose=False)
    model = joblib.load(MODEL_PATH)


def scheduled_retrain_job():
    """
    Runs on a timer. Retrains on synthetic + whatever human-verified/labeled
    data exists, then hot-swaps the in-memory model - no restart needed.
    """
    global last_verified_count
    print("[scheduler] Running automatic retrain check...")
    verified_count = retrain()
    if verified_count != last_verified_count:
        load_model()
        print(f"[scheduler] Model reloaded ({verified_count} verified samples, was {last_verified_count})")
        last_verified_count = verified_count
    else:
        print(f"[scheduler] No new verified data since last check ({verified_count} samples) - model unchanged")


def generate_reasoning(data, prediction, confidence):
    """
    Builds a plain-language explanation for the prediction by comparing
    the input values against the ranges used to define each class during
    training (see train_model.py). This is template-based, not a literal
    trace of the Random Forest's internal splits - but it reflects the
    same signals (flame presence, smoke level, temperature, humidity)
    the model was trained to weigh, so it's an honest explanation of why
    a reading looks like fire vs. a false alarm.
    """
    flame = bool(data["flameDetected"])
    smoke = float(data["smokeValue"])
    temp = float(data["temperature"])
    humidity = float(data["humidity"])

    reasons = []

    if flame:
        reasons.append("flame was directly detected, the strongest single signal of an active fire")
    else:
        reasons.append("no flame was detected by the sensor")

    if smoke >= 450:
        reasons.append(f"smoke level ({smoke:.0f} ppm) is in the range typically seen during real fires")
    elif smoke >= 300:
        reasons.append(f"smoke level ({smoke:.0f} ppm) is elevated but also consistent with cooking, steam, or dust")
    else:
        reasons.append(f"smoke level ({smoke:.0f} ppm) is close to normal background levels")

    if temp >= 45:
        reasons.append(f"temperature ({temp:.1f}C) is high enough to be consistent with fire")
    elif temp >= 32:
        reasons.append(f"temperature ({temp:.1f}C) is elevated but not extreme")
    else:
        reasons.append(f"temperature ({temp:.1f}C) is within a normal range")

    if humidity <= 40:
        reasons.append(f"humidity ({humidity:.0f}%) is low, which is common near an active fire")
    else:
        reasons.append(f"humidity ({humidity:.0f}%) is within a normal range")

    verdict = "This combination points to a real fire" if prediction == "fire" else "This combination looks like a false alarm rather than a real fire"
    confidence_pct = round(confidence * 100)

    return f"{verdict} ({confidence_pct}% confidence): " + "; ".join(reasons) + "."


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    required_fields = ["flameDetected", "smokeValue", "temperature", "humidity"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    features = np.array([[
        1 if data["flameDetected"] else 0,
        float(data["smokeValue"]),
        float(data["temperature"]),
        float(data["humidity"]),
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])
    reasoning = generate_reasoning(data, prediction, confidence)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
    })


@app.route("/retrain", methods=["POST"])
def manual_retrain():
    """
    Triggers retraining immediately (doesn't wait for the scheduled
    interval) - useful right after verifying/labeling a reading, to show
    the model incorporating it live.
    """
    global last_verified_count
    verified_count = retrain()
    load_model()
    last_verified_count = verified_count
    return jsonify({"status": "retrained", "verified_samples_used": verified_count})


# ---- runs on import, so this works under BOTH `python app.py` AND
# gunicorn (which imports the module and never hits the __main__ block) ----
load_model()
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_retrain_job, "interval", minutes=RETRAIN_INTERVAL_MINUTES)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())
print(f"[scheduler] Automatic retraining enabled - every {RETRAIN_INTERVAL_MINUTES} minutes")


if __name__ == "__main__":
    # local dev only - in production (Render) gunicorn runs this instead,
    # see the module docstring at the top of this file
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
