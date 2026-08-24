# Fire Detection ML Service (Flask)

Classifies a suspicious sensor reading as `fire` or `false_alarm`. Called by
the Node.js backend's `services/mlService.js`.

## Deploying for free (Render)

1. Push this folder to its own GitHub repo (separate from the backend repo).
2. On [render.com](https://render.com), New + → Web Service → connect the repo.
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT --timeout 120`
     (`--workers 1` matters - the retrain scheduler is stateful, more than
     one worker would retrain/reload redundantly)
   - Instance type: Free
3. Set environment variables: `BACKEND_URL` (the deployed Node backend's
   URL, no trailing slash), `RETRAIN_INTERVAL_MINUTES=30`.
4. No model file needs to be committed - on first boot with no
   `model/fire_model.joblib` present, `app.py` automatically trains one on
   synthetic data before serving requests.
5. Copy the resulting URL (e.g. `https://fire-ml-xxxx.onrender.com`) into
   the backend's `ML_API_URL` env var as `<that-url>/predict`.

**Free tier caveat:** same 15-min-idle sleep as the backend. While asleep,
the background retrain scheduler is also paused - it resumes automatically
once a request wakes the service back up.

## Setup

```bash
pip install -r requirements.txt
python train_model.py   # trains + saves model/fire_model.joblib
python app.py             # starts the API on http://localhost:8000
```

Point the backend's `.env` at it:
```
ML_API_URL=http://localhost:8000/predict
```

## Automatic retraining from real, verified data

The model no longer just trains once and sits static. `app.py` runs a
**background scheduler** that periodically:

1. Calls the backend's `GET /api/alerts/training-data` to fetch every alert
   a **human has actually verified** (confirmed real fire, or marked false
   alarm) - not the model's own past predictions, which would just
   reinforce whatever it already believes instead of learning anything.
2. Combines that real data with the synthetic baseline and retrains.
3. Hot-swaps the in-memory model - no restart needed.

Configure via `.env`:
```
BACKEND_URL=http://localhost:5000
RETRAIN_INTERVAL_MINUTES=30
```

**Manual trigger** (retrain immediately, e.g. right after verifying a demo
alert, to show the model incorporating it live):
```bash
curl -X POST http://localhost:8000/retrain
```
Response: `{ "status": "retrained", "verified_samples_used": N }`

Until any alerts have been human-verified, `verified_samples_used` will be
0 and the model trains on synthetic data only (same as before) - this is
expected and not an error.

## Why synthetic training data (for now)

Real hardware hasn't been purchased yet, so `train_model.py` generates a
synthetic dataset shaped like two realistic scenarios:

- **fire**: flame detected, smoke 450-800, temperature 45-70C, humidity 15-40%
- **false_alarm**: no flame, smoke 300-520, temperature 32-46C, humidity 35-65%
  (models cooking, steam, a heater - situations that trip the smoke/temp
  threshold but aren't a real fire)

Both classes only cover the "already suspicious" range, because the Node
backend only calls this service after a reading crosses its own basic
threshold (see `sensorRoutes.js`) - the model's job is to separate real
fire from a false alarm within that range, not to notice normal readings.

## Manually retraining from a CSV instead

If you'd rather not wait for the automatic scheduler, or want to bulk-load
readings from a CSV instead of relying on the backend export:

1. Log real readings to a CSV with columns: `flameDetected,smokeValue,temperature,humidity,label`
   (label the fire tests as `fire`, the cooking/steam/etc tests as `false_alarm`)
2. In `train_model.py`'s `main()`, load your CSV with pandas and concatenate
   it with `generate_synthetic_data()` before calling `train_and_save()`
3. Re-run `python train_model.py`

## API

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true }`.

### `POST /predict`
Request:
```json
{ "flameDetected": true, "smokeValue": 620, "temperature": 55, "humidity": 25 }
```
Response:
```json
{ "prediction": "fire", "confidence": 0.97 }
```

## Deployment

Free option: **Hugging Face Spaces** (Docker or Gradio SDK, CPU tier, free) -
or Render/Railway free tier for a plain Flask app. Update `ML_API_URL` in
the backend's `.env` to the deployed URL once live.
