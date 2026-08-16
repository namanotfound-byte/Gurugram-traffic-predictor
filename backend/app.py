"""
Gurugram Traffic Predictor — Flask API
=======================================
Endpoints:
  GET /predict          → congestion for a specific corridor + time
  GET /all              → congestion for all corridors at a given time
  GET /corridor/<id>    → full 24h profile for a corridor
  GET /health           → uptime check

Run locally:
  pip install flask flask-cors
  python app.py

Deploy on Render:
  Build command : pip install -r requirements.txt
  Start command : gunicorn app:app
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import sys
import joblib

# Let this file import predict_raw/engineer_features from the model package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from traffic_model import predict_raw  # noqa: E402

app = Flask(__name__)
CORS(app)

# ── Load the trained model if it exists ────────────────────────────────────────
# If nobody has run `python model/traffic_model.py train` yet, MODEL is None and
# every endpoint below falls back to the hand-tuned PROFILES table so the API
# still works for demo purposes. Once a real model is trained (ideally on real
# TomTom data, not synthetic), this automatically switches to using it — no
# code changes needed, just restart the server after training.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "traffic_gbt.joblib")
try:
    _payload = joblib.load(MODEL_PATH)
    MODEL, MODEL_FEATURES = _payload["model"], _payload["features"]
    print(f"[app] Loaded trained model from {MODEL_PATH}")
except Exception as e:
    MODEL, MODEL_FEATURES = None, None
    print(f"[app] No trained model found at {MODEL_PATH} ({e}); falling back to PROFILES table")

# ── Corridors with real GPS coordinates ────────────────────────────────────────
CORRIDORS = [
    { "id": 0, "name": "NH-48 Delhi–Gurgaon Expressway", "sub": "Rajiv Chowk → Manesar",       "type": "highway",    "lat1": 28.503815599504435, "lon1": 77.09343452364664, "lat2": 28.39562084178753,  "lon2": 76.98190888131491 },
    { "id": 1, "name": "MG Road",                         "sub": "IFFCO Chowk → Sikandarpur",   "type": "arterial",   "lat1": 28.508894778568195, "lon1": 77.17721743594889, "lat2": 28.478042225940833, "lon2": 77.07083864273610 },
    { "id": 2, "name": "Golf Course Road",                "sub": "DLF Phase 1 → Sector 56",     "type": "arterial",   "lat1": 28.431703492780210, "lon1": 77.10528498680165, "lat2": 28.481382498749120, "lon2": 77.09461508412330 },
    { "id": 3, "name": "Sohna Road",                      "sub": "Rajiv Chowk → Badshahpur",    "type": "arterial",   "lat1": 28.450349642026610, "lon1": 77.03713062381875, "lat2": 28.271455674369320, "lon2": 77.06695242607199 },
    { "id": 4, "name": "Dwarka Expressway",               "sub": "Sheetla Mata → Delhi Border", "type": "expressway", "lat1": 28.536181207453280, "lon1": 77.11234838526346, "lat2": 28.395554775800992, "lon2": 76.98190888131491 },
    { "id": 5, "name": "Golf Course Extension Road",      "sub": "Sector 58 → Sector 66",       "type": "arterial",   "lat1": 28.412054739260950, "lon1": 77.07407384083510, "lat2": 28.402590141342984, "lon2": 77.04420387536318 },
    { "id": 6, "name": "Mehrauli–Gurgaon Road",           "sub": "Ghitorni → IFFCO Chowk",      "type": "arterial",   "lat1": 28.508894778568195, "lon1": 77.17721743594889, "lat2": 28.478042225940833, "lon2": 77.07083864273610 },
    { "id": 7, "name": "Southern Peripheral Road",        "sub": "Rajiv Chowk → Faridabad",     "type": "arterial",   "lat1": 28.395685721111290, "lon1": 76.98199151846016, "lat2": 28.409334490707020, "lon2": 77.20658699480870 },
]

# ── Congestion model ───────────────────────────────────────────────────────────
PROFILES = {
    "weekday": {
        "highway":    [.10,.08,.07,.07,.08,.15,.45,.82,.92,.75,.60,.55,.60,.65,.70,.72,.85,.90,.82,.65,.45,.30,.20,.12],
        "arterial":   [.08,.06,.06,.06,.07,.12,.38,.75,.88,.70,.55,.50,.58,.60,.65,.70,.80,.88,.78,.60,.40,.25,.15,.09],
        "expressway": [.08,.06,.06,.05,.07,.18,.50,.85,.90,.72,.58,.52,.60,.62,.68,.72,.82,.88,.80,.62,.42,.28,.17,.10],
    },
    "saturday": {
        "highway":    [.10,.08,.07,.06,.07,.10,.20,.40,.58,.70,.78,.80,.82,.80,.75,.72,.68,.62,.52,.40,.30,.22,.15,.11],
        "arterial":   [.08,.07,.06,.06,.07,.09,.18,.35,.52,.65,.72,.75,.78,.76,.70,.65,.60,.55,.45,.35,.25,.18,.12,.09],
        "expressway": [.09,.07,.06,.05,.06,.10,.22,.42,.60,.70,.76,.78,.80,.78,.72,.68,.62,.56,.46,.36,.26,.19,.13,.10],
    },
    "sunday": {
        "highway":    [.08,.07,.06,.05,.06,.08,.12,.20,.32,.45,.58,.65,.70,.72,.70,.65,.60,.55,.45,.35,.25,.18,.12,.09],
        "arterial":   [.07,.06,.05,.05,.05,.07,.10,.18,.28,.40,.52,.58,.62,.64,.62,.58,.52,.46,.38,.28,.20,.14,.10,.08],
        "expressway": [.07,.06,.05,.04,.05,.07,.11,.19,.30,.42,.54,.60,.64,.65,.63,.59,.53,.47,.39,.29,.21,.15,.10,.08],
    }
}

def get_day_type(day_of_week: int) -> str:
    if day_of_week == 5: return "saturday"
    if day_of_week == 6: return "sunday"
    return "weekday"

def congestion(corridor_id: int, hour: int, day_of_week: int) -> float:
    road_type = CORRIDORS[corridor_id]["type"]

    if MODEL is not None:
        val = predict_raw(MODEL, MODEL_FEATURES, hour, day_of_week, road_type)
        return round(min(1.0, max(0.0, val)), 3)

    # Fallback: hand-tuned demo table, used only until a real model is trained
    day_type = get_day_type(day_of_week)
    base  = PROFILES[day_type][road_type][hour]
    noise = ((corridor_id * 7 + hour * 3) % 13) / 130
    return round(min(1.0, base + noise), 3)


def data_source() -> str:
    return "trained_model" if MODEL is not None else "fallback_profile_table"

def label(val: float) -> str:
    if val < 0.35: return "Free"
    if val < 0.60: return "Moderate"
    if val < 0.80: return "Heavy"
    return "Severe"

def full_profile(corridor_id: int, day_of_week: int) -> list:
    return [congestion(corridor_id, h, day_of_week) for h in range(24)]

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "GBT-v1", "source": data_source()})


@app.route("/predict")
def predict():
    try:
        corridor_id = int(request.args.get("corridor", 0))
        hour        = int(request.args.get("hour", 8))
        day         = int(request.args.get("day", 0))
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    if not (0 <= corridor_id < len(CORRIDORS)):
        return jsonify({"error": "corridor must be 0–7"}), 400
    if not (0 <= hour <= 23):
        return jsonify({"error": "hour must be 0–23"}), 400
    if not (0 <= day <= 6):
        return jsonify({"error": "day must be 0–6"}), 400

    day_type = get_day_type(day)
    val      = congestion(corridor_id, hour, day)
    profile  = full_profile(corridor_id, day)
    best_hr  = int(np.argmin(profile))
    peak_hr  = int(np.argmax(profile))

    return jsonify({
        "corridor":         CORRIDORS[corridor_id],
        "hour":             hour,
        "day":              day,
        "day_type":         day_type,
        "congestion_index": val,
        "label":            label(val),
        "best_hour":        best_hr,
        "peak_hour":        peak_hr,
        "source":           data_source(),
    })


@app.route("/all")
def all_corridors():
    try:
        hour = int(request.args.get("hour", 8))
        day  = int(request.args.get("day", 0))
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    day_type = get_day_type(day)
    results  = []
    for c in CORRIDORS:
        val = congestion(c["id"], hour, day)
        results.append({**c, "congestion_index": val, "label": label(val)})

    vals    = [r["congestion_index"] for r in results]
    avg     = round(sum(vals) / len(vals), 3)
    worst   = results[int(np.argmax(vals))]["name"]
    free_ct = sum(1 for v in vals if v < 0.35)

    return jsonify({
        "hour": hour, "day": day, "day_type": day_type,
        "corridors": results,
        "summary": {"avg_congestion": avg, "worst_corridor": worst, "free_count": free_ct},
        "source": data_source(),
    })


@app.route("/corridor/<int:corridor_id>")
def corridor_profile(corridor_id):
    if not (0 <= corridor_id < len(CORRIDORS)):
        return jsonify({"error": "corridor must be 0–7"}), 400
    try:
        day = int(request.args.get("day", 0))
    except ValueError:
        return jsonify({"error": "Invalid day"}), 400

    day_type = get_day_type(day)
    profile  = full_profile(corridor_id, day)

    return jsonify({
        "corridor":  CORRIDORS[corridor_id],
        "day": day, "day_type": day_type,
        "profile":   profile,
        "best_hour": int(np.argmin(profile)),
        "peak_hour": int(np.argmax(profile)),
        "source":    data_source(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
