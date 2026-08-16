"""
Gurugram Traffic Congestion Predictor
======================================
ML pipeline: data collection (TomTom API) → feature engineering → model training → prediction

Author: [Your Name]
Stack : Python 3.10+, scikit-learn, pandas, requests

HOW TO RUN:
  - Set your key first (required for real collection):
      export TOMTOM_API_KEY="your_key_here"

  - Collect data every 30 mins (runs forever in foreground, e.g. on a VM):
      python traffic_model.py collect

  - Collect ONE round and exit (for cron / GitHub Actions schedulers):
      python traffic_model.py collect-once

  - Train model on collected data:
      python traffic_model.py train

  - Make a prediction:
      python traffic_model.py predict
"""

import os
import sys
import time
import datetime
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")
DATA_FILE      = "data/gurugram_traffic_raw.csv"
MODEL_FILE     = "models/traffic_gbt.joblib"
COLLECT_INTERVAL_MINUTES = 30

# ─────────────────────────────────────────────
# CORRIDORS — real GPS coordinates (right-clicked from Google Maps)
# ─────────────────────────────────────────────
CORRIDORS = [
    ("NH-48 Delhi-Gurgaon Expressway", "highway",    28.503815599504435, 77.09343452364664),
    ("MG Road",                        "arterial",   28.508894778568195, 77.17721743594889),
    ("Golf Course Road",               "arterial",   28.431703492780210, 77.10528498680165),
    ("Sohna Road",                     "arterial",   28.450349642026610, 77.03713062381875),
    ("Dwarka Expressway",              "expressway", 28.536181207453280, 77.11234838526346),
    ("Golf Course Extension Road",     "arterial",   28.412054739260950, 77.07407384083510),
    ("Mehrauli-Gurgaon Road",          "arterial",   28.508894778568195, 77.17721743594889),
    ("Southern Peripheral Road",       "arterial",   28.395685721111290, 76.98199151846016),
]


# ─────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────

def fetch_tomtom_flow(lat, lon):
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={TOMTOM_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()["flowSegmentData"]
        return {
            "free_flow_speed": d["freeFlowSpeed"],
            "current_speed":   d["currentSpeed"],
            "confidence":      d["confidence"],
            "congestion_idx":  round(1 - (d["currentSpeed"] / max(d["freeFlowSpeed"], 1)), 4),
        }
    except Exception as e:
        print(f"  [WARN] TomTom fetch failed ({lat},{lon}): {e}")
        return None


def collect_once():
    if not TOMTOM_API_KEY:
        print("  [ERROR] TOMTOM_API_KEY environment variable is not set.")
        print("  Get a free key at https://developer.tomtom.com and run:")
        print("    export TOMTOM_API_KEY=\"your_key_here\"")
        return pd.DataFrame()

    records = []
    now = datetime.datetime.now()
    print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Collecting data for all corridors...")

    for name, road_class, lat, lon in CORRIDORS:
        print(f"  Fetching: {name}...", end=" ", flush=True)
        flow = fetch_tomtom_flow(lat, lon)
        if flow is None:
            print("FAILED")
            continue

        record = {
            "corridor":        name,
            "road_class":      road_class,
            "hour":            now.hour,
            "day_of_week":     now.weekday(),
            "is_weekend":      int(now.weekday() >= 5),
            "is_peak_morning": int(7 <= now.hour <= 10),
            "is_peak_evening": int(17 <= now.hour <= 20),
            **flow,
            "timestamp":       now.isoformat(),
        }
        records.append(record)
        print(f"OK  (speed: {flow['current_speed']} km/h, congestion: {flow['congestion_idx']:.2f})")
        time.sleep(0.5)

    if not records:
        print("  [ERROR] No data collected this round.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(DATA_FILE)
    df.to_csv(DATA_FILE, mode="a", header=not file_exists, index=False)
    print(f"  Saved {len(df)} records to {DATA_FILE}")
    return df


def collect_loop():
    print("=" * 55)
    print("  Gurugram Traffic Collector")
    print(f"  Collecting every {COLLECT_INTERVAL_MINUTES} mins for all 8 corridors")
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    while True:
        try:
            collect_once()
            next_time = datetime.datetime.now() + datetime.timedelta(minutes=COLLECT_INTERVAL_MINUTES)
            print(f"  Next collection at {next_time.strftime('%H:%M:%S')}\n")
            time.sleep(COLLECT_INTERVAL_MINUTES * 60)
        except KeyboardInterrupt:
            print("\n\nStopped. Data saved to", DATA_FILE)
            break
        except Exception as e:
            print(f"  [ERROR] {e} — retrying in 5 minutes")
            time.sleep(300)


# ─────────────────────────────────────────────
# 2. SYNTHETIC DATASET (fallback)
# ─────────────────────────────────────────────

def generate_synthetic_data(n=5000):
    rng = np.random.default_rng(42)
    hours      = rng.integers(0, 24, n)
    dow        = rng.integers(0, 7, n)
    road_class = rng.choice(["highway", "arterial", "expressway"], n, p=[0.2, 0.65, 0.15])

    def base_congestion(h, d, rc):
        weekend = d >= 5
        am_peak = np.exp(-((h - 8.5) ** 2) / 3)
        pm_peak = np.exp(-((h - 18)  ** 2) / 3)
        base    = 0.25 * am_peak + 0.30 * pm_peak + 0.05
        if weekend:
            base *= 0.55
        mult = {"highway": 1.1, "expressway": 1.05, "arterial": 0.95}[rc]
        return np.clip(base * mult + rng.normal(0, 0.04), 0.0, 1.0)

    congestion = np.array([base_congestion(h, d, rc)
                           for h, d, rc in zip(hours, dow, road_class)])

    return pd.DataFrame({
        "hour":            hours,
        "day_of_week":     dow,
        "road_class":      road_class,
        "is_weekend":      (dow >= 5).astype(int),
        "is_peak_morning": ((hours >= 7) & (hours <= 10)).astype(int),
        "is_peak_evening": ((hours >= 17) & (hours <= 20)).astype(int),
        "hour_sin":        np.sin(2 * np.pi * hours / 24),
        "hour_cos":        np.cos(2 * np.pi * hours / 24),
        "congestion_idx":  congestion,
    })


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    le = LabelEncoder()
    df = df.copy()
    if "road_class" in df.columns:
        df["road_class_enc"] = le.fit_transform(df["road_class"])
    else:
        df["road_class_enc"] = 1
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    feature_cols = [
        "hour", "hour_sin", "hour_cos",
        "day_of_week", "is_weekend",
        "is_peak_morning", "is_peak_evening",
        "road_class_enc",
    ]
    return df, feature_cols


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"Training on REAL data: {len(df)} rows from {DATA_FILE}")
        if len(df) < 100:
            print(f"  Only {len(df)} real rows — mixing with synthetic data for stability")
            df = pd.concat([df, generate_synthetic_data(2000)], ignore_index=True)
    else:
        print("No real data found — training on synthetic data")
        df = generate_synthetic_data(5000)

    df, feature_cols = engineer_features(df)
    X = df[feature_cols]
    y = df["congestion_idx"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=10,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    cv  = cross_val_score(model, X, y, cv=5, scoring="r2")

    print("\n-- Model Performance --")
    print(f"  R2  (test set)   : {r2:.4f}")
    print(f"  MAE (test set)   : {mae:.4f}")
    print(f"  R2  (5-fold CV)  : {cv.mean():.4f} +/- {cv.std():.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string())

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    return model


# ─────────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────────

def predict_raw(model, feature_cols, hour, dow, road_class):
    rc_map = {"highway": 0, "arterial": 1, "expressway": 2}
    row = {
        "hour": hour, "day_of_week": dow,
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "is_weekend": int(dow >= 5),
        "is_peak_morning": int(7 <= hour <= 10),
        "is_peak_evening": int(17 <= hour <= 20),
        "road_class_enc": rc_map.get(road_class, 1),
    }
    return float(model.predict(pd.DataFrame([row])[feature_cols])[0])


def predict(hour, day_of_week, road_class="arterial"):
    payload = joblib.load(MODEL_FILE)
    model, feature_cols = payload["model"], payload["features"]
    idx = round(predict_raw(model, feature_cols, hour, day_of_week, road_class), 3)
    if idx < 0.35:   label = "Free flow"
    elif idx < 0.60: label = "Moderate"
    elif idx < 0.80: label = "Heavy"
    else:            label = "Severe"
    best_hour = min(range(24), key=lambda h: predict_raw(model, feature_cols, h, day_of_week, road_class))
    return {"congestion_index": idx, "label": label, "best_travel_hour": best_hour}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "collect":
        collect_loop()

    elif mode == "collect-once":
        # Single-shot collection, meant to be triggered by an external
        # scheduler (cron, GitHub Actions, etc.) rather than run forever.
        collect_once()

    elif mode == "train":
        print("=== Gurugram Traffic Predictor - Training ===\n")
        train_model()
        print("\n-- Sample predictions --")
        tests = [
            (8,  0, "highway",    "Monday 8 AM, NH-48"),
            (13, 2, "arterial",   "Wednesday 1 PM, MG Road"),
            (18, 4, "expressway", "Friday 6 PM, Dwarka Expy"),
            (11, 6, "arterial",   "Sunday 11 AM, Sohna Road"),
        ]
        for h, d, rc, lbl in tests:
            r = predict(h, d, rc)
            print(f"  {lbl:40s} -> {r['label']:12s} ({r['congestion_index']:.2f})  best: {r['best_travel_hour']:02d}:00")

    elif mode == "predict":
        print("=== Gurugram Traffic Predictor - Live Predictions ===\n")
        now = datetime.datetime.now()
        print(f"Current time: {now.strftime('%A %H:%M')}\n")
        for name, road_class, _, _ in CORRIDORS:
            r = predict(now.hour, now.weekday(), road_class)
            print(f"  {name:40s} -> {r['label']:12s} ({r['congestion_index']:.2f})")

    else:
        print("Usage: python traffic_model.py [collect|collect-once|train|predict]")
