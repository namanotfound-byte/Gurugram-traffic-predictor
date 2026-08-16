"""
Gurugram Traffic Congestion Predictor — training + prediction
================================================================
ML pipeline: real TomTom traffic data -> feature engineering -> model
training -> prediction.

Data collection now lives in `bootstrap_collect.py` at the repo root (it
sweeps the TomTom Routing API's `departAt` + `computeTravelTimeFor=all`
trick, since this project's API key is Routing-only and the Flow API
returns 403). This file only trains on and serves predictions from
whatever real data has been collected — it never invents synthetic data.

HOW TO RUN:
  - Collect real data first (from repo root):
      python bootstrap_collect.py --max-requests 1400

  - Train model on collected data:
      python model/traffic_model.py train

  - Make a prediction for the current moment, all corridors:
      python model/traffic_model.py predict
"""

import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import corridors  # noqa: E402

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(_REPO_ROOT, "data", "gurugram_bootstrap.csv")
MODEL_FILE = os.path.join(_REPO_ROOT, "models", "traffic_gbt.joblib")

FEATURE_COLS = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "is_weekend",
    "is_peak_morning", "is_peak_evening",
    "road_class_enc",
]


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered columns in place of the buggy LabelEncoder approach.

    Uses corridors.ROAD_CLASS_ENC as the single source of truth for the
    road_class -> integer mapping, on BOTH the training side (here) and the
    prediction side (predict_raw below) — this is the fix for the bug where
    training used alphabetical LabelEncoder order (arterial=0, expressway=1,
    highway=2) while predict_raw hardcoded {"highway":0,"arterial":1,
    "expressway":2}, silently scrambling every road class at inference time.
    """
    df = df.copy()

    if "road_class" not in df.columns:
        raise ValueError("engineer_features: input data has no 'road_class' column")

    unknown = set(df["road_class"].unique()) - set(corridors.ROAD_CLASS_ENC)
    if unknown:
        raise ValueError(
            f"engineer_features: road_class value(s) {unknown} not in "
            f"corridors.ROAD_CLASS_ENC ({corridors.ROAD_CLASS_ENC})"
        )
    df["road_class_enc"] = df["road_class"].map(corridors.ROAD_CLASS_ENC)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "is_peak_morning" not in df.columns:
        df["is_peak_morning"] = ((df["hour"] >= 7) & (df["hour"] <= 10)).astype(int)
    if "is_peak_evening" not in df.columns:
        df["is_peak_evening"] = ((df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)

    return df


def load_bootstrap_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        print(f"[FATAL] {DATA_FILE} not found.")
        print("        Run `python bootstrap_collect.py` from the repo root first.")
        print("        This project trains on real TomTom data only — there is no")
        print("        synthetic fallback.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        print(f"[FATAL] {DATA_FILE} exists but has 0 rows.")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=["congestion_idx", "hour", "day_of_week", "road_class"])
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} row(s) with missing required fields.")

    out_of_range = df[(df["congestion_idx"] < 0) | (df["congestion_idx"] > 1)]
    if len(out_of_range):
        print(f"[WARN] {len(out_of_range)} row(s) have congestion_idx outside "
              f"[0, 1]; clipping.")
        df["congestion_idx"] = df["congestion_idx"].clip(0.0, 1.0)

    # ── Route-consistency filter ───────────────────────────────────────
    # The routing engine sometimes reroutes a corridor onto a physically
    # different path at certain hours (observed directly: Golf Course
    # Extension Rd, Mehrauli-Gurgaon Rd, and Southern Peripheral Rd all
    # showed >9% length_m swings between hours). Those rows' congestion_idx
    # describes a different road than the one the corridor is defined as, so
    # they are excluded from training by default rather than silently mixed
    # in. bootstrap_collect.py stamps this per-row as `route_stable`.
    if "route_stable" in df.columns:
        def _to_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("true", "1")
        df["route_stable"] = df["route_stable"].map(_to_bool)
        n_unstable = int((~df["route_stable"]).sum())
        if n_unstable:
            per_corridor = (
                df[~df["route_stable"]]
                .groupby(["corridor_id", "corridor_name"])
                .size()
            )
            print(f"[INFO] Excluding {n_unstable}/{len(df)} row(s) flagged "
                  f"route_stable=False (routing engine picked a different "
                  f"physical road at that hour):")
            for (cid, name), n in per_corridor.items():
                print(f"         corridor {cid} ({name}): {n} row(s) excluded")
        df = df[df["route_stable"]].drop(columns=["route_stable"])
    else:
        print("[WARN] No 'route_stable' column in the data — cannot filter out "
              "hours where the routing engine picked a different physical road. "
              "Re-run bootstrap_collect.py to get this column.")

    return df


# ─────────────────────────────────────────────
# 2. MODEL TRAINING
# ─────────────────────────────────────────────

def _fit_gbt(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=10,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _evaluate_naive_split(df, feature_cols):
    """Random 80/20 split. THIS LEAKS: adjacent hours for the same corridor
    are near-duplicates, so the test set is not really held out. Reported
    for comparison only — do not trust this number as generalization."""
    X = df[feature_cols]
    y = df["congestion_idx"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = _fit_gbt(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def _evaluate_group_kfold(df, feature_cols, group_col="corridor_id"):
    """Leave-one-corridor-out CV. Honest generalization estimate: no hour
    from a held-out corridor is ever seen during that fold's training, so
    this can't cheat off near-duplicate adjacent-hour rows for the SAME
    corridor. Caveat: NH-48 (id 0) is the only 'highway' corridor and Dwarka
    Expressway (id 4) is the only 'expressway' corridor, so their folds also
    test extrapolation to a road_class unseen anywhere in that fold's
    training data — expect those folds to score worse for that reason, not
    because the model is bad."""
    X = df[feature_cols]
    y = df["congestion_idx"]
    groups = df[group_col]
    n_groups = groups.nunique()

    gkf = GroupKFold(n_splits=n_groups)
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        held_out_corridor = groups.iloc[test_idx].unique()
        model = _fit_gbt(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        fold_r2 = r2_score(y.iloc[test_idx], y_pred)
        fold_mae = mean_absolute_error(y.iloc[test_idx], y_pred)
        fold_results.append({
            "held_out_corridor_id": int(held_out_corridor[0]),
            "r2": fold_r2,
            "mae": fold_mae,
            "n_test": len(test_idx),
        })

    r2s = np.array([f["r2"] for f in fold_results])
    maes = np.array([f["mae"] for f in fold_results])
    return {
        "r2_mean": r2s.mean(), "r2_std": r2s.std(),
        "mae_mean": maes.mean(), "mae_std": maes.std(),
        "folds": fold_results,
    }


def train_model():
    df = load_bootstrap_data()
    print(f"Training on REAL bootstrap data: {len(df)} rows from {DATA_FILE}")

    df = engineer_features(df)

    # ── Evaluation 1: naive random split (leaky, reported for contrast) ──
    naive = _evaluate_naive_split(df, FEATURE_COLS)
    print("\n-- Naive random 80/20 split (LEAKY — adjacent hours of the same "
          "corridor are near-duplicates, this overstates real performance) --")
    print(f"  R2  : {naive['r2']:.4f}")
    print(f"  MAE : {naive['mae']:.4f}")
    print(f"  n_train={naive['n_train']}  n_test={naive['n_test']}")

    # ── Evaluation 2: GroupKFold leave-one-corridor-out (honest) ──────────
    gkf_result = _evaluate_group_kfold(df, FEATURE_COLS, group_col="corridor_id")
    print("\n-- GroupKFold, leave-one-corridor-out (HONEST generalization "
          "estimate) --")
    for f in gkf_result["folds"]:
        c = corridors.by_id(f["held_out_corridor_id"])
        print(f"  held out corridor {f['held_out_corridor_id']:>2} "
              f"({c['name']:38s} / {c['road_class']:10s}) "
              f"R2={f['r2']:7.4f}  MAE={f['mae']:.4f}  n={f['n_test']}")
    print(f"  MEAN  R2  = {gkf_result['r2_mean']:.4f} +/- {gkf_result['r2_std']:.4f}")
    print(f"  MEAN  MAE = {gkf_result['mae_mean']:.4f} +/- {gkf_result['mae_std']:.4f}")

    # ── Does adding corridor_id as a feature help? ─────────────────────────
    feature_cols_with_corridor = FEATURE_COLS + ["corridor_id"]
    gkf_with_corridor = _evaluate_group_kfold(
        df, feature_cols_with_corridor, group_col="corridor_id"
    )
    print("\n-- Does adding corridor_id as a feature help? (GroupKFold "
          "leave-one-corridor-out) --")
    print(f"  WITHOUT corridor_id: mean R2 = {gkf_result['r2_mean']:.4f}")
    print(f"  WITH    corridor_id: mean R2 = {gkf_with_corridor['r2_mean']:.4f}")
    print("  NOTE: corridor_id is NOT shipped in the production model even if it")
    print("  helps here — backend/app.py calls predict_raw(model, features, hour,")
    print("  day_of_week, road_type) with no corridor_id argument, so the deployed")
    print("  feature set must stay road_class-based only, to match that contract.")
    if gkf_with_corridor["r2_mean"] > gkf_result["r2_mean"]:
        print(f"  -> corridor_id would improve leave-one-corridor-out R2 by "
              f"{gkf_with_corridor['r2_mean'] - gkf_result['r2_mean']:.4f}, but note that "
              f"this metric is somewhat unfair to corridor_id (each test fold's "
              f"corridor_id value is, by construction, unseen during that fold's "
              f"training, so this understates its value in normal in-sample use).")
    else:
        print("  -> corridor_id does not improve leave-one-corridor-out generalization "
              "(unsurprising: the model can't use a categorical value for a corridor "
              "it never saw in training).")

    # ── Final production model: fit on ALL data, fixed feature set ────────
    X_full = df[FEATURE_COLS]
    y_full = df["congestion_idx"]
    final_model = _fit_gbt(X_full, y_full)

    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
    print("\n-- Feature importances (final production model, trained on all data) --")
    print(importances.sort_values(ascending=False).to_string())

    model_version = f"gbt-{datetime.date.today().isoformat()}"
    metrics = {
        "r2": naive["r2"],
        "mae": naive["mae"],
        "cv_r2": gkf_result["r2_mean"],
        "cv_r2_std": gkf_result["r2_std"],
        "cv_mae": gkf_result["mae_mean"],
        "cv_mae_std": gkf_result["mae_std"],
    }
    payload = {
        "model": final_model,
        "features": FEATURE_COLS,
        "provenance": "bootstrap",
        "model_version": model_version,
        "trained_rows": len(df),
        "road_class_enc": corridors.ROAD_CLASS_ENC,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(payload, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"  model_version = {model_version}")
    print(f"  trained_rows  = {len(df)}")
    print(f"  metrics       = {metrics}")
    return final_model


# ─────────────────────────────────────────────
# 3. PREDICTION
# ─────────────────────────────────────────────

def predict_raw(model, feature_cols, hour, dow, road_class):
    """Signature intentionally matches the pre-existing contract used by
    backend/app.py: predict_raw(MODEL, MODEL_FEATURES, hour, day_of_week,
    road_type). Do not change this signature without updating backend/app.py.

    Uses corridors.ROAD_CLASS_ENC — the same mapping used at training time —
    instead of the old hardcoded (and wrong) {"highway":0,"arterial":1,
    "expressway":2} dict.
    """
    road_class_enc = corridors.ROAD_CLASS_ENC.get(
        road_class, corridors.ROAD_CLASS_ENC["arterial"]
    )
    row = {
        "hour": hour, "day_of_week": dow,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_weekend": int(dow >= 5),
        "is_peak_morning": int(7 <= hour <= 10),
        "is_peak_evening": int(17 <= hour <= 20),
        "road_class_enc": road_class_enc,
    }
    return float(model.predict(pd.DataFrame([row])[feature_cols])[0])


def predict(hour, day_of_week, road_class="arterial"):
    """Returns the raw congestion index only — NOT a human-readable label.
    Labelling (Free/Moderate/Heavy/Severe and their cutoffs) is owned by the
    API layer (see docs/api_contract.md); duplicating threshold constants
    here would just create another copy that can drift out of sync, the same
    class of bug that made predict_raw's road-class mapping wrong before."""
    payload = joblib.load(MODEL_FILE)
    model, feature_cols = payload["model"], payload["features"]
    idx = round(predict_raw(model, feature_cols, hour, day_of_week, road_class), 3)
    best_hour = min(
        range(24),
        key=lambda h: predict_raw(model, feature_cols, h, day_of_week, road_class),
    )
    return {"congestion_index": idx, "best_travel_hour": best_hour}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
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
            print(f"  {lbl:40s} -> congestion_index={r['congestion_index']:.3f}  "
                  f"best: {r['best_travel_hour']:02d}:00")

    elif mode == "predict":
        print("=== Gurugram Traffic Predictor - Live Predictions ===\n")
        now = datetime.datetime.now()
        print(f"Current time: {now.strftime('%A %H:%M')}\n")
        for c in corridors.CORRIDORS:
            r = predict(now.hour, now.weekday(), c["road_class"])
            print(f"  {c['name']:40s} -> congestion_index={r['congestion_index']:.3f}")

    else:
        print("Usage: python model/traffic_model.py [train|predict]")
        print("  (data collection now lives in bootstrap_collect.py at the repo root)")
