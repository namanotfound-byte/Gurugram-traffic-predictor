"""
Gurugram Traffic Predictor — Flask API v2
===========================================
Implements docs/api_contract.md exactly. See that file for the frozen field
names/types — this module must not diverge from it.

Endpoints:
  GET /health          -> service + model status
  GET /corridors       -> static corridor list (from corridors.py, never hardcoded)
  GET /predict          -> single (corridor, day, hour) cell
  GET /advice            -> "when should I go?" for one corridor/day
  GET /advice/all        -> /advice for all 8 corridors in one call
  GET /best-time         -> best departure inside a user time window
  GET /now                -> live verdict for all corridors, current IST time

Design notes:
  - The full 8x7x24 (corridor x day x hour) grid is predicted ONCE at startup
    and cached in memory (GRID). No endpoint calls model.predict() per request.
  - No hand-invented numbers. If no model is loaded, every model-backed
    endpoint returns HTTP 503 {"error": "no model trained yet"}. There is no
    PROFILES fallback table — that was the v1 bug this rebuild removes.
  - road_class encoding always comes from the model payload's own
    "road_class_enc" map when present, else corridors.ROAD_CLASS_ENC. Never
    hardcoded/reconstructed ad hoc (that mismatch was a real v1 bug).

Run locally:
  pip install -r requirements.txt
  python app.py
"""

import datetime
import math
import os
import sys
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python >=3.9 always has zoneinfo
    from backports.zoneinfo import ZoneInfo  # type: ignore

# ── Import the single source of truth for corridors ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from corridors import CORRIDORS, ROAD_CLASS_ENC  # noqa: E402
from model.forecast_model import lags_at_datetime  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_gbt.joblib")
FORECAST_MODEL_PATH = os.path.join(BASE_DIR, "models", "forecast_residual_gbt.joblib")
BOOTSTRAP_CSV = os.path.join(BASE_DIR, "data", "gurugram_bootstrap.csv")
OBSERVED_CSV = os.path.join(BASE_DIR, "data", "gurugram_observed.csv")

IST = ZoneInfo("Asia/Kolkata")

app = Flask(__name__)
CORS(app)

N_CORRIDORS = len(CORRIDORS)
VALID_CORRIDOR_IDS = {c["id"] for c in CORRIDORS}

# ── Real, off-peak (least-congested) TomTom free-flow reference ────────────
# Sourced from a live TomTom Routing API call (traffic=false, departAt Tue
# 2026-08-18 03:00 IST — a low-traffic reference time) against each
# corridor's exact start->end pair from corridors.py, run 2026-08-16 while
# building this API. This is real measured data, not invented. It is used
# ONLY as a last-resort fallback for converting an already-model-predicted
# congestion_index into real-world minutes, for any corridor not (yet)
# covered by data/gurugram_bootstrap.csv or data/gurugram_observed.csv.
FREE_FLOW_MINUTES_FALLBACK = {
    0: 33.20,  # NH-48 Delhi-Gurgaon Expressway
    1: 6.80,   # MG Road
    2: 12.37,  # Golf Course Road
    3: 11.17,  # Sohna Road
    4: 22.43,  # Dwarka Expressway
    5: 23.98,  # Golf Course Extension Road
    6: 16.30,  # Mehrauli-Gurgaon Road
    7: 14.93,  # Southern Peripheral Road
}

# Recalibrated 2026-08-16: real bootstrap data (1344 cells) peaks at
# congestion_index=0.435, so the original 0.35/0.60/0.80 thresholds
# (inherited from the synthetic generator, which ranged up to 0.92) made
# Heavy/Severe unreachable and labelled 98.5% of the real city "Free".
# See docs/api_contract.md "Conventions -> label" for the full rationale.
# Defined once, here, and used everywhere labels are derived — never
# duplicated/hardcoded per endpoint.
LABEL_THRESHOLDS = (
    (0.091, "Free"),
    (0.200, "Moderate"),
    (0.310, "Heavy"),
)


def label_for(idx: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if idx < threshold:
            return label
    return "Severe"


# Sourced verbatim from docs/accuracy_report.md (generated 2026-09-18 by
# tools/evaluate_accuracy.py, n=15234 observed rows / 100% cell coverage --
# see that file for the full methodology and confidence-tier caveats).
# NOT re-derived here -- these are the headline figures from that report,
# kept in sync by hand whenever the report is regenerated with a
# materially larger n. Surfaced via /health (and the static bundle) so the
# product's real strength (ranking hours against each other) and real
# weakness (matching an exact label) are both discoverable. These numbers
# compare TomTom historical bootstrap vs live observations -- NOT a GBT
# model-accuracy score.
ACCURACY_SUMMARY = {
    "label_agreement_pct": 43.0,
    "hour_ranking_concordance_pct": 57.2,
    "sample_size": 15234,
    "as_of": "2026-09-18",
    "note": (
        "Measured against 15,234 real TomTom live observations (100% cell "
        "coverage): compares the TomTom historical bootstrap each cell "
        "serves against what was actually observed for that same "
        "(corridor, day, hour) -- NOT a GradientBoosting model score. "
        "Hour-vs-hour ranking concordance is 57.2% (n=91 corridor/day "
        "groups); exact label match is 43.0%. Best-hour hit 13.2%, "
        "worst-hour hit 0.0%. Treat labels as typical day-of-week values, "
        "not a forecast for a specific date. Source: docs/accuracy_report.md."
    ),
}


def fmt_ampm(hour: int) -> str:
    hour = hour % 24
    period = "AM" if hour < 12 else "PM"
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    return f"{h12} {period}"


# ─────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────

def load_model_payload():
    """Load models/traffic_gbt.joblib and normalize its shape.

    Returns None if no usable model is present. Handles the legacy payload
    shape (only "model" + "features", from the pre-v2 synthetic-data
    pipeline) by honestly labeling it provenance="synthetic" rather than
    pretending it is the same as a real bootstrap/observed model.
    """
    try:
        payload = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[app] no usable model at {MODEL_PATH}: {e}")
        return None

    if not isinstance(payload, dict) or "model" not in payload or "features" not in payload:
        print(f"[app] model payload at {MODEL_PATH} is malformed; ignoring")
        return None

    new_shape_keys = ("provenance", "model_version", "trained_rows", "road_class_enc", "metrics")
    if all(k in payload for k in new_shape_keys):
        return payload

    # Legacy shape: {"model", "features"} only. This is the pre-Phase-2
    # payload, trained purely on model/traffic_model.py's synthetic data
    # generator. Label it accordingly rather than implying real data.
    try:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        version_stamp = mtime.strftime("%Y-%m-%d")
    except OSError:
        version_stamp = "unknown"
    print("[app] legacy model payload (model+features only) -> provenance=synthetic")
    return {
        "model": payload["model"],
        "features": payload["features"],
        "provenance": "synthetic",
        "model_version": f"gbt-legacy-{version_stamp}",
        "trained_rows": None,
        "road_class_enc": dict(ROAD_CLASS_ENC),
        "metrics": None,
    }


def load_free_flow_minutes():
    """Real free-flow minutes per corridor, preferring measured data.

    Priority:
      1. data/gurugram_bootstrap.csv  -> free_flow_s column (TomTom's
         no-traffic routing time), averaged per corridor.
      2. data/gurugram_observed.csv   -> same column, live snapshots.
      3. FREE_FLOW_MINUTES_FALLBACK   -> our own real TomTom measurement
         (see comment above), used only for corridors missing from both
         CSVs.
    Never fabricated.
    """
    ff_minutes = {}
    sources_used = {}
    for path, tag in ((BOOTSTRAP_CSV, "bootstrap_csv"), (OBSERVED_CSV, "observed_csv")):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[app] could not read {path}: {e}")
            continue
        if "corridor_id" not in df.columns or "free_flow_s" not in df.columns:
            continue
        grouped = df.groupby("corridor_id")["free_flow_s"].mean()
        for cid, secs in grouped.items():
            cid = int(cid)
            if cid not in ff_minutes:
                ff_minutes[cid] = round(float(secs) / 60.0, 2)
                sources_used[cid] = tag

    missing = [cid for cid in VALID_CORRIDOR_IDS if cid not in ff_minutes]
    for cid in missing:
        ff_minutes[cid] = FREE_FLOW_MINUTES_FALLBACK[cid]
        sources_used[cid] = "tomtom_fallback_constant"

    if missing:
        print(f"[app] free-flow minutes: corridors {sorted(missing)} used the TomTom fallback constant")
    print(f"[app] free-flow minutes sources: {sources_used}")
    return ff_minutes


def load_measured_grid():
    """Build the full measured-value lookup straight from real data.

    Every cell your API can be asked about should be served from an actual
    measurement when one exists -- the model is for filling gaps, not for
    lossily re-compressing a value we already hold exactly. Priority per
    (corridor_id, day, hour) cell:
      1. data/gurugram_observed.csv  -- freshest, live-measured by us.
      2. data/gurugram_bootstrap.csv -- TomTom historical-model measured.
         This CSV carries a `route_stable` column (TomTom occasionally
         routes a materially different, longer path for the same corridor
         across sweep timestamps) -- read it directly rather than
         re-deriving route stability ourselves.

    Returns {(corridor_id, day, hour): {"congestion_idx": float,
                                          "origin": "observed"|"bootstrap",
                                          "route_stable": bool}}
    Any (corridor, day, hour) NOT present here has no measurement and
    must fall back to model.predict() at grid-build time.
    """
    measured = {}

    if os.path.exists(BOOTSTRAP_CSV):
        try:
            df = pd.read_csv(BOOTSTRAP_CSV)
            has_stability_col = "route_stable" in df.columns
            for _, row in df.iterrows():
                cid, d, h = int(row["corridor_id"]), int(row["day_of_week"]), int(row["hour"])
                stable = bool(row["route_stable"]) if has_stability_col else True
                measured[(cid, d, h)] = {
                    "congestion_idx": float(row["congestion_idx"]),
                    "origin": "bootstrap",
                    "route_stable": stable,
                }
        except Exception as e:
            print(f"[app] could not read {BOOTSTRAP_CSV} for the measured grid: {e}")

    if os.path.exists(OBSERVED_CSV):
        try:
            df = pd.read_csv(OBSERVED_CSV)
            for _, row in df.iterrows():
                cid, d, h = int(row["corridor_id"]), int(row["day_of_week"]), int(row["hour"])
                measured[(cid, d, h)] = {
                    "congestion_idx": float(row["congestion_idx"]),
                    "origin": "observed",
                    "route_stable": True,  # a fresh live measurement is our best signal
                }
        except Exception as e:
            print(f"[app] could not read {OBSERVED_CSV} for the measured grid: {e}")

    return measured


def extract_within_class_quality(metrics):
    """A same-road-class / within-corridor quality figure, if the training
    pipeline has published one under a recognizable key. NEVER read
    metrics["cv_r2"] here: that figure is leave-one-CORRIDOR-out, and it is
    dominated by the two corridors that are the sole member of their road
    class -- Dwarka Expressway (the only "expressway", per-fold R2 -25.5)
    and NH-48 (the only "highway", R2 -0.08). The six arterial corridors,
    which do have same-class siblings to generalize from, individually
    average R2 ~0.90. We never serve a road class the model hasn't seen,
    so cross-corridor-class generalization is not the risk that matters
    for an inferred cell here.
    """
    if not metrics or not isinstance(metrics, dict):
        return None
    for key in ("within_corridor_r2", "within_class_r2", "same_class_r2", "holdout_hours_r2"):
        val = metrics.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


# Confidence tiers, strictly ordered: a live observation beats a stable
# bootstrap measurement beats an unstable one beats any model-inferred
# value -- never a flat constant, and cv_r2 (leave-one-corridor-out, not
# representative of this product) never enters the calculation.
CONFIDENCE_OBSERVED = 0.97
CONFIDENCE_MEASURED_STABLE = 0.92
CONFIDENCE_MEASURED_UNSTABLE = 0.50
INFERRED_CONFIDENCE_DEFAULT = 0.35
INFERRED_CONFIDENCE_CAP = 0.45  # even a strong within-class score can't outrank a real measurement
# Residual forecast: holdout skill maps to confidence between 0.55 and 0.85.
# Never outranks a direct measurement — this is a condition-adjusted forecast.
NEAREST_INCIDENT_SENTINEL_M = 5000.0


def compute_confidence(provenance, metrics, measured_cell, cell_origin=None, forecast_skill=None):
    """Honest per-cell confidence -- how trustworthy is the SERVED value.

    Priority:
      1. synthetic model -> flat low. No real data exists anywhere to
         differentiate cells by, so a flat number is the honest answer.
      2. cell has a live "observed" measurement -> highest (0.97).
      3. cell has a bootstrap measurement, route_stable -> high (0.92).
      4. cell has a bootstrap measurement, NOT route_stable (TomTom routed
         a materially different/longer path for that hour, so the
         free-flow/expected ratio isn't apples-to-apples) -> materially
         lower (0.50).
      5. cell origin is "residual_adjusted" -- weather/incident-conditioned
         forecast over the bootstrap baseline; confidence scales with the
         model's time-holdout skill score (see model/forecast_model.py).
      6. no measurement at all -- the GBT model fills the gap -> lower
         still, using a within-corridor/same-class quality figure if the
         training pipeline publishes one, else a conservative default.
    """
    if provenance == "synthetic":
        return 0.15

    if cell_origin == "residual_adjusted":
        if forecast_skill is None or forecast_skill <= 0:
            # Residual model refused or failed holdout — must not be served.
            return INFERRED_CONFIDENCE_DEFAULT
        return round(min(0.85, max(0.55, 0.55 + 0.35 * forecast_skill)), 2)

    if measured_cell is not None:
        if measured_cell["origin"] == "observed":
            return CONFIDENCE_OBSERVED
        return CONFIDENCE_MEASURED_STABLE if measured_cell["route_stable"] else CONFIDENCE_MEASURED_UNSTABLE

    quality = extract_within_class_quality(metrics)
    if quality is None:
        return INFERRED_CONFIDENCE_DEFAULT
    conf = 0.20 + 0.25 * max(0.0, min(1.0, quality))
    return round(min(INFERRED_CONFIDENCE_CAP, max(0.05, conf)), 2)


def minutes_from_index(free_flow_minutes: float, idx: float):
    idx_safe = min(idx, 0.97)  # guard against divide-by-zero blowup near 1.0
    typical = free_flow_minutes / (1.0 - idx_safe)
    delay = typical - free_flow_minutes
    return round(typical, 1), round(delay, 1)


def build_feature_row(hour: int, day: int, road_class: str, road_class_enc: dict) -> dict:
    return {
        "hour": hour,
        "day_of_week": day,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_weekend": int(day >= 5),
        "is_peak_morning": int(7 <= hour <= 10),
        "is_peak_evening": int(17 <= hour <= 20),
        "road_class_enc": road_class_enc[road_class],
    }


# ─────────────────────────────────────────────────────────────────────────
# Residual forecast model (model/forecast_model.py)
# ─────────────────────────────────────────────────────────────────────────

def load_forecast_payload():
    """Load models/forecast_residual_gbt.joblib only if holdout skill > 0.

    A non-positive skill score means the residual model does NOT beat the
    bootstrap baseline on the designed time holdout — serving it would be
    dishonest. Fail loudly (return None), never silently fall back.
    """
    try:
        payload = joblib.load(FORECAST_MODEL_PATH)
    except Exception as e:
        print(f"[app] no residual forecast model at {FORECAST_MODEL_PATH}: {e}")
        return None

    if not isinstance(payload, dict) or "model" not in payload or "features" not in payload:
        print(f"[app] forecast payload at {FORECAST_MODEL_PATH} is malformed; ignoring")
        return None

    skill = payload.get("skill_score")
    if skill is None:
        print("[app] forecast model artifact has no skill_score; NOT serving residual adjustments")
        return None
    try:
        skill = float(skill)
    except (TypeError, ValueError):
        print("[app] forecast model skill_score is not numeric; NOT serving residual adjustments")
        return None
    if skill <= 0:
        print(f"[app] forecast model holdout skill={skill:.4f} <= 0; NOT serving residual "
              f"adjustments (baseline alone is better or equal)")
        return None

    print(f"[app] residual forecast model loaded: skill={skill:.4f}, "
          f"holdout MAE model={payload.get('model_mae')} baseline={payload.get('baseline_mae')}")
    return payload


def load_baseline_grid():
    """Bootstrap baseline grid: {(corridor_id, day, hour): baseline_idx}.

    Prefers route_stable=True rows; fills any missing cell from the full
    bootstrap sweep so every corridor/day/hour has a baseline for forecasting.
    """
    baseline = {}
    if not os.path.exists(BOOTSTRAP_CSV):
        return baseline
    try:
        df = pd.read_csv(BOOTSTRAP_CSV)
    except Exception as e:
        print(f"[app] could not read {BOOTSTRAP_CSV} for baseline grid: {e}")
        return baseline

    def _to_bool(v):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1")

    stable_df = df
    if "route_stable" in df.columns:
        stable_df = df[df["route_stable"].map(_to_bool)]

    for source_df in (stable_df, df):
        grouped = (
            source_df.groupby(["corridor_id", "day_of_week", "hour"])["congestion_idx"]
            .mean()
            .reset_index()
        )
        for _, row in grouped.iterrows():
            key = (int(row["corridor_id"]), int(row["day_of_week"]), int(row["hour"]))
            if key not in baseline:
                baseline[key] = float(row["congestion_idx"])
    return baseline


def _neutral_incident_features():
    """No-incident defaults for grid gap-filling (not a live fetch)."""
    return {
        "incident_count": 0,
        "incident_total_delay_s": 0.0,
        "incident_max_magnitude": 0,
        "has_road_closure": False,
        "has_jam": False,
        "nearest_incident_m": NEAREST_INCIDENT_SENTINEL_M,
        "incident_data_known": 1,
    }


def target_date_for_weekday(day_of_week: int) -> datetime.date:
    """Next occurrence of weekday in IST (including today when it matches)."""
    today = datetime.datetime.now(IST).date()
    delta = (int(day_of_week) - today.weekday()) % 7
    return today + datetime.timedelta(days=delta)


def load_observed_for_lags():
    """Observed rows indexed for leak-safe lag lookup at inference."""
    if not os.path.exists(OBSERVED_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(OBSERVED_CSV)
    except Exception as e:
        print(f"[app] could not read {OBSERVED_CSV} for lag lookup: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True).dt.tz_convert(IST)
    df["date"] = df["collected_at"].dt.date
    df["corridor_id"] = df["corridor_id"].astype(int)
    df["hour"] = df["hour"].astype(int)
    return df.sort_values("collected_at")


def build_forecast_feature_row(corridor, day, hour, target_date, incident_feat,
                               lag_prior_idx=None, lag_week_idx=None, baseline_idx=0.0):
    """Feature vector matching model/forecast_model.py FEATURE_COLS."""
    try:
        import weather as wx  # noqa: WPS433 — lazy: optional at app import time
    except ImportError:
        wx = None
    w = (wx.get_hourly_weather(target_date, hour) or {}) if wx else {}
    e = wx.get_event_features(target_date) if wx else {
        "is_holiday": False,
        "is_festival_period": False,
        "is_month_end": int(target_date.day >= 28),
        "days_to_nearest_holiday": 999,
    }

    def _bool_i(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        if isinstance(val, bool):
            return int(val)
        return int(str(val).strip().lower() in ("true", "1"))

    vis = w.get("visibility_m")
    if vis is None or (isinstance(vis, float) and np.isnan(vis)):
        vis = 8000.0
    temp = w.get("temperature_c")
    if temp is None or (isinstance(temp, float) and np.isnan(temp)):
        temp = 28.0

    nearest_m = incident_feat.get("nearest_incident_m")
    if nearest_m is None:
        nearest_m = NEAREST_INCIDENT_SENTINEL_M

    days_hol = e.get("days_to_nearest_holiday")
    if days_hol is None:
        days_hol = 999

    if lag_prior_idx is None:
        lag_prior_idx = baseline_idx
    if lag_week_idx is None:
        lag_week_idx = baseline_idx

    return {
        "temperature_c": float(temp),
        "precipitation_mm": float(w.get("precipitation_mm") or 0.0),
        "is_raining_i": _bool_i(w.get("is_raining")),
        "rain_last_3h": float(w.get("rain_last_3h") or 0.0),
        "visibility_m": float(vis),
        "low_visibility_i": _bool_i(w.get("low_visibility")),
        "is_holiday_i": _bool_i(e.get("is_holiday")),
        "is_festival_period_i": _bool_i(e.get("is_festival_period")),
        "is_month_end_i": _bool_i(e.get("is_month_end")),
        "days_to_nearest_holiday": int(days_hol),
        "road_class_enc": ROAD_CLASS_ENC[corridor["road_class"]],
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_weekend": int(day >= 5),
        "corridor_id": int(corridor["id"]),
        "lag_prior_idx": float(lag_prior_idx),
        "lag_week_idx": float(lag_week_idx),
        "incident_count": int(incident_feat.get("incident_count") or 0),
        "incident_total_delay_s": float(incident_feat.get("incident_total_delay_s") or 0.0),
        "incident_max_magnitude": int(incident_feat.get("incident_max_magnitude") or 0),
        "has_road_closure_i": int(bool(incident_feat.get("has_road_closure"))),
        "has_jam_i": int(bool(incident_feat.get("has_jam"))),
        "nearest_incident_m": float(nearest_m),
        "incident_data_known": int(incident_feat.get("incident_data_known", 1)),
    }


def _incident_features_for_date(target_date, live_incidents_by_cid):
    """Live incidents for today; neutral + unknown flag for other dates."""
    today = datetime.datetime.now(IST).date()
    if target_date == today and live_incidents_by_cid:
        return live_incidents_by_cid
    neutral = _neutral_incident_features()
    neutral["incident_data_known"] = 0
    return {cid: dict(neutral) for cid in VALID_CORRIDOR_IDS}


def residual_forecast_idx(corridor, day, hour, target_date, incident_feat,
                          baseline_grid, forecast_model, forecast_features,
                          observed_df=None, dwarka_override=None):
    """baseline + predicted_residual, clipped to [0, 1]. Returns None if no baseline."""
    cell = (corridor["id"], day, hour)
    baseline = baseline_grid.get(cell)
    if baseline is None:
        return None

    target_dt = datetime.datetime.combine(target_date, datetime.time(hour, 0), tzinfo=IST)
    lag_prior, lag_week = lags_at_datetime(
        observed_df, corridor["id"], day, hour, target_dt, baseline_idx=baseline,
    )
    feat_row = build_forecast_feature_row(
        corridor, day, hour, target_date, incident_feat,
        lag_prior_idx=lag_prior, lag_week_idx=lag_week,
    )
    X = pd.DataFrame([feat_row])[forecast_features]
    model = dwarka_override if (dwarka_override is not None and corridor["id"] == 4) else forecast_model
    pred_resid = float(model.predict(X)[0])
    return round(float(np.clip(baseline + pred_resid, 0.0, 1.0)), 3)


def forecast_explanation(baseline, forecast_idx, feat_row):
    """Template English from numbers — no LLM."""
    residual = round(forecast_idx - baseline, 3)
    parts = [
        f"Typical baseline {baseline:.2f} for this corridor/day/hour",
    ]
    if abs(residual) >= 0.005:
        direction = "higher" if residual > 0 else "lower"
        parts.append(f"adjusted {direction} by {abs(residual):.2f} for conditions")
    else:
        parts.append("conditions match the typical pattern")
    if feat_row.get("is_raining_i"):
        parts.append("rain flagged")
    if feat_row.get("incident_count", 0) > 0 and feat_row.get("incident_data_known"):
        parts.append(f"{int(feat_row['incident_count'])} nearby incident(s)")
    elif not feat_row.get("incident_data_known"):
        parts.append("incident status unknown for this date")
    return "; ".join(parts) + "."


def fetch_live_incident_features():
    """Per-corridor incident features for /now (one TomTom bbox request)."""
    try:
        import incidents as inc  # noqa: WPS433
        feats, _n = inc.get_corridor_incident_features()
    except Exception as e:
        print(f"[app] live incident fetch failed ({e}); using no-incident defaults for /now")
        return {}
    out = {}
    for cid, f in feats.items():
        out[int(cid)] = {
            **f,
            "incident_data_known": 1,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
# Startup: load model, load real-data backing, precompute the full grid
# ─────────────────────────────────────────────────────────────────────────

MODEL_PAYLOAD = load_model_payload()
MODEL_READY = MODEL_PAYLOAD is not None

FORECAST_PAYLOAD = load_forecast_payload()
FORECAST_READY = FORECAST_PAYLOAD is not None
FORECAST_SKILL = FORECAST_PAYLOAD["skill_score"] if FORECAST_READY else None
FORECAST_MODEL = FORECAST_PAYLOAD["model"] if FORECAST_READY else None
FORECAST_FEATURES = FORECAST_PAYLOAD["features"] if FORECAST_READY else None
FORECAST_DWARKA_OVERRIDE = (
    FORECAST_PAYLOAD.get("dwarka_override") if FORECAST_READY else None
)
OBSERVED_LAG_DF = load_observed_for_lags()
FORECAST_HOLDOUT = (
    {
        "model_mae": FORECAST_PAYLOAD.get("model_mae"),
        "baseline_mae": FORECAST_PAYLOAD.get("baseline_mae"),
        "skill_score": FORECAST_PAYLOAD.get("skill_score"),
        "trained_at": FORECAST_PAYLOAD.get("trained_at"),
    }
    if FORECAST_READY else None
)

MODEL_VERSION = None
MODEL_PROVENANCE = None
TRAINED_ROWS = None
GRID = {}  # (corridor_id, day, hour) -> dict
BASELINE_GRID = {}

if MODEL_READY:
    MODEL = MODEL_PAYLOAD["model"]
    FEATURES = MODEL_PAYLOAD["features"]
    MODEL_VERSION = MODEL_PAYLOAD["model_version"]
    MODEL_PROVENANCE = MODEL_PAYLOAD["provenance"]
    TRAINED_ROWS = MODEL_PAYLOAD["trained_rows"]
    METRICS = MODEL_PAYLOAD.get("metrics")
    ROAD_CLASS_ENC_USED = MODEL_PAYLOAD.get("road_class_enc") or dict(ROAD_CLASS_ENC)
else:
    MODEL = None
    FEATURES = []
    MODEL_VERSION = None
    MODEL_PROVENANCE = None
    TRAINED_ROWS = None
    METRICS = None
    ROAD_CLASS_ENC_USED = dict(ROAD_CLASS_ENC)
    print(f"[app] no legacy GBT at {MODEL_PATH}; grid uses residual forecast only")

FREE_FLOW_MINUTES = load_free_flow_minutes()
MEASURED_GRID = load_measured_grid()  # training/evaluation only — not served as forecasts
BASELINE_GRID = load_baseline_grid()

GRID_READY = FORECAST_READY
if GRID_READY:
    try:
        import weather as wx  # noqa: WPS433

        target_dates = [target_date_for_weekday(d) for d in range(7)]
        wx.get_weather_range(min(target_dates), max(target_dates))
        today = datetime.datetime.now(IST).date()
        live_incidents_today = fetch_live_incident_features() if today in target_dates else {}

        residual_count = 0
        bootstrap_fallback_count = 0
        corridor_by_id = {c["id"]: c for c in CORRIDORS}

        for c in CORRIDORS:
            for day in range(7):
                target_date = target_date_for_weekday(day)
                inc_by_cid = _incident_features_for_date(target_date, live_incidents_today)
                for hour in range(24):
                    cell = (c["id"], day, hour)
                    inc_feat = inc_by_cid.get(c["id"], _neutral_incident_features())
                    adj = residual_forecast_idx(
                        c, day, hour, target_date, inc_feat,
                        BASELINE_GRID, FORECAST_MODEL, FORECAST_FEATURES,
                        observed_df=OBSERVED_LAG_DF, dwarka_override=FORECAST_DWARKA_OVERRIDE,
                    )
                    baseline = BASELINE_GRID.get(cell)
                    if adj is not None:
                        idx = adj
                        origin = "residual_adjusted"
                        residual_count += 1
                    elif baseline is not None:
                        idx = round(float(baseline), 3)
                        origin = "bootstrap"
                        bootstrap_fallback_count += 1
                    else:
                        raise RuntimeError(f"no baseline or forecast for cell {cell}")

                    ff_minutes = FREE_FLOW_MINUTES[c["id"]]
                    typical, delay = minutes_from_index(ff_minutes, idx)
                    conf = compute_confidence(
                        MODEL_PROVENANCE or "bootstrap", METRICS, None,
                        cell_origin=origin, forecast_skill=FORECAST_SKILL,
                    )
                    GRID[cell] = {
                        "congestion_index": idx,
                        "label": label_for(idx),
                        "free_flow_minutes": round(ff_minutes, 1),
                        "typical_minutes": typical,
                        "delay_minutes": delay,
                        "confidence": conf,
                        "origin": origin,
                    }

        parts = [f"{residual_count} residual-adjusted"]
        if bootstrap_fallback_count:
            parts.append(f"{bootstrap_fallback_count} bootstrap fallback")
        print(f"[app] precomputed forecast grid: {len(GRID)} cells ({', '.join(parts)}), "
              f"forecast_skill={FORECAST_SKILL:.4f}")
    except Exception as e:
        print(f"[app] FAILED to precompute forecast grid ({e}); disabling model-backed endpoints")
        GRID_READY = False
        GRID = {}
else:
    print("[app] residual forecast model not ready (missing or skill <= 0); "
          "model-backed endpoints will return 503")


# ─────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────

def parse_query_int(name: str, lo: int, hi: int, required: bool = True, default=None):
    """Returns (value, error_message). error_message is None on success."""
    raw = request.args.get(name)
    if raw is None or raw == "":
        if required:
            return None, f"missing required parameter '{name}'"
        return default, None
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return None, f"'{name}' must be an integer, got '{raw}'"
    if not (lo <= val <= hi):
        return None, f"'{name}' must be between {lo} and {hi}, got {val}"
    return val, None


def require_model(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not GRID_READY:
            return jsonify({"error": "no forecast model trained yet"}), 503
        return f(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────
# Window detection (contiguous good/bad hour runs, wrap-aware)
# ─────────────────────────────────────────────────────────────────────────

def _merge_runs_raw(flags):
    """flags: list[bool] length n -> list of (start_index, end_index) inclusive
    contiguous True-runs, with NO wrap merge between the first and last index.
    Shared by both the full 24h (circular) and period-local (linear) window
    detectors below."""
    n = len(flags)
    if all(flags):
        return [(0, n - 1)]
    if not any(flags):
        return []

    runs = []
    start = None
    for h in range(n):
        if flags[h] and start is None:
            start = h
        if not flags[h] and start is not None:
            runs.append((start, h - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    return runs


def _merge_runs(flags):
    """flags: list[bool] length 24 -> list of (start_hour, end_hour) inclusive,
    with a run touching both hour 0 and hour 23 merged into one wrapping run.
    Circular -- appropriate for a full 24h clock, where hour 23 and hour 0
    really are adjacent."""
    n = len(flags)
    runs = _merge_runs_raw(flags)

    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        first = runs.pop(0)
        last = runs.pop(-1)
        runs.append((last[0], first[1]))  # wrapping run: e.g. (22, 5)

    return runs


def _window_hours(start, end):
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 24)) + list(range(0, end + 1))


def _window_text(start, end, kind):
    verb = "Clear" if kind == "best" else "Avoid"
    wraps = start > end
    if start == 0 and end == 23:
        return f"{verb} all day"
    if wraps:
        if kind == "best" and end <= 6:
            return f"Clear after {fmt_ampm(start)}"
        return f"{verb} {fmt_ampm(start)}-{fmt_ampm(end)}"
    if kind == "best" and start == 0 and end < 23:
        return f"Clear before {fmt_ampm(end + 1)}"
    if kind == "best" and end == 23 and start > 0:
        return f"Clear after {fmt_ampm(start)}"
    if start == end:
        return f"{verb} around {fmt_ampm(start)}"
    return f"{verb} {fmt_ampm(start)}-{fmt_ampm(end)}"


def find_windows(profile):
    """Collapse a 24-hour congestion_index profile into contiguous best/worst
    windows, relative to that corridor/day's own range (not a fixed global
    threshold) so a mild day still surfaces its relatively-worst hours."""
    lo, hi = min(profile), max(profile)
    span = hi - lo
    if span < 1e-9:
        return [], []  # flat profile: no meaningful window to call out

    low_thr = lo + 0.15 * span
    high_thr = hi - 0.15 * span

    best_runs = _merge_runs([profile[h] <= low_thr for h in range(24)])
    worst_runs = _merge_runs([profile[h] >= high_thr for h in range(24)])

    def build(runs, kind):
        out = []
        for start, end in runs:
            hours = _window_hours(start, end)
            avg = round(sum(profile[h] for h in hours) / len(hours), 3)
            out.append({
                "start_hour": start, "end_hour": end,
                "avg_index": avg, "label": label_for(avg),
                "text": _window_text(start, end, kind),
            })
        return out

    best_windows = build(best_runs, "best")
    worst_windows = build(worst_runs, "worst")
    best_windows.sort(key=lambda w: w["avg_index"])
    worst_windows.sort(key=lambda w: -w["avg_index"])
    return best_windows, worst_windows


# ─────────────────────────────────────────────────────────────────────────
# Day/night split (added 2026-08-17)
# ─────────────────────────────────────────────────────────────────────────
# The whole-day best_hour is midnight for essentially every corridor --
# roads are simply empty overnight -- which made the whole-day "best time"
# figure look identical and useless for every corridor to a daytime
# traveller (real user complaint: "it always shows that night is free...
# going in at night is not viable"). The boundary below is derived from the
# actual measured grid, not asserted: averaging congestion_index across all
# 13 corridors x 7 days at each hour (see GRID) shows a near-zero, flat
# floor overnight (0.0005-0.0009 avg, max <=0.009 -- i.e. every corridor
# reads "Free") that holds from 22:00 through 03:00, a small uptick at
# 04:00-05:00 (avg <=0.012, still Free-band), then a sharp order-of-magnitude
# climb starting 06:00 (avg 0.019) and steepening fast by 07:00-08:00 (avg
# 0.054 -> 0.114). The evening side shows the mirror-image cliff: 21:00
# still averages 0.087 (max 0.15) but 22:00 drops to 0.0006 (max 0.009) --
# a >100x collapse in one hour. So 22:00-05:59 is where congestion is
# structurally, measurably absent across the whole dataset, and 06:00-21:59
# is where it actually varies by corridor and hour -- which is exactly the
# window a daytime traveller needs advice about. Night advice is still
# served (truck/shift-worker use case), just as its own explicit period
# rather than silently winning every "best time" comparison.
DAY_HOURS = list(range(6, 22))                       # 06:00-21:59
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))  # 22:00-05:59, in real-time order


def find_windows_for_hours(profile, hours):
    """Like find_windows, but restricted to an explicit, already-ordered list
    of hours (e.g. DAY_HOURS or NIGHT_HOURS) rather than the full 24h clock.

    Uses the non-circular run merge (_merge_runs_raw): the first and last
    hour of a period (e.g. hour 6 and hour 21 for "day") are NOT adjacent in
    real time -- there's a whole other period in between -- so, unlike
    find_windows's full-clock wrap, runs must never be merged across that
    boundary. Hour ordering within NIGHT_HOURS (22,23,0,...,5) is already
    real-time-contiguous, so a run spanning the whole period still comes out
    as a correctly-wrapping (start_hour > end_hour) window via _window_text.
    """
    sub = [profile[h] for h in hours]
    lo, hi = min(sub), max(sub)
    span = hi - lo
    if span < 1e-9:
        return [], []

    low_thr = lo + 0.15 * span
    high_thr = hi - 0.15 * span

    best_runs = _merge_runs_raw([sub[i] <= low_thr for i in range(len(sub))])
    worst_runs = _merge_runs_raw([sub[i] >= high_thr for i in range(len(sub))])

    def build(runs, kind):
        out = []
        for start_i, end_i in runs:
            hrs = hours[start_i:end_i + 1]
            avg = round(sum(profile[h] for h in hrs) / len(hrs), 3)
            start_hour, end_hour = hrs[0], hrs[-1]
            out.append({
                "start_hour": start_hour, "end_hour": end_hour,
                "avg_index": avg, "label": label_for(avg),
                "text": _window_text(start_hour, end_hour, kind),
            })
        return out

    best_windows = build(best_runs, "best")
    worst_windows = build(worst_runs, "worst")
    best_windows.sort(key=lambda w: w["avg_index"])
    worst_windows.sort(key=lambda w: -w["avg_index"])
    return best_windows, worst_windows


_PERIOD_LABELS = {"day": "daytime", "night": "nighttime", "any": "all-day"}


def _period_summary_text(period_name, best_hour, worst_hour, worst_delay_minutes, worst_delay_pct, saving_minutes):
    period_label = _PERIOD_LABELS.get(period_name, period_name)
    if saving_minutes >= 0.5:
        return (f"Best {period_label} departure: {fmt_ampm(best_hour)}. "
                f"Avoid {fmt_ampm(worst_hour)} ({worst_delay_minutes:+.0f} min, {worst_delay_pct:.0f}% longer).")
    return f"Traffic barely varies across {period_label} hours today."


def period_payload_for(corridor_id: int, day: int, profile, hours, period_name: str) -> dict:
    """Best/worst hour, windows, and saving figures restricted to one period
    (day/night) of a corridor/day's profile. Reuses find_windows_for_hours
    and the same GRID-derived delay/pct math as the whole-day figures --
    nothing here is a parallel reimplementation."""
    best_windows, worst_windows = find_windows_for_hours(profile, hours)
    best_hour = min(hours, key=lambda h: profile[h])
    worst_hour = max(hours, key=lambda h: profile[h])
    best_cell = GRID[(corridor_id, day, best_hour)]
    worst_cell = GRID[(corridor_id, day, worst_hour)]

    saving_minutes = round(worst_cell["delay_minutes"] - best_cell["delay_minutes"], 1)
    saving_pct = (
        round(saving_minutes / worst_cell["typical_minutes"] * 100, 1)
        if worst_cell["typical_minutes"] > 0 else 0.0
    )
    worst_delay_pct = (
        round(worst_cell["delay_minutes"] / worst_cell["free_flow_minutes"] * 100, 1)
        if worst_cell["free_flow_minutes"] > 0 else 0.0
    )

    summary = _period_summary_text(period_name, best_hour, worst_hour,
                                    worst_cell["delay_minutes"], worst_delay_pct, saving_minutes)
    if worst_cell["confidence"] < 0.5:
        summary += " (Limited data for this corridor/day — treat as a rough guide.)"

    return {
        "period": period_name,
        "start_hour": hours[0],
        "end_hour": hours[-1],
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "best_hour_delay_minutes": best_cell["delay_minutes"],
        "worst_hour_delay_minutes": worst_cell["delay_minutes"],
        "worst_hour_delay_pct": worst_delay_pct,
        "saving_minutes": saving_minutes,
        "saving_pct": saving_pct,
        "best_windows": best_windows,
        "worst_windows": worst_windows,
        "summary": summary,
        "confidence": worst_cell["confidence"],
    }


def build_advice_summary(best_windows, peak_hour, peak_delay_minutes,
                          whole_day_saving_minutes, whole_day_saving_pct,
                          peak_delay_pct, confidence):
    if not best_windows:
        leave = "No clearly free window today"
    else:
        w = best_windows[0]
        s, e = w["start_hour"], w["end_hour"]
        if s > e:  # wraps past midnight
            before_h = (e + 1) % 24
            leave = f"Leave before {fmt_ampm(before_h)} or after {fmt_ampm(s)}"
        elif s == 0 and e == 23:
            leave = "Leave anytime today — it stays clear"
        elif s == 0:
            leave = f"Leave before {fmt_ampm(e + 1)}"
        elif e == 23:
            leave = f"Leave after {fmt_ampm(s)}"
        else:
            leave = f"Leave between {fmt_ampm(s)} and {fmt_ampm(e)}"
    worst = f"Worst is {fmt_ampm(peak_hour)} ({peak_delay_minutes:+.0f} min, {peak_delay_pct:.0f}% longer than free-flow)."

    if whole_day_saving_minutes >= 0.5:
        saving = (f" Timing it right saves ~{whole_day_saving_minutes:.0f} min "
                  f"({whole_day_saving_pct:.0f}% shorter trip) versus the worst hour.")
    else:
        saving = " Traffic barely varies by hour on this corridor today."

    caveat = ""
    if confidence < 0.5:
        caveat = " (Limited data for this corridor/day — treat as a rough guide.)"

    return f"{leave}. {worst}{saving}{caveat}"


def advice_payload_for(corridor_id: int, day: int) -> dict:
    profile = [GRID[(corridor_id, day, h)]["congestion_index"] for h in range(24)]
    best_windows, worst_windows = find_windows(profile)
    best_hour = int(np.argmin(profile))
    peak_hour = int(np.argmax(profile))
    best_cell = GRID[(corridor_id, day, best_hour)]
    peak_cell = GRID[(corridor_id, day, peak_hour)]
    best_hour_delay = best_cell["delay_minutes"]
    peak_delay = peak_cell["delay_minutes"]

    whole_day_saving_minutes = round(peak_delay - best_hour_delay, 1)
    whole_day_saving_pct = (
        round(whole_day_saving_minutes / peak_cell["typical_minutes"] * 100, 1)
        if peak_cell["typical_minutes"] > 0 else 0.0
    )
    peak_delay_pct = (
        round(peak_delay / peak_cell["free_flow_minutes"] * 100, 1)
        if peak_cell["free_flow_minutes"] > 0 else 0.0
    )

    confidence = GRID[(corridor_id, day, peak_hour)]["confidence"]
    summary = build_advice_summary(
        best_windows, peak_hour, peak_delay,
        whole_day_saving_minutes, whole_day_saving_pct, peak_delay_pct, confidence,
    )
    peak_cell_origin = GRID[(corridor_id, day, peak_hour)]["origin"]
    if peak_cell_origin == "residual_adjusted":
        baseline = BASELINE_GRID.get((corridor_id, day, peak_hour))
        if baseline is not None:
            summary += f" Peak-hour forecast: baseline {baseline:.2f}, adjusted to {peak_cell['congestion_index']:.2f}."

    # Day/night split (added 2026-08-17) -- see DAY_HOURS/NIGHT_HOURS above
    # for why this boundary. Additive: every field above is unchanged, for
    # backwards compatibility with the frontend build already in progress.
    day_period = period_payload_for(corridor_id, day, profile, DAY_HOURS, "day")
    night_period = period_payload_for(corridor_id, day, profile, NIGHT_HOURS, "night")

    profile_origins = [GRID[(corridor_id, day, h)]["origin"] for h in range(24)]

    return {
        "corridor_id": corridor_id,
        "profile": profile,
        "profile_origins": profile_origins,
        "best_windows": best_windows,
        "worst_windows": worst_windows,
        "best_hour": best_hour,
        "peak_hour": peak_hour,
        "best_hour_delay_minutes": best_hour_delay,
        "peak_delay_minutes": peak_delay,
        "peak_delay_pct": peak_delay_pct,
        "whole_day_saving_minutes": whole_day_saving_minutes,
        "whole_day_saving_pct": whole_day_saving_pct,
        "summary": summary,
        "confidence": confidence,
        "day_period": day_period,
        "night_period": night_period,
    }


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    body = {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "provenance": MODEL_PROVENANCE,
        "corridors": N_CORRIDORS,
        "trained_rows": TRAINED_ROWS,
        "accuracy": ACCURACY_SUMMARY,
    }
    if GRID_READY:
        body["forecast"] = {
            "enabled": True,
            "skill_score": round(float(FORECAST_SKILL), 4),
            "holdout": FORECAST_HOLDOUT,
        }
    else:
        body["forecast"] = {"enabled": False}
    return jsonify(body)


@app.route("/corridors")
def corridors_list():
    return jsonify({"corridors": [
        {
            "id": c["id"],
            "name": c["name"],
            "sub": c["sub"],
            "road_class": c["road_class"],
            "start": list(c["start"]),
            "end": list(c["end"]),
            "length_km": c["verified_km"],
        }
        for c in CORRIDORS
    ]})


@app.route("/predict")
@require_model
def predict():
    corridor_id, err = parse_query_int("corridor", 0, N_CORRIDORS - 1)
    if err:
        return jsonify({"error": err}), 400
    day, err = parse_query_int("day", 0, 6)
    if err:
        return jsonify({"error": err}), 400
    hour, err = parse_query_int("hour", 0, 23)
    if err:
        return jsonify({"error": err}), 400

    cell = GRID[(corridor_id, day, hour)]
    return jsonify({
        "corridor_id": corridor_id,
        "day": day,
        "hour": hour,
        "congestion_index": cell["congestion_index"],
        "label": cell["label"],
        "delay_minutes": cell["delay_minutes"],
        "typical_minutes": cell["typical_minutes"],
        "free_flow_minutes": cell["free_flow_minutes"],
        "provenance": MODEL_PROVENANCE,
        "origin": cell["origin"],
        "confidence": cell["confidence"],
        "model_version": MODEL_VERSION,
    })


@app.route("/advice")
@require_model
def advice():
    corridor_id, err = parse_query_int("corridor", 0, N_CORRIDORS - 1)
    if err:
        return jsonify({"error": err}), 400
    day, err = parse_query_int("day", 0, 6)
    if err:
        return jsonify({"error": err}), 400

    payload = advice_payload_for(corridor_id, day)
    payload["day"] = day
    payload["provenance"] = MODEL_PROVENANCE
    return jsonify(payload)


@app.route("/advice/all")
@require_model
def advice_all():
    day, err = parse_query_int("day", 0, 6)
    if err:
        return jsonify({"error": err}), 400

    corridors_out = [advice_payload_for(c["id"], day) for c in CORRIDORS]
    return jsonify({
        "day": day,
        "corridors": corridors_out,
        "provenance": MODEL_PROVENANCE,
        "model_version": MODEL_VERSION,
    })


@app.route("/best-time")
@require_model
def best_time():
    corridor_id, err = parse_query_int("corridor", 0, N_CORRIDORS - 1)
    if err:
        return jsonify({"error": err}), 400
    day, err = parse_query_int("day", 0, 6)
    if err:
        return jsonify({"error": err}), 400

    # Added 2026-08-17: an explicit period ("day"/"night"/"any") as an
    # alternative to earliest/latest -- see DAY_HOURS/NIGHT_HOURS above.
    # Without this, the whole-day scan silently recommends midnight for
    # almost every corridor (roads are empty overnight), which is
    # technically correct but useless to a daytime commuter. earliest/latest
    # keep working exactly as before when period is omitted.
    period = request.args.get("period")
    if period is not None and period not in ("day", "night", "any"):
        return jsonify({"error": f"'period' must be one of day, night, any, got '{period}'"}), 400

    if period == "day":
        hours = DAY_HOURS
        period_value = "day"
        earliest, latest = hours[0], hours[-1]
    elif period == "night":
        hours = NIGHT_HOURS
        period_value = "night"
        earliest, latest = hours[0], hours[-1]
    elif period == "any":
        hours = list(range(24))
        period_value = "any"
        earliest, latest = hours[0], hours[-1]
    else:
        earliest, err = parse_query_int("earliest", 0, 23)
        if err:
            return jsonify({"error": err}), 400
        latest, err = parse_query_int("latest", 0, 23)
        if err:
            return jsonify({"error": err}), 400
        period_value = "custom"
        if latest < earliest:
            hours = list(range(earliest, 24)) + list(range(0, latest + 1))
        else:
            hours = list(range(earliest, latest + 1))

    def idx_of(h):
        return GRID[(corridor_id, day, h)]["congestion_index"]

    recommended_hour = min(hours, key=idx_of)
    worst_hour = max(hours, key=idx_of)
    rec_cell = GRID[(corridor_id, day, recommended_hour)]
    worst_cell = GRID[(corridor_id, day, worst_hour)]

    saving = round(worst_cell["delay_minutes"] - rec_cell["delay_minutes"], 1)
    saving_pct = (
        round(saving / worst_cell["typical_minutes"] * 100, 1)
        if worst_cell["typical_minutes"] > 0 else 0.0
    )

    other_hours = sorted((h for h in hours if h != recommended_hour), key=idx_of)
    alternatives = [
        {
            "hour": h,
            "congestion_index": GRID[(corridor_id, day, h)]["congestion_index"],
            "delay_minutes": GRID[(corridor_id, day, h)]["delay_minutes"],
        }
        for h in other_hours[:3]
    ]

    # Whole-day (unconstrained) best/worst for this corridor/day, so a small
    # window-limited saving can be reported honestly as "within your window"
    # rather than implying that's the best this corridor can ever do -- see
    # docs/api_contract.md "best-time" for the rationale (a narrow window can
    # bury a much larger saving available just outside it).
    all_hours = list(range(24))
    day_idx_of = lambda h: GRID[(corridor_id, day, h)]["congestion_index"]  # noqa: E731
    whole_day_best_hour = min(all_hours, key=day_idx_of)
    whole_day_worst_hour = max(all_hours, key=day_idx_of)
    day_best_cell = GRID[(corridor_id, day, whole_day_best_hour)]
    day_worst_cell = GRID[(corridor_id, day, whole_day_worst_hour)]
    whole_day_saving_minutes = round(day_worst_cell["delay_minutes"] - day_best_cell["delay_minutes"], 1)
    whole_day_saving_pct = (
        round(whole_day_saving_minutes / day_worst_cell["typical_minutes"] * 100, 1)
        if day_worst_cell["typical_minutes"] > 0 else 0.0
    )
    if period_value == "custom":
        # Legacy explicit earliest/latest path -- unchanged. True if a
        # strictly better hour exists outside the user's chosen window.
        window_constrained = (
            whole_day_best_hour not in hours
            and whole_day_saving_minutes > saving + 0.5
        )
        if window_constrained:
            lead = (f"Best overall on this day: avoid {fmt_ampm(whole_day_worst_hour)}, leave around "
                     f"{fmt_ampm(whole_day_best_hour)} instead — saves ~{whole_day_saving_minutes:.0f} min "
                     f"({whole_day_saving_pct:.0f}% shorter trip) across the full day.")
            if saving >= 0.5:
                window_part = (f" Within your {fmt_ampm(earliest)}-{fmt_ampm(latest)} window, leave at "
                                f"{fmt_ampm(recommended_hour)}: saves ~{saving:.0f} min ({saving_pct:.0f}%) vs "
                                f"{fmt_ampm(worst_hour)} within that window, but a bigger saving is available "
                                f"outside it.")
            else:
                window_part = (f" Within your {fmt_ampm(earliest)}-{fmt_ampm(latest)} window, traffic is about "
                                f"the same throughout — widen your window for a bigger saving.")
            summary = lead + window_part
        elif saving >= 0.5:
            summary = (f"Of {fmt_ampm(earliest)}-{fmt_ampm(latest)}, leave at {fmt_ampm(recommended_hour)}. "
                       f"Saves ~{saving:.0f} min ({saving_pct:.0f}%) vs leaving at {fmt_ampm(worst_hour)}.")
        else:
            summary = (f"Of {fmt_ampm(earliest)}-{fmt_ampm(latest)}, leave at {fmt_ampm(recommended_hour)}. "
                       f"Traffic is about the same all through this window.")
    else:
        # Explicit period ("day"/"night"/"any") -- the caller deliberately
        # scoped the search, so never second-guess it by comparing back
        # against the unconstrained whole day (that comparison is exactly
        # what made night silently "win" every time before this feature
        # existed). Reuses the same phrasing as /advice's day_period /
        # night_period blocks.
        window_constrained = False
        worst_delay_pct = (
            round(worst_cell["delay_minutes"] / worst_cell["free_flow_minutes"] * 100, 1)
            if worst_cell["free_flow_minutes"] > 0 else 0.0
        )
        summary = _period_summary_text(period_value, recommended_hour, worst_hour,
                                        worst_cell["delay_minutes"], worst_delay_pct, saving)

    if rec_cell["confidence"] < 0.5:
        summary += " (Limited data for this corridor/day — treat as a rough guide.)"

    return jsonify({
        "corridor_id": corridor_id,
        "day": day,
        "period": period_value,
        "earliest": earliest,
        "latest": latest,
        "recommended_hour": recommended_hour,
        "congestion_index": rec_cell["congestion_index"],
        "label": rec_cell["label"],
        "delay_minutes": rec_cell["delay_minutes"],
        "saving_vs_worst_minutes": saving,
        "saving_vs_worst_pct": saving_pct,
        "window_constrained": window_constrained,
        "whole_day_best_hour": whole_day_best_hour,
        "whole_day_worst_hour": whole_day_worst_hour,
        "whole_day_saving_minutes": whole_day_saving_minutes,
        "whole_day_saving_pct": whole_day_saving_pct,
        "alternatives": alternatives,
        "summary": summary,
        "provenance": MODEL_PROVENANCE,
        "origin": rec_cell.get("origin", GRID[(corridor_id, day, recommended_hour)]["origin"]),
        "confidence": rec_cell["confidence"],
    })


def _trend_for(corridor_id, day, hour):
    cur = GRID[(corridor_id, day, hour)]["congestion_index"]
    next_day, next_hour = (day, hour + 1) if hour < 23 else ((day + 1) % 7, 0)
    nxt = GRID[(corridor_id, next_day, next_hour)]["congestion_index"]
    if nxt - cur > 0.01:
        return "rising"
    if cur - nxt > 0.01:
        return "falling"
    return "flat"


def _now_text(label, verdict, confidence, explanation=None):
    # Forecast for this corridor/day/hour — not a live sensor replay.
    if verdict == "go_now":
        base = "Forecast: clear now. Good time to travel." if label == "Free" \
            else f"Forecast: {label.lower()} but manageable. Good time to travel."
    elif verdict == "wait":
        base = f"Forecast: {label.lower()} now, easing soon. Consider waiting a bit."
    else:
        base = f"Forecast: {label.lower()} at this hour — consider avoiding."
    if explanation:
        base += f" {explanation}"
    if confidence < 0.5:
        base += " Limited data for this hour — treat as a rough guide."
    return base


def _verdict_for(label, trend):
    if label == "Severe":
        return "avoid"
    if label == "Heavy":
        return "wait" if trend == "falling" else "avoid"
    if label == "Moderate":
        return "wait" if trend == "rising" else "go_now"
    return "go_now"  # Free


@app.route("/now")
@require_model
def now():
    current = datetime.datetime.now(IST)
    day, hour = current.weekday(), current.hour
    today = current.date()

    live_incidents = fetch_live_incident_features() if GRID_READY else {}

    results = []
    for c in CORRIDORS:
        cid = c["id"]
        origin = GRID[(cid, day, hour)]["origin"]
        ff_minutes = FREE_FLOW_MINUTES[cid]

        cell = GRID[(cid, day, hour)]
        idx = cell["congestion_index"]
        origin = cell["origin"]
        baseline = BASELINE_GRID.get((cid, day, hour), idx)
        inc_feat = live_incidents.get(cid, _neutral_incident_features())
        lag_prior, lag_week = lags_at_datetime(
            OBSERVED_LAG_DF, cid, day, hour,
            datetime.datetime.combine(today, datetime.time(hour, 0), tzinfo=IST),
            baseline_idx=baseline,
        )
        feat_row = build_forecast_feature_row(
            c, day, hour, today, inc_feat,
            lag_prior_idx=lag_prior, lag_week_idx=lag_week, baseline_idx=baseline,
        )
        explanation = forecast_explanation(baseline, idx, feat_row)

        label = label_for(idx)
        typical, delay = minutes_from_index(ff_minutes, idx)
        conf = compute_confidence(
            MODEL_PROVENANCE or "bootstrap", METRICS, None,
            cell_origin=origin, forecast_skill=FORECAST_SKILL,
        )
        trend = _trend_for(cid, day, hour)
        verdict = _verdict_for(label, trend)
        results.append({
            "id": cid,
            "name": c["name"],
            "congestion_index": idx,
            "label": label,
            "delay_minutes": delay,
            "trend": trend,
            "verdict": verdict,
            "text": _now_text(label, verdict, conf, explanation),
            "origin": origin,
            "explanation": explanation,
        })

    vals = [r["congestion_index"] for r in results]
    avg_congestion = round(sum(vals) / len(vals), 3)
    worst_corridor = results[int(np.argmax(vals))]["name"]
    clear_count = sum(1 for r in results if r["label"] == "Free")

    payload = {
        "now_ist": current.isoformat(timespec="seconds"),
        "day": day,
        "hour": hour,
        "corridors": results,
        "summary": {
            "avg_congestion": avg_congestion,
            "worst_corridor": worst_corridor,
            "clear_count": clear_count,
        },
        "provenance": MODEL_PROVENANCE,
    }
    if GRID_READY:
        payload["forecast"] = {
            "enabled": True,
            "skill_score": round(float(FORECAST_SKILL), 4),
        }
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
