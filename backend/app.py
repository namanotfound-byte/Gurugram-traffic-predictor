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

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_gbt.joblib")
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


def load_real_backing():
    """Which (corridor, day, hour) cells have real measured data behind them.

    Returns (bootstrap_cells, observed_cells, bootstrap_corridors_seen).
    Used only to make `confidence` honest instead of a flat constant.
    """
    bootstrap_cells, observed_cells, bootstrap_corridors = set(), set(), set()

    if os.path.exists(BOOTSTRAP_CSV):
        try:
            df = pd.read_csv(BOOTSTRAP_CSV, usecols=["corridor_id", "day_of_week", "hour"])
            for cid, d, h in zip(df["corridor_id"], df["day_of_week"], df["hour"]):
                cid, d, h = int(cid), int(d), int(h)
                bootstrap_cells.add((cid, d, h))
                bootstrap_corridors.add(cid)
        except Exception as e:
            print(f"[app] could not read {BOOTSTRAP_CSV} for confidence data: {e}")

    if os.path.exists(OBSERVED_CSV):
        try:
            df = pd.read_csv(OBSERVED_CSV, usecols=["corridor_id", "day_of_week", "hour"])
            for cid, d, h in zip(df["corridor_id"], df["day_of_week"], df["hour"]):
                observed_cells.add((int(cid), int(d), int(h)))
        except Exception as e:
            print(f"[app] could not read {OBSERVED_CSV} for confidence data: {e}")

    return bootstrap_cells, observed_cells, bootstrap_corridors


def load_route_stability():
    """Per-corridor route-length stability, computed from the bootstrap CSV.

    TomTom's routing occasionally returns a materially different path
    (different length_m) for the same corridor across different sweep
    timestamps. When that happens, the free-flow/congestion figures for
    that corridor aren't perfectly apples-to-apples hour to hour, so
    confidence should reflect it. This is computed generically for every
    corridor from whatever the sweep actually measured -- never a
    per-corridor hardcoded value.
    """
    if not os.path.exists(BOOTSTRAP_CSV):
        return {}
    try:
        df = pd.read_csv(BOOTSTRAP_CSV, usecols=["corridor_id", "length_m"])
    except Exception as e:
        print(f"[app] could not read {BOOTSTRAP_CSV} for route stability: {e}")
        return {}

    stability = {}
    for cid, group in df.groupby("corridor_id"):
        lo, hi = group["length_m"].min(), group["length_m"].max()
        variance_frac = (hi - lo) / lo if lo else 0.0
        # every extra point of route-length variance costs confidence,
        # floored so it degrades gracefully rather than collapsing to zero
        stability[int(cid)] = round(max(0.5, 1.0 - variance_frac), 3)
    return stability


def compute_confidence(provenance, metrics, cell, corridor_id,
                        bootstrap_cells, observed_cells, bootstrap_corridors,
                        route_stability):
    """Honest per-cell confidence: how much real data backs this prediction.

    Never a flat constant. Combines:
      - the model's own cross-validated fit quality (metrics.cv_r2),
      - whether THIS SPECIFIC (corridor, day, hour) cell was actually
        measured (observed > bootstrapped > extrapolated-within-known-
        corridor > fully extrapolated),
      - that corridor's route-length stability across the sweep (an
        unstable route, e.g. TomTom picking a different path hour to
        hour, means the underlying numbers are less trustworthy),
    or, if the whole model is synthetic (no real data anywhere), a flat
    low confidence -- that case genuinely has no data to differentiate on.
    """
    if provenance == "synthetic":
        return 0.15

    if metrics and isinstance(metrics, dict) and metrics.get("cv_r2") is not None:
        try:
            base_quality = max(0.1, min(0.95, float(metrics["cv_r2"])))
        except (TypeError, ValueError):
            base_quality = 0.5
    else:
        base_quality = 0.5

    if cell in observed_cells:
        data_factor = 1.0
    elif cell in bootstrap_cells:
        data_factor = 0.9
    elif corridor_id in bootstrap_corridors:
        data_factor = 0.55
    else:
        data_factor = 0.3

    stability_factor = route_stability.get(corridor_id, 1.0)

    return round(max(0.05, min(0.95, base_quality * data_factor * stability_factor)), 2)


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
# Startup: load model, load real-data backing, precompute the full grid
# ─────────────────────────────────────────────────────────────────────────

MODEL_PAYLOAD = load_model_payload()
MODEL_READY = MODEL_PAYLOAD is not None

MODEL_VERSION = None
MODEL_PROVENANCE = None
TRAINED_ROWS = None
GRID = {}  # (corridor_id, day, hour) -> dict

if MODEL_READY:
    MODEL = MODEL_PAYLOAD["model"]
    FEATURES = MODEL_PAYLOAD["features"]
    MODEL_VERSION = MODEL_PAYLOAD["model_version"]
    MODEL_PROVENANCE = MODEL_PAYLOAD["provenance"]
    TRAINED_ROWS = MODEL_PAYLOAD["trained_rows"]
    METRICS = MODEL_PAYLOAD.get("metrics")
    ROAD_CLASS_ENC_USED = MODEL_PAYLOAD.get("road_class_enc") or dict(ROAD_CLASS_ENC)

    FREE_FLOW_MINUTES = load_free_flow_minutes()
    BOOTSTRAP_CELLS, OBSERVED_CELLS, BOOTSTRAP_CORRIDORS = load_real_backing()
    ROUTE_STABILITY = load_route_stability()

    try:
        rows = []
        keys = []
        for c in CORRIDORS:
            for day in range(7):
                for hour in range(24):
                    rows.append(build_feature_row(hour, day, c["road_class"], ROAD_CLASS_ENC_USED))
                    keys.append((c["id"], day, hour))

        X = pd.DataFrame(rows)[FEATURES]
        preds = MODEL.predict(X)

        for (cid, day, hour), raw_idx in zip(keys, preds):
            idx = round(float(min(1.0, max(0.0, raw_idx))), 3)
            ff_minutes = FREE_FLOW_MINUTES[cid]
            typical, delay = minutes_from_index(ff_minutes, idx)
            cell = (cid, day, hour)
            conf = compute_confidence(
                MODEL_PROVENANCE, METRICS, cell, cid,
                BOOTSTRAP_CELLS, OBSERVED_CELLS, BOOTSTRAP_CORRIDORS,
                ROUTE_STABILITY,
            )
            GRID[cell] = {
                "congestion_index": idx,
                "label": label_for(idx),
                "free_flow_minutes": round(ff_minutes, 1),
                "typical_minutes": typical,
                "delay_minutes": delay,
                "confidence": conf,
            }
        print(f"[app] precomputed grid: {len(GRID)} cells, provenance={MODEL_PROVENANCE}, "
              f"model_version={MODEL_VERSION}")
    except Exception as e:
        print(f"[app] FAILED to precompute grid ({e}); disabling model-backed endpoints")
        MODEL_READY = False
        GRID = {}
else:
    print(f"[app] no model loaded from {MODEL_PATH}; model-backed endpoints will return 503")


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
        if not MODEL_READY:
            return jsonify({"error": "no model trained yet"}), 503
        return f(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────
# Window detection (contiguous good/bad hour runs, wrap-aware)
# ─────────────────────────────────────────────────────────────────────────

def _merge_runs(flags):
    """flags: list[bool] length 24 -> list of (start_hour, end_hour) inclusive,
    with a run touching both hour 0 and hour 23 merged into one wrapping run."""
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


def build_advice_summary(best_windows, peak_hour, peak_delay_minutes):
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
    worst = f"Worst is {fmt_ampm(peak_hour)} ({peak_delay_minutes:+.0f} min)."
    return f"{leave}. {worst}"


def advice_payload_for(corridor_id: int, day: int) -> dict:
    profile = [GRID[(corridor_id, day, h)]["congestion_index"] for h in range(24)]
    best_windows, worst_windows = find_windows(profile)
    best_hour = int(np.argmin(profile))
    peak_hour = int(np.argmax(profile))
    peak_delay = GRID[(corridor_id, day, peak_hour)]["delay_minutes"]
    summary = build_advice_summary(best_windows, peak_hour, peak_delay)
    return {
        "corridor_id": corridor_id,
        "profile": profile,
        "best_windows": best_windows,
        "worst_windows": worst_windows,
        "best_hour": best_hour,
        "peak_hour": peak_hour,
        "summary": summary,
        "confidence": GRID[(corridor_id, day, peak_hour)]["confidence"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_version": MODEL_VERSION,
        "provenance": MODEL_PROVENANCE,
        "corridors": N_CORRIDORS,
        "trained_rows": TRAINED_ROWS,
    })


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
    earliest, err = parse_query_int("earliest", 0, 23)
    if err:
        return jsonify({"error": err}), 400
    latest, err = parse_query_int("latest", 0, 23)
    if err:
        return jsonify({"error": err}), 400

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

    other_hours = sorted((h for h in hours if h != recommended_hour), key=idx_of)
    alternatives = [
        {
            "hour": h,
            "congestion_index": GRID[(corridor_id, day, h)]["congestion_index"],
            "delay_minutes": GRID[(corridor_id, day, h)]["delay_minutes"],
        }
        for h in other_hours[:3]
    ]

    if saving >= 0.5:
        summary = (f"Of {fmt_ampm(earliest)}-{fmt_ampm(latest)}, leave at {fmt_ampm(recommended_hour)}. "
                   f"Saves ~{saving:.0f} min vs leaving at {fmt_ampm(worst_hour)}.")
    else:
        summary = (f"Of {fmt_ampm(earliest)}-{fmt_ampm(latest)}, leave at {fmt_ampm(recommended_hour)}. "
                   f"Traffic is about the same all through this window.")

    return jsonify({
        "corridor_id": corridor_id,
        "day": day,
        "earliest": earliest,
        "latest": latest,
        "recommended_hour": recommended_hour,
        "congestion_index": rec_cell["congestion_index"],
        "label": rec_cell["label"],
        "delay_minutes": rec_cell["delay_minutes"],
        "saving_vs_worst_minutes": saving,
        "alternatives": alternatives,
        "summary": summary,
        "provenance": MODEL_PROVENANCE,
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


def _now_text(label, verdict):
    if verdict == "go_now":
        return "Clear. Good time to travel." if label == "Free" else f"{label} but manageable. Good time to travel."
    if verdict == "wait":
        return f"{label} now, easing soon. Consider waiting a bit."
    return f"Avoid — {label.lower()} congestion right now."


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

    results = []
    for c in CORRIDORS:
        cell = GRID[(c["id"], day, hour)]
        trend = _trend_for(c["id"], day, hour)
        verdict = _verdict_for(cell["label"], trend)
        results.append({
            "id": c["id"],
            "name": c["name"],
            "congestion_index": cell["congestion_index"],
            "label": cell["label"],
            "delay_minutes": cell["delay_minutes"],
            "trend": trend,
            "verdict": verdict,
            "text": _now_text(cell["label"], verdict),
        })

    vals = [r["congestion_index"] for r in results]
    avg_congestion = round(sum(vals) / len(vals), 3)
    worst_corridor = results[int(np.argmax(vals))]["name"]
    clear_count = sum(1 for r in results if r["label"] == "Free")

    return jsonify({
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
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
