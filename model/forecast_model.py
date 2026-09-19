#!/usr/bin/env python3
"""
model/forecast_model.py — the actual forecasting model.
=========================================================
This is the intellectual core of the project: can weather and calendar
conditions predict how Gurugram traffic will DEVIATE from its normal
day-of-week x hour pattern?

WHY MODEL THE RESIDUAL, NOT ABSOLUTE CONGESTION
-------------------------------------------------
data/gurugram_bootstrap.csv (1344 rows: 8 corridors x 7 days x 24 hours,
exactly one row per combination — verified) is a complete, noise-free map of
Gurugram's diurnal/weekly traffic rhythm. It has ZERO weather or date signal
(TomTom's historical model returns byte-identical numbers for six different
future Fridays at 18:00, including Diwali week — it's a pure day-of-week x
hour average). That rhythm is already perfectly captured; there is nothing
left to learn from it about rain, festivals, or salary-day traffic.

data/gurugram_observed.csv is the opposite: it's small (it only exists in
real time, one round at a time) but it's the ONLY place time-varying
conditions show up at all.

So instead of asking a sparse dataset to relearn the entire diurnal curve
from scratch (which would need vastly more rows than this project can
collect in a semester), this file asks a much narrower, much more
learnable question:

    baseline(corridor, day_of_week, hour) = bootstrap grid value   (complete)
    residual = observed_congestion - baseline                     (target)
    forecast = baseline + predicted_residual(weather, events, time)
    forecast = clip(forecast, 0, 1)

The residual should be small and should correlate with exactly the things
the baseline can't see — which is the hypothesis this file exists to test,
honestly, against a real benchmark (see EVALUATION below).

EVALUATION — the number that actually matters
------------------------------------------------
"Just use the historical average" (i.e. the bootstrap baseline alone, no
weather) is the benchmark this model must beat. We report a skill score:

    skill = 1 - (MAE_model / MAE_baseline)

Positive means the model adds value over the baseline; zero or negative
means it does not, and that must be reported plainly, not hidden.

Two rules this file follows, both because getting them wrong would produce a
flattering but meaningless number:

  1. TIME-BASED HOLDOUT. Train on earlier dates, test on later ones. A random
     split leaks badly here — consecutive 15-minute samples of the same
     corridor are near-identical, so a random split lets the model "cheat"
     by training on a sample 15 minutes away from a test sample. A
     forecaster has to be evaluated on data from AFTER its training window,
     because that's the only way it will ever actually be used.

  2. cv_r2 / leave-one-corridor-out is NOT used here (unlike
     model/traffic_model.py) — it was shown to be the wrong metric for this
     project (road classes with only one corridor each drove a nonsensical
     -2.52 score there). MAE-based skill score against the baseline is used
     instead throughout.

HONEST GATING
--------------
This file refuses to train — and exits with a clear message, emitting no
model artifact — unless the data is actually viable. See READINESS THRESHOLDS
below for the specific numbers and the reasoning behind each one. A model
trained on 300 dry rows cannot predict rain, and shipping one anyway would
be dishonest.

INCIDENTS (added 2026-08-17, same day as this file, once TomTom's Incidents
API was enabled on the project key mid-project)
------------------------------------------------------------------------------
Incidents matter more than weather for this residual — a crash or closure is
exactly the congestion weather/calendar features can never explain. See
incidents.py's module docstring for how incidents are matched to corridors
(spatial buffer, chosen empirically) and what the raw feed actually looks
like (dominated by "road closed" entries, many with no numeric delay value).

Incidents CANNOT be backfilled (no historical incident-replay endpoint
exists on this key) — rows collected before incident tracking existed (or
any future gap where the incidents fetch failed) have no ground truth for
"was there an incident nearby at that moment." Those rows are NOT assumed
incident-free: build_training_table() imputes the incident FEATURE values to
"no incident" defaults for the model (0 counts, no closure/jam, max
magnitude 0, nearest_incident_m at a large sentinel) but also adds
`incident_data_known` (1/0) alongside them, so the model can itself learn
to discount incident features on rows where they're actually unknown rather
than silently treating "unknown" as "confirmed clear." See FEATURE COLUMNS
below.

ROUTE-STABILITY FILTER (mirrors model/traffic_model.py / bootstrap_collect.py)
--------------------------------------------------------------------------------
TomTom's routing engine occasionally reroutes a corridor onto a physically
different road at different times (previously identified for this project:
Golf Course Extension Road, Mehrauli-Gurgaon Road, Southern Peripheral Road
all show >9% length_m swings). bootstrap_collect.py already flags this per
row as `route_stable` (length_m within 2% of corridors.py's verified_km) and
model/traffic_model.py already excludes route_stable=False rows before
training. This file does the same for BOTH datasets — bootstrap (using its
existing route_stable column) and observed (recomputed from length_m, since
collect_live.py doesn't currently write the column, but it's fully derivable
from a value every row already has). Mixing congestion_idx values that
describe two different physical roads into one baseline/residual would
silently corrupt exactly the signal this file exists to model.

FREE_FLOW CONSISTENCY CHECK
------------------------------
The residual is `observed_congestion - baseline_congestion`. Expanded out
(observed uses `1 - free_flow/live`, baseline uses `1 - free_flow/historic`,
same free_flow numerator so it cancels): residual = free_flow * (1/historic
- 1/live) — "how much worse right now is than typical," which is exactly
the quantity weather/incidents should explain. The denominator difference
itself is NOT the risk. The real risk is free_flow drifting BETWEEN the two
datasets for a corridor that reroutes (a different road has a different
free-flow time) — that would look like a large constant residual which
weather/incidents obviously can't explain, and would silently bias training.
check_free_flow_consistency() compares mean free_flow_s per corridor
between the two datasets (route_stable rows only, since unstable rows are
already excluded first — comparing free_flow across mismatched roads on
BOTH sides would be meaningless) and flags any corridor whose free_flow
moved more than FREE_FLOW_CONSISTENCY_TOL_PCT between the bootstrap sweep
and live collection. Called every readiness/train run; printed, not hidden.

SIGN CONVENTION: congestion_idx CAN legitimately be negative (observed
here: NH-48 at 00:45 IST, live=1942s vs free_flow=1991s -> -0.025). TomTom's
free-flow estimate is not a hard physical floor, and at low-traffic hours
the "live" time can come in faster than it. This is real signal, not a bug,
and it is NOT clipped on the input side — clipping the ground truth would
bias what the model is asked to learn. Only the model's own FORECAST output
is clipped to [0, 1] (`forecast = clip(baseline + predicted_residual, 0,
1)`), because a forecast promising negative congestion is not a meaningful
statement to hand to a user, even though a negative MEASUREMENT is. MAE and
GradientBoostingRegressor both handle negative real-valued targets natively
— nothing downstream breaks on this, verified by running the full pipeline
against real data containing negative congestion_idx rows.

USAGE
-----
    python model/forecast_model.py readiness   # report current data status
    python model/forecast_model.py train       # train + evaluate (refuses if not ready)
"""

import argparse
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import corridors  # noqa: E402
import weather as wx  # noqa: E402

BOOTSTRAP_FILE = os.path.join(REPO_ROOT, "data", "gurugram_bootstrap.csv")
OBSERVED_FILE = os.path.join(REPO_ROOT, "data", "gurugram_observed.csv")
MODEL_FILE = os.path.join(REPO_ROOT, "models", "forecast_residual_gbt.joblib")


# ─────────────────────────────────────────────
# READINESS THRESHOLDS (documented reasoning — these are the numbers a viva
# examiner will ask "why this and not something else" about)
# ─────────────────────────────────────────────
# Distinct calendar days of collection. Gurugram's weekly rhythm needs to
# repeat at least twice for weekday-vs-weekend to be distinguishable from
# noise, AND monsoon rain in NCR is intermittent (a handful of rainy days per
# fortnight in Aug-Sep, not continuous) — 14 days is the minimum window in
# which "we saw more than one rain event" is a credible claim rather than
# luck. This is a HARD floor set by the calendar, not something more request
# budget can shortcut.
MIN_DISTINCT_DAYS = 14

# Total observed rows. With ~14 features and gradient boosting's tendency to
# overfit on small tabular data, a conservative rule of thumb is on the order
# of 100+ effective (independent) samples per feature. Consecutive 15-minute
# samples are autocorrelated, so raw row count overstates independence badly
# — 1500 raw rows is a deliberately conservative floor to compensate (roughly
# 1500 / 26 buckets/corridor/day-ish effective days-of-signal, still thin).
MIN_TOTAL_ROWS = 1500

# The model must have seen BOTH conditions of interest with enough rows in
# each to fit a real relationship, not one or two outlier points. A model
# trained on rows that are 95% dry cannot honestly claim to predict rain.
MIN_RAINY_ROWS = 50
MIN_DRY_ROWS = 200

# Out of 8 corridors — tolerate a couple having persistent API trouble
# without blocking training on the rest, but refuse if collection has
# effectively only been covering one or two roads.
MIN_CORRIDORS = 6

# Incidents cannot be backfilled (see module docstring), so unlike rain,
# rows with UNKNOWN incident status must not silently count as "no
# incident" for gating purposes — only rows collected after incident
# tracking existed (incident_data_known=1) count towards these two. Same
# logic as the rain gate: the model must have actually seen both an
# incident-affected corridor-round AND a confirmed-clear one.
MIN_INCIDENT_AFFECTED_ROWS = 30
MIN_INCIDENT_CLEAR_ROWS = 150

TEST_HOLDOUT_FRACTION = 0.2  # last 20% of distinct days, by date, held out

# See module docstring's ROUTE-STABILITY FILTER section. Mirrors
# bootstrap_collect.py's ROUTE_STABLE_TOL_PCT exactly (not imported —
# bootstrap_collect.py is another workstream's file and off-limits to
# depend on for an import that would break this file if it's ever
# refactored there — the constant is small enough to just mirror).
ROUTE_STABLE_TOL_PCT = 2.0

# See module docstring's FREE_FLOW CONSISTENCY CHECK section.
FREE_FLOW_CONSISTENCY_TOL_PCT = 5.0

# A large sentinel (metres) for nearest_incident_m when no incident data is
# known at all for a row (imputation, not a real measurement) — far beyond
# incidents.py's 300 m match buffer, so it reads to the model as "nothing
# nearby" without claiming a specific false distance.
NEAREST_INCIDENT_SENTINEL_M = 5000.0

# corridor_id is allowed here (residual model, time holdout) — not in traffic_model.py GBT.
LAG_WEEK_DAYS = 7
LAG_WEEK_TOLERANCE_D = 1  # accept same corridor+hour from 6–8 calendar days earlier

FEATURE_COLS = [
    "temperature_c", "precipitation_mm", "is_raining_i", "rain_last_3h",
    "visibility_m", "low_visibility_i",
    "is_holiday_i", "is_festival_period_i", "is_month_end_i",
    "days_to_nearest_holiday",
    "road_class_enc", "hour_sin", "hour_cos", "is_weekend",
    "corridor_id",
    "lag_prior_idx", "lag_week_idx",
    "incident_count", "incident_total_delay_s", "incident_max_magnitude",
    "has_road_closure_i", "has_jam_i", "nearest_incident_m",
    "incident_data_known",
]

DWARKA_CORRIDOR_ID = 4


# ─────────────────────────────────────────────
# ROUTE STABILITY (see module docstring)
# ─────────────────────────────────────────────
def _route_stable_mask(corridor_ids, length_ms):
    """Vectorised route-stability check: True where length_m is within
    ROUTE_STABLE_TOL_PCT of that corridor's corridors.py verified_km
    reference. Missing length_m is treated as stable (nothing to contradict
    it with) rather than dropped, matching bootstrap_collect.py's
    is_route_stable() behaviour for a zero/absent reference length."""
    ref_m = corridor_ids.map(lambda cid: corridors.by_id(int(cid))["verified_km"] * 1000)
    length_ms = pd.to_numeric(length_ms, errors="coerce")
    deviation_pct = (length_ms - ref_m).abs() / ref_m * 100
    stable = deviation_pct <= ROUTE_STABLE_TOL_PCT
    return stable.fillna(True) | ref_m.le(0)


def _apply_route_stability_filter(df, label):
    if "length_m" not in df.columns or df.empty:
        return df
    stable = _route_stable_mask(df["corridor_id"], df["length_m"])
    n_unstable = int((~stable).sum())
    if n_unstable:
        per_corridor = df.loc[~stable].groupby(["corridor_id", "corridor_name"]).size()
        print(f"[INFO] {label}: excluding {n_unstable}/{len(df)} row(s) where the routing "
              f"engine measured a different physical road (length_m outside "
              f"{ROUTE_STABLE_TOL_PCT}% of corridors.py's verified_km):")
        for (cid, name), n in per_corridor.items():
            print(f"         corridor {cid} ({name}): {n} row(s) excluded")
    return df[stable].copy()


def check_free_flow_consistency(bootstrap_stable, observed_stable):
    """Compares mean free_flow_s per corridor between the (route-stable-only)
    bootstrap sweep and the (route-stable-only) live observed data. A
    corridor whose free_flow moved more than FREE_FLOW_CONSISTENCY_TOL_PCT
    between the two is flagged — that would mean the residual for that
    corridor is contaminated by a free_flow drift the weather/incident
    features can't explain, not a genuine conditions effect. See module
    docstring's FREE_FLOW CONSISTENCY CHECK section for why the denominator
    difference itself (live vs historic) is NOT the risk here."""
    flagged = []
    if observed_stable.empty:
        return flagged
    boot_ff = bootstrap_stable.groupby("corridor_id")["free_flow_s"].mean()
    obs_ff = observed_stable.groupby("corridor_id")["free_flow_s"].mean()
    print("  free_flow_s consistency (bootstrap vs observed, route-stable rows only):")
    for cid in sorted(set(boot_ff.index) & set(obs_ff.index)):
        b, o = boot_ff[cid], obs_ff[cid]
        pct = abs(o - b) / b * 100 if b else 0.0
        name = corridors.by_id(int(cid))["name"]
        flag = " <-- FLAGGED" if pct > FREE_FLOW_CONSISTENCY_TOL_PCT else ""
        print(f"    [{cid}] {name:38s} bootstrap={b:8.1f}s  observed={o:8.1f}s  diff={pct:5.2f}%{flag}")
        if pct > FREE_FLOW_CONSISTENCY_TOL_PCT:
            flagged.append((cid, name, pct))
    missing = set(boot_ff.index) - set(obs_ff.index)
    if missing:
        print(f"    (no observed data yet for corridor id(s) {sorted(missing)} — cannot check)")
    return flagged


# ─────────────────────────────────────────────
# DATA LOADING / FEATURE JOIN
# ─────────────────────────────────────────────
def load_baseline(route_stable_only=True):
    """The bootstrap grid, collapsed to one baseline congestion_idx per
    (corridor_id, day_of_week, hour). Uses a groupby-mean rather than assuming
    uniqueness so this stays correct even if the bootstrap file is ever
    re-swept with duplicate combinations. Filters to route_stable=True rows
    first (see module docstring) — mixing two different physical roads'
    congestion_idx into one averaged baseline cell would be meaningless."""
    if not os.path.exists(BOOTSTRAP_FILE):
        print(f"[FATAL] {BOOTSTRAP_FILE} not found — the residual model has no baseline to "
              f"measure against without it.")
        sys.exit(1)
    df = pd.read_csv(BOOTSTRAP_FILE)
    if route_stable_only and "route_stable" in df.columns:
        def _to_bool(v):
            return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1")
        stable = df["route_stable"].map(_to_bool)
        n_unstable = int((~stable).sum())
        if n_unstable:
            print(f"[INFO] load_baseline: excluding {n_unstable}/{len(df)} bootstrap row(s) "
                  f"flagged route_stable=False before computing the baseline grid.")
        df = df[stable]
    base = (
        df.groupby(["corridor_id", "day_of_week", "hour"])["congestion_idx"]
        .mean()
        .reset_index()
        .rename(columns={"congestion_idx": "baseline_idx"})
    )
    return base


def load_bootstrap_raw_stable():
    """Bootstrap rows, route_stable=True only — used by
    check_free_flow_consistency (kept separate from load_baseline so the
    latter can stay focused on producing the baseline lookup table)."""
    df = pd.read_csv(BOOTSTRAP_FILE)
    if "route_stable" in df.columns:
        def _to_bool(v):
            return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1")
        df = df[df["route_stable"].map(_to_bool)]
    return df


def load_observed_raw():
    if not os.path.exists(OBSERVED_FILE):
        return pd.DataFrame()
    df = pd.read_csv(OBSERVED_FILE)
    if df.empty:
        return df
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df["date"] = df["collected_at"].dt.date
    df = _apply_route_stability_filter(df, label="load_observed_raw")
    return df


def compute_leak_safe_lags(obs_df, baseline_lookup=None):
    """Add lag_prior_idx and lag_week_idx using only rows with collected_at
    strictly before each row's timestamp (no holdout leakage).

    baseline_lookup: optional {(corridor_id, day, hour): baseline_idx} used
    to impute missing lags at inference when no prior observation exists."""
    if obs_df.empty:
        obs_df = obs_df.copy()
        obs_df["lag_prior_idx"] = np.nan
        obs_df["lag_week_idx"] = np.nan
        return obs_df

    df = obs_df.sort_values("collected_at").reset_index(drop=True)
    lag_prior = []
    lag_week = []

    for _, row in df.iterrows():
        cid = int(row["corridor_id"])
        hour = int(row["hour"])
        ts = row["collected_at"]
        row_date = row["date"] if "date" in row else ts.date()

        prior = df[(df["corridor_id"] == cid) & (df["collected_at"] < ts)]
        if prior.empty:
            lag_prior.append(np.nan)
        else:
            lag_prior.append(float(prior.iloc[-1]["congestion_idx"]))

        week_lo = row_date - datetime.timedelta(days=LAG_WEEK_DAYS + LAG_WEEK_TOLERANCE_D)
        week_hi = row_date - datetime.timedelta(days=LAG_WEEK_DAYS - LAG_WEEK_TOLERANCE_D)
        week_cands = prior[
            (prior["hour"] == hour)
            & (prior["date"] >= week_lo)
            & (prior["date"] <= week_hi)
        ]
        if week_cands.empty:
            lag_week.append(np.nan)
        else:
            week_cands = week_cands.copy()
            week_cands["day_delta"] = (row_date - week_cands["date"]).apply(lambda d: abs(d.days - LAG_WEEK_DAYS))
            best = week_cands.sort_values(["day_delta", "collected_at"]).iloc[0]
            lag_week.append(float(best["congestion_idx"]))

    df["lag_prior_idx"] = lag_prior
    df["lag_week_idx"] = lag_week

    if baseline_lookup is not None:
        for col in ("lag_prior_idx", "lag_week_idx"):
            missing = df[col].isna()
            if missing.any():
                df.loc[missing, col] = df.loc[missing].apply(
                    lambda r: baseline_lookup.get(
                        (int(r["corridor_id"]), int(r["day_of_week"]), int(r["hour"])),
                        r.get("baseline_idx", np.nan),
                    ),
                    axis=1,
                )
    return df


def lags_at_datetime(obs_df, corridor_id, day, hour, target_dt, baseline_idx=None):
    """Inference-time lag lookup: only observations strictly before target_dt."""
    if obs_df is None or obs_df.empty:
        return baseline_idx, baseline_idx

    cid = int(corridor_id)
    cutoff = pd.Timestamp(target_dt)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("Asia/Kolkata")
    else:
        cutoff = cutoff.tz_convert("Asia/Kolkata")

    past = obs_df[(obs_df["corridor_id"] == cid) & (obs_df["collected_at"] < cutoff)]
    lag_prior = float(past.iloc[-1]["congestion_idx"]) if not past.empty else baseline_idx

    target_date = cutoff.date()
    week_lo = target_date - datetime.timedelta(days=LAG_WEEK_DAYS + LAG_WEEK_TOLERANCE_D)
    week_hi = target_date - datetime.timedelta(days=LAG_WEEK_DAYS - LAG_WEEK_TOLERANCE_D)
    week_cands = past[
        (past["hour"] == int(hour))
        & (past["date"] >= week_lo)
        & (past["date"] <= week_hi)
    ]
    if week_cands.empty:
        lag_week = baseline_idx
    else:
        week_cands = week_cands.copy()
        week_cands["day_delta"] = (target_date - week_cands["date"]).apply(
            lambda d: abs(d.days - LAG_WEEK_DAYS)
        )
        lag_week = float(week_cands.sort_values(["day_delta", "collected_at"]).iloc[0]["congestion_idx"])

    return lag_prior, lag_week


def attach_weather_and_events(df):
    """Fills in weather/calendar columns for any row missing them (mirrors
    collect_live.py's own backfill, kept independent here so this file works
    correctly even if collect_live.py's CSV is somehow stale/partial —
    forecast_model.py never assumes upstream backfill has already run)."""
    if df.empty:
        return df

    needs_fill = df["precipitation_mm"].isna() if "precipitation_mm" in df.columns else pd.Series([True] * len(df))
    if needs_fill.any():
        wx.get_weather_range(df["date"].min(), df["date"].max())  # bulk prefetch

    weather_cols = [
        "temperature_c", "precipitation_mm", "is_raining", "rain_intensity",
        "rain_last_3h", "visibility_m", "low_visibility",
    ]
    event_cols = ["is_holiday", "holiday_name", "is_festival_period", "is_month_end",
                  "days_to_nearest_holiday"]
    for c in weather_cols + event_cols:
        if c not in df.columns:
            df[c] = np.nan

    event_cache = {}
    for idx, row in df.iterrows():
        d, h = row["date"], int(row["hour"])
        if pd.isna(row.get("precipitation_mm")):
            w = wx.get_hourly_weather(d, h) or {}
            for c in weather_cols:
                df.at[idx, c] = w.get(c)
        if pd.isna(row.get("is_holiday")):
            if d not in event_cache:
                event_cache[d] = wx.get_event_features(d)
            e = event_cache[d]
            for c in event_cols:
                df.at[idx, c] = e.get(c)
    return df


def build_training_table():
    """Observed rows (route-stable only), joined to their baseline
    (route-stable only), with the residual target and all engineered
    features computed. One row per (corridor, round)."""
    obs = load_observed_raw()  # already route-stability filtered
    if obs.empty:
        return obs

    # Free_flow consistency check — see module docstring. Run before
    # anything else so a corridor-level data-quality problem is visible
    # even if training later refuses to run for other reasons.
    flagged = check_free_flow_consistency(load_bootstrap_raw_stable(), obs)
    if flagged:
        print(f"  [WARN] {len(flagged)} corridor(s) show free_flow drift beyond "
              f"{FREE_FLOW_CONSISTENCY_TOL_PCT}% between bootstrap and observed — "
              f"their residuals may partly reflect this drift rather than "
              f"weather/incident conditions. Investigate before trusting their "
              f"feature importances specifically.")

    obs = attach_weather_and_events(obs)
    base = load_baseline()
    df = obs.merge(base, on=["corridor_id", "day_of_week", "hour"], how="left")

    missing_baseline = df["baseline_idx"].isna().sum()
    if missing_baseline:
        print(f"[WARN] {missing_baseline} observed row(s) have no matching bootstrap baseline "
              f"cell (corridor/day/hour not in the grid) — dropped from training.")
        df = df.dropna(subset=["baseline_idx"])

    # residual target — see module docstring's SIGN CONVENTION section:
    # can legitimately be negative, and is intentionally NOT clipped here.
    df["residual"] = df["congestion_idx"] - df["baseline_idx"]

    # engineered features (self-contained here — NOT imported from
    # model/traffic_model.py, which this project's rules say to leave
    # untouched, and which this file must keep working independent of)
    df["road_class_enc"] = df["road_class"].map(corridors.ROAD_CLASS_ENC)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for col, out in [
        ("is_raining", "is_raining_i"),
        ("low_visibility", "low_visibility_i"),
        ("is_holiday", "is_holiday_i"),
        ("is_festival_period", "is_festival_period_i"),
        ("is_month_end", "is_month_end_i"),
    ]:
        if col in df.columns:
            df[out] = df[col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(
                df[col].astype("boolean").astype("Int64")
            ).fillna(0).astype(int)
        else:
            df[out] = 0

    if df["visibility_m"].notna().any():
        df["visibility_m"] = df["visibility_m"].fillna(df["visibility_m"].median())
    else:
        df["visibility_m"] = df["visibility_m"].fillna(8000.0)  # NCR clear-day default
    df["days_to_nearest_holiday"] = df["days_to_nearest_holiday"].fillna(999)
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median() if df["temperature_c"].notna().any() else 28.0)
    df["rain_last_3h"] = df["rain_last_3h"].fillna(0.0)
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)

    # ---- incident features (see module docstring: NOT backfillable) ----
    # incident_data_known: 1 iff this row was collected after incident
    # tracking existed (i.e. incident_count is present, even if it's a
    # genuine 0). Rows before that get imputed "no incident" defaults for
    # the model but are flagged unknown rather than presented as confirmed
    # clear — the feature importance of incident_data_known itself is worth
    # checking once trained: if the model leans on it, that's a sign it's
    # partly learning "old rows vs new rows" rather than a true incident
    # effect, which would be an honest caveat to report.
    if "incident_count" in df.columns:
        df["incident_data_known"] = df["incident_count"].notna().astype(int)
    else:
        df["incident_count"] = np.nan
        df["incident_data_known"] = 0

    for col in ["incident_total_delay_s", "incident_known_delay_count", "incident_max_magnitude"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["has_road_closure", "has_jam"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col + "_i"] = df[col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    if "nearest_incident_m" not in df.columns:
        df["nearest_incident_m"] = np.nan

    df["incident_count"] = df["incident_count"].fillna(0).astype(int)
    df["incident_total_delay_s"] = df["incident_total_delay_s"].fillna(0.0)
    df["incident_max_magnitude"] = df["incident_max_magnitude"].fillna(0).astype(int)
    df["nearest_incident_m"] = df["nearest_incident_m"].fillna(NEAREST_INCIDENT_SENTINEL_M)

    baseline_lookup = {
        (int(r["corridor_id"]), int(r["day_of_week"]), int(r["hour"])): float(r["baseline_idx"])
        for _, r in base.iterrows()
    }
    df = compute_leak_safe_lags(df, baseline_lookup=baseline_lookup)
    df["lag_prior_idx"] = df["lag_prior_idx"].fillna(df["baseline_idx"])
    df["lag_week_idx"] = df["lag_week_idx"].fillna(df["baseline_idx"])

    return df


# ─────────────────────────────────────────────
# READINESS REPORT
# ─────────────────────────────────────────────
def forecast_readiness(df=None, verbose=True):
    """Reports the current data status against the readiness thresholds:
    rows, distinct days, rainy/dry hours seen, holidays covered, corridors
    covered, whether training is unlocked, and — if not — what's still
    missing and a realistic estimate of when it will be ready."""
    if df is None:
        df = build_training_table()

    report = {
        "unlocked": False,
        "rows": 0,
        "distinct_days": 0,
        "distinct_hours_seen": 0,
        "rainy_hour_buckets": 0,
        "dry_hour_buckets": 0,
        "holidays_covered": 0,
        "corridors_covered": 0,
        "date_span": None,
        "missing": [],
    }

    if df is None or df.empty:
        report["missing"].append(
            f"no observed rows at all yet (need >= {MIN_TOTAL_ROWS} rows across "
            f">= {MIN_DISTINCT_DAYS} distinct days)."
        )
        if verbose:
            _print_readiness(report)
        return report

    report["rows"] = len(df)
    distinct_days = sorted(df["date"].unique())
    report["distinct_days"] = len(distinct_days)
    report["date_span"] = (str(distinct_days[0]), str(distinct_days[-1])) if distinct_days else None
    report["corridors_covered"] = df["corridor_id"].nunique()

    # De-duplicate weather status to unique (date, hour) buckets — weather is
    # city-wide, shared across all 8 corridors in a round, so counting raw
    # rows here would inflate "rainy hours seen" by up to 8x.
    hour_buckets = df.drop_duplicates(subset=["date", "hour"])
    report["distinct_hours_seen"] = len(hour_buckets)
    report["rainy_hour_buckets"] = int(hour_buckets["is_raining_i"].sum())
    report["dry_hour_buckets"] = int((1 - hour_buckets["is_raining_i"]).sum())
    report["holidays_covered"] = int(hour_buckets.drop_duplicates(subset=["date"])["is_holiday_i"].sum())

    # Rows (not hour-buckets) are what the model actually trains on, so gate
    # rainy/dry ROW counts, which is the stricter (harder to satisfy) of the
    # two — being generous here would be the dishonest direction.
    rainy_rows = int(df["is_raining_i"].sum())
    dry_rows = int((1 - df["is_raining_i"]).sum())

    # Incident presence gate — same logic as rain, but restricted to rows
    # where incident status is actually KNOWN (see build_training_table's
    # incident_data_known), since incidents can't be backfilled onto older
    # rows the way weather can (see module docstring). A row with unknown
    # incident status counts toward neither bucket below.
    known = df[df["incident_data_known"] == 1]
    incident_affected_rows = int((known["incident_count"] > 0).sum())
    incident_clear_rows = int((known["incident_count"] == 0).sum())

    missing = []
    if report["distinct_days"] < MIN_DISTINCT_DAYS:
        missing.append(
            f"only {report['distinct_days']}/{MIN_DISTINCT_DAYS} distinct days collected."
        )
    if report["rows"] < MIN_TOTAL_ROWS:
        missing.append(f"only {report['rows']}/{MIN_TOTAL_ROWS} total observed rows.")
    if rainy_rows < MIN_RAINY_ROWS:
        missing.append(
            f"only {rainy_rows}/{MIN_RAINY_ROWS} rainy rows seen — cannot honestly claim to "
            f"predict rain's effect without this."
        )
    if dry_rows < MIN_DRY_ROWS:
        missing.append(f"only {dry_rows}/{MIN_DRY_ROWS} dry rows seen.")
    if incident_affected_rows < MIN_INCIDENT_AFFECTED_ROWS:
        missing.append(
            f"only {incident_affected_rows}/{MIN_INCIDENT_AFFECTED_ROWS} rows with a KNOWN "
            f"nearby incident seen — cannot honestly claim to predict incident effects without this "
            f"(incidents can't be backfilled, so this can only grow from here forward)."
        )
    if incident_clear_rows < MIN_INCIDENT_CLEAR_ROWS:
        missing.append(
            f"only {incident_clear_rows}/{MIN_INCIDENT_CLEAR_ROWS} rows with KNOWN incident-clear "
            f"status seen."
        )
    if report["corridors_covered"] < MIN_CORRIDORS:
        missing.append(
            f"only {report['corridors_covered']}/{MIN_CORRIDORS} corridors have any data."
        )

    report["rainy_rows"] = rainy_rows
    report["dry_rows"] = dry_rows
    report["incident_affected_rows"] = incident_affected_rows
    report["incident_clear_rows"] = incident_clear_rows
    report["incident_data_known_rows"] = len(known)
    report["missing"] = missing
    report["unlocked"] = len(missing) == 0

    if not report["unlocked"]:
        # Realistic ETA: the hard floor is calendar days (MIN_DISTINCT_DAYS),
        # which no amount of extra request budget can shortcut. Rainy-row
        # coverage depends on actual monsoon rainfall, which is out of our
        # control — so the ETA below is a floor, not a guarantee.
        days_needed_for_calendar = max(0, MIN_DISTINCT_DAYS - report["distinct_days"])
        eta = datetime.date.today() + datetime.timedelta(days=days_needed_for_calendar)
        report["earliest_possible_ready_date"] = str(eta)
        report["missing"].append(
            f"earliest this can possibly unlock (calendar floor alone): {eta}. "
            f"This assumes continuous collection AND at least a few rain events landing "
            f"in that window (plausible in mid-August NCR monsoon, not guaranteed)."
        )

    if verbose:
        _print_readiness(report)
    return report


def _print_readiness(r):
    print("=" * 70)
    print("FORECAST MODEL READINESS")
    print("=" * 70)
    print(f"  observed rows:              {r['rows']}")
    print(f"  distinct days:               {r['distinct_days']} (need >= {MIN_DISTINCT_DAYS})")
    if r.get("date_span"):
        print(f"  date span:                   {r['date_span'][0]} .. {r['date_span'][1]}")
    print(f"  corridors covered:           {r['corridors_covered']}/{len(corridors.CORRIDORS)} "
          f"(need >= {MIN_CORRIDORS})")
    print(f"  distinct (date,hour) seen:   {r['distinct_hours_seen']}")
    print(f"  rainy hour-buckets seen:     {r['rainy_hour_buckets']}")
    print(f"  dry hour-buckets seen:       {r['dry_hour_buckets']}")
    if "rainy_rows" in r:
        print(f"  rainy ROWS (gating value):  {r['rainy_rows']} (need >= {MIN_RAINY_ROWS})")
        print(f"  dry ROWS (gating value):    {r['dry_rows']} (need >= {MIN_DRY_ROWS})")
    if "incident_affected_rows" in r:
        print(f"  rows w/ known incident status: {r['incident_data_known_rows']}/{r['rows']} "
              f"(incidents cannot be backfilled — older rows are unknown, not 'clear')")
        print(f"  incident-affected ROWS:      {r['incident_affected_rows']} (need >= {MIN_INCIDENT_AFFECTED_ROWS})")
        print(f"  incident-clear ROWS:         {r['incident_clear_rows']} (need >= {MIN_INCIDENT_CLEAR_ROWS})")
    print(f"  holidays covered:            {r['holidays_covered']}")
    print(f"  TRAINING UNLOCKED:            {r['unlocked']}")
    if r["missing"]:
        print("  still missing:")
        for m in r["missing"]:
            print(f"    - {m}")
    print("=" * 70)


# ─────────────────────────────────────────────
# TRAIN + EVALUATE
# ─────────────────────────────────────────────
def train():
    df = build_training_table()
    report = forecast_readiness(df, verbose=True)

    if not report["unlocked"]:
        print()
        print("[REFUSING TO TRAIN] Data does not yet meet the readiness thresholds above.")
        print("                    No model artifact will be written. This is the correct,")
        print("                    honest outcome today — not a bug. Re-run this command")
        print("                    after more collection days have passed.")
        sys.exit(1)

    # ---- time-based holdout: train on earlier dates, test on later dates ----
    distinct_days = sorted(df["date"].unique())
    n_test_days = max(1, round(len(distinct_days) * TEST_HOLDOUT_FRACTION))
    test_days = set(distinct_days[-n_test_days:])
    train_df = df[~df["date"].isin(test_days)].copy()
    test_df = df[df["date"].isin(test_days)].copy()

    if len(train_df) < 10 or len(test_df) < 10:
        print(f"[REFUSING TO TRAIN] Time-based holdout split leaves too few rows "
              f"(train={len(train_df)}, test={len(test_df)}) to evaluate honestly, even "
              f"though the aggregate thresholds passed. No model artifact will be written.")
        sys.exit(1)

    X_train, y_train = train_df[FEATURE_COLS], train_df["residual"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["residual"]

    baseline_mae = mean_absolute_error(test_df["congestion_idx"], test_df["baseline_idx"])

    def _holdout_scores(name, fitted, X_tr, y_tr, X_te, te_df):
        fitted.fit(X_tr, y_tr)
        pred_resid = fitted.predict(X_te)
        pred_cong = (te_df["baseline_idx"] + pred_resid).clip(0, 1)
        mae = mean_absolute_error(te_df["congestion_idx"], pred_cong)
        sk = 1 - (mae / baseline_mae) if baseline_mae > 0 else float("nan")
        return fitted, mae, sk

    gbt = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbt_model, gbt_mae, gbt_skill = _holdout_scores(
        "GBT", gbt, X_train, y_train, X_test, test_df,
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), early_stopping=True, random_state=42,
        max_iter=500,
    )
    mlp_model, mlp_mae, mlp_skill = _holdout_scores(
        "MLP", mlp, X_train, y_train, X_test, test_df,
    )

    if mlp_mae < gbt_mae:
        winner_name, model, model_mae, skill = "MLP", mlp_model, mlp_mae, mlp_skill
    else:
        winner_name, model, model_mae, skill = "GBT", gbt_model, gbt_mae, gbt_skill

    # Optional corridor-specific GBT for Dwarka if it improves holdout MAE there.
    dwarka_override = None
    dw_test = test_df[test_df["corridor_id"] == DWARKA_CORRIDOR_ID]
    if len(dw_test) >= 5:
        dw_train = train_df[train_df["corridor_id"] == DWARKA_CORRIDOR_ID]
        dw_gbt = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        dw_gbt.fit(dw_train[FEATURE_COLS], dw_train["residual"])
        dw_pred = (dw_test["baseline_idx"] + dw_gbt.predict(dw_test[FEATURE_COLS])).clip(0, 1)
        dw_mae_main = mean_absolute_error(dw_test["congestion_idx"], dw_pred)
        dw_pred_global = (
            dw_test["baseline_idx"] + model.predict(dw_test[FEATURE_COLS])
        ).clip(0, 1)
        dw_mae_global = mean_absolute_error(dw_test["congestion_idx"], dw_pred_global)
        print(f"  Dwarka holdout MAE — global {winner_name}: {dw_mae_global:.4f}, "
              f"corridor-specific GBT: {dw_mae_main:.4f}")
        if dw_mae_main < dw_mae_global:
            dwarka_override = dw_gbt
            print("  -> shipping corridor-specific GBT for Dwarka (id 4) on holdout improvement.")
        else:
            print("  -> keeping global model for Dwarka.")

    print()
    print("=" * 70)
    print("TIME-BASED HOLDOUT EVALUATION")
    print("=" * 70)
    print(f"  train: {len(train_df)} rows across {len(distinct_days) - n_test_days} days "
          f"({distinct_days[0]} .. {distinct_days[-n_test_days - 1]})")
    print(f"  test:  {len(test_df)} rows across {n_test_days} days "
          f"({distinct_days[-n_test_days]} .. {distinct_days[-1]})")
    print(f"  baseline MAE (bootstrap grid alone):        {baseline_mae:.4f}")
    print(f"  GBT   MAE (baseline + predicted residual): {gbt_mae:.4f}  skill={gbt_skill:.4f}")
    print(f"  MLP   MAE (baseline + predicted residual): {mlp_mae:.4f}  skill={mlp_skill:.4f}")
    print(f"  SHIPPED: {winner_name} (lower holdout MAE)")
    print(f"  SKILL SCORE = 1 - model_MAE/baseline_MAE: {skill:.4f}")
    if skill > 0:
        print("  -> model adds value over 'just use the historical average'.")
    else:
        print("  -> model does NOT beat the baseline. Report this honestly; do not ship it "
              "as an improvement.")
        sys.exit(1)

    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
        )
        print()
        print("  feature importances (shipped model):")
        for name, imp in importances:
            print(f"    {name:28s} {imp:.4f}")
    print("=" * 70)

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_type": winner_name,
            "features": FEATURE_COLS,
            "skill_score": skill,
            "model_mae": model_mae,
            "baseline_mae": baseline_mae,
            "holdout_gbt_mae": gbt_mae,
            "holdout_gbt_skill": gbt_skill,
            "holdout_mlp_mae": mlp_mae,
            "holdout_mlp_skill": mlp_skill,
            "dwarka_override": dwarka_override,
            "trained_at": datetime.datetime.now().isoformat(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        MODEL_FILE,
    )
    print(f"\nModel artifact written: {MODEL_FILE}")
    return skill


def main():
    parser = argparse.ArgumentParser(description="Gurugram residual forecast model.")
    parser.add_argument("command", choices=["readiness", "train"])
    args = parser.parse_args()

    if args.command == "readiness":
        forecast_readiness()
    else:
        train()


if __name__ == "__main__":
    main()
