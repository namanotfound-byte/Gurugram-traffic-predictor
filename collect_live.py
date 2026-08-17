#!/usr/bin/env python3
"""
collect_live.py — Live OBSERVED traffic collector (TomTom Routing API)
========================================================================

WHY THIS EXISTS
----------------
The original collector (model/traffic_model.py: collect_once) polls TomTom's
Flow Segment Data API (/traffic/services/4/flowSegmentData/...). Tested
2026-08-16 against this project's real TomTom key:

    Flow API      (flowSegmentData)  -> HTTP 403 Forbidden
    Search/Geocode API                -> HTTP 403 Forbidden
    Routing API   (calculateRoute)    -> HTTP 200 OK

So collect_once() could never have worked on this key's plan tier — it was
silently failing every 30-minute run (the "[WARN] TomTom fetch failed"
branch) for weeks, which is why the repo has zero rows of real traffic
data despite the scheduled workflow "running" the whole time.

This script gets equivalent signal through an endpoint the key actually
has access to: TomTom's Routing API, called with traffic=true and
computeTravelTimeFor=all. For a start->end pair it returns, in one
response:

    travelTimeInSeconds                 live, current-conditions time
    noTrafficTravelTimeInSeconds        free-flow baseline
    historicTrafficTravelTimeInSeconds  historical-model time
    trafficDelayInSeconds               live - free-flow-ish delay

We compute an observed congestion index from the LIVE time:

    congestion_idx = 1 - (noTrafficTravelTimeInSeconds / travelTimeInSeconds)

This lands on the same 0..1 scale as the bootstrap sweep's congestion_idx,
so the two are mixable as training rows — but they are NOT computed the
same way (bootstrap divides free-flow by *historic* time; this divides
free-flow by *live* time). Every row here is tagged source="observed" so
downstream consumers can tell the two apart, weight them differently, or
filter to only real, present-moment observations.

QUOTA
-----
TomTom's free tier is 2,500 requests/day. One round here is
len(CORRIDORS) == 13 routing requests (one calculateRoute call per
corridor) + 1 incidents bbox request (see incidents.py) = 14 requests/round.

Cadence target is every 15 minutes:

    14 requests/round * 96 rounds/day = 1,344 requests/day

against the 2,500/day cap — leaving ~1,150/day of headroom for the
bootstrap sweep / manual experimentation sharing the same key.

HONEST CAVEAT on what a 15-minute cadence actually buys: consecutive
15-minute samples of the same corridor are strongly autocorrelated (traffic
15 minutes from now looks a lot like traffic now), so more rows is far
less than proportionally more *information* for a model that predicts the
diurnal curve. The real payoff is temporal resolution on RAIN ONSET/OFFSET,
which are short-lived (often 15-45 minutes) — at 30-minute cadence a rain
event can start and finish between two samples and look like it never
happened; at 15-minute cadence it's far more likely to be caught
mid-transition, which is exactly the regime the residual model most needs
to see (see model/forecast_model.py and weather.py's rain_last_3h feature).

GITHUB CRON THROTTLING AND THE HOURLY-JOB / INTERNAL-LOOP FIX (2026-08-17)
---------------------------------------------------------------------------
Scheduling `collect.yml` at `*/15 * * * *` does NOT get you 96 rounds/day.
GitHub's scheduler is documented best-effort for short intervals; observed
real gaps between CI rounds were 30-45 minutes, i.e. ~36 rounds/day (~37%
of nominal) instead of 96. Row counts and the 14-distinct-days gate were
unaffected by this, but temporal RESOLUTION during weather events — the
signal above says is the actual payoff of 15-minute sampling — was being
lost to the exact degree the schedule was being throttled.

The fix is not to fight GitHub's scheduler with a tighter cron (that makes
the throttling worse, not better — GitHub best-effort-schedules more
generously at longer intervals). Instead `collect.yml` now fires HOURLY
(`0 * * * *`, an interval GitHub reliably honors close to on-time) and
calls `--loop --max-rounds 4` here: this process itself loops internally,
firing one round every ROUND_MINUTES (15) minutes for up to 4 rounds
(~45 minutes wall-clock) before exiting, so every job that runs at all
delivers ~4 evenly-spaced-by-15-minutes rounds regardless of scheduler lag
on the outer hourly trigger. `--max-minutes` is also accepted as a safety
cap (used by CI as a belt-and-suspenders bound) so a slow round (retries/
backoff) can't push the job past the ~1-hour window before the next
scheduled run, which would risk overlapping jobs.

This does NOT change the daily request math above (still ~14 req/round,
now reliably ~4 rounds/hour * 24h = 96 rounds/day -> ~1,344 req/day) — it
changes how reliably that nominal rate is actually delivered, since the
job no longer depends on GitHub firing a fresh workflow run every 15
minutes to hit it.

WEATHER + CALENDAR FEATURES (2026-08-17)
------------------------------------------
Every row now also carries the Gurugram weather (from weather.py, backed by
Open-Meteo, cached on disk) and calendar features (public holidays / festival
window / month-end) for that row's date+hour. Weather is fetched ONCE per
round (it's a single city-wide reading, not per-corridor) and stamped onto
all 8 corridor rows for that round. This is what lets model/forecast_model.py
learn a weather/calendar-conditioned residual on top of the bootstrap
baseline — see that file's module docstring for the modelling rationale.

`--backfill-weather` retroactively attaches these columns to rows collected
before this feature existed (via weather.py's archive/forecast auto-select),
rewriting the CSV in place. It is a one-time migration path, not something
the CI workflow runs on a schedule — collect.yml is deliberately left alone
(see .github/workflows/collect.yml) and only gets weather on rows collected
after this change via the normal --once path.

INCIDENT FEATURES (added 2026-08-17, same day)
--------------------------------------------------
TomTom's Traffic Incidents API, previously 403 on this project's key, was
enabled on the TomTom portal partway through this project and is now live
(re-verified: 200 OK, real incidents returned for the Gurugram bbox). This
matters more than weather for the residual model — a crash or closure is
exactly the congestion that weather/calendar features can never explain.
See incidents.py's module docstring for the full design: incidents are
matched to corridors by distance from the incident's own geometry to the
corridor's real digitized polyline (frontend/corridors.geojson, read-only),
with a 300 m buffer chosen empirically from a real pull of Gurugram
incidents (not guessed).

ONE bbox request per round covers all 8 corridors (not one request per
corridor — see incidents.py's QUOTA section), so this adds only ~1
request/round (~+96/day) on top of the 768/day the routing calls already
use — still well inside the 2,500/day free tier.

UNLIKE WEATHER, INCIDENTS CANNOT BE BACKFILLED. There is no historical
incident-replay endpoint on this key (only current, live incidents were
ever confirmed reachable) — a closure that happened yesterday and has
since cleared is gone, there is no record to backfill onto yesterday's
rows. Rows collected before this feature existed (and any future gap in
collection) will have blank incident columns; model/forecast_model.py
documents exactly how it treats that gap (see its FEATURE IMPUTATION
section) rather than silently pretending "blank" means "confirmed clear."

BUDGET GUARD
------------
`--budget-guard` (optional) tracks TomTom requests made today in
data/.tomtom_budget.json (git-ignored — it's local run state, not a data
asset) and refuses to start a new round once `--daily-cap` (default 2000,
leaving headroom under the 2,500 hard limit for other consumers of the same
key) would be exceeded. Off by default so a bare --once/--loop behaves
exactly as before.

USAGE
-----
    python collect_live.py --once                         # single round, then exit (used by CI)
    python collect_live.py --loop                          # loop forever, one round / 15 min (VM)
    python collect_live.py --loop --max-rounds 4           # loop 4 rounds (~45 min) then exit (CI, hourly job)
    python collect_live.py --loop --max-rounds 4 --max-minutes 50   # + safety time cap
    python collect_live.py --backfill-weather              # attach weather/calendar cols to old rows
    python collect_live.py --once --budget-guard --daily-cap 2000

`--max-rounds` / `--max-minutes` only apply with `--loop` (error otherwise).
`--once` behaviour is completely unchanged by their existence — it still
runs exactly one round and exits, same as before this flag was added.

The TomTom key is read from the TOMTOM_API_KEY environment variable, or
else from a .env file (KEY=VALUE lines) in this file's directory. The key
is never printed, logged, or written anywhere by this script.

Imports corridor definitions from corridors.py — the single source of
truth for this repo. Corridors are never redefined here.
"""

import argparse
import csv
import datetime
import json
import os
import random
import sys
import time

import requests

from corridors import CORRIDORS
import weather as wx
import incidents as inc

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(REPO_ROOT, "data", "gurugram_observed.csv")
ENV_FILE = os.path.join(REPO_ROOT, ".env")
BUDGET_STATE_FILE = os.path.join(REPO_ROOT, "data", ".tomtom_budget.json")

ROUTING_URL_TMPL = "https://api.tomtom.com/routing/1/calculateRoute/{start}:{end}/json"

CSV_COLUMNS = [
    "corridor_id", "corridor_name", "road_class", "day_of_week", "hour",
    "minute", "length_m", "free_flow_s", "live_s", "historic_s",
    "traffic_delay_s", "congestion_idx", "source", "collected_at",
    # weather (weather.py / Open-Meteo) — see module docstring
    "temperature_c", "precipitation_mm", "is_raining", "rain_intensity",
    "rain_last_3h", "visibility_m", "low_visibility",
    # calendar (weather.py / holidays package) — see module docstring
    "is_holiday", "holiday_name", "is_festival_period", "is_month_end",
    "days_to_nearest_holiday",
    # incidents (incidents.py / TomTom Incidents API) — see module docstring.
    # NOT backfillable onto pre-existing rows (no historical incident feed).
    "incident_count", "incident_total_delay_s", "incident_known_delay_count",
    "incident_max_magnitude", "has_road_closure", "has_jam", "nearest_incident_m",
]

ROUND_MINUTES = 15      # target collection cadence (also the dedupe bucket size —
                         # this single constant controls both; changing the
                         # cadence without changing this would silently start
                         # discarding samples at the old bucket size)
MAX_RETRIES = 4         # per-corridor retry budget on 429 / 5xx / network errors
BACKOFF_BASE_S = 2.0
REQUEST_TIMEOUT_S = 15
INTER_REQUEST_SLEEP_S = 0.3   # be polite to the API; spreads the 8-request burst out a bit
DEFAULT_DAILY_CAP = 2000      # used only when --budget-guard is passed


# ─────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────
def load_api_key():
    """TOMTOM_API_KEY from the environment, else parsed out of a .env file. Never logged."""
    key = os.environ.get("TOMTOM_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == "TOMTOM_API_KEY":
                    return v.strip().strip('"').strip("'")
    return None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def round_timestamp(dt, minutes=ROUND_MINUTES):
    """Round a datetime down to the nearest `minutes` boundary — used as the dedupe bucket."""
    discard = dt.minute % minutes
    return dt.replace(minute=dt.minute - discard, second=0, microsecond=0)


def fetch_corridor_summary(corridor, api_key):
    """Call the Routing API for one corridor. Retries on 429/5xx/network errors with backoff.
    Returns the 'summary' dict on success, or None if every attempt failed (caller skips it)."""
    lat1, lon1 = corridor["start"]
    lat2, lon2 = corridor["end"]
    url = ROUTING_URL_TMPL.format(start=f"{lat1},{lon1}", end=f"{lat2},{lon2}")
    params = {
        "key": api_key,
        "traffic": "true",
        "computeTravelTimeFor": "all",
        "routeRepresentation": "summaryOnly",   # skip the point geometry, we only need the summary
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r.json()["routes"][0]["summary"]
        except Exception as e:
            last_err = str(e)

        if attempt < MAX_RETRIES:
            sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_s)

    print(f"  [WARN] {corridor['name']}: failed after {MAX_RETRIES} attempts ({last_err}); skipping this corridor.")
    return None


def ensure_csv_schema():
    """Migrate data/gurugram_observed.csv to the current CSV_COLUMNS if the
    on-disk header is stale (e.g. rows collected before the weather/calendar
    columns existed). Without this, appending new-schema rows under an old
    header silently corrupts the file — csv.DictReader has no way to know
    the extra trailing fields belong to columns that didn't exist yet when
    the header was written, and mis-parses every row after the schema
    changed. Safe to call on every run; a no-op once the header is current."""
    if not os.path.exists(OUTPUT_CSV):
        return

    with open(OUTPUT_CSV, newline="") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return

    header = raw_rows[0]
    if header == CSV_COLUMNS:
        return  # already current

    print(f"[MIGRATE] {OUTPUT_CSV} header is stale ({len(header)} cols vs {len(CSV_COLUMNS)} "
          f"expected) — migrating to current schema, padding missing columns with blanks.")

    migrated = []
    for row in raw_rows[1:]:
        if len(row) == len(header):
            d = dict(zip(header, row))
        else:
            # Row already has more fields than this stale header names (e.g. a
            # prior partial migration) — take positionally by CSV_COLUMNS order.
            d = dict(zip(CSV_COLUMNS, row))
        migrated.append({col: d.get(col, "") for col in CSV_COLUMNS})

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)

    print(f"[MIGRATE] done: {len(migrated)} rows now under the current {len(CSV_COLUMNS)}-column schema.")


def load_existing_keys():
    """Set of (corridor_id, 'YYYY-MM-DDTHH:MM') already in the CSV, for dedupe against re-runs."""
    keys = set()
    if not os.path.exists(OUTPUT_CSV):
        return keys
    with open(OUTPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("corridor_id")
            ts = row.get("collected_at")
            if cid is not None and ts:
                keys.add((str(cid), ts[:16]))
    return keys


# ─────────────────────────────────────────────
# BUDGET GUARD (optional, --budget-guard)
# ─────────────────────────────────────────────
def _load_budget_state():
    if not os.path.exists(BUDGET_STATE_FILE):
        return {"date": None, "requests": 0}
    try:
        with open(BUDGET_STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"date": None, "requests": 0}


def _save_budget_state(state):
    os.makedirs(os.path.dirname(BUDGET_STATE_FILE), exist_ok=True)
    tmp = BUDGET_STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, BUDGET_STATE_FILE)


def budget_requests_used_today():
    """Requests already spent today, per the local state file. Resets across an
    IST day boundary (matches weather.py's day convention for consistency)."""
    today = str(datetime.date.today())
    state = _load_budget_state()
    if state.get("date") != today:
        return 0
    return int(state.get("requests", 0))


def budget_record_requests(n):
    today = str(datetime.date.today())
    state = _load_budget_state()
    if state.get("date") != today:
        state = {"date": today, "requests": 0}
    state["requests"] = int(state.get("requests", 0)) + n
    _save_budget_state(state)


# ─────────────────────────────────────────────
# COLLECTION
# ─────────────────────────────────────────────
def _fetch_weather_and_events(rounded):
    """One weather + calendar lookup per round (city-wide, not per-corridor) —
    stamped onto all corridor rows for this round. Never raises: a weather API
    hiccup should not lose a round of TomTom data, so failures are logged and
    the weather/event columns are left blank for this round instead."""
    date = rounded.date()
    hour = rounded.hour
    try:
        w = wx.get_hourly_weather(date, hour) or {}
    except Exception as e:
        print(f"  [WARN] weather lookup failed ({e}); weather columns left blank for this round.")
        w = {}
    try:
        e = wx.get_event_features(date)
    except Exception as ex:
        print(f"  [WARN] event-feature lookup failed ({ex}); calendar columns left blank for this round.")
        e = {}
    return {
        "temperature_c": w.get("temperature_c"),
        "precipitation_mm": w.get("precipitation_mm"),
        "is_raining": w.get("is_raining"),
        "rain_intensity": w.get("rain_intensity"),
        "rain_last_3h": w.get("rain_last_3h"),
        "visibility_m": w.get("visibility_m"),
        "low_visibility": w.get("low_visibility"),
        "is_holiday": e.get("is_holiday"),
        "holiday_name": e.get("holiday_name"),
        "is_festival_period": e.get("is_festival_period"),
        "is_month_end": e.get("is_month_end"),
        "days_to_nearest_holiday": e.get("days_to_nearest_holiday"),
    }


def _fetch_incident_features(api_key):
    """One TomTom bbox request per round, covering all 8 corridors (see
    incidents.py's QUOTA section) — NOT one request per corridor. Never
    raises: an incidents-API hiccup should not lose a round of routing data,
    so failures are logged and every corridor gets the all-zero/None default
    feature set for this round instead (see incidents.match_and_aggregate).
    Returns (per_corridor_features, tomtom_requests_used) — used is 1 on a
    real attempt (success or failure both cost the request) so the budget
    guard accounts for it; 0 if skipped outright (no key)."""
    if not api_key:
        return inc.match_and_aggregate([]), 0
    try:
        features, n_raw = inc.get_corridor_incident_features(api_key=api_key)
        print(f"  incidents: {n_raw} raw in bbox")
        return features, 1
    except Exception as e:
        print(f"  [WARN] incident fetch failed ({e}); incident columns left at defaults for this round.")
        return inc.match_and_aggregate([]), 1


def collect_round(budget_guard=False, daily_cap=DEFAULT_DAILY_CAP):
    """Run one round (up to len(CORRIDORS) requests). Returns rows written (>=0),
    or -1 if the round could not start at all (no API key configured, or the
    budget guard refuses the whole round outright)."""
    api_key = load_api_key()
    if not api_key:
        print("[ERROR] TOMTOM_API_KEY not set (checked environment and .env). Aborting round.")
        return -1

    if budget_guard:
        used = budget_requests_used_today()
        # +1 for the single incidents bbox request, on top of one routing
        # request per corridor (see _fetch_incident_features / incidents.py).
        needed = len(CORRIDORS) + 1
        if used + needed > daily_cap:
            print(f"[BUDGET GUARD] {used} TomTom requests already used today; this round needs up to "
                  f"{needed} more, which would exceed --daily-cap={daily_cap}. Refusing to start.")
            return -1

    ensure_csv_schema()

    now = datetime.datetime.now()
    rounded = round_timestamp(now)
    collected_at = rounded.isoformat()
    existing = load_existing_keys()
    weather_and_events = _fetch_weather_and_events(rounded)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)

    print(f"[{now.isoformat(timespec='seconds')}] Collecting live-observed round "
          f"({len(CORRIDORS)} corridors, bucket={collected_at})...")
    print(f"  weather: temp={weather_and_events['temperature_c']}C "
          f"precip={weather_and_events['precipitation_mm']}mm "
          f"raining={weather_and_events['is_raining']} "
          f"rain_last_3h={weather_and_events['rain_last_3h']}mm "
          f"visibility={weather_and_events['visibility_m']}m | "
          f"holiday={weather_and_events['holiday_name']} "
          f"festival_period={weather_and_events['is_festival_period']} "
          f"month_end={weather_and_events['is_month_end']}")

    all_deduped = all(
        (str(c["id"]), collected_at[:16]) in existing for c in CORRIDORS
    )
    if all_deduped:
        # Every corridor already has a row for this bucket (e.g. a re-run
        # moments later) — nothing will be written, so skip the incidents
        # bbox request rather than spending real TomTom quota on a round
        # that writes 0 rows.
        print("  (all corridors already collected for this round — skipping incidents fetch too)")
        incident_features, incident_requests_used = inc.match_and_aggregate([]), 0
    else:
        incident_features, incident_requests_used = _fetch_incident_features(api_key)

    rows_written = 0
    tomtom_requests_made = incident_requests_used
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for corridor in CORRIDORS:
            dedupe_key = (str(corridor["id"]), collected_at[:16])
            if dedupe_key in existing:
                print(f"  {corridor['name']:38s} already collected for this round — skipping (dedupe).")
                continue

            if budget_guard:
                used = budget_requests_used_today() + tomtom_requests_made
                if used >= daily_cap:
                    print(f"  [BUDGET GUARD] daily cap ({daily_cap}) reached mid-round; "
                          f"stopping early, {corridor['name']} onward skipped.")
                    break

            summary = fetch_corridor_summary(corridor, api_key)
            tomtom_requests_made += 1
            if summary is None:
                continue  # already logged in fetch_corridor_summary

            live_s = summary.get("travelTimeInSeconds")
            free_flow_s = summary.get("noTrafficTravelTimeInSeconds")
            historic_s = summary.get("historicTrafficTravelTimeInSeconds")
            delay_s = summary.get("trafficDelayInSeconds")
            length_m = summary.get("lengthInMeters")

            if not live_s or not free_flow_s:
                print(f"  [WARN] {corridor['name']}: response missing travel times; skipping.")
                continue

            congestion_idx = round(1 - (free_flow_s / live_s), 4)
            inc_feat = incident_features.get(corridor["id"], {})

            row = {
                "corridor_id": corridor["id"],
                "corridor_name": corridor["name"],
                "road_class": corridor["road_class"],
                "day_of_week": rounded.weekday(),
                "hour": rounded.hour,
                "minute": rounded.minute,
                "length_m": length_m,
                "free_flow_s": free_flow_s,
                "live_s": live_s,
                "historic_s": historic_s,
                "traffic_delay_s": delay_s,
                "congestion_idx": congestion_idx,
                "source": "observed",
                "collected_at": collected_at,
                **weather_and_events,
                "incident_count": inc_feat.get("incident_count"),
                "incident_total_delay_s": inc_feat.get("incident_total_delay_s"),
                "incident_known_delay_count": inc_feat.get("incident_known_delay_count"),
                "incident_max_magnitude": inc_feat.get("incident_max_magnitude"),
                "has_road_closure": inc_feat.get("has_road_closure"),
                "has_jam": inc_feat.get("has_jam"),
                "nearest_incident_m": inc_feat.get("nearest_incident_m"),
            }
            writer.writerow(row)
            f.flush()
            rows_written += 1
            print(f"  {corridor['name']:38s} OK  live={live_s}s free_flow={free_flow_s}s "
                  f"historic={historic_s}s delay={delay_s}s congestion_idx={congestion_idx:.3f} "
                  f"| incidents={inc_feat.get('incident_count')} "
                  f"closure={inc_feat.get('has_road_closure')} "
                  f"nearest_m={inc_feat.get('nearest_incident_m')}")
            time.sleep(INTER_REQUEST_SLEEP_S)

    if budget_guard and tomtom_requests_made:
        budget_record_requests(tomtom_requests_made)

    print(f"Round complete: {rows_written}/{len(CORRIDORS)} corridors written to {OUTPUT_CSV}")
    return rows_written


def loop_forever(budget_guard=False, daily_cap=DEFAULT_DAILY_CAP, max_rounds=None, max_minutes=None):
    """One round every ROUND_MINUTES minutes.

    With max_rounds and/or max_minutes both None (the original VM usage:
    `--loop` with nothing else), this runs forever exactly as before —
    unchanged behaviour, unchanged signature-compatible default.

    With either bound set (CI's hourly-job usage: `--loop --max-rounds 4`),
    this instead runs a BOUNDED number of evenly-15-minutes-spaced rounds
    and then returns, so the calling workflow can do one commit covering
    every round collected in this invocation. See the GITHUB CRON
    THROTTLING section of this file's module docstring for why CI wants
    this instead of a tighter cron.

    A round that raises is logged and skipped, same as always — it does
    NOT stop the loop and does NOT lose rows already written by earlier
    rounds in this same invocation (collect_round flushes each row to disk
    as it's written, so a crash mid-round only loses that round's
    not-yet-written corridors, never previously completed rounds).
    """
    bound_desc = (f"bounded: max_rounds={max_rounds}, max_minutes={max_minutes}"
                  if (max_rounds is not None or max_minutes is not None)
                  else "unbounded — Ctrl+C to stop")
    print(f"collect_live.py --loop : one round every {ROUND_MINUTES} minutes ({bound_desc}).")

    start = time.monotonic()
    round_num = 0
    while True:
        round_num += 1
        try:
            collect_round(budget_guard=budget_guard, daily_cap=daily_cap)
        except Exception as e:
            # Never let one bad round kill the whole loop, and never let it
            # take down rounds already written earlier in this invocation.
            print(f"[ERROR] round raised an unexpected exception: {e}")

        if max_rounds is not None and round_num >= max_rounds:
            print(f"[LOOP] max_rounds={max_rounds} reached after round {round_num}; exiting.")
            return
        if max_minutes is not None and (time.monotonic() - start) >= max_minutes * 60:
            print(f"[LOOP] max_minutes={max_minutes} elapsed after round {round_num}; exiting.")
            return

        time.sleep(ROUND_MINUTES * 60)


# ─────────────────────────────────────────────
# BACKFILL (one-time migration for rows collected before weather/calendar cols existed)
# ─────────────────────────────────────────────
def backfill_weather():
    """Rewrite gurugram_observed.csv in place, filling in weather/calendar
    columns for any existing row that predates this feature (or has them
    blank for any other reason). Uses weather.get_weather_range to bulk-fetch
    the whole date span covered by the file in as few HTTP requests as
    possible, rather than one request per row.

    Idempotent: rows that already have weather populated are left untouched
    (and cost no extra network calls). Safe to run repeatedly."""
    if not os.path.exists(OUTPUT_CSV):
        print(f"[INFO] {OUTPUT_CSV} does not exist yet — nothing to backfill.")
        return 0

    ensure_csv_schema()

    with open(OUTPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[INFO] observed CSV is empty — nothing to backfill.")
        return 0

    dates = []
    for r in rows:
        ts = r.get("collected_at")
        if ts:
            try:
                dates.append(datetime.date.fromisoformat(ts[:10]))
            except ValueError:
                pass
    if not dates:
        print("[WARN] no parseable collected_at timestamps found — nothing to backfill.")
        return 0

    min_date, max_date = min(dates), max(dates)
    print(f"Backfilling weather/calendar features for {len(rows)} rows spanning "
          f"{min_date} .. {max_date}...")
    wx.get_weather_range(min_date, max_date)  # bulk-prefetch into weather.py's cache

    updated = 0
    event_cache = {}
    for r in rows:
        # Skip rows that already have weather (idempotent re-runs, no extra work).
        if r.get("precipitation_mm") not in (None, ""):
            continue
        ts = r.get("collected_at")
        if not ts:
            continue
        try:
            date = datetime.date.fromisoformat(ts[:10])
            hour = int(r["hour"])
        except (ValueError, KeyError):
            continue

        w = wx.get_hourly_weather(date, hour) or {}
        if date not in event_cache:
            event_cache[date] = wx.get_event_features(date)
        e = event_cache[date]

        r["temperature_c"] = w.get("temperature_c")
        r["precipitation_mm"] = w.get("precipitation_mm")
        r["is_raining"] = w.get("is_raining")
        r["rain_intensity"] = w.get("rain_intensity")
        r["rain_last_3h"] = w.get("rain_last_3h")
        r["visibility_m"] = w.get("visibility_m")
        r["low_visibility"] = w.get("low_visibility")
        r["is_holiday"] = e.get("is_holiday")
        r["holiday_name"] = e.get("holiday_name")
        r["is_festival_period"] = e.get("is_festival_period")
        r["is_month_end"] = e.get("is_month_end")
        r["days_to_nearest_holiday"] = e.get("days_to_nearest_holiday")
        updated += 1

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    print(f"Backfill complete: {updated}/{len(rows)} rows updated, {OUTPUT_CSV} rewritten.")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Collect live observed Gurugram traffic congestion via the TomTom Routing API."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run a single collection round and exit (for CI).")
    mode.add_argument("--loop", action="store_true",
                       help="Run one round every 15 minutes. Forever by default (for a VM); pass "
                            "--max-rounds and/or --max-minutes to stop after a bound instead (CI).")
    mode.add_argument("--backfill-weather", action="store_true",
                       help="One-time migration: attach weather/calendar columns to existing rows, then exit.")
    parser.add_argument("--budget-guard", action="store_true",
                         help="Track TomTom requests used today and refuse to exceed --daily-cap.")
    parser.add_argument("--daily-cap", type=int, default=DEFAULT_DAILY_CAP,
                         help=f"Max TomTom requests/day when --budget-guard is set (default {DEFAULT_DAILY_CAP}).")
    parser.add_argument("--max-rounds", type=int, default=None,
                         help="With --loop, stop after this many rounds instead of looping forever "
                              "(e.g. CI's hourly job uses --max-rounds 4 to collect ~45 minutes of "
                              "15-minute-spaced rounds per invocation). Ignored/invalid without --loop.")
    parser.add_argument("--max-minutes", type=int, default=None,
                         help="With --loop, stop after this many minutes elapsed, even if --max-rounds "
                              "hasn't been reached yet — a safety cap so a slow round can't push the "
                              "job past the next scheduled invocation. Ignored/invalid without --loop.")
    args = parser.parse_args()

    if (args.max_rounds is not None or args.max_minutes is not None) and not args.loop:
        parser.error("--max-rounds/--max-minutes only apply with --loop.")

    if args.backfill_weather:
        backfill_weather()
        sys.exit(0)
    elif args.once:
        n = collect_round(budget_guard=args.budget_guard, daily_cap=args.daily_cap)
        sys.exit(1 if n < 0 else 0)
    else:
        loop_forever(budget_guard=args.budget_guard, daily_cap=args.daily_cap,
                      max_rounds=args.max_rounds, max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
