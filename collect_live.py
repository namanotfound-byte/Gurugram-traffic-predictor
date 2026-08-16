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
TomTom's free tier is 2,500 requests/day. One round here is exactly
len(CORRIDORS) == 8 requests (one calculateRoute call per corridor). At
the intended cadence of one round every 30 minutes:

    8 requests/round * 48 rounds/day = 384 requests/day

That leaves roughly 2,500 - 384 ~= 2,100 requests/day of headroom, but be
aware the one-off bootstrap historical sweep (bootstrap_collect.py) draws
from the *same* daily quota if it shares a key — run it accordingly.

USAGE
-----
    python collect_live.py --once      # single round, then exit (used by CI)
    python collect_live.py --loop      # loop forever, one round / 30 min (VM)

The TomTom key is read from the TOMTOM_API_KEY environment variable, or
else from a .env file (KEY=VALUE lines) in this file's directory. The key
is never printed, logged, or written anywhere by this script.

Imports corridor definitions from corridors.py — the single source of
truth for this repo. Corridors are never redefined here.
"""

import argparse
import csv
import datetime
import os
import random
import sys
import time

import requests

from corridors import CORRIDORS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(REPO_ROOT, "data", "gurugram_observed.csv")
ENV_FILE = os.path.join(REPO_ROOT, ".env")

ROUTING_URL_TMPL = "https://api.tomtom.com/routing/1/calculateRoute/{start}:{end}/json"

CSV_COLUMNS = [
    "corridor_id", "corridor_name", "road_class", "day_of_week", "hour",
    "minute", "length_m", "free_flow_s", "live_s", "historic_s",
    "traffic_delay_s", "congestion_idx", "source", "collected_at",
]

ROUND_MINUTES = 30      # target collection cadence (also the dedupe bucket size)
MAX_RETRIES = 4         # per-corridor retry budget on 429 / 5xx / network errors
BACKOFF_BASE_S = 2.0
REQUEST_TIMEOUT_S = 15
INTER_REQUEST_SLEEP_S = 0.3   # be polite to the API; spreads the 8-request burst out a bit


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
# COLLECTION
# ─────────────────────────────────────────────
def collect_round():
    """Run one round (up to len(CORRIDORS) requests). Returns rows written (>=0),
    or -1 if the round could not start at all (no API key configured)."""
    api_key = load_api_key()
    if not api_key:
        print("[ERROR] TOMTOM_API_KEY not set (checked environment and .env). Aborting round.")
        return -1

    now = datetime.datetime.now()
    rounded = round_timestamp(now)
    collected_at = rounded.isoformat()
    existing = load_existing_keys()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)

    print(f"[{now.isoformat(timespec='seconds')}] Collecting live-observed round "
          f"({len(CORRIDORS)} corridors, bucket={collected_at})...")

    rows_written = 0
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for corridor in CORRIDORS:
            dedupe_key = (str(corridor["id"]), collected_at[:16])
            if dedupe_key in existing:
                print(f"  {corridor['name']:38s} already collected for this round — skipping (dedupe).")
                continue

            summary = fetch_corridor_summary(corridor, api_key)
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
            }
            writer.writerow(row)
            f.flush()
            rows_written += 1
            print(f"  {corridor['name']:38s} OK  live={live_s}s free_flow={free_flow_s}s "
                  f"historic={historic_s}s delay={delay_s}s congestion_idx={congestion_idx:.3f}")
            time.sleep(INTER_REQUEST_SLEEP_S)

    print(f"Round complete: {rows_written}/{len(CORRIDORS)} corridors written to {OUTPUT_CSV}")
    return rows_written


def loop_forever():
    print(f"collect_live.py --loop : one round every {ROUND_MINUTES} minutes. Ctrl+C to stop.")
    while True:
        try:
            collect_round()
        except Exception as e:
            # Never let one bad round kill the whole loop.
            print(f"[ERROR] round raised an unexpected exception: {e}")
        time.sleep(ROUND_MINUTES * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect live observed Gurugram traffic congestion via the TomTom Routing API."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run a single collection round and exit (for CI).")
    mode.add_argument("--loop", action="store_true", help="Run forever, one round every 30 minutes (for a VM).")
    args = parser.parse_args()

    if args.once:
        n = collect_round()
        sys.exit(1 if n < 0 else 0)
    else:
        loop_forever()


if __name__ == "__main__":
    main()
