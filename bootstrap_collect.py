"""
bootstrap_collect.py
=====================
Sweeps TomTom's Routing API (with a future departAt + computeTravelTimeFor=all)
across all 8 frozen corridors, 7 days, 24 hours to bootstrap a real-data
training set for the congestion model.

Technique (validated separately): TomTom Routing accepts a future `departAt`
and, with `computeTravelTimeFor=all`, returns TomTom's own historical traffic
model for that road at that time of day/week — even though this API key is
NOT entitled to the Flow API. See routes[0].summary in the response:

  https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json
    ?key=KEY&departAt=YYYY-MM-DDTHH:00:00&computeTravelTimeFor=all&traffic=true

congestion_index = 1 - (noTrafficTravelTimeInSeconds / historicTrafficTravelTimeInSeconds)

Usage:
  python bootstrap_collect.py [--max-requests 1400] [--out data/gurugram_bootstrap.csv]

Resumable: on restart, any (corridor_id, day_of_week, hour) cell already
present in the output CSV is skipped, so a crash never wastes prior requests.

Route-consistency: the routing engine sometimes reroutes a corridor onto a
physically different path at different departure times (observed directly:
Golf Course Extension Road, Mehrauli-Gurgaon Road, and Southern Peripheral
Road all showed >9% length_m swings between hours, while the other 5
corridors stayed within 1%). Each row gets a `route_stable` boolean: True iff
`length_m` is within ROUTE_STABLE_TOL_PCT of the corridor's `verified_km`
reference length from corridors.py (the length recorded when that corridor
was validated). Rows with route_stable=False are measuring a different road
than the frozen corridor definition and are excluded from training by
default — see model/traffic_model.py.
"""

import argparse
import csv
import datetime
import os
import sys
import time

import requests

import corridors

DEFAULT_OUT = "data/gurugram_bootstrap.csv"
FIELDNAMES = [
    "corridor_id", "corridor_name", "road_class", "day_of_week", "hour",
    "date", "length_m", "route_stable", "free_flow_s", "historic_s", "live_s",
    "traffic_delay_s", "congestion_idx", "source", "collected_at",
]

# A row's length_m must be within this % of the corridor's verified_km
# reference (corridors.py) to count as measuring the same physical road.
# Empirically the split is clean: every corridor's rows are either within
# ~1% of the reference or >9% off — there is no ambiguous middle ground.
ROUTE_STABLE_TOL_PCT = 2.0


def is_route_stable(corridor: dict, length_m: float) -> bool:
    ref_m = corridor["verified_km"] * 1000
    if ref_m <= 0:
        return True
    deviation_pct = abs(length_m - ref_m) / ref_m * 100
    return deviation_pct <= ROUTE_STABLE_TOL_PCT

# Upcoming week: Mon 2026-08-17 .. Sun 2026-08-23 (must be in the future
# relative to "today" = 2026-08-16 so TomTom treats departAt as a forecast).
WEEK_START = datetime.date(2026, 8, 17)  # Monday, day_of_week = 0

SLEEP_MIN = 0.15
SLEEP_MAX = 0.25
MAX_RETRIES = 3


def load_api_key() -> str:
    """Parse .env by hand — never rely on the key being exported already."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        print("[ERROR] .env not found; cannot load TOMTOM_API_KEY.", file=sys.stderr)
        sys.exit(1)
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "TOMTOM_API_KEY":
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    print("[ERROR] TOMTOM_API_KEY not found in .env.", file=sys.stderr)
    sys.exit(1)


def date_for(day_of_week: int) -> datetime.date:
    """day_of_week: 0=Mon .. 6=Sun, mapped onto WEEK_START's week."""
    return WEEK_START + datetime.timedelta(days=day_of_week)


def load_existing_keys(out_path: str) -> set:
    """Returns set of (corridor_id, day_of_week, hour) already collected."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["corridor_id"]), int(row["day_of_week"]), int(row["hour"]))
                done.add(key)
            except (KeyError, ValueError):
                continue
    return done


def fetch_route(session, api_key, corridor, depart_at_iso):
    """One TomTom Routing API call. Returns summary dict or raises."""
    pair = corridors.route_pair(corridor)
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{pair}/json"
    params = {
        "key": api_key,
        "departAt": depart_at_iso,
        "computeTravelTimeFor": "all",
        "traffic": "true",
    }
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 429 or r.status_code >= 500:
                wait = (2 ** (attempt - 1)) * 1.0
                print(f"    [RETRY] HTTP {r.status_code}, backing off {wait:.1f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                continue
            r.raise_for_status()
            data = r.json()
            summary = data["routes"][0]["summary"]
            return summary
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = (2 ** (attempt - 1)) * 1.0
                print(f"    [RETRY] {e}, backing off {wait:.1f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
    raise last_exc


def main():
    ap = argparse.ArgumentParser(description="Bootstrap real TomTom traffic data.")
    ap.add_argument("--max-requests", type=int, default=1400,
                     help="Hard cap on API requests this run (default 1400).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT,
                     help=f"Output CSV path (default {DEFAULT_OUT}).")
    args = ap.parse_args()

    api_key = load_api_key()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    existing = load_existing_keys(args.out)
    print(f"[INFO] {len(existing)} cells already collected in {args.out}; will skip those.")

    # Build the full work plan: 8 corridors x 7 days x 24 hours.
    plan = []
    for corridor in corridors.CORRIDORS:
        for dow in range(7):
            for hour in range(24):
                key = (corridor["id"], dow, hour)
                if key in existing:
                    continue
                plan.append((corridor, dow, hour))

    total_cells = len(corridors.CORRIDORS) * 7 * 24
    print(f"[INFO] Full sweep = {total_cells} cells. {len(plan)} remaining to collect.")
    print(f"[INFO] Quota guard: --max-requests={args.max_requests}")

    file_exists = os.path.exists(args.out)
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    session = requests.Session()
    request_count = 0
    ok_count = 0
    fail_count = 0
    t_start = time.time()

    try:
        for corridor, dow, hour in plan:
            if request_count >= args.max_requests:
                print(f"[STOP] Hit --max-requests={args.max_requests}. "
                      f"{len(plan) - request_count} cells remain for next run.")
                break

            the_date = date_for(dow)
            depart_at_iso = f"{the_date.isoformat()}T{hour:02d}:00:00"

            request_count += 1
            elapsed = time.time() - t_start
            print(f"[{request_count}/{min(len(plan), args.max_requests)}] "
                  f"corridor={corridor['id']} ({corridor['name']}) "
                  f"dow={dow} hour={hour:02d} depart={depart_at_iso} "
                  f"(elapsed {elapsed:.0f}s)...", end=" ", flush=True)

            try:
                summary = fetch_route(session, api_key, corridor, depart_at_iso)
            except Exception as e:
                fail_count += 1
                print(f"FAILED ({e})")
                time.sleep(SLEEP_MIN)
                continue

            length_m = summary.get("lengthInMeters")
            free_flow_s = summary.get("noTrafficTravelTimeInSeconds")
            historic_s = summary.get("historicTrafficTravelTimeInSeconds")
            live_s = summary.get("travelTimeInSeconds")
            delay_s = summary.get("trafficDelayInSeconds")

            if not free_flow_s or not historic_s or historic_s <= 0:
                fail_count += 1
                print(f"SKIP (missing/zero timing fields: {summary})")
                time.sleep(SLEEP_MIN)
                continue

            congestion_idx = round(1 - (free_flow_s / historic_s), 4)
            route_stable = is_route_stable(corridor, length_m)

            row = {
                "corridor_id": corridor["id"],
                "corridor_name": corridor["name"],
                "road_class": corridor["road_class"],
                "day_of_week": dow,
                "hour": hour,
                "date": the_date.isoformat(),
                "length_m": length_m,
                "route_stable": route_stable,
                "free_flow_s": free_flow_s,
                "historic_s": historic_s,
                "live_s": live_s,
                "traffic_delay_s": delay_s,
                "congestion_idx": congestion_idx,
                "source": "bootstrap",
                "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            writer.writerow(row)
            out_f.flush()
            ok_count += 1
            stable_tag = "" if route_stable else "  [UNSTABLE ROUTE]"
            print(f"OK  len={length_m}m ff={free_flow_s}s hist={historic_s}s "
                  f"idx={congestion_idx:.3f}{stable_tag}")

            sleep_s = SLEEP_MIN + (SLEEP_MAX - SLEEP_MIN) * ((request_count % 7) / 7)
            time.sleep(sleep_s)
    finally:
        out_f.close()

    print("\n" + "=" * 60)
    print(f"[DONE] requests made: {request_count}  ok: {ok_count}  failed: {fail_count}")
    print(f"[DONE] total elapsed: {time.time() - t_start:.0f}s")
    print(f"[DONE] output: {args.out}")

    # ── Route-consistency check ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[CHECK] Per-corridor length_m consistency (min/max/median):")
    check_route_consistency(args.out)


def check_route_consistency(out_path):
    import statistics
    from collections import defaultdict

    lengths = defaultdict(list)
    unstable_counts = defaultdict(int)
    names = {}
    with open(out_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row["corridor_id"])
                length = float(row["length_m"])
            except (KeyError, ValueError, TypeError):
                continue
            lengths[cid].append(length)
            names[cid] = row["corridor_name"]
            if row.get("route_stable", "True").strip().lower() in ("false", "0"):
                unstable_counts[cid] += 1

    flagged = []
    for cid in sorted(lengths):
        vals = lengths[cid]
        if not vals:
            continue
        lo, hi, med = min(vals), max(vals), statistics.median(vals)
        variance_pct = (hi - lo) / med * 100 if med else 0.0
        n_unstable = unstable_counts.get(cid, 0)
        flag = " *** FLAGGED: >15% length variance ***" if variance_pct > 15 else ""
        if flag:
            flagged.append((cid, names[cid], variance_pct, n_unstable))
        print(f"  corridor {cid:>2} {names[cid]:38s} n={len(vals):4d} "
              f"min={lo:8.0f}m max={hi:8.0f}m median={med:8.0f}m "
              f"variance={variance_pct:5.1f}%  route_stable=False: {n_unstable:3d}{flag}")

    if flagged:
        print("\n[WARN] Corridors with inconsistent routing (different physical "
              "route chosen at different times) — the route_stable column marks "
              "which rows deviate from the corridors.py verified_km reference; "
              "model/traffic_model.py excludes route_stable=False rows by default:")
        for cid, name, pct, n_unstable in flagged:
            print(f"  - corridor {cid} ({name}): {pct:.1f}% length variance, "
                  f"{n_unstable} row(s) flagged route_stable=False")
    else:
        print("\n[OK] All corridors within 15% length variance.")


if __name__ == "__main__":
    main()
