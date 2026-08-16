#!/usr/bin/env python3
"""
Generate frontend/corridors.geojson from real TomTom road geometry.
=====================================================================
One-time, offline generator. Run manually whenever corridors.py changes;
do NOT call this at page load — the frontend reads the committed
corridors.geojson as a static file.

Usage:
    python3 tools/build_corridor_geojson.py

Reads TOMTOM_API_KEY from .env at the repo root (never printed/logged).
Imports corridor definitions from corridors.py (the single source of
truth) — never redefines them here.

For each corridor, requests routeRepresentation=polyline geometry from the
TomTom Routing API at an off-peak departure time (03:00 local) so that
corridors which reroute by time of day (Golf Course Extension Rd, Southern
Peripheral Rd, Mehrauli-Gurgaon Rd) resolve to their stable/modal route —
the one whose length matches `verified_km` in corridors.py — rather than a
congestion-driven detour.

After fetching, each corridor's polyline length is checked against
`verified_km` (+/- 5%). Any corridor outside tolerance is reported and the
script exits non-zero WITHOUT writing output, so a bad fetch can never
silently produce a geojson that disagrees with the frozen corridor data.
"""
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from corridors import CORRIDORS  # noqa: E402  (single source of truth)

ENV_PATH = os.path.join(REPO_ROOT, ".env")
OUTPUT_PATH = os.path.join(REPO_ROOT, "frontend", "corridors.geojson")
TOLERANCE = 0.05  # +/- 5% of verified_km


def load_tomtom_key():
    """Parse TOMTOM_API_KEY out of .env without ever printing its value."""
    if not os.path.exists(ENV_PATH):
        raise SystemExit(".env not found at repo root — cannot read TOMTOM_API_KEY")
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "TOMTOM_API_KEY":
                v = v.strip().strip('"').strip("'")
                if not v:
                    raise SystemExit("TOMTOM_API_KEY is present in .env but empty")
                return v
    raise SystemExit("TOMTOM_API_KEY not found in .env")


def next_offpeak_depart_at():
    """Next 03:00 IST that is at least 1 hour in the future, as an ISO8601
    string with the IST offset (what TomTom's departAt expects)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    candidate = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if candidate <= now + timedelta(hours=1):
        candidate += timedelta(days=1)
    return candidate.strftime("%Y-%m-%dT%H:%M:%S") + "+05:30"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def polyline_length_km(points):
    total = 0.0
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        total += haversine_km(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
    return total


def fetch_route(api_key, corridor, depart_at, retries=3):
    lat1, lon1 = corridor["start"]
    lat2, lon2 = corridor["end"]
    path = f"{lat1},{lon1}:{lat2},{lon2}"
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/{path}/json"
        f"?key={api_key}&routeRepresentation=polyline&departAt={urllib.parse.quote(depart_at)}"
    )
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            last_err = f"HTTP {e.code}: {body[:300]}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        if attempt < retries:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"TomTom request failed for corridor {corridor['id']} ({corridor['name']}): {last_err}")


def main():
    api_key = load_tomtom_key()
    depart_at = next_offpeak_depart_at()
    print(f"Using off-peak departAt={depart_at} (key redacted)")

    features = []
    mismatches = []

    for c in CORRIDORS:
        print(f"[{c['id']}] {c['name']} ({c['sub']}) ...", end=" ", flush=True)
        data = fetch_route(api_key, c, depart_at)
        routes = data.get("routes") or []
        if not routes:
            mismatches.append((c, None, f"no routes returned: {json.dumps(data)[:300]}"))
            print("NO ROUTE")
            continue
        legs = routes[0].get("legs") or []
        if not legs:
            mismatches.append((c, None, "no legs in route"))
            print("NO LEGS")
            continue
        points = legs[0].get("points") or []
        if len(points) < 2:
            mismatches.append((c, None, f"only {len(points)} points"))
            print("TOO FEW POINTS")
            continue

        length_km = polyline_length_km(points)
        verified = c["verified_km"]
        rel_diff = abs(length_km - verified) / verified
        status = "OK" if rel_diff <= TOLERANCE else "MISMATCH"
        print(f"{len(points)} pts, {length_km:.2f} km (verified {verified} km, diff {rel_diff*100:.1f}%) {status}")

        if rel_diff > TOLERANCE:
            mismatches.append((c, length_km, f"{length_km:.2f} km vs verified {verified} km ({rel_diff*100:.1f}% off, tolerance {TOLERANCE*100:.0f}%)"))
            continue

        coords = [[p["longitude"], p["latitude"]] for p in points]
        features.append({
            "type": "Feature",
            "properties": {
                "corridor_id": c["id"],
                "name": c["name"],
                "sub": c["sub"],
                "road_class": c["road_class"],
                "length_km": round(length_km, 2),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
        })

    if mismatches:
        print("\nFAILED — the following corridors did not resolve to their verified route:", file=sys.stderr)
        for c, length_km, msg in mismatches:
            print(f"  [{c['id']}] {c['name']}: {msg}", file=sys.stderr)
        print("\nNo output written. Investigate before committing corridors.geojson.", file=sys.stderr)
        sys.exit(1)

    features.sort(key=lambda f: f["properties"]["corridor_id"])
    geojson = {"type": "FeatureCollection", "features": features}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(geojson, f, indent=1)
        f.write("\n")

    print(f"\nWrote {len(features)} corridor geometries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
