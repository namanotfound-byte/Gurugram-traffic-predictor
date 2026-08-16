#!/usr/bin/env python3
"""
incidents.py — TomTom Traffic Incidents, matched spatially to corridors.
============================================================================
WHY THIS MATTERS MORE THAN WEATHER
------------------------------------
weather.py and the calendar features answer "is this an unusual DAY" (rain,
a holiday, salary week). They can never explain a crash, a stalled truck, or
a road closure on one specific corridor at one specific moment — the single
largest driver of the residual model/forecast_model.py is trying to predict.
This file is what lets the model see that.

WHAT THE API ACTUALLY RETURNS (tested live, 2026-08-17, not assumed)
-----------------------------------------------------------------------
A single `bbox` query covering the whole Gurugram area (derived from
corridors.py's own coordinate envelope, not hardcoded) returned 77 live
incidents in one request:

    iconCategory:      {8 (road closed): 70, 6 (jam): 5, 9 (roadworks): 2}
    magnitudeOfDelay:  {4 (severe): 70, 3: 3, 2: 1, 1: 1, 0: 2}
    delay (seconds):   NULL on 72/77 (93.5%) — only 5 incidents carry a
                        numeric delay. magnitudeOfDelay, by contrast, is
                        populated on all 77 — it is the reliable severity
                        signal here, not raw delay.
    length (metres):   mostly small (33-527 m), with 19 duplicate values
                        among the 70 "Closed" entries.

HONEST CHARACTERIZATION: the "Closed" (iconCategory 8) entries dominate the
feed overwhelmingly and mostly carry no numeric delay. Matched against the
8 corridors (see buffer discussion below), several cluster tightly around
Dwarka Expressway specifically — consistent with that corridor's real,
long-running construction rather than a single fresh crash. This means
`has_road_closure` / `incident_count` should be read as "this corridor
currently has an active closure/worksite nearby," which can be a
slowly-changing (day-to-day) signal for some corridors, not a
minute-to-minute one. That's still a real and useful feature (a closure
sustained across a work-shift genuinely does explain sustained extra
congestion) — it's just not automatically "a crash just happened," and the
README/report says so plainly rather than overselling it. No incidents were
dropped on account of this; see FILTERING below for what actually is (and
isn't) filtered, and why.

SPATIAL MATCHING
------------------
Corridors are named roads; incidents come with their own point/line
geometry. An incident 5 km away on an unrelated road must not be counted
against a corridor just because it's inside the same bounding box. Each
incident is matched to a corridor by the minimum distance from ANY vertex
of the incident's geometry to ANY segment of the corridor's digitized
polyline (frontend/corridors.geojson — read-only, owned by another agent;
this file only reads it, never writes it).

BUFFER THRESHOLD: 300 metres, chosen empirically (not guessed) by pulling
77 real Gurugram incidents and computing each one's distance to its nearest
corridor: distances split cleanly into a small cluster under a few hundred
metres and the large majority (65/77, ~84%) beyond 500 m on unrelated
roads. 300 m sits inside that gap — wide enough to absorb a divided
highway's carriageway width, service-road offsets, and ordinary
GPS/geocoding slop, but narrower than the typical spacing between distinct
named roads in Gurugram's arterial grid (which would otherwise wrongly
attribute a neighbouring corridor's incident to this one). It's a tunable
constant (`DEFAULT_BUFFER_M`), not a hardcoded assumption — re-derive it if
the incident mix looks different once more rounds of data come in.

FILTERING: no length- or magnitude-based hard filter is applied. Spatial
matching (the 300 m buffer) already removes ~84% of the raw feed as
irrelevant to these 8 corridors, which is the filter that actually matters.
`magnitudeOfDelay` is exposed as `incident_max_magnitude` per corridor
(reliable, never null in the sample pulled). `delay` is summed where
present and simply excluded (not assumed zero) where null — see
aggregate_corridor_incident_features's docstring for the exact contract.

QUOTA
-----
ONE bbox request per round covers all 8 corridors (not one request per
corridor) — this is what makes incidents cheap to add: it's +1 request per
15-minute round (roughly +96/day), stacked on top of collect_live.py's 768
TomTom-routing requests/day, still comfortably inside the 2,500/day free
tier shared across every TomTom API this project uses.
"""

import json
import math
import os

import requests

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORRIDORS_GEOJSON = os.path.join(REPO_ROOT, "frontend", "corridors.geojson")  # read-only
ENV_FILE = os.path.join(REPO_ROOT, ".env")

INCIDENTS_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"
INCIDENT_FIELDS = (
    "{incidents{type,geometry{type,coordinates},"
    "properties{iconCategory,magnitudeOfDelay,delay,length,roadNumbers,events{description}}}}"
)
REQUEST_TIMEOUT_S = 20

# ~3km margin beyond the corridors' own coordinate envelope, so incidents
# just past a corridor's endpoint (e.g. just past Manesar on NH-48) aren't
# clipped out of the query itself before spatial matching even runs.
BBOX_MARGIN_DEG = 0.03

# TomTom iconCategory reference (the ones actually seen in this bbox; TomTom
# defines more categories than this, unused ones aren't enumerated here).
ICON_ROAD_CLOSED = 8
ICON_JAM = 6
ICON_ROAD_WORKS = 9

DEFAULT_BUFFER_M = 300.0  # see module docstring for the empirical justification

_EARTH_R_M = 6371000.0
# Single projection origin for the whole bbox (~30km across) — equirectangular
# approximation, error well under 1% at this scale, consistent with the
# approach used for the empirical buffer-threshold analysis above.
_LAT0 = 28.45
_LON0 = 77.02


# ─────────────────────────────────────────────
# API KEY (duplicated, minimal, from collect_live.py's load_api_key — kept
# separate rather than imported to avoid a circular import, since
# collect_live.py is the one that imports THIS module)
# ─────────────────────────────────────────────
def _load_tomtom_key():
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
# GEOMETRY (pure-python; no shapely dependency)
# ─────────────────────────────────────────────
def _to_xy(lat, lon):
    """Local equirectangular projection to metres, centred on the bbox."""
    x = math.radians(lon - _LON0) * _EARTH_R_M * math.cos(math.radians(_LAT0))
    y = math.radians(lat - _LAT0) * _EARTH_R_M
    return x, y


def _point_segment_distance_m(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _min_distance_point_to_polyline_m(pt_xy, polyline_xy):
    px, py = pt_xy
    best = float("inf")
    for i in range(len(polyline_xy) - 1):
        ax, ay = polyline_xy[i]
        bx, by = polyline_xy[i + 1]
        d = _point_segment_distance_m(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    if len(polyline_xy) == 1:  # degenerate single-point "polyline"
        ax, ay = polyline_xy[0]
        best = math.hypot(px - ax, py - ay)
    return best


def _incident_to_corridor_distance_m(incident_coords_lonlat, corridor_xy):
    """Min distance from ANY vertex of the incident's geometry to ANY segment
    of the corridor's polyline. Approximates true polyline-to-polyline
    distance well for this use case (short incident geometries, corridor
    polylines with dense vertices — see corridors.geojson point counts)."""
    best = float("inf")
    for lon, lat in incident_coords_lonlat:
        pt_xy = _to_xy(lat, lon)
        d = _min_distance_point_to_polyline_m(pt_xy, corridor_xy)
        if d < best:
            best = d
    return best


# ─────────────────────────────────────────────
# CORRIDOR GEOMETRY (read-only load of frontend/corridors.geojson)
# ─────────────────────────────────────────────
_corridor_polylines_cache = None  # {corridor_id: (name, [(x,y), ...])}


def load_corridor_polylines():
    """Reads frontend/corridors.geojson (owned by another agent — read-only,
    never written here) and returns {corridor_id: (name, [(x,y) metres, ...])}
    in the projected coordinate system used for distance calculations."""
    global _corridor_polylines_cache
    if _corridor_polylines_cache is not None:
        return _corridor_polylines_cache

    if not os.path.exists(CORRIDORS_GEOJSON):
        raise FileNotFoundError(
            f"{CORRIDORS_GEOJSON} not found — incident matching needs real corridor "
            f"polylines and will not fall back to straight-line start->end approximations."
        )

    with open(CORRIDORS_GEOJSON) as f:
        geo = json.load(f)

    out = {}
    for feat in geo["features"]:
        cid = feat["properties"]["corridor_id"]
        name = feat["properties"]["name"]
        coords = feat["geometry"]["coordinates"]  # [lon, lat] pairs
        xy = [_to_xy(lat, lon) for lon, lat in coords]
        out[cid] = (name, xy)

    _corridor_polylines_cache = out
    return out


def _bbox_from_corridors():
    polylines = load_corridor_polylines()
    lats, lons = [], []
    # Recompute lat/lon envelope from the raw geojson rather than the
    # projected xy (keeps this independent of the projection choice).
    with open(CORRIDORS_GEOJSON) as f:
        geo = json.load(f)
    for feat in geo["features"]:
        for lon, lat in feat["geometry"]["coordinates"]:
            lats.append(lat)
            lons.append(lon)
    min_lon, max_lon = min(lons) - BBOX_MARGIN_DEG, max(lons) + BBOX_MARGIN_DEG
    min_lat, max_lat = min(lats) - BBOX_MARGIN_DEG, max(lats) + BBOX_MARGIN_DEG
    return f"{min_lon:.5f},{min_lat:.5f},{max_lon:.5f},{max_lat:.5f}"


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────
def fetch_incidents(bbox=None, api_key=None):
    """One request, covering the whole corridor bbox. Returns a list of
    dicts: {icon_category, magnitude, delay_s (or None), length_m (or None),
    description, coords (list of (lon, lat))}. Returns [] (not an
    exception) on any request failure, with a printed warning — a incidents
    outage should not take down a whole collection round."""
    api_key = api_key or _load_tomtom_key()
    if not api_key:
        print("[WARN] incidents.fetch_incidents: no TOMTOM_API_KEY available; returning no incidents.")
        return []

    bbox = bbox or _bbox_from_corridors()
    params = {
        "key": api_key,
        "bbox": bbox,
        "language": "en-GB",
        "fields": INCIDENT_FIELDS,
    }
    try:
        r = requests.get(INCIDENTS_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"[WARN] incidents.fetch_incidents: request failed ({e}); returning no incidents.")
        return []

    out = []
    for inc in payload.get("incidents", []):
        props = inc.get("properties", {})
        geom = inc.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Point":
            coords = [coords]  # normalise to a list of [lon, lat] pairs
        events = props.get("events") or []
        description = events[0].get("description") if events else None
        out.append({
            "icon_category": props.get("iconCategory"),
            "magnitude": props.get("magnitudeOfDelay"),
            "delay_s": props.get("delay"),          # often None — see module docstring
            "length_m": props.get("length"),
            "description": description,
            "coords": [(c[0], c[1]) for c in coords],  # (lon, lat)
        })
    return out


# ─────────────────────────────────────────────
# MATCH + AGGREGATE
# ─────────────────────────────────────────────
def match_and_aggregate(incidents, buffer_m=DEFAULT_BUFFER_M):
    """Per-corridor incident features for one round.

    Returns {corridor_id: {...}} with:
        incident_count          # incidents within buffer_m of this corridor
        incident_total_delay_s  # sum of delay_s among matched incidents that
                                 # HAVE a numeric delay; incidents with
                                 # delay_s=None contribute nothing to this sum
                                 # (excluded, not assumed 0) — so treat this
                                 # figure as a floor, not an exact total, given
                                 # ~93% of raw incidents carry no delay value.
        incident_known_delay_count  # how many matched incidents contributed
                                     # to incident_total_delay_s, so a
                                     # consumer can tell "0 because no delay"
                                     # apart from "0 because no data".
        incident_max_magnitude  # 0-4, max magnitudeOfDelay among matched
                                 # incidents (0 if none matched) — the
                                 # reliable severity signal (never null in
                                 # the raw feed, unlike delay_s).
        has_road_closure        # any matched incident with iconCategory==8
        has_jam                 # any matched incident with iconCategory==6
        nearest_incident_m      # distance to the single nearest incident
                                 # IN THE WHOLE FETCHED SET, regardless of
                                 # buffer_m — a continuous feature so the
                                 # model isn't blind to "almost affected"
                                 # corridors just outside the hard cutoff.
                                 # None if no incidents were fetched at all.
    """
    polylines = load_corridor_polylines()
    result = {}
    for cid, (name, _) in polylines.items():
        result[cid] = {
            "incident_count": 0,
            "incident_total_delay_s": 0.0,
            "incident_known_delay_count": 0,
            "incident_max_magnitude": 0,
            "has_road_closure": False,
            "has_jam": False,
            "nearest_incident_m": None,
        }

    if not incidents:
        return result

    for cid, (name, xy) in polylines.items():
        nearest = float("inf")
        for inc in incidents:
            if not inc["coords"]:
                continue
            d = _incident_to_corridor_distance_m(inc["coords"], xy)
            if d < nearest:
                nearest = d
            if d <= buffer_m:
                r = result[cid]
                r["incident_count"] += 1
                if inc["delay_s"] is not None:
                    r["incident_total_delay_s"] += float(inc["delay_s"])
                    r["incident_known_delay_count"] += 1
                mag = inc["magnitude"] or 0
                if mag > r["incident_max_magnitude"]:
                    r["incident_max_magnitude"] = mag
                if inc["icon_category"] == ICON_ROAD_CLOSED:
                    r["has_road_closure"] = True
                if inc["icon_category"] == ICON_JAM:
                    r["has_jam"] = True
        result[cid]["nearest_incident_m"] = round(nearest, 1) if nearest != float("inf") else None

    return result


def get_corridor_incident_features(buffer_m=DEFAULT_BUFFER_M, api_key=None):
    """Convenience wrapper: fetch + match + aggregate in one call. This is
    what collect_live.py calls once per round (ONE HTTP request covers all
    8 corridors — see module docstring's QUOTA section)."""
    incidents = fetch_incidents(api_key=api_key)
    return match_and_aggregate(incidents, buffer_m=buffer_m), len(incidents)


if __name__ == "__main__":
    # Manual smoke test: python incidents.py
    feats, n_raw = get_corridor_incident_features()
    print(f"Fetched {n_raw} raw incidents in bbox. Per-corridor (buffer={DEFAULT_BUFFER_M}m):")
    for cid, f in sorted(feats.items()):
        name = load_corridor_polylines()[cid][0]
        print(f"  [{cid}] {name:38s} count={f['incident_count']:>2} "
              f"max_mag={f['incident_max_magnitude']} closure={f['has_road_closure']} "
              f"jam={f['has_jam']} nearest_m={f['nearest_incident_m']} "
              f"total_delay_s={f['incident_total_delay_s']} "
              f"(known_delay_n={f['incident_known_delay_count']})")
