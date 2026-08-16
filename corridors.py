"""
Corridor definitions — SINGLE SOURCE OF TRUTH.
==============================================
Every component (collector, trainer, API, frontend) imports corridors from here.
Do not redefine corridors anywhere else.

Coordinates were resolved via OpenStreetMap Nominatim geocoding and then each
corridor was validated end-to-end against the TomTom Routing API: the returned
road length must fall inside `expect_km`. All 8 passed on 2026-08-16.

History: the previous definitions (in model/traffic_model.py and backend/app.py,
which disagreed with each other) had MG Road and Mehrauli-Gurgaon Road pointing
at byte-identical coordinates, and NH-48 / Dwarka Expressway / SPR sharing a
single endpoint within 17 m. Both bugs are fixed here.
"""

CORRIDORS = [
    {
        "id": 0,
        "name": "NH-48 Delhi-Gurgaon Expressway",
        "sub": "Rajiv Chowk -> Manesar",
        "road_class": "highway",
        "start": (28.44747, 77.03284),   # Rajiv Chowk
        "end":   (28.32471, 76.92638),   # Manesar
        "expect_km": (15, 30),
        "verified_km": 21.90,
    },
    {
        "id": 1,
        "name": "MG Road",
        "sub": "IFFCO Chowk -> Sikandarpur",
        "road_class": "arterial",
        "start": (28.47233, 77.07242),   # IFFCO Chowk
        "end":   (28.48170, 77.09470),   # Sikandarpur
        "expect_km": (2, 6),
        "verified_km": 3.85,
    },
    {
        "id": 2,
        "name": "Golf Course Road",
        "sub": "Sikandarpur -> Sector 56",
        "road_class": "arterial",
        "start": (28.48170, 77.09470),   # Sikandarpur
        "end":   (28.42532, 77.09852),   # Sector 56
        "expect_km": (5, 12),
        "verified_km": 7.13,
    },
    {
        "id": 3,
        "name": "Sohna Road",
        "sub": "Rajiv Chowk -> Badshahpur",
        "road_class": "arterial",
        "start": (28.44747, 77.03284),   # Rajiv Chowk
        "end":   (28.39328, 77.04842),   # Badshahpur
        "expect_km": (6, 14),
        "verified_km": 7.16,
    },
    {
        "id": 4,
        "name": "Dwarka Expressway",
        "sub": "Dwarka Sector 21 -> Kherki Daula",
        "road_class": "expressway",
        "start": (28.55192, 77.05856),   # Dwarka Sector 21
        "end":   (28.40850, 76.98200),   # Kherki Daula
        "expect_km": (15, 30),
        "verified_km": 24.30,
    },
    {
        "id": 5,
        "name": "Golf Course Extension Road",
        "sub": "Sector 56 -> Vatika Chowk",
        "road_class": "arterial",
        "start": (28.42532, 77.09852),   # Sector 56
        "end":   (28.39060, 77.04090),   # Vatika Chowk
        "expect_km": (6, 14),
        "verified_km": 10.02,
    },
    {
        "id": 6,
        "name": "Mehrauli-Gurgaon Road",
        "sub": "Ghitorni -> IFFCO Chowk",
        "road_class": "arterial",
        "start": (28.49358, 77.14929),   # Ghitorni
        "end":   (28.47233, 77.07242),   # IFFCO Chowk
        "expect_km": (6, 14),
        "verified_km": 9.41,
    },
    {
        "id": 7,
        "name": "Southern Peripheral Road",
        "sub": "Vatika Chowk -> Kherki Daula",
        "road_class": "arterial",
        "start": (28.39060, 77.04090),   # Vatika Chowk
        "end":   (28.40850, 76.98200),   # Kherki Daula
        "expect_km": (5, 14),
        "verified_km": 8.02,
    },
]

ROAD_CLASSES = ["arterial", "expressway", "highway"]   # sorted; index == encoding
ROAD_CLASS_ENC = {rc: i for i, rc in enumerate(ROAD_CLASSES)}


def by_id(corridor_id: int) -> dict:
    for c in CORRIDORS:
        if c["id"] == corridor_id:
            return c
    raise KeyError(f"no corridor with id {corridor_id}")


def route_pair(c: dict) -> str:
    """TomTom routing path segment: 'lat1,lon1:lat2,lon2'."""
    return f"{c['start'][0]},{c['start'][1]}:{c['end'][0]},{c['end'][1]}"
