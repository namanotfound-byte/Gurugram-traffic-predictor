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

Corridors 8-12 (2026-08-17): added to fix a structural road_class imbalance.
The original 8 had 6 arterial / 1 expressway / 1 highway, so leave-one-
corridor-out CV had zero same-class siblings for Dwarka Expressway or NH-48
— holding either out left the model with no training example of that class
at all, and it scored R2=-25.5 (Dwarka) / -0.08 (NH-48) as a result, dragging
mean CV R2 to -2.52. These 5 corridors were chosen specifically to give both
under-represented classes same-class siblings (expressway: 1 -> 3, highway:
1 -> 4), not to expand coverage for its own sake. Coordinates resolved via
TomTom Geocoding (search/2/geocode), routed at an off-peak hour (03:00 IST)
against TomTom Routing exactly like corridors 0-7, and checked pairwise
against all existing + new corridor endpoints for duplicate/near-duplicate
geometry (see tools/_validate_new_corridors.py) — one intentional shared
endpoint was found (KMP Expressway and Pataudi Road both terminate at
Pataudi Chowk, a real physical junction where both roads meet — same pattern
already present among corridors 0-7, e.g. Rajiv Chowk shared by corridors 0
and 3), no unintentional duplicates.
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
    {
        "id": 8,
        "name": "KMP Expressway (Western Peripheral Expressway)",
        "sub": "Sidhrawali -> Pataudi Chowk",
        "road_class": "expressway",
        # Justification: the ONLY access-controlled peripheral expressway on
        # Gurugram's west side, structurally and behaviorally distinct from
        # Dwarka Expressway (real-estate-corridor, still-under-construction
        # traffic) — gives the expressway class a second, genuinely
        # different-pattern sibling rather than a near-copy.
        "start": (28.26347, 76.83235),   # Sidhrawali (Manesar)
        "end":   (28.32837, 76.77790),   # Pataudi Chowk
        "expect_km": (10, 18),
        "verified_km": 13.59,
    },
    {
        "id": 9,
        "name": "Delhi-Mumbai Expressway",
        "sub": "Sohna Interchange -> Nuh",
        "road_class": "expressway",
        # Justification: newly operational 8-lane greenfield expressway
        # (NE-4), the Haryana-Nuh stretch south of Sohna. Third expressway
        # sample and the longest/highest-speed one, giving that class real
        # variance in geometry and traffic character instead of relying on
        # a single corridor to represent the whole class.
        "start": (28.25651, 77.10909),   # Sohna DME interchange
        "end":   (28.10515, 77.00848),   # Nuh
        "expect_km": (25, 40),
        "verified_km": 33.59,
    },
    {
        "id": 10,
        "name": "NH-352W (Gurugram-Sohna-Alwar Road)",
        "sub": "Sohna -> Taoru",
        "road_class": "highway",
        # Justification: a distinct national highway number from NH-48, a
        # 2-lane semi-rural highway carrying very different traffic than
        # NH-48's 8-lane expressway-grade corridor. Gives the highway class
        # its second sibling so leave-one-corridor-out no longer has to
        # extrapolate NH-48's pattern onto a road with no same-class
        # training example.
        "start": (28.24915, 77.06533),   # Old Alwar Road, Sohna
        "end":   (28.20990, 76.95030),   # Taoru
        "expect_km": (10, 20),
        "verified_km": 14.60,
    },
    {
        "id": 11,
        "name": "Old Delhi-Gurgaon Road",
        "sub": "Kapashera Border -> Hero Honda Chowk",
        "road_class": "highway",
        # Justification: the old NH-8 service road/alignment through the
        # city, physically distinct from the new NH-48 expressway bypass
        # (corridor 0) even though they run roughly parallel — dense urban
        # signal-controlled arterial-like traffic rather than free-flow
        # expressway traffic. Third highway sibling.
        "start": (28.52196, 77.08888),   # Kapashera Border (Delhi/Haryana)
        "end":   (28.46033, 77.04983),   # Hero Honda Chowk
        "expect_km": (10, 20),
        "verified_km": 15.26,
    },
    {
        "id": 12,
        "name": "Pataudi Road",
        "sub": "Basai Chowk -> Pataudi Chowk",
        "road_class": "highway",
        # Justification: state highway connecting Gurugram city to Pataudi
        # tehsil, serving industrial/rural commuter traffic unlike any other
        # corridor in the set. Fourth highway sibling, and the longest
        # highway sample, adding range to that class's training data.
        "start": (28.45514, 76.98790),   # Basai Chowk
        "end":   (28.32837, 76.77790),   # Pataudi Chowk (also KMP's end —
                                          # real junction where both roads meet)
        "expect_km": (22, 35),
        "verified_km": 30.04,
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
