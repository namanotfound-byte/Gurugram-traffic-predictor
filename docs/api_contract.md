# API Contract v2 — FROZEN 2026-08-16

This contract is frozen so the API and frontend can be built in parallel.
**Do not change a field name or type without updating this file first.**

Base URL: `http://localhost:5000` (dev)

## Conventions

- `day` — integer 0–6, Monday=0 … Sunday=6 (matches Python `weekday()`).
- `hour` — integer 0–23, local Gurugram time (IST, UTC+05:30).
- `congestion_index` — float 0.0–1.0, rounded to 3 dp.
  `1 - (free_flow_travel_time / expected_travel_time)`.
  0.0 = free flowing, 1.0 = standstill.
- `label` — one of `"Free"`, `"Moderate"`, `"Heavy"`, `"Severe"`.
  Thresholds: `<0.35` Free, `<0.60` Moderate, `<0.80` Heavy, else Severe.
- `provenance` — one of:
  - `"observed"`  — trained on live data we measured ourselves
  - `"bootstrap"` — trained on TomTom historical-model data
  - `"synthetic"` — trained on generated data (NOT real; must be surfaced in UI)
- `confidence` — float 0.0–1.0. Derived from how much real data backs the
  prediction for that corridor/day/hour cell. Frontend must visibly degrade
  the display when `< 0.5`.

Every response includes `provenance` and `model_version`. The frontend is
required to display provenance to the user — never present synthetic numbers
as if they were measured.

## Endpoints

### `GET /health`
```json
{ "status": "ok", "model_version": "gbt-2026-08-16", "provenance": "bootstrap",
  "corridors": 8, "trained_rows": 1344 }
```

### `GET /corridors`
Static list. Frontend uses this to build its map/selector — it must NOT hardcode corridors.
```json
{ "corridors": [
  { "id": 0, "name": "NH-48 Delhi-Gurgaon Expressway", "sub": "Rajiv Chowk -> Manesar",
    "road_class": "highway", "start": [28.44747, 77.03284], "end": [28.32471, 76.92638],
    "length_km": 21.9 }
]}
```

### `GET /predict?corridor=<id>&day=<0-6>&hour=<0-23>`
Single cell.
```json
{ "corridor_id": 0, "day": 1, "hour": 8,
  "congestion_index": 0.185, "label": "Free",
  "delay_minutes": 3.7, "typical_minutes": 19.8, "free_flow_minutes": 16.1,
  "provenance": "bootstrap", "confidence": 0.9, "model_version": "gbt-2026-08-16" }
```

### `GET /advice?corridor=<id>&day=<0-6>`
**The primary endpoint.** Answers "when should I go?"
```json
{ "corridor_id": 0, "day": 1,
  "profile": [0.0, 0.0, ...],                    // 24 floats, hour 0..23
  "best_windows":  [ {"start_hour": 22, "end_hour": 5, "avg_index": 0.01,
                      "label": "Free", "text": "Clear after 10 PM"} ],
  "worst_windows": [ {"start_hour": 17, "end_hour": 20, "avg_index": 0.34,
                      "label": "Moderate", "text": "Avoid 5-8 PM"} ],
  "best_hour": 2, "peak_hour": 18,
  "summary": "Leave before 7 AM or after 9 PM. Worst is 6 PM (+11 min).",
  "provenance": "bootstrap", "confidence": 0.9 }
```
Windows are contiguous hour runs. `end_hour` is inclusive and may wrap past
midnight (`start_hour > end_hour` means the window wraps).

### `GET /best-time?corridor=<id>&day=<0-6>&earliest=<0-23>&latest=<0-23>`
Best departure inside the user's own constraint.
```json
{ "corridor_id": 0, "day": 1, "earliest": 8, "latest": 12,
  "recommended_hour": 12, "congestion_index": 0.187, "label": "Free",
  "delay_minutes": 3.9,
  "saving_vs_worst_minutes": 5.2,
  "alternatives": [ {"hour": 11, "congestion_index": 0.19, "delay_minutes": 4.0} ],
  "summary": "Of 8 AM-12 PM, leave at 12 PM. Saves ~5 min vs leaving at 10 AM.",
  "provenance": "bootstrap", "confidence": 0.9 }
```
If `latest < earliest` the window wraps past midnight.

### `GET /now`
Live verdict for all corridors, using current IST time.
```json
{ "now_ist": "2026-08-16T21:45:00+05:30", "day": 6, "hour": 21,
  "corridors": [ { "id": 0, "name": "NH-48 ...", "congestion_index": 0.02,
                   "label": "Free", "delay_minutes": 0.4, "trend": "falling",
                   "verdict": "go_now",
                   "text": "Clear. Good time to travel." } ],
  "summary": { "avg_congestion": 0.03, "worst_corridor": "Sohna Road",
               "clear_count": 7 },
  "provenance": "bootstrap" }
```
`trend` ∈ `"rising" | "falling" | "flat"` (compare next hour vs current).
`verdict` ∈ `"go_now" | "wait" | "avoid"`.

## Errors

All errors: HTTP 400 with `{"error": "<human readable message>"}`.
Validate: corridor id in range, day 0–6, hour 0–23. Never 500 on bad input.

## Non-negotiables

1. **No hand-invented numbers.** The old `PROFILES` lookup table in
   `backend/app.py` is deleted. If no model is loaded, endpoints return
   HTTP 503 `{"error": "no model trained yet"}` — they must never silently
   serve made-up values while implying a model produced them.
2. **Precompute.** Build the full 8×7×24 grid once at startup, serve from
   memory. Do not call `model.predict()` per request.
3. **Road class encoding** comes from `corridors.ROAD_CLASS_ENC` on both the
   training and serving side. Never hardcode it.
