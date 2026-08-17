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

  **Thresholds (RECALIBRATED 2026-08-16 — the earlier values are dead):**

  | label | congestion_index | means |
  |---|---|---|
  | `Free`     | `< 0.091` | under 1.10x free-flow travel time |
  | `Moderate` | `< 0.200` | 1.10x - 1.25x |
  | `Heavy`    | `< 0.310` | 1.25x - 1.45x |
  | `Severe`   | `>= 0.310` | over 1.45x |

  The old thresholds (`0.35/0.60/0.80`) were inherited from the synthetic data
  generator, whose values ranged up to 0.92. **Real TomTom data for Gurugram
  peaks at 0.435**, so under the old scale 98.5% of all 1344 measured cells
  labelled `Free` and `Heavy`/`Severe` were unreachable — the worst cell in the
  city (MG Road, Friday 19:00, a 7-minute trip taking 12) would have displayed
  as `Free`.

  The new boundaries are anchored on round travel-time multipliers rather than
  percentiles of one week's data, so they stay meaningful as more data arrives.
  Observed distribution across the bootstrap set: Free 52.5%, Moderate 25.4%,
  Heavy 18.2%, Severe 3.9%.

  **Both the API and the frontend must use these exact numbers.** The frontend
  derives per-hour labels client-side from the `profile` array, so any drift
  between the two shows up as a visible contradiction on screen.
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

### Label honesty (added 2026-08-17)

A served `label`/`congestion_index` is a **typical** value for that
(corridor, day-of-week, hour) cell — it is not a live sensor reading and not
a forecast for one specific date. Measured against 115 real observations
(`docs/accuracy_report.md`, 3.5% cell coverage): exact label agreement is
**58.3%**, while hour-vs-hour ranking (pairwise concordance) is **89.4%**.
The gap is not a uniform bias (per-band bias is +0.032/-0.048/-0.052/+0.037
near-free/light/moderate/peak) — it is variance at thin coverage, so
thresholds are **not** shifted and **no** bias correction is applied; both
would fit noise. Instead:
- `GET /health` (and the bundle's top-level `accuracy` key) serve
  `ACCURACY_SUMMARY`: `label_agreement_pct`, `hour_ranking_concordance_pct`,
  `sample_size`, `as_of`, `note` — so this strength/weakness split is
  discoverable, not just documented.
- Natural-language `text`/`summary` fields (`/now`, `/advice`,
  `/advice/all`) say "Typically <label>…", never state the label as an
  unqualified fact.
- Any summary/text built from a cell with `confidence < 0.5` appends
  `" (Limited data for this corridor/day — treat as a rough guide.)"` (or
  the `/now`-specific `" Limited data for this hour — treat as a rough
  guide."`) inline in the string, so a low-confidence cell degrades
  visibly even before a frontend wires up `confidence` itself.

### Best-time / saving fields (added 2026-08-17)

Gurugram corridors are short (the longest realistic corridor trip is under
25 minutes), so a raw minutes-saved figure can look unconvincingly small —
even though it is correct. Two changes address this without inventing any
number:
1. **Whole-day framing.** `/advice`, `/advice/all`, and the bundle's
   `advice` objects now include `best_hour_delay_minutes`,
   `peak_delay_minutes`, `peak_delay_pct` (delay at the peak hour as a % of
   free-flow time — "how much longer at the worst hour"), and
   `whole_day_saving_minutes` / `whole_day_saving_pct` (best-vs-worst saving
   across the full day, minutes and % of the peak hour's trip time). The
   `summary` string leads with this whole-day figure.
2. **Window-constrained honesty.** `/best-time` now also returns
   `saving_vs_worst_pct`, `whole_day_best_hour`, `whole_day_worst_hour`,
   `whole_day_saving_minutes`, `whole_day_saving_pct`, and a boolean
   `window_constrained` (true when a strictly better hour exists outside
   the caller's `earliest`–`latest` window). When `window_constrained` is
   true, `summary` leads with the corridor-wide best-vs-worst saving, then
   explicitly says the window-limited number is "within your window" and
   that a bigger saving exists outside it — instead of implying the small
   window number is the best the corridor can ever do.

Percentage is expressed both ways deliberately: `peak_delay_pct` answers
"how much longer is the worst hour than free-flow" (e.g. a 5-minute delay
on a 7-minute drive is ~76% longer — often the more compelling framing for
short corridors), while `whole_day_saving_pct` / `saving_vs_worst_pct`
answer "how much shorter is my trip if I time it right." Minutes are always
still present alongside the percentage — neither replaces the other.

### Day/night split (added 2026-08-17)

The whole-day `best_hour` is midnight for essentially every corridor —
roads are simply empty overnight — so the "best time" figure looked
identical and useless across every corridor for a daytime traveller (real
user complaint: "it always shows that night is free ... going in at night
is not viable"). The boundary is derived from the measured grid, not
asserted: averaging `congestion_index` across all 13 corridors × 7 days per
hour shows a near-zero, flat floor (avg 0.0005–0.0009, max ≤0.009 — every
corridor reads "Free") from 22:00 through 03:00, then a sharp climb
starting 06:00. The evening side mirrors it: 21:00 still averages 0.087
(max 0.15) but 22:00 collapses to 0.0006 (max 0.009) — a >100x drop in one
hour. So:

```python
DAY_HOURS = list(range(6, 22))                        # 06:00-21:59
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))  # 22:00-05:59
```

Night advice is still served (truck/shift-worker use case) — just as its
own explicit period, never silently winning every "best time" comparison.

**`/advice` and `/advice/all`** (and the bundle's `advice` objects) now
additionally return `day_period` and `night_period`, each shaped:
```json
{ "period": "day", "start_hour": 6, "end_hour": 21,
  "best_hour": 6, "worst_hour": 18,
  "best_hour_delay_minutes": 0.8, "worst_hour_delay_minutes": 12.8,
  "worst_hour_delay_pct": 38.6,
  "saving_minutes": 12.0, "saving_pct": 26.1,
  "best_windows": [...], "worst_windows": [...],
  "summary": "Best daytime departure: 6 AM. Avoid 6 PM (+13 min, 39% longer).",
  "confidence": 0.92 }
```
All existing whole-day fields (`profile`, `best_windows`, `best_hour`,
`peak_hour`, `summary`, `confidence`, etc.) are unchanged — this is purely
additive.

**`/best-time`** now accepts an optional `period` query param —
`"day"` | `"night"` | `"any"` — as an alternative to `earliest`/`latest`.
When given, `earliest`/`latest` are derived from the period (`6`/`21` for
day, `22`/`5` for night, `0`/`23` for any) and echoed in the response as
before; a new top-level `period` field reports which mode was used
(`"day"`, `"night"`, `"any"`, or `"custom"` when explicit `earliest`/
`latest` were passed instead — the pre-existing, still-fully-supported
behavior). `earliest`/`latest` remain **required together** when `period`
is omitted, exactly as before.

Crucially, when `period` is `"day"` or `"night"`, `window_constrained` is
always `false` and the summary is period-worded (e.g. `"Best daytime
departure: 6 AM. Avoid 6 PM (+13 min, 39% longer)."`) — the search never
falls back to comparing against the unconstrained 24-hour scan, because
that comparison is exactly what made night silently "win" every time
before this feature existed. The `whole_day_best_hour` /
`whole_day_worst_hour` / `whole_day_saving_minutes` / `whole_day_saving_pct`
fields are still populated for reference in every response regardless of
`period`.

## Endpoints

### `GET /health`
```json
{ "status": "ok", "model_version": "gbt-2026-08-16", "provenance": "bootstrap",
  "corridors": 8, "trained_rows": 1344,
  "accuracy": { "label_agreement_pct": 58.3, "hour_ranking_concordance_pct": 89.4,
                "sample_size": 115, "as_of": "2026-08-17",
                "note": "Measured against 115 real observations: this site is much better at RANKING which hour is better... (see docs/accuracy_report.md)" } }
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
  "best_hour_delay_minutes": 0.0, "peak_delay_minutes": 5.2, "peak_delay_pct": 76.5,
  "whole_day_saving_minutes": 5.2, "whole_day_saving_pct": 43.3,
  "summary": "Leave before 7 AM or after 9 PM. Worst is 6 PM (+5 min, 76% longer than free-flow). Timing it right saves ~5 min (43% shorter trip) versus the worst hour.",
  "provenance": "bootstrap", "confidence": 0.9 }
```
Windows are contiguous hour runs. `end_hour` is inclusive and may wrap past
midnight (`start_hour > end_hour` means the window wraps). `best_hour_delay_minutes`,
`peak_delay_minutes`, `peak_delay_pct`, `whole_day_saving_minutes`, and
`whole_day_saving_pct` were added 2026-08-17 — see "Best-time / saving fields" above.

### `GET /advice/all?day=<0-6>`
**Added 2026-08-16** after the frontend build showed the map + sidebar for a
chosen day otherwise needs 8 parallel `/advice` calls. Additive — `/advice`
is unchanged.

Returns the same object as `/advice` for every corridor in one response:
```json
{ "day": 1,
  "corridors": [ { "corridor_id": 0, "profile": [...], "best_windows": [...],
                   "worst_windows": [...], "best_hour": 2, "peak_hour": 18,
                   "summary": "...", "confidence": 0.9 } ],
  "provenance": "bootstrap", "model_version": "gbt-2026-08-16" }
```
Cheap to serve — the 8×7×24 grid is already precomputed in memory.

### `GET /best-time?corridor=<id>&day=<0-6>&earliest=<0-23>&latest=<0-23>`
Best departure inside the user's own constraint.
```json
{ "corridor_id": 0, "day": 1, "earliest": 8, "latest": 12,
  "recommended_hour": 12, "congestion_index": 0.187, "label": "Free",
  "delay_minutes": 3.9,
  "saving_vs_worst_minutes": 5.2, "saving_vs_worst_pct": 20.1,
  "window_constrained": false,
  "whole_day_best_hour": 12, "whole_day_worst_hour": 18,
  "whole_day_saving_minutes": 5.2, "whole_day_saving_pct": 20.1,
  "alternatives": [ {"hour": 11, "congestion_index": 0.19, "delay_minutes": 4.0} ],
  "summary": "Of 8 AM-12 PM, leave at 12 PM. Saves ~5 min (20%) vs leaving at 10 AM.",
  "provenance": "bootstrap", "confidence": 0.9 }
```
If `latest < earliest` the window wraps past midnight.

Added 2026-08-17: `saving_vs_worst_pct`, `window_constrained`,
`whole_day_best_hour`, `whole_day_worst_hour`, `whole_day_saving_minutes`,
`whole_day_saving_pct` — see "Best-time / saving fields" above. When
`window_constrained` is `true` (a strictly better hour exists outside
`earliest`–`latest`), `summary` leads with the whole-day saving instead of
the window-limited one, e.g.:
```
"Best overall on this day: avoid 7 PM, leave around 12 AM instead — saves
~5 min (43% shorter trip) across the full day. Within your 7 AM-11 AM
window, leave at 7 AM: saves ~2 min (21%) vs 11 AM within that window, but
a bigger saving is available outside it."
```
(Real example: MG Road, Friday, `/best-time?corridor=1&day=4&earliest=7&latest=11`.)

### `GET /now`
Live verdict for all corridors, using current IST time.
```json
{ "now_ist": "2026-08-16T21:45:00+05:30", "day": 6, "hour": 21,
  "corridors": [ { "id": 0, "name": "NH-48 ...", "congestion_index": 0.02,
                   "label": "Free", "delay_minutes": 0.4, "trend": "falling",
                   "verdict": "go_now",
                   "text": "Typically clear now. Good time to travel." } ],
  "summary": { "avg_congestion": 0.03, "worst_corridor": "Sohna Road",
               "clear_count": 7 },
  "provenance": "bootstrap" }
```
`trend` ∈ `"rising" | "falling" | "flat"` (compare next hour vs current).
`verdict` ∈ `"go_now" | "wait" | "avoid"`.

`text` wording changed 2026-08-17 to lead with "Typically …" (see "Label
honesty" above) — the served value is a typical value for this hour, not a
live sensor reading, so the copy must not claim it as an unqualified
present-tense fact. When the served cell's `confidence < 0.5`, `text` also
appends `" Limited data for this hour — treat as a rough guide."`.

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
