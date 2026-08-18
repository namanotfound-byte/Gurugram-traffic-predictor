# Project Explainer — Gurugram Traffic Congestion Predictor

This file exists so you can explain **every part of this project** in a CS
viva/exam without having to re-derive it from scratch, and so it reads well
as a CV artifact. It walks through every file, explains every import (what
library, why it's used), and goes line-by-line through the parts of the
code that actually matter — feature engineering, model training, the Flask
API, the frontend — and then spends real space on the *history*: the bugs
that were found, the metrics that were wrong, and why the current design
looks the way it does. That history is the most defensible material in a
viva and it does not exist anywhere else in the repo.

**A note on numbers.** Every figure in this document was re-derived
directly from the files in this repo while writing it (either read out of
source/data files, or recomputed with a short script against the live
model and CSVs) — not copied from an earlier draft or from memory. Where a
number in `README.md` or `docs/api_contract.md` disagreed with what the
code and data actually show, that is called out explicitly rather than
silently "corrected" — those files describe an earlier state of the
project (8 corridors, not the current 13) and have not been updated since.
If you are asked "how do you know your numbers are real," the honest
answer is: *recompute them from source every time, don't trust a cached
document* — which is exactly what this file's own history is about.

Keep this file updated as the code changes — ask Claude to re-sync it
whenever you add/modify something.

---

## 1. The one-sentence pitch

> Predict how congested a road in Gurugram will be, for any day of the
> week and hour of the day, using a machine learning model trained on
> real traffic data pulled from TomTom's API — serve *measured* values
> wherever they exist and only fall back to the model where they don't —
> then layer weather, calendar and live-incident signals on top to try to
> explain the day-to-day deviation from that baseline, and show all of it
> on a live web dashboard.

The second half of that sentence (weather/incidents/deviation) is newer
than the first half and is currently **gated off** — see §8. Both halves
are worth understanding, and the gating itself is a design decision worth
defending, not a gap to hide.

---

## 2. The big picture: how data flows through the system

```
TomTom Routing API            TomTom Incidents API      Open-Meteo (weather)
(real traffic, live)          (real, live, unauthenticated (no key))
        │                          │                          │
        │ bootstrap_collect.py     │ incidents.py             │ weather.py
        │ (one-time historical     │ (matched to corridor     │ (hourly weather +
        │  sweep, 13x7x24 cells)   │  geometry, per round)     │  Indian holiday
        │                          │                          │  calendar)
        │ collect_live.py          │                          │
        │ (ongoing live sampling,  │◄─────────────────────────┘
        │  ~every 15 min via CI)   │
        ▼                          ▼
data/gurugram_bootstrap.csv   data/gurugram_observed.csv (weather + calendar +
(2,184 rows = 13×7×24,         incident columns stamped onto every row)
 TomTom's OWN historical model)
        │                          │
        │ model/traffic_model.py   │ model/forecast_model.py
        │ train_model()            │ train() — residual on top of the
        ▼                          │ bootstrap baseline (GATED — see §8)
models/traffic_gbt.joblib          ▼
        │                    models/forecast_residual_gbt.joblib
        │                    (does not exist yet — data not sufficient)
        │
        │ backend/app.py (loads model, precomputes full grid, serves
        │                  MEASURED values first, model only fills gaps)
        ▼
Flask REST API (localhost:5000) → /predict /advice /advice/all /best-time /now
        │
        │ tools/build_static_bundle.py (imports app.py, snapshots GRID)
        ▼
frontend/data/bundle.json (~98 KB, everything the API can answer, frozen)
        │
        │ fetch() once, at page load — no server needed
        ▼
frontend/index.html (MapLibre map + dashboard, served on GitHub Pages)
```

Everything downstream of the two CSV files is *derived* — the model file,
the API responses, the static bundle and the frontend numbers all
ultimately trace back to real numbers TomTom's (or Open-Meteo's) servers
returned. There is no made-up data anywhere in the production path — that
was a real, documented bug earlier in the project's history (§9).

Two things changed since an earlier version of this document that are
worth stating up front, because they reshape almost every section below:

1. **8 corridors → 13 corridors** (2026-08-17), to fix a structural
   cross-validation problem (§7.3).
2. **The API stopped trusting the model as its primary source of truth.**
   `backend/app.py` now serves the *measured* CSV value for any cell that
   has one, and only calls the trained model for cells with no
   measurement at all. Right now that means the model is inferring for
   **zero** of the 2,184 cells — see §10.1 for the number that motivated
   this and why it matters more than it sounds.

---

## 3. `corridors.py` — the single source of truth for "which roads are we tracking"

**What it is:** a plain Python list of 13 dictionaries, plus two tiny
helper functions and one lookup table. No ML, no I/O.

```python
CORRIDORS = [
    {
        "id": 0,
        "name": "NH-48 Delhi-Gurgaon Expressway",
        "sub": "Rajiv Chowk -> Manesar",
        "road_class": "highway",
        "start": (28.44747, 77.03284),
        "end":   (28.32471, 76.92638),
        "expect_km": (15, 30),
        "verified_km": 21.90,
    },
    ...  # 13 corridors total, ids 0-12
]
```

- **Why this file exists at all:** every other file (both collectors, both
  trainers, the API, the incident matcher, the geojson builder) needs to
  know "what are the roads we track and where are they." Defined **once**
  here and imported everywhere else, instead of four copies that can
  quietly disagree — which is literally what happened before this file
  existed (§9.2).
- `road_class` — `"arterial"`, `"expressway"`, or `"highway"`. Matters
  because a highway absorbs more cars before jamming than a small
  arterial road, so the model needs to know which type of road it's
  predicting for.
- `start` / `end` — GPS coordinates sent straight to TomTom's routing API.
- `expect_km` / `verified_km` — every corridor's real-world length
  (returned by TomTom) was checked against a plausible range once, and the
  actual measured length recorded, to catch coordinate mistakes.

### 3.1 Why 13 corridors, not 8

The module docstring (read it — it's short and it's the primary source
for this) explains the original 8 corridors were 6 arterial / 1
expressway / 1 highway. That imbalance is what broke leave-one-corridor-out
cross-validation (full story in §7.3) — holding out the *only* expressway
or the *only* highway corridor left the model with zero same-class
training examples for that fold, which is a much harder problem than
"generalizing to a new road" and scored accordingly badly (documented in
the code as R²=-25.5 for Dwarka, -0.08 for NH-48, dragging the mean to
-2.52).

Five corridors were added specifically to fix this — not to expand
coverage for its own sake:

| new corridor | road_class | why this one |
|---|---|---|
| KMP Expressway | expressway | only other access-controlled peripheral expressway in western Gurugram; structurally distinct from Dwarka (real-estate-corridor, still-under-construction traffic) |
| Delhi–Mumbai Expressway (NE-4) | expressway | newly operational 8-lane greenfield expressway, longest/highest-speed sample |
| NH-352W (Gurugram-Sohna-Alwar Rd) | highway | a 2-lane semi-rural highway, very different character from NH-48's 8-lane corridor |
| Old Delhi-Gurgaon Road | highway | the old NH-8 alignment, dense urban signal-controlled traffic, physically distinct from the NH-48 bypass even though they run roughly parallel |
| Pataudi Road | highway | state highway serving industrial/rural traffic, longest highway sample |

Result: arterial 6 / expressway 3 / highway 4. Every road class now has at
least one same-class sibling for GroupKFold to train on when another
member of its class is held out.

Coordinates for the 5 new corridors were resolved via TomTom's Geocoding
API and routed at 03:00 IST (off-peak, to avoid picking up a
congestion-driven detour), checked against `expect_km`, and checked
pairwise against every existing endpoint for duplicate/near-duplicate
geometry. One shared endpoint was found on purpose and kept: KMP
Expressway and Pataudi Road both terminate at Pataudi Chowk, a real
physical road junction — the same pattern already existed among the
original 8 (Rajiv Chowk is shared by corridors 0 and 3).

```python
ROAD_CLASSES = ["arterial", "expressway", "highway"]   # sorted; index == encoding
ROAD_CLASS_ENC = {rc: i for i, rc in enumerate(ROAD_CLASSES)}
```
Builds `{"arterial": 0, "expressway": 1, "highway": 2}`. Because this
dictionary lives here and only here, training and prediction are
guaranteed to use the *same* mapping — an earlier version of this project
had two different, disagreeing mappings (§9.3).

```python
def by_id(corridor_id: int) -> dict: ...
def route_pair(c: dict) -> str:
    return f"{c['start'][0]},{c['start'][1]}:{c['end'][0]},{c['end'][1]}"
```
`by_id` is a lookup by id. `route_pair` formats a corridor's coordinates
into the exact string TomTom's routing URL expects, so that formatting
logic isn't duplicated in every script that talks to TomTom.

---

## 4. Data collection

### 4.1 `bootstrap_collect.py` — the one-time historical sweep

**Purpose:** fill a full training grid in one sitting instead of waiting
weeks for live sampling to accumulate. It queries TomTom for **every**
combination of `13 corridors × 7 days × 24 hours = 2,184 cells`.

**Key trick — how it gets "historical" data from a Routing API:**
TomTom's dedicated Flow API returns 403 on this project's key (§9.1). But
the Routing API accepts a future `departAt` and, with
`computeTravelTimeFor=all`, returns TomTom's own historical traffic
model's estimate for that time — even though this key can't query Flow
directly. That's the entire reason this script can produce real,
non-invented data.

```python
congestion_idx = round(1 - (free_flow_s / historic_s), 4)
```
`free_flow_s` = trip time with zero traffic; `historic_s` = TomTom's
historical-model time for that specific day/hour. Ratio near 1 → `idx`
near 0 (free flowing). Historic time double the free-flow time → `idx =
0.5` (effectively half-speed). Verified range in the actual collected
data: **0.0 to 0.4349** (see §10 for why that number matters so much).

**Route-stability check** — TomTom's routing engine sometimes picks a
*physically different road* at different times of day (a side-street
reroute at 3 AM, say). `is_route_stable()` flags a row `route_stable=False`
when the returned route length deviates from the corridor's
`verified_km` reference by more than 2%, so the trainer can exclude rows
that measured the wrong road. Running the check against the full,
completed sweep: **146 of 2,184 rows (6.7%)** are flagged unstable,
concentrated on 5 corridors (Golf Course Extension Road, Mehrauli-Gurgaon
Road, Southern Peripheral Road, KMP Expressway, Old Delhi-Gurgaon Road).
Those rows are excluded from training by default.

**Resumability:** before starting, the script reads whatever's already in
the output CSV and skips any `(corridor, day, hour)` cell already
collected — a crash or a deliberate `--max-requests` stop just picks back
up where it left off on the next run, instead of re-spending API quota.

**Retry logic:** HTTP 429 (rate-limited) or 5xx is retried with
exponential backoff (1s, 2s, 4s…) rather than given up on immediately or
hammered instantly — standard practice for a flaky/rate-limited remote
API.

At the end, `check_route_consistency()` prints per-corridor min/max/median
route length across the whole sweep — good material if asked "how did
you validate data quality?"

### 4.2 `collect_live.py` — the ongoing live collector

**Purpose:** unlike the bootstrap sweep ("what does history say?"), this
asks "what is traffic doing **right now**?" — one live snapshot per
corridor, run repeatedly to build a real-time stream over calendar time.

```python
congestion_idx = round(1 - (free_flow_s / live_s), 4)
```
Divides by `live_s` (current-conditions time TomTom just measured), not
`historic_s`. That's a subtly different quantity from the bootstrap
sweep's number — see §8's "free-flow consistency" discussion for why this
matters and how it's guarded against. Every row is tagged
`source="observed"`.

**Every row also carries weather, calendar, and incident features**,
fetched once per round (city-wide / bbox-wide, not per corridor) and
stamped onto all 13 corridor rows for that round:

```
temperature_c, precipitation_mm, is_raining, rain_intensity, rain_last_3h,
visibility_m, low_visibility,
is_holiday, holiday_name, is_festival_period, is_month_end, days_to_nearest_holiday,
incident_count, incident_total_delay_s, incident_known_delay_count,
incident_max_magnitude, has_road_closure, has_jam, nearest_incident_m
```
This is what makes `model/forecast_model.py` (§8) possible at all — none
of it existed when the bootstrap grid was built, and none of it can be
retrofitted onto the bootstrap grid (see §5 and §6 for why).

**Cadence, and a real CI scheduling bug found and fixed.** The intended
cadence is one round every 15 minutes: 13 routing requests + 1 incidents
bbox request = 14 requests/round, × 96 rounds/day = 1,344 requests/day,
comfortably under TomTom's 2,500/day free-tier cap. The first
implementation scheduled the GitHub Actions workflow directly at
`*/15 * * * *`. In practice this did **not** deliver 96 rounds/day — GitHub
documents `schedule:` triggers as best-effort, and the observed real gap
between runs was 30–45 minutes, i.e. roughly 36 rounds/day (~37% of
nominal). Total row count wasn't badly hurt by this, but the actual reason
for 15-minute sampling — catching short-lived monsoon rain onset/offset,
which can start and finish inside a half-hour gap — was being lost in
direct proportion to the throttling.

The fix (documented in both `collect_live.py`'s module docstring and
`.github/workflows/collect.yml`) is not a tighter cron — that makes
GitHub's throttling worse, not better. Instead the workflow now fires
**hourly** (an interval GitHub honors reliably) and calls
`collect_live.py --loop --max-rounds 4 --max-minutes 50`: the script loops
*internally*, firing 4 rounds ~15 minutes apart inside one ~45-minute CI
job, with a wall-clock safety cap so a slow round can't run past the next
hourly trigger. This is a good example of "the platform's scheduler lies
to you, so move the timing logic into your own process" — a real,
debuggable production concern, not a toy one.

**Two run modes** remain: `--once` (single round, used by CI before this
fix, still available), `--loop [--max-rounds N] [--max-minutes M]` (loop
forever or for a bounded number of rounds).

**`--backfill-weather`** retroactively attaches weather/calendar columns to
rows collected before those columns existed. **Incidents cannot be
backfilled** the same way — there is no historical incident-replay
endpoint on this key, only live/current incidents were ever confirmed
reachable, so a closure that happened yesterday and has since cleared is
simply gone. Rows predating incident tracking are *not* assumed
incident-free; `model/forecast_model.py` carries an explicit
`incident_data_known` flag for exactly this reason (§8).

**`--budget-guard`** (optional) tracks requests spent today in a
git-ignored local JSON file and refuses to start a round that would blow
past `--daily-cap`. Deliberately **not** used by the CI workflow — its
state file doesn't survive between ephemeral GitHub-hosted runners, so it
would provide false confidence there; it's meant for a persistent VM
running `--loop`, where both the process and the state file survive
between rounds. Quota safety in CI instead comes from the request math
itself.

---

## 5. `weather.py` — Open-Meteo weather + Indian holiday calendar

**Why this file exists**, stated directly in its own docstring and worth
repeating because it's one of the most important design facts in the
project: querying TomTom's historical model for **six different future
Fridays at 18:00, including Diwali week**, returned **byte-identical
results** every time. The bootstrap grid is a pure day-of-week × hour
average with **zero** weather or date sensitivity — it can never teach a
model anything about rain, festivals, or salary-day traffic. This file is
the sole source of the two feature families (weather, calendar) that
might explain deviations the bootstrap grid structurally cannot see.

**Weather source: Open-Meteo** (free, no API key). Two endpoints, chosen
between per-request:
- **Forecast endpoint** (`api.open-meteo.com/v1/forecast`) — today,
  future, and — via `past_days` — up to 92 days into the past. Carries a
  real `visibility` field.
- **Archive endpoint** (`archive-api.open-meteo.com/v1/archive`) — any
  historical date, no 92-day limit, but **`visibility` comes back null on
  every value** (confirmed empirically against real Gurugram dates before
  writing this file, per the docstring — not assumed).

So the forecast endpoint is preferred whenever a date falls inside its
92-day trailing window (a strict upgrade: same precipitation/temperature
data, plus real visibility); the archive endpoint is only a fallback once
a date is older than that, and visibility is honestly left `None` for
those rows rather than invented. In practice, since this project's
observed collection only started 2026-08-16, every row this project will
ever backfill falls inside the 92-day window — the archive branch exists
for correctness on a much older/revived dataset, not because it is
expected to run.

**Derived features** (not raw values): `is_raining` (precip > 0.1mm),
`rain_intensity` (none/light/moderate/heavy, following standard
meteorological bands), and — the one flagged as hypothesis-worth-testing
in the docstring — `rain_last_3h`, a trailing 3-hour cumulative
precipitation sum. The stated reasoning: roads stay slick/slow *after*
rain stops, so this is hypothesized to be a stronger predictor of
congestion than the instantaneous reading. This is exactly the kind of
claim you test against feature importances once the residual model
actually trains (§8) — right now it's a documented hypothesis, not yet a
verified result, and it's honest to say so in a viva.

**Calendar features**: `holidays.India(subdiv="HR")` (Haryana) for
`is_holiday` / `holiday_name` / `days_to_nearest_holiday`, plus two
hand-built features not in that package because they aren't public
holidays but are real Gurugram traffic effects: `is_festival_period` (±2
days around a holiday — pre-festival shopping / post-festival return
travel is a multi-day effect) and `is_month_end` (day ≤ 2 or within the
last 3 days of the month — most Indian salaries land there, driving
discretionary/shopping trips).

**Caching**: every `(date, hour)` fetched is written to a git-ignored
on-disk JSON cache so the same hour is never re-requested, and fetches are
batched whole-day-or-wider rather than one request per hour.

---

## 6. `incidents.py` — TomTom Traffic Incidents, matched to corridor geometry

**Why this matters more than weather**, per its own docstring: weather and
calendar features answer "is this an unusual *day*." They can never
explain a crash, a stalled truck, or a closure on one specific corridor at
one specific moment — arguably the single largest driver of the residual
the forecasting model (§8) is trying to predict.

**What the API actually returns** — tested live, not assumed, and the
finding is candidly reported: one bbox query covering the whole corridor
envelope returned 77 real incidents, dominated overwhelmingly by
`iconCategory 8` ("road closed", 70/77) with `magnitudeOfDelay=4`
("severe") on almost all of them, and **no numeric `delay` value on 93.5%
of them** — only 5 of 77 carry a usable delay figure. `magnitudeOfDelay`,
by contrast, is populated on all 77, so it's treated as the reliable
severity signal, not raw delay. Several of the closures cluster tightly
around Dwarka Expressway specifically, consistent with that corridor's
real, long-running construction rather than a fresh crash — so
`has_road_closure` should be read as "an active closure/worksite nearby,"
a slowly-changing signal for some corridors, not a minute-to-minute one.
The docstring says this plainly rather than overselling the feature.

**Spatial matching**: each incident is matched to a corridor by the
minimum distance from *any* vertex of the incident's geometry to *any*
segment of the corridor's digitized polyline
(`frontend/corridors.geojson`, read-only from this file's perspective — it
never writes it, only reads it). Pure-Python point-to-segment distance, no
`shapely` dependency.

**Buffer threshold: 300 m**, chosen empirically rather than guessed —
computing every one of the 77 real incidents' distance to its nearest
corridor showed a clean split: a small cluster under a few hundred metres,
and 65/77 (~84%) beyond 500 m on unrelated roads. 300 m sits in that gap:
wide enough to absorb a divided highway's carriageway width and ordinary
GPS slop, narrow enough not to wrongly attribute a neighbouring corridor's
incident.

**Quota**: one bbox request per round covers all 13 corridors — not one
per corridor — so incidents add roughly +1 request/round (~+96/day) on
top of the routing requests, still comfortably inside the 2,500/day free
tier.

---

## 7. `model/traffic_model.py` — the grid model: feature engineering, training, prediction

### 7.1 Feature engineering — `engineer_features()`

```python
df["road_class_enc"] = df["road_class"].map(corridors.ROAD_CLASS_ENC)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
df["is_peak_morning"] = ((df["hour"] >= 7)  & (df["hour"] <= 10)).astype(int)
df["is_peak_evening"] = ((df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)
```
- `road_class_enc` uses `corridors.ROAD_CLASS_ENC` — the single mapping
  from §3 — on **both** the training side and the prediction side
  (`predict_raw`, below). This is the direct fix for a real bug (§9.3).
- **Cyclical hour encoding** (`hour_sin`/`hour_cos`) is the single most
  important trick in the project. Raw `hour=23` and `hour=0` look maximally
  far apart to a model that just sees the number, when they're actually
  one hour apart. Plotting each hour around a 24-point circle and taking
  its (x, y) coordinate fixes that — midnight and 11 PM become neighbours,
  exactly as real traffic behaves across that boundary.
- `is_weekend` / `is_peak_morning` / `is_peak_evening` are hand-built
  domain-knowledge flags rather than hoping the model discovers rush hour
  purely from the raw hour number — this helps a model learn reliably on
  a dataset that, per corridor, is one row per hour of one representative
  week (168 rows), not thousands.

### 7.2 `load_bootstrap_data()` — cleaning

Reads the CSV, drops rows missing required fields, clips any
out-of-[0,1] `congestion_idx` defensively, and — the important one —
excludes every row flagged `route_stable=False` (§4.1): 146 of 2,184 rows,
concentrated on 5 corridors. Training on 2,038 rows.

### 7.3 Training — `train_model()`, and the corridor-imbalance story

```python
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    min_samples_leaf=10, subsample=0.8, random_state=42,
)
```
**Why Gradient Boosting:** an ensemble of many small, shallow trees fit in
sequence, each one correcting the previous trees' errors — strong on
small/medium tabular data without needing the volume a neural network
would, and it's what the README documents as the winner in a comparison
against Random Forest, plain Linear Regression, and an MLP.

**Two evaluations, run deliberately side by side, to be honest about
quality:**

**Naive random 80/20 split** — the everyday approach, but misleading here:
a row for "MG Road, Monday, 8 AM" and "MG Road, Monday, 9 AM" are nearly
identical, so if one lands in train and the other in test, the model
isn't really being tested on anything new. Reported *for contrast*, not
trusted. Current measured value: **R² = 0.786, MAE = 0.0285** (2,038 rows).

**GroupKFold leave-one-corridor-out** — the honest number. Every fold
trains on 12 corridors and tests on the 13th, with that corridor's
`road_class` never having appeared during that fold's training if it's the
sole member of its class. Current result:

```
MEAN CV R2  = -0.353  (+/- 3.52 across folds)
MEAN CV MAE =  0.0317
```

Per-fold detail (recomputed directly against the current model/data for
this document):

| held-out corridor | road_class | R² | MAE |
|---|---|---:|---:|
| Sohna Road | arterial | 0.980 | 0.012 |
| Golf Course Extension Road | arterial | 0.968 | 0.008 |
| MG Road | arterial | 0.930 | 0.022 |
| NH-48 | highway | 0.908 | 0.017 |
| Southern Peripheral Road | arterial | 0.914 | 0.010 |
| Mehrauli-Gurgaon Road | arterial | 0.893 | 0.025 |
| NH-352W | highway | 0.852 | 0.023 |
| Delhi-Mumbai Expressway | expressway | 0.757 | 0.019 |
| Golf Course Road | arterial | 0.757 | 0.033 |
| Old Delhi-Gurgaon Road | highway | 0.740 | 0.035 |
| Pataudi Road | highway | **-0.741** | 0.044 |
| KMP Expressway | expressway | **-0.117** | 0.091 |
| **Dwarka Expressway** | expressway | **-12.425** | 0.075 |

By class: **arterial** (6 corridors, all have several same-class
siblings) averages a healthy ~0.907. **highway** (4 corridors, expanded
from 1) averages **+0.440** — up from -0.08 when NH-48 was the only
highway corridor and had no same-class sibling to generalize from at all.
**expressway** (3 corridors, expanded from 1) averages **-3.928** — a big
improvement on the old -25.5, but still dragged deeply negative by one
corridor.

**Be honest about what's still wrong**: Dwarka Expressway is *still* a
terrible fold at R² = -12.4. The corridor-imbalance fix (giving every
class same-class siblings) was necessary, but Dwarka's real problem is
different — its true mean congestion is roughly 5× lower than its
nominal sibling KMP Expressway, because it's a genuine outlier
(real-estate-corridor traffic + active construction, per §6) that a single
categorical `road_class` feature cannot express. A model trained on 12
other corridors, none of which look like Dwarka, will always struggle to
predict Dwarka's specific pattern from `road_class` alone. This is a real,
current limitation — say so plainly if asked, it's exactly the kind of
answer examiners reward.

**Why `corridor_id` isn't a production feature**, even though the code
tests whether it would help GroupKFold: `backend/app.py` calls
`predict_raw(model, features, hour, day_of_week, road_class)` with no
`corridor_id` argument, so the deployed feature set has to stay
road-class-based to match that contract. The training script tests and
reports the counterfactual anyway (with a caveat that leave-one-corridor-out
is somewhat unfair to `corridor_id`, since by construction each fold's
value is unseen during that fold's training) — useful "did you consider
X" material.

**Feature importances** (from the current final production model, fit on
all 2,038 rows): `hour_sin` and `is_peak_morning` dominate, consistent
with intuition — *when* it is, especially whether it's rush hour, is the
single biggest driver of Gurugram congestion in a model that has no
weather/incident signal at all.

### 7.4 Prediction — `predict_raw()` / `predict()`

```python
def predict_raw(model, feature_cols, hour, dow, road_class):
    road_class_enc = corridors.ROAD_CLASS_ENC.get(road_class, ...)
    row = { "hour": hour, "day_of_week": dow,
            "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
            "is_weekend": int(dow >= 5), "is_peak_morning": int(7<=hour<=10),
            "is_peak_evening": int(17<=hour<=20), "road_class_enc": road_class_enc }
    return float(model.predict(pd.DataFrame([row])[feature_cols])[0])
```
Rebuilds the *exact same* transformation used at training time for a
single hypothetical input. It's essential this matches `engineer_features`
exactly — any mismatch silently produces wrong predictions with no error,
because the model happily predicts on garbage input (which is exactly
what happened before — §9.3).

`predict()` (the CLI entry point) returns the raw congestion index only,
deliberately not a label — labelling thresholds are owned by the API
layer (`backend/app.py`), so they exist in exactly one place.

---

## 8. `model/forecast_model.py` — the residual model, and why it refuses to train right now

This is the intellectual core of the newer half of the project: **can
weather, calendar and incident conditions predict how traffic will
*deviate* from its normal day-of-week × hour pattern?**

### 8.1 Why model a residual, not absolute congestion

The bootstrap grid is a complete, noise-free map of the diurnal/weekly
rhythm — but, per §5, it has **zero** date/weather sensitivity. There is
nothing left to learn from it about rain or festivals. The observed data
is the opposite: small, but the *only* place time-varying conditions show
up at all. So instead of asking a small dataset to relearn an entire
diurnal curve from scratch:

```
baseline(corridor, day_of_week, hour) = bootstrap grid value    (complete)
residual = observed_congestion - baseline                       (the target)
forecast = clip(baseline + predicted_residual(weather, events, incidents, time), 0, 1)
```

**A denominator subtlety worth having ready for a probing question**:
observed congestion uses `1 - free_flow/live`; the baseline uses
`1 - free_flow/historic`. Since `free_flow` is the same value on both
sides, it cancels in the subtraction: `residual = free_flow × (1/historic
- 1/live)` — "how much worse right now is than typical," exactly the
quantity weather/incidents should explain. The docstring is explicit that
the *real* risk isn't this denominator difference — it's `free_flow`
itself drifting between the two datasets for a corridor that reroutes
(§4.1), which would look like a large constant residual weather/incidents
obviously can't explain and would silently bias training.
`check_free_flow_consistency()` guards against exactly this, comparing
mean `free_flow_s` per corridor between the two datasets (route-stable
rows only) and flagging any corridor that drifted more than 5%. It's
printed on every readiness/train run, not hidden.

**Sign convention**: `congestion_idx` can legitimately go negative (one
real example logged: NH-48 at 00:45 IST, live time *faster* than the
free-flow estimate, `idx = -0.025`). This is real signal about a soft
free-flow reference at very low-traffic hours, not a bug, and it is **not**
clipped on the input side — clipping ground truth would bias what the
model is asked to learn. Only the final *forecast* output is clipped to
[0, 1], because promising a user negative congestion isn't meaningful even
though a negative *measurement* is real.

### 8.2 Evaluation: a skill score, not R²

```python
skill = 1 - (MAE_model / MAE_baseline)
```
This directly answers the obvious challenge — *"couldn't you just average
the past?"* — which R² can't. Positive means the model beats "just use the
historical average"; zero or negative means it doesn't, and the code
prints that outcome plainly rather than hiding it (`train()`'s final
branch literally prints "model does NOT beat the baseline. Report this
honestly").

Two deliberate methodology choices, both because getting them wrong would
produce a flattering but meaningless number:
1. **Time-based holdout** — train on the earliest ~80% of collection
   days, test on the most recent ~20%. A random split leaks: consecutive
   15-minute samples of the same corridor are near-duplicates, so a
   random split lets the model "cheat" on a sample 15 minutes from a
   training sample.
2. **`cv_r2` / leave-one-corridor-out is deliberately not reused here** —
   §7.3 already showed it's the wrong metric when a class has few
   corridors. MAE-based skill score against the baseline is used
   throughout instead.

### 8.3 Honest gating — why no forecast model exists yet

```
MIN_DISTINCT_DAYS            = 14    # need the weekly rhythm to repeat >=2x,
                                      #   AND monsoon rain is intermittent —
                                      #   14 days is the floor for "we saw
                                      #   more than one rain event" to be
                                      #   credible rather than luck
MIN_TOTAL_ROWS                = 1500 # conservative floor given autocorrelated
                                      #   15-min samples overstate independence
MIN_RAINY_ROWS = 50, MIN_DRY_ROWS = 200   # must see BOTH conditions with
                                      #   enough rows to fit a real relationship
MIN_CORRIDORS                  = 6
MIN_INCIDENT_AFFECTED_ROWS = 30, MIN_INCIDENT_CLEAR_ROWS = 150
                                      # restricted to rows with KNOWN incident
                                      #   status (incident_data_known == 1),
                                      #   since incidents can't be backfilled
```
If any of these thresholds aren't met, `train()` **refuses to train and
writes no model artifact**, exiting with a clear message. This is stated
in the file's own docstring as a feature, not a bug: *"A model trained on
300 dry rows cannot predict rain, and shipping one anyway would be
dishonest."*

**Current status, checked directly against the repo while writing this
document**: `data/gurugram_observed.csv` has **115 rows** (CI appends more
roughly every 40 minutes, per the collection cadence in §4.2, so this
number is a moving target — re-run `python model/forecast_model.py
readiness` for the live figure). 115 rows is nowhere close to the 1,500-row
floor, and no `models/forecast_residual_gbt.joblib` exists in the repo —
confirmed by listing `models/`. **Training is currently and correctly
refused.** If a teacher asks "does the weather/incident model work,"
the honest and correct answer is: *it hasn't been able to train yet,
by design, because the data isn't sufficient — and that refusal is itself
the feature being demonstrated.*

### 8.4 Feature imputation for incidents specifically

Because incidents can't be backfilled, a row collected before incident
tracking existed has genuinely *unknown* incident status — not "known
clear." `build_training_table()` imputes "no incident" defaults for the
model to consume (0 counts, no closure/jam, a large sentinel distance) but
also adds `incident_data_known` (1/0) as its own feature, so the model can
itself learn to discount incident features on rows where they're actually
unknown, and so the readiness gate (§8.3) can count only rows with known
status toward the incident-coverage thresholds. The docstring flags a
specific thing to check once this trains: if `incident_data_known` itself
ends up high in feature importance, that's a sign the model is partly
learning "old rows vs. new rows" rather than a genuine incident effect —
an honest caveat to report rather than something to hide.

---

## 9. History worth knowing (bugs found and fixed)

This is the single most valuable section of this document for a viva —
each one is a real, traceable defect with a concrete before/after, not a
hypothetical, and the fixes are still visible in code comments if you want
to point at them live.

### 9.0 The origin story: a pipeline that had never collected a row

Worth stating plainly before the individual bugs, because it's the frame
all of them sit inside. The project's first committed state (`22e1049`,
"Baseline + Phase 0 foundations") had two properties at once: `README.md`
opened by claiming **"Answer: Yes, with R² = 0.83"** and quoted `R² (test
set) 0.83` / `R² (5-fold CV) 0.84 ± 0.003` — and `backend/app.py`, whenever
no trained model file was found, silently fell back to serving predictions
out of a hand-typed `PROFILES` dictionary (`PROFILES[day_type][road_type][hour]`)
instead of a 503 error. Neither number came from real traffic: the R² was
measured against a **synthetic data generator** (the same commit's README
says so directly — *"the pipeline uses a realistic synthetic dataset for
demo purposes"*), and `PROFILES` was simply guessed by a person, not
learned by anything. At that point in the project's history, **zero rows
of real TomTom data had ever been collected** — the original collector
targeted TomTom's Flow API, which 403s on this project's key (§9.1), so
every scheduled run quietly hit a `[WARN] TomTom fetch failed` branch and
produced nothing, while GitHub Actions still reported the workflow as a
successful run.

Both were deleted, on purpose, in later commits: the `PROFILES` fallback
was removed for good in `fe46fa3` ("Rebuild Flask API as v2: delete
PROFILES fallback, precompute 8x7x24 grid…") — confirmed directly against
the current `backend/app.py`, which contains no `PROFILES` dictionary
anywhere, only a comment at its own top noting the removal. Every metric
anywhere else in this document is computed against real, TomTom-sourced
rows collected after that rebuild.

**A documentation-drift example that has since been fixed** — kept here
because an earlier draft of this document flagged it as still broken, and
that claim is itself now stale: `README.md` used to open with *"This repo
is a working pipeline with no real traffic data in it yet"* and quote the
original synthetic-data R²=0.83 figures, unrewritten after the real-data
pipeline replaced it. Checked directly against the current commit while
writing this: that's no longer true. `README.md` now opens *"the whole
pipeline runs on real, measured data, not synthetic samples"* (fixed in
`22e9e78`, "Rewrite README to match the working system, not the
synthetic-era draft") and its own "Accuracy" section quotes the same
58.3%/89.4% figures as §12.3 below. The lesson still stands even though
the specific example doesn't anymore: this file gets re-derived from the
actual code and data every time it's touched — and the proof is that its
own claims about *other* files can go stale too, and did, which is exactly
why "recompute, don't trust a cached document" has to apply recursively,
including to this document.

### 9.1 The original data collector never worked

The very first collector targeted TomTom's Flow Segment Data API. Tested
directly against this project's key: **Flow API → 403 Forbidden,
Search/Geocode API → 403 Forbidden, Routing API → 200 OK.** So the
original collector had been silently failing on the `"[WARN] TomTom fetch
failed"` branch every scheduled run for weeks, producing **zero rows** of
real data the whole time with no visible top-level error — the workflow
"ran successfully" by CI's own bookkeeping while doing nothing useful.
The fix was discovering the `departAt` + `computeTravelTimeFor=all` trick
on the Routing API (§4.1) that this key *can* reach.

### 9.2 Duplicate/conflicting corridor definitions

Before `corridors.py` existed as a single source of truth, "MG Road" and
"Mehrauli-Gurgaon Road" were defined with **byte-identical coordinates**
in two different files (0.0 m apart — the same road, duplicated forever
under two names), and NH-48/Dwarka Expressway/Southern Peripheral Road
shared a GPS endpoint within 17 m of each other. The "8 corridors" of that
era weren't actually 8 distinct roads to the model. Found by computing
pairwise haversine distances between every corridor's start/end points.

### 9.3 Mismatched road-class encoding

Training used scikit-learn's default `LabelEncoder`, which sorts
alphabetically: `arterial=0, expressway=1, highway=2`. The prediction code
(`predict_raw`) independently hardcoded a *different* dictionary:
`{highway:0, arterial:1, expressway:2}`. **Every prediction the model ever
served used the wrong road class** — silently, because both sides produce
valid-looking integers and nothing errors. Fixed by making
`corridors.ROAD_CLASS_ENC` (§3) the single import both sides use.

### 9.4 Congestion label thresholds calibrated to fake data

The Free/Moderate/Heavy/Severe cutoffs were inherited from a synthetic
data generator whose values ranged up to 0.92 (`Free < 0.35`, etc.). Real
Gurugram data peaks at **0.4349** (verified directly from the current
2,184-row bootstrap grid). Under the old thresholds, computing labels
against the real grid: **98.86%** of all cells label "Free" (the frozen
`docs/api_contract.md`, written against the earlier 8-corridor/1,344-cell
grid, quotes 98.5% — same phenomenon, consistent number). The worst cell
in the entire dataset — MG Road, **Friday 19:00**, `congestion_idx =
0.4349`, a 6.8-minute free-flow trip taking 12.0 minutes at that hour —
would have displayed as "Free."

Recalibrated on **travel-time multipliers** rather than percentiles of one
week's data, so the boundaries stay meaningful as more data arrives:

| label | `congestion_index` | means |
|---|---|---|
| Free | `< 0.091` | under 1.10× free-flow time |
| Moderate | `< 0.200` | 1.10×–1.25× |
| Heavy | `< 0.310` | 1.25×–1.45× |
| Severe | `>= 0.310` | over 1.45× |

Resulting distribution across the current, complete 13-corridor/2,184-cell
grid (recomputed directly from `data/gurugram_bootstrap.csv` for this
document): **Free 52.2%, Moderate 27.7%, Heavy 16.7%, Severe 3.4%.**

**A documentation discrepancy worth flagging on its own** (good "how do
you verify your own numbers" material): `docs/api_contract.md` — marked
"FROZEN 2026-08-16" — still quotes the *previous*, 8-corridor-era
distribution (52.5/25.4/18.2/3.9%) and an 8-corridor `/health` example.
It was frozen the day before the 13-corridor expansion and was never
updated afterward. The thresholds themselves are unchanged and still
correct; only the illustrative distribution and corridor count in that
doc are stale. This document uses numbers recomputed directly from the
current data, not copied from that file.

### 9.5 The road-class imbalance (full story in §7.3)

Leave-one-corridor-out CV originally scored **R² = -25.5** on Dwarka
Expressway and -0.08 on NH-48 — each the sole member of its road class —
dragging the mean to **-2.52**. Fixed by expanding 8 → 13 corridors
specifically to give `expressway` and `highway` same-class siblings
(§3.1). Result: mean **-2.52 → -0.35**; highway class mean **-0.08 →
+0.44**; expressway class mean **-25.5 → -3.93** (still dragged deeply
negative by Dwarka specifically, at -12.4 — a genuine outlier, not a
class-imbalance artifact anymore; see §7.3 for why).

### 9.6 Unprotected concurrent git pushes in CI

Three scheduled GitHub Actions workflows (`collect.yml` roughly hourly,
`retrain.yml` weekly, `refresh_bundle.yml` daily) all push commits to the
same branch. A plain `git push` fails the moment any two of them race.
Fixed with a fetch-rebase-retry loop with jittered backoff in every
workflow that commits data.

### 9.7 GitHub's `schedule:` cron does not deliver its nominal rate

Covered in full in §4.2. Worth restating here as its own bug: scheduling
`*/15 * * * *` measurably delivered ~37% of the nominal 96 rounds/day.
Fixed by triggering hourly (an interval GitHub honors reliably) and moving
the 15-minute spacing into the collector's own internal loop.

### 9.8 Whole-day "best time" was midnight for almost every corridor, and minutes-only savings undersold the product

Two real user complaints prompted `7d32d12` ("Communicate time savings
honestly and split best-time by day/night"): the site's "saves ~6 min"
framing read as too small to matter, and the whole-day best-hour advice
sometimes recommended midnight — technically correct, useless to someone
who has to leave for work in daylight.

**The midnight problem, re-verified against the live grid for this
document, not just the commit message.** Averaging `congestion_index`
across all 13 corridors × 7 days at each hour:

| hour | avg `congestion_index` | max |
|---|---:|---:|
| 20:00 | 0.151 | 0.322 |
| 21:00 | 0.075 | 0.150 |
| **22:00** | **0.0008** | 0.031 |
| 23:00 | 0.0006 | 0.010 |
| 06:00 | 0.019 | 0.053 |

21:00 → 22:00 is roughly a **93× collapse** in the hourly average — the
commit's own inline comment in `backend/app.py` quotes 0.087 → 0.0006 (a
>100× drop), measured the day it was written; the numbers above are
slightly different because `data/gurugram_observed.csv` keeps growing
under live CI collection and now overrides more bootstrap cells than it
did on 2026-08-17, but the same cliff is still there. That evidence is
what `DAY_HOURS`/`NIGHT_HOURS` are built from:

```python
DAY_HOURS = list(range(6, 22))                        # 06:00-21:59
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))  # 22:00-05:59
```

Night advice is still fully served, not dropped — truck drivers and
shift workers genuinely do travel then — it just no longer silently wins
every "best time" comparison against daytime hours it was never being
fairly compared against. `/advice` now returns `day_period`/`night_period`
blocks and `/best-time` accepts `period=day|night|any` (§10.10 has the
full API shape).

**One honest caveat this document adds that the commit message doesn't
make**: "near-zero, flat floor overnight — every corridor reads Free" is
an *average*-case claim, not a claim about every individual cell.
Spot-checking the live grid directly for this document turns up a real
counterexample: **Old Delhi-Gurgaon Road, Tuesday 2 AM** — four real
live observations 15 minutes apart on 2026-08-18 (`source="observed"`,
`is_raining=True`, `has_jam=True`, one row carrying `incident_max_magnitude=2`)
show `congestion_idx` climbing **0.151 → 0.193 → 0.236 → 0.275** across
that half hour, served today as `"Heavy"` at `confidence=0.97` (observed
data is the *highest*-confidence tier, §10.3 — this is not low-quality
noise). The day/night boundary is still the right call: it's derived from
the average shape of the week, and one real jam-plus-rain night out of
2,184 cells doesn't change that night hours are overwhelmingly, reliably
quieter than day hours — but a teacher asking "is 22:00–05:59 *always*
empty?" deserves "no, and here's the one real exception I found," not a
restatement of the average as if it were a guarantee.

**The savings-framing half of the same commit.** MG Road is 3.85 km with
a 6.8-minute free-flow time (§10.9 has the full corridor-length table), so
its whole-day best-vs-worst saving is capped at ~5 minutes no matter how
bad the worst hour gets — a number that reads as unimpressive next to
"Gurugram traffic is terrible," even though it's correct. Recomputed live
for this document (`/advice?corridor=1&day=4`, MG Road, Friday): best hour
6 AM (0.1 min delay) vs. worst hour 7 PM (5.2 min delay, **76.5% longer
than free-flow**) — the same 5.2-minute gap as always, but expressed as
**"+5 min, 76% longer"** it reads as a materially different claim, and
it's equally true. Minutes and percent are now both served side by side
(`whole_day_saving_minutes`/`whole_day_saving_pct`,
`peak_delay_minutes`/`peak_delay_pct`) — neither replaces the other.

**The same commit also tightened label-honesty wording**, and this ties
directly back to §12.3's 58.3% label-agreement figure: `/now` and
`/advice` text now say *"Typically \<label\>…"* instead of stating a
typical-hour value as an unqualified present-tense fact, and any summary
built from a cell with `confidence < 0.5` gets an inline `" (Limited data
for this corridor/day — treat as a rough guide.)"` caveat appended —
visible in the sentence a user actually reads, not just in a `confidence`
number a frontend has to remember to check.

`backend/test_api.py` grew from 57 to **79 tests** in this commit across
14 test classes (re-collected directly with `pytest --collect-only` while
writing this document, not taken from the commit message).

---

## 10. `backend/app.py` — the Flask REST API

**What Flask is**: a lightweight Python web framework — write a function,
decorate it with a route, and it becomes an HTTP endpoint.

The module docstring states the two non-negotiables this file implements:
1. **No hand-invented numbers.** If no model is loaded, every model-backed
   endpoint returns HTTP 503 `{"error": "no model trained yet"}`. There is
   no `PROFILES` fallback table — that table, and the invented numbers it
   served, is the v1 bug this rebuild deletes (full origin story in §9.0).
2. **`road_class` encoding always comes from the model payload's own
   `road_class_enc` map when present, else `corridors.ROAD_CLASS_ENC`** —
   never hardcoded/reconstructed ad hoc, the direct fix for §9.3.

### 10.1 The most important design decision in this file: serve measured values, not model predictions

Stated in `load_measured_grid()`'s own docstring: *"Every cell your API
can be asked about should be served from an actual measurement when one
exists — the model is for filling gaps, not for lossily re-compressing a
value we already hold exactly."*

At startup, `load_measured_grid()` builds a `{(corridor, day, hour):
value}` lookup straight from the two CSVs (observed takes priority over
bootstrap where both exist), and the model is used **only** to fill any
`(corridor, day, hour)` cell that has no measurement at all.

**Just how big a difference this makes — verified directly for this
document, not carried over from an earlier draft**: comparing the trained
model's own predictions against every one of the 2,184 measured grid
cells gives **MAE = 0.0265, max absolute error = 0.242**, and — the number
that actually matters for a user-facing label — **490 of 2,184 cells
(22.4%) would show a different Free/Moderate/Heavy/Severe label** if the
model's prediction were served instead of the real measurement. The worst
individual mismatches are on the corridors with the least stable data
(Old Delhi-Gurgaon Road, KMP Expressway): e.g. Old Delhi-Gurgaon Road,
Monday 10 AM, measured `0.3735` ("Severe") vs. model-predicted `0.132`
("Moderate") — two full label bands apart. Serving the model prediction
where ground truth exists would be strictly worse than serving the
measurement, for close to a quarter of the whole grid. That's the concrete
answer if a teacher asks "why not just always use the model."

Right now, because the bootstrap sweep is fully complete (2,184/2,184
cells measured) and the model is retained purely as a gap-filler,
**`inferred_cells = 0`** in the live grid (confirmed directly from
`frontend/data/bundle.json`) — every cell the API can currently answer is
a real measurement. The model becomes load-bearing again the moment a
corridor/day/hour combination exists that neither CSV covers, which is
exactly the situation sparse live data will eventually create for
newer/rarer cells.

### 10.2 `load_model_payload()` — handling old vs. new model formats

Defensive backwards-compatibility: an early version of the model file only
contained `{"model", "features"}` and was trained on synthetic data. If
that shape is ever loaded, the missing metadata keys
(`provenance`, `model_version`, `trained_rows`, `road_class_enc`,
`metrics`) are detected and the payload is explicitly labeled
`provenance="synthetic"` — a direct, deliberate guard against quietly
treating fake-data output as equivalent to real-data output.

### 10.3 `compute_confidence()` — an honest "how much do we trust this number" score

This has been rebuilt since an earlier draft of this document and is worth
re-reading carefully — it **no longer uses `cv_r2` at all**. The current
code's own comment explains exactly why:

> *"NEVER read `metrics["cv_r2"]` here: that figure is
> leave-one-corridor-out, and it is dominated by the two/three corridors
> that are the sole member of their road class... We never serve a road
> class the model hasn't seen, so cross-corridor-class generalization is
> not the risk that matters for an inferred cell here."*

Instead, confidence is a strictly ordered tier system:

```python
CONFIDENCE_OBSERVED           = 0.97  # a live measurement — the best signal
CONFIDENCE_MEASURED_STABLE    = 0.92  # bootstrap measurement, route_stable
CONFIDENCE_MEASURED_UNSTABLE  = 0.50  # bootstrap measurement, route unstable
INFERRED_CONFIDENCE_DEFAULT   = 0.35  # no measurement -> model fills the gap
INFERRED_CONFIDENCE_CAP       = 0.45  # even a strong within-class score can't
                                       #   outrank a real measurement
```
A synthetic-provenance model short-circuits to a flat `0.15` — the one
case where a flat number actually *is* the honest answer, since nothing
real exists anywhere to differentiate cells by. This is a cleaner design
than the multiplicative `base_quality × data_factor × stability_factor`
approach an earlier version of this project used (and an earlier draft of
this document described) — the current version doesn't need a
model-quality figure at all for the common case, because it just checks
"is there a real measurement for this exact cell."

### 10.4 `label_for()` / `LABEL_THRESHOLDS`

Covered fully in §9.4. Defined once here and nowhere else — the frontend
imports the same four numbers rather than redefining them (§11 flags where
that duplication still exists and must be kept in sync by hand).

### 10.5 Startup: precompute the whole grid once

```python
for c in CORRIDORS:
    for day in range(7):
        for hour in range(24):
            rows.append(build_feature_row(...))
X = pd.DataFrame(rows)[FEATURES]
preds = MODEL.predict(X)     # ONE batched call, 2,184 rows
```
The full 13×7×24 grid is predicted **once** at startup (cheap — 2,184
rows in one batched call) and cached in a dict, `GRID`. Per §10.1, the
measured-value lookup then overwrites almost every cell of that
prediction. Every endpoint afterward is a dictionary lookup — the module
docstring states as a design rule that no endpoint calls `model.predict()`
per request.

### 10.6 The endpoints

- **`/health`** — model status, version, row count. No model required.
- **`/corridors`** — the static list from `corridors.py` as JSON; the
  frontend never hardcodes it.
- **`/predict?corridor=&day=&hour=`** — single `GRID[...]` lookup.
- **`/advice?corridor=&day=`** — "the primary endpoint": full 24-hour
  profile plus computed best/worst windows and a natural-language summary
  that leads with the whole-day best-vs-worst saving in minutes *and*
  percent (e.g., recomputed live for this document, MG Road/Friday: *"Leave
  before 8 AM or after 10 PM. Worst is 7 PM (+5 min, 76% longer than
  free-flow). Timing it right saves ~5 min (43% shorter trip) versus the
  worst hour."*). Also returns `day_period`/`night_period` blocks — §10.10.
- **`/advice/all?day=`** — the same payload (including `day_period`/
  `night_period`) for all 13 corridors in one response, so the frontend
  isn't firing 13 parallel requests.
- **`/best-time?corridor=&day=&earliest=&latest=`** — best departure
  inside a user-specified window. Also accepts `period=day|night|any` as
  an alternative to `earliest`/`latest` — §10.10.
- **`/now`** — live verdict for all 13 corridors at the current IST time,
  including a `trend` (rising/falling/flat, comparing this hour to the
  next) and a `verdict` (`go_now` / `wait` / `avoid`).

### 10.7 `find_windows()` — 24 numbers into readable time blocks

```python
lo, hi = min(profile), max(profile)
low_thr, high_thr = lo + 0.15*(hi-lo), hi - 0.15*(hi-lo)
```
"Good" and "bad" are defined *relative to that specific corridor/day's own
range*, not a fixed global cutoff — so even a corridor that's mild all day
still gets its relatively-best/worst hours surfaced rather than nothing
qualifying. `_merge_runs()` then collapses consecutive true/false hour
flags into contiguous windows, specifically handling a window that spans
midnight (e.g. quiet 10 PM–5 AM) by merging the run touching hour 23 with
the run touching hour 0 into one wrapping window `(22, 5)` rather than two
disconnected ones — `backend/test_api.py`'s `TestWindowDetectionWrap`
class exists specifically to pin this edge case.

### 10.8 `_trend_for()` / `_verdict_for()` — the `/now` endpoint's logic

Compares this hour's congestion to next hour's (correctly wrapping past
midnight and past Sunday→Monday) to call `rising`/`falling`/`flat`, then a
small decision table combines label + trend into one of three
user-actionable verdicts — e.g. "Heavy but easing" becomes `wait`, "Heavy
and getting worse" becomes `avoid`.

### 10.9 Why "minutes saved" figures look modest

The frontend and every `/advice`-family endpoint express delay in real
minutes, not just a raw index, via `minutes_from_index()`:

```python
typical = free_flow_minutes / (1.0 - idx)
delay = typical - free_flow_minutes
```

That's arithmetically correct, but it produces numbers that can look
underwhelming next to the "Gurugram traffic is a nightmare" framing a
teacher might expect — worth being ready to explain rather than being
caught off guard by it. **The reason is that Gurugram corridors, as
tracked here, are mostly short**, so even a large *proportional* slowdown
caps out at a small number of *absolute* minutes. Recomputed directly
from `data/gurugram_bootstrap.csv` and `corridors.py` for this document
(weekday cells only, using each corridor's mean measured `free_flow_s`
and its single worst measured `congestion_idx`):

| corridor | free-flow time | peak `congestion_idx` | max weekday delay |
|---|---:|---:|---:|
| Dwarka Expressway (24.30 km) | 22.4 min | 0.095 | **2.4 min** |
| MG Road (3.85 km) | 6.8 min | **0.435** (the dataset's single highest cell) | **5.2 min** |
| … 9 corridors in between … | | | |
| NH-48 Delhi-Gurgaon Expressway | 33.2 min | 0.305 | 14.5 min |
| Old Delhi-Gurgaon Road | 22.2 min | 0.412 | **15.5 min** |

Median across all 13 corridors' weekday maximum: **8.7 minutes**. MG Road
carries the single highest congestion index anywhere in the dataset
(0.4349, §9.4) and still only ever costs **5.2 minutes**, because it is a
3.85 km road with a 6.8-minute free-flow time — you cannot lose 30 minutes
on a 7-minute drive, no matter how jammed it is. Dwarka Expressway sits at
the other extreme: it is long (22.4 min free-flow) but structurally
uncongested (§7.3's outlier finding again, from a different angle), so its
absolute ceiling is the smallest of all 13 corridors even though it's one
of the longest. Old Delhi-Gurgaon Road has the largest ceiling precisely
because it combines real length *and* real congestion (0.412). **The
lesson for a viva answer**: "modest-looking minutes" is not the model
hedging or underselling — it is what a short, real road network
arithmetically produces, and a single global "typical" minutes-saved
number would hide that this varies more than 6× (2.4 to 15.5 minutes)
corridor to corridor for reasons the model can point to directly.

**The same commit that split day/night (`7d32d12`, §9.8) also added
percentage framing alongside minutes**, for exactly the reason this
section lays out — "saves 5 min" reads as unimpressive, but the identical
fact reads differently as "43% shorter trip," and "+5 min" reads as
unimpressive next to "+76% longer than free-flow." Every `/advice`-family
response now carries both forms: `whole_day_saving_minutes` *and*
`whole_day_saving_pct` (the best-vs-worst saving as a percent of the peak
hour's trip time), `peak_delay_minutes` *and* `peak_delay_pct` (how much
longer the worst hour is than free-flow, as a percent). Recomputed live
for this document (`/advice?corridor=1&day=4`, MG Road, Friday): the worst
hour (7 PM) takes **76.5% longer** than free-flow, and leaving at the best
hour instead saves the same 5.2 minutes as **43.3% off the trip**. Neither
framing is invented or replaces the other — both are the same underlying
minutes figure expressed two ways, and minutes are always still shown
alongside the percentage.

### 10.10 The day/night split (added `7d32d12`, 2026-08-17)

Full derivation and a real counterexample the commit message doesn't
mention are in §9.8 — this subsection is just the API shape. What changed:

- `DAY_HOURS = 06:00-21:59`, `NIGHT_HOURS = 22:00-05:59` — a boundary
  derived from a >90x average collapse in `congestion_index` between
  21:00 and 22:00 (§9.8's table), not asserted.
- `/advice` and `/advice/all` now additionally return `day_period` and
  `night_period` blocks, each with its own `best_hour`/`worst_hour`,
  `saving_minutes`/`saving_pct`, best/worst windows, and a period-worded
  `summary` (e.g. *"Best daytime departure: 6 AM. Avoid 7 PM (+5 min, 76%
  longer)."*). Windows within a period are found by
  `find_windows_for_hours()`, a non-circular variant of §10.7's
  `find_windows()`: day and night are linear ranges, not a 24-hour wheel,
  so hour 6 and hour 21 must never be merged as adjacent the way hour 23
  and hour 0 are on the full clock.
- `/best-time` accepts `period=day|night|any` as an alternative to
  `earliest`/`latest`. When a period is given, `window_constrained` is
  always `false` and the summary never second-guesses the caller by
  comparing back against the unconstrained 24-hour scan — that comparison
  is exactly what made night silently "win" every best-time query before
  this feature existed.
- Night advice is still fully served, not dropped — real audience: truck
  drivers and shift workers do travel then. The frontend defaults to
  showing Day, since daytime commuters are the primary audience.

All fields from before this commit (`profile`, `best_windows`, `best_hour`,
`peak_hour`, `summary`, `confidence`, `earliest`/`latest`, …) are
unchanged — this is additive, not a breaking change.

---

## 11. `frontend/index.html` — the dashboard

Still a single self-contained HTML file, but this has grown substantially
since an earlier draft of this document and now includes a real map
library — worth being accurate about if asked to walk through it.

### 11.1 What actually renders the map now

`frontend/vendor/maplibre-gl.js` + `.css` are vendored (checked into the
repo, not pulled from a CDN at runtime) and used to render real map tiles
from `https://tiles.openfreemap.org/styles/positron`, with each corridor
drawn as a real line along its digitized polyline
(`frontend/corridors.geojson`, produced by `tools/build_corridor_geojson.py`
— §12.2) colored by current congestion. **The original hand-rolled
schematic SVG projection is kept as an offline fallback** — `MAP_LOAD_TIMEOUT_MS
= 7000`, and if MapLibre's tiles don't load in time (offline, tile host
down), `fallbackToSchematic()` switches to the old not-to-scale SVG line
drawing so the page still works with zero network access. This is a
sensible degrade-gracefully design, not two competing half-finished
features.

### 11.2 Where the frontend gets its data — the static-bundle-first design

```js
const API_PARAM = new URLSearchParams(location.search).get('api');
const BUNDLE_URL = 'data/bundle.json';
```
On load (`init()`), the page picks one of three data sources, in order:
1. **`?mock=1`** — never touches the network; renders from local fixture
   data (`mockFetch()`), for demoing/screenshotting without a backend.
   `?mock=1&synth=1` additionally simulates a synthetic-provenance model,
   specifically to prove the "this is not real data" warning banner
   actually renders differently — a deliberate test path, not leftover
   debug code.
2. **`?api=<url>`** — explicit opt-in to a live Flask backend (this is
   what `run.sh` passes automatically for local development).
3. **Default (what the public GitHub Pages site actually uses)**: fetch
   `frontend/data/bundle.json` once and set `BUNDLE_MODE = true`. Every
   subsequent "API call" (`apiGet()`) is served from that already-loaded
   JSON via `bundleFetch()`, which reimplements the same trend/verdict/
   window logic as `backend/app.py` client-side in JavaScript. If the
   bundle can't be loaded, the page falls through to attempting the live
   API, and if that also fails, shows the existing honest "can't reach the
   API" banner — the page never silently shows nothing.

**Why static-first, not API-first, for the public deployment**: the grid
the API serves only changes when data is recollected or the model is
retrained — nothing about `/now` or `/best-time` is actually dynamic
server-side computation, both are just different views over the same
precomputed grid evaluated against whatever time the browser's clock
reports. So a frozen JSON snapshot reproduces every endpoint's behaviour
with **no server required**: instant load, zero hosting cost, nothing that
can be down during a demo. `tools/build_static_bundle.py` (§12.1)
generates this bundle by *importing* `backend/app.py` and reading its
already-tested `GRID` and `advice_payload_for()` directly — not
reimplementing the logic — so the two can never silently diverge.

**A concrete "does this actually work offline" fact worth having ready**:
opening `index.html` directly via `file://` (not through `run.sh`'s local
HTTP server) triggers a `protoOverlay` warning telling the user to run
`./run.sh` — because a bare `file://` origin can't reliably `fetch()` even
a same-directory JSON file in every browser. The public GitHub Pages
deployment doesn't have this problem since it's served over `https://`.

### 11.3 `labelFor()` / `colorFor()` — a duplication that must be kept in sync by hand

```js
function labelFor(v){ return v<0.091?'Free':v<0.200?'Moderate':v<0.310?'Heavy':'Severe'; }
```
This is the frontend's own copy of `backend/app.py`'s `LABEL_THRESHOLDS`
(§9.4/§10.4). It has to be duplicated because the frontend is plain
JavaScript with no build step that could import a Python constant. A
mismatch here would visibly self-contradict on screen (a bar colored
"orange" next to text saying "Moderate") — good material for "what's a
maintenance risk in your own codebase" if asked.

### 11.4 Route planner — `geocodeLocation()` / `matchCorridor()`

Free-text place names ("Cyber Hub" → coordinates) go to OpenStreetMap's
free Nominatim geocoding service; `matchCorridor()` then finds whichever
tracked corridor's polyline is geometrically closest to the two resolved
points (point-to-line-segment distance — the same kind of 2D geometry
`incidents.py` uses for a different purpose, §6), and calls `/best-time`
(or its bundle equivalent) for that corridor. If geocoding fails (offline,
Nominatim unreachable), it falls back gracefully to whichever corridor is
already selected.

### 11.5 `render()`

One master function re-draws every UI section (map, corridor list, advice
banner, detail popup, stats, model/provenance card, travel windows) from
current state. Every event handler (day button, hour slider, corridor
click) just updates a state variable and calls `render()` — simple and
predictable rather than patching individual DOM elements from many
different places.

---

## 12. `tools/` — build-time scripts (not run at request time)

### 12.1 `tools/build_static_bundle.py`

Generates `frontend/data/bundle.json`. **Deliberately does not
reimplement `backend/app.py`'s logic** — it imports `app.py` as a module
(exactly like `backend/test_api.py` does) and reads its already-computed,
already-tested `GRID`, `CORRIDORS`, `FREE_FLOW_MINUTES`, and
`advice_payload_for()`. If the two ever diverged, the public static site
would silently disagree with the live API; importing makes that
impossible by construction.

One small but real precision detail documented in the script's own
comments: it reads `free_flow_minutes` from `FREE_FLOW_MINUTES` (2 decimal
places) rather than from the already-1dp-rounded value inside `GRID`,
because computing `typical_minutes`/`delay_minutes` from the coarser
1dp value client-side would double-round and drift up to ~0.1 minute (~6
seconds) from what the live API reports for the same cell — found and
fixed empirically by comparing bundle output against a running backend.

Current bundle stats (verified directly from the committed file for this
document): **13 corridors, 91 advice objects (13×7), 2,184 measured
cells, 0 inferred cells**, ~98 KB. The claim in an earlier draft that this
bundle was "verified against the live API across all 2,184 cells and 91
advice objects with zero mismatches" is architecturally guaranteed here
(by the import-not-reimplement design), not a one-time manual check that
could go stale — every rebuild re-derives the bundle from the same code
path the live API uses.

### 12.2 `tools/build_corridor_geojson.py`

One-time/manual generator for `frontend/corridors.geojson` — the real,
digitized polyline geometry each corridor's map line and incident-matching
buffer are drawn from (as opposed to a straight line between `start` and
`end`). Requests `routeRepresentation=polyline` from TomTom at an
off-peak departure time (03:00 IST) specifically so corridors known to
reroute by time of day resolve to their stable/modal route, checks the
resulting polyline length against `verified_km` (±5%), and **refuses to
write any output at all** if any corridor is outside tolerance — a bad
fetch can never silently produce a geojson that disagrees with the frozen
corridor data in `corridors.py`.

### 12.3 `tools/evaluate_accuracy.py` — checking the site's own numbers against reality

Every section above discusses *how* the numbers are produced. This script
asks the different, more uncomfortable question: **now that the site is
live, how close are its served values to what actually happened?**
Committed as `80f1586` ("Add accuracy evaluation: measure served numbers
against observed reality") along with its output, `docs/accuracy_report.md`
and `docs/accuracy_history.csv`. It is read-only against the two CSVs — no
network calls, nothing it touches feeds back into training — and it
deliberately imports `LABEL_THRESHOLDS` / `label_for()` from
`backend/app.py` and corridor metadata from `corridors.py` rather than
redefining them, for the same reason `tools/build_static_bundle.py` imports
`app.py` (§12.1): so this report can never silently disagree with what the
live site actually serves.

**What it compares**: for every row in `data/gurugram_observed.csv` (a real
live measurement, `congestion_idx = 1 - free_flow/live`), it joins on
`(corridor_id, day_of_week, hour)` to the matching cell in
`data/gurugram_bootstrap.csv` — the **served** value, i.e. what
`backend/app.py` actually returns for that cell today, since the bootstrap
grid is what's served for the ~96% of cells never directly observed. Both
sides share the same `free_flow` numerator, so the two `congestion_idx`
values are directly comparable, not two different quantities.

**Headline numbers, as committed in `docs/accuracy_report.md` (generated
2026-08-17, n=115 observed rows, 76/2,184 cells = 3.5% of the grid ever
observed)** — re-verified directly against the CSVs while writing this
document, filtering `data/gurugram_observed.csv` to rows collected at or
before the report's own cutoff timestamp (`2026-08-17T11:45:00`) to
reproduce its exact n=115 sample:

- **Point error**: MAE = **0.057**, RMSE = **0.075**, bias = **-0.017**
  (95% bootstrap CI -0.031 to -0.004). Bias is signed `served - observed`,
  so a negative bias means the site **understates** real congestion on
  average — confirmed by direct recomputation.
- **Label agreement**: only **58.3%** of the time did the served
  Free/Moderate/Heavy/Severe label exactly match the observed one (95%
  Wilson CI 49.1%-66.9%, n=115). Of the mismatches, **28.7%** were the
  *dangerous* direction (site showed a better label than reality — a user
  who trusted "Moderate" actually met "Heavy"), **13.0%** were the merely
  annoying direction (site showed worse than reality).
- **Hour-ranking (advice-level) accuracy**: pairwise concordance
  **89.4%**, best-hour-hit rate **76.9%** (10/13 corridor/day groups),
  worst-hour-hit rate **76.9%** — computed over 142 comparable hour-pairs
  across 13 corridor/day groups where at least two distinct hours had been
  observed. All of those groups happen to be Monday, since that's the only
  day-of-week with enough observed-hour diversity yet (§8.3's readiness
  gate again, from a different angle: thin data limits what can honestly
  be measured, not just what can be trained).

**The interpretation that matters most for a viva answer**: *the site is
far better at "when should I go" (89.4% hour-ranking concordance) than at
"how bad is it, exactly" (58.3% label agreement)*. That's not a
contradiction — it's a direct consequence of what the bootstrap grid
actually is. It's a day-of-week × hour **average**, so it captures the
*shape* of a typical day (rush hour is worse than 2 AM, Friday evening is
worse than Sunday morning) reliably — which is exactly what ordering two
hours against each other tests — but it cannot capture *today's specific*
deviation from that average (today's rain, today's one-off jam), which is
exactly what an exact label match on one specific date requires. A model
whose ranking survives even when its absolute values drift is precisely
what you'd expect from averaging over many days and then being checked
against one.

**A discrepancy worth naming directly**: an earlier draft of this document
(not this one — a project brief written before this section existed)
described the accuracy report's bias as varying by "congestion band" with
specific per-band figures. That per-band breakdown does not exist anywhere
in the repo — not in `docs/accuracy_report.md`, not printed by
`tools/evaluate_accuracy.py`. Recomputing it directly (grouping the same
n=115 join by the *served* Free/Moderate/Heavy/Severe label instead of a
quartile split) gives: **Free bias +0.015 (n=33), Moderate bias -0.058
(n=56), Heavy bias +0.016 (n=22), Severe bias +0.108 (n=4, far below even
the "insufficient" reporting floor this project uses elsewhere — do not
quote that last one as a real finding)**. The clear signal in that table is
that **Moderate-labelled cells are the most reliably understated** (n=56,
the largest bucket, bias -0.058) — worth stating as the actual, verified
finding instead of numbers that don't trace back to any file in this repo.

**Coverage is the real caveat over all of the above**: 3.5% of the grid,
concentrated on Monday and Sunday only (Tuesday-Saturday have zero
observed coverage as of the committed report), collected over a single
~15-hour window spanning two calendar days. `docs/accuracy_report.md`
itself tags every per-corridor and per-road-class breakdown as "LOW" or
"INSUFFICIENT" for exactly this reason and refuses to present them as
standalone conclusions — the same honesty discipline as §8.3's training
gate, applied to evaluation instead of training. As CI keeps collecting
(`data/gurugram_observed.csv` had already grown to 128+ rows by the time
this document was finished, ahead of the committed report's 115 — it's a
live-growing file, re-run `python tools/evaluate_accuracy.py` for the
current figure), coverage and confidence both improve without any code
change.

`backend/app.py` now surfaces the two headline figures itself, via an
`ACCURACY_SUMMARY` constant (`label_agreement_pct: 58.3`,
`hour_ranking_concordance_pct: 89.4`) sourced verbatim from the committed
report and exposed through `/health` — so a user of the live site (or
static bundle) can see the same honest self-assessment discussed here,
not just this document.

**The same commit that added `ACCURACY_SUMMARY` (`7d32d12`, §9.8) also
changed how individual sentences are worded**, for the same underlying
reason: a served label is a *typical* value, and stating it as an
unqualified present-tense fact ("Heavy now") overclaims exactly what the
58.3% figure above says not to trust. `/now` and `/advice` text now reads
*"Typically \<label\>…"*, and any summary built from a cell with
`confidence < 0.5` gets an inline caveat — `" (Limited data for this
corridor/day — treat as a rough guide.)"` (or the `/now`-specific
wording) — appended directly to the sentence a user reads, not just
exposed as a separate `confidence` number a frontend has to remember to
check.

---

## 13. Supporting/config files

| File | Purpose |
|---|---|
| `requirements.txt` | Training-side packages: scikit-learn, pandas, numpy, joblib, requests |
| `backend/requirements.txt` | API-side: adds Flask, flask-cors, gunicorn, pytest |
| `requirements-collect.txt` | Just `requests` (+ `holidays` for calendar features) — deliberately minimal, since `collect_live.py` runs every CI round and shouldn't reinstall pandas/scikit-learn just to make an HTTP call |
| `backend/test_api.py` | pytest suite: label thresholds, health/corridors, confidence honesty, measured-vs-inferred grid behaviour, route-instability confidence, `/predict`/`/advice`/`/advice/all`/`/best-time`/`/now`, day/night saving fields (§9.8/§10.10), midnight-wrap window detection, and the no-model 503 path. **79 tests across 14 test classes** (verified directly with `pytest --collect-only`, not carried over from an earlier draft — it was 57 tests/12 classes before `7d32d12`). |
| `docs/api_contract.md` | The frozen API specification — **stale in one respect** (§9.4: it still describes 8 corridors / 1,344 cells / the older label distribution, frozen the day before the 13-corridor expansion). The field names, types, and label thresholds it documents are still accurate; the illustrative numbers are not. |
| `.github/workflows/collect.yml` | Runs `collect_live.py --loop --max-rounds 4 --max-minutes 50` roughly hourly (§4.2/§9.7), commits new rows with a fetch-rebase-retry loop. |
| `.github/workflows/retrain.yml` | Weekly (Monday 03:00 UTC): retrains `model/traffic_model.py`, rebuilds the static bundle, commits both. |
| `.github/workflows/refresh_bundle.yml` | Daily (04:10 UTC), separate from retraining: rebuilds just the static bundle, so newly-collected observed rows show up in the public grid well before the next weekly retrain. |
| `run.sh` | One-command local launcher: starts the Flask backend on `.venv_backend`, serves `frontend/` over real HTTP (never `file://`), opens the browser at `index.html?api=http://localhost:<port>`, cleans up both processes on Ctrl-C. Reuses an already-running backend on the target port if it answers `/health`; finds a free port otherwise. |
| `.env` | Holds the real `TOMTOM_API_KEY` locally — gitignored, never committed. |
| `models/traffic_gbt.joblib` | The trained grid model. |
| `models/forecast_residual_gbt.joblib` | Does **not** exist yet — see §8.3. |
| `data/gurugram_bootstrap.csv` | 2,184 rows (13×7×24), TomTom's historical model, complete. |
| `data/gurugram_observed.csv` | Growing live-observed dataset, currently 115 rows, appended roughly every 40 minutes by CI. |
| `data/weather_cache.json`, `data/.tomtom_budget.json` | Local run-state caches — gitignored, rebuildable, not data assets. |
| `tools/evaluate_accuracy.py` | Read-only accuracy self-evaluation: served (bootstrap) vs. observed values. §12.3. |
| `docs/accuracy_report.md` | Generated output of the above — point error, label agreement, hour-ranking accuracy, all with sample sizes and confidence tiers. §12.3. |
| `docs/accuracy_history.csv` | Append-only run log of the accuracy script, so accuracy-over-time can be tracked as `data/gurugram_observed.csv` grows. |

---

## 14. Current limitations — be explicit about these

Examiners reward candour here; do not undersell it.

1. **Dwarka Expressway is a genuine outlier the model can't currently
   express** (§7.3) — its real congestion pattern doesn't resemble any
   other corridor's, and `road_class` alone can't capture that. A
   corridor-specific or embedding-based feature would help; it isn't
   built.
2. **The forecasting/residual model has never trained** (§8.3) — not a
   failed experiment, a correctly-refused one, but until the observed
   dataset clears ~1,500 rows across 14+ distinct days with meaningful
   rain and incident coverage, there is no evidence either way about
   whether weather/incidents actually improve predictions here. The
   hypothesis (`rain_last_3h` mattering more than instantaneous
   precipitation) is untested.
3. **Incidents cannot be backfilled** (§4.2/§8.4) — the incident feed only
   ever showed live/current incidents on this key, so historical
   incident-affected rows can only accumulate going forward, permanently
   capping how fast that specific readiness gate can clear.
4. **The bootstrap grid is a single week, extrapolated forever** — TomTom's
   historical model itself has zero month-to-month or seasonal variation
   (proven by the byte-identical-Fridays test, §5); if Gurugram's actual
   traffic pattern shifts structurally (new road opens, a corridor's
   construction finishes), the bootstrap baseline won't know until it's
   re-swept.
5. **`docs/api_contract.md` is stale** in its illustrative numbers (§9.4) —
   a real example of documentation drifting out of sync with a fast-moving
   codebase, worth naming directly if asked "is your documentation
   trustworthy."
6. **The frontend duplicates label thresholds in JavaScript** (§11.3) by
   necessity (no shared build step with the Python backend) — a real,
   acknowledged maintenance risk.
7. **`corridor_id` as a feature was tested and shown to help
   leave-one-corridor-out CV, but isn't shipped**, because the deployed
   prediction contract doesn't pass it (§7.3) — a deliberate
   interface-vs-accuracy tradeoff, not an oversight.
8. **Label agreement against real observed data is only 58.3%** (§12.3,
   n=115, 3.5% cell coverage) — the site's exact "Free/Moderate/Heavy/
   Severe" call is right little better than half the time on any single
   real date, and understates congestion (the more dangerous error
   direction) more often than it overstates it. The mitigating fact, also
   measured rather than asserted, is that hour-ranking concordance is
   89.4% — the site is much more trustworthy for "which hour is better"
   than for "exactly how bad is this hour," and it says so about itself
   (`ACCURACY_SUMMARY` in `backend/app.py`). Coverage is still thin and
   concentrated on two days of the week, so both figures will keep moving
   as more data accumulates — that is a limitation of the evaluation's
   current sample, not evidence the site is more or less accurate than
   measured.

**What's next, with more time/data**: let the observed dataset clear the
`forecast_model.py` readiness gate and actually train the residual model;
test the `rain_last_3h` hypothesis against real feature importances; add a
corridor-specific or geometric feature (e.g. distance from CBD, number of
signals) so Dwarka Expressway's outlier status is something the model can
learn rather than something GroupKFold just has to absorb; consider
re-sweeping the bootstrap grid periodically to catch structural drift.

---

## 15. Quick reference: how to run everything

```bash
# One command, does everything (backend + static frontend server + browser):
./run.sh

# Or manually:
pip install -r requirements.txt -r backend/requirements.txt

# (Data already collected in data/*.csv — to re-collect from scratch:)
python bootstrap_collect.py --max-requests 1400   # one-time historical sweep, 2,184 cells
python collect_live.py --once                     # one live snapshot round (13 corridors + incidents)

# Train the grid model
python model/traffic_model.py train

# Check / train the forecasting (residual) model
python model/forecast_model.py readiness           # will currently report NOT ready
python model/forecast_model.py train                # will currently refuse — by design

# Run the API server
cd backend && python app.py          # serves http://localhost:5000

# Build the static bundle the public site actually uses
python tools/build_static_bundle.py  # writes frontend/data/bundle.json

# Open the frontend (bundle-backed, no server needed)
open frontend/index.html
# ...or against the live local API:
open "frontend/index.html?api=http://localhost:5000"

# Run the test suite
pytest backend/test_api.py -v
```

---

## 16. Likely viva questions, and how to answer them

**"Is this really machine learning if you serve measured values instead
of the model's predictions?"**
Yes, and the measured-first design is itself the more defensible
engineering decision, not a way of avoiding ML. The model exists, is
trained on real data with an honestly-reported (negative) leave-one-
corridor-out score, and is what fills any gap in the measured grid. Right
now the grid happens to be 100% measured (2,184/2,184 cells), so the
model isn't currently load-bearing for any served value — but it becomes
load-bearing the instant a new corridor/day/hour combination appears that
neither CSV covers. I verified directly (§10.1) that if the model's
predictions were served instead of the measurements, 22.4% of cells would
show a different congestion label — serving the real measurement where
one exists is strictly more accurate, and pretending otherwise would be
the dishonest choice, not the more "ML" one.

**"Why is your cross-validation R² negative?"**
Leave-one-corridor-out CV holds an *entire corridor* out, including its
`road_class` if that corridor is the only member of its class in that
fold's training set — that's extrapolating to an unseen category, a much
harder task than ordinary generalization, and R² can go arbitrarily
negative when a model's predictions are worse than just guessing the
training mean. The mean is -0.35 overall, but that average hides real
structure: arterial corridors (which have several same-class siblings)
average ~0.91; the negative mean is driven almost entirely by Dwarka
Expressway (-12.4), a corridor whose real traffic pattern is a genuine
outlier — its true congestion is roughly 5× lower than its nominal
sibling — that a single categorical feature can't express. I can show the
full per-corridor table (§7.3) and the specific numeric reasoning for why
adding 5 corridors moved the mean from -2.52 to -0.35 without fully fixing
Dwarka specifically.

**"How do you know your data is real, not generated?"**
Every row in `data/gurugram_bootstrap.csv` and `data/gurugram_observed.csv`
is a logged response from a live TomTom API call — I can point to the
exact request URL format and the exact JSON fields (`noTrafficTravelTimeInSeconds`,
`historicTrafficTravelTimeInSeconds`, `travelTimeInSeconds`) each number
comes from. There is a documented, deleted synthetic-data era earlier in
this project (§9.1) whose numbers (R² = 0.83, a hand-typed lookup table)
are the counter-example — I can explain exactly what was fake about them
and exactly what replaced them. The backend still carries an explicit
`provenance` field (`"observed"` / `"bootstrap"` / `"synthetic"`) on every
response specifically so a synthetic-era model, if one were ever loaded
by accident, could never silently masquerade as real.

**"What happens when it rains?"**
Honestly: right now, nothing different — the currently-serving grid model
(`model/traffic_model.py`) has no weather input at all, only time-of-day
and road class, because it's trained on the bootstrap grid, which is
provably weather-blind (§5's byte-identical-Fridays test). A second model
(`model/forecast_model.py`) exists specifically to learn a
weather/calendar/incident-conditioned deviation on top of that baseline,
but it has correctly refused to train so far because there isn't enough
observed data yet (115 rows against a 1,500-row floor, §8.3). That refusal
is a deliberate, defensible design choice — the alternative was shipping
a model that claims to predict rain's effect off a handful of rainy
samples, which would be a confident-sounding but dishonest number.

**"Why 13 corridors and not, say, 20 or just the original 8?"**
8 was too few to give every road class (arterial/expressway/highway) a
same-class sibling for cross-validation, which produced a badly misleading
mean CV score (§7.3/§9.5). 13 was chosen as the minimum expansion that
gives every class at least 3 members, each new corridor picked for a
specific, stated structural reason (different construction status,
different lane count, different urban-vs-rural character — §3.1), not
just "more data." Going further would help Dwarka's specific
outlier-corridor problem less than a genuinely different kind of feature
(§14) would.

**"Why does the site treat day and night differently, and why 6 AM-10 PM
specifically?"**
Because the whole-day best-hour was midnight for almost every corridor —
roads are structurally empty overnight — which made "best time to leave"
identical and useless across every corridor for the audience that actually
needs it: daytime commuters. The boundary isn't a guess: averaging
`congestion_index` across all 13 corridors × 7 days per hour shows roughly
a 90x collapse between 21:00 and 22:00, and a near-flat floor holding
through the early morning before climbing sharply from 06:00 (§9.8, with
the current recomputed numbers). Night advice is still fully served — real
audience: truck drivers and shift workers do travel then — it's just its
own explicit `period=night` rather than silently outcompeting every
daytime hour it was never being fairly compared against. I'd also volunteer,
unprompted, that "every corridor reads Free overnight" is an average-case
claim, not a universal one — I found a real counterexample myself in the
live-observed data (Old Delhi-Gurgaon Road, a real 2 AM jam during rain,
§9.8) and can point to it directly, which is a stronger answer than
restating the average as if it were a guarantee.

**"Your labels are only 58% accurate against real data — is this actually
useful?"**
Yes, and I measured that number myself rather than waiting to be caught
out on it (§12.3, `tools/evaluate_accuracy.py`, n=115 real observations):
label agreement is 58.3%, and the site's own `/health` endpoint says so
via `ACCURACY_SUMMARY`, not just this document. But label agreement is the
wrong single number to judge the product by, because the site's actual
promise is comparative — "leave now, not at 6 PM" — not "the congestion
index at 6 PM is exactly 0.31." Measured the same way, hour-ranking
concordance (does the site correctly say hour A is better or worse than
hour B, for pairs of hours we actually observed) is **89.4%**, and
best-hour-hit rate is 76.9%. A model can have its absolute values
shifted and still rank hours correctly relative to each other, and that's
what happened here — the bootstrap grid is a day-of-week × hour *average*,
so it reliably captures the *shape* of a typical day even when it can't
capture one specific date's deviation from it. I'd rather report both
numbers honestly, at their real sample size (n=115, 3.5% of the grid, two
days of the week), than pick whichever one looks better.

**"What was the single hardest bug to find?"** (good if asked for a
debugging story)
The road-class encoding mismatch (§9.3): training used scikit-learn's
alphabetical `LabelEncoder` ordering, prediction hardcoded a different,
wrong mapping. It never crashed — both sides produced syntactically valid
integers, so every prediction the model ever served was silently wrong in
a way that looked completely normal. Found by reading the two pieces of
code side by side and noticing the dictionaries disagreed, not from any
runtime error. The fix (§3) was making the encoding live in exactly one
place that both sides import.
