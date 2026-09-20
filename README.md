# Gurugram Traffic Congestion Predictor

> Predicts road congestion on 13 of Gurugram's key corridors, by day of week and hour, from a
> grid of real TomTom traffic measurements — and tells you the best time to leave. Built as a
> CS portfolio project; the whole pipeline runs on real, measured data, not synthetic samples.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBT-orange?style=flat-square)
![Data](https://img.shields.io/badge/Data-TomTom%20API-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

For the full line-by-line technical writeup (every file, every import, every historical bug
and how it was found) see [`PROJECT_EXPLAINER.md`](PROJECT_EXPLAINER.md) — this README is the
short version.

---

## Live demo

**Live site (Vercel):** https://gurugram-traffic-predictor.vercel.app

The GitHub repo is connected; each push to `main` redeploys. Root Directory stays the repo
root; `vercel.json` sets `outputDirectory` to `frontend/`. No Flask, no build step — the
static prediction bundle (`data/bundle.json`) is what the map reads.

**GitHub Pages (optional):**
[namanotfound-byte.github.io/Gurugram-traffic-predictor/frontend/index.html](https://namanotfound-byte.github.io/Gurugram-traffic-predictor/frontend/index.html)
— same static bundle, but the URL includes `/frontend/` because Pages serves the repo root. Enable
under **Settings → Pages → Deploy from a branch → `main` / `/root`** if you want a second mirror.

---

## How to run it locally

```bash
git clone https://github.com/namanotfound-byte/Gurugram-traffic-predictor.git
cd Gurugram-traffic-predictor
./run.sh
```

That's it. `run.sh` starts the Flask backend, serves `frontend/` over HTTP on its own port, and
opens it in your browser — you land on a live map with an advice banner, sidebar, and popups.
Press `Ctrl-C` to stop both servers.

**Don't open `frontend/index.html` directly** (double-clicking it, or a `file://` URL) — browsers
block a `file://` page from reaching the API, the static bundle, or the map's road geometry. Serve
it over HTTP instead: `./run.sh`, or any static server (`python3 -m http.server`) pointed at
`frontend/`.

Requirements: Python 3 and a `.venv_backend` virtualenv with `backend/requirements.txt`
installed. If `run.sh` can't find it, it prints the exact `python3 -m venv` / `pip install`
commands to create it — run those once, then `./run.sh` again. If port 5000 or 8000 is already
taken (common on macOS: AirPlay Receiver uses 5000), `run.sh` finds the next free port
automatically.

---

## Deploying / viewing the live site

The frontend can run entirely without a backend. **Production is on Vercel:**
https://gurugram-traffic-predictor.vercel.app — the GitHub repo is already connected, so each
push to `main` redeploys. Root Directory stays the repo root (`vercel.json` points
`outputDirectory` at `frontend/`). No build command, no Node package — Vercel uploads `frontend/` as a plain static site.
There are no catch-all rewrites to `index.html` (that would break `data/bundle.json` and
`vendor/`); unknown paths fall through to `frontend/404.html` if present.

GitHub Pages works the same way but is optional: it serves static files only, so
`frontend/index.html` reads `frontend/data/bundle.json` instead of calling the Flask API.

In both cases the page uses a **precomputed snapshot**, not a live API. The bundle is built from
the same 13-corridor × 7-day × 24-hour grid the backend serves locally (via
`tools/build_static_bundle.py`, refreshed automatically once a day by
`.github/workflows/refresh_bundle.yml`) — same thresholds, same `label`/`typical_minutes` math,
same advice text, just baked into a file.

The page always tells you which data source it's using:
- The header pill reads **`DATA: static bundle`** by default, or **`API: <url>`** when talking to
  a live backend.
- A **`data as of <date> IST`** pill shows the bundle's `generated_at` timestamp, so a stale
  bundle is never mistaken for live data.
- `?api=<backend-url>` (what `./run.sh` passes automatically) skips the bundle and talks to that
  backend directly — for local development.
- `?mock=1` renders offline fixture data behind a "DEMONSTRATION DATA" banner, for UI work with no
  data dependency.
- If the bundle can't be loaded and no `?api=` was given, the page falls back to
  `http://localhost:5000` and shows an honest "can't reach the API" message if that fails too — it
  never silently shows nothing or fabricates numbers.

---

## What it predicts

```
congestion_index = 1 − (free-flow travel time / expected travel time)
```
`0.0` = free flowing, `1.0` = standstill. Collected via the **TomTom Routing API**: a request with
a future `departAt` and `computeTravelTimeFor=all` returns TomTom's own historical traffic model
for that road at that time of week, even though this project's API key is not entitled to the
Flow Segment Data API (see "How the data is collected" below for why that mattered).

| label | `congestion_index` | means |
|---|---|---|
| `Free`     | `< 0.091`  | under 1.10x free-flow travel time |
| `Moderate` | `< 0.200`  | 1.10x – 1.25x |
| `Heavy`    | `< 0.310`  | 1.25x – 1.45x |
| `Severe`   | `>= 0.310` | over 1.45x |

These boundaries are anchored on round travel-time multipliers, not percentiles of one week's
data, so they stay meaningful as more data arrives. Real Gurugram data currently peaks around
0.44 — under the project's original thresholds (inherited from a now-deleted synthetic data
generator, which ranged up to 0.92), the worst cell in the city would have displayed as `Free`.

## Corridors covered

13 corridors — 6 arterial, 3 expressway, 4 highway — each geocoded and validated end-to-end
against TomTom's Routing API (returned road length checked against a plausible range for that
road). The single source of truth is [`corridors.py`](corridors.py); every collector, the
trainer, and the API import from it rather than redefining coordinates.

| Corridor | Type | Route |
|---|---|---|
| NH-48 Delhi-Gurgaon Expressway | Highway | Rajiv Chowk → Manesar |
| MG Road | Arterial | IFFCO Chowk → Sikandarpur |
| Golf Course Road | Arterial | Sikandarpur → Sector 56 |
| Sohna Road | Arterial | Rajiv Chowk → Badshahpur |
| Dwarka Expressway | Expressway | Dwarka Sector 21 → Kherki Daula |
| Golf Course Extension Road | Arterial | Sector 56 → Vatika Chowk |
| Mehrauli-Gurgaon Road | Arterial | Ghitorni → IFFCO Chowk |
| Southern Peripheral Road | Arterial | Vatika Chowk → Kherki Daula |
| KMP Expressway (Western Peripheral Expressway) | Expressway | Sidhrawali → Pataudi Chowk |
| Delhi-Mumbai Expressway | Expressway | Sohna Interchange → Nuh |
| NH-352W (Gurugram-Sohna-Alwar Road) | Highway | Sohna → Taoru |
| Old Delhi-Gurgaon Road | Highway | Kapashera Border → Hero Honda Chowk |
| Pataudi Road | Highway | Basai Chowk → Pataudi Chowk |

The last 5 were added specifically to fix a class-imbalance problem in cross-validation — see
"Model evaluation" below.

---

## How it works

### 1. The measured grid

`data/gurugram_bootstrap.csv` is a **complete, zero-gap grid**: 13 corridors × 7 days × 24 hours
= 2,184 rows, one per (corridor, day-of-week, hour) combination, swept once via `bootstrap_collect.py`.
This is TomTom's own historical-average model for each cell — real, but date-insensitive (it
returns identical numbers for six different future Fridays at 18:00, including a festival week).

### 2. Live collection on top of it

`collect_live.py` polls the same Routing API in real time and derives an **observed**
congestion index from the live travel time (`congestion_idx = 1 − noTraffic/live`, tagged
`source="observed"` since it's computed from live time rather than TomTom's historical average).
Each row also carries:
- **Weather** (`weather.py`, via [Open-Meteo](https://open-meteo.com/), free/no key) — rain,
  trailing 3-hour rainfall, visibility, temperature.
- **Calendar** — Indian holidays (`holidays.India(subdiv="HR")`), festival periods, month-end.
- **Incidents** (`incidents.py`, TomTom Traffic Incidents API) — matched to a corridor if within
  300 m of its real digitized polyline (`frontend/corridors.geojson`), a buffer chosen empirically
  from real incident-distance data.

This runs autonomously on **GitHub Actions**: an hourly scheduled job (`.github/workflows/collect.yml`)
that internally loops ~4 rounds, 15 minutes apart, before exiting — because GitHub's own scheduler
is unreliable at short (15-minute) cron intervals but reliable at hourly ones, so the fine-grained
timing is done inside the job instead of relying on the outer trigger. That's ~96 rounds/day ×
14 requests/round (13 routing + 1 incidents bbox call) ≈ **1,344 requests/day**, against TomTom's
2,500/day free-tier cap. Scheduled workflows are still best-effort (GitHub can delay them, and
disables them after 60 days of repo inactivity) — for a guaranteed cadence, run
`python collect_live.py --loop` on a small always-on machine instead.

### 3. The API

`backend/app.py` (Flask) precomputes the full 13×7×24 grid once at startup and serves it from
memory. Critically, it serves **measured** values first — the observed CSV overwrites a cell if
a real observation exists for it — and only calls the trained model to fill cells that have never
been measured at all (currently zero of 2,184). If no model is loaded, model-backed endpoints
return HTTP 503 rather than inventing a number; there is no hand-typed fallback table. 79 tests
in `backend/test_api.py` currently pass. Full endpoint contract: [`docs/api_contract.md`](docs/api_contract.md).

### 4. The static bundle

`tools/build_static_bundle.py` imports `backend/app.py` as a module and dumps its precomputed
grid to `frontend/data/bundle.json`, so the exact same numbers the live API would serve are
available with zero server — this is what the GitHub Pages deployment reads.

### 5. The forecasting model (gated)

`model/forecast_model.py` is the product model: instead of relearning the whole diurnal curve,
it predicts the **residual** — how far actual conditions deviate from the bootstrap baseline — as
a function of weather, calendar, incidents, corridor, and leak-safe lags from observed history.
The trained artifact is `models/forecast_residual_gbt.joblib` (time holdout; re-run
`model/forecast_model.py train` for current numbers). As of the last training run on this machine:
holdout **skill = 0.6125** (baseline MAE 0.116 vs GBT MAE 0.045; MLP was worse and not shipped).

**The weekly map, advice, best-time, and `/now` all serve this forecast**
(`clip(baseline + predicted_residual, 0, 1)`, origin `residual_adjusted`). The observed CSV is
kept for training, lags, and evaluation only — it is **not** replayed as the displayed congestion
value. If the residual model is missing or holdout skill ≤ 0, model-backed endpoints return HTTP 503.

`tools/build_static_bundle.py` snapshots the same forecast grid into `frontend/data/bundle.json`
for GitHub Pages (weather for the next occurrence of each weekday, live incidents for today only).

---

## Accuracy — bootstrap vs. live observations (diagnostic)

`docs/accuracy_report.md` (regenerated by `tools/evaluate_accuracy.py`) compares the TomTom
**historical bootstrap** against **live TomTom observations** for the same (corridor, day-of-week,
hour) bucket. As of the last run (**2026-09-18**): **15,234 observed rows**, **2,184 / 2,184
cells covered (100%)**. This is a **lookup diagnostic** — how well the date-blind TomTom typical
pattern matches what was measured — **not** the residual forecast the map now shows, and **not** a
GradientBoosting ranking-model score.

**The headline number is ranking accuracy, because ranking is the product.** The site's actual
promise is comparative — "leave at this hour, not that one" — not "the congestion index will be
exactly 0.237."

- **Hour-ranking concordance: 57.2%** (23,704 comparable hour-pairs across 91 corridor/day
  groups). When the bootstrap says hour A is better than hour B, live observations agree 57.2% of
  the time.
- **Best-hour hit rate: 13.2%**; **worst-hour hit rate: 0.0%**.
- **Exact label match: 43.0%** (n=15,234, 95% CI 42.2%–43.7%) — the weaker, more honest number.
  The site's `Free`/`Moderate`/`Heavy`/`Severe` label matches the observed label a little under
  half the time.
- **Point error:** MAE 0.116, bias **+0.003** — near-zero systematic bias (30.8% of mismatches
  understate congestion; 26.3% overstate).

In plain terms: with full grid coverage the bootstrap-vs-observed gap is now visible at scale —
ranking is modestly better than a coin flip, and exact labels are weaker still. Both numbers are
served to the frontend (`GET /health`'s `accuracy` block) rather than hidden, and every
`/now`/`/advice` summary is worded "Typically …" rather than as an unqualified fact.

**Model cross-validation.** Leave-one-corridor-out R² improved from **−2.52 to −0.35** after
expanding from 8 to 13 corridors (the original set was 6 arterial / 1 expressway / 1 highway, so
holding out the sole expressway or highway corridor left the model with zero same-class training
examples). Dwarka Expressway remains a poor fold at **R² = −12.4** even after the fix — real
under-construction, real-estate-corridor traffic that doesn't resemble any other corridor in the
set, not a bug to be fixed by more averaging.

**Forecast holdout eval.** `docs/forecast_eval.md` is regenerated weekly by `.github/workflows/evaluate_forecast.yml` (Mondays ~03:30 UTC, after retrain); trigger manually from Actions → *Evaluate Forecast Model* → *Run workflow*.

---

## Limitations

- **Accuracy is honest, not flattering.** Full 2,184-cell coverage means every figure in
  `docs/accuracy_report.md` is computed at scale (n=15,234), but bootstrap-vs-observed concordance
  is still only ~57% and label agreement ~43% — the product ranks hours better than it pins exact
  severity, and neither is a model CV score.
- **Served values are residual forecasts, not live replays.** Each cell is
  `clip(bootstrap_baseline + predicted_residual, 0, 1)` for the next occurrence of that weekday
  (weather/calendar/incidents/lags applied). It is not a copy of the latest observed CSV row.
- **Exact severity is the weak point.** Expect the label to be right a little under half the time;
  hour ranking concordance is modestly above chance — see "Accuracy" above.
- **Forecast quality is holdout-gated.** Residual model holdout skill must be > 0 to serve at all;
  re-derive from `model/forecast_model.py train` after new data — do not quote stale README figures.
- **GitHub Actions scheduling is best-effort.** Collection can lag or, after 60 days of repo
  inactivity, stop until someone pushes a commit.
- **Dwarka Expressway is a known hard case** for the underlying model (see cross-validation above)
  — real ongoing construction traffic that doesn't generalize from the other corridors.

---

## Project structure

```
Gurugram-traffic-predictor/
├── corridors.py                 # single source of truth: 13 corridors, coordinates, road class
├── bootstrap_collect.py         # one-time sweep: builds the 2,184-cell measured grid
├── collect_live.py              # ongoing live collector (routing + weather + incidents)
├── weather.py                   # Open-Meteo weather + Indian holiday/calendar features
├── incidents.py                 # TomTom Traffic Incidents, matched to corridor geometry
├── run.sh                       # one-command local launcher (backend + frontend)
├── requirements.txt             # training pipeline deps
├── requirements-collect.txt     # collect_live.py-only deps (kept minimal for CI)
│
├── backend/
│   ├── app.py                   # Flask API v2 — precomputed residual-forecast grid
│   ├── requirements.txt
│   └── test_api.py              # 79 tests
│
├── model/
│   └── forecast_model.py        # gated residual (weather/calendar/incident) model
│
├── models/
│   └── forecast_residual_gbt.joblib  # shipped forecast artifact
│
├── data/
│   ├── gurugram_bootstrap.csv   # the complete 2,184-cell measured grid
│   └── gurugram_observed.csv    # ongoing live observations (weather/incident columns attached)
│
├── docs/
│   ├── api_contract.md          # frozen API contract
│   ├── accuracy_report.md       # served-vs-observed accuracy, regenerated from real data
│   ├── forecast_eval.md         # forecast holdout eval, regenerated weekly by CI
│   └── accuracy_history.csv
│
├── tools/
│   ├── build_static_bundle.py   # builds frontend/data/bundle.json from the live grid
│   ├── build_corridor_geojson.py
│   ├── evaluate_accuracy.py     # generates docs/accuracy_report.md
│   └── evaluate_forecast.py     # holdout eval of forecast_residual_gbt.joblib
│
├── frontend/
│   ├── index.html               # MapLibre map + dashboard
│   ├── corridors.geojson        # real digitized corridor polylines
│   ├── data/bundle.json         # static snapshot served on GitHub Pages
│   └── vendor/                  # vendored MapLibre GL JS/CSS
│
├── .github/workflows/
│   ├── collect.yml              # hourly live-collection job
│   ├── refresh_bundle.yml       # daily static-bundle rebuild
│   ├── retrain.yml              # weekly model retrain
│   └── evaluate_forecast.yml    # weekly forecast holdout eval
│
├── PROJECT_EXPLAINER.md         # full technical writeup, file-by-file
└── README.md
```

---

## License

MIT — use freely, attribution appreciated. See [`LICENSE`](LICENSE).

---

*Built as a CS + AI portfolio project exploring urban mobility prediction in the NCR region.*
