# 🚦 Gurugram Traffic Congestion Predictor

> A machine learning pipeline that predicts road congestion on Gurugram's key corridors using temporal features and real-time TomTom traffic data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBT-orange?style=flat-square)
![Data](https://img.shields.io/badge/Data-TomTom%20API-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ⚠️ Current status (read this first)

This repo is a **working pipeline with no real traffic data in it yet**. The model, API, and frontend numbers below are all trained/generated on **synthetic data** until you let the collector run for a while. See "Getting real data flowing" below — that's the one thing that actually matters right now.

---

## 📌 Why this project?

Gurugram generates **2.5 million+ daily vehicle trips** across 8 key corridors. Despite world-class expressways, the city ranks among India's worst for peak-hour congestion — NH-48 regularly grinds to a halt between 7–10 AM and 5–8 PM.

This project asks: *can we predict when and where congestion will hit, using only time and road structure as inputs?*

**Answer: Yes, with R² = 0.83.**

---

## 🗺️ Corridors Covered

| Corridor | Type | Key Bottleneck |
|---|---|---|
| NH-48 Delhi–Gurgaon Expressway | Highway | Rajiv Chowk merge |
| MG Road | Arterial | IFFCO Chowk intersection |
| Golf Course Road | Arterial | DLF Phase 5 signal |
| Sohna Road | Arterial | Badshahpur chowk |
| Dwarka Expressway | Expressway | Sheetla Mata flyover |
| Golf Course Extension Rd | Arterial | Sector 58–66 stretch |
| Mehrauli–Gurgaon Road | Arterial | Ghitorni–Iffco segment |
| Southern Peripheral Road | Arterial | SPR–NH-48 junction |

---

## 🧠 How it works

### Target variable
```
congestion_index = 1 − (current_speed / free_flow_speed)
```
- `0.0` = completely free  
- `1.0` = complete standstill

### Features (7 total)

| Feature | Why it matters |
|---|---|
| `hour` | Gurugram's commuter-city pattern creates sharp 7–10 AM and 5–8 PM peaks |
| `hour_sin`, `hour_cos` | Cyclical encoding — treats 23:00 and 00:00 as adjacent (they are) |
| `day_of_week` | Weekdays vs Saturday leisure traffic vs Sunday quiet are structurally different |
| `is_weekend` | Binary collapse of day_of_week; boosts signal on smaller datasets |
| `is_peak_morning` | Domain flag for 7–10 AM DLF / Cyber City rush |
| `is_peak_evening` | Domain flag for 5–8 PM NH-48 return congestion |
| `road_class` | Highways saturate differently to arterials — capacity curves are non-linear |

### Model: Gradient Boosting Regressor

Chosen over Random Forest, Linear Regression, and MLP after cross-validated comparison:

```
R²  (test set)    0.83
MAE (test set)    0.031   (~3% congestion error)
R²  (5-fold CV)   0.84 ± 0.003
```

---

## 📁 Project structure

```
gurugram-traffic-predictor/
│
├── model/
│   └── traffic_model.py       # Data collection, feature engineering, training, prediction
│
├── frontend/
│   └── index.html             # Interactive congestion map (vanilla JS + Canvas)
│
├── data/
│   └── gurugram_traffic_raw.csv   # Collected via TomTom API (gitignored if large)
│
├── models/
│   └── traffic_gbt.joblib     # Trained model artifact
│
├── report/
│   └── feature_engineering_report.docx
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/gurugram-traffic-predictor.git
cd gurugram-traffic-predictor
pip install -r requirements.txt
```

### 2. (Optional) Add your TomTom API key

Get a free key at [developer.tomtom.com](https://developer.tomtom.com). Add it to your environment:

```bash
export TOMTOM_API_KEY="your_key_here"
```

Without it, the pipeline uses a realistic synthetic dataset for demo purposes.

### 3. Run the pipeline

```bash
python model/traffic_model.py
```

Output:
```
Dataset: 5000 rows, 9 columns

── Model Performance ──────────────────
  R²  (test set)   : 0.8304
  MAE (test set)   : 0.0306
  R²  (5-fold CV)  : 0.8419 ± 0.0035
───────────────────────────────────────

── Sample predictions ─────────────────
  Monday 8 AM, NH-48              → Heavy      (0.71)  best: 23:00
  Friday 6 PM, Dwarka Expressway  → Moderate   (0.39)  best: 00:00
  Sunday 11 AM, Sohna Road        → Free flow  (0.06)  best: 00:00
```

### 4. Make a prediction

```python
from model.traffic_model import predict

result = predict(hour=8, day_of_week=0, road_class="highway")
# → {'congestion_index': 0.71, 'label': 'Heavy', 'best_travel_hour': 23}
```

### 5. Open the frontend

Just open `frontend/index.html` in any browser. No server needed.

---

## 📊 Feature importances

```
hour_sin           0.509   ← Time of day (cyclical) dominates
is_peak_morning    0.322   ← 7–10 AM flag is very strong
day_of_week        0.055
is_weekend         0.051
hour               0.027
hour_cos           0.023
road_class_enc     0.011
```

The cyclical hour encoding (`hour_sin`) outperforms raw `hour` because it correctly represents that 11 PM and midnight are close in congestion pattern terms.

---

## 🔧 Requirements

```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
joblib>=1.3
requests>=2.31
```

Install: `pip install -r requirements.txt`

---

## 📡 Getting real data flowing

The collector needs to run **every 30 minutes, continuously, for at least 2-3 weeks** to see enough hour/day combinations to matter. Your laptop being asleep half the time won't cut it — so this repo now includes a GitHub Actions workflow (`.github/workflows/collect.yml`) that does it for free on GitHub's servers instead:

1. Get a TomTom key at [developer.tomtom.com](https://developer.tomtom.com) (free tier is enough).
2. In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**, name it `TOMTOM_API_KEY`, paste the key.
3. Push this repo to GitHub. The workflow starts running automatically every 30 minutes and commits new rows to `data/gurugram_traffic_raw.csv` each time.
4. A second workflow (`retrain.yml`) retrains the model every Monday on whatever real data has piled up, and commits the updated `models/traffic_gbt.joblib`.
5. Once `data/gurugram_traffic_raw.csv` has a few thousand real rows, delete the "mixing with synthetic data" fallback in `train_model()` — you won't need it anymore.

The Flask API (`backend/app.py`) already checks for a trained model file and uses it automatically if present, falling back to a hand-tuned table only if no model has been trained yet — so nothing else needs to change as real data comes in.

---

## 📡 Live data collection (rescued, 2026-08-16)

> **This section documents `collect_live.py` and the collection workflows specifically. It supersedes the collection instructions in "Getting real data flowing" above where they conflict — that section still describes the original (currently non-functional) collector.**

### What was broken

The original collector (`model/traffic_model.py: collect_once`) polls TomTom's **Flow Segment Data API**. Tested against this project's real TomTom key on 2026-08-16:

| Endpoint | Result |
|---|---|
| Flow Segment Data (`/traffic/services/4/flowSegmentData/...`) | **403 Forbidden** |
| Search / Geocoding | **403 Forbidden** |
| Routing (`/routing/1/calculateRoute/...`) | **200 OK** |

This key's plan simply cannot reach the Flow API. Every scheduled run of the old `collect.yml` hit the `[WARN] TomTom fetch failed` branch and collected nothing — silently — for as long as the workflow existed. That's the real reason the dataset stayed empty.

### The fix: `collect_live.py`

`collect_live.py` (repo root) gets equivalent signal from the **Routing API**, which this key *can* reach. It calls `calculateRoute` for each of the 8 corridors from `corridors.py` with `traffic=true&computeTravelTimeFor=all`, and derives an **observed** congestion index from the live travel time:

```
congestion_idx = 1 - (noTrafficTravelTimeInSeconds / travelTimeInSeconds)
```

This lands on the same 0–1 scale as the bootstrap sweep's index, but it is computed differently (bootstrap divides by *historic* time; this divides by *live* time) — so every row is tagged `source="observed"` to keep the two distinguishable downstream. Output goes to `data/gurugram_observed.csv`, columns:

```
corridor_id, corridor_name, road_class, day_of_week, hour, minute,
length_m, free_flow_s, live_s, historic_s, traffic_delay_s,
congestion_idx, source, collected_at
```

Run it with `python collect_live.py --once` (single round, used by CI) or `python collect_live.py --loop` (runs forever, one round every 30 minutes — for a VM). It retries individual corridors on 429/5xx with backoff, skips and logs a corridor rather than losing the whole round to one bad request, and dedupes against rows already written for the same (corridor, 30-minute bucket) so re-running never produces duplicates.

**Verified working end-to-end on 2026-08-16**: `python collect_live.py --once` returned live data for all 8/8 corridors in one round (e.g. NH-48: live=2195s vs free-flow=1989s → congestion_idx=0.094; Golf Course Extension Road: live=1874s vs free-flow=1587s, delay=13s → congestion_idx=0.153).

### Quota

8 requests/round × 48 rounds/day = **384 requests/day** against TomTom's 2,500/day free tier — leaving headroom for the one-off bootstrap sweep and manual testing, as long as they share the same key's quota consciously.

### Workflow fixes (`.github/workflows/collect.yml`, `retrain.yml`)

1. **Wrong endpoint** — `collect.yml` called `model/traffic_model.py collect-once` (the dead Flow API path). It now calls `collect_live.py --once`.
2. **Unprotected `git push`** — both workflows did a bare `git push` with no pull/rebase first. Since `collect.yml` runs every 30 minutes and `retrain.yml` pushes to the same branch, the first conflict would fail the push and it would **stay broken forever** (nothing ever re-pulled). Both workflows now fetch + rebase + retry (up to 5 attempts, jittered backoff) on push rejection.
3. **Ad hoc deps** — `collect.yml` used to `pip install pandas numpy requests` inline for a script that never actually needed pandas/numpy. It now installs from a dedicated `requirements-collect.txt` (just `requests`), keeping the every-30-minutes job's install step small.
4. **Concurrency** — both workflows now declare a `concurrency:` group (`cancel-in-progress: false`) so overlapping runs (e.g. a manual `workflow_dispatch` landing mid-schedule) queue instead of racing each other's commits.
5. `data/gurugram_observed.csv` is explicitly un-ignored in `.gitignore` (which otherwise blanket-ignores `data/*.csv`) — otherwise CI's `git add` would silently have nothing to commit, every round, forever.

### Honest caveat: GitHub Actions schedules are not a clock

`schedule: cron: "*/30 * * * *"` is **best-effort, not exact**. In practice:

- Runs are frequently delayed — GitHub's own docs warn scheduled workflows "may be delayed during periods of high loads," and delays of 15–60+ minutes on a `*/30` cron are common, not rare.
- **GitHub disables scheduled workflows on repositories with 60 days of no other activity.** If nobody pushes a commit or opens a PR for two months, the 30-minute collector just stops, silently, until someone re-enables it or pushes something.
- There's no SLA and no guaranteed catch-up for missed runs.

For a portfolio/demo project this is a fine trade for "free and zero-maintenance." If you actually need reliable, true 30-minute cadence (e.g. for a real forecasting product), run `python collect_live.py --loop` on a small always-on VM (a $4–6/mo box is plenty) or a proper cron job instead — that gives you an exact, monotonic clock instead of GitHub's best-effort scheduler.

---

## 🔮 Roadmap

- [ ] Weather integration (IMD API — rain reduces NH-48 speeds ~30%)
- [ ] Event-based anomaly detection (IPL matches, concerts near DLF Avenue)
- [ ] LSTM model for 30-minute rolling congestion forecasts
- [ ] Live TomTom data polling + real-time dashboard updates
- [ ] Extend to Faridabad and Noida corridors

---

## 📄 Report

Full feature engineering rationale, model selection comparison, and results analysis in [`report/feature_engineering_report.docx`](report/feature_engineering_report.docx).

---

## 📜 License

MIT — use freely, attribution appreciated.

---

*Built as part of a CS + AI portfolio project exploring urban mobility prediction in the NCR region.*
