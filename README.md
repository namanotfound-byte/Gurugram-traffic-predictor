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
