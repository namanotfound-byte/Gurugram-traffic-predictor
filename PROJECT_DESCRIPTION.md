# Gurugram Traffic Congestion Predictor

**Naman Arora — Class 12**

Repository: https://github.com/namanotfound-byte/Gurugram-traffic-predictor  
Live site: https://gurugram-traffic-predictor.vercel.app

| Item | Fact |
|---|---|
| Use | Tells you when to leave on 13 Gurugram roads. |
| Site | https://gurugram-traffic-predictor.vercel.app |
| Grid | 13 × 7 × 24 = **2,184** slots |
| Data | TomTom Routing. Weather: Open-Meteo. Incidents: TomTom. |
| Period | 16 Aug 2026 – 15 Oct 2026 (61 days) |
| Live rows | 24,490 |
| Trained rows | **20,198** |
| Congestion | `1 − (empty time ÷ expected time)` |
| Prediction | `clip(baseline + residual, 0, 1)` |
| Extra minutes | `empty × index / (1 − index)` |
| Labels | Free &lt; 0.091 · Moderate &lt; 0.200 · Heavy &lt; 0.310 · Severe ≥ 0.310 |
| Model | Gradient boosting of the leftover vs TomTom’s typical hour |
| What it looks at | Hour of day, recent lag, incidents, road class |
| Accuracy | Skill **0.5417** · labels 77.9% · ranking 88.1% (later days) |
| Biggest typical extra | ~13 min (KMP, Monday 08:00 vs 22:00) |
| Stack | Python, Flask, MapLibre · 77 tests passed |
| Limit | A typical week, not this minute. Missing model → HTTP 503. |

Run locally: `./run.sh`. Open the site link, not a saved HTML file.

*Numbers from the live CSV, the 15 Oct 2026 model file, and the evaluation script.*
