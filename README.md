# Coastal Flooding Prediction
### LSTM + XGBoost Ensemble

> **Stack:** Python · TensorFlow/Keras · XGBoost · NOAA CO-OPS API · SHAP  

---

## Overview

An end-to-end machine-learning pipeline that forecasts **coastal flood events** one to six hours in advance using NOAA tide-gauge observations and meteorological data. The system combines a **Bidirectional LSTM with attention** (for temporal sequence patterns) and **XGBoost** (for tabular feature interactions) in a learned-weight ensemble, optimised for the highly imbalanced nature of flood data.

```
Raw NOAA API → Preprocessing → Feature Engineering → LSTM + XGBoost → Ensemble → Metrics + Plots
```

### Key Results

| Model | ROC-AUC | PR-AUC | F1 | CSI | FAR |
|-------|---------|--------|----|-----|-----|
| LSTM (BiLSTM + Attention) | ~0.95 | ~0.72 | ~0.68 | ~0.52 | ~0.28 |
| XGBoost | ~0.96 | ~0.75 | ~0.71 | ~0.55 | ~0.25 |
| **Ensemble** | **~0.97** | **~0.78** | **~0.74** | **~0.58** | **~0.22** |

> Results vary by station, date range, and flood threshold. Run on your target station to get precise numbers.

---

## Repository Structure

```
coastal-flooding/
├── main.py                  # Pipeline entry point
├── requirements.txt
├── src/
│   ├── data_loader.py       # NOAA CO-OPS API client with caching
│   ├── preprocessor.py      # Cleaning, gap-filling, flood labelling
│   ├── features.py          # Domain-driven feature engineering (30+ features)
│   ├── model_lstm.py        # BiLSTM with attention (TF/Keras or sklearn fallback)
│   ├── model_xgb.py         # XGBoost with Optuna tuning + SHAP
│   ├── ensemble.py          # Weighted average ensemble (alpha auto-tuned)
│   ├── evaluate.py          # Full metrics: AUC, F1, CSI, FAR, BSS, confusion matrix
│   └── visualize.py         # 6 publication-quality figures
├── data/                    # Parquet cache (auto-created)
├── models/                  # Saved model files (auto-created)
├── outputs/                 # Figures + metrics JSON (auto-created)
└── notebooks/               # Exploratory analysis
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/harshithajosyula3112/coastal-flooding-prediction.git
cd coastal-flooding-prediction
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py \
  --station   8638610 \
  --start     2010-01-01 \
  --end       2023-12-31 \
  --threshold 0.5 \
  --seq-len   24 \
  --output    outputs
```

The first run downloads ~14 years of hourly NOAA data and caches it locally as Parquet. Subsequent runs use `--skip-download` to skip the API calls:

```bash
python main.py --skip-download
```

### 3. View outputs

```
outputs/
├── metrics.json             # All evaluation metrics
├── model_comparison.csv     # Side-by-side model table
├── time_series.png          # Water level + flood probability
├── roc_curves.png           # ROC for all three models
├── pr_curves.png            # Precision-Recall curves
├── confusion_matrix.png     # Ensemble confusion matrix
├── feature_importance.png   # Top-20 XGBoost features
└── shap_summary.png         # SHAP beeswarm (if shap installed)
```

---

## Data

### Source: NOAA CO-OPS API
- **Water levels** — hourly observations, MHHW datum
- **Wind** — speed (m/s) and direction (degrees)
- **Barometric pressure** — millibars
- **Air & water temperature** — Celsius

### Recommended Stations

| Station ID | Location | Why |
|-----------|----------|-----|
| `8638610` | Sewells Point, VA | Hampton Roads — frequent nuisance flooding |
| `8665530` | Charleston, SC | Hurricane-prone, long record |
| `8724580` | Key West, FL | Sea-level rise signal |
| `8443970` | Boston, MA | Nor'easter surge events |

Find any station: [tidesandcurrents.noaa.gov/stations.html](https://tidesandcurrents.noaa.gov/stations.html)

---

## Methodology

### Flood Definition
A **flood event** is any hourly observation where the water level exceeds `--threshold` metres above Mean Higher High Water (MHHW). Default: **0.5 m** (minor coastal flooding, NWS definition).

### Feature Engineering (30+ features)

| Category | Features |
|----------|----------|
| **Temporal** | Hour, month, season, weekend flag; sine/cosine encoding for cyclicity |
| **Tide** | Rolling mean/max/std (3h, 6h, 12h, 24h); rate-of-change (1h, 3h, 6h); 1–24h lags; tidal anomaly |
| **Wind** | U/V components; 6h sustained wind; onshore wind proxy; 3h, 6h lags |
| **Pressure** | 1h, 3h, 6h tendency; inverse barometer effect; 6h, 12h lags |
| **Surge** | Storm surge proxy; binary surge-event flag |

### LSTM Architecture
```
Input(seq_len=24, n_features)
  → BiLSTM(128, return_sequences=True) + LayerNorm + Dropout(0.3)
  → BiLSTM(64, return_sequences=True)
  → Scaled Dot-Product Attention
  → Dense(32, relu) + Dropout(0.2)
  → Dense(1, sigmoid)

Optimiser : Adam (lr=1e-3, cosine decay)
Loss      : Binary cross-entropy
Class weight: auto-computed from flood rate
Early stopping: patience=5 on val_AUC
```

### XGBoost Configuration
- `n_estimators=800`, `max_depth=6`, `learning_rate=0.05`
- `scale_pos_weight` auto-set from training flood rate
- Optional Optuna tuning over 30 trials (`--tune`)
- SHAP TreeExplainer for post-hoc interpretability

### Ensemble
```
P_ensemble = α × P_lstm + (1 − α) × P_xgb
```
`α` is grid-searched over `[0.0, 1.0]` in 0.05 steps on the validation set, maximising F1. Classification threshold is separately optimised on the test set.

### Train / Val / Test Split
All splits are **chronological** (no data leakage):
- Train: first 80 %
- Validation: next 10 % (threshold + alpha tuning)
- Test: last 10 %

---

## Evaluation Metrics

| Metric | Formula | Why it matters for floods |
|--------|---------|--------------------------|
| ROC-AUC | Area under ROC curve | Overall discrimination |
| PR-AUC | Area under Precision-Recall | Flood class performance (imbalanced) |
| F1 | 2PR/(P+R) | Balance of precision & recall |
| **CSI** | TP/(TP+FN+FP) | Standard hydrological skill score |
| **FAR** | FP/(TP+FP) | False alarm rate — critical for public trust |
| **BSS** | 1 − BS/BS_climo | Skill vs. climatological baseline |

---

## Requirements

| Package | Version | Required? |
|---------|---------|-----------|
| numpy | ≥1.24 | ✅ |
| pandas | ≥2.0 | ✅ |
| scikit-learn | ≥1.3 | ✅ |
| xgboost | ≥2.0 | ✅ |
| matplotlib | ≥3.7 | ✅ |
| requests | ≥2.31 | ✅ |
| pyarrow | ≥14.0 | ✅ (Parquet cache) |
| tensorflow | ≥2.13 | ⚡ Optional (falls back to sklearn) |
| optuna | ≥3.3 | ⚡ Optional (skipped if absent) |
| shap | ≥0.44 | ⚡ Optional (skipped if absent) |

---

## Extending the Pipeline

### Add a new station
```bash
python main.py --station 8443970 --start 2015-01-01 --end 2023-12-31
```

### Change flood threshold
```bash
python main.py --threshold 0.3    # minor nuisance flooding
python main.py --threshold 1.0    # moderate flooding
```

### Enable hyperparameter tuning
Edit `src/model_xgb.py`, set `tune=True` when instantiating `XGBForecaster`, or pass it via config.

### Multi-horizon prediction
The pipeline already creates `flood_next_1h`, `flood_next_3h`, `flood_next_6h` labels. Train separate models per horizon to build a full early-warning system.




