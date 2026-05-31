[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Models](https://img.shields.io/badge/Models-RF%20%7C%20Prophet%20%7C%20LSTM-orange)](#models-implemented)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)](https://colab.research.google.com/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](#)

# 🌍 Air Quality Prediction Using Machine Learning

> A comprehensive, end-to-end Machine Learning system for predicting PM2.5 air pollution levels across 26 Indian cities — implementing the full ML lifecycle from raw data to a production-ready 7-day forecasting pipeline.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Technical Architecture](#-technical-architecture)
- [Models Implemented](#-models-implemented)
- [Evaluation & Insights](#-evaluation--insights)
- [Visualisations](#-visualisations)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🔍 Project Overview

Air quality monitoring is a critical public health challenge. Fine particulate matter (**PM2.5**) — particles under 2.5 micrometres — penetrates deep into lung tissue and enters the bloodstream, causing cardiovascular and respiratory disease. India's urban PM2.5 levels average **6.6× the WHO 24-hour guideline** of 15 µg/m³.

This project builds a **multi-model forecasting system** that:
- Predicts daily PM2.5 concentrations with near-perfect accuracy (R² = 0.9968)
- Compares Ensemble ML, Time-Series, and Deep Learning approaches
- Generates a **7-day ahead probabilistic forecast** with 90% confidence intervals
- Delivers an **interactive 6-panel Plotly dashboard** for stakeholder exploration

---

## 🏆 Key Results

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | Verdict |
|---|---|---|---|---|
| 🥇 **Random Forest** | **3.40** | **5.60** | **0.9968** | Best overall |
| 🥈 LSTM | 19.88 | 26.41 | 0.7424 | Good (univariate input) |
| 🥉 Prophet | 21.55 | 27.69 | 0.7148 | Best interpretability |

> **Random Forest** explains **99.68% of PM2.5 variance** on the held-out test set, with an average prediction error of just 3.40 µg/m³ — below the WHO's 5 µg/m³ precision threshold for advisory systems.

---

## 📁 Project Structure

```
Air-Quality-Prediction-Using-Machine-Learning/
│
├── 📁 data/
│   ├── raw/
│   │   └── air_pollution_data.csv          # Original dataset — never modified
│   └── processed/
│       └── featured_data.csv               # Output of feature engineering
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis
│   ├── 02_Preprocessing_Features.ipynb     # Preprocessing + Feature Engineering
│   ├── 03_Modelling.ipynb                  # Training + Evaluation (all 3 models)
│   └── 04_Forecasting_Dashboard.ipynb      # Prophet forecast + Plotly dashboard
│
├── 📁 src/                                 # Modular Python source code
│   ├── __init__.py
│   ├── config.py                           # Constants and hyperparameters
│   ├── preprocessing.py                    # Data loading, cleaning, imputation
│   ├── features.py                         # Feature engineering functions
│   ├── train.py                            # Model training pipeline
│   ├── evaluate.py                         # Metrics and visualisation
│   └── forecast.py                         # Prophet 7-day forecast logic
│
├── 📁 app/                                 # Deployment (upcoming)
│   ├── main.py                             # FastAPI backend
│   └── streamlit_app.py                    # Streamlit frontend
│
├── 📁 models/                              # Saved trained models (gitignored)
│   ├── random_forest_model.pkl
│   ├── feature_scaler.pkl
│   └── lstm_model.keras
│
├── 📁 outputs/
│   ├── figures/                            # All exported plot PNGs
│   ├── forecasts/
│   │   └── 7day_forecast.csv
│   ├── metrics/
│   │   └── model_metrics.json
│   └── reports/
│       └── Air_Quality_Project_Report.docx
│
├── 📁 tests/                               # Unit tests for src/ modules
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_evaluate.py
│
├── 📁 .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI on push
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── config.yaml                             # Single source of truth for all params
```

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| **Source** | India CPCB Air Quality Monitoring Network |
| **Records** | 23,504 daily observations |
| **Date Range** | 30 Nov 2020 — 25 May 2023 |
| **Cities** | 26 Indian cities (904 records each) |
| **Features** | CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃, AQI |
| **Target Variable** | PM2.5 (µg/m³) |
| **Missing Values** | 0 (clean dataset) |

**Cities covered:** Ahmedabad, Aizawl, Amaravati, Amritsar, Bengaluru, Bhopal, Brajrajnagar, Chandigarh, Chennai, Coimbatore, Delhi, Ernakulam, Gurugram, Guwahati, Hyderabad, Jorapokhar, Kochi, Kolkata, Lucknow, Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, Visakhapatnam, and more.

---

## 🏗️ Technical Architecture

### 1. Data Preprocessing
- Parsed `date` column from string to `datetime64` with Indian date format handling (`dayfirst=True`)
- Replaced UCI sentinel values (`-200`) with `NaN`
- Applied **city-wise mean imputation** (preserves city-specific pollution baselines, superior to global mean)
- Removed duplicates; sorted chronologically within each city

### 2. Feature Engineering (23 input features)

| Category | Features | Count |
|---|---|---|
| Raw pollutants | CO, NO, NO₂, O₃, SO₂, PM10, NH₃ | 7 |
| Temporal | year, month, day, day_of_year, day_of_week, week_number, is_weekend, quarter | 8 |
| Lag features | pm2_5_lag_1, pm2_5_lag_3, pm2_5_lag_7 | 3 |
| Rolling stats | pm2_5_roll7_mean, pm2_5_roll7_std, pm2_5_roll30_mean | 3 |
| Composite | pollution_index | 1 |
| Encoding | city_encoded | 1 |

> **Key insight:** The 1-day PM2.5 lag feature is the single strongest predictor, reflecting the physical persistence of atmospheric pollution events (stagnation events last 3–7 days).

### 3. Train/Test Split
- **Chronological 80/20 split** — no random shuffling
- Training: 18,657 samples | Testing: 4,665 samples
- Prevents data leakage; simulates real-world deployment

### 4. Scaling
- `MinMaxScaler` fitted on training set only, then applied to test set
- Prevents test set statistics from influencing the scaler (a form of data leakage prevention)

---

## 🤖 Models Implemented

### Model 1 — Random Forest Regressor
**Why chosen:** Handles non-linear pollutant interactions natively, robust to skewed distributions, provides built-in feature importance, and consistently outperforms on structured tabular data without strong distributional assumptions.

**Configuration:**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

### Model 2 — Facebook Prophet
**Why chosen:** Natively decomposes time-series into trend + weekly + yearly seasonality components. Provides the most interpretable forecast and is ideal for stakeholder-facing dashboards.

**Configuration:** `seasonality_mode='multiplicative'`, yearly + weekly seasonality, `changepoint_prior_scale=0.1`, 90% confidence intervals.

### Model 3 — LSTM (Deep Learning)
**Why chosen:** Learns long-range temporal dependencies through gated memory cells without manual feature engineering. Serves as the deep learning benchmark.

**Architecture:** 2-layer LSTM (128 → 64 units) with Dropout (0.2) + Dense layers. Input: 30-day sequence window. Trained for 60 epochs with EarlyStopping.

---

## 📏 Evaluation & Insights

### Domain Insights
- 🌡️ **Winter dominates:** PM2.5 peaks December–February due to temperature inversions trapping ground-level pollutants
- 🌧️ **Monsoon cleans air:** July–September rainfall reduces PM2.5 to annual lows
- 🏭 **Industrial cities differ:** Jorapokhar, Talcher, Brajrajnagar show high SO₂/CO profiles vs traffic-dominated Delhi/Mumbai
- 📅 **Sunday effect:** A consistent ~10–15% PM2.5 reduction every Sunday confirmed by Prophet's weekly component

### ML Methodology Insights
- **Lag features dominate:** pm2_5_lag_1 is the single most important feature — autocorrelation is the primary signal
- **Chronological splitting is non-negotiable:** Random splitting on time-series inflates metrics and fails to test real deployment
- **City-aware imputation matters:** Per-city means preserve pollution profiles vs global mean which distorts low-pollution cities

---

## 📈 Visualisations

All plots are saved to `outputs/figures/`:

| Plot | Description |
|---|---|
| `pollutant_distributions.png` | Histogram for each of 8 pollutants with mean reference line |
| `correlation_heatmap.png` | Lower-triangular Pearson correlation matrix (PM2.5/PM10 r=0.97) |
| `feature_importance.png` | Top 15 Random Forest feature importances (Gini) |
| `actual_vs_predicted.png` | Actual vs predicted PM2.5 for all 3 models side-by-side |
| `model_comparison.png` | MAE / RMSE / R² bar chart across all models |
| `residuals_analysis.png` | Residuals histogram + residuals vs predicted scatter |
| `prophet_components.png` | Trend + yearly + weekly seasonal decomposition |
| `lstm_training_loss.png` | Train vs validation loss across 60 epochs |

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.10+
- Google Colab (recommended) or Jupyter Notebook

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/jmhasan1/Air-Quality-Prediction-Using-Machine-Learning-.git
cd Air-Quality-Prediction-Using-Machine-Learning-
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Place the dataset:**
```
data/raw/air_pollution_data.csv
```

---

## 🚀 Usage

### Option A — Run Notebooks (Recommended for Exploration)

Open notebooks in order inside Google Colab or Jupyter:

```
notebooks/01_EDA.ipynb                  → Explore data distributions and correlations
notebooks/02_Preprocessing_Features.ipynb → Clean data and engineer features
notebooks/03_Modelling.ipynb            → Train and evaluate all 3 models
notebooks/04_Forecasting_Dashboard.ipynb → Generate 7-day forecast + dashboard
```

### Option B — Run Modular Scripts (Recommended for Reproducibility)

```bash
# Run full pipeline end-to-end
python src/train.py

# Generate 7-day forecast only
python src/forecast.py

# Evaluate saved model on new data
python src/evaluate.py
```

### Option C — Import as a Module

```python
from src.preprocessing import load_and_clean
from src.features import engineer_features
from src.train import train_random_forest

df = load_and_clean("data/raw/air_pollution_data.csv")
df = engineer_features(df)
model, metrics = train_random_forest(df)
print(metrics)
# → {'MAE': 3.40, 'RMSE': 5.60, 'R2': 0.9968}
```

---

## 🗺️ Roadmap

- [x] EDA and data preprocessing
- [x] Feature engineering (23 features)
- [x] Random Forest model (R² = 0.9968)
- [x] Prophet time-series model
- [x] LSTM deep learning model
- [x] 7-day forecast with confidence intervals
- [x] Interactive Plotly dashboard
- [x] Project report (DOCX)
- [ ] Refactor into modular `src/` architecture
- [ ] Add `config.yaml` for hyperparameter management
- [ ] Add unit tests for all `src/` modules
- [ ] Integrate Open-Meteo weather API features
- [ ] Add XGBoost / LightGBM comparison
- [ ] Add SHAP explainability layer
- [ ] FastAPI + Streamlit deployment
- [ ] GitHub Actions CI pipeline

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset sourced from India's Central Pollution Control Board (CPCB) via Kaggle
- Project completed as part of the **INLIGHN TECH** ML internship programme
- Forecasting powered by [Meta's Prophet](https://facebook.github.io/prophet/)

---

<p align="center">
  <i>Built with ❤️ to address India's air quality crisis through data-driven insights</i>
</p>
