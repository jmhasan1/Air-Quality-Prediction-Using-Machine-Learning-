This project represents a comprehensive implementation of the Machine Learning Project Lifecycle to address the critical issue of air pollution. By analyzing historical daily measurements of pollutants across various Indian cities, I developed a multi-model forecasting system to predict PM2.5 levels—the most significant indicator of air quality.

# 🌍 Air Quality Prediction Using Machine Learning

This repository features a comprehensive end-to-end Machine Learning pipeline designed to analyze and forecast air pollution levels (PM2.5) across various Indian cities. The project implements the full ML lifecycle, from exploratory data analysis to a production-ready 7-day forecasting system.

---

## Project Overview
Air quality monitoring is essential for public health. This project focuses on predicting **PM2.5**, the most hazardous air quality indicator, by comparing three distinct modeling approaches: Ensemble Learning, Time-Series Forecasting, and Deep Learning.

### Key Features
* **Multi-Model Strategy**: Implements and compares **Random Forest**, **Facebook Prophet**, and **LSTM** architectures.
* **Advanced Feature Engineering**: Leverages temporal features, Indian meteorological seasons, lag features (1, 3, 7 days), and 7-day rolling statistics.
* **Interactive Dashboard**: Features a 6-panel Plotly dashboard for city-wise trends, AQI distributions, and model comparison.
* **7-Day Future Forecast**: Provides a production-ready forecasting pipeline with 90% confidence intervals.

---

## Technical Architecture

### 1. Data Preprocessing
* **Cleaning**: Handled missing values via city-wise mean imputation and replaced -200 sentinel values with NaN.
* **Feature Selection**: Included pollutants (CO, NO, NO2, O3, SO2, PM10, NH3) alongside engineered temporal and lag features.
* **Scaling**: Applied `MinMaxScaler` for normalization, particularly critical for the LSTM model.

### 2. Models Implemented
| Model | Type | Strength |
|---|---|---|
| **Random Forest** | Ensemble | High accuracy on tabular data; provides feature importance. |
| **Prophet** | Time-Series | Captures complex weekly and yearly seasonality natively. |
| **LSTM** | Deep Learning | Learns long-range temporal dependencies using sequence data. |

---

## Evaluation & Insights
The models were evaluated using MAE, RMSE, and $R^2$ scores. 

**Key findings include:**
* **Lag Significance**: Lag features, particularly the 1-day lag, are the strongest predictors for PM2.5 levels.
* **Seasonal Trends**: PM2.5 levels dramatically increase during **Winter** due to atmospheric temperature inversions.
* **Pollutant Correlation**: A strong correlation exists between PM2.5 and PM10.

---

## Setup and Installation

### Prerequisites
* Python 3.10+
* Google Colab (Recommended) or Jupyter Notebook

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/air-quality-prediction.git](https://github.com/your-username/air-quality-prediction.git)