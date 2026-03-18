#  Smart Delivery ETA & Delay Prediction System

An end-to-end Machine Learning pipeline that predicts **delivery delay risk** and **Estimated Time of Arrival (ETA)** using logistics, weather, and temporal features.
The project also includes a **Streamlit dashboard** for interactive predictions.

---

#  Project Overview

Modern logistics systems require accurate delivery predictions to improve customer experience and optimize operations.

This project builds a **two-model ML pipeline**:

1️⃣ **Delay Classification Model**
Predicts whether a delivery will be delayed.

2️⃣ **ETA Regression Model**
Predicts the expected delivery time in hours.

The system integrates:

* Feature engineering
* Holiday enrichment
* Weather enrichment
* Synthetic noise injection
* Model training & evaluation
* Streamlit frontend deployment

---

#  Machine Learning Pipeline

The project follows a complete ML workflow:

```
EDA
↓
Data Cleaning & Preprocessing
↓
Holiday API Feature Engineering
↓
Weather Feature Enrichment
↓
Noise Injection (to simulate real-world variability)
↓
Delay Prediction Model (Classification)
↓
ETA Prediction Model (Regression)
↓
Model Evaluation
↓
Model Saving
↓
Streamlit Dashboard
```

---

#  Models Used

### Classification (Delay Prediction)

| Model                    | Purpose                    |
| ------------------------ | -------------------------- |
| Random Forest Classifier | baseline delay prediction  |
| XGBoost Classifier       | optimized delay prediction |

### Regression (ETA Prediction)

| Model                   | Purpose                  |
| ----------------------- | ------------------------ |
| Random Forest Regressor | baseline ETA prediction  |
| XGBoost Regressor       | optimized ETA prediction |

---

#  Model Performance

### Delay Prediction

| Metric         | Value |
| -------------- | ----- |
| Accuracy       | ~98%  |
| Recall (Delay) | ~0.86 |
| F1 Score       | ~0.82 |

### ETA Prediction

| Metric   | Value                     |
| -------- | ------------------------- |
| MAE      | ~0.37 hours (~22 minutes) |
| RMSE     | ~0.47                     |
| R² Score | ~0.97                     |

The model demonstrates strong predictive performance while maintaining generalization.

---

#  Project Structure

```
ETA-delay-prediction/
│
├── data/
│   └── dataset_with_weather_features.csv
│
├── models/
│   ├── xgboost_Classifier.pkl
│   ├── xgboost_regressor.pkl
│   ├── classification_feature_columns.pkl
│   └── regression_feature_columns.pkl
│
├── notebooks/
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_holiday_api_integration.ipynb
│   ├── 04_weather_enrichment.ipynb
│   ├── 05_model_building_classification.ipynb
│   └── 06_model_building_regression.ipynb
│
├── src/
│   └── app.py        # Streamlit dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🖥 Streamlit Dashboard

The project includes an interactive dashboard where users can input delivery details and obtain predictions.

### Features

* Delivery ETA prediction
* Delay probability prediction
* Interactive input controls
* Clean UI for logistics analytics

### Run the App

```bash
streamlit run src/app.py
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/SinehaManjary28/Noisy-dataset---ETA-.git
cd Noisy-dataset---ETA-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Feature Engineering

Key features used in modeling:

| Feature Category | Examples                                 |
| ---------------- | ---------------------------------------- |
| Logistics        | distance_km, vehicle_type, delivery_mode |
| Package          | package_weight_kg                        |
| Weather          | temperature, humidity, wind_speed        |
| Calendar         | order_hour, order_day, weekend_flag      |
| Operational      | region, delivery_partner                 |

Synthetic noise (9%) was injected into numerical features to simulate real-world variability.

---

#  Visualizations Included

* Confusion Matrix
* ROC Curve
* Actual vs Predicted ETA
* Residual Analysis
* Feature Importance

---

#  Future Improvements

* Real-time traffic data integration
* Live weather API integration
* Route optimization models
* API deployment using FastAPI
* Cloud deployment (AWS / Azure)

---

#  License

This project is for **educational and research purposes**.