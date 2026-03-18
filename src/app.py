import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ETA & Delay Prediction",
    layout="wide"
)

st.title(" Smart Delivery ETA & Delay Prediction")

st.markdown(
    """
Predict **delivery delay risk** and **estimated delivery time**
using trained Machine Learning models.
"""
)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

DELAY_MODEL_PATH = "../models/xgboost_Classifier.pkl"
ETA_MODEL_PATH = "../models/xgboost_regressor.pkl"

delay_model = joblib.load(DELAY_MODEL_PATH)
eta_model = joblib.load(ETA_MODEL_PATH)

# ---------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------

st.sidebar.header(" Delivery Details")

delivery_partner = st.sidebar.selectbox(
    "Delivery Partner",
    ["amazon logistics", "dhl", "fedex"]
)

package_type = st.sidebar.selectbox(
    "Package Type",
    ["electronics", "clothing", "automobile parts"]
)

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    ["bike", "ev bike", "van", "truck"]
)

delivery_mode = st.sidebar.selectbox(
    "Delivery Mode",
    ["standard", "same day", "two day"]
)

region = st.sidebar.selectbox(
    "Region",
    ["north", "south", "east", "west", "central"]
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["clear", "cloudy", "rainy", "stormy"]
)

distance_km = st.sidebar.slider("Distance (km)", 1, 500, 50)

package_weight_kg = st.sidebar.slider("Package Weight (kg)", 1, 50, 5)

api_temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 30)

api_humidity = st.sidebar.slider("Humidity (%)", 30, 100, 70)

api_wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 20, 6)

holiday_flag = st.sidebar.selectbox(
    "Holiday / Weekend Transit",
    [0, 1]
)

order_hour = st.sidebar.slider("Order Hour", 0, 23, 12)

order_day = st.sidebar.slider("Order Day (0=Mon)", 0, 6, 2)

is_weekend = st.sidebar.selectbox(
    "Is Weekend?",
    [0, 1]
)

predict_button = st.sidebar.button("Predict Delivery")

# ---------------------------------------------------
# CREATE INPUT DATA
# ---------------------------------------------------

bad_weather_flag = 1 if weather_condition in ["rainy", "stormy"] else 0

input_data = pd.DataFrame({
    "delivery_partner": [delivery_partner],
    "package_type": [package_type],
    "vehicle_type": [vehicle_type],
    "delivery_mode": [delivery_mode],
    "region": [region],
    "weather_condition": [weather_condition],
    "distance_km": [distance_km],
    "package_weight_kg": [package_weight_kg],
    "api_temperature": [api_temperature],
    "api_humidity": [api_humidity],
    "api_wind_speed": [api_wind_speed],
    "bad_weather_flag_api": [bad_weather_flag],
    "holiday_or_weekend_transit_flag": [holiday_flag],
    "order_hour": [order_hour],
    "order_day": [order_day],
    "is_weekend": [is_weekend]
})

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

st.subheader(" Prediction Result")

col1, col2, col3 = st.columns(3)

if predict_button:

    delay_prediction = delay_model.predict(input_data)[0]
    eta_prediction = eta_model.predict(input_data)[0]

    delay_prob = delay_model.predict_proba(input_data)[0][1]

    if delay_prediction == 1:
        delay_label = " Likely Delayed"
    else:
        delay_label = " On Time"

    col1.metric("Estimated ETA (hours)", f"{eta_prediction:.2f}")

    col2.metric("Estimated ETA (minutes)", f"{eta_prediction*60:.0f}")

    col3.metric("Delay Probability", f"{delay_prob*100:.1f}%")

    st.subheader("Delay Prediction")

    if delay_prediction == 1:
        st.error(delay_label)
    else:
        st.success(delay_label)

else:

    col1.metric("Estimated ETA (hours)", "-")
    col2.metric("Estimated ETA (minutes)", "-")
    col3.metric("Delay Probability", "-")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
© 2026 Smart Logistics ETA Prediction System
"""
)