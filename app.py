import streamlit as st
import numpy as np
import joblib
import pickle

# Load model
model = joblib.load('stacked_rainfall_model.pkl')

st.title("🌧️ Rainfall Prediction")
st.write("Binary classification to predict rainfall based on weather parameters.")

# Input fields
pressure = st.number_input("Pressure", format="%.2f")
temparature = st.number_input("Temperature", format="%.2f")
dewpoint = st.number_input("Dew Point", format="%.2f")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f")
cloud = st.number_input("Cloud (%)", min_value=0.0, max_value=100.0, format="%.2f")
sunshine = st.number_input("Sunshine (hrs)", format="%.2f")
winddirection = st.number_input("Wind Direction (degrees)", format="%.2f")
windspeed = st.number_input("Wind Speed (km/h)", format="%.2f")

# Predict
if st.button("Predict Rainfall"):
    input_features = [pressure, temparature, dewpoint, humidity,
                      cloud, sunshine, winddirection, windspeed]

    # === Basic Input Validation ===
    if any(x == 0.0 for x in input_features):
        st.error("⚠️ Please make sure all input fields are filled with non-zero values.")
    elif not (850 <= pressure <= 1100):
        st.warning("⚠️ Pressure value seems unrealistic.")
    elif not (-50 <= temparature <= 60):
        st.warning("⚠️ Temperature is out of realistic range.")
    elif not (0 <= dewpoint <= 60):
        st.warning("⚠️ Dew point is out of realistic range.")
    elif not (0 <= sunshine <= 24):
        st.warning("⚠️ Sunshine hours should be between 0 and 24.")
    elif not (0 <= winddirection <= 360):
        st.warning("⚠️ Wind direction must be between 0° and 360°.")
    elif not (0 <= windspeed <= 150):
        st.warning("⚠️ Wind speed seems too high.")
    else:
        # All inputs valid — make prediction
        prediction = model.predict([input_features])[0]

        result = "🌧️ Rain Predicted" if prediction == 1 else "☀️ No Rain Predicted"
        st.subheader(result)
