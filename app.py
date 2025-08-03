import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_input

# Load trained model
model = joblib.load("stacked_rainfall_model.pkl")

st.title("🌧️ Rainfall Prediction App")
st.write("Enter the weather information to predict rainfall:")

# Input fields
pressure = st.number_input("Pressure (hPa)", min_value=850.0, max_value=1100.0, step=1.0)
temparature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
dewpoint = st.number_input("Dew Point (°C)", min_value=0.0, max_value=60.0)
humidity = st.slider("Humidity (%)", 0, 100, 50)
cloud = st.slider("Cloud Cover (%)", 0, 100, 50)
sunshine = st.slider("Sunshine Hours", 0.0, 24.0, 5.0)
winddirection = st.slider("Wind Direction (°)", 0, 360, 180)
windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, step=0.1)

# Predict
if st.button("Predict Rainfall"):
    input_features = [pressure, temparature, dewpoint, humidity,
                      cloud, sunshine, winddirection, windspeed]

    # === Input Validation ===
    if any(x is None or x == "" for x in input_features):
        st.error("⚠️ Please fill all input fields before predicting.")

    elif not (850 <= pressure <= 1100):
        st.warning("⚠️ Pressure must be between 850 and 1100 hPa.")

    elif not (-50 <= temparature <= 60):
        st.warning("⚠️ Temperature must be between -50°C and 60°C.")

    elif not (0 <= dewpoint <= 60):
        st.warning("⚠️ Dew Point must be between 0°C and 60°C.")

    elif not (0 <= humidity <= 100):
        st.warning("⚠️ Humidity must be between 0% and 100%.")

    elif not (0 <= cloud <= 100):
        st.warning("⚠️ Cloud cover must be between 0% and 100%.")

    elif not (0 <= sunshine <= 24):
        st.warning("⚠️ Sunshine hours must be between 0 and 24.")

    elif not (0 <= winddirection <= 360):
        st.warning("⚠️ Wind direction must be between 0° and 360°.")

    elif not (0 <= windspeed <= 150):
        st.warning("⚠️ Wind speed must be between 0 and 150 km/h.")

    else:
        # All inputs are valid — preprocess and predict
        input_df = pd.DataFrame([{
            'pressure': pressure,
            'temparature': temparature,
            'dewpoint': dewpoint,
            'humidity': humidity,
            'cloud': cloud,
            'sunshine': sunshine,
            'winddirection': winddirection,
            'windspeed': windspeed
        }])

        processed = preprocess_input(input_df)
        prediction = model.predict(processed)[0]

        result = "🌧️ Rain Predicted" if prediction == 1 else "☀️ No Rain Predicted"
        st.subheader(result)


