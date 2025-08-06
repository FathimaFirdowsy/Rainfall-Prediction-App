import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and preprocessing objects
model = joblib.load("stacked_model_calibrate.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly_features.pkl")

# === Streamlit App Title ===
st.title("🌦️ Rainfall Prediction App")
st.markdown("Provide weather details to predict whether it will rain.")

# === Input Form ===
with st.form("rainfall_form"):
    pressure = st.number_input("Pressure (hPa)", min_value=850.0, max_value=1100.0, value=1010.0, step=1.0)
    temparature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0, step=0.1)
    dewpoint = st.number_input("Dew Point (°C)", min_value=0.0, max_value=60.0, value=15.0)
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    cloud = st.slider("Cloud Cover (%)", 0, 100, 50)
    sunshine = st.slider("Sunshine Hours", 0.0, 24.0, 6.0)
    winddirection = st.slider("Wind Direction (°)", 0, 360, 180)
    windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0, step=0.1)

    submitted = st.form_submit_button("🔍 Predict Rainfall")

# === Prediction Block ===
if submitted:
    # Input validation
    if not (850 <= pressure <= 1100):
        st.warning("⚠️ Pressure must be between 850 and 1100 hPa.")
    elif not (-50 <= temparature <= 60):
        st.warning("⚠️ Temperature must be between -50°C and 60°C.")
    elif not (0 <= dewpoint <= 60):
        st.warning("⚠️ Dew Point must be between 0°C and 60°C.")
    elif not (0 <= humidity <= 100):
        st.warning("⚠️ Humidity must be between 0% and 100%.")
    elif not (0 <= cloud <= 100):
        st.warning("⚠️ Cloud Cover must be between 0% and 100%.")
    elif not (0 <= sunshine <= 24):
        st.warning("⚠️ Sunshine hours must be between 0 and 24.")
    elif not (0 <= winddirection <= 360):
        st.warning("⚠️ Wind direction must be between 0° and 360°.")
    elif not (0 <= windspeed <= 150):
        st.warning("⚠️ Wind speed must be between 0 and 150 km/h.")
    else:
        # Assemble input into a DataFrame
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

        print("\n--- Raw Input ---")
        print(input_df)

        try:
            # === Preprocessing ===

            # 1. Log transform
            input_df['windspeed_log'] = np.log1p(input_df['windspeed'])

            # 2. Power transform replacements (already transformed in training)
            input_df['dewpoint_yeo'] = input_df['dewpoint']
            input_df['cloud_yeo'] = input_df['cloud']

            # 3. Wind direction sin/cos
            input_df['winddirection_rad'] = np.deg2rad(input_df['winddirection'])
            input_df['winddirection_sin'] = np.sin(input_df['winddirection_rad'])
            input_df['winddirection_cos'] = np.cos(input_df['winddirection_rad'])

            # 4. Drop unused original columns
            input_df.drop(columns=['winddirection', 'winddirection_rad', 'windspeed', 'dewpoint', 'cloud'], inplace=True)

            # 5. Polynomial features
            features_for_poly = ['sunshine', 'dewpoint_yeo', 'cloud_yeo']
            poly_transformed = poly.transform(input_df[features_for_poly])
            poly_feature_names = [name.replace(" ", "_") for name in poly.get_feature_names_out(features_for_poly)]

            poly_df = pd.DataFrame(poly_transformed, columns=poly_feature_names, index=input_df.index)

            expected_poly_features = [
                'sunshine^2',
                'sunshine_dewpoint_yeo',
                'sunshine_cloud_yeo',
                'dewpoint_yeo^2',
                'dewpoint_yeo_cloud_yeo',
                'cloud_yeo^2'
            ]

            input_df.drop(columns=features_for_poly, inplace=True)
            input_df = pd.concat([input_df, poly_df[expected_poly_features]], axis=1)

            print("\n--- After Preprocessing ---")
            print(input_df)

            # 6. Scale
            df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns, index=input_df.index)

            print("\n--- Scaled Data ---")
            print(df_scaled)

            # Predict probability of rainfall
            probas = model.predict_proba(df_scaled)[0]
            rain_probability = probas[1]  # Probability of class '1' = Rain

            # Apply custom threshold
            prediction = 1 if rain_probability > 0.6 else 0

            print(f"\n--- Prediction Output ---\nPredicted Value: {prediction}")
            proba = model.predict_proba(df_scaled)[0]
            print(f"Probability of Rain: {proba[1]:.4f}, No Rain: {proba[0]:.4f}")

            # Output result
            if prediction == 1:
                st.success(f"🌧️ Rain Predicted")
            else:
                st.success(f"☀️ No Rain Expected")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
