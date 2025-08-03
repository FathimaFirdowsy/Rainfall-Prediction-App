import numpy as np
import pandas as pd
import joblib

# Load preprocessing objects
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_features.pkl')

def preprocess_input(df):
    df = df.copy()

    # 1. Log transform windspeed
    df['windspeed_log'] = np.log1p(df['windspeed'])

    # 2. Yeo-Johnson replacements — already power transformed
    df['dewpoint_yeo'] = df['dewpoint']
    df['cloud_yeo'] = df['cloud']

    # 3. Wind direction → radians → sin, cos
    df['winddirection_rad'] = np.deg2rad(df['winddirection'])
    df['winddirection_sin'] = np.sin(df['winddirection_rad'])
    df['winddirection_cos'] = np.cos(df['winddirection_rad'])

    # 4. Drop raw columns not used by model
    df.drop(columns=['winddirection', 'winddirection_rad', 'windspeed', 'dewpoint', 'cloud'], inplace=True)

    # 5. Polynomial features
    features_for_poly = ['sunshine', 'dewpoint_yeo', 'cloud_yeo']
    poly_transformed = poly.transform(df[features_for_poly])
    poly_feature_names = poly.get_feature_names_out(features_for_poly)

    # Replace space with underscore to match training
    poly_feature_names = [name.replace(' ', '_') for name in poly_feature_names]
    poly_df = pd.DataFrame(poly_transformed, columns=poly_feature_names, index=df.index)

    # Retain only selected poly features
    expected_poly_features = [
        'sunshine^2',
        'sunshine_dewpoint_yeo',
        'sunshine_cloud_yeo',
        'dewpoint_yeo^2',
        'dewpoint_yeo_cloud_yeo',
        'cloud_yeo^2'
    ]
    df.drop(columns=features_for_poly, inplace=True)
    df = pd.concat([df, poly_df[expected_poly_features]], axis=1)

    # 6. Scale features using fitted scaler
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    return df_scaled


