import pandas as pd
import joblib
import streamlit as st

# Load model and column structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Set page config
st.set_page_config(page_title="Water Pollutants Predictor", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #1f77b4;'> Water Pollutants Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict water pollution levels based on Year and Station ID</p>", unsafe_allow_html=True)
st.markdown("---")

# Input form layout
col1, col2 = st.columns(2)

with col1:
    year_input = st.number_input(" Enter Year", min_value=2000, max_value=2100, value=2022)

with col2:
    station_id = st.text_input("Enter Station ID", value='1')

# Predict button
if st.button(' Predict'):
    if not station_id.strip():
        st.warning('Please enter a valid station ID.')
    else:
        # Prepare input
        input_df = pd.DataFrame({'year': [year_input], 'station_id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['station_id'])

        # Align columns with model
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.markdown("---")
        st.success(f" Predicted pollutant levels for station **{station_id}** in **{year_input}**:")

        if len(predicted_pollutants) != len(pollutants):
            st.error(" Mismatch between model output and expected pollutants.")
        else:
            results_df = pd.DataFrame({
                "Pollutant": pollutants,
                "Predicted Value": [round(val, 2) for val in predicted_pollutants]
            })
            st.table(results_df)

        st.markdown("---")
