import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the necessary artifacts
try:
    with open('gb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('rentang_fitur.pkl', 'rb') as file:
        rentang_fitur = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        encoders = pickle.load(file)
    with open('scaler_fitur.pkl', 'rb') as file:
        scaler_fitur = pickle.load(file)
    with open('target_scaler.pkl', 'rb') as file:
        target_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'rf_model.pkl', 'rentang_fitur.pkl', 'label_encoder.pkl', 'scaler_fitur.pkl', and 'target_scaler.pkl' are in the same directory.")
    st.stop()

def main():
    st.title("House Price Prediction App")
    st.write("Enter the details of the house to get a price prediction.")

    # Numerical Inputs (using sliders based on min/max from rentang_fitur)
    st.header("House Features (Numerical)")

    bedrooms = st.slider(
        "Bedrooms",
        min_value=int(rentang_fitur['bedrooms']['min']),
        max_value=int(rentang_fitur['bedrooms']['max']),
        value=3
    )

    bathrooms = st.slider(
        "Bathrooms",
        min_value=float(rentang_fitur['bathrooms']['min']),
        max_value=float(rentang_fitur['bathrooms']['max']),
        value=2.0,
        step=0.25
    )

    sqft_living = st.slider(
        "Sqft Living",
        min_value=int(rentang_fitur['sqft_living']['min']),
        max_value=int(rentang_fitur['sqft_living']['max']),
        value=1500
    )

    floors = st.slider(
        "Floors",
        min_value=float(rentang_fitur['floors']['min']),
        max_value=float(rentang_fitur['floors']['max']),
        value=1.0,
        step=0.5
    )

    sqft_above = st.slider(
        "Sqft Above",
        min_value=int(rentang_fitur['sqft_above']['min']),
        max_value=int(rentang_fitur['sqft_above']['max']),
        value=1000
    )

    # Categorical Inputs (using selectbox based on unique values from encoders)
    st.header("House Features (Categorical)")

    city_options = list(encoders['city'].classes_)
    city = st.selectbox("City", options=city_options, index=city_options.index('Seattle') if 'Seattle' in city_options else 0)

    statezip_options = list(encoders['statezip'].classes_)
    statezip = st.selectbox("State/Zip", options=statezip_options, index=statezip_options.index('WA 98115') if 'WA 98115' in statezip_options else 0)

    # Prediction Button
    if st.button("Predict Price"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'floors': [floors],
            'sqft_above': [sqft_above],
            'city': [encoders['city'].transform([city])[0]],
            'statezip': [encoders['statezip'].transform([statezip])[0]]
        })

        # Scale the numerical features
        X_scaled_input = scaler_fitur.transform(input_data)

        # Make prediction
        scaled_prediction = model.predict(X_scaled_input)

        # Inverse transform to get original price
        actual_price = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        st.success(f"Predicted House Price: ${actual_price:,.2f}")

if __name__ == "__main__":
    main()
