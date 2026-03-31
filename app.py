import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
try:
    model = pickle.load(open('LinearRegressionModel (1).pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Extract unique values from the model's metadata (based on your file content)
companies = [
    'Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 
    'Hindustan', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Land', 
    'Mahindra', 'Maruti', 'Mercedes', 'Mini', 'Mitsubishi', 
    'Nissan', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
]

fuel_types = ['Petrol', 'Diesel', 'LPG']

# Streamlit UI
st.set_page_config(page_title="Car Price Predictor")
st.title("🚗 Car Price Prediction Model")
st.markdown("Enter the details of the car to estimate its resale value.")

# Form for user input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Select Company", sorted(companies))
        # In a production app, you'd filter 'name' based on 'company'
        car_name = st.text_input("Car Model Name", placeholder="e.g. Maruti Suzuki Swift")
        year = st.number_input("Year of Manufacture", min_value=1900, max_value=2026, value=2015)

    with col2:
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500, value=10000)
    
    submit_button = st.form_submit_button("Predict Price")

if submit_button:
    # Create a DataFrame for the model input
    # Note: Column names must exactly match the names used during model training
    input_data = pd.DataFrame(
        [[car_name, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    try:
        prediction = model.predict(input_data)
        st.success(f"### Predicted Resale Price: ₹{np.round(prediction[0], 2):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure the 'Car Model Name' matches the format used in the training data.")