import streamlit as st
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# -------------------------------
# Load model + pipeline
# -------------------------------
model = load_model("loan_interest_model.h5", compile=False)
pipeline = joblib.load("pipeline.pkl")   # contains scaler + encoder

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Loan Interest Rate Prediction App")

loan_amnt = st.number_input("Loan Amount", min_value=0)
annual_inc = st.number_input("Annual Income", min_value=0)
fico_low = st.number_input("FICO Range Low", min_value=0)
fico_high = st.number_input("FICO Range High", min_value=0)
dti = st.number_input("DTI (Debt-to-Income Ratio)", min_value=0.0)
delinq_2yrs = st.number_input("Delinquencies (Past 2 years)", min_value=0)
emp_length = st.number_input("Employment Length (Years)", min_value=0)

purpose = st.text_input("Loan Purpose (must match dataset values)")
term = st.text_input("Term (e.g., ' 36 months')")   # Note: leading space is required if in dataset

revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Interest Rate"):

    # Build dataframe with EXACT columns used during training
    input_df = pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'annual_inc': annual_inc,
        'fico_range_low': fico_low,
        'fico_range_high': fico_high,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'emp_length': emp_length,
        'purpose': purpose,
        'term': term,
        'revol_util': revol_util
    }])

    # Transform using the full pipeline
    processed_data = pipeline.transform(input_df)

    # Predict
    prediction = model.predict(processed_data)[0][0]

    st.success(f"Predicted Interest Rate: {prediction:.2f}%")
