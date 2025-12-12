from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# -------------------------------
# Load Model & Pipeline
# -------------------------------
model = load_model("loan_interest_model.h5", compile=False)
pipeline = joblib.load("pipeline.pkl")

# -------------------------------
# FASTAPI App
# -------------------------------
app = FastAPI(title="Loan Interest Rate Prediction API")

# -------------------------------
# Request Body Schema
# -------------------------------
class LoanInput(BaseModel):
    loan_amnt: float
    annual_inc: float
    fico_range_low: float
    fico_range_high: float
    dti: float
    delinq_2yrs: int
    emp_length: int
    purpose: str
    term: str
    revol_util: float

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/predict")
def predict_rate(data: LoanInput):

    # Convert input to DataFrame EXACTLY like training columns
    input_df = pd.DataFrame([{
        "loan_amnt": data.loan_amnt,
        "annual_inc": data.annual_inc,
        "fico_range_low": data.fico_range_low,
        "fico_range_high": data.fico_range_high,
        "dti": data.dti,
        "delinq_2yrs": data.delinq_2yrs,
        "emp_length": data.emp_length,
        "purpose": data.purpose,
        "term": data.term,
        "revol_util": data.revol_util
    }])

    # Process input using pipeline
    processed = pipeline.transform(input_df)

    # Predict using Keras model
    prediction = model.predict(processed)[0][0]

    return {"predicted_interest_rate": round(float(prediction), 2)}
