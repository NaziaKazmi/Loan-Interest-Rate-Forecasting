import gradio as gr
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and pipeline
model = load_model("loan_interest_model.h5", compile=False)
pipeline = joblib.load("pipeline.pkl")

# Prediction function
def predict_rate(loan_amnt, annual_inc, fico_low, fico_high, dti,
                 delinq_2yrs, emp_length, purpose, term, revol_util):

    input_df = pd.DataFrame([[
        loan_amnt, annual_inc, fico_low, fico_high, dti,
        delinq_2yrs, emp_length, purpose, term, revol_util
    ]], columns=[
        'loan_amnt', 'annual_inc', 'fico_range_low', 'fico_range_high', 'dti',
        'delinq_2yrs', 'emp_length', 'purpose', 'term', 'revol_util'
    ])

    processed = pipeline.transform(input_df)
    processed = np.array(processed).astype("float32")

    prediction = model.predict(processed)[0][0]
    return round(float(prediction), 2)

# Gradio UI
inputs = [
    gr.Number(label="Loan Amount"),
    gr.Number(label="Annual Income"),
    gr.Number(label="FICO Range Low"),
    gr.Number(label="FICO Range High"),
    gr.Number(label="DTI"),
    gr.Number(label="Delinquencies (2yrs)"),
    gr.Number(label="Employment Length"),
    gr.Textbox(label="Purpose"),
    gr.Textbox(label="Term"),
    gr.Number(label="Revolving Utilization")
]

output = gr.Number(label="Predicted Interest Rate (%)")

gr.Interface(fn=predict_rate, inputs=inputs, outputs=output, title="Loan Interest Predictor").launch()
