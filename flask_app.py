from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Load Model & Pipeline
# -------------------------------
model = load_model("loan_interest_model.h5", compile=False)
pipeline = joblib.load("pipeline.pkl")

# -------------------------------
# Initialize Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Home Page (HTML Form)
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get form data
            data = {
                "loan_amnt": float(request.form["loan_amnt"]),
                "annual_inc": float(request.form["annual_inc"]),
                "fico_range_low": float(request.form["fico_range_low"]),
                "fico_range_high": float(request.form["fico_range_high"]),
                "dti": float(request.form["dti"]),
                "delinq_2yrs": int(request.form["delinq_2yrs"]),
                "emp_length": int(request.form["emp_length"]),
                "purpose": request.form["purpose"],
                "term": request.form["term"],
                "revol_util": float(request.form["revol_util"])
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([data])

            # Preprocess input
            processed_input = pipeline.transform(input_df)

            # Predict using Keras model
            prediction = model.predict(processed_input)[0][0]

            # Show prediction in template
            return render_template("index.html", prediction=round(float(prediction), 2))
        except Exception as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")
    
# -------------------------------
# API Endpoint for JSON requests
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        processed_input = pipeline.transform(input_df)
        prediction = model.predict(processed_input)[0][0]
        return jsonify({"predicted_interest_rate": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
