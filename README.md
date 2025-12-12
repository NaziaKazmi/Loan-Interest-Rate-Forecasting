# Loan Interest Rate Forecasting Using ANN

A **Deep Learning (ANN) project** to predict loan interest rates based on historical data. This repository includes the trained model, preprocessing pipelines, and multiple interfaces to interact with the model.

## Project Structure

```
LoanInterestApp/
├── models/               # Trained ANN model and preprocessing pipelines
├── apps/                 # Interfaces for prediction
│   ├── flask_app.py
│   ├── gradio_app.py
│   ├── fastapi_app.py
│  └── streamlit_app.py
|___ Images
|___ Templates
```

## Features

- Predict loan interest rates using a fully trained **Artificial Neural Network (ANN)**.
- Multiple deployment options for experimentation:
  - **Flask**
  - **FastAPI**
  - **Gradio**
  - **Streamlit**
- Preprocessing pipelines included to handle new data before prediction.
- Easy-to-use interfaces for interactive predictions.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd LoanInterestApp
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the trained models and preprocessing files are placed inside the `models/` folder.

## How to Run

Run the desired interface:

- **Flask**

```bash
python apps/flask_app.py
```

- **Gradio**

```bash
python apps/gradio_app.py
```

- **FastAPI**

```bash
python apps/fastapi_app.py
```

- **Streamlit**

```bash
streamlit run apps/streamlit_app.py
```
**Dataset**

The full dataset is large (>100MB) and not uploaded to GitHub.
You can download it from this link: https://www.kaggle.com/datasets/wordsforthewise/lending-club

>  Note: Large files such as the trained model may need to be downloaded separately.

## Contributing

Contributions are welcome! Feel free to fork the repo, add features, or improve documentation.

**Author**

Nazia Kazmi




