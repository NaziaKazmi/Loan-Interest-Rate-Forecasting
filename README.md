# Loan Interest Rate Forecasting Using ANN

A **Deep Learning (ANN) project** to predict loan interest rates based on historical data. This repository includes the trained model, preprocessing pipelines, and multiple interfaces to interact with the model.

## Project Structure

```
LoanInterestApp/
│
├── data/                 # Dataset(s) used for training
├── models/               # Trained ANN model and preprocessing pipelines
├── apps/                 # Interfaces for prediction
│   ├── flask_app.py
│   ├── gradio_app.py
│   ├── fastapi_app.py
│   └── streamlit_app.py
├── notebook/             # Jupyter notebook for training and experimentation
└── requirements.txt      # Python dependencies
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

> ⚠️ Note: Large files such as the trained model or dataset may need to be downloaded separately.

## Contributing

Contributions are welcome! Feel free to fork the repo, add features, or improve documentation.




