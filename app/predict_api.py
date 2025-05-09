from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# === Absolute path setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

# === Load model and scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Define input format ===
class CustomerData(BaseModel):
    features: list  # Should contain exactly 30 numeric features

# === Inference endpoint ===
@app.post("/predict")
def predict_churn(data: CustomerData):
    x_input = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x_input)  # If not already scaled
    pred = model.predict(x_scaled)
    return {"churn_prediction": int(pred[0])}

