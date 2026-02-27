import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "models/credit_model.pkl"

print("Loading trained model...")
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Credit Default Prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Make prediction
    probability = model.predict_proba(input_df)[:, 1][0]
    prediction = int(probability > 0.5)

    return {
        "default_probability": round(float(probability), 4),
        "prediction": prediction
    }