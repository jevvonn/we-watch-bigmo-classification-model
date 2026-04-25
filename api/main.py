from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Maternal Health Risk API")

# Load model once at startup
bundle = joblib.load("model/model.pkl")
model  = bundle["model"]
le     = bundle["label_encoder"]
feature_names = bundle["feature_names"]


class PatientData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float

class PredictionResponse(BaseModel):
    risk_level: str
    probabilities: dict


@app.get("/")
def root():
    return {"message": "Maternal Health Risk Classifier API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    data = pd.DataFrame([patient.model_dump()])
    data = data[feature_names]  # ensure correct column order

    pred_encoded = model.predict(data)[0]
    pred_proba   = model.predict_proba(data)[0]

    risk_label = le.inverse_transform([pred_encoded])[0]
    proba_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(le.classes_, pred_proba)
    }

    return PredictionResponse(
        risk_level=risk_label,
        probabilities=proba_dict
    )


@app.get("/health")
def health():
    return {"status": "ok"}