from typing import Literal

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "best_mlp.keras"
SCALER_PATH = "scaler.joblib"

app = FastAPI(title="Diabetes Prediction API", version="1.0")


class PatientInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=17)
    Glucose: float = Field(..., ge=0, le=200)
    BloodPressure: float = Field(..., ge=0, le=122)
    SkinThickness: float = Field(..., ge=0, le=99)
    Insulin: float = Field(..., ge=0, le=846)
    BMI: float = Field(..., ge=0, le=67.1)
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=2.42)
    Age: float = Field(..., ge=21, le=81)


class PredictionResponse(BaseModel):
    probability: float
    prediction: Literal[0, 1]
    risk_level: Literal["LOW", "MODERATE", "HIGH"]


# Load model + scaler once
model = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model from {MODEL_PATH}: {e}")

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler from {SCALER_PATH}: {e}")


@app.get("/")
def root():
    return {
        "service": "Diabetes Prediction API",
        "endpoints": {"POST": "/predict"},
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Keep column order exactly consistent with Streamlit preprocessing
    input_df = pd.DataFrame(
        [
            [
                payload.Pregnancies,
                payload.Glucose,
                payload.BloodPressure,
                payload.SkinThickness,
                payload.Insulin,
                payload.BMI,
                payload.DiabetesPedigreeFunction,
                payload.Age,
            ]
        ],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    )

    try:
        scaled = scaler.transform(input_df)
        proba = float(model.predict(scaled, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    pred = 1 if proba >= 0.5 else 0

    # Same cutoffs as Streamlit UI logic
    if proba >= 0.7:
        risk = "HIGH"
    elif proba >= 0.4:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return {
        "probability": proba,
        "prediction": pred,
        "risk_level": risk,
    }

