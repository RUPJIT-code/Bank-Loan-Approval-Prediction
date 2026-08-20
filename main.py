from fastapi import FastAPI, HTTPException
from schema.user_input import UserInput
from schema.predict_response import PredictionResponse
from models.predict import (predict_output,model,scaler,MODEL_VERSION)

app = FastAPI(
    title="Bank Loan Approval Prediction API",
    description="Loan Approval Prediction using Logistic Regression",
    version=MODEL_VERSION)

@app.get("/")
def root():
    return {
        "message": "Bank Loan Approval Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: UserInput):
    try:
        return predict_output(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))