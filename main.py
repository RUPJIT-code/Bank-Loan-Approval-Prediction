import subprocess
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from schema.user_input import UserInput
from schema.predict_response import PredictionResponse
from models.predict import predict_output, model, scaler, MODEL_VERSION

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:8501")

app = FastAPI(
    title="Bank Loan Approval Prediction API",
    description="Loan Approval Prediction using Logistic Regression",
    version=MODEL_VERSION,
)

streamlit_process = None


@app.get("/")
def root():
    return RedirectResponse(url=FRONTEND_URL)


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.get("/predict")
def predict_info():
    return {
        "message": "This endpoint only accepts POST requests with a JSON body.",
        "how_to_test": "Use the interactive docs at /docs, or send a POST request with applicant data as JSON.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: UserInput):
    try:
        return predict_output(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
def launch_streamlit():
    global streamlit_process
    frontend_path = Path(__file__).resolve().parent / "frontend" / "app.py"
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(frontend_path),
         "--server.headless", "true"]
    )


@app.on_event("shutdown")
def stop_streamlit():
    global streamlit_process
    if streamlit_process:
        streamlit_process.terminate()