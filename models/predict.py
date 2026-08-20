from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent

# Load Trained Model
model = joblib.load(MODEL_DIR / "logistic_regression.pkl")
# Load Scaler
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

MODEL_VERSION = "1.0.0"

CLASS_LABELS = {
    0: "Loan Rejected",
    1: "Loan Approved"}

FEATURE_ORDER = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents_1",
    "Dependents_2",
    "Dependents_3+",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]
 

def predict_output(user_input: dict):
    dependents = user_input["dependents"]
    property_area = user_input["property_area"]
    total_months = user_input["loan_term_years"] * 12 + user_input["loan_term_months"]

    row = {
        "Gender": 1 if user_input["Gender"] == "Male" else 0,
        "Married": 1 if user_input["married"] == "Yes" else 0,
        "Education": 0 if user_input["education"] == "Graduated" else 1,
        "Self_Employed": 1 if user_input["self_employed"] == "Yes" else 0,
        "ApplicantIncome": user_input["applicant_income"],
        "CoapplicantIncome": user_input["coapplicant_income"],
        "LoanAmount": user_input["loan_amount"],
        "Loan_Amount_Term": total_months,
        "Credit_History": 1 if user_input["credit_history"].startswith("Has") else 0,
        "Dependents_1": 1 if dependents == "1" else 0,
        "Dependents_2": 1 if dependents == "2" else 0,
        "Dependents_3+": 1 if dependents == "3+" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    # Scale features
    scaled_input = scaler.transform(df)
    # Prediction
    prediction = model.predict(scaled_input)[0]
    # Probabilities
    probabilities = model.predict_proba(scaled_input)[0]
    confidence = float(probabilities[prediction])

    return {
        "prediction": CLASS_LABELS[prediction],
        "confidence": round(confidence, 4),
        "probabilities": {
            "Loan Rejected": round(float(probabilities[0]), 4),
            "Loan Approved": round(float(probabilities[1]), 4),
        }
    }