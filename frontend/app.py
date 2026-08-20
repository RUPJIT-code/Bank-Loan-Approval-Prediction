import requests
import joblib
import pandas as pd
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# ----------------------------------------------------------------------
# Load model + scaler (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

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

# Header
st.title("🏦 Bank Loan Approval Predictor")
st.write(
    "Fill in the applicant details below and click **Predict** to find out "
    "whether the loan is likely to be **approved** or **rejected**, based on "
    "a Logistic Regression model trained on historical loan application data."
)

st.divider()

# Input form
with st.form("loan_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with col2:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        credit_history = st.selectbox(
            "Credit History",
            ["Has credit history (1)", "No credit history (0)"],
        )

    st.subheader("Financial Information")

    col3, col4 = st.columns(2)

    with col3:
        applicant_income = st.number_input(
            "Applicant Monthly Income ($)",
            min_value=0,
            value=5000,
            step=100,
        )

        loan_amount = st.number_input(
            "Loan Amount (in thousands, e.g. 128 = $128,000)",
            min_value=0,
            value=128,
            step=1,
        )

    with col4:
        coapplicant_income = st.number_input(
            "Coapplicant Monthly Income ($)",
            min_value=0,
            value=0,
            step=100,
        )

        st.write("Loan Term")

        term_col1, term_col2 = st.columns(2)

        with term_col1:
            loan_term_years = st.selectbox(
                "Years",
                list(range(0, 31)),
                index=30,
            )

        with term_col2:
            loan_term_months = st.selectbox(
                "Months",
                list(range(0, 13)),
                index=0,
            )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True,
    )

# Prediction
if submitted:
    # --- Encode inputs exactly as done in the training notebook ---
    gender_enc = 1 if gender == "Male" else 0            # Female=0, Male=1
    married_enc = 1 if married == "Yes" else 0            # No=0, Yes=1
    education_enc = 0 if education == "Graduate" else 1   # Graduate=0, Not Graduate=1
    self_employed_enc = 1 if self_employed == "Yes" else 0  # No=0, Yes=1
    credit_history_enc = 1 if credit_history.startswith("Has") else 0
    loan_term_total_months = loan_term_years * 12 + loan_term_months

    # one-hot (drop_first=True), baseline = Dependents "0"
    dependents_1 = 1 if dependents == "1" else 0
    dependents_2 = 1 if dependents == "2" else 0
    dependents_3plus = 1 if dependents == "3+" else 0

    # one-hot (drop_first=True), baseline = Property_Area "Rural"
    property_semiurban = 1 if property_area == "Semiurban" else 0
    property_urban = 1 if property_area == "Urban" else 0

    row = {
        "Gender": gender_enc,
        "Married": married_enc,
        "Education": education_enc,
        "Self_Employed": self_employed_enc,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term_total_months,
        "Credit_History": credit_history_enc,
        "Dependents_1": dependents_1,
        "Dependents_2": dependents_2,
        "Dependents_3+": dependents_3plus,
        "Property_Area_Semiurban": property_semiurban,
        "Property_Area_Urban": property_urban,
    }

    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.success(f"✅ **Loan Approved** — confidence: {proba[1]*100:.1f}%")
    else:
        st.error(f"❌ **Loan Rejected** — confidence: {proba[0]*100:.1f}%")

    with st.expander("See prediction probabilities"):
        st.write(
            pd.DataFrame(
                {"Outcome": ["Rejected", "Approved"], "Probability": proba}
            ).set_index("Outcome")
        )

    with st.expander("See the exact feature values sent to the model"):
        st.dataframe(X)

st.divider()
st.caption(
    "Model: scikit-learn LogisticRegression · Trained on historical bank loan "
    "application data · This tool is for demonstration purposes only and is "
    "not financial advice."
)