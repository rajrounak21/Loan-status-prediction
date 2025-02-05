import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("loan.pkl")

# Streamlit UI
st.title("Loan Status Prediction App")
st.sidebar.write("Enter the loan application details to predict if the loan will be granted.")

# Input fields
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 4])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, step=100)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0, step=100)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0, step=1)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term (in months)", min_value=0, step=1)
Credit_History = st.sidebar.selectbox("Credit History", [0, 1])
Property_Area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Convert categorical inputs
Gender = 1 if Gender == "Male" else 0
Married = 1 if Married == "Yes" else 0
Education = 1 if Education == "Graduate" else 0
Self_Employed = 1 if Self_Employed == "Yes" else 0
Property_Area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[Property_Area]

# Predict button
if st.sidebar.button("Predict Loan Status"):
    input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                            Credit_History, Property_Area]])

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.error("The loan is NOT granted.")
    else:
        st.success("The loan is granted!")
