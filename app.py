import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🔮 Customer Churn Prediction App")

st.markdown("Enter customer details below to predict churn.")

# Create inputs (30 features)
inputs = []

# Basic binary + numeric fields
inputs.append(st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female"))
inputs.append(st.selectbox("Senior Citizen", [0, 1]))
inputs.append(st.selectbox("Partner", [0, 1]))
inputs.append(st.selectbox("Dependents", [0, 1]))
inputs.append(st.number_input("Tenure (months)", min_value=0, max_value=72, value=12))
inputs.append(st.selectbox("Phone Service", [0, 1]))
inputs.append(st.selectbox("Paperless Billing", [0, 1]))
inputs.append(st.number_input("Monthly Charges", min_value=0.0, value=50.0))
inputs.append(st.number_input("Total Charges", min_value=0.0, value=500.0))

# Remaining one-hot fields (21 dummy values for example)
st.markdown("**One-Hot Encoded Features**")
for i in range(21):
    inputs.append(st.selectbox(f"Feature {i+1}", [0, 1]))

if st.button("Predict Churn"):
    try:
        x_input = np.array(inputs).reshape(1, -1)
        x_scaled = scaler.transform(x_input)
        prediction = model.predict(x_scaled)
        if prediction[0] == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is not likely to churn.")
    except Exception as e:
        st.exception(e)
