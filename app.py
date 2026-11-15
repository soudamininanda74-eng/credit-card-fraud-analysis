import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.title("💳 Credit Card Fraud Detection Portal")
st.write("Enter the transaction details below to verify if the transaction is legitimate or likely fraudulent.")

# Load artifacts safely
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading models. Please ensure 'fraud_rf_model.joblib' and 'scaler.joblib' are in the same folder. Error: {e}")
    st.stop()

# Split input configuration into columns
col1, col2, col3 = st.columns(3)

with col1:
    raw_time = st.number_input("Transaction Time (Seconds)", min_value=0.0, value=100.0, step=1.0)
    raw_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=0.01)
    
with col2:
    st.write("**Principal Component Anonymized Features (V1-V14)**")
    v_inputs = {}
    for i in range(1, 15):
        v_inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0, step=0.1)

with col3:
    st.write("**Principal Component Anonymized Features (V15-V28)**")
    for i in range(15, 29):
        v_inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0, step=0.1)

if st.button("Run Fraud Audit Prediction", type="primary"):
    # Correctly scale the Amount and Time variables
    scaled_amount = scaler.transform(np.array([[raw_amount]]))[0][0]
    # We fit scaled time manually based on how scaler was used during development
    # Note: If time was scaled separately, we use the same scaler or transformation logic:
    scaled_time = scaler.transform(np.array([[raw_time]]))[0][0] # Adjust according to dataset constraints
    
    # Reconstruct exact feature order expected by our random forest classifier
    feature_order = [f'V{i}' for i in range(1, 29)] + ['Scaled_Amount', 'Scaled_Time']
    
    input_dict = {f'V{i}': [v_inputs[f'V{i}']] for i in range(1, 29)}
    input_dict['Scaled_Amount'] = [scaled_amount]
    input_dict['Scaled_Time'] = [scaled_time]
    
    input_df = pd.DataFrame(input_dict)[feature_order]
    
    # Classify
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Audit Verdict:")
    if prediction == 1:
        st.error(f"🚨 ALERT: FRAUD DETECTED (Confidence: {probability * 100:.2f}%)")
    else:
        st.success(f"✅ Transaction Validated (Genuine Probability: {(1 - probability) * 100:.2f}%)")
