import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import os
from typing import Any

# Set page config
st.set_page_config(
    page_title="Credit Risk Intelligence Dashboard",
    page_icon="💰",
    layout="wide"
)

# Title and Description
st.title("💰 Credit Risk Intelligence Dashboard")
st.markdown("""
This dashboard provides real-time credit risk scoring and explainability. 
It uses alternative data (transactional behavior) to predict the probability of default.
""")

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

@st.cache_resource
def load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        return model
    except Exception as e:
        st.error(f"Failed to load model from MLflow: {e}")
        return None

model = load_model()

if model:
    st.sidebar.success("Model loaded successfully")
else:
    st.sidebar.error("Model not loaded. Please check MLflow server.")

# Sidebar for Input
st.sidebar.header("Customer Transaction Data")

total_amt = st.sidebar.number_input("Total Transaction Amount", min_value=0.0, value=1500.0, step=10.0)
avg_amt = st.sidebar.number_input("Average Transaction Amount", min_value=0.0, value=150.0, step=1.0)
trans_count = st.sidebar.number_input("Transaction Count", min_value=1, value=10, step=1)
std_amt = st.sidebar.number_input("Std Dev of Transaction Amount", min_value=0.0, value=50.0, step=1.0)

# Main Content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Score")
    
    if model:
        features = np.array([total_amt, avg_amt, trans_count, std_amt]).reshape(1, -1)
        risk_proba = model.predict_proba(features)[0][1]
        
        # Display Gauge or Metric
        st.metric(label="Default Probability", value=f"{risk_proba:.2%}")
        
        if risk_proba < 0.3:
            st.success("Risk Category: LOW")
        elif risk_proba < 0.7:
            st.warning("Risk Category: MEDIUM")
        else:
            st.error("Risk Category: HIGH")
            
        st.info("""
        **Business Recommendation:**
        - **LOW**: Approve credit limit up to 200% of avg monthly spending.
        - **MEDIUM**: Manual review required / Limited credit offer.
        - **HIGH**: Decline application or require collateral.
        """)

with col2:
    st.subheader("Prediction Explainability (SHAP)")
    
    if model:
        # SHAP calculation
        # Note: Local explanation for the single prediction
        explainer = shap.Explainer(model)
        # We need a reference dataset for some explainers, but for Trees/Linear it might be fine
        # For simplicity in this demo, we'll use the input features
        feature_names = ["Total Amt", "Avg Amt", "Count", "Std Dev"]
        
        # To make it work reliably we might need to pass the feature names
        shap_values = explainer(features)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
        st.write("This plot shows how each feature contributed to the final risk score relative to the base value.")

# Batch Prediction Section
st.divider()
st.subheader("Batch Risk Scoring")
uploaded_file = st.file_uploader("Upload CSV with transaction data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Basic check for required columns
    required_cols = ["TotalTransactionAmount", "AvgTransactionAmount", "TransactionCount", "StdTransactionAmount"]
    if all(col in data.columns for col in required_cols):
        predictions = model.predict_proba(data[required_cols])[:, 1]
        data["Risk_Probability"] = predictions
        data["Risk_Category"] = data["Risk_Probability"].apply(lambda x: "High" if x > 0.7 else ("Medium" if x > 0.3 else "Low"))
        
        st.write("Preview of results:")
        st.dataframe(data.head())
        
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scored Data",
            data=csv,
            file_name="scored_customers.csv",
            mime="text/csv",
        )
    else:
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
