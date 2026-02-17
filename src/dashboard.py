import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import os
from typing import Any, Optional
from src.config import config
from src.data_processing import process_data_end_to_end

# Set page config
st.set_page_config(
    page_title=config.ui.app_title,
    page_icon="💰",
    layout="wide"
)

# Title and Description
st.title(f"💰 {config.ui.app_title}")
st.markdown("""
This dashboard provides real-time credit risk scoring, explainability, and business impact analysis.
It leverages alternative transactional data and Basel II principles for transparent credit decisioning.
""")

# Configuration
MODEL_NAME = config.model.model_name
MODEL_STAGE = config.model.model_stage

@st.cache_resource
def load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        return model
    except Exception as e:
        # Fallback for local development if MLflow is not running
        st.sidebar.warning(f"Could not load production model: {e}")
        return None

@st.cache_data
def get_global_shap_data():
    if not os.path.exists(config.data.raw_data_path):
        return None, None
    df = pd.read_csv(config.data.raw_data_path)
    final_df = process_data_end_to_end(df)
    X = final_df.drop(columns=[config.data.target_col, config.data.customer_id_col])
    return X, df

model = load_model()

if model:
    st.sidebar.success("Production Model Loaded")
else:
    st.sidebar.error("Model not found in registry")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📂 Batch Analysis", "🌍 Global Insights & Impact"])

with tab1:
    st.subheader("Customer Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Input Transaction Data")
        total_amt = st.number_input("Total Transaction Amount", min_value=0.0, value=1500.0, step=10.0)
        avg_amt = st.number_input("Average Transaction Amount", min_value=0.0, value=150.0, step=1.0)
        trans_count = st.number_input("Transaction Count", min_value=1, value=10, step=1)
        std_amt = st.number_input("Std Dev of Transaction Amount", min_value=0.0, value=50.0, step=1.0)
        
        if model:
            features = np.array([total_amt, avg_amt, trans_count, std_amt]).reshape(1, -1)
            risk_proba = model.predict_proba(features)[0][1]
            
            st.metric(label="Default Probability", value=f"{risk_proba:.2%}")
            
            if risk_proba < config.ui.risk_threshold_low:
                st.success("Risk Category: LOW")
            elif risk_proba < config.ui.risk_threshold_high:
                st.warning("Risk Category: MEDIUM")
            else:
                st.error("Risk Category: HIGH")
                
            st.info(f"**Thresholds:** Low < {config.ui.risk_threshold_low*100}%, High > {config.ui.risk_threshold_high*100}%")

    with col2:
        st.subheader("Local Explainability (SHAP)")
        if model:
            explainer = shap.Explainer(model)
            shap_values = explainer(features)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("How features influenced this score")
            st.pyplot(fig)
            st.caption("Waterfall plot showing the contribution of each feature to the final risk score.")

with tab2:
    st.subheader("Bulk Assessment")
    uploaded_file = st.file_uploader("Upload CSV for Scoring", type=["csv"], key="batch")
    
    if uploaded_file is not None and model:
        data = pd.read_csv(uploaded_file)
        required_cols = config.data.numerical_features
        if all(col in data.columns for col in required_cols):
            predictions = model.predict_proba(data[required_cols])[:, 1]
            data["Risk_Probability"] = predictions
            data["Risk_Category"] = data["Risk_Probability"].apply(
                lambda x: "High" if x > config.ui.risk_threshold_high 
                else ("Medium" if x > config.ui.risk_threshold_low else "Low")
            )
            
            st.write("### Processed Results")
            st.dataframe(data.head(20))
            
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Report", data=csv, file_name="risk_report.csv", mime="text/csv")
        else:
            st.error(f"Missing columns: {set(required_cols) - set(data.columns)}")

with tab3:
    st.subheader("Global Model Insights")
    
    col1, col2 = st.columns(2)
    
    X_global, df_raw = get_global_shap_data()
    
    if X_global is not None and model:
        with col1:
            st.write("### Feature Importance (Global)")
            explainer = shap.Explainer(model)
            shap_vals = explainer(X_global)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_vals, X_global, show=False)
            st.pyplot(fig)
            st.write("""
            **Analysis:** 
            - Higher transaction counts generally indicate lower risk (engagement).
            - Large transaction amounts with high standard deviations often signal volatility and higher risk.
            """)
            
        with col2:
            st.write("### Business Impact Simulator")
            loan_amt = st.number_input("Average Loan Amount ($)", value=1000)
            default_cost = st.number_input("Cost per Default ($)", value=1200)
            
            # Simple simulation
            raw_preds = model.predict_proba(X_global)[:, 1]
            high_risk_count = np.sum(raw_preds > config.ui.risk_threshold_high)
            
            total_savings = high_risk_count * default_cost
            
            st.metric("Potential Loss Avoidance", f"${total_savings:,.0f}")
            st.write(f"By rejecting the top {high_risk_count} high-risk applications identified by the model, the institution could potentially save **${total_savings:,.0f}** in default costs.")
            
            st.divider()
            st.write("### Risk Distribution")
            hist_fig, hist_ax = plt.subplots()
            pd.Series(raw_preds).hist(bins=20, ax=hist_ax, color='skyblue', edgecolor='black')
            plt.title("Distribution of Customer Risk Probabilities")
            plt.xlabel("Probability of Default")
            st.pyplot(hist_fig)
    else:
        st.warning("Global insights require the raw data file and a registered model.")
