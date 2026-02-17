import mlflow
import mlflow.sklearn
import numpy as np
import logging
import os
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts credit risk probability using alternative data",
    version="1.1.0"
)

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        logger.info(f"Loading model: {MODEL_NAME}, Stage: {MODEL_STAGE}")
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # In production, we might want to fail hard, but for now we'll log it
        # Raising an error here will prevent the app from starting
        # raise RuntimeError(f"Could not load model: {e}")

def get_risk_category(proba: float) -> str:
    if proba < 0.3:
        return "Low"
    elif proba < 0.7:
        return "Medium"
    else:
        return "High"

@app.get("/")
def health_check():
    return {
        "status": "API is running",
        "model_loaded": model is not None,
        "model_info": {"name": MODEL_NAME, "stage": MODEL_STAGE}
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        # Convert Pydantic model to numpy array for prediction
        # The order depends on what was used during training
        features = [
            request.TotalTransactionAmount,
            request.AvgTransactionAmount,
            request.TransactionCount,
            request.StdTransactionAmount
        ]
        
        X = np.array(features).reshape(1, -1)
        risk_proba = float(model.predict_proba(X)[0][1])
        
        category = get_risk_category(risk_proba)
        
        logger.info(f"Risk prediction: {risk_proba:.4f} -> {category}")
        
        return PredictionResponse(
            risk_probability=risk_proba,
            risk_category=category
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
