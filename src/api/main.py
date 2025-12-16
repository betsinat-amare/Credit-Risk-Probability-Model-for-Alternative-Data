import mlflow
import mlflow.sklearn
import numpy as np

from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts credit risk probability using alternative data",
    version="1.0.0"
)

# Load model from MLflow registry
MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "Production"

model = mlflow.sklearn.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)

    risk_proba = model.predict_proba(X)[0][1]

    return PredictionResponse(
        risk_probability=float(risk_proba)
    )
