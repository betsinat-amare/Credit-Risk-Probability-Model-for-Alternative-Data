import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import numpy as np

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "API is running"

def test_predict_validation_error(client):
    # Missing required fields
    response = client.post("/predict", json={"TotalTransactionAmount": 1000.0})
    assert response.status_code == 422 # Unprocessable Entity

def test_predict_success(client, mocker):
    # Mock the model since we don't want to rely on MLflow in unit tests
    mock_model = mocker.Mock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]]) # 0.2 probability of risk
    
    # Patch the model in main app
    from src.api import main
    main.model = mock_model
    
    payload = {
        "TotalTransactionAmount": 1500.0,
        "AvgTransactionAmount": 150.0,
        "TransactionCount": 10,
        "StdTransactionAmount": 50.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "risk_category" in data
    assert data["risk_probability"] == 0.2
    assert data["risk_category"] == "Low"

def test_predict_high_risk(client, mocker):
    mock_model = mocker.Mock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]]) # 0.8 probability of risk
    
    from src.api import main
    main.model = mock_model
    
    payload = {
        "TotalTransactionAmount": 100.0,
        "AvgTransactionAmount": 10.0,
        "TransactionCount": 2,
        "StdTransactionAmount": 5.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_probability"] == 0.8
    assert data["risk_category"] == "High"
