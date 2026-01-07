# tests/test_api.py
import os
from fastapi.testclient import TestClient
from src.api.main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        
        expected_model_loaded = os.path.exists("models/model_pipeline.pkl")
        assert data["model_loaded"] == expected_model_loaded

def test_predict_valid_payload():
    payload = {
        "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0.0,
        "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09,
        "RAD": 1, "TAX": 296, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98
    }
    
    with TestClient(app) as client:
        if os.path.exists("models/model_pipeline.pkl"):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "predicted_price" in data
            assert isinstance(data["predicted_price"], float)
        else:
            response = client.post("/predict", json=payload)
            assert response.status_code == 500

def test_predict_invalid_payload():
    payload = {"CRIM": "esto no es un número"}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422