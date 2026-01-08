# tests/test_api.py
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.main import app, ml_models

client = TestClient(app)


# Mocked successful prediction test
def test_predict_endpoint_mocked_success():
    """
    Unit test of the /predict endpoint with a mocked successful model.
    1. Create a fake model that always returns 25.0
    2. Inject this fake model into the API's global model storage
    """
    fake_model = MagicMock()
    fake_model.predict.return_value = [25.0]  # Mock response

    # Injection of fake model
    # patch.dict replaces the 'ml_models' dict temporarily
    with patch.dict(ml_models, {"model": fake_model}):
        payload = {
            "CRIM": 0.1,
            "ZN": 18.0,
            "INDUS": 2.31,
            "CHAS": 0.0,
            "NOX": 0.538,
            "RM": 6.575,
            "AGE": 65.2,
            "DIS": 4.09,
            "RAD": 1,
            "TAX": 296,
            "PTRATIO": 15.3,
            "B": 396.9,
            "LSTAT": 4.98,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json() == {"predicted_price": 25.0}
        # Verify that the model's predict method was called once
        fake_model.predict.assert_called_once()


# Failure case: Mock of a model that raises an exception
def test_predict_endpoint_handles_error():
    """
    Test the /predict endpoint when the model raises an exception.
    1. Create a fake model that raises an exception on predict
    2. Inject this fake model into the API's global model storage
    3. Verify that the API returns a 400 or 500 error code
    """
    broken_model = MagicMock()
    broken_model.predict.side_effect = Exception("Model failure")

    with patch.dict(ml_models, {"model": broken_model}):
        payload = {
            "CRIM": 0.1,
            "ZN": 18.0,
            "INDUS": 2.31,
            "CHAS": 0.0,
            "NOX": 0.538,
            "RM": 6.575,
            "AGE": 65.2,
            "DIS": 4.09,
            "RAD": 1,
            "TAX": 296,
            "PTRATIO": 15.3,
            "B": 396.9,
            "LSTAT": 4.98,
        }

        response = client.post("/predict", json=payload)

        # Assertions
        assert response.status_code == 400
        assert "Model failure" in response.json()["detail"]
