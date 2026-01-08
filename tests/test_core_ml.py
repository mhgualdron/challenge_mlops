# tests/test_core_ml.py
import pytest
import pandas as pd
import joblib
import os
import numpy as np


def test_model_loading_and_prediction():
    """
    Strict test to ensure the model loads correctly and can make a prediction.
    """
    model_path = "models/model_pipeline.pkl"

    if not os.path.exists(model_path):
        pytest.skip("No model file found. Skipping test.")

    model = joblib.load(model_path)

    # Dummy data to predict
    input_data = pd.DataFrame(
        [
            {
                "CRIM": 0.01,
                "ZN": 18.0,
                "INDUS": 2.31,
                "CHAS": 0.0,
                "NOX": 0.53,
                "RM": 6.5,
                "AGE": 65.2,
                "DIS": 4.0,
                "RAD": 1,
                "TAX": 296,
                "PTRATIO": 15.3,
                "B": 396.9,
                "LSTAT": 4.9,
            }
        ]
    )

    prediction = model.predict(input_data)

    # Assertions
    assert isinstance(prediction[0], (float, np.floating))
    assert prediction[0] > 0
