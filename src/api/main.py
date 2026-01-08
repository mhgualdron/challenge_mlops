# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import uvicorn
import os

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "models/model_pipeline.pkl"
    if os.path.exists(model_path):
        ml_models["model"] = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model not found in {model_path}")
        ml_models["model"] = None

    yield

    ml_models.clear()
    print("Models cleared from memory.")


app = FastAPI(title="Boston Housing Price Predictor", version="1.0", lifespan=lifespan)


class HousingFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: int
    PTRATIO: float
    B: float
    LSTAT: float

    class Config:
        json_schema_extra = {
            "example": {
                "CRIM": 0.00632,
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
        }


@app.post("/predict", tags=["Inferencia"])
def predict_price(features: HousingFeatures):
    if ml_models.get("model") is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        prediction = ml_models["model"].predict(input_df)
        return {"predicted_price": round(prediction[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health", tags=["Estado"])
def health_check():
    return {"status": "ok", "model_loaded": ml_models.get("model") is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
