# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# Pydantic model to define the input data
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

    # Example for documentation
    model_config = {
        "json_schema_extra": {
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
                "LSTAT": 4.98
            }
        }
    }

# Initialize FastAPI app
app = FastAPI(title="Boston Housing Price Predictor", version="1.0")

model = None

# Startup event to load the model
@app.on_event("startup")
def load_model():
    global model
    model_path = "models/model_pipeline.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print("Model loaded successfully into memory")

# Predict endpoint
@app.post("/predict", tags=["Inference"])
def predict_price(features: HousingFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # Try to make prediction
        prediction = model.predict(input_df)
        
        # Convert numpy.floatXX to native Python float for correct serialization
        predicted_price = float(prediction[0])
        
        # Return result rounded to 2 decimal places
        return {"predicted_price": round(predicted_price, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")
    
# Health check endpoint
@app.get("/health", tags=["State"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)