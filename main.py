from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import os

app = FastAPI(title="Titanic Survival Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define input data model
class TitanicPassenger(BaseModel):
    pclass: int
    sex: int  # 0 for female, 1 for male
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: int  # 0, 1, or 2 for different ports
    family_size: Optional[int] = None
    is_alone: Optional[int] = None
    age_group: Optional[int] = None

# Load models and preprocessing objects
try:
    dt_model = joblib.load('models/decision_tree_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load feature columns
    with open('models/feature_columns.txt', 'r') as f:
        feature_columns = f.read().split('\n')
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    dt_model = None
    lr_model = None
    scaler = None
    feature_columns = None

def preprocess_input(passenger: TitanicPassenger) -> pd.DataFrame:
    """Preprocess input data to match training format"""
    
    # Calculate derived features if not provided
    if passenger.family_size is None:
        passenger.family_size = passenger.sibsp + passenger.parch + 1
    
    if passenger.is_alone is None:
        passenger.is_alone = 1 if passenger.family_size == 1 else 0
    
    if passenger.age_group is None:
        if passenger.age <= 12:
            passenger.age_group = 0
        elif passenger.age <= 18:
            passenger.age_group = 1
        elif passenger.age <= 60:
            passenger.age_group = 2
        else:
            passenger.age_group = 3
    
    # Create DataFrame with all features
    data = {
        'Pclass': passenger.pclass,
        'Sex': passenger.sex,
        'Age': passenger.age,
        'SibSp': passenger.sibsp,
        'Parch': passenger.parch,
        'Fare': passenger.fare,
        'Embarked': passenger.embarked,
        'FamilySize': passenger.family_size,
        'IsAlone': passenger.is_alone,
        'AgeGroup': passenger.age_group
    }
    
    df = pd.DataFrame([data])
    
    # Ensure columns match training data
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training
    df = df[feature_columns]
    
    return df

@app.get("/")
async def root():
    return {"message": "Titanic Survival Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    if dt_model is None or lr_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict/decision-tree")
async def predict_decision_tree(passenger: TitanicPassenger):
    if dt_model is None:
        raise HTTPException(status_code=500, detail="Decision Tree model not loaded")
    
    try:
        # Preprocess input
        df = preprocess_input(passenger)
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = dt_model.predict(scaled_features)[0]
        probability = dt_model.predict_proba(scaled_features)[0].max()
        
        return {
            "model": "Decision Tree",
            "prediction": int(prediction),
            "survived": bool(prediction),
            "probability": float(probability),
            "message": "Survived" if prediction == 1 else "Did not survive"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/logistic-regression")
async def predict_logistic_regression(passenger: TitanicPassenger):
    if lr_model is None:
        raise HTTPException(status_code=500, detail="Logistic Regression model not loaded")
    
    try:
        # Preprocess input
        df = preprocess_input(passenger)
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = lr_model.predict(scaled_features)[0]
        probability = lr_model.predict_proba(scaled_features)[0].max()
        
        return {
            "model": "Logistic Regression",
            "prediction": int(prediction),
            "survived": bool(prediction),
            "probability": float(probability),
            "message": "Survived" if prediction == 1 else "Did not survive"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/both")
async def predict_both(passenger: TitanicPassenger):
    if dt_model is None or lr_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Preprocess input
        df = preprocess_input(passenger)
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make predictions
        dt_pred = dt_model.predict(scaled_features)[0]
        dt_prob = dt_model.predict_proba(scaled_features)[0].max()
        
        lr_pred = lr_model.predict(scaled_features)[0]
        lr_prob = lr_model.predict_proba(scaled_features)[0].max()
        
        return {
            "decision_tree": {
                "prediction": int(dt_pred),
                "survived": bool(dt_pred),
                "probability": float(dt_prob),
                "message": "Survived" if dt_pred == 1 else "Did not survive"
            },
            "logistic_regression": {
                "prediction": int(lr_pred),
                "survived": bool(lr_pred),
                "probability": float(lr_prob),
                "message": "Survived" if lr_pred == 1 else "Did not survive"
            },
            "consensus": {
                "agreement": dt_pred == lr_pred,
                "majority_prediction": int(dt_pred) if dt_pred == lr_pred else "disagreement"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
