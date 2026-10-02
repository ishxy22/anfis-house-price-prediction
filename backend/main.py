from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import joblib
import json
import os
from anfis import ANFIS

app = FastAPI(title="ANFIS House Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
X_scaler = None
y_scaler = None
schema = None
metrics_data = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
EVALUATION_DIR = os.path.join(BASE_DIR, 'evaluation')

class PredictionRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float

@app.on_event("startup")
async def startup_event():
    global model, X_scaler, y_scaler, schema, metrics_data
    
    try:
        schema_path = os.path.join(DATASETS_DIR, 'schema.json')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                
            X_scaler = joblib.load(os.path.join(CHECKPOINTS_DIR, 'X_scaler.pkl'))
            y_scaler = joblib.load(os.path.join(CHECKPOINTS_DIR, 'y_scaler.pkl'))
            
            input_dim = len(schema['features'])
            num_mfs = 3
            model = ANFIS(input_dim=input_dim, num_mfs=num_mfs)
            
            model_path = os.path.join(CHECKPOINTS_DIR, 'anfis_model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            metrics_path = os.path.join(EVALUATION_DIR, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
        else:
            print("Warning: Data not preprocessed yet. Run preprocess.py first.")
            
    except Exception as e:
        print(f"Error during startup: {e}")

@app.post("/predict")
async def predict(req: PredictionRequest):
    if not model or not X_scaler or not y_scaler:
        return {"error": "Model or scalers not loaded. Run preprocess.py first."}
        
    features = np.array([[req.MedInc, req.HouseAge, req.AveRooms]])
    
    scaled_features = X_scaler.transform(features)
    x_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    
    with torch.no_grad():
        output, firing = model(x_tensor)
        
    predicted_scaled = output.numpy()
    predicted_price = float(y_scaler.inverse_transform(predicted_scaled)[0][0])
    
    firing_strengths = firing.numpy()[0]
    
    top_indices = np.argsort(firing_strengths)[-5:][::-1]
    top_rules = [
        {"rule": f"Rule {int(idx)}", "weight": float(firing_strengths[idx])}
        for idx in top_indices
    ]
    
    all_rules = [
        {"rule": f"Rule {int(idx)}", "weight": float(firing_strengths[idx])}
        for idx in range(len(firing_strengths))
    ]
    
    return {
        "predictedPrice": predicted_price,
        "topRules": top_rules,
        "allRules": all_rules
    }

@app.get("/metrics")
async def metrics():
    if metrics_data:
        return metrics_data
    return {
        "error": "Metrics not found. Run training script."
    }

@app.get("/fuzzy-sets")
async def get_fuzzy_sets():
    if not model or not schema:
        return {"error": "Model not loaded."}
        
    mu = model.layer1.mu.detach().numpy()
    sigma = model.layer1.sigma.detach().numpy()
    
    result = {}
    for i, feature in enumerate(schema['features']):
        mfs = []
        for j in range(len(mu[i])):
            mfs.append({"mean": float(mu[i][j]), "sigma": float(sigma[i][j])})
        result[feature] = mfs
    return {"membershipFunctions": result}
