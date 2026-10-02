import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from anfis import ANFIS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
EVALUATION_DIR = os.path.join(BASE_DIR, 'evaluation')

def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def train_and_evaluate():
    print("Fetching California Housing Dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    features = ['MedInc', 'HouseAge', 'AveRooms']
    target = 'MedHouseVal'
    
    X = df[features].values
    y = df[[target]].values
    
    print("Scaling features and target...")
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    joblib.dump(X_scaler, os.path.join(CHECKPOINTS_DIR, 'X_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(CHECKPOINTS_DIR, 'y_scaler.pkl'))
    
    schema = {
        "features": features,
        "target": target,
        "feature_ranges": {
            feat: {"min": float(X_scaler.data_min_[i]), "max": float(X_scaler.data_max_[i])} 
            for i, feat in enumerate(features)
        },
        "target_range": {"min": float(y_scaler.data_min_[0]), "max": float(y_scaler.data_max_[0])}
    }
    with open(os.path.join(DATASETS_DIR, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=4)
        
    print("Training Baseline Models...")
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train.ravel())
    lr_pred = lr.predict(X_test)
    lr_metrics = evaluate(y_test, lr_pred)
    
    # MLP Regressor
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train.ravel())
    mlp_pred = mlp.predict(X_test)
    mlp_metrics = evaluate(y_test, mlp_pred)
    
    print("Training PyTorch ANFIS Model...")
    input_dim = len(features)
    num_mfs = 3
    model = ANFIS(input_dim=input_dim, num_mfs=num_mfs)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    epochs = 200
    batch_size = 256
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    # Evaluate ANFIS
    model.eval()
    with torch.no_grad():
        anfis_pred, _ = model(X_test_t)
        anfis_pred_np = anfis_pred.numpy()
        
    anfis_metrics = evaluate(y_test, anfis_pred_np)
    
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, 'anfis_model.pth'))
    
    metrics = {
        "models": {
            "ANFIS": anfis_metrics,
            "MLP": mlp_metrics,
            "LinearRegression": lr_metrics
        }
    }
    
    with open(os.path.join(EVALUATION_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Training complete. Artifacts saved.")

if __name__ == "__main__":
    train_and_evaluate()
