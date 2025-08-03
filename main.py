from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel
from typing import List
import mlflow
import mlflow.sklearn
from datetime import datetime

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn_prediction")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="A FastAPI application for predicting customer churn using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = "churn_prediction_model.pkl"
model = joblib.load(model_path)

# Define Pydantic models for request/response
class UserFeatures(BaseModel):
    total_sessions: int
    avg_session_duration: float
    total_songs_played: int
    avg_songs_per_session: float
    thumbs_up_count: int
    thumbs_down_count: int
    add_playlist_count: int
    add_friend_count: int
    time_since_last_activity: int
    days_since_registration: int
    thumbs_up_ratio: float
    thumbs_down_ratio: float
    is_paid_user: int

class ChurnPrediction(BaseModel):
    churn_prediction: int
    churn_probability: float
    no_churn_probability: float

class BatchChurnPrediction(BaseModel):
    predictions: List[dict]

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    precision_recall_auc: float

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/api/churn/predict", response_model=ChurnPrediction)
async def predict_churn(user_features: UserFeatures):
    """
    Predict churn for a single user based on their features.
    """
    try:
        # Convert Pydantic model to DataFrame
        input_data = user_features.dict()
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Log prediction to MLflow
        with mlflow.start_run():
            mlflow.log_params(input_data)
            mlflow.log_metrics({
                "prediction": int(prediction),
                "churn_probability": float(prediction_proba[1])
            })
        
        return ChurnPrediction(
            churn_prediction=int(prediction),
            churn_probability=float(prediction_proba[1]),
            no_churn_probability=float(prediction_proba[0])
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/churn/batch_predict", response_model=BatchChurnPrediction)
async def batch_predict_churn(user_features_list: List[UserFeatures]):
    """
    Predict churn for multiple users based on their features.
    """
    try:
        # Convert list of Pydantic models to DataFrame
        input_data = [user_features.dict() for user_features in user_features_list]
        input_df = pd.DataFrame(input_data)
        
        # Make predictions
        predictions = model.predict(input_df)
        predictions_proba = model.predict_proba(input_df)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            results.append({
                "index": i,
                "churn_prediction": int(pred),
                "churn_probability": float(proba[1]),
                "no_churn_probability": float(proba[0])
            })
        
        # Log batch prediction to MLflow
        with mlflow.start_run():
            mlflow.log_param("batch_size", len(user_features_list))
            mlflow.log_metric("avg_churn_probability", np.mean([r["churn_probability"] for r in results]))
        
        return BatchChurnPrediction(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/churn/health")
async def health_check():
    """
    Check the health of the API and model.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/churn/model/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """
    Get the performance metrics of the current model.
    """
    try:
        metrics_path = "model_evaluation_metrics.txt"
        if os.path.exists(metrics_path):
            metrics = {}
            with open(metrics_path, 'r') as f:
                for line in f:
                    if line.startswith("Accuracy:"):
                        metrics["accuracy"] = float(line.split(":")[1].strip())
                    elif line.startswith("Precision:"):
                        metrics["precision"] = float(line.split(":")[1].strip())
                    elif line.startswith("Recall:"):
                        metrics["recall"] = float(line.split(":")[1].strip())
                    elif line.startswith("F1-Score:"):
                        metrics["f1_score"] = float(line.split(":")[1].strip())
                    elif line.startswith("ROC AUC:"):
                        metrics["roc_auc"] = float(line.split(":")[1].strip())
                    elif line.startswith("Precision-Recall AUC:"):
                        metrics["precision_recall_auc"] = float(line.split(":")[1].strip())
            
            return ModelMetrics(**metrics)
        else:
            raise HTTPException(status_code=404, detail="Metrics file not found")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/churn/model/info")
async def get_model_info():
    """
    Get information about the current model.
    """
    return {
        "model_type": "XGBoost Classifier",
        "features": [
            "total_sessions", "avg_session_duration", "total_songs_played",
            "avg_songs_per_session", "thumbs_up_count", "thumbs_down_count",
            "add_playlist_count", "add_friend_count", "time_since_last_activity",
            "days_since_registration", "thumbs_up_ratio", "thumbs_down_ratio",
            "is_paid_user"
        ],
        "target": "churn",
        "model_path": model_path,
        "last_updated": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

