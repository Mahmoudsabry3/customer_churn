"""
API routes for the churn prediction system.
"""

import os
from datetime import datetime
from typing import List
import mlflow
import numpy as np

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from ..models.predictor import ChurnPredictor
from ..config.settings import (
    MODEL_PATH, METRICS_PATH, FEATURE_COLUMNS, 
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from .models import (
    UserFeatures, ChurnPrediction, BatchChurnPrediction,
    ModelMetrics, HealthResponse, ModelInfo
)
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global predictor instance
_predictor = None


def get_predictor() -> ChurnPredictor:
    """Get or create the predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor(str(MODEL_PATH))
    return _predictor


@router.get("/", response_class=FileResponse)
async def read_index():
    """Serve the main web interface."""
    return FileResponse("static/index.html")


@router.post("/api/churn/predict", response_model=ChurnPrediction)
async def predict_churn(user_features: UserFeatures, predictor: ChurnPredictor = Depends(get_predictor)):
    """
    Predict churn for a single user based on their features.
    
    Args:
        user_features: User feature data
        predictor: Churn predictor instance
        
    Returns:
        ChurnPrediction: Prediction result
    """
    try:
        # Convert Pydantic model to dict
        input_data = user_features.dict()
        
        # Make prediction
        result = predictor.predict_single(input_data)
        
        # Log prediction to MLflow
        with mlflow.start_run():
            mlflow.log_params(input_data)
            mlflow.log_metrics({
                "prediction": result["churn_prediction"],
                "churn_probability": result["churn_probability"]
            })
        
        return ChurnPrediction(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/churn/batch_predict", response_model=BatchChurnPrediction)
async def batch_predict_churn(
    user_features_list: List[UserFeatures], 
    predictor: ChurnPredictor = Depends(get_predictor)
):
    """
    Predict churn for multiple users based on their features.
    
    Args:
        user_features_list: List of user feature data
        predictor: Churn predictor instance
        
    Returns:
        BatchChurnPrediction: Batch prediction results
    """
    try:
        # Convert list of Pydantic models to list of dicts
        input_data = [user_features.dict() for user_features in user_features_list]
        
        # Make predictions
        results = predictor.predict_batch(input_data)
        
        # Log batch prediction to MLflow
        with mlflow.start_run():
            mlflow.log_param("batch_size", len(user_features_list))
            avg_churn_prob = np.mean([r["churn_probability"] for r in results])
            mlflow.log_metric("avg_churn_probability", avg_churn_prob)
        
        return BatchChurnPrediction(predictions=results)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/churn/health", response_model=HealthResponse)
async def health_check(predictor: ChurnPredictor = Depends(get_predictor)):
    """
    Check the health of the API and model.
    
    Args:
        predictor: Churn predictor instance
        
    Returns:
        HealthResponse: Health status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded(),
        timestamp=datetime.now()
    )


@router.get("/api/churn/model/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """
    Get the performance metrics of the current model.
    
    Returns:
        ModelMetrics: Model performance metrics
    """
    try:
        if not os.path.exists(METRICS_PATH):
            raise HTTPException(status_code=404, detail="Metrics file not found")
        
        metrics = {}
        with open(METRICS_PATH, 'r') as f:
            for line in f:
                if line.startswith("Accuracy:"):
                    metrics["accuracy"] = float(line.split(":")[1].strip())
                elif line.startswith("Precision:"):
                    metrics["precision"] = float(line.split(":")[1].strip())
                elif line.startswith("Recall:"):
                    metrics["recall"] = float(line.split(":")[1].strip())
                elif line.startswith("F1-Score:"):
                    metrics["f1_score"] = float(line.split(":")[1].strip())
                elif line.startswith("Roc Auc:"):
                    metrics["roc_auc"] = float(line.split(":")[1].strip())
                elif line.startswith("Precision-Recall Auc:"):
                    metrics["precision_recall_auc"] = float(line.split(":")[1].strip())
        
        return ModelMetrics(**metrics)
    
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/churn/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current model.
    
    Returns:
        ModelInfo: Model information
    """
    try:
        return ModelInfo(
            model_type="XGBoost Classifier",
            features=FEATURE_COLUMNS,
            target="churn",
            model_path=str(MODEL_PATH),
            last_updated=datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/churn/feature_importance")
async def get_feature_importance(predictor: ChurnPredictor = Depends(get_predictor)):
    """
    Get feature importance from the trained model.
    
    Args:
        predictor: Churn predictor instance
        
    Returns:
        dict: Feature importance scores
    """
    try:
        importance = predictor.get_feature_importance()
        if not importance:
            raise HTTPException(status_code=404, detail="Feature importance not available")
        
        return {"feature_importance": importance}
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=400, detail=str(e))
