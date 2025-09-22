"""
Pydantic models for the churn prediction API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime


class UserFeatures(BaseModel):
    """User features for churn prediction."""
    total_sessions: int = Field(..., description="Total number of user sessions")
    avg_session_duration: float = Field(..., description="Average session duration in seconds")
    total_songs_played: int = Field(..., description="Total number of songs played")
    avg_songs_per_session: float = Field(..., description="Average songs per session")
    thumbs_up_count: int = Field(..., description="Number of thumbs up given")
    thumbs_down_count: int = Field(..., description="Number of thumbs down given")
    add_playlist_count: int = Field(..., description="Number of playlist additions")
    add_friend_count: int = Field(..., description="Number of friends added")
    time_since_last_activity: int = Field(..., description="Days since last activity")
    days_since_registration: int = Field(..., description="Days since registration")
    thumbs_up_ratio: float = Field(..., description="Thumbs up ratio")
    thumbs_down_ratio: float = Field(..., description="Thumbs down ratio")
    is_paid_user: int = Field(..., description="Paid user indicator (1=paid, 0=free)")

    class Config:
        schema_extra = {
            "example": {
                "total_sessions": 10,
                "avg_session_duration": 3600.0,
                "total_songs_played": 100,
                "avg_songs_per_session": 10.0,
                "thumbs_up_count": 20,
                "thumbs_down_count": 5,
                "add_playlist_count": 15,
                "add_friend_count": 8,
                "time_since_last_activity": 2,
                "days_since_registration": 30,
                "thumbs_up_ratio": 0.2,
                "thumbs_down_ratio": 0.05,
                "is_paid_user": 1
            }
        }


class ChurnPrediction(BaseModel):
    """Single churn prediction result."""
    churn_prediction: int = Field(..., description="Binary churn prediction (0=no churn, 1=churn)")
    churn_probability: float = Field(..., description="Probability of churn")
    no_churn_probability: float = Field(..., description="Probability of no churn")

    class Config:
        schema_extra = {
            "example": {
                "churn_prediction": 0,
                "churn_probability": 0.23,
                "no_churn_probability": 0.77
            }
        }


class BatchChurnPrediction(BaseModel):
    """Batch churn prediction result."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of prediction results")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "index": 0,
                        "churn_prediction": 0,
                        "churn_probability": 0.23,
                        "no_churn_probability": 0.77
                    }
                ]
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    roc_auc: float = Field(..., description="ROC AUC score")
    precision_recall_auc: float = Field(..., description="Precision-Recall AUC score")

    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.8889,
                "precision": 0.8000,
                "recall": 0.6923,
                "f1_score": 0.7423,
                "roc_auc": 0.9116,
                "precision_recall_auc": 0.7620
            }
        }


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(..., description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2025-09-23T00:18:55.945401"
            }
        }


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str = Field(..., description="Type of model")
    features: List[str] = Field(..., description="List of feature names")
    target: str = Field(..., description="Target variable name")
    model_path: str = Field(..., description="Path to model file")
    last_updated: str = Field(..., description="Model last updated timestamp")

    class Config:
        schema_extra = {
            "example": {
                "model_type": "XGBoost Classifier",
                "features": ["total_sessions", "avg_session_duration", "total_songs_played"],
                "target": "churn",
                "model_path": "churn_prediction_model.pkl",
                "last_updated": "2025-09-23T00:18:39.149136"
            }
        }
