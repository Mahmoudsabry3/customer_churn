"""
Configuration settings for the customer churn prediction system.
"""

import os
from pathlib import Path
from typing import List

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "processed_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "preprocessed_data.csv"
FEATURED_DATA_PATH = DATA_DIR / "features" / "featured_data.csv"

# Model paths
MODEL_PATH = MODELS_DIR / "churn_prediction_model.pkl"
METRICS_PATH = MODELS_DIR / "model_evaluation_metrics.txt"

# MLflow settings
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "churn_prediction"

# Model configuration
MODEL_CONFIG = {
    "model_type": "XGBClassifier",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_splits": 5,
    "use_label_encoder": False
}

# Feature configuration
FEATURE_COLUMNS = [
    "total_sessions",
    "avg_session_duration", 
    "total_songs_played",
    "avg_songs_per_session",
    "thumbs_up_count",
    "thumbs_down_count",
    "add_playlist_count",
    "add_friend_count",
    "time_since_last_activity",
    "days_since_registration",
    "thumbs_up_ratio",
    "thumbs_down_ratio",
    "is_paid_user"
]

TARGET_COLUMN = "churn"
USER_ID_COLUMN = "userId"

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "title": "Customer Churn Prediction API",
    "description": "A FastAPI application for predicting customer churn using machine learning",
    "version": "1.0.0"
}

# CORS settings
CORS_ORIGINS = ["*"]
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
