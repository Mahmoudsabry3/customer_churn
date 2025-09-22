"""
Model prediction module for customer churn prediction.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Handles churn predictions using trained models."""
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model from file."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def validate_input(self, data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        Validate and convert input data to DataFrame.
        
        Args:
            data (Union[Dict, List[Dict]]): Input data for prediction
            
        Returns:
            pd.DataFrame: Validated and formatted data
        """
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = [
            "total_sessions", "avg_session_duration", "total_songs_played",
            "avg_songs_per_session", "thumbs_up_count", "thumbs_down_count",
            "add_playlist_count", "add_friend_count", "time_since_last_activity",
            "days_since_registration", "thumbs_up_ratio", "thumbs_down_ratio",
            "is_paid_user"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure correct data types
        numeric_columns = required_columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for missing values
        if df.isnull().any().any():
            raise ValueError("Input data contains missing values")
        
        return df
    
    def predict_single(self, user_features: Dict) -> Dict[str, Union[int, float]]:
        """
        Predict churn for a single user.
        
        Args:
            user_features (Dict): User feature dictionary
            
        Returns:
            Dict[str, Union[int, float]]: Prediction results
        """
        logger.info("Making single prediction")
        
        # Validate input
        df = self.validate_input(user_features)
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        prediction_proba = self.model.predict_proba(df)[0]
        
        result = {
            "churn_prediction": int(prediction),
            "churn_probability": float(prediction_proba[1]),
            "no_churn_probability": float(prediction_proba[0])
        }
        
        logger.info(f"Prediction completed: {result}")
        return result
    
    def predict_batch(self, user_features_list: List[Dict]) -> List[Dict[str, Union[int, float]]]:
        """
        Predict churn for multiple users.
        
        Args:
            user_features_list (List[Dict]): List of user feature dictionaries
            
        Returns:
            List[Dict[str, Union[int, float]]]: List of prediction results
        """
        logger.info(f"Making batch prediction for {len(user_features_list)} users")
        
        # Validate input
        df = self.validate_input(user_features_list)
        
        # Make predictions
        predictions = self.model.predict(df)
        predictions_proba = self.model.predict_proba(df)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            result = {
                "index": i,
                "churn_prediction": int(pred),
                "churn_probability": float(proba[1]),
                "no_churn_probability": float(proba[0])
            }
            results.append(result)
        
        logger.info(f"Batch prediction completed for {len(results)} users")
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        try:
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
                importances = self.model.named_steps['classifier'].feature_importances_
                
                importance_dict = dict(zip(feature_names, importances))
                return importance_dict
            else:
                logger.warning("Model does not support feature importance")
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


def load_predictor(model_path: str) -> ChurnPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        ChurnPredictor: Loaded predictor instance
    """
    return ChurnPredictor(model_path)
