"""
Model training module for customer churn prediction.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Tuple, Any
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ..config.settings import MODEL_CONFIG, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation for churn prediction."""
    
    def __init__(self):
        self.model_pipeline = None
        self.feature_columns = None
        self.evaluation_metrics = {}
        
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        logger.info("Setting up MLflow tracking")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        logger.info("Preparing data for training")
        
        # Define features and target
        self.feature_columns = [col for col in df.columns if col not in ["userId", target_column]]
        X = df[self.feature_columns]
        y = df[target_column]
        
        logger.info(f"Features: {self.feature_columns}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_model_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Create the model pipeline with preprocessing and classifier.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            
        Returns:
            Pipeline: Model pipeline
        """
        logger.info("Creating model pipeline")
        
        # Identify numerical features
        numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features)
            ]
        )
        
        # Create model pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                objective=MODEL_CONFIG["objective"],
                eval_metric=MODEL_CONFIG["eval_metric"],
                use_label_encoder=MODEL_CONFIG["use_label_encoder"],
                random_state=MODEL_CONFIG["random_state"]
            ))
        ])
        
        return pipeline
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Dict[str, Any]: Evaluation metrics and predictions
        """
        logger.info("Starting k-fold cross-validation")
        
        # Setup cross-validation
        kf = StratifiedKFold(
            n_splits=MODEL_CONFIG["n_splits"],
            shuffle=True,
            random_state=MODEL_CONFIG["random_state"]
        )
        
        # Storage for predictions
        all_y_test = []
        all_y_pred = []
        all_y_proba = []
        
        # Cross-validation loop
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            logger.info(f"Fold {fold+1}/{MODEL_CONFIG['n_splits']}")
            
            # Split data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Create and train model
            model_pipeline = self.create_model_pipeline(X)
            model_pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred_fold = model_pipeline.predict(X_test)
            y_proba_fold = model_pipeline.predict_proba(X_test)[:, 1]
            
            # Store predictions
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred_fold)
            all_y_proba.extend(y_proba_fold)
        
        # Convert to numpy arrays
        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)
        
        return {
            "y_test": all_y_test,
            "y_pred": all_y_pred,
            "y_proba": all_y_proba
        }
    
    def calculate_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Prediction probabilities
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Calculating evaluation metrics")
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # Calculate Precision-Recall AUC
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        metrics["precision_recall_auc"] = auc(rec, prec)
        
        return metrics
    
    def log_to_mlflow(self, metrics: Dict[str, float], cm: np.ndarray) -> str:
        """
        Log training results to MLflow.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
            cm (np.ndarray): Confusion matrix
            
        Returns:
            str: MLflow run ID
        """
        logger.info("Logging results to MLflow")
        
        with mlflow.start_run(run_name=f"churn_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "model_type": MODEL_CONFIG["model_type"],
                "n_splits": MODEL_CONFIG["n_splits"],
                "random_state": MODEL_CONFIG["random_state"],
                "objective": MODEL_CONFIG["objective"],
                "eval_metric": MODEL_CONFIG["eval_metric"]
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix components
            mlflow.log_metrics({
                "true_negative": cm[0, 0],
                "false_positive": cm[0, 1],
                "false_negative": cm[1, 0],
                "true_positive": cm[1, 1]
            })
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run logged successfully! Run ID: {run_id}")
            
        return run_id
    
    def save_model(self, model_pipeline: Pipeline, model_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            model_pipeline (Pipeline): Trained model pipeline
            model_path (str): Path to save the model
        """
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model_pipeline, model_path)
        logger.info("Model saved successfully")
    
    def save_metrics(self, metrics: Dict[str, float], cm: np.ndarray, metrics_path: str) -> None:
        """
        Save evaluation metrics to file.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
            cm (np.ndarray): Confusion matrix
            metrics_path (str): Path to save metrics
        """
        logger.info(f"Saving metrics to {metrics_path}")
        
        with open(metrics_path, "w") as f:
            f.write("--- Model Evaluation Results ---\n")
            for metric, value in metrics.items():
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
            
            f.write("\n--- Confusion Matrix ---\n")
            f.write(str(cm) + "\n")
            f.write(f"True Negative (TN): {cm[0, 0]}\n")
            f.write(f"False Positive (FP): {cm[0, 1]}\n")
            f.write(f"False Negative (FN): {cm[1, 0]}\n")
            f.write(f"True Positive (TP): {cm[1, 1]}\n")
            
            f.write("\n--- Business Implications ---\n")
            f.write(f"False Positives: {cm[0, 1]} - Wasted resources on retention efforts\n")
            f.write(f"False Negatives: {cm[1, 0]} - Lost revenue from missed opportunities\n")
        
        logger.info("Metrics saved successfully")
    
    def train(self, df: pd.DataFrame, model_path: str, metrics_path: str) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            df (pd.DataFrame): Feature-engineered dataframe
            model_path (str): Path to save the model
            metrics_path (str): Path to save metrics
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Starting model training pipeline")
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Cross-validation
        cv_results = self.cross_validate_model(X, y)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            cv_results["y_test"],
            cv_results["y_pred"],
            cv_results["y_proba"]
        )
        
        # Confusion matrix
        cm = confusion_matrix(cv_results["y_test"], cv_results["y_pred"])
        
        # Log to MLflow
        run_id = self.log_to_mlflow(metrics, cm)
        
        # Train final model on full dataset
        logger.info("Training final model on full dataset")
        final_model = self.create_model_pipeline(X)
        final_model.fit(X, y)
        
        # Save model and metrics
        self.save_model(final_model, model_path)
        self.save_metrics(metrics, cm, metrics_path)
        
        # Store results
        self.evaluation_metrics = metrics
        self.model_pipeline = final_model
        
        logger.info("Model training completed successfully")
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "mlflow_run_id": run_id,
            "model": final_model
        }
    
    def get_feature_columns(self) -> list:
        """Get feature column names."""
        return self.feature_columns
    
    def get_evaluation_metrics(self) -> Dict[str, float]:
        """Get evaluation metrics."""
        return self.evaluation_metrics


def train_model(df: pd.DataFrame, model_path: str, metrics_path: str) -> Dict[str, Any]:
    """
    Convenience function for model training.
    
    Args:
        df (pd.DataFrame): Feature-engineered dataframe
        model_path (str): Path to save the model
        metrics_path (str): Path to save metrics
        
    Returns:
        Dict[str, Any]: Training results
    """
    trainer = ModelTrainer()
    return trainer.train(df, model_path, metrics_path)
