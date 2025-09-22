"""
Complete pipeline for data processing and model training.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.utils.logger import setup_logging
from src.config.settings import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURED_DATA_PATH,
    MODEL_PATH, METRICS_PATH
)

def main():
    """Run the complete pipeline from raw data to trained model."""
    # Setup logging
    setup_logging()
    
    print("=== Complete Customer Churn Prediction Pipeline ===")
    
    # Step 1: Data Preprocessing
    print("\n1. Data Preprocessing...")
    preprocessor = DataPreprocessor()
    preprocessed_df = preprocessor.preprocess(str(RAW_DATA_PATH))
    preprocessor.save_preprocessed_data(preprocessed_df, str(PROCESSED_DATA_PATH))
    
    # Step 2: Feature Engineering
    print("\n2. Feature Engineering...")
    engineer = FeatureEngineer()
    featured_df = engineer.engineer_features(preprocessed_df)
    engineer.save_features(featured_df, str(FEATURED_DATA_PATH))
    
    # Step 3: Model Training
    print("\n3. Model Training...")
    trainer = ModelTrainer()
    results = trainer.train(featured_df, str(MODEL_PATH), str(METRICS_PATH))
    
    # Print final results
    print("\n=== Pipeline Completed Successfully ===")
    print(f"Preprocessed data: {PROCESSED_DATA_PATH}")
    print(f"Featured data: {FEATURED_DATA_PATH}")
    print(f"Trained model: {MODEL_PATH}")
    print(f"Model metrics: {METRICS_PATH}")
    print(f"MLflow Run ID: {results['mlflow_run_id']}")
    
    print("\nFinal Model Performance:")
    for metric, value in results["metrics"].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    main()
