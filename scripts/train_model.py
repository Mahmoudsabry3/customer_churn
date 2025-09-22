"""
Model training pipeline for churn prediction.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.trainer import ModelTrainer
from src.utils.logger import setup_logging
from src.config.settings import FEATURED_DATA_PATH, MODEL_PATH, METRICS_PATH

def main():
    """Run the model training pipeline."""
    # Setup logging
    setup_logging()
    
    print("=== Customer Churn Prediction Model Training ===")
    
    # Load feature-engineered data
    print("\nLoading feature-engineered data...")
    df = pd.read_csv(FEATURED_DATA_PATH)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Train model
    print("\nTraining model...")
    trainer = ModelTrainer()
    results = trainer.train(df, str(MODEL_PATH), str(METRICS_PATH))
    
    # Print results
    print("\n=== Model Training Completed Successfully ===")
    print("\nModel Performance Metrics:")
    for metric, value in results["metrics"].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"MLflow Run ID: {results['mlflow_run_id']}")

if __name__ == "__main__":
    main()
