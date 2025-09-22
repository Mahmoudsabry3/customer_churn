"""
Complete data processing pipeline for churn prediction.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logging
from src.config.settings import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURED_DATA_PATH
)

def main():
    """Run the complete data processing pipeline."""
    # Setup logging
    setup_logging()
    
    print("=== Customer Churn Prediction Data Pipeline ===")
    
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
    
    print("\n=== Data Pipeline Completed Successfully ===")
    print(f"Preprocessed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Featured data saved to: {FEATURED_DATA_PATH}")
    print(f"Final dataset shape: {featured_df.shape}")
    print(f"Features created: {engineer.get_feature_columns()}")

if __name__ == "__main__":
    main()
