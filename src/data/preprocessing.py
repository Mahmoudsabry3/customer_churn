"""
Data preprocessing module for customer churn prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing operations for churn prediction."""
    
    def __init__(self):
        self.churn_users = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp columns to datetime objects.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with converted timestamps
        """
        logger.info("Converting timestamp columns to datetime")
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df["registration"] = pd.to_datetime(df["registration"], unit="ms")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        logger.info("Handling missing values")
        df = df.copy()
        
        # Handle missing userId by dropping rows as it's critical for user-level analysis
        initial_count = len(df)
        df.dropna(subset=["userId"], inplace=True)
        logger.info(f"Dropped {initial_count - len(df)} rows with missing userId")
        
        # Fill missing artist, song, length with 'Unknown' or 0
        df["artist"].fillna("Unknown", inplace=True)
        df["song"].fillna("Unknown", inplace=True)
        df["length"].fillna(0, inplace=True)
        
        # Fill missing location, userAgent, lastName, firstName, gender with 'Unknown'
        categorical_columns = ["location", "userAgent", "lastName", "firstName", "gender"]
        for col in categorical_columns:
            df[col].fillna("Unknown", inplace=True)
        
        return df
    
    def create_churn_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create churn label based on cancellation confirmation events.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with churn label
        """
        logger.info("Creating churn labels")
        df = df.copy()
        
        # Identify users who have cancelled
        self.churn_users = df[df["page"] == "Cancellation Confirmation"]["userId"].unique()
        df["churn"] = df["userId"].apply(lambda x: 1 if x in self.churn_users else 0)
        
        churn_count = df["churn"].sum()
        logger.info(f"Identified {churn_count} churn events across {len(self.churn_users)} unique users")
        
        return df
    
    def preprocess(self, file_path: str) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path (str): Path to the raw data file
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Load data
        df = self.load_data(file_path)
        
        # Convert timestamps
        df = self.convert_timestamps(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create churn labels
        df = self.create_churn_label(df)
        
        logger.info("Data preprocessing completed successfully")
        return df
    
    def save_preprocessed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save preprocessed data to CSV.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            output_path (str): Output file path
        """
        logger.info(f"Saving preprocessed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Preprocessed data saved successfully")


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Convenience function for data preprocessing.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save preprocessed data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(input_path)
    preprocessor.save_preprocessed_data(df, output_path)
    return df
