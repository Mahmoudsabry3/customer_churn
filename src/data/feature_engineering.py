"""
Feature engineering module for customer churn prediction.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering operations for churn prediction."""
    
    def __init__(self):
        self.feature_columns = None
        
    def calculate_session_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate session duration for each session.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with session duration
        """
        logger.info("Calculating session durations")
        df = df.copy()
        df["session_duration"] = df.groupby(["userId", "sessionId"])["ts"].transform(
            lambda x: (x.max() - x.min()).total_seconds()
        )
        return df
    
    def create_user_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level aggregated features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: User-level features dataframe
        """
        logger.info("Creating user-level features")
        
        # Group by user to create user-level features
        user_features = df.groupby("userId").agg(
            # Engagement Metrics
            last_ts=("ts", "max"),  # To calculate time since last activity
            first_ts=("ts", "min"),  # To calculate days since registration
            total_sessions=("sessionId", "nunique"),  # Number of distinct sessions
            avg_session_duration=("session_duration", "mean"),  # Average duration
            total_songs_played=("song", lambda x: (x != "Unknown").sum()),  # Total songs
            avg_songs_per_session=("song", lambda x: (x != "Unknown").sum() / x.nunique()),
            thumbs_up_count=("page", lambda x: (x == "Thumbs Up").sum()),
            thumbs_down_count=("page", lambda x: (x == "Thumbs Down").sum()),
            add_playlist_count=("page", lambda x: (x == "Add to Playlist").sum()),
            add_friend_count=("page", lambda x: (x == "Add Friend").sum()),
            # Subscription Health
            last_level=("level", "last"),  # User's last known subscription level
            # Churn label
            churn=("churn", "max")  # If a user churned at any point
        ).reset_index()
        
        return user_features
    
    def create_temporal_features(self, df: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features like time since last activity and days since registration.
        
        Args:
            df (pd.DataFrame): Original dataframe for reference timestamps
            user_features (pd.DataFrame): User features dataframe
            
        Returns:
            pd.DataFrame: User features with temporal features
        """
        logger.info("Creating temporal features")
        
        # Time since last activity: captures recency of engagement
        user_features["time_since_last_activity"] = (
            df["ts"].max() - user_features["last_ts"]
        ).dt.days
        
        # Days since registration: captures user tenure
        user_features["days_since_registration"] = (
            df["ts"].max() - user_features["first_ts"]
        ).dt.days
        
        return user_features
    
    def create_ratio_features(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features for normalized engagement metrics.
        
        Args:
            user_features (pd.DataFrame): User features dataframe
            
        Returns:
            pd.DataFrame: User features with ratio features
        """
        logger.info("Creating ratio features")
        
        # Ratio features: normalize engagement metrics
        user_features["thumbs_up_ratio"] = (
            user_features["thumbs_up_count"] / user_features["total_songs_played"]
        )
        user_features["thumbs_down_ratio"] = (
            user_features["thumbs_down_count"] / user_features["total_songs_played"]
        )
        
        # Handle division by zero and infinite values
        user_features.replace([np.inf, -np.inf], 0, inplace=True)
        user_features.fillna(0, inplace=True)
        
        return user_features
    
    def create_subscription_features(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create subscription-related features.
        
        Args:
            user_features (pd.DataFrame): User features dataframe
            
        Returns:
            pd.DataFrame: User features with subscription features
        """
        logger.info("Creating subscription features")
        
        # Convert last_level to numerical: 'paid' as 1, 'free' as 0
        user_features["is_paid_user"] = user_features["last_level"].apply(
            lambda x: 1 if x == "paid" else 0
        )
        
        return user_features
    
    def clean_features(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and finalize features.
        
        Args:
            user_features (pd.DataFrame): User features dataframe
            
        Returns:
            pd.DataFrame: Cleaned user features dataframe
        """
        logger.info("Cleaning features")
        
        # Drop intermediate timestamp columns and last_level
        columns_to_drop = ["last_ts", "first_ts", "last_level"]
        user_features = user_features.drop(columns=columns_to_drop)
        
        # Store feature column names
        self.feature_columns = [col for col in user_features.columns if col not in ["userId", "churn"]]
        
        return user_features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            pd.DataFrame: Feature-engineered dataframe
        """
        logger.info("Starting feature engineering pipeline")
        
        # Sort by userId and timestamp for time-based features
        df = df.sort_values(by=["userId", "ts"])
        
        # Calculate session duration
        df = self.calculate_session_duration(df)
        
        # Create user-level features
        user_features = self.create_user_level_features(df)
        
        # Create temporal features
        user_features = self.create_temporal_features(df, user_features)
        
        # Create ratio features
        user_features = self.create_ratio_features(user_features)
        
        # Create subscription features
        user_features = self.create_subscription_features(user_features)
        
        # Clean features
        user_features = self.clean_features(user_features)
        
        logger.info(f"Feature engineering completed. Created {len(self.feature_columns)} features")
        return user_features
    
    def save_features(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save feature-engineered data to CSV.
        
        Args:
            df (pd.DataFrame): Feature-engineered dataframe
            output_path (str): Output file path
        """
        logger.info(f"Saving feature-engineered data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Feature-engineered data saved successfully")
    
    def get_feature_columns(self) -> List[str]:
        """
        Get the list of feature column names.
        
        Returns:
            List[str]: List of feature column names
        """
        return self.feature_columns


def engineer_features(input_df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Convenience function for feature engineering.
    
    Args:
        input_df (pd.DataFrame): Preprocessed dataframe
        output_path (str, optional): Path to save feature-engineered data
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe
    """
    engineer = FeatureEngineer()
    df = engineer.engineer_features(input_df)
    
    if output_path:
        engineer.save_features(df, output_path)
    
    return df
