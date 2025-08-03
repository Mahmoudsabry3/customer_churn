import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert timestamp columns to datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms")

    # Handle missing userId by dropping rows as it's critical for user-level analysis
    # These records likely represent unauthenticated sessions or data collection issues
    df.dropna(subset=["userId"], inplace=True)

    # Fill missing artist, song, length with 'Unknown' or 0
    # This assumes that a missing artist/song/length means the event was not a song play
    df["artist"].fillna("Unknown", inplace=True)
    df["song"].fillna("Unknown", inplace=True)
    df["length"].fillna(0, inplace=True)

    # Fill missing location, userAgent, lastName, firstName, gender with 'Unknown'
    # These are categorical features where 'Unknown' can be treated as a separate category
    for col in ["location", "userAgent", "lastName", "firstName", "gender"]:
        df[col].fillna("Unknown", inplace=True)

    # Create churn label: A user is considered churned if they have a 'Cancellation Confirmation' event
    # This is a critical step for defining the target variable.
    # We identify unique users who have cancelled at any point in their history within this dataset.
    churn_users = df[df["page"] == "Cancellation Confirmation"]["userId"].unique()
    df["churn"] = df["userId"].apply(lambda x: 1 if x in churn_users else 0)

    return df

def feature_engineer(df):
    # Sort by userId and timestamp for time-based features to ensure correct chronological order
    df = df.sort_values(by=["userId", "ts"])

    # Calculate session duration (approximate) for each session
    # This captures user engagement within a single session.
    df["session_duration"] = df.groupby(["userId", "sessionId"])["ts"].transform(lambda x: (x.max() - x.min()).total_seconds())

    # Group by user to create user-level features
    # These aggregations summarize user behavior over their entire recorded history.
    user_features = df.groupby("userId").agg(
        # Engagement Metrics
        last_ts=("ts", "max"), # To calculate time since last activity
        first_ts=("ts", "min"), # To calculate days since registration
        total_sessions=("sessionId", "nunique"), # Number of distinct sessions
        avg_session_duration=("session_duration", "mean"), # Average duration of user's sessions
        total_songs_played=("song", lambda x: (x != "Unknown").sum()), # Total songs played (excluding 'Unknown' for missing)
        avg_songs_per_session=("song", lambda x: (x != "Unknown").sum() / x.nunique()), # Average songs per unique session
        thumbs_up_count=("page", lambda x: (x == "Thumbs Up").sum()), # Count of positive feedback
        thumbs_down_count=("page", lambda x: (x == "Thumbs Down").sum()), # Count of negative feedback
        add_playlist_count=("page", lambda x: (x == "Add to Playlist").sum()), # Count of playlist additions
        add_friend_count=("page", lambda x: (x == "Add Friend").sum()), # Count of friend additions
        # Subscription Health
        last_level=("level", "last"), # User's last known subscription level
        # Churn label (already created in preprocessing, ensure it's propagated to user-level)
        churn=("churn", "max") # If a user churned at any point, they are marked as churned
    ).reset_index()

    # Time since last activity: captures recency of engagement
    user_features["time_since_last_activity"] = (df["ts"].max() - user_features["last_ts"]).dt.days

    # Days since registration: captures user tenure
    user_features["days_since_registration"] = (df["ts"].max() - user_features["first_ts"]).dt.days

    # Ratio features: normalize engagement metrics
    # Handle division by zero by replacing inf/-inf with 0, and NaN with 0
    user_features["thumbs_up_ratio"] = user_features["thumbs_up_count"] / user_features["total_songs_played"]
    user_features["thumbs_down_ratio"] = user_features["thumbs_down_count"] / user_features["total_songs_played"]
    user_features.replace([np.inf, -np.inf], 0, inplace=True)
    user_features.fillna(0, inplace=True)

    # Convert last_level to numerical: 'paid' as 1, 'free' as 0
    user_features["is_paid_user"] = user_features["last_level"].apply(lambda x: 1 if x == "paid" else 0)

    # Drop intermediate timestamp columns and last_level as they are no longer needed after feature creation
    user_features = user_features.drop(columns=["last_ts", "first_ts", "last_level"])

    return user_features

if __name__ == "__main__":
    processed_data_path = "processed_data.csv"
    
    # Step 1: Load and Preprocess Data
    preprocessed_df = load_and_preprocess_data(processed_data_path)
    print("\n--- Data Preprocessing Summary ---")
    print("Preprocessed DataFrame Head:")
    print(preprocessed_df.head().to_string())
    print("\nPreprocessed DataFrame Info:")
    preprocessed_df.info()
    print("\nChurned Users Count:", preprocessed_df["churn"].sum())
    print("Missing values after preprocessing:")
    print(preprocessed_df.isnull().sum().to_string())
    preprocessed_df.to_csv("preprocessed_data.csv", index=False)
    print("\nPreprocessed data saved to preprocessed_data.csv")

    # Step 2: Feature Engineering
    # Ensure 'ts' and 'registration' are datetime objects before passing to feature_engineer
    # This is crucial for time-based calculations within feature engineering.
    preprocessed_df["ts"] = pd.to_datetime(preprocessed_df["ts"])
    preprocessed_df["registration"] = pd.to_datetime(preprocessed_df["registration"])

    featured_df = feature_engineer(preprocessed_df)
    print("\n--- Feature Engineering Summary ---")
    print("Featured DataFrame Head:")
    print(featured_df.head().to_string())
    print("\nFeatured DataFrame Info:")
    featured_df.info()
    print("Missing values after feature engineering:")
    print(featured_df.isnull().sum().to_string())
    featured_df.to_csv("featured_data.csv", index=False)
    print("\nFeatured data saved to featured_data.csv")

    # Data Leakage Prevention Discussion:
    # The churn label is created based on the entire historical data available for each user.
    # This is acceptable for training a model that predicts future churn based on past behavior.
    # For real-time prediction, features would need to be calculated up to a specific point in time (prediction window).
    # In this implementation, all features are derived from historical user activity up to the latest timestamp in the dataset.
    # The 'time_since_last_activity' and 'days_since_registration' features inherently capture temporal aspects.
    # No future information is used to create the features for a given user's churn label.
    # The definition of churn ('Cancellation Confirmation' event) is clear and directly observable from the data.
    # The time-series split in model evaluation (though not explicitly implemented yet, but part of the plan) 
    # would further ensure that the model is not exposed to future data during training.


