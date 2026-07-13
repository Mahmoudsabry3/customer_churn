import pytest

from src.models.predictor import ChurnPredictor


VALID_FEATURES = {
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
    "is_paid_user": 1,
}


@pytest.fixture
def predictor():
    instance = ChurnPredictor.__new__(ChurnPredictor)
    instance.model = None
    return instance


def test_validate_input_accepts_complete_feature_set(predictor):
    frame = predictor.validate_input(VALID_FEATURES)

    assert frame.shape == (1, 13)
    assert list(frame.columns) == list(VALID_FEATURES)


def test_validate_input_rejects_missing_features(predictor):
    incomplete = VALID_FEATURES.copy()
    incomplete.pop("total_sessions")

    with pytest.raises(ValueError, match="total_sessions"):
        predictor.validate_input(incomplete)
