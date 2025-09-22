# Customer Churn Prediction Platform

A production-ready machine learning platform for predicting customer churn in subscription-based services. Built with FastAPI, XGBoost, and MLflow following modern software engineering best practices.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager
- 4GB+ RAM

### Installation & Running
```bash
# Clone the repository
git clone https://github.com/Mahmoudsabry3/customer_churn.git
cd customer_churn

# Quick start (install, process data, train model, and run API)
make quick-start

# Or step by step:
make install          # Install dependencies
make full-pipeline    # Process data and train model
make run             # Start the API server

# Access the application
# Web Interface: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Quick Test
```python
import requests

# Test prediction
response = requests.post("http://localhost:8000/api/churn/predict", json={
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
    "is_paid_user": 1
})

print(f"Churn Probability: {response.json()['churn_probability']:.3f}")
```

## 📊 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **ROC AUC** | 91.16% | ✅ Excellent |
| **Accuracy** | 88.89% | ✅ Good |
| **Precision** | 80.00% | ✅ Good |
| **Recall** | 69.23% | ✅ Good |
| **F1-Score** | 74.23% | ✅ Good |

## 🏗️ Project Structure

```
customer_churn/
├── src/                          # Main source code
│   ├── api/                      # API layer
│   │   ├── app.py               # FastAPI app factory
│   │   ├── models.py            # Pydantic models
│   │   └── routes.py            # API endpoints
│   ├── config/                   # Configuration
│   │   └── settings.py          # All settings and paths
│   ├── data/                     # Data processing
│   │   ├── preprocessing.py     # Data preprocessing
│   │   └── feature_engineering.py # Feature creation
│   ├── models/                   # ML models
│   │   ├── trainer.py           # Model training
│   │   └── predictor.py         # Model prediction
│   └── utils/                    # Utilities
│       └── logger.py            # Logging configuration
├── scripts/                      # Pipeline scripts
│   ├── data_pipeline.py         # Data processing pipeline
│   ├── train_model.py           # Model training script
│   └── full_pipeline.py         # Complete pipeline
├── data/                         # Data storage
│   ├── raw/                     # Raw data
│   ├── processed/               # Preprocessed data
│   └── features/                # Feature-engineered data
├── models/                       # Model artifacts
├── logs/                         # Log files
├── static/                       # Web interface
├── main.py                       # API entry point
├── Makefile                      # Build commands
└── requirements.txt              # Dependencies
```

## 🛠️ Available Commands

```bash
# Data and Model Operations
make setup-data      # Run data processing pipeline
make train-model     # Train the churn prediction model
make full-pipeline   # Run complete pipeline (data + training)

# Development
make install         # Install dependencies
make run            # Start FastAPI server
make clean          # Clean temporary files
make lint           # Run code linting
make format         # Format code

# Quick Start
make quick-start    # Setup + pipeline + run
```

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Web interface
- `GET /docs` - Interactive API documentation
- `GET /api/churn/health` - Health check
- `GET /api/churn/model/metrics` - Model performance metrics
- `GET /api/churn/model/info` - Model information

### Prediction Endpoints
- `POST /api/churn/predict` - Single user prediction
- `POST /api/churn/batch_predict` - Batch prediction

### Example API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/api/churn/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "total_sessions": 10,
    "avg_session_duration": 3600,
    "total_songs_played": 100,
    "avg_songs_per_session": 10,
    "thumbs_up_count": 20,
    "thumbs_down_count": 5,
    "add_playlist_count": 15,
    "add_friend_count": 8,
    "time_since_last_activity": 2,
    "days_since_registration": 30,
    "thumbs_up_ratio": 0.2,
    "thumbs_down_ratio": 0.05,
    "is_paid_user": 1
  }'
```

#### Health Check
```bash
curl http://localhost:8000/api/churn/health
```

## 🔧 Development

### Setting up Development Environment
```bash
make dev-setup      # Create directories and install dependencies
```

### Running Individual Components
```bash
# Data processing only
python scripts/data_pipeline.py

# Model training only
python scripts/train_model.py

# Full pipeline
python scripts/full_pipeline.py

# API server
python main.py
```

### Code Quality
```bash
make lint           # Run flake8 linting
make format         # Format code with black
make clean          # Clean temporary files
```

## 📈 Features

### Data Processing
- **Robust Preprocessing**: Handles missing values and data quality issues
- **Feature Engineering**: Creates 13 meaningful user-level features
- **Churn Labeling**: Identifies churn based on cancellation events

### Machine Learning
- **XGBoost Classifier**: High-performance gradient boosting model
- **Cross-Validation**: 5-fold stratified cross-validation
- **MLflow Integration**: Automatic experiment tracking and model versioning
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, ROC AUC

### API & Web Interface
- **FastAPI Backend**: High-performance async API
- **Type Safety**: Pydantic models for request/response validation
- **Interactive Docs**: Automatic OpenAPI documentation
- **Web Interface**: User-friendly frontend for testing
- **Health Monitoring**: Built-in health checks and metrics

### Production Ready
- **Modular Architecture**: Clean separation of concerns
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Centralized settings
- **Docker Support**: Container-ready deployment
- **Scalable Design**: Built for production environments

## 🔍 Model Details

### Features Used
1. `total_sessions` - Number of user sessions
2. `avg_session_duration` - Average session duration (seconds)
3. `total_songs_played` - Total songs played
4. `avg_songs_per_session` - Average songs per session
5. `thumbs_up_count` - Positive feedback count
6. `thumbs_down_count` - Negative feedback count
7. `add_playlist_count` - Playlist additions
8. `add_friend_count` - Friend additions
9. `time_since_last_activity` - Days since last activity
10. `days_since_registration` - User tenure
11. `thumbs_up_ratio` - Positive feedback rate
12. `thumbs_down_ratio` - Negative feedback rate
13. `is_paid_user` - Subscription tier (1=paid, 0=free)

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Preprocessing**: StandardScaler for numerical features
- **Validation**: 5-fold stratified cross-validation
- **Evaluation**: Multiple metrics for comprehensive assessment

## 🚀 Deployment

### Local Development
```bash
make quick-start
```

### Production Deployment
```bash
# Build Docker image
make docker-build

# Run Docker container
make docker-run
```

### Environment Variables
```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export MLFLOW_TRACKING_URI=file:///app/mlruns
```

## 📚 Dependencies

Core dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting model
- `mlflow` - Experiment tracking
- `pydantic` - Data validation

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For questions or issues:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the logs in the `logs/` directory

---

**Built with ❤️ for customer retention and business growth**
