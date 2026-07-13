.PHONY: install run test clean lint format help train all docker-build docker-run setup-data train-model full-pipeline

# =====================
# Default target
# =====================
help:
	@echo "Available commands:"
	@echo "  setup-data     - Run data processing pipeline"
	@echo "  train-model    - Train the churn prediction model"
	@echo "  full-pipeline  - Run complete pipeline (data + training)"
	@echo "  install        - Install dependencies"
	@echo "  run            - Run the FastAPI server"
	@echo "  test           - Run tests with pytest"
	@echo "  clean          - Clean up temporary files"
	@echo "  lint           - Run linting with flake8"
	@echo "  format         - Format code with black"
	@echo "  all            - Clean, install, lint, test, run"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"

# =====================
# Data & Model Pipeline
# =====================
setup-data:
	@echo "Running data processing pipeline..."
	python scripts/data_pipeline.py

train-model:
	@echo "Training churn prediction model..."
	python scripts/train_model.py

full-pipeline:
	@echo "Running complete pipeline (data processing + model training)..."
	python scripts/full_pipeline.py

# =====================
# Setup & Run
# =====================
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

run:
	@echo "Starting FastAPI server..."
	python main.py

# =====================
# Quality & Testing
# =====================
test:
	@echo "Running tests..."
	python -m pytest tests/

lint:
	@echo "Running linting..."
	python -m flake8 src/ scripts/ tests/ main.py

format:
	@echo "Formatting code..."
	python -m black src/ scripts/ tests/ main.py

# =====================
# Utilities
# =====================
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf mlruns
	rm -rf logs/*.log

all: clean install lint test run

# =====================
# Docker (optional)
# =====================
docker-build:
	@echo "Building Docker image..."
	docker build -t churn-app .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 -v "$(PWD)/models:/app/models:ro" churn-app

# =====================
# Development
# =====================
dev-setup:
	@echo "Setting up development environment..."
	python -m pip install -r requirements-dev.txt
	mkdir -p data/{raw,processed,features}
	mkdir -p models artifacts logs
	@echo "Development environment ready!"

# Quick start for new development
quick-start: dev-setup full-pipeline run
