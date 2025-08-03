.PHONY: install run test clean lint format help

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  run        - Run the FastAPI server"
	@echo "  test       - Run tests (placeholder)"
	@echo "  clean      - Clean up temporary files"
	@echo "  lint       - Run linting (placeholder)"
	@echo "  format     - Format code (placeholder)"
	@echo "  help       - Show this help message"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

run:
	@echo "Starting FastAPI server..."
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "Running tests..."
	@echo "Tests not implemented yet"

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf mlruns

lint:
	@echo "Running linting..."
	@echo "Linting not configured yet"

format:
	@echo "Formatting code..."
	@echo "Code formatting not configured yet"

