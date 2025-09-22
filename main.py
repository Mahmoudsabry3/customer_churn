"""
Main entry point for the churn prediction API.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from src.api.app import create_app
from src.utils.logger import setup_logging
from src.config.settings import API_CONFIG

# Setup logging
setup_logging()

# Create FastAPI app
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )