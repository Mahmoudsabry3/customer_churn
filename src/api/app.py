"""
FastAPI application factory for the churn prediction system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config.settings import API_CONFIG, CORS_ORIGINS, CORS_METHODS, CORS_HEADERS
from .routes import router
import logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    logger.info("Creating FastAPI application")
    
    # Create FastAPI app
    app = FastAPI(
        title=API_CONFIG["title"],
        description=API_CONFIG["description"],
        version=API_CONFIG["version"]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=CORS_METHODS,
        allow_headers=CORS_HEADERS,
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Include routers
    app.include_router(router)
    
    logger.info("FastAPI application created successfully")
    return app


def get_app() -> FastAPI:
    """
    Get the FastAPI application instance.
    
    Returns:
        FastAPI: Application instance
    """
    return create_app()
