"""
Configuration settings for the application
"""
import os
from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/stockdb"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Market Data APIs
    ALPHA_VANTAGE_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    
    # ML Model Settings
    MODEL_UPDATE_INTERVAL: int = 3600  # seconds
    PREDICTION_HORIZON: int = 30  # days
    LOOKBACK_PERIOD: int = 252  # trading days (1 year)
    
    # Trading Settings
    RISK_FREE_RATE: float = 0.05
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    
    # Supported symbols
    DEFAULT_SYMBOLS: List[str] = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
        "NVDA", "META", "NFLX", "SPY", "QQQ"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()