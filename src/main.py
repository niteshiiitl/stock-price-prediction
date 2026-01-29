"""
Main entry point for the Stock Price Prediction System
"""
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import stock_routes, prediction_routes, options_routes
from services.data_collector import DataCollector
from services.model_trainer import ModelTrainer
from config.settings import settings

app = FastAPI(
    title="AI Stock Price Prediction API",
    description="Advanced stock price prediction with options trading support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stock_routes.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(prediction_routes.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(options_routes.router, prefix="/api/options", tags=["options"])

@app.on_startup
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting AI Stock Prediction System...")
    
    # Initialize data collector
    data_collector = DataCollector()
    await data_collector.initialize()
    
    print("✅ System ready!")

@app.get("/")
async def root():
    return {"message": "AI Stock Price Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )