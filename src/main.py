"""
Main entry point for the Stock Price Prediction System - Render Optimized
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import stock_routes, prediction_routes, options_routes
from services.data_collector import DataCollector
from services.model_trainer import ModelTrainer

# Get port from environment (Render sets this)
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

app = FastAPI(
    title="AI Stock Price Prediction API",
    description="Advanced stock price prediction with options trading support",
    version="1.0.0"
)

# CORS middleware - Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(stock_routes.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(prediction_routes.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(options_routes.router, prefix="/api/options", tags=["options"])

# Serve static files (frontend)
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.on_startup
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting AI Stock Prediction System on Render...")
    print(f"📊 Server will run on {HOST}:{PORT}")
    
    # Initialize data collector
    try:
        data_collector = DataCollector()
        await data_collector.initialize()
        print("✅ Data collector initialized")
    except Exception as e:
        print(f"⚠️ Data collector initialization failed: {e}")
    
    print("✅ System ready!")

@app.get("/")
async def root():
    """Root endpoint - serve frontend or API info"""
    return {
        "message": "AI Stock Price Prediction API", 
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "/static/index.html" if os.path.exists("frontend/index.html") else "Not available"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy", 
        "service": "stock-prediction-api",
        "version": "1.0.0"
    }

# Serve frontend at root if available
@app.get("/app")
async def serve_frontend():
    """Serve the frontend application"""
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "Frontend not available", "api_docs": "/docs"}

if __name__ == "__main__":
    print(f"🚀 Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False  # Disable reload in production
    )