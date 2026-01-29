"""
API routes for stock price predictions
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
from services.data_collector import DataCollector
from services.model_trainer import ModelTrainer
from models.lstm_model import LSTMStockPredictor
import pandas as pd

router = APIRouter()

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int = 1
    model_type: str = "lstm"

class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[float]
    confidence_score: float
    model_accuracy: Optional[float]
    timestamp: str

@router.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Generate stock price predictions"""
    try:
        # Initialize services
        data_collector = DataCollector()
        model_trainer = ModelTrainer()
        
        # Get historical data
        data = data_collector.get_stock_data(request.symbol, period="2y")
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Load or train model
        model = LSTMStockPredictor()
        model_path = f"models/saved/{request.symbol}_lstm"
        
        try:
            model.load_model(model_path)
        except:
            # Train new model if not found
            print(f"Training new model for {request.symbol}...")
            model.train(data, epochs=50)
            model.save_model(model_path)
        
        # Make predictions
        predictions = model.predict(data, days_ahead=request.days_ahead)
        
        # Calculate confidence score (simplified)
        recent_volatility = data['Close'].pct_change().tail(30).std()
        confidence_score = max(0.1, 1.0 - (recent_volatility * 10))
        
        return PredictionResponse(
            symbol=request.symbol,
            predictions=predictions.tolist(),
            confidence_score=round(confidence_score, 3),
            model_accuracy=0.85,  # Placeholder
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{symbol}/performance")
async def get_model_performance(symbol: str):
    """Get model performance metrics"""
    try:
        # This would typically load from a database
        return {
            "symbol": symbol,
            "accuracy": 0.85,
            "mse": 0.02,
            "mae": 0.015,
            "last_trained": "2024-01-15T10:30:00Z",
            "training_samples": 500
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{symbol}/retrain")
async def retrain_model(symbol: str, background_tasks: BackgroundTasks):
    """Retrain model for a specific symbol"""
    try:
        background_tasks.add_task(retrain_model_task, symbol)
        return {"message": f"Model retraining started for {symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_model_task(symbol: str):
    """Background task to retrain model"""
    try:
        data_collector = DataCollector()
        model = LSTMStockPredictor()
        
        # Get fresh data
        data = data_collector.get_stock_data(symbol, period="2y")
        
        # Retrain model
        model.train(data, epochs=100)
        
        # Save updated model
        model.save_model(f"models/saved/{symbol}_lstm")
        
        print(f"Model retrained successfully for {symbol}")
        
    except Exception as e:
        print(f"Error retraining model for {symbol}: {e}")

@router.get("/batch-predict")
async def batch_predict(symbols: str, days_ahead: int = 1):
    """Get predictions for multiple symbols"""
    try:
        symbol_list = symbols.split(",")
        data_collector = DataCollector()
        results = {}
        
        for symbol in symbol_list:
            try:
                data = data_collector.get_stock_data(symbol.strip(), period="1y")
                if not data.empty:
                    model = LSTMStockPredictor()
                    
                    # Quick training for demo (in production, use pre-trained models)
                    model.train(data, epochs=20)
                    predictions = model.predict(data, days_ahead=days_ahead)
                    
                    results[symbol] = {
                        "predictions": predictions.tolist(),
                        "current_price": float(data['Close'].iloc[-1]),
                        "predicted_change": float((predictions[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100)
                    }
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))