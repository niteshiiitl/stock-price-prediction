"""
Vercel serverless function entry point for Stock Price Prediction API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from datetime import datetime
import random
import math

# Initialize FastAPI app
app = FastAPI(
    title="AI Stock Price Prediction API",
    description="Serverless stock price prediction and options analysis",
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

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int = 1
    model_type: str = "simple"

class OptionsPricingRequest(BaseModel):
    symbol: str
    strike_price: float
    expiry_days: int
    option_type: str
    volatility: Optional[float] = None

class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[float]
    confidence_score: float
    current_price: float
    timestamp: str

# Simple mock functions for demo (to avoid heavy dependencies)
def generate_mock_price(symbol: str) -> float:
    """Generate mock stock price based on symbol"""
    base_prices = {
        'AAPL': 150.0,
        'GOOGL': 2800.0,
        'MSFT': 300.0,
        'AMZN': 3200.0,
        'TSLA': 200.0,
        'NVDA': 450.0,
        'META': 280.0
    }
    base = base_prices.get(symbol.upper(), 100.0)
    # Add some randomness
    return round(base * (0.95 + random.random() * 0.1), 2)

def simple_prediction(symbol: str, days_ahead: int = 1) -> List[float]:
    """Simple prediction using mock data"""
    current_price = generate_mock_price(symbol)
    predictions = []
    
    for day in range(days_ahead):
        # Simple random walk with slight upward bias
        change = random.gauss(0.001, 0.02)  # 0.1% daily trend, 2% volatility
        current_price *= (1 + change)
        predictions.append(round(current_price, 2))
    
    return predictions

def black_scholes_simple(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Simplified Black-Scholes calculation"""
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    # Simplified calculation for demo
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    time_value = S * sigma * math.sqrt(T) * 0.4
    
    return round(intrinsic + time_value, 4)

def calculate_simple_greeks(S: float, K: float, T: float, option_type: str) -> Dict[str, float]:
    """Simplified Greeks calculation"""
    delta = 0.5 if option_type == 'call' else -0.5
    gamma = 0.01 / S
    theta = -0.05
    vega = S * 0.01
    rho = K * T * 0.01 if option_type == 'call' else -K * T * 0.01
    
    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4)
    }

# API Routes
@app.get("/")
async def root():
    return {
        "message": "AI Stock Price Prediction API", 
        "status": "running", 
        "version": "1.0.0",
        "endpoints": [
            "/predictions/predict",
            "/options/price",
            "/stocks/data/{symbol}",
            "/stocks/realtime/{symbol}"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predictions/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Generate stock price predictions"""
    try:
        # Generate predictions
        predictions = simple_prediction(request.symbol, request.days_ahead)
        current_price = generate_mock_price(request.symbol)
        
        # Mock confidence score
        confidence_score = round(0.7 + random.random() * 0.2, 3)
        
        return PredictionResponse(
            symbol=request.symbol.upper(),
            predictions=predictions,
            confidence_score=confidence_score,
            current_price=current_price,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/options/price")
async def calculate_option_price(request: OptionsPricingRequest):
    """Calculate theoretical option price using simplified Black-Scholes"""
    try:
        # Get mock current price
        current_price = generate_mock_price(request.symbol)
        
        # Use provided volatility or default
        volatility = request.volatility or 0.25
        
        # Convert days to years
        time_to_expiry = request.expiry_days / 365.0
        risk_free_rate = 0.05
        
        # Calculate option price
        option_price = black_scholes_simple(
            current_price, request.strike_price, time_to_expiry, 
            risk_free_rate, volatility, request.option_type
        )
        
        # Calculate Greeks
        greeks = calculate_simple_greeks(
            current_price, request.strike_price, time_to_expiry, request.option_type
        )
        
        return {
            "symbol": request.symbol.upper(),
            "current_stock_price": current_price,
            "strike_price": request.strike_price,
            "option_type": request.option_type,
            "theoretical_price": option_price,
            "volatility_used": volatility,
            "time_to_expiry_days": request.expiry_days,
            "greeks": greeks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/data/{symbol}")
async def get_stock_data_endpoint(symbol: str):
    """Get mock historical stock data"""
    try:
        current_price = generate_mock_price(symbol)
        
        # Generate mock historical data
        data_points = []
        price = current_price
        
        for i in range(30):  # Last 30 days
            date = datetime.now().replace(day=1).replace(hour=0, minute=0, second=0, microsecond=0)
            date = date.replace(day=min(30, date.day + i))
            
            # Random price movement
            change = random.gauss(0, 0.02)
            price *= (1 + change)
            
            data_points.append({
                "Date": date.isoformat(),
                "Open": round(price * 0.99, 2),
                "High": round(price * 1.02, 2),
                "Low": round(price * 0.98, 2),
                "Close": round(price, 2),
                "Volume": random.randint(1000000, 10000000)
            })
        
        return {
            "symbol": symbol.upper(),
            "period": "1mo",
            "data_points": len(data_points),
            "data": data_points,
            "latest_price": current_price,
            "price_change": round(random.gauss(0, 0.02), 4),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/realtime/{symbol}")
async def get_realtime_data(symbol: str):
    """Get mock real-time stock data"""
    try:
        current_price = generate_mock_price(symbol)
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "previous_close": round(current_price * 0.99, 2),
            "open": round(current_price * 0.98, 2),
            "day_high": round(current_price * 1.02, 2),
            "day_low": round(current_price * 0.97, 2),
            "volume": random.randint(1000000, 50000000),
            "market_cap": random.randint(100000000, 3000000000),
            "pe_ratio": round(15 + random.random() * 20, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch-predict")
async def batch_predict(symbols: str, days_ahead: int = 1):
    """Get predictions for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        results = {}
        
        for symbol in symbol_list:
            try:
                predictions = simple_prediction(symbol, days_ahead)
                current_price = generate_mock_price(symbol)
                
                results[symbol] = {
                    "predictions": predictions,
                    "current_price": current_price,
                    "predicted_change": round((predictions[0] - current_price) / current_price * 100, 2)
                }
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Export the app for Vercel
app_handler = app