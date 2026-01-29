"""
Vercel serverless function entry point for Stock Price Prediction API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import json
from datetime import datetime, timedelta
import os

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

# Utility functions
def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Get stock data with technical indicators"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return pd.DataFrame()
        
        # Add simple technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
        
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def simple_prediction(data: pd.DataFrame, days_ahead: int = 1) -> List[float]:
    """Simple prediction using moving averages and trend analysis"""
    if len(data) < 50:
        # Not enough data, return current price
        return [float(data['Close'].iloc[-1])] * days_ahead
    
    # Calculate trend components
    recent_prices = data['Close'].tail(20)
    sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else recent_prices.mean()
    sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else data['Close'].tail(50).mean()
    
    # Trend direction
    trend = 1 if sma_20 > sma_50 else -1
    
    # Volatility
    volatility = recent_prices.pct_change().std()
    
    # Current price
    current_price = float(data['Close'].iloc[-1])
    
    # Simple prediction logic
    predictions = []
    for day in range(1, days_ahead + 1):
        # Base prediction on trend and some randomness
        trend_factor = trend * 0.001 * day  # Small daily trend
        volatility_factor = np.random.normal(0, volatility * 0.5)  # Random component
        
        predicted_price = current_price * (1 + trend_factor + volatility_factor)
        predictions.append(max(predicted_price, current_price * 0.8))  # Floor at 20% drop
    
    return predictions

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option pricing"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option pricing"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """Calculate option Greeks"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type.lower() == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
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
    return {"message": "AI Stock Price Prediction API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predictions/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Generate stock price predictions"""
    try:
        # Get historical data
        data = get_stock_data(request.symbol, period="1y")
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Make predictions
        predictions = simple_prediction(data, request.days_ahead)
        
        # Calculate confidence score
        recent_volatility = data['Close'].pct_change().tail(30).std()
        confidence_score = max(0.1, 1.0 - (recent_volatility * 5))
        
        return PredictionResponse(
            symbol=request.symbol,
            predictions=predictions,
            confidence_score=round(confidence_score, 3),
            current_price=float(data['Close'].iloc[-1]),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/options/price")
async def calculate_option_price(request: OptionsPricingRequest):
    """Calculate theoretical option price using Black-Scholes"""
    try:
        # Get current stock data
        ticker = yf.Ticker(request.symbol)
        info = ticker.info
        current_price = info.get('currentPrice', 0)
        
        if current_price == 0:
            # Fallback to recent price
            data = get_stock_data(request.symbol, period="5d")
            if data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
            current_price = float(data['Close'].iloc[-1])
        
        # Calculate volatility if not provided
        if request.volatility is None:
            data = get_stock_data(request.symbol, period="1y")
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized
        else:
            volatility = request.volatility
        
        # Convert days to years
        time_to_expiry = request.expiry_days / 365.0
        risk_free_rate = 0.05  # 5% risk-free rate
        
        # Calculate option price
        if request.option_type.lower() == 'call':
            option_price = black_scholes_call(
                current_price, request.strike_price, time_to_expiry, risk_free_rate, volatility
            )
        else:
            option_price = black_scholes_put(
                current_price, request.strike_price, time_to_expiry, risk_free_rate, volatility
            )
        
        # Calculate Greeks
        greeks = calculate_greeks(
            current_price, request.strike_price, time_to_expiry,
            risk_free_rate, volatility, request.option_type
        )
        
        return {
            "symbol": request.symbol,
            "current_stock_price": round(current_price, 2),
            "strike_price": request.strike_price,
            "option_type": request.option_type,
            "theoretical_price": round(option_price, 4),
            "volatility_used": round(volatility, 4),
            "time_to_expiry_days": request.expiry_days,
            "greeks": greeks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/data/{symbol}")
async def get_stock_data_endpoint(symbol: str, period: str = "1y"):
    """Get historical stock data"""
    try:
        data = get_stock_data(symbol, period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Convert to JSON-serializable format
        result = {
            "symbol": symbol,
            "period": period,
            "data_points": len(data),
            "data": data.tail(100).reset_index().to_dict('records'),
            "latest_price": float(data['Close'].iloc[-1]),
            "price_change": float(data['Close'].pct_change().iloc[-1]),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/realtime/{symbol}")
async def get_realtime_data(symbol: str):
    """Get real-time stock data"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            "symbol": symbol,
            "current_price": info.get('currentPrice', 0),
            "previous_close": info.get('previousClose', 0),
            "open": info.get('open', 0),
            "day_high": info.get('dayHigh', 0),
            "day_low": info.get('dayLow', 0),
            "volume": info.get('volume', 0),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch-predict")
async def batch_predict(symbols: str, days_ahead: int = 1):
    """Get predictions for multiple symbols"""
    try:
        symbol_list = symbols.split(",")
        results = {}
        
        for symbol in symbol_list:
            try:
                symbol = symbol.strip().upper()
                data = get_stock_data(symbol, period="6mo")
                if not data.empty:
                    predictions = simple_prediction(data, days_ahead=days_ahead)
                    current_price = float(data['Close'].iloc[-1])
                    
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

# Vercel handler
def handler(request, response):
    """Vercel serverless function handler"""
    return app(request, response)

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)