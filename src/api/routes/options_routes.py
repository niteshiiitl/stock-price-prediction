"""
API routes for options trading and analysis
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from services.options_calculator import OptionsCalculator
from services.data_collector import DataCollector
import pandas as pd

router = APIRouter()

class OptionsPricingRequest(BaseModel):
    symbol: str
    strike_price: float
    expiry_days: int
    option_type: str  # 'call' or 'put'
    volatility: Optional[float] = None

class GreeksRequest(BaseModel):
    symbol: str
    strike_price: float
    expiry_days: int
    option_type: str
    volatility: Optional[float] = None

class PortfolioPosition(BaseModel):
    symbol: str
    stock_price: float
    strike: float
    time_to_expiry: float
    volatility: float
    option_type: str
    quantity: int
    entry_price: float

@router.post("/price")
async def calculate_option_price(request: OptionsPricingRequest):
    """Calculate theoretical option price using Black-Scholes"""
    try:
        # Get current stock data
        data_collector = DataCollector()
        real_time_data = data_collector.get_real_time_price(request.symbol)
        
        if not real_time_data:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        current_price = real_time_data['current_price']
        
        # Calculate volatility if not provided
        if request.volatility is None:
            historical_data = data_collector.get_stock_data(request.symbol, period="1y")
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        else:
            volatility = request.volatility
        
        # Initialize options calculator
        calc = OptionsCalculator()
        
        # Convert days to years
        time_to_expiry = request.expiry_days / 365.0
        
        # Calculate option price
        if request.option_type.lower() == 'call':
            option_price = calc.black_scholes_call(
                current_price, request.strike_price, time_to_expiry, calc.risk_free_rate, volatility
            )
        else:
            option_price = calc.black_scholes_put(
                current_price, request.strike_price, time_to_expiry, calc.risk_free_rate, volatility
            )
        
        # Calculate Greeks
        greeks = calc.calculate_greeks(
            current_price, request.strike_price, time_to_expiry, 
            calc.risk_free_rate, volatility, request.option_type
        )
        
        return {
            "symbol": request.symbol,
            "current_stock_price": current_price,
            "strike_price": request.strike_price,
            "option_type": request.option_type,
            "theoretical_price": round(option_price, 4),
            "volatility_used": round(volatility, 4),
            "time_to_expiry_days": request.expiry_days,
            "greeks": {k: round(v, 4) for k, v in greeks.items()},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/greeks")
async def calculate_greeks(request: GreeksRequest):
    """Calculate option Greeks"""
    try:
        data_collector = DataCollector()
        real_time_data = data_collector.get_real_time_price(request.symbol)
        
        if not real_time_data:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        current_price = real_time_data['current_price']
        
        # Calculate volatility if not provided
        if request.volatility is None:
            historical_data = data_collector.get_stock_data(request.symbol, period="1y")
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)
        else:
            volatility = request.volatility
        
        calc = OptionsCalculator()
        time_to_expiry = request.expiry_days / 365.0
        
        greeks = calc.calculate_greeks(
            current_price, request.strike_price, time_to_expiry,
            calc.risk_free_rate, volatility, request.option_type
        )
        
        return {
            "symbol": request.symbol,
            "greeks": {k: round(v, 6) for k, v in greeks.items()},
            "parameters": {
                "stock_price": current_price,
                "strike_price": request.strike_price,
                "time_to_expiry": time_to_expiry,
                "volatility": volatility,
                "option_type": request.option_type
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/greeks")
async def calculate_portfolio_greeks(positions: List[PortfolioPosition]):
    """Calculate portfolio-level Greeks"""
    try:
        calc = OptionsCalculator()
        
        # Convert positions to the format expected by the calculator
        position_data = []
        for pos in positions:
            position_data.append({
                'stock_price': pos.stock_price,
                'strike': pos.strike,
                'time_to_expiry': pos.time_to_expiry,
                'volatility': pos.volatility,
                'option_type': pos.option_type,
                'quantity': pos.quantity
            })
        
        portfolio_greeks = calc.calculate_portfolio_greeks(position_data)
        
        return {
            "portfolio_greeks": {k: round(v, 4) for k, v in portfolio_greeks.items()},
            "position_count": len(positions),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chain/{symbol}")
async def get_options_chain(symbol: str):
    """Get options chain for a symbol"""
    try:
        data_collector = DataCollector()
        options_data = data_collector.get_options_data(symbol)
        
        if not options_data:
            raise HTTPException(status_code=404, detail=f"No options data found for {symbol}")
        
        return {
            "symbol": symbol,
            "expiry_date": options_data['expiry_date'],
            "calls": options_data['calls'][:10],  # Limit to first 10
            "puts": options_data['puts'][:10],
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/implied-volatility")
async def calculate_implied_volatility(
    symbol: str,
    market_price: float,
    strike_price: float,
    expiry_days: int,
    option_type: str
):
    """Calculate implied volatility from market price"""
    try:
        data_collector = DataCollector()
        real_time_data = data_collector.get_real_time_price(symbol)
        
        if not real_time_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current_price = real_time_data['current_price']
        calc = OptionsCalculator()
        time_to_expiry = expiry_days / 365.0
        
        implied_vol = calc.implied_volatility(
            market_price, current_price, strike_price, 
            time_to_expiry, calc.risk_free_rate, option_type
        )
        
        return {
            "symbol": symbol,
            "implied_volatility": round(implied_vol, 4),
            "market_price": market_price,
            "parameters": {
                "stock_price": current_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "option_type": option_type
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))