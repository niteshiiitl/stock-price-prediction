"""
API routes for stock data and market information
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from services.data_collector import DataCollector
import pandas as pd

router = APIRouter()

@router.get("/data/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y", interval: str = "1d"):
    """Get historical stock data"""
    try:
        data_collector = DataCollector()
        data = data_collector.get_stock_data(symbol, period, interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Convert to JSON-serializable format
        result = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data_points": len(data),
            "data": data.tail(100).to_dict('records'),  # Last 100 records
            "latest_price": float(data['Close'].iloc[-1]),
            "price_change": float(data['Close'].pct_change().iloc[-1]),
            "volume": int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime/{symbol}")
async def get_realtime_data(symbol: str):
    """Get real-time stock data"""
    try:
        data_collector = DataCollector()
        data = data_collector.get_real_time_price(symbol)
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No real-time data found for {symbol}")
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-sentiment")
async def get_market_sentiment():
    """Get overall market sentiment indicators"""
    try:
        data_collector = DataCollector()
        sentiment = data_collector.get_market_sentiment()
        
        return sentiment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch")
async def get_batch_data(symbols: str):
    """Get data for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        data_collector = DataCollector()
        
        results = {}
        for symbol in symbol_list:
            try:
                real_time = data_collector.get_real_time_price(symbol)
                if real_time:
                    results[symbol] = real_time
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/technical-analysis/{symbol}")
async def get_technical_analysis(symbol: str):
    """Get technical analysis indicators"""
    try:
        data_collector = DataCollector()
        data = data_collector.get_stock_data(symbol, period="6mo")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Get latest values of technical indicators
        latest = data.iloc[-1]
        
        analysis = {
            "symbol": symbol,
            "current_price": float(latest['Close']),
            "sma_20": float(latest['SMA_20']) if 'SMA_20' in data.columns else None,
            "sma_50": float(latest['SMA_50']) if 'SMA_50' in data.columns else None,
            "rsi": float(latest['RSI']) if 'RSI' in data.columns else None,
            "macd": float(latest['MACD']) if 'MACD' in data.columns else None,
            "macd_signal": float(latest['MACD_Signal']) if 'MACD_Signal' in data.columns else None,
            "bollinger_upper": float(latest['BB_Upper']) if 'BB_Upper' in data.columns else None,
            "bollinger_lower": float(latest['BB_Lower']) if 'BB_Lower' in data.columns else None,
            "volume_ratio": float(latest['Volume_Ratio']) if 'Volume_Ratio' in data.columns else None,
        }
        
        # Add trading signals
        signals = []
        
        if analysis['rsi']:
            if analysis['rsi'] > 70:
                signals.append("RSI indicates overbought condition")
            elif analysis['rsi'] < 30:
                signals.append("RSI indicates oversold condition")
        
        if analysis['sma_20'] and analysis['sma_50']:
            if analysis['current_price'] > analysis['sma_20'] > analysis['sma_50']:
                signals.append("Price above both moving averages - bullish trend")
            elif analysis['current_price'] < analysis['sma_20'] < analysis['sma_50']:
                signals.append("Price below both moving averages - bearish trend")
        
        if analysis['macd'] and analysis['macd_signal']:
            if analysis['macd'] > analysis['macd_signal']:
                signals.append("MACD above signal line - bullish momentum")
            else:
                signals.append("MACD below signal line - bearish momentum")
        
        analysis['signals'] = signals
        analysis['timestamp'] = pd.Timestamp.now().isoformat()
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{query}")
async def search_stocks(query: str):
    """Search for stocks by symbol or company name"""
    try:
        # This is a simplified search - in production, you'd use a proper search API
        common_stocks = {
            "AAPL": "Apple Inc.",
            "GOOGL": "Alphabet Inc.",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com Inc.",
            "TSLA": "Tesla Inc.",
            "NVDA": "NVIDIA Corporation",
            "META": "Meta Platforms Inc.",
            "NFLX": "Netflix Inc.",
            "SPY": "SPDR S&P 500 ETF",
            "QQQ": "Invesco QQQ Trust"
        }
        
        query_upper = query.upper()
        results = []
        
        for symbol, name in common_stocks.items():
            if query_upper in symbol or query.lower() in name.lower():
                results.append({
                    "symbol": symbol,
                    "name": name
                })
        
        return {"query": query, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))