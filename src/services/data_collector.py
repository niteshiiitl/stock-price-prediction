"""
Data Collection Service for Stock Market Data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import aiohttp
from config.settings import settings

class DataCollector:
    def __init__(self):
        self.symbols = settings.DEFAULT_SYMBOLS
        self.cache = {}
        
    async def initialize(self):
        """Initialize the data collector"""
        print("📊 Initializing data collector...")
        
    def get_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Add technical indicators
            data = self.add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        return df
    
    def get_options_data(self, symbol: str) -> Dict:
        """Get options data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {}
            
            # Get options for the nearest expiration
            nearest_expiry = options_dates[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            return {
                'expiry_date': nearest_expiry,
                'calls': options_chain.calls.to_dict('records'),
                'puts': options_chain.puts.to_dict('records')
            }
            
        except Exception as e:
            print(f"Error fetching options data for {symbol}: {e}")
            return {}
    
    def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching real-time data for {symbol}: {e}")
            return {}
    
    def get_market_sentiment(self) -> Dict:
        """Get market sentiment indicators"""
        try:
            # VIX (Fear & Greed Index)
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            
            # S&P 500
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="5d")
            
            return {
                'vix_current': float(vix_data['Close'].iloc[-1]),
                'vix_change': float(vix_data['Close'].pct_change().iloc[-1]),
                'spy_current': float(spy_data['Close'].iloc[-1]),
                'spy_change': float(spy_data['Close'].pct_change().iloc[-1]),
                'market_sentiment': self._calculate_sentiment(vix_data['Close'].iloc[-1])
            }
            
        except Exception as e:
            print(f"Error fetching market sentiment: {e}")
            return {}
    
    def _calculate_sentiment(self, vix_value: float) -> str:
        """Calculate market sentiment based on VIX"""
        if vix_value < 15:
            return "Very Bullish"
        elif vix_value < 20:
            return "Bullish"
        elif vix_value < 25:
            return "Neutral"
        elif vix_value < 30:
            return "Bearish"
        else:
            return "Very Bearish"
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks concurrently"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol)
                if not data.empty:
                    results[symbol] = data
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        return results