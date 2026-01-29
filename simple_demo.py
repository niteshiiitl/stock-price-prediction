"""
Simple Stock Price Prediction Demo
Works without heavy ML dependencies
"""
import json
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List

class SimpleStockPredictor:
    """Simple stock predictor using basic statistical methods"""
    
    def __init__(self):
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]
        
    def generate_mock_data(self, symbol: str, days: int = 100) -> List[Dict]:
        """Generate mock historical stock data"""
        data = []
        base_price = random.uniform(100, 300)
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            # Simple random walk with trend
            change = random.gauss(0, 0.02)  # 2% daily volatility
            base_price *= (1 + change)
            
            # Ensure positive price
            base_price = max(base_price, 10)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(base_price * random.uniform(0.99, 1.01), 2),
                "high": round(base_price * random.uniform(1.00, 1.03), 2),
                "low": round(base_price * random.uniform(0.97, 1.00), 2),
                "close": round(base_price, 2),
                "volume": random.randint(1000000, 10000000)
            })
            
        return data
    
    def predict_price(self, symbol: str, days_ahead: int = 1) -> Dict:
        """Predict stock price using simple moving average"""
        # Generate mock historical data
        historical_data = self.generate_mock_data(symbol, 30)
        
        # Calculate simple moving average
        recent_prices = [d["close"] for d in historical_data[-10:]]
        sma = sum(recent_prices) / len(recent_prices)
        
        # Calculate volatility
        returns = []
        for i in range(1, len(recent_prices)):
            returns.append((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1])
        
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 if returns else 0.02
        
        # Generate predictions with some randomness
        predictions = []
        current_price = recent_prices[-1]
        
        for day in range(days_ahead):
            # Simple trend + random component
            trend = 0.001  # Slight upward bias
            random_component = random.gauss(0, volatility)
            
            predicted_price = current_price * (1 + trend + random_component)
            predictions.append(round(predicted_price, 2))
            current_price = predicted_price
        
        # Calculate confidence based on volatility
        confidence = max(0.1, 1.0 - (volatility * 10))
        
        return {
            "symbol": symbol,
            "current_price": recent_prices[-1],
            "predictions": predictions,
            "confidence_score": round(confidence, 3),
            "volatility": round(volatility, 4),
            "sma_10": round(sma, 2),
            "timestamp": datetime.now().isoformat()
        }

class SimpleOptionsCalculator:
    """Simple Black-Scholes options calculator"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Simplified Black-Scholes call option pricing"""
        if T <= 0:
            return max(S - K, 0)
        
        # Simplified calculation (not exact Black-Scholes)
        intrinsic_value = max(S - K, 0)
        time_value = S * sigma * (T ** 0.5) * 0.4  # Simplified time value
        
        return round(intrinsic_value + time_value, 4)
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Simplified Black-Scholes put option pricing"""
        if T <= 0:
            return max(K - S, 0)
        
        # Simplified calculation
        intrinsic_value = max(K - S, 0)
        time_value = S * sigma * (T ** 0.5) * 0.4
        
        return round(intrinsic_value + time_value, 4)
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict:
        """Simplified Greeks calculation"""
        # Simplified Greeks (not exact)
        if option_type.lower() == 'call':
            delta = 0.5 if S > K else 0.3  # Simplified delta
        else:
            delta = -0.5 if S < K else -0.3
        
        gamma = 0.01 / S  # Simplified gamma
        theta = -0.05  # Simplified theta (time decay)
        vega = S * 0.01  # Simplified vega
        rho = K * T * 0.01  # Simplified rho
        
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4)
        }

def demo_stock_prediction():
    """Demo stock prediction functionality"""
    print("🚀 Stock Price Prediction Demo")
    print("=" * 50)
    
    predictor = SimpleStockPredictor()
    
    # Test predictions for multiple symbols
    symbols = ["AAPL", "GOOGL", "TSLA"]
    
    for symbol in symbols:
        print(f"\n📈 {symbol} Prediction:")
        result = predictor.predict_price(symbol, days_ahead=5)
        
        print(f"   Current Price: ${result['current_price']}")
        print(f"   5-Day Predictions: {result['predictions']}")
        print(f"   Confidence: {result['confidence_score']*100:.1f}%")
        print(f"   Volatility: {result['volatility']*100:.2f}%")

def demo_options_calculator():
    """Demo options calculation functionality"""
    print("\n⚡ Options Calculator Demo")
    print("=" * 50)
    
    calc = SimpleOptionsCalculator()
    
    # Example calculations
    examples = [
        {"symbol": "AAPL", "S": 150, "K": 155, "T": 30/365, "sigma": 0.25},
        {"symbol": "GOOGL", "S": 2800, "K": 2900, "T": 45/365, "sigma": 0.30},
        {"symbol": "TSLA", "S": 200, "K": 210, "T": 60/365, "sigma": 0.40}
    ]
    
    for example in examples:
        print(f"\n🎯 {example['symbol']} Options:")
        print(f"   Stock Price: ${example['S']}")
        print(f"   Strike Price: ${example['K']}")
        print(f"   Days to Expiry: {int(example['T']*365)}")
        
        # Calculate call and put prices
        call_price = calc.black_scholes_call(
            example['S'], example['K'], example['T'], 0.05, example['sigma']
        )
        put_price = calc.black_scholes_put(
            example['S'], example['K'], example['T'], 0.05, example['sigma']
        )
        
        print(f"   Call Price: ${call_price}")
        print(f"   Put Price: ${put_price}")
        
        # Calculate Greeks for call
        greeks = calc.calculate_greeks(
            example['S'], example['K'], example['T'], 0.05, example['sigma'], 'call'
        )
        
        print(f"   Greeks - Delta: {greeks['delta']}, Gamma: {greeks['gamma']}, Theta: {greeks['theta']}")

def demo_api_simulation():
    """Simulate API responses"""
    print("\n🌐 API Simulation Demo")
    print("=" * 50)
    
    predictor = SimpleStockPredictor()
    calc = SimpleOptionsCalculator()
    
    # Simulate API endpoints
    print("\n1. Stock Prediction API:")
    api_response = predictor.predict_price("AAPL", 3)
    print(json.dumps(api_response, indent=2))
    
    print("\n2. Options Pricing API:")
    options_response = {
        "symbol": "AAPL",
        "current_stock_price": 150.0,
        "strike_price": 155.0,
        "option_type": "call",
        "theoretical_price": calc.black_scholes_call(150, 155, 30/365, 0.05, 0.25),
        "greeks": calc.calculate_greeks(150, 155, 30/365, 0.05, 0.25, 'call'),
        "timestamp": datetime.now().isoformat()
    }
    print(json.dumps(options_response, indent=2))

def main():
    """Run the complete demo"""
    print("🎯 AI Stock Price Prediction & Options Trading System")
    print("📊 Simplified Demo Version")
    print("=" * 70)
    
    try:
        demo_stock_prediction()
        demo_options_calculator()
        demo_api_simulation()
        
        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("📝 This demo shows the core functionality without heavy ML dependencies.")
        print("🚀 Install the full requirements to get real data and advanced ML models.")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main()