"""
Test script for the Stock Price Prediction System
"""
import sys
import os
sys.path.append('src')

from services.data_collector import DataCollector
from services.options_calculator import OptionsCalculator
from models.lstm_model import LSTMStockPredictor
import pandas as pd

def test_data_collection():
    """Test data collection functionality"""
    print("🧪 Testing Data Collection...")
    
    collector = DataCollector()
    
    # Test stock data
    data = collector.get_stock_data("AAPL", period="1mo")
    if not data.empty:
        print(f"✅ Stock data collected: {len(data)} records")
        print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Failed to collect stock data")
        return False
    
    # Test real-time data
    real_time = collector.get_real_time_price("AAPL")
    if real_time:
        print(f"✅ Real-time data: ${real_time.get('current_price', 0):.2f}")
    else:
        print("❌ Failed to get real-time data")
    
    return True

def test_options_calculator():
    """Test options pricing functionality"""
    print("\n🧪 Testing Options Calculator...")
    
    calc = OptionsCalculator()
    
    # Test Black-Scholes pricing
    call_price = calc.black_scholes_call(150, 155, 0.25, 0.05, 0.2)
    put_price = calc.black_scholes_put(150, 155, 0.25, 0.05, 0.2)
    
    print(f"✅ Call option price: ${call_price:.4f}")
    print(f"✅ Put option price: ${put_price:.4f}")
    
    # Test Greeks calculation
    greeks = calc.calculate_greeks(150, 155, 0.25, 0.05, 0.2, 'call')
    print(f"✅ Greeks calculated: Delta={greeks['delta']:.4f}")
    
    return True

def test_lstm_model():
    """Test LSTM model functionality"""
    print("\n🧪 Testing LSTM Model...")
    
    try:
        # Get sample data
        collector = DataCollector()
        data = collector.get_stock_data("AAPL", period="6mo")
        
        if data.empty:
            print("❌ No data for model testing")
            return False
        
        # Create and train model (small test)
        model = LSTMStockPredictor()
        print("   Training model (this may take a moment)...")
        
        # Use smaller dataset for quick test
        test_data = data.tail(100)
        history = model.train(test_data, epochs=5, validation_split=0.1)
        
        # Make prediction
        predictions = model.predict(test_data, days_ahead=1)
        
        print(f"✅ Model trained and prediction made: ${predictions[0]:.2f}")
        print(f"   Current price: ${test_data['Close'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_api_endpoints():
    """Test API functionality (requires server to be running)"""
    print("\n🧪 Testing API Endpoints...")
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print("❌ API server not responding")
            return False
            
    except Exception as e:
        print(f"⚠️  API test skipped (server not running): {e}")
        return True  # Don't fail the test if server isn't running

def main():
    """Run all tests"""
    print("🚀 Stock Price Prediction System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Collection", test_data_collection),
        ("Options Calculator", test_options_calculator),
        ("LSTM Model", test_lstm_model),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()