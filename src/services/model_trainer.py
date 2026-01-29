"""
Model Training Service
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from models.lstm_model import LSTMStockPredictor
from services.data_collector import DataCollector
import joblib
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.data_collector = DataCollector()
        self.models = {}
        self.model_dir = "models/saved"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_model_for_symbol(self, symbol: str, epochs: int = 50, retrain: bool = False) -> Dict:
        """Train or load model for a specific symbol"""
        model_path = f"{self.model_dir}/{symbol}_lstm"
        
        # Check if model exists and retrain is not forced
        if not retrain and os.path.exists(f"{model_path}_model.h5"):
            print(f"Loading existing model for {symbol}")
            model = LSTMStockPredictor()
            model.load_model(model_path)
            self.models[symbol] = model
            return {"status": "loaded", "symbol": symbol}
        
        print(f"Training new model for {symbol}")
        
        # Get training data
        data = self.data_collector.get_stock_data(symbol, period="2y")
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Initialize and train model
        model = LSTMStockPredictor()
        history = model.train(data, epochs=epochs)
        
        # Save model
        model.save_model(model_path)
        self.models[symbol] = model
        
        # Calculate performance metrics
        performance = self.evaluate_model(model, data)
        
        return {
            "status": "trained",
            "symbol": symbol,
            "epochs": epochs,
            "performance": performance,
            "training_samples": len(data)
        }
    
    def evaluate_model(self, model: LSTMStockPredictor, data: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        try:
            # Use last 30 days for evaluation
            test_data = data.tail(60)  # Need 60 for sequence length + 30 for testing
            
            # Make predictions for last 30 days
            predictions = []
            actual_prices = []
            
            for i in range(30):
                # Get data up to current point
                current_data = test_data.iloc[:30+i]
                pred = model.predict(current_data, days_ahead=1)
                
                if len(pred) > 0:
                    predictions.append(pred[0])
                    if 30+i < len(test_data):
                        actual_prices.append(test_data.iloc[30+i]['Close'])
            
            if len(predictions) > 0 and len(actual_prices) > 0:
                # Calculate metrics
                predictions = np.array(predictions[:len(actual_prices)])
                actual_prices = np.array(actual_prices)
                
                mse = np.mean((predictions - actual_prices) ** 2)
                mae = np.mean(np.abs(predictions - actual_prices))
                mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
                
                # Direction accuracy
                pred_direction = np.diff(predictions) > 0
                actual_direction = np.diff(actual_prices) > 0
                direction_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0
                
                return {
                    "mse": float(mse),
                    "mae": float(mae),
                    "mape": float(mape),
                    "direction_accuracy": float(direction_accuracy),
                    "r_squared": float(1 - (mse / np.var(actual_prices))) if np.var(actual_prices) > 0 else 0
                }
        
        except Exception as e:
            print(f"Error evaluating model: {e}")
        
        return {
            "mse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
            "direction_accuracy": 0.0,
            "r_squared": 0.0
        }
    
    def batch_train_models(self, symbols: List[str], epochs: int = 50) -> Dict:
        """Train models for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                result = self.train_model_for_symbol(symbol, epochs)
                results[symbol] = result
                print(f"✅ Completed training for {symbol}")
            except Exception as e:
                results[symbol] = {"status": "error", "error": str(e)}
                print(f"❌ Error training {symbol}: {e}")
        
        return results
    
    def get_model_info(self, symbol: str) -> Dict:
        """Get information about a trained model"""
        model_path = f"{self.model_dir}/{symbol}_lstm_model.h5"
        
        if not os.path.exists(model_path):
            return {"status": "not_found", "symbol": symbol}
        
        # Get file modification time
        mod_time = os.path.getmtime(model_path)
        last_trained = datetime.fromtimestamp(mod_time).isoformat()
        
        return {
            "status": "exists",
            "symbol": symbol,
            "last_trained": last_trained,
            "model_path": model_path
        }
    
    def predict_with_model(self, symbol: str, days_ahead: int = 1) -> Dict:
        """Make prediction using trained model"""
        # Load model if not in memory
        if symbol not in self.models:
            model_info = self.get_model_info(symbol)
            if model_info["status"] == "not_found":
                # Train new model
                self.train_model_for_symbol(symbol)
            else:
                # Load existing model
                model = LSTMStockPredictor()
                model.load_model(f"{self.model_dir}/{symbol}_lstm")
                self.models[symbol] = model
        
        # Get recent data
        data = self.data_collector.get_stock_data(symbol, period="1y")
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Make prediction
        model = self.models[symbol]
        predictions = model.predict(data, days_ahead=days_ahead)
        
        # Calculate confidence based on recent volatility
        recent_returns = data['Close'].pct_change().tail(30)
        volatility = recent_returns.std()
        confidence = max(0.1, 1.0 - (volatility * 5))  # Simple confidence metric
        
        return {
            "symbol": symbol,
            "predictions": predictions.tolist(),
            "confidence_score": float(confidence),
            "current_price": float(data['Close'].iloc[-1]),
            "prediction_change": float((predictions[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100)
        }
    
    def update_all_models(self, symbols: List[str] = None) -> Dict:
        """Update all models with fresh data"""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        print("🔄 Starting batch model update...")
        results = self.batch_train_models(symbols, epochs=30)  # Fewer epochs for updates
        
        return {
            "update_timestamp": datetime.now().isoformat(),
            "symbols_updated": len([r for r in results.values() if r.get("status") == "trained"]),
            "results": results
        }