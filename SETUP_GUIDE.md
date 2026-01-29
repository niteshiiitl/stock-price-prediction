# 🚀 AI Stock Price Prediction & Options Trading System - Setup Guide

## Overview
This system provides AI-powered stock price predictions using LSTM neural networks and comprehensive options trading analysis with Black-Scholes pricing and Greeks calculations.

## Features
- **Stock Price Prediction**: LSTM-based machine learning models
- **Options Pricing**: Black-Scholes model with Greeks calculation
- **Real-time Data**: Live market data integration
- **Technical Analysis**: Multiple technical indicators
- **Web Interface**: Interactive dashboard
- **REST API**: Complete API for integration
- **Portfolio Analysis**: Multi-position Greeks and P&L

## Quick Start

### 1. Install Dependencies
```bash
# Clone or download the project
cd stock_price_prediction

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run the System
```bash
# Option 1: Use the quick start script
python run.py

# Option 2: Manual startup
cd src
python main.py

# Option 3: Using Docker
docker-compose up
```

### 3. Access the Interface
- **Web Interface**: Open `frontend/index.html` in your browser
- **API Documentation**: http://localhost:8000/docs
- **API Base URL**: http://localhost:8000/api

## Detailed Setup

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (for ML models)
- Internet connection (for market data)

### Environment Setup
1. Copy `.env.example` to `.env`
2. Configure API keys (optional - yfinance works without keys):
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   POLYGON_API_KEY=your_key_here
   FINNHUB_API_KEY=your_key_here
   ```

### Database Setup (Optional)
For production use with PostgreSQL:
```bash
# Start PostgreSQL and Redis
docker-compose up db redis

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://postgres:password@localhost:5432/stockdb
```

## Usage Examples

### 1. Stock Price Prediction
```python
from services.data_collector import DataCollector
from models.lstm_model import LSTMStockPredictor

# Get data and train model
collector = DataCollector()
data = collector.get_stock_data("AAPL", period="2y")

model = LSTMStockPredictor()
model.train(data, epochs=50)

# Make predictions
predictions = model.predict(data, days_ahead=5)
print(f"5-day predictions: {predictions}")
```

### 2. Options Pricing
```python
from services.options_calculator import OptionsCalculator

calc = OptionsCalculator()

# Calculate option price
call_price = calc.black_scholes_call(
    S=150,      # Current stock price
    K=155,      # Strike price
    T=0.25,     # Time to expiry (years)
    r=0.05,     # Risk-free rate
    sigma=0.2   # Volatility
)

# Calculate Greeks
greeks = calc.calculate_greeks(150, 155, 0.25, 0.05, 0.2, 'call')
print(f"Delta: {greeks['delta']}")
```

### 3. API Usage
```bash
# Get stock prediction
curl -X POST "http://localhost:8000/api/predictions/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 5}'

# Calculate option price
curl -X POST "http://localhost:8000/api/options/price" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "strike_price": 155, "expiry_days": 30, "option_type": "call"}'
```

## API Endpoints

### Stock Data
- `GET /api/stocks/data/{symbol}` - Historical data
- `GET /api/stocks/realtime/{symbol}` - Real-time price
- `GET /api/stocks/technical-analysis/{symbol}` - Technical indicators

### Predictions
- `POST /api/predictions/predict` - Generate price predictions
- `GET /api/predictions/models/{symbol}/performance` - Model metrics

### Options
- `POST /api/options/price` - Calculate option price
- `POST /api/options/greeks` - Calculate Greeks
- `GET /api/options/chain/{symbol}` - Options chain data

## Testing

Run the test suite to verify everything works:
```bash
python test_system.py
```

## Model Training

### Automatic Training
Models are trained automatically when first requested for a symbol.

### Manual Training
```python
from services.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Train single symbol
result = trainer.train_model_for_symbol("AAPL", epochs=100)

# Batch training
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
results = trainer.batch_train_models(symbols)
```

### Model Performance
- **Accuracy**: Typically 70-85% direction prediction
- **Training Time**: 2-5 minutes per symbol
- **Memory Usage**: ~500MB per model
- **Update Frequency**: Daily recommended

## Production Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.yml up -d

# Scale API service
docker-compose up --scale api=3
```

### Environment Variables
```bash
# Production settings
DEBUG=False
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379

# Model settings
MODEL_UPDATE_INTERVAL=3600  # 1 hour
PREDICTION_HORIZON=30       # 30 days
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade tensorflow pandas numpy yfinance
   ```

2. **Memory Issues**
   - Reduce batch size in model training
   - Use smaller sequence lengths
   - Train fewer symbols simultaneously

3. **Data Issues**
   - Check internet connection
   - Verify symbol names are correct
   - Some symbols may not have options data

4. **Model Training Slow**
   - Reduce epochs for testing
   - Use GPU if available
   - Consider using pre-trained models

### Performance Optimization

1. **Model Caching**
   - Models are automatically saved and loaded
   - Use Redis for prediction caching

2. **Data Caching**
   - Historical data is cached locally
   - Real-time data has short TTL

3. **Batch Processing**
   - Use batch endpoints for multiple symbols
   - Implement async processing for large requests

## Advanced Features

### Custom Models
Extend the `LSTMStockPredictor` class to implement:
- Transformer models
- Ensemble methods
- Feature engineering
- Custom loss functions

### Risk Management
- Position sizing algorithms
- Portfolio optimization
- VaR calculations
- Stress testing

### Integration
- Trading platform APIs
- Real-time data feeds
- Alert systems
- Backtesting frameworks

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite
3. Review API documentation at `/docs`
4. Check logs for detailed error messages

## License
This project is for educational and research purposes. Use at your own risk for actual trading decisions.