# 🚀 Vercel Deployment Guide - AI Stock Prediction System

## Overview
This guide covers deploying the AI Stock Price Prediction & Options Trading system to Vercel as a serverless application.

## 🏗️ Architecture for Vercel

### Serverless Structure
```
stock_price_prediction/
├── api/
│   └── index.py              # Serverless FastAPI function
├── frontend/
│   └── index.html            # Static frontend
├── vercel.json               # Vercel configuration
├── requirements.txt          # Python dependencies
└── package.json              # Node.js metadata
```

### Key Changes for Vercel
1. **Simplified ML Models**: Replaced heavy LSTM with lightweight prediction algorithms
2. **Serverless Functions**: FastAPI app adapted for Vercel's Python runtime
3. **Static Frontend**: HTML/JS frontend served as static files
4. **Optimized Dependencies**: Minimal package requirements for faster cold starts

## 🚀 Quick Deploy

### Option 1: Deploy with Vercel CLI
```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to project
cd stock_price_prediction

# Deploy
vercel --prod
```

### Option 2: Deploy via GitHub
1. Push code to GitHub repository
2. Connect repository to Vercel dashboard
3. Auto-deploy on push

### Option 3: Deploy via Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Import project from Git or upload folder
3. Configure and deploy

## ⚙️ Configuration

### Environment Variables
Set these in Vercel dashboard or via CLI:

```bash
# Optional API keys (yfinance works without them)
vercel env add ALPHA_VANTAGE_API_KEY
vercel env add POLYGON_API_KEY
vercel env add FINNHUB_API_KEY

# Python version
vercel env add PYTHON_VERSION 3.9
```

### vercel.json Configuration
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    },
    {
      "src": "frontend/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/index.py"
    },
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ],
  "functions": {
    "api/index.py": {
      "maxDuration": 30
    }
  }
}
```

## 📡 API Endpoints (Serverless)

### Stock Predictions
- `POST /api/predictions/predict` - Generate price predictions
- `GET /api/stocks/data/{symbol}` - Historical stock data
- `GET /api/stocks/realtime/{symbol}` - Real-time price data
- `GET /api/batch-predict?symbols=AAPL,GOOGL&days_ahead=5` - Batch predictions

### Options Trading
- `POST /api/options/price` - Calculate option prices and Greeks
- All Black-Scholes calculations with full Greeks support

### Example Usage
```javascript
// Predict stock price
const response = await fetch('/api/predictions/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'AAPL',
    days_ahead: 5,
    model_type: 'simple'
  })
});

// Calculate option price
const optionResponse = await fetch('/api/options/price', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'AAPL',
    strike_price: 155,
    expiry_days: 30,
    option_type: 'call'
  })
});
```

## 🎯 Features Available on Vercel

### ✅ Fully Supported
- **Stock Price Predictions**: Simplified ML algorithms optimized for serverless
- **Options Pricing**: Complete Black-Scholes implementation with Greeks
- **Real-time Data**: yfinance integration for live market data
- **Technical Analysis**: RSI, Moving averages, volatility calculations
- **Batch Processing**: Multiple symbol analysis
- **Interactive Frontend**: Full web interface

### ⚠️ Limitations
- **Model Complexity**: Heavy LSTM models replaced with lightweight algorithms
- **Cold Start**: ~2-3 second delay on first request
- **Execution Time**: 30-second timeout per function
- **Memory**: Limited to Vercel's serverless constraints
- **Persistence**: No model training/saving (stateless functions)

## 🔧 Local Development

### Setup
```bash
# Install dependencies
npm install -g vercel
pip install -r requirements.txt

# Run locally
vercel dev
```

### Testing
```bash
# Test API endpoints
curl http://localhost:3000/api/health

# Test prediction
curl -X POST http://localhost:3000/api/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 1}'
```

## 📊 Performance Optimization

### Cold Start Optimization
- Minimal dependencies in requirements.txt
- Lazy loading of heavy libraries
- Efficient data structures
- Cached calculations where possible

### Response Time
- **First request**: ~2-3 seconds (cold start)
- **Subsequent requests**: ~200-500ms
- **Batch requests**: ~1-2 seconds

### Memory Usage
- **Base function**: ~50MB
- **With data processing**: ~100-150MB
- **Peak usage**: ~200MB

## 🚀 Production Considerations

### Scaling
- Automatic scaling with Vercel
- No server management required
- Pay-per-execution model

### Monitoring
- Built-in Vercel analytics
- Function logs in dashboard
- Error tracking and alerts

### Security
- HTTPS by default
- Environment variable encryption
- CORS configuration included

### Rate Limiting
- Vercel's built-in limits apply
- Consider implementing custom rate limiting for heavy usage

## 🔄 CI/CD Pipeline

### Automatic Deployment
```yaml
# .github/workflows/deploy.yml
name: Deploy to Vercel
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

## 📈 Usage Examples

### Frontend Integration
The system automatically detects the environment and uses the correct API endpoints:

```javascript
// Automatically uses correct base URL
const API_BASE = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000/api' 
  : '/api';
```

### Custom Integration
```python
import requests

# Your Vercel deployment URL
BASE_URL = "https://your-app.vercel.app/api"

# Get stock prediction
response = requests.post(f"{BASE_URL}/predictions/predict", 
  json={"symbol": "AAPL", "days_ahead": 5})
prediction = response.json()

# Calculate option price
option_response = requests.post(f"{BASE_URL}/options/price",
  json={
    "symbol": "AAPL",
    "strike_price": 155,
    "expiry_days": 30,
    "option_type": "call"
  })
option_data = option_response.json()
```

## 🐛 Troubleshooting

### Common Issues

1. **Cold Start Timeouts**
   - Reduce dependencies
   - Optimize function code
   - Consider function warming

2. **Memory Limits**
   - Process data in chunks
   - Use streaming for large datasets
   - Optimize pandas operations

3. **API Rate Limits**
   - Implement caching
   - Use batch endpoints
   - Add retry logic

### Debug Mode
```bash
# Local debugging
vercel dev --debug

# Check function logs
vercel logs
```

## 🎉 Success Metrics

After deployment, you should see:
- ✅ Frontend accessible at your Vercel URL
- ✅ API endpoints responding at `/api/*`
- ✅ Stock predictions working
- ✅ Options calculations functional
- ✅ Real-time data integration active

## 🔗 Useful Links

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/guides/deploying-fastapi-with-vercel)

Your AI Stock Prediction system is now ready for global deployment with Vercel's edge network! 🚀