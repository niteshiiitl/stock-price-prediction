#!/bin/bash

# Render Deployment Script for AI Stock Price Prediction System

set -o errexit  # exit on error

echo "🚀 Deploying AI Stock Price Prediction System to Render..."

# Install dependencies with fallback
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

if pip install -r requirements.txt; then
    echo "✅ Full dependencies installed"
    export APP_MODE="full"
else
    echo "⚠️  Full dependencies failed, installing minimal set..."
    pip install fastapi uvicorn pydantic
    export APP_MODE="lightweight"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/saved
mkdir -p logs

# Set environment variables for production
export DEBUG=false
export HOST=0.0.0.0
export PORT=${PORT:-8000}

# Validate installation
echo "🧪 Validating installation..."
python -c "
import fastapi, uvicorn, pydantic
print('✅ Core dependencies available')

try:
    if '$APP_MODE' == 'full':
        from src.main import app
        print('✅ Full system available')
    else:
        from api.index import app
        print('✅ Lightweight API available')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo "✅ Deployment preparation complete!"
echo "🌐 Starting server on $HOST:$PORT"

# Start the appropriate application
if [ "$APP_MODE" = "full" ]; then
    echo "🚀 Starting full system..."
    uvicorn src.main:app --host $HOST --port $PORT --workers 1
else
    echo "🚀 Starting lightweight API..."
    uvicorn api.index:app --host $HOST --port $PORT --workers 1
fi