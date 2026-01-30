#!/usr/bin/env bash
# Render build script

set -o errexit  # exit on error

echo "🚀 Starting build process..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Try to install full requirements, fallback to minimal if needed
if pip install -r requirements.txt; then
    echo "✅ Full dependencies installed"
    export APP_MODE="full"
else
    echo "⚠️  Full dependencies failed, installing minimal set..."
    pip install fastapi uvicorn pydantic
    export APP_MODE="lightweight"
fi

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models/saved
mkdir -p logs

# Run basic validation
echo "🧪 Running validation..."
python -c "
import sys
try:
    import fastapi
    import pydantic
    print('✅ Core FastAPI modules available')
    
    # Try to import our modules
    try:
        from src.services.data_collector import DataCollector
        from src.services.options_calculator import OptionsCalculator
        print('✅ Full system modules imported successfully')
        mode = 'full'
    except ImportError as e:
        print(f'⚠️  Full system import failed: {e}')
        print('✅ Will use lightweight API mode')
        mode = 'lightweight'
        
    # Test API import
    try:
        from api.index import app
        print('✅ Lightweight API available')
    except ImportError as e:
        print(f'❌ API import failed: {e}')
        sys.exit(1)
        
except ImportError as e:
    print(f'❌ Critical dependency missing: {e}')
    sys.exit(1)
"

echo "✅ Build completed successfully!"
echo "💡 Use deploy-render.sh to start the server"