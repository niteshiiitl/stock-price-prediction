"""
Quick start script for the Stock Price Prediction System
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import pandas
        import numpy
        import tensorflow
        import yfinance
        import fastapi
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    os.chdir("src")
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def open_frontend():
    """Open the frontend in browser"""
    frontend_path = Path("frontend/index.html").absolute()
    if frontend_path.exists():
        webbrowser.open(f"file://{frontend_path}")
        print(f"🌐 Frontend opened: file://{frontend_path}")
    else:
        print("❌ Frontend file not found")

def main():
    print("🎯 AI Stock Price Prediction & Options Trading System")
    print("=" * 60)
    
    if not check_requirements():
        return
    
    print("\nChoose an option:")
    print("1. Start API server only")
    print("2. Open frontend only")
    print("3. Start server and open frontend")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        start_api_server()
    elif choice == "2":
        open_frontend()
    elif choice == "3":
        print("Starting server in background...")
        print("Open http://localhost:8000 for API docs")
        open_frontend()
        start_api_server()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()