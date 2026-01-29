"""
Simple HTTP Server for Stock Prediction System
Works without heavy dependencies
"""
import json
import http.server
import socketserver
import urllib.parse
from datetime import datetime
import os
import webbrowser
from simple_demo import SimpleStockPredictor, SimpleOptionsCalculator

class StockPredictionHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.predictor = SimpleStockPredictor()
        self.calculator = SimpleOptionsCalculator()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve the main HTML file
            self.path = '/frontend/index.html'
        
        # Handle API endpoints
        if self.path.startswith('/api/'):
            self.handle_api_request()
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path.startswith('/api/'):
            self.handle_api_request()
        else:
            self.send_error(404)
    
    def handle_api_request(self):
        """Handle API requests"""
        try:
            # Parse the URL
            parsed_path = urllib.parse.urlparse(self.path)
            path_parts = parsed_path.path.split('/')
            
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            response_data = {}
            
            # Route API requests
            if len(path_parts) >= 4:
                endpoint_type = path_parts[2]  # stocks, predictions, options
                endpoint_action = path_parts[3]  # specific action
                
                if endpoint_type == 'predictions' and endpoint_action == 'predict':
                    response_data = self.handle_prediction_request()
                elif endpoint_type == 'options' and endpoint_action == 'price':
                    response_data = self.handle_options_request()
                elif endpoint_type == 'stocks' and endpoint_action == 'realtime':
                    symbol = path_parts[4] if len(path_parts) > 4 else 'AAPL'
                    response_data = self.handle_stock_data_request(symbol)
                else:
                    response_data = {"error": "Endpoint not found"}
            else:
                response_data = {"error": "Invalid API path"}
            
            # Send JSON response
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def handle_prediction_request(self):
        """Handle stock prediction requests"""
        try:
            # Get POST data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode())
            else:
                request_data = {"symbol": "AAPL", "days_ahead": 1}
            
            symbol = request_data.get('symbol', 'AAPL').upper()
            days_ahead = request_data.get('days_ahead', 1)
            
            # Generate prediction
            result = self.predictor.predict_price(symbol, days_ahead)
            
            return {
                "symbol": result["symbol"],
                "predictions": result["predictions"],
                "confidence_score": result["confidence_score"],
                "model_accuracy": 0.85,  # Mock accuracy
                "timestamp": result["timestamp"]
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def handle_options_request(self):
        """Handle options pricing requests"""
        try:
            # Get POST data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode())
            else:
                request_data = {
                    "symbol": "AAPL",
                    "strike_price": 150,
                    "expiry_days": 30,
                    "option_type": "call"
                }
            
            symbol = request_data.get('symbol', 'AAPL').upper()
            strike_price = float(request_data.get('strike_price', 150))
            expiry_days = int(request_data.get('expiry_days', 30))
            option_type = request_data.get('option_type', 'call')
            
            # Get mock current price
            current_price = strike_price * (0.95 + 0.1 * hash(symbol) % 100 / 100)
            
            # Calculate option price
            time_to_expiry = expiry_days / 365.0
            volatility = 0.25  # Mock volatility
            
            if option_type.lower() == 'call':
                option_price = self.calculator.black_scholes_call(
                    current_price, strike_price, time_to_expiry, 0.05, volatility
                )
            else:
                option_price = self.calculator.black_scholes_put(
                    current_price, strike_price, time_to_expiry, 0.05, volatility
                )
            
            # Calculate Greeks
            greeks = self.calculator.calculate_greeks(
                current_price, strike_price, time_to_expiry, 0.05, volatility, option_type
            )
            
            return {
                "symbol": symbol,
                "current_stock_price": round(current_price, 2),
                "strike_price": strike_price,
                "option_type": option_type,
                "theoretical_price": option_price,
                "volatility_used": volatility,
                "time_to_expiry_days": expiry_days,
                "greeks": greeks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Options calculation error: {str(e)}"}
    
    def handle_stock_data_request(self, symbol):
        """Handle stock data requests"""
        try:
            # Generate mock real-time data
            base_price = 100 + (hash(symbol) % 200)
            
            return {
                "symbol": symbol,
                "current_price": round(base_price * (0.95 + 0.1 * hash(datetime.now().second) % 100 / 100), 2),
                "previous_close": round(base_price, 2),
                "open": round(base_price * 0.99, 2),
                "day_high": round(base_price * 1.02, 2),
                "day_low": round(base_price * 0.98, 2),
                "volume": hash(symbol) % 10000000 + 1000000,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Stock data error: {str(e)}"}

def start_server(port=8000):
    """Start the HTTP server"""
    try:
        # Change to the project directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Create server
        with socketserver.TCPServer(("", port), StockPredictionHandler) as httpd:
            print(f"🚀 Stock Prediction Server Starting...")
            print(f"📊 Server running at: http://localhost:{port}")
            print(f"🌐 Frontend: http://localhost:{port}/frontend/index.html")
            print(f"📖 API Base: http://localhost:{port}/api/")
            print(f"⏹️  Press Ctrl+C to stop the server")
            
            # Try to open browser
            try:
                webbrowser.open(f"http://localhost:{port}/frontend/index.html")
                print("🌍 Browser opened automatically")
            except:
                print("💡 Please open your browser manually")
            
            print("\n" + "="*60)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_server()