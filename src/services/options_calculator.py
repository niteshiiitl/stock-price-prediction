"""
Options Pricing and Greeks Calculator
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple
import math

class OptionsCalculator:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks
        
        Returns:
            Dictionary containing Delta, Gamma, Theta, Vega, Rho
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, market_price: float, S: float, K: float, T: float, r: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        sigma = 0.2  # Initial guess
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            if option_type.lower() == 'call':
                price = self.black_scholes_call(S, K, T, r, sigma)
            else:
                price = self.black_scholes_put(S, K, T, r, sigma)
            
            # Vega for Newton-Raphson
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            price_diff = price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            if vega == 0:
                break
                
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive
            sigma = max(sigma, 0.001)
        
        return sigma
    
    def calculate_portfolio_greeks(self, positions: list) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks
        
        Args:
            positions: List of dictionaries with position details
        """
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for position in positions:
            greeks = self.calculate_greeks(
                position['stock_price'],
                position['strike'],
                position['time_to_expiry'],
                position.get('risk_free_rate', self.risk_free_rate),
                position['volatility'],
                position['option_type']
            )
            
            quantity = position['quantity']
            
            total_delta += greeks['delta'] * quantity
            total_gamma += greeks['gamma'] * quantity
            total_theta += greeks['theta'] * quantity
            total_vega += greeks['vega'] * quantity
            total_rho += greeks['rho'] * quantity
        
        return {
            'portfolio_delta': total_delta,
            'portfolio_gamma': total_gamma,
            'portfolio_theta': total_theta,
            'portfolio_vega': total_vega,
            'portfolio_rho': total_rho
        }
    
    def calculate_profit_loss(self, positions: list, new_stock_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate P&L for options positions"""
        total_pnl = 0
        position_pnls = {}
        
        for i, position in enumerate(positions):
            symbol = position['symbol']
            new_price = new_stock_prices.get(symbol, position['stock_price'])
            
            # Calculate new option price
            if position['option_type'].lower() == 'call':
                new_option_price = self.black_scholes_call(
                    new_price,
                    position['strike'],
                    position['time_to_expiry'],
                    position.get('risk_free_rate', self.risk_free_rate),
                    position['volatility']
                )
            else:
                new_option_price = self.black_scholes_put(
                    new_price,
                    position['strike'],
                    position['time_to_expiry'],
                    position.get('risk_free_rate', self.risk_free_rate),
                    position['volatility']
                )
            
            # Calculate P&L
            pnl = (new_option_price - position['entry_price']) * position['quantity'] * 100
            position_pnls[f'position_{i}'] = pnl
            total_pnl += pnl
        
        return {
            'total_pnl': total_pnl,
            'position_pnls': position_pnls
        }
    
    def monte_carlo_option_pricing(self, S: float, K: float, T: float, r: float, sigma: float, 
                                 option_type: str = 'call', num_simulations: int = 10000) -> Dict[str, float]:
        """
        Monte Carlo simulation for option pricing
        """
        dt = T / 252  # Daily steps
        num_steps = int(T * 252)
        
        # Generate random paths
        Z = np.random.standard_normal((num_simulations, num_steps))
        
        # Initialize price paths
        S_paths = np.zeros((num_simulations, num_steps + 1))
        S_paths[:, 0] = S
        
        # Generate stock price paths
        for t in range(1, num_steps + 1):
            S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return {
            'option_price': option_price,
            'standard_error': np.std(payoffs) / np.sqrt(num_simulations),
            'confidence_interval_95': [
                option_price - 1.96 * np.std(payoffs) / np.sqrt(num_simulations),
                option_price + 1.96 * np.std(payoffs) / np.sqrt(num_simulations)
            ]
        }