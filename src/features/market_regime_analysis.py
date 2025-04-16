import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeAnalyzer:
    def __init__(self):
        self.features = {}
        
    def identify_market_regimes(self, prices: np.ndarray,
                              window: int = 20,
                              threshold: float = 0.1) -> np.ndarray:
        """Identify different market regimes using volatility and trend analysis"""
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window).std()
        trend = pd.Series(prices).rolling(window).mean()
        
        # Regime classification
        regime = np.zeros(len(prices))
        regime[volatility > threshold] = 1  # High volatility
        regime[trend.diff() > 0] += 2  # Uptrend
        regime[trend.diff() < 0] += 4  # Downtrend
        
        return regime
    
    def calculate_volatility_regimes(self, prices: np.ndarray,
                                   window: int = 20) -> Dict:
        """Calculate detailed volatility regimes"""
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window).std()
        
        # Volatility regimes
        low_vol = volatility < volatility.quantile(0.25)
        medium_vol = (volatility >= volatility.quantile(0.25)) & (volatility <= volatility.quantile(0.75))
        high_vol = volatility > volatility.quantile(0.75)
        
        return {
            'low_volatility': low_vol,
            'medium_volatility': medium_vol,
            'high_volatility': high_vol,
            'volatility_level': volatility
        }
    
    def calculate_trend_regimes(self, prices: np.ndarray,
                              window: int = 20) -> Dict:
        """Calculate detailed trend regimes"""
        trend = pd.Series(prices).rolling(window).mean()
        trend_direction = trend.diff()
        
        # Trend regimes
        uptrend = trend_direction > 0
        downtrend = trend_direction < 0
        sideways = trend_direction.abs() < trend_direction.abs().mean() * 0.1
        
        return {
            'uptrend': uptrend,
            'downtrend': downtrend,
            'sideways': sideways,
            'trend_strength': trend_direction
        }
    
    def calculate_market_phases(self, prices: np.ndarray,
                              window: int = 20) -> Dict:
        """Calculate market phases combining volatility and trend"""
        volatility_regimes = self.calculate_volatility_regimes(prices, window)
        trend_regimes = self.calculate_trend_regimes(prices, window)
        
        # Market phases
        accumulation = trend_regimes['sideways'] & volatility_regimes['low_volatility']
        markup = trend_regimes['uptrend'] & volatility_regimes['medium_volatility']
        distribution = trend_regimes['sideways'] & volatility_regimes['high_volatility']
        markdown = trend_regimes['downtrend'] & volatility_regimes['medium_volatility']
        
        return {
            'accumulation': accumulation,
            'markup': markup,
            'distribution': distribution,
            'markdown': markdown
        } 