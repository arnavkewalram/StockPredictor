import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MarketMicrostructureAnalyzer:
    def __init__(self):
        self.features = {}
        
    def analyze_price_impact(self, prices: np.ndarray,
                           volume: np.ndarray) -> Dict:
        """Analyze price impact of trades"""
        returns = np.diff(np.log(prices))
        price_impact = np.abs(returns) / volume[1:]
        
        # Price impact metrics
        avg_price_impact = np.mean(price_impact)
        max_price_impact = np.max(price_impact)
        price_impact_std = np.std(price_impact)
        
        return {
            'price_impact_series': price_impact,
            'average_price_impact': avg_price_impact,
            'max_price_impact': max_price_impact,
            'price_impact_std': price_impact_std
        }
    
    def analyze_order_flow(self, volume: np.ndarray,
                         window: int = 20) -> Dict:
        """Analyze order flow characteristics"""
        # Order flow imbalance
        ofi = np.diff(volume) / volume[1:]
        
        # Volume profile
        volume_profile = pd.Series(volume).rolling(window).mean()
        
        # Order flow persistence
        of_persistence = pd.Series(ofi).rolling(window).autocorr()
        
        return {
            'order_flow_imbalance': ofi,
            'volume_profile': volume_profile,
            'order_flow_persistence': of_persistence
        }
    
    def analyze_liquidity(self, prices: np.ndarray,
                        volume: np.ndarray) -> Dict:
        """Analyze market liquidity"""
        # Spread calculation
        spread = np.diff(prices) / prices[:-1]
        
        # Liquidity measures
        liquidity = 1 / (spread * volume[1:])
        illiquidity = spread * volume[1:]
        
        # Liquidity metrics
        avg_liquidity = np.mean(liquidity)
        liquidity_std = np.std(liquidity)
        liquidity_ratio = avg_liquidity / liquidity_std
        
        return {
            'liquidity_series': liquidity,
            'illiquidity_series': illiquidity,
            'average_liquidity': avg_liquidity,
            'liquidity_std': liquidity_std,
            'liquidity_ratio': liquidity_ratio
        }
    
    def analyze_market_depth(self, prices: np.ndarray,
                           volume: np.ndarray,
                           window: int = 20) -> Dict:
        """Analyze market depth characteristics"""
        # Price levels
        price_levels = np.unique(prices)
        
        # Volume at price levels
        volume_at_price = {}
        for price in price_levels:
            volume_at_price[price] = np.sum(volume[prices == price])
            
        # Depth profile
        depth_profile = pd.Series(volume).rolling(window).sum()
        
        # Depth concentration
        depth_concentration = np.sum(volume_at_price.values()) / len(price_levels)
        
        return {
            'price_levels': price_levels,
            'volume_at_price': volume_at_price,
            'depth_profile': depth_profile,
            'depth_concentration': depth_concentration
        }
    
    def analyze_trade_characteristics(self, prices: np.ndarray,
                                   volume: np.ndarray) -> Dict:
        """Analyze trade characteristics"""
        # Trade size distribution
        trade_sizes = volume[volume > 0]
        avg_trade_size = np.mean(trade_sizes)
        trade_size_std = np.std(trade_sizes)
        
        # Trade frequency
        trade_frequency = len(trade_sizes) / len(prices)
        
        # Trade clustering
        trade_clusters = np.zeros(len(volume))
        trade_clusters[volume > avg_trade_size + trade_size_std] = 1
        trade_clusters[volume < avg_trade_size - trade_size_std] = -1
        
        return {
            'average_trade_size': avg_trade_size,
            'trade_size_std': trade_size_std,
            'trade_frequency': trade_frequency,
            'trade_clusters': trade_clusters
        }
    
    def analyze_market_quality(self, prices: np.ndarray,
                             volume: np.ndarray,
                             window: int = 20) -> Dict:
        """Analyze overall market quality"""
        # Price efficiency
        returns = np.diff(np.log(prices))
        price_efficiency = 1 - np.abs(pd.Series(returns).autocorr())
        
        # Market resilience
        price_reversion = np.mean(np.abs(returns))
        market_resilience = 1 / (1 + price_reversion)
        
        # Market fragmentation
        volume_concentration = np.sum(volume**2) / (np.sum(volume)**2)
        market_fragmentation = 1 - volume_concentration
        
        return {
            'price_efficiency': price_efficiency,
            'market_resilience': market_resilience,
            'market_fragmentation': market_fragmentation
        } 