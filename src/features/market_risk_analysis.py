import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MarketRiskAnalyzer:
    def __init__(self):
        self.features = {}
        
    def calculate_value_at_risk(self, prices: np.ndarray,
                              confidence_level: float = 0.95) -> Dict:
        """Calculate Value at Risk (VaR) metrics"""
        returns = np.diff(np.log(prices))
        
        # Historical VaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (ES)
        es_95 = np.mean(returns[returns <= var_95])
        es_99 = np.mean(returns[returns <= var_99])
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99
        }
    
    def calculate_drawdown_metrics(self, prices: np.ndarray) -> Dict:
        """Calculate drawdown-related metrics"""
        # Maximum drawdown
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Drawdown duration
        drawdown_duration = np.zeros(len(prices))
        for i in range(1, len(prices)):
            if prices[i] < cummax[i]:
                drawdown_duration[i] = drawdown_duration[i-1] + 1
                
        return {
            'max_drawdown': max_drawdown,
            'drawdown_series': drawdown,
            'drawdown_duration': drawdown_duration
        }
    
    def calculate_risk_adjusted_returns(self, prices: np.ndarray,
                                      risk_free_rate: float = 0.02) -> Dict:
        """Calculate risk-adjusted return metrics"""
        returns = np.diff(np.log(prices))
        
        # Sharpe Ratio
        sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns)
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = (np.mean(returns) - risk_free_rate) / np.std(downside_returns)
        
        # Information Ratio
        market_returns = np.diff(np.log(prices))  # Using same returns as benchmark
        tracking_error = np.std(returns - market_returns)
        information_ratio = (np.mean(returns) - np.mean(market_returns)) / tracking_error
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'information_ratio': information_ratio
        }
    
    def calculate_volatility_metrics(self, prices: np.ndarray,
                                   window: int = 20) -> Dict:
        """Calculate volatility-related metrics"""
        returns = np.diff(np.log(prices))
        
        # Historical volatility
        historical_vol = pd.Series(returns).rolling(window).std()
        
        # Volatility clustering
        vol_clusters = np.zeros(len(returns))
        vol_mean = historical_vol.mean()
        vol_std = historical_vol.std()
        vol_clusters[historical_vol > vol_mean + vol_std] = 1
        vol_clusters[historical_vol < vol_mean - vol_std] = -1
        
        # Volatility persistence
        vol_persistence = pd.Series(returns).rolling(window).autocorr()
        
        return {
            'historical_volatility': historical_vol,
            'volatility_clusters': vol_clusters,
            'volatility_persistence': vol_persistence
        }
    
    def calculate_tail_risk_metrics(self, prices: np.ndarray) -> Dict:
        """Calculate tail risk metrics"""
        returns = np.diff(np.log(prices))
        
        # Skewness and Kurtosis
        skewness = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        # Tail risk measures
        left_tail = np.percentile(returns, 1)
        right_tail = np.percentile(returns, 99)
        tail_ratio = abs(right_tail / left_tail)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'left_tail': left_tail,
            'right_tail': right_tail,
            'tail_ratio': tail_ratio
        }
    
    def calculate_correlation_risk(self, prices: np.ndarray,
                                 market_index: np.ndarray,
                                 window: int = 20) -> Dict:
        """Calculate correlation-based risk metrics"""
        returns = np.diff(np.log(prices))
        market_returns = np.diff(np.log(market_index))
        
        # Rolling correlation
        rolling_corr = pd.Series(returns).rolling(window).corr(pd.Series(market_returns))
        
        # Beta
        beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
        
        # Correlation breakdown
        corr_breakdown = np.zeros(len(returns))
        corr_breakdown[rolling_corr > rolling_corr.mean() + rolling_corr.std()] = 1
        corr_breakdown[rolling_corr < rolling_corr.mean() - rolling_corr.std()] = -1
        
        return {
            'rolling_correlation': rolling_corr,
            'beta': beta,
            'correlation_breakdown': corr_breakdown
        } 