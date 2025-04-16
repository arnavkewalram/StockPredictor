import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
import yfinance as yf
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class MarketAnalyzer:
    def __init__(self):
        self.features = {}
        self.indicators = {}
        
    def calculate_market_regime(self, prices: np.ndarray,
                              window: int = 20,
                              threshold: float = 0.1) -> np.ndarray:
        """Calculate market regime using volatility and trend"""
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window).std()
        trend = pd.Series(prices).rolling(window).mean()
        
        # Regime classification
        regime = np.zeros(len(prices))
        regime[volatility > threshold] = 1  # High volatility
        regime[trend.diff() > 0] += 2  # Uptrend
        regime[trend.diff() < 0] += 4  # Downtrend
        
        return regime
    
    def calculate_market_sentiment(self, ticker: str,
                                 window: int = 5) -> Dict:
        """Calculate market sentiment using multiple sources"""
        stock = yf.Ticker(ticker)
        
        # News sentiment
        news = stock.news
        news_sentiment = np.mean([n.get('sentiment', 0) for n in news])
        
        # Social media sentiment
        # Note: This would require API access to Twitter, Reddit, etc.
        social_sentiment = 0.0  # Placeholder
        
        # Options sentiment
        options = stock.options
        put_call_ratio = stock.info.get('putCallRatio', 1.0)
        
        # Analyst sentiment
        analyst_ratings = stock.recommendations
        if not analyst_ratings.empty:
            analyst_sentiment = analyst_ratings['To Grade'].mean()
        else:
            analyst_sentiment = 0.0
            
        return {
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'put_call_ratio': put_call_ratio,
            'analyst_sentiment': analyst_sentiment
        }
    
    def calculate_market_microstructure(self, prices: np.ndarray,
                                      volume: np.ndarray,
                                      window: int = 20) -> Dict:
        """Calculate market microstructure features"""
        # Price impact
        returns = np.diff(np.log(prices))
        price_impact = np.abs(returns) / volume[1:]
        
        # Order flow imbalance
        ofi = np.diff(volume) / volume[1:]
        
        # Volume profile
        volume_profile = pd.Series(volume).rolling(window).mean()
        
        # Liquidity measures
        spread = np.diff(prices) / prices[:-1]
        liquidity = 1 / (spread * volume[1:])
        
        return {
            'price_impact': price_impact,
            'order_flow_imbalance': ofi,
            'volume_profile': volume_profile,
            'liquidity': liquidity
        }
    
    def calculate_market_anomalies(self, prices: np.ndarray,
                                 window: int = 20) -> Dict:
        """Detect market anomalies and patterns"""
        # Momentum anomalies
        returns = np.diff(np.log(prices))
        momentum = pd.Series(returns).rolling(window).mean()
        
        # Volatility clustering
        volatility = pd.Series(returns).rolling(window).std()
        vol_clusters = find_peaks(volatility, height=volatility.mean())[0]
        
        # Price patterns
        peaks, _ = find_peaks(prices)
        troughs, _ = find_peaks(-prices)
        
        # Autocorrelation
        autocorr = pd.Series(returns).autocorr()
        
        return {
            'momentum': momentum,
            'volatility_clusters': vol_clusters,
            'price_patterns': {'peaks': peaks, 'troughs': troughs},
            'autocorrelation': autocorr
        }
    
    def calculate_market_risk_metrics(self, prices: np.ndarray,
                                    risk_free_rate: float = 0.02) -> Dict:
        """Calculate advanced risk metrics"""
        returns = np.diff(np.log(prices))
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (ES)
        es_95 = np.mean(returns[returns <= var_95])
        es_99 = np.mean(returns[returns <= var_99])
        
        # Maximum Drawdown
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Sharpe Ratio
        sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns)
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = (np.mean(returns) - risk_free_rate) / np.std(downside_returns)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino
        }
    
    def calculate_market_correlation(self, prices: np.ndarray,
                                   market_index: np.ndarray,
                                   window: int = 20) -> Dict:
        """Calculate market correlation features"""
        returns = np.diff(np.log(prices))
        market_returns = np.diff(np.log(market_index))
        
        # Rolling correlation
        rolling_corr = pd.Series(returns).rolling(window).corr(pd.Series(market_returns))
        
        # Beta calculation
        beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
        
        # Cointegration test
        coint_test = stats.coint(prices, market_index)
        
        # Granger causality
        # Note: This would require statsmodels
        granger_causality = None  # Placeholder
        
        return {
            'rolling_correlation': rolling_corr,
            'beta': beta,
            'cointegration': coint_test,
            'granger_causality': granger_causality
        }
    
    def calculate_market_volume_analysis(self, volume: np.ndarray,
                                       prices: np.ndarray,
                                       window: int = 20) -> Dict:
        """Calculate advanced volume analysis features"""
        # Volume profile
        volume_profile = pd.Series(volume).rolling(window).mean()
        
        # Volume-price relationship
        vwap = np.cumsum(volume * prices) / np.cumsum(volume)
        volume_price_corr = pd.Series(volume).rolling(window).corr(pd.Series(prices))
        
        # Volume momentum
        volume_momentum = pd.Series(volume).pct_change(window)
        
        # Volume volatility
        volume_volatility = pd.Series(volume).rolling(window).std()
        
        # Volume clustering
        volume_clusters = find_peaks(volume, height=volume.mean())[0]
        
        return {
            'volume_profile': volume_profile,
            'vwap': vwap,
            'volume_price_correlation': volume_price_corr,
            'volume_momentum': volume_momentum,
            'volume_volatility': volume_volatility,
            'volume_clusters': volume_clusters
        }
    
    def calculate_market_timing(self, prices: np.ndarray,
                              indicators: Dict,
                              window: int = 20) -> Dict:
        """Calculate market timing signals"""
        # Trend following
        ma_short = pd.Series(prices).rolling(window//2).mean()
        ma_long = pd.Series(prices).rolling(window).mean()
        trend_signal = np.where(ma_short > ma_long, 1, -1)
        
        # Momentum timing
        returns = np.diff(np.log(prices))
        momentum = pd.Series(returns).rolling(window).mean()
        momentum_signal = np.where(momentum > 0, 1, -1)
        
        # Volatility timing
        volatility = pd.Series(returns).rolling(window).std()
        vol_signal = np.where(volatility > volatility.mean(), -1, 1)
        
        # Combined timing signal
        combined_signal = trend_signal + momentum_signal + vol_signal
        
        return {
            'trend_signal': trend_signal,
            'momentum_signal': momentum_signal,
            'volatility_signal': vol_signal,
            'combined_signal': combined_signal
        }
    
    def calculate_market_cycle(self, prices: np.ndarray,
                             window: int = 20) -> Dict:
        """Calculate market cycle features"""
        # Hodrick-Prescott filter
        def hp_filter(y, lamb=1600):
            n = len(y)
            I = np.eye(n)
            D = np.diff(I, 2)
            return np.linalg.solve(I + lamb * D.T @ D, y)
            
        trend = hp_filter(prices)
        cycle = prices - trend
        
        # Cycle phase
        cycle_phase = np.arctan2(np.diff(cycle), cycle[:-1])
        
        # Cycle amplitude
        cycle_amplitude = np.abs(cycle)
        
        # Cycle frequency
        fft = np.fft.fft(cycle)
        frequencies = np.fft.fftfreq(len(cycle))
        dominant_freq = frequencies[np.argmax(np.abs(fft))]
        
        return {
            'trend': trend,
            'cycle': cycle,
            'cycle_phase': cycle_phase,
            'cycle_amplitude': cycle_amplitude,
            'dominant_frequency': dominant_freq
        }
    
    def calculate_market_fractal(self, prices: np.ndarray,
                               window: int = 20) -> Dict:
        """Calculate market fractal features"""
        # Hurst exponent
        def hurst_exponent(ts):
            lags = range(2, 100)
            tau = []; laggedvar = []
            for lag in lags:
                tau.append(lag)
                laggedvar.append(np.var(ts[lag:] - ts[:-lag]))
            m = np.polyfit(np.log(tau), np.log(laggedvar), 1)
            return m[0] / 2.0
            
        hurst = hurst_exponent(prices)
        
        # Fractal dimension
        def fractal_dimension(ts):
            n = len(ts)
            scales = np.logspace(0, np.log10(n/4), 20)
            scales = scales[scales > 1]
            scales = scales.astype(int)
            q = 2
            Fq = np.zeros(len(scales))
            for i, scale in enumerate(scales):
                reshaped = ts[:n//scale*scale].reshape(-1, scale)
                Fq[i] = np.mean(np.sum(np.abs(np.diff(reshaped, axis=1))**q, axis=1))**(1/q)
            m = np.polyfit(np.log(scales), np.log(Fq), 1)
            return -m[0]
            
        fractal_dim = fractal_dimension(prices)
        
        # Multifractal spectrum
        def multifractal_spectrum(ts, qs=np.arange(-5, 6)):
            n = len(ts)
            scales = np.logspace(0, np.log10(n/4), 20)
            scales = scales[scales > 1]
            scales = scales.astype(int)
            Fq = np.zeros((len(qs), len(scales)))
            for i, q in enumerate(qs):
                for j, scale in enumerate(scales):
                    reshaped = ts[:n//scale*scale].reshape(-1, scale)
                    Fq[i,j] = np.mean(np.sum(np.abs(np.diff(reshaped, axis=1))**q, axis=1))**(1/q)
            tau = np.zeros(len(qs))
            for i, q in enumerate(qs):
                m = np.polyfit(np.log(scales), np.log(Fq[i]), 1)
                tau[i] = m[0]
            alpha = np.gradient(tau)
            f = qs * alpha - tau
            return alpha, f
            
        alpha, f = multifractal_spectrum(prices)
        
        return {
            'hurst_exponent': hurst,
            'fractal_dimension': fractal_dim,
            'multifractal_spectrum': {'alpha': alpha, 'f': f}
        } 