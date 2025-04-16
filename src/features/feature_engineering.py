import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import talib
from textblob import TextBlob
import tweepy
import praw
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.feature_selector = SelectKBest(score_func=f_regression, k=20)
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators to the dataframe"""
        # Price-based indicators
        df['RSI'] = talib.RSI(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
        
        # Volume-based indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Momentum indicators
        df['MOM'] = talib.MOM(df['Close'])
        df['ROC'] = talib.ROC(df['Close'])
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        
        # Volatility indicators
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'])
        
        # Trend indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
        
        return df
    
    def add_fundamental_features(self, df: pd.DataFrame, 
                               financial_data: Dict) -> pd.DataFrame:
        """Add fundamental analysis features"""
        # Financial ratios
        df['PE_Ratio'] = financial_data['price'] / financial_data['eps']
        df['PB_Ratio'] = financial_data['price'] / financial_data['book_value']
        df['PS_Ratio'] = financial_data['price'] / financial_data['revenue']
        df['Debt_to_Equity'] = financial_data['debt'] / financial_data['equity']
        
        # Growth metrics
        df['Revenue_Growth'] = financial_data['revenue_growth']
        df['EPS_Growth'] = financial_data['eps_growth']
        df['ROE'] = financial_data['roe']
        df['ROA'] = financial_data['roa']
        
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame, 
                             news_data: List[Dict]) -> pd.DataFrame:
        """Add sentiment analysis features from news and social media"""
        # News sentiment
        news_sentiments = []
        for article in news_data:
            blob = TextBlob(article['text'])
            news_sentiments.append(blob.sentiment.polarity)
        df['News_Sentiment'] = np.mean(news_sentiments)
        
        # Social media sentiment
        twitter_sentiments = self._get_twitter_sentiment(df.index[-1])
        reddit_sentiments = self._get_reddit_sentiment(df.index[-1])
        
        df['Twitter_Sentiment'] = np.mean(twitter_sentiments)
        df['Reddit_Sentiment'] = np.mean(reddit_sentiments)
        
        return df
    
    def _get_twitter_sentiment(self, date: datetime) -> List[float]:
        """Get sentiment from Twitter data"""
        # Implementation for Twitter API
        sentiments = []
        # ... Twitter API calls and sentiment analysis
        return sentiments
    
    def _get_reddit_sentiment(self, date: datetime) -> List[float]:
        """Get sentiment from Reddit data"""
        # Implementation for Reddit API
        sentiments = []
        # ... Reddit API calls and sentiment analysis
        return sentiments
    
    def add_market_features(self, df: pd.DataFrame, 
                          market_data: Dict) -> pd.DataFrame:
        """Add market-wide features"""
        # Market indices correlation
        for index, data in market_data.items():
            df[f'{index}_Correlation'] = df['Close'].rolling(window=20).corr(data)
        
        # Market volatility
        df['VIX'] = market_data['vix']
        df['Market_Volume'] = market_data['volume']
        
        # Sector performance
        df['Sector_Return'] = market_data['sector_return']
        df['Sector_Volatility'] = market_data['sector_volatility']
        
        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess and select the most important features"""
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Feature selection
        selected_features = self.feature_selector.fit_transform(
            pca_features, df['Target']
        )
        
        # Get feature names
        feature_names = df.columns[self.feature_selector.get_support()]
        
        return selected_features, feature_names
    
    def create_sequences(self, data: np.ndarray, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features for prediction"""
        # Fourier transform for seasonality
        fft = np.fft.fft(df['Close'])
        df['FFT_Amplitude'] = np.abs(fft)
        df['FFT_Phase'] = np.angle(fft)
        
        # Wavelet transform for multi-scale analysis
        # ... Implementation of wavelet transform
        
        # Entropy-based features
        df['Price_Entropy'] = self._calculate_entropy(df['Close'])
        df['Volume_Entropy'] = self._calculate_entropy(df['Volume'])
        
        # Fractal features
        df['Hurst_Exponent'] = self._calculate_hurst(df['Close'])
        
        return df
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a time series"""
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))
    
    def _calculate_hurst(self, series: pd.Series) -> float:
        """Calculate Hurst exponent for fractal analysis"""
        lags = range(2, 100)
        tau = [np.sqrt(np.var(np.subtract(series[lag:], series[:-lag]))) 
               for lag in lags]
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        return m[0] / 2.0 