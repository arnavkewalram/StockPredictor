import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import ta
from ta.trend import IchimokuIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import WilliamsRIndicator
import yfinance as yf
import requests
from datetime import datetime, timedelta
import tweepy
from textblob import TextBlob
import praw
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatures:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        self.load_api_keys()
        
    def load_api_keys(self):
        """Load API keys from environment variables"""
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
    def add_advanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(high=data['High'], low=data['Low'])
        data['Ichimoku_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Average True Range
        atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'])
        data['ATR'] = atr.average_true_range()
        
        # Williams %R
        williams_r = WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'])
        data['Williams_R'] = williams_r.williams_r()
        
        # Fibonacci Retracement Levels
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - 0.236 * diff
        data['Fib_38.2'] = high - 0.382 * diff
        data['Fib_50.0'] = high - 0.500 * diff
        data['Fib_61.8'] = high - 0.618 * diff
        
        return data
    
    def get_options_data(self) -> pd.DataFrame:
        """Fetch and process options data"""
        try:
            options = self.stock.options
            options_data = []
            
            for expiry in options:
                opt = self.stock.option_chain(expiry)
                calls = opt.calls
                puts = opt.puts
                
                # Calculate put/call ratio
                pcr = len(puts) / len(calls)
                
                # Calculate implied volatility
                iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2
                
                options_data.append({
                    'expiry': expiry,
                    'put_call_ratio': pcr,
                    'implied_volatility': iv
                })
            
            return pd.DataFrame(options_data)
        except Exception as e:
            print(f"Error fetching options data: {str(e)}")
            return pd.DataFrame()
    
    def get_social_sentiment(self) -> Dict[str, float]:
        """Get sentiment from social media"""
        sentiment = {
            'twitter': self._get_twitter_sentiment(),
            'reddit': self._get_reddit_sentiment()
        }
        return sentiment
    
    def _get_twitter_sentiment(self) -> float:
        """Get sentiment from Twitter"""
        if not all([self.twitter_api_key, self.twitter_api_secret, 
                   self.twitter_access_token, self.twitter_access_secret]):
            return 0.0
            
        try:
            auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
            auth.set_access_token(self.twitter_access_token, self.twitter_access_secret)
            api = tweepy.API(auth)
            
            tweets = api.search_tweets(q=f"${self.symbol}", lang="en", count=100)
            sentiments = []
            
            for tweet in tweets:
                analysis = TextBlob(tweet.text)
                sentiments.append(analysis.sentiment.polarity)
            
            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            print(f"Error fetching Twitter sentiment: {str(e)}")
            return 0.0
    
    def _get_reddit_sentiment(self) -> float:
        """Get sentiment from Reddit"""
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            return 0.0
            
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='stock_analyzer'
            )
            
            subreddits = ['stocks', 'investing', 'wallstreetbets']
            sentiments = []
            
            for subreddit in subreddits:
                for submission in reddit.subreddit(subreddit).search(f"${self.symbol}", limit=50):
                    analysis = TextBlob(submission.title)
                    sentiments.append(analysis.sentiment.polarity)
            
            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            print(f"Error fetching Reddit sentiment: {str(e)}")
            return 0.0
    
    def calculate_var(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        returns = data['Close'].pct_change().dropna()
        var = norm.ppf(1 - confidence_level) * returns.std()
        return var
    
    def monte_carlo_simulation(self, data: pd.DataFrame, 
                             days: int = 30, 
                             simulations: int = 1000) -> np.ndarray:
        """Perform Monte Carlo simulation"""
        returns = data['Close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        simulations = np.zeros((days, simulations))
        last_price = data['Close'].iloc[-1]
        
        for i in range(simulations):
            price_series = [last_price]
            for j in range(days):
                price = price_series[-1] * (1 + np.random.normal(mu, sigma))
                price_series.append(price)
            simulations[:, i] = price_series[1:]
        
        return simulations
    
    def stress_test(self, data: pd.DataFrame, 
                   scenarios: Dict[str, float] = None) -> Dict[str, float]:
        """Perform stress testing"""
        if scenarios is None:
            scenarios = {
                'market_crash': -0.20,  # 20% drop
                'market_correction': -0.10,  # 10% drop
                'market_rally': 0.10,  # 10% rise
                'volatility_spike': 0.30  # 30% increase in volatility
            }
        
        results = {}
        current_price = data['Close'].iloc[-1]
        
        for scenario, change in scenarios.items():
            if 'volatility' in scenario:
                # Handle volatility scenarios differently
                returns = data['Close'].pct_change().dropna()
                new_volatility = returns.std() * (1 + change)
                results[scenario] = current_price * (1 + np.random.normal(0, new_volatility))
            else:
                results[scenario] = current_price * (1 + change)
        
        return results
    
    def correlation_analysis(self, data: pd.DataFrame, 
                           indices: List[str] = None) -> Dict[str, float]:
        """Analyze correlation with market indices"""
        if indices is None:
            indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
        
        correlations = {}
        stock_returns = data['Close'].pct_change().dropna()
        
        for index in indices:
            try:
                index_data = yf.download(index, start=data.index[0], end=data.index[-1])
                index_returns = index_data['Close'].pct_change().dropna()
                correlation = stock_returns.corr(index_returns)
                correlations[index] = correlation
            except Exception as e:
                print(f"Error calculating correlation with {index}: {str(e)}")
                correlations[index] = 0.0
        
        return correlations 