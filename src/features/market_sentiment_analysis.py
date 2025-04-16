import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class MarketSentimentAnalyzer:
    def __init__(self):
        self.features = {}
        
    def analyze_news_sentiment(self, ticker: str) -> Dict:
        """Analyze sentiment from financial news"""
        stock = yf.Ticker(ticker)
        news = stock.news
        
        # News sentiment analysis
        sentiments = []
        for article in news:
            sentiment = article.get('sentiment', 0)
            sentiments.append(sentiment)
            
        return {
            'average_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'positive_articles': sum(1 for s in sentiments if s > 0),
            'negative_articles': sum(1 for s in sentiments if s < 0)
        }
    
    def analyze_options_sentiment(self, ticker: str) -> Dict:
        """Analyze sentiment from options market"""
        stock = yf.Ticker(ticker)
        
        # Options data
        put_call_ratio = stock.info.get('putCallRatio', 1.0)
        options = stock.options
        
        # Options sentiment
        options_sentiment = {
            'put_call_ratio': put_call_ratio,
            'options_volume': stock.info.get('optionsVolume', 0),
            'options_open_interest': stock.info.get('optionsOpenInterest', 0)
        }
        
        return options_sentiment
    
    def analyze_analyst_sentiment(self, ticker: str) -> Dict:
        """Analyze sentiment from analyst recommendations"""
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations.empty:
            return {
                'average_rating': 0,
                'rating_std': 0,
                'num_analysts': 0
            }
            
        # Convert ratings to numerical values
        rating_map = {
            'Strong Buy': 2,
            'Buy': 1,
            'Hold': 0,
            'Sell': -1,
            'Strong Sell': -2
        }
        
        ratings = recommendations['To Grade'].map(rating_map)
        
        return {
            'average_rating': ratings.mean(),
            'rating_std': ratings.std(),
            'num_analysts': len(ratings)
        }
    
    def analyze_social_sentiment(self, ticker: str) -> Dict:
        """Analyze sentiment from social media (placeholder for API integration)"""
        # Note: This would require API access to Twitter, Reddit, etc.
        return {
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'social_volume': 0
        }
    
    def calculate_composite_sentiment(self, ticker: str) -> Dict:
        """Calculate composite sentiment score combining all sources"""
        news_sentiment = self.analyze_news_sentiment(ticker)
        options_sentiment = self.analyze_options_sentiment(ticker)
        analyst_sentiment = self.analyze_analyst_sentiment(ticker)
        social_sentiment = self.analyze_social_sentiment(ticker)
        
        # Weighted average of different sentiment sources
        weights = {
            'news': 0.3,
            'options': 0.2,
            'analyst': 0.3,
            'social': 0.2
        }
        
        composite_score = (
            weights['news'] * news_sentiment['average_sentiment'] +
            weights['options'] * (1 - options_sentiment['put_call_ratio']) +
            weights['analyst'] * analyst_sentiment['average_rating'] +
            weights['social'] * (social_sentiment['twitter_sentiment'] + social_sentiment['reddit_sentiment']) / 2
        )
        
        return {
            'composite_score': composite_score,
            'news_sentiment': news_sentiment,
            'options_sentiment': options_sentiment,
            'analyst_sentiment': analyst_sentiment,
            'social_sentiment': social_sentiment
        } 