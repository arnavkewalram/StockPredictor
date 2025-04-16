import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, Tuple, List, Optional
import os
from dotenv import load_dotenv
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class DataCollector:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        load_dotenv()
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.fred_key = os.getenv('FRED_KEY')
        
    def get_historical_data(self, period: str = '5y', interval: str = '1d') -> pd.DataFrame:
        """Fetch historical stock data with technical indicators"""
        data = self.stock.history(period=period, interval=interval)
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        # Add economic indicators
        data = self._add_economic_indicators(data)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add various technical indicators to the data"""
        # Trend indicators
        data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
        data['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        
        # Momentum indicators
        data['RSI'] = RSIIndicator(close=data['Close']).rsi()
        data['MACD'] = MACD(close=data['Close']).macd()
        data['MACD_Signal'] = MACD(close=data['Close']).macd_signal()
        data['Stochastic_K'] = StochasticOscillator(high=data['High'], low=data['Low'], 
                                                  close=data['Close']).stoch()
        data['Stochastic_D'] = StochasticOscillator(high=data['High'], low=data['Low'], 
                                                  close=data['Close']).stoch_signal()
        
        # Volatility indicators
        bollinger = BollingerBands(close=data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        data['BB_Middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        data['VWAP'] = VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], 
                                                close=data['Close'], volume=data['Volume']).volume_weighted_average_price()
        
        return data
    
    def _add_economic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add economic indicators to the data"""
        if not self.fred_key:
            return data
            
        # Fetch economic indicators from FRED
        indicators = {
            'GDP': 'GDP',
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'CPI',
            'FEDFUNDS': 'Federal Funds Rate'
        }
        
        for code, name in indicators.items():
            try:
                url = f'https://api.stlouisfed.org/fred/series/observations'
                params = {
                    'series_id': code,
                    'api_key': self.fred_key,
                    'file_type': 'json',
                    'observation_start': data.index[0].strftime('%Y-%m-%d'),
                    'observation_end': data.index[-1].strftime('%Y-%m-%d')
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    observations = response.json()['observations']
                    values = {pd.to_datetime(obs['date']): float(obs['value']) 
                            for obs in observations if obs['value'] != '.'}
                    data[name] = pd.Series(values)
            except Exception as e:
                print(f"Error fetching {name}: {str(e)}")
        
        return data
    
    def get_financial_statements(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch and process financial statements"""
        balance_sheet = self.stock.balance_sheet
        income_stmt = self.stock.income_stmt
        cash_flow = self.stock.cashflow
        
        # Calculate financial ratios
        ratios = self._calculate_financial_ratios(balance_sheet, income_stmt, cash_flow)
        
        return balance_sheet, income_stmt, cash_flow, ratios
    
    def _calculate_financial_ratios(self, balance_sheet: pd.DataFrame, 
                                  income_stmt: pd.DataFrame, 
                                  cash_flow: pd.DataFrame) -> pd.DataFrame:
        """Calculate important financial ratios"""
        ratios = pd.DataFrame()
        
        # Liquidity ratios
        ratios['Current_Ratio'] = balance_sheet.loc['Total Current Assets'] / balance_sheet.loc['Total Current Liabilities']
        ratios['Quick_Ratio'] = (balance_sheet.loc['Total Current Assets'] - balance_sheet.loc['Inventory']) / balance_sheet.loc['Total Current Liabilities']
        
        # Profitability ratios
        ratios['ROE'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Stockholder Equity']
        ratios['ROA'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Assets']
        ratios['Profit_Margin'] = income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue']
        
        # Leverage ratios
        ratios['Debt_to_Equity'] = balance_sheet.loc['Total Liab'] / balance_sheet.loc['Total Stockholder Equity']
        ratios['Debt_to_Assets'] = balance_sheet.loc['Total Liab'] / balance_sheet.loc['Total Assets']
        
        # Efficiency ratios
        ratios['Asset_Turnover'] = income_stmt.loc['Total Revenue'] / balance_sheet.loc['Total Assets']
        
        return ratios
    
    def get_news_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch and process news data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch news from multiple sources
        news_data = self._fetch_news_data(start_date, end_date)
        
        # Process and analyze news
        processed_news = self._process_news_data(news_data)
        
        return processed_news
    
    def _fetch_news_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch news from multiple sources"""
        news_data = []
        
        # Add news from different sources here
        # For now, we'll just use the basic implementation
        try:
            news = self.stock.news
            news_data.extend(news)
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
        
        return news_data
    
    def _process_news_data(self, news_data: List[Dict]) -> pd.DataFrame:
        """Process and analyze news data"""
        # Convert news data to DataFrame
        df = pd.DataFrame(news_data)
        
        if not df.empty:
            # Add sentiment analysis
            df['sentiment'] = df['title'].apply(self._analyze_sentiment)
            
            # Add importance score
            df['importance'] = df.apply(self._calculate_importance, axis=1)
            
            # Add topic classification
            df['topic'] = df['title'].apply(self._classify_topic)
        
        return df
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        # Implement more sophisticated sentiment analysis
        # For now, return a simple score
        return 0.0
    
    def _calculate_importance(self, row: pd.Series) -> float:
        """Calculate importance score for news item"""
        # Implement importance scoring
        return 0.0
    
    def _classify_topic(self, text: str) -> str:
        """Classify news topic"""
        # Implement topic classification
        return "General" 