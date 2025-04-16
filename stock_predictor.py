import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# Download NLTK data
nltk.download('vader_lexicon')

class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        load_dotenv()
        self.news_api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

    def get_historical_data(self, period='5y'):
        """Fetch historical stock data"""
        data = self.stock.history(period=period)
        return data

    def get_financial_data(self):
        """Fetch financial statements"""
        balance_sheet = self.stock.balance_sheet
        income_stmt = self.stock.income_stmt
        cash_flow = self.stock.cashflow
        return balance_sheet, income_stmt, cash_flow

    def get_news_sentiment(self, days=30):
        """Fetch and analyze news sentiment"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        news = self.news_api.get_everything(
            q=self.symbol,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy'
        )
        
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        
        for article in news['articles']:
            sentiment = sia.polarity_scores(article['title'])
            sentiments.append(sentiment['compound'])
        
        return np.mean(sentiments) if sentiments else 0

    def prepare_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, data, epochs=50, batch_size=32):
        """Train the LSTM model"""
        X, y = self.prepare_data(data)
        self.model = self.build_model((X.shape[1], 1))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, data, days=30):
        """Make predictions"""
        if not self.model:
            self.train_model(data)
        
        last_60_days = data['Close'].values[-60:]
        last_60_days_scaled = self.scaler.transform(last_60_days.reshape(-1, 1))
        
        predictions = []
        current_batch = last_60_days_scaled.reshape(1, 60, 1)
        
        for i in range(days):
            current_pred = self.model.predict(current_batch)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions

    def plot_predictions(self, data, predictions):
        """Plot historical data and predictions"""
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[-60:], data['Close'].values[-60:], label='Historical Data')
        future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1)[1:]
        plt.plot(future_dates, predictions, label='Predictions')
        plt.title(f'{self.symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    # Example usage
    symbol = input("Enter stock symbol (e.g., AAPL): ")
    predictor = StockPredictor(symbol)
    
    # Get historical data
    data = predictor.get_historical_data()
    
    # Get news sentiment
    sentiment = predictor.get_news_sentiment()
    print(f"\nNews Sentiment Score: {sentiment:.2f}")
    
    # Train model and make predictions
    predictions = predictor.predict(data)
    
    # Plot results
    predictor.plot_predictions(data, predictions)

if __name__ == "__main__":
    main() 