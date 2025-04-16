import os
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from data.data_collector import DataCollector
from models.model_builder import ModelBuilder
from visualization.dashboard import StockDashboard

def main():
    # Load environment variables
    load_dotenv()
    
    # Get stock symbol from user
    symbol = input("Enter stock symbol (e.g., AAPL): ")
    
    # Initialize data collector
    collector = DataCollector(symbol)
    
    # Collect and process data
    print("Collecting historical data...")
    historical_data = collector.get_historical_data()
    
    print("Collecting financial statements...")
    balance_sheet, income_stmt, cash_flow, ratios = collector.get_financial_statements()
    
    print("Collecting news data...")
    news_data = collector.get_news_data()
    
    # Initialize model builder
    model_builder = ModelBuilder(historical_data)
    
    # Prepare data for training
    print("Preparing data for model training...")
    X, y = model_builder.prepare_data()
    
    # Train models
    print("Training models...")
    models = model_builder.train_models(X, y)
    
    # Make predictions
    print("Making predictions...")
    predictions = model_builder.make_predictions(X[-1:])  # Use last sequence for prediction
    
    # Evaluate models
    print("Evaluating models...")
    metrics = model_builder.evaluate_models(y[-1])  # Use last target for evaluation
    
    # Create and run dashboard
    print("Launching dashboard...")
    dashboard = StockDashboard(historical_data, predictions, metrics)
    dashboard.run()

if __name__ == "__main__":
    main() 