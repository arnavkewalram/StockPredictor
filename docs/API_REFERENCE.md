# API Reference

## StockPredictor Class

The main class for stock price prediction.

### Methods

#### `__init__(self)`
Initialize the StockPredictor.

#### `train_model(self, ticker: str, start_date: str, end_date: str)`
Train the model on historical data.

Parameters:
- `ticker` (str): Stock symbol (e.g., "AAPL")
- `start_date` (str): Start date in YYYY-MM-DD format
- `end_date` (str): End date in YYYY-MM-DD format

#### `predict_next_days(self, n_days: int)`
Make predictions for the next n days.

Parameters:
- `n_days` (int): Number of days to predict

Returns:
- `predictions` (DataFrame): Predicted prices with confidence intervals

## NeuralArchitectures Class

Class for creating various neural network architectures.

### Methods

#### `create_transformer_model(self, input_shape: tuple, num_heads: int = 8, num_layers: int = 6, d_model: int = 512)`
Create a transformer-based model.

Parameters:
- `input_shape` (tuple): Shape of input data
- `num_heads` (int): Number of attention heads
- `num_layers` (int): Number of transformer layers
- `d_model` (int): Dimension of the model

#### `create_attention_lstm(self, input_shape: tuple, hidden_dim: int = 64, num_layers: int = 2)`
Create an LSTM model with attention mechanism.

Parameters:
- `input_shape` (tuple): Shape of input data
- `hidden_dim` (int): Dimension of hidden layers
- `num_layers` (int): Number of LSTM layers

#### `create_hybrid_model(self, input_shape: tuple)`
Create a hybrid model combining multiple architectures.

Parameters:
- `input_shape` (tuple): Shape of input data

## ModelOptimizer Class

Class for optimizing model parameters and architecture.

### Methods

#### `optimize_architecture(self, model, n_trials: int = 100)`
Optimize model architecture using Optuna.

Parameters:
- `model`: Model to optimize
- `n_trials` (int): Number of optimization trials

#### `optimize_hyperparameters(self, model, X, y, param_space: dict)`
Optimize model hyperparameters using Bayesian optimization.

Parameters:
- `model`: Model to optimize
- `X`: Training data
- `y`: Target values
- `param_space` (dict): Parameter space for optimization

## MarketAnalyzer Class

Class for market analysis and feature extraction.

### Methods

#### `calculate_market_regime(self, prices: pd.Series)`
Calculate current market regime.

Parameters:
- `prices` (pd.Series): Price series

Returns:
- `regime` (str): Current market regime

#### `calculate_sentiment(self, ticker: str)`
Calculate market sentiment for a given ticker.

Parameters:
- `ticker` (str): Stock symbol

Returns:
- `sentiment` (float): Sentiment score

## ModelTrainer Class

Class for training and evaluating models.

### Methods

#### `train_ensemble(self, X, y, validation_split: float = 0.2)`
Train an ensemble of models.

Parameters:
- `X`: Training data
- `y`: Target values
- `validation_split` (float): Proportion of data to use for validation

#### `evaluate_model(self, model, X, y)`
Evaluate model performance.

Parameters:
- `model`: Model to evaluate
- `X`: Test data
- `y`: True values

Returns:
- `metrics` (dict): Performance metrics

## ModelEnhancer Class

Class for enhancing model performance.

### Methods

#### `create_adaptive_model(self, input_shape: tuple)`
Create an adaptive model that adjusts to market conditions.

Parameters:
- `input_shape` (tuple): Shape of input data

#### `create_uncertainty_estimator(self, model)`
Create a model that estimates prediction uncertainty.

Parameters:
- `model`: Base model

## Performance Metrics

### Available Metrics

- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
- **Root Mean Square Error (RMSE)**: Square root of average squared differences
- **R-squared**: Proportion of variance explained by the model
- **Directional Accuracy**: Percentage of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline

### Usage

```python
from src.models.model_performance import calculate_metrics

metrics = calculate_metrics(y_true, y_pred)
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R-squared: {metrics['r2']:.4f}")
``` 