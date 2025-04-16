import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ModelBuilder:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def prepare_data(self, target_col: str = 'Close', 
                    lookback: int = 60, 
                    forecast_horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data[target_col].values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data) - forecast_horizon):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i:i+forecast_horizon, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_lstm_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Build a sophisticated LSTM model"""
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        lstm1 = LSTM(units=100, return_sequences=True)(inputs)
        dropout1 = Dropout(0.2)(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(units=100, return_sequences=True)(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        
        # Third LSTM layer
        lstm3 = LSTM(units=50)(dropout2)
        dropout3 = Dropout(0.2)(lstm3)
        
        # Dense layers
        dense1 = Dense(units=50, activation='relu')(dropout3)
        dense2 = Dense(units=25, activation='relu')(dense1)
        outputs = Dense(units=1)(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        return model
    
    def build_xgboost_model(self) -> xgb.XGBRegressor:
        """Build XGBoost model"""
        return xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            nthread=4,
            scale_pos_weight=1,
            seed=27
        )
    
    def build_lightgbm_model(self) -> lgb.LGBMRegressor:
        """Build LightGBM model"""
        return lgb.LGBMRegressor(
            objective='regression',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=1000,
            max_bin=55,
            bagging_fraction=0.8,
            bagging_freq=5,
            feature_fraction=0.8,
            feature_fraction_seed=9,
            bagging_seed=9,
            min_data_in_leaf=6,
            min_sum_hessian_in_leaf=11
        )
    
    def build_catboost_model(self) -> CatBoostRegressor:
        """Build CatBoost model"""
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            l2_leaf_reg=3,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42
        )
    
    def build_prophet_model(self) -> Prophet:
        """Build Prophet model"""
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
    
    def build_arima_model(self, order: Tuple[int, int, int] = (5, 1, 0)) -> ARIMA:
        """Build ARIMA model"""
        return ARIMA(self.data['Close'], order=order)
    
    def build_sarima_model(self, order: Tuple[int, int, int] = (1, 1, 1),
                          seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)) -> SARIMAX:
        """Build SARIMA model"""
        return SARIMAX(self.data['Close'],
                      order=order,
                      seasonal_order=seasonal_order)
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    validation_split: float = 0.2) -> Dict[str, Model]:
        """Train all models"""
        # Split data
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train LSTM
        lstm_model = self.build_lstm_model(X_train.shape[1:])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
        
        lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        self.models['lstm'] = lstm_model
        
        # Train XGBoost
        xgb_model = self.build_xgboost_model()
        xgb_model.fit(
            X_train.reshape(X_train.shape[0], -1),
            y_train,
            eval_set=[(X_val.reshape(X_val.shape[0], -1), y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM
        lgb_model = self.build_lightgbm_model()
        lgb_model.fit(
            X_train.reshape(X_train.shape[0], -1),
            y_train,
            eval_set=[(X_val.reshape(X_val.shape[0], -1), y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        self.models['lightgbm'] = lgb_model
        
        # Train CatBoost
        cat_model = self.build_catboost_model()
        cat_model.fit(
            X_train.reshape(X_train.shape[0], -1),
            y_train,
            eval_set=[(X_val.reshape(X_val.shape[0], -1), y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        self.models['catboost'] = cat_model
        
        # Train Prophet
        prophet_data = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data['Close']
        })
        prophet_model = self.build_prophet_model()
        prophet_model.fit(prophet_data)
        self.models['prophet'] = prophet_model
        
        # Train ARIMA
        arima_model = self.build_arima_model()
        arima_model = arima_model.fit()
        self.models['arima'] = arima_model
        
        # Train SARIMA
        sarima_model = self.build_sarima_model()
        sarima_model = sarima_model.fit()
        self.models['sarima'] = sarima_model
        
        return self.models
    
    def make_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using all models"""
        for name, model in self.models.items():
            if name == 'lstm':
                pred = model.predict(X)
            elif name in ['xgboost', 'lightgbm', 'catboost']:
                pred = model.predict(X.reshape(X.shape[0], -1))
            elif name == 'prophet':
                future = model.make_future_dataframe(periods=len(X))
                pred = model.predict(future)['yhat'].values[-len(X):]
            elif name == 'arima':
                pred = model.forecast(len(X))
            elif name == 'sarima':
                pred = model.forecast(len(X))
            
            self.predictions[name] = self.scaler.inverse_transform(pred.reshape(-1, 1))
        
        # Calculate ensemble prediction
        ensemble_pred = np.mean(list(self.predictions.values()), axis=0)
        self.predictions['ensemble'] = ensemble_pred
        
        return self.predictions
    
    def evaluate_models(self, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance"""
        for name, pred in self.predictions.items():
            mse = mean_squared_error(y_true, pred)
            mae = mean_absolute_error(y_true, pred)
            r2 = r2_score(y_true, pred)
            
            self.metrics[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }
        
        return self.metrics 