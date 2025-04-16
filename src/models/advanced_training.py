import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

class AdvancedTrainer:
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.best_params = {}
        
    def train_ensemble(self, X: np.ndarray, y: np.ndarray,
                      models: List[Tuple[str, Callable]]) -> Dict[str, object]:
        """Train an ensemble of models"""
        trained_models = {}
        
        for name, model_class in models:
            if name.startswith('xgb'):
                model = self._train_xgboost(X, y, model_class)
            elif name.startswith('lgb'):
                model = self._train_lightgbm(X, y, model_class)
            elif name.startswith('cat'):
                model = self._train_catboost(X, y, model_class)
            elif name.startswith('nn'):
                model = self._train_neural_network(X, y, model_class)
            else:
                model = self._train_sklearn_model(X, y, model_class)
                
            trained_models[name] = model
            
        self.models = trained_models
        return trained_models
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray,
                      model_class: Callable) -> xgb.XGBRegressor:
        """Train XGBoost model with advanced features"""
        model = model_class(
            objective='reg:squarederror',
            tree_method='hist',
            grow_policy='lossguide',
            eval_metric='rmse',
            random_state=42
        )
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train with early stopping
        model.fit(
            X, y,
            eval_set=[(X, y)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray,
                       model_class: Callable) -> lgb.LGBMRegressor:
        """Train LightGBM model with advanced features"""
        model = model_class(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            random_state=42
        )
        
        # Create Dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train with early stopping
        model.fit(
            X, y,
            eval_set=[(X, y)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def _train_catboost(self, X: np.ndarray, y: np.ndarray,
                       model_class: Callable) -> cb.CatBoostRegressor:
        """Train CatBoost model with advanced features"""
        model = model_class(
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            task_type='CPU'
        )
        
        # Train with early stopping
        model.fit(
            X, y,
            eval_set=[(X, y)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray,
                            model_class: Callable) -> tf.keras.Model:
        """Train neural network with advanced features"""
        model = model_class()
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.histories['nn'] = history
        return model
    
    def _train_sklearn_model(self, X: np.ndarray, y: np.ndarray,
                           model_class: Callable) -> object:
        """Train scikit-learn model"""
        model = model_class()
        model.fit(X, y)
        return model
    
    def optimize_hyperparameters(self, model_class: Callable, X: np.ndarray, y: np.ndarray,
                               param_space: Dict, method: str = 'optuna',
                               n_trials: int = 100) -> Dict:
        """Optimize hyperparameters using different methods"""
        if method == 'optuna':
            return self._optimize_with_optuna(model_class, X, y, param_space, n_trials)
        elif method == 'bayesian':
            return self._optimize_with_bayesian(model_class, X, y, param_space, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_with_optuna(self, model_class: Callable, X: np.ndarray, y: np.ndarray,
                            param_space: Dict, n_trials: int) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, 
                                                             param_range[0], 
                                                             param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, 
                                                               param_range[0], 
                                                               param_range[1])
            
            model = model_class(**params)
            scores = self._cross_validate(model, X, y)
            return -np.mean(scores)  # Minimize RMSE
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['optuna'] = study.best_params
        return study.best_params
    
    def _optimize_with_bayesian(self, model_class: Callable, X: np.ndarray, y: np.ndarray,
                              param_space: Dict, n_iter: int) -> Dict:
        """Optimize hyperparameters using Bayesian Optimization"""
        def objective(**params):
            model = model_class(**params)
            scores = self._cross_validate(model, X, y)
            return -np.mean(scores)  # Minimize RMSE
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_space,
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=n_iter)
        
        self.best_params['bayesian'] = optimizer.max['params']
        return optimizer.max['params']
    
    def _cross_validate(self, model: object, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)
            
        return scores
    
    def create_ensemble_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble prediction with uncertainty estimation"""
        predictions = []
        
        for name, model in self.models.items():
            if name.startswith('nn'):
                pred = model.predict(X).flatten()
            else:
                pred = model.predict(X)
            predictions.append(pred)
            
        # Calculate mean prediction
        mean_prediction = np.mean(predictions, axis=0)
        
        # Calculate prediction uncertainty
        uncertainty = np.std(predictions, axis=0)
        
        return mean_prediction, uncertainty
    
    def save_models(self, path: str):
        """Save trained models"""
        import joblib
        import os
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        for name, model in self.models.items():
            if name.startswith('nn'):
                model.save(os.path.join(path, f'{name}_model.h5'))
            else:
                joblib.dump(model, os.path.join(path, f'{name}_model.joblib'))
                
        # Save histories
        if self.histories:
            joblib.dump(self.histories, os.path.join(path, 'training_histories.joblib'))
            
        # Save best parameters
        if self.best_params:
            joblib.dump(self.best_params, os.path.join(path, 'best_parameters.joblib'))
    
    def load_models(self, path: str):
        """Load trained models"""
        import joblib
        import os
        
        for file in os.listdir(path):
            if file.endswith('_model.h5'):
                name = file.replace('_model.h5', '')
                self.models[name] = tf.keras.models.load_model(os.path.join(path, file))
            elif file.endswith('_model.joblib'):
                name = file.replace('_model.joblib', '')
                self.models[name] = joblib.load(os.path.join(path, file))
                
        # Load histories
        history_file = os.path.join(path, 'training_histories.joblib')
        if os.path.exists(history_file):
            self.histories = joblib.load(history_file)
            
        # Load best parameters
        params_file = os.path.join(path, 'best_parameters.joblib')
        if os.path.exists(params_file):
            self.best_params = joblib.load(params_file) 