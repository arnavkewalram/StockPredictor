import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers, callbacks
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import List, Dict, Tuple, Optional, Union
import optuna
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.histories = {}
        
    def train_ensemble(self, X: np.ndarray, y: np.ndarray,
                      validation_split: float = 0.2,
                      epochs: int = 100,
                      batch_size: int = 32) -> Dict:
        """Train an ensemble of models"""
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train XGBoost
        xgb_model = self._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train LightGBM
        lgb_model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Train CatBoost
        cb_model = self._train_catboost(X_train, y_train, X_val, y_val)
        
        # Train Neural Network
        nn_model, nn_history = self._train_neural_network(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size
        )
        
        # Train scikit-learn models
        sklearn_models = self._train_sklearn_model(X_train, y_train)
        
        return {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'catboost': cb_model,
            'neural_network': nn_model,
            'sklearn_models': sklearn_models,
            'history': nn_history
        }
        
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> xgb.Booster:
        """Train XGBoost model with advanced features"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        return model
        
    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> lgb.Booster:
        """Train LightGBM model with advanced features"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0,
            'lambda_l2': 0
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        return model
        
    def _train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> cb.CatBoostRegressor:
        """Train CatBoost model with advanced features"""
        model = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            od_type='Iter',
            od_wait=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        return model
        
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            epochs: int = 100,
                            batch_size: int = 32) -> Tuple[tf.keras.Model, Dict]:
        """Train neural network with advanced features"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=0
        )
        
        return model, history.history
        
    def _train_sklearn_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train scikit-learn models"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import ElasticNet
        
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
        
        for name, model in models.items():
            model.fit(X, y)
            
        return models
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                               model_type: str,
                               n_trials: int = 100) -> Dict:
        """Optimize hyperparameters using Optuna or Bayesian Optimization"""
        if model_type in ['xgboost', 'lightgbm', 'catboost']:
            return self._optimize_with_optuna(X, y, model_type, n_trials)
        else:
            return self._optimize_with_bayesian(X, y, model_type, n_trials)
            
    def _optimize_with_optuna(self, X: np.ndarray, y: np.ndarray,
                            model_type: str,
                            n_trials: int) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 1)
                }
                model = xgb.XGBRegressor(**params)
            elif model_type == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50)
                }
                model = lgb.LGBMRegressor(**params)
            elif model_type == 'catboost':
                params = {
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
                }
                model = cb.CatBoostRegressor(**params)
                
            # Cross-validation
            scores = self._cross_validate(model, X, y)
            return np.mean(scores)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
        
    def _optimize_with_bayesian(self, X: np.ndarray, y: np.ndarray,
                              model_type: str,
                              n_iter: int) -> Dict:
        """Optimize hyperparameters using Bayesian Optimization"""
        def objective(**params):
            if model_type == 'random_forest':
                model = RandomForestRegressor(**params)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(**params)
            elif model_type == 'svr':
                model = SVR(**params)
                
            scores = self._cross_validate(model, X, y)
            return np.mean(scores)
            
        if model_type == 'random_forest':
            param_space = {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 5)
            }
        elif model_type == 'gradient_boosting':
            param_space = {
                'n_estimators': (50, 200),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (2, 10)
            }
        elif model_type == 'svr':
            param_space = {
                'C': (0.1, 10),
                'epsilon': (0.01, 0.1),
                'gamma': (0.01, 1)
            }
            
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_space,
            random_state=42
        )
        
        optimizer.maximize(
            init_points=5,
            n_iter=n_iter
        )
        
        return optimizer.max['params']
        
    def _cross_validate(self, model: Union[xgb.Booster, lgb.Booster, cb.CatBoostRegressor],
                       X: np.ndarray, y: np.ndarray,
                       n_splits: int = 5) -> np.ndarray:
        """Perform time series cross-validation"""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if isinstance(model, (xgb.Booster, lgb.Booster)):
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                model = xgb.train(
                    model.get_params(),
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
            else:
                model.fit(X_train, y_train)
                
            y_pred = model.predict(X_val)
            score = np.sqrt(np.mean((y_val - y_pred) ** 2))
            scores.append(score)
            
        return np.array(scores)
        
    def create_ensemble_prediction(self, models: Dict,
                                 X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble predictions with uncertainty estimation"""
        predictions = []
        
        for name, model in models.items():
            if name == 'xgboost':
                pred = model.predict(xgb.DMatrix(X))
            elif name == 'lightgbm':
                pred = model.predict(X)
            elif name == 'catboost':
                pred = model.predict(X)
            elif name == 'neural_network':
                pred = model.predict(X).flatten()
            else:
                pred = model.predict(X)
                
            predictions.append(pred)
            
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
        
    def save_models(self, models: Dict, path: str) -> None:
        """Save trained models and their histories"""
        import joblib
        import os
        
        os.makedirs(path, exist_ok=True)
        
        for name, model in models.items():
            if name == 'neural_network':
                model.save(os.path.join(path, f'{name}.h5'))
            else:
                joblib.dump(model, os.path.join(path, f'{name}.joblib'))
                
        if 'history' in models:
            joblib.dump(models['history'], os.path.join(path, 'history.joblib'))
            
    def load_models(self, path: str) -> Dict:
        """Load trained models and their histories"""
        import joblib
        import os
        
        models = {}
        
        for file in os.listdir(path):
            if file.endswith('.h5'):
                models[file[:-3]] = tf.keras.models.load_model(os.path.join(path, file))
            elif file.endswith('.joblib'):
                name = file[:-7]
                if name == 'history':
                    models[name] = joblib.load(os.path.join(path, file))
                else:
                    models[name] = joblib.load(os.path.join(path, file))
                    
        return models 