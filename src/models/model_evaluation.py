import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
from bayes_opt import BayesianOptimization
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.metrics_history = {}
        self.feature_importance = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Directional_Accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == 
                                         np.sign(y_pred[1:] - y_pred[:-1]))
        }
        return metrics
    
    def time_series_cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                                   n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'MAPE': [], 
                  'Directional_Accuracy': []}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            fold_metrics = self.calculate_metrics(y_test, y_pred)
            for metric, value in fold_metrics.items():
                metrics[metric].append(value)
                
        return metrics
    
    def calculate_feature_importance(self, model, X: np.ndarray, 
                                   feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance using SHAP values"""
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        importance = {}
        for i, feature in enumerate(feature_names):
            importance[feature] = np.abs(shap_values.values[:, i]).mean()
            
        self.feature_importance = importance
        return importance
    
    def plot_feature_importance(self):
        """Plot feature importance using SHAP values"""
        if not self.feature_importance:
            raise ValueError("Feature importance not calculated yet")
            
        plt.figure(figsize=(10, 6))
        features = list(self.feature_importance.keys())
        importance = list(self.feature_importance.values())
        
        sns.barplot(x=importance, y=features)
        plt.title('Feature Importance (SHAP Values)')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.tight_layout()
        plt.show()
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate prediction confidence intervals"""
        errors = y_true - y_pred
        std_error = np.std(errors)
        z_score = norm.ppf((1 + confidence) / 2)
        
        lower_bound = y_pred - z_score * std_error
        upper_bound = y_pred + z_score * std_error
        
        return lower_bound, upper_bound
    
    def calculate_uncertainty(self, model, X: np.ndarray) -> np.ndarray:
        """Calculate prediction uncertainty using ensemble methods"""
        if hasattr(model, 'predict_std'):
            return model.predict_std(X)
        else:
            # Use bootstrapping for uncertainty estimation
            n_samples = 100
            predictions = []
            
            for _ in range(n_samples):
                # Bootstrap sample
                idx = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X[idx]
                y_boot = model.predict(X_boot)
                predictions.append(y_boot)
                
            return np.std(predictions, axis=0)
    
    def plot_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                lower_bound: np.ndarray, upper_bound: np.ndarray):
        """Plot prediction intervals"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red')
        plt.fill_between(range(len(y_pred)), lower_bound, upper_bound, 
                        alpha=0.2, color='gray', label='95% Confidence Interval')
        plt.title('Predictions with Confidence Intervals')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def calculate_model_stability(self, model, X: np.ndarray, 
                                n_iterations: int = 10) -> float:
        """Calculate model stability across multiple runs"""
        predictions = []
        
        for _ in range(n_iterations):
            # Add small noise to input data
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            y_pred = model.predict(X_noisy)
            predictions.append(y_pred)
            
        # Calculate stability as the inverse of prediction variance
        stability = 1 / np.var(predictions, axis=0).mean()
        return stability
    
    def optimize_hyperparameters(self, model_class, X: np.ndarray, y: np.ndarray,
                               param_space: Dict, n_trials: int = 100) -> Dict:
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
            metrics = self.time_series_cross_validation(model, X, y)
            return -np.mean(metrics['RMSE'])  # Minimize RMSE
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def bayesian_optimization(self, model_class, X: np.ndarray, y: np.ndarray,
                            param_space: Dict, n_iter: int = 50) -> Dict:
        """Optimize hyperparameters using Bayesian Optimization"""
        def objective(**params):
            model = model_class(**params)
            metrics = self.time_series_cross_validation(model, X, y)
            return -np.mean(metrics['RMSE'])  # Minimize RMSE
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_space,
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=n_iter)
        return optimizer.max['params']
    
    def calculate_model_robustness(self, model, X: np.ndarray, y: np.ndarray,
                                 noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict:
        """Calculate model robustness to input noise"""
        robustness = {}
        
        for noise in noise_levels:
            X_noisy = X + np.random.normal(0, noise, X.shape)
            y_pred = model.predict(X_noisy)
            metrics = self.calculate_metrics(y, y_pred)
            robustness[f'noise_{noise}'] = metrics
            
        return robustness 