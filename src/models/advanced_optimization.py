import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import List, Dict, Tuple, Optional
import optuna
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

class AdvancedOptimizer:
    def __init__(self):
        self.optimizers = {}
        self.schedulers = {}
        self.callbacks = {}
        
    def create_optimizer(self, model: tf.keras.Model,
                        optimizer_type: str = 'adam',
                        learning_rate: float = 0.001) -> tf.keras.Model:
        """Create an advanced optimizer"""
        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
        elif optimizer_type == 'adamw':
            optimizer = optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01
            )
        elif optimizer_type == 'radam':
            optimizer = optimizers.RectifiedAdam(
                learning_rate=learning_rate,
                min_lr=1e-5
            )
        elif optimizer_type == 'lookahead':
            base_optimizer = optimizers.Adam(learning_rate=learning_rate)
            optimizer = optimizers.Lookahead(base_optimizer)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.optimizers[optimizer_type] = optimizer
        return model
    
    def create_pytorch_optimizer(self, model: nn.Module,
                               optimizer_type: str = 'adam',
                               learning_rate: float = 0.001) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Create an advanced PyTorch optimizer with scheduler"""
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        elif optimizer_type == 'radam':
            optimizer = optim.RAdam(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        # Create learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            verbose=True
        )
        
        self.optimizers[optimizer_type] = optimizer
        self.schedulers[optimizer_type] = scheduler
        return optimizer, scheduler
    
    def create_callbacks(self, model: tf.keras.Model,
                        monitor: str = 'val_loss',
                        patience: int = 10) -> List[callbacks.Callback]:
        """Create advanced training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor=monitor,
                save_best_only=True
            ),
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        self.callbacks['default'] = callbacks_list
        return callbacks_list
    
    def create_advanced_callbacks(self, model: tf.keras.Model) -> List[callbacks.Callback]:
        """Create more advanced training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            ),
            callbacks.LearningRateScheduler(
                self._cosine_decay_schedule
            ),
            callbacks.CSVLogger(
                'training.log'
            )
        ]
        
        self.callbacks['advanced'] = callbacks_list
        return callbacks_list
    
    def _cosine_decay_schedule(self, epoch: int, lr: float) -> float:
        """Create a cosine decay learning rate schedule"""
        initial_lr = 0.001
        decay_steps = 100
        alpha = 0.0
        
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_lr * decayed
    
    def optimize_learning_rate(self, model: tf.keras.Model,
                             X: np.ndarray, y: np.ndarray,
                             min_lr: float = 1e-6,
                             max_lr: float = 1e-2,
                             num_iterations: int = 100) -> float:
        """Find optimal learning rate using learning rate range test"""
        initial_weights = model.get_weights()
        best_loss = float('inf')
        best_lr = min_lr
        
        for lr in np.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations):
            model.set_weights(initial_weights)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=lr),
                loss='mse'
            )
            
            history = model.fit(
                X, y,
                epochs=1,
                batch_size=32,
                verbose=0
            )
            
            loss = history.history['loss'][0]
            if loss < best_loss:
                best_loss = loss
                best_lr = lr
                
        return best_lr
    
    def optimize_batch_size(self, model: tf.keras.Model,
                          X: np.ndarray, y: np.ndarray,
                          min_batch_size: int = 16,
                          max_batch_size: int = 256,
                          num_iterations: int = 10) -> int:
        """Find optimal batch size"""
        initial_weights = model.get_weights()
        best_loss = float('inf')
        best_batch_size = min_batch_size
        
        for batch_size in np.linspace(min_batch_size, max_batch_size, num_iterations, dtype=int):
            model.set_weights(initial_weights)
            model.compile(
                optimizer=optimizers.Adam(),
                loss='mse'
            )
            
            history = model.fit(
                X, y,
                epochs=5,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            loss = min(history.history['val_loss'])
            if loss < best_loss:
                best_loss = loss
                best_batch_size = batch_size
                
        return best_batch_size
    
    def optimize_architecture(self, model_class: type,
                            X: np.ndarray, y: np.ndarray,
                            param_space: Dict,
                            n_trials: int = 50) -> Dict:
        """Optimize model architecture using Optuna"""
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
            model.compile(
                optimizer=optimizers.Adam(),
                loss='mse',
                metrics=['mae']
            )
            
            history = model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def optimize_hyperparameters(self, model: tf.keras.Model,
                               X: np.ndarray, y: np.ndarray,
                               param_space: Dict,
                               n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Bayesian Optimization"""
        def objective(**params):
            model.compile(
                optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
                loss='mse'
            )
            
            history = model.fit(
                X, y,
                epochs=50,
                batch_size=params['batch_size'],
                validation_split=0.2,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            return -min(history.history['val_loss'])
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_space,
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=n_trials)
        return optimizer.max['params']
    
    def create_ensemble_optimizer(self, models: List[tf.keras.Model],
                                weights: Optional[List[float]] = None) -> tf.keras.Model:
        """Create an ensemble optimizer"""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
            
        class EnsembleModel(tf.keras.Model):
            def __init__(self, models, weights):
                super(EnsembleModel, self).__init__()
                self.models = models
                self.weights = weights
                
            def call(self, inputs):
                outputs = [model(inputs) for model in self.models]
                weighted_outputs = [w * o for w, o in zip(self.weights, outputs)]
                return tf.reduce_sum(weighted_outputs, axis=0)
                
        return EnsembleModel(models, weights)
    
    def create_meta_optimizer(self, base_models: List[tf.keras.Model],
                            meta_model: tf.keras.Model) -> tf.keras.Model:
        """Create a meta-optimizer for stacking"""
        class MetaOptimizer(tf.keras.Model):
            def __init__(self, base_models, meta_model):
                super(MetaOptimizer, self).__init__()
                self.base_models = base_models
                self.meta_model = meta_model
                
            def call(self, inputs):
                base_predictions = [model(inputs) for model in self.base_models]
                stacked_predictions = tf.concat(base_predictions, axis=1)
                return self.meta_model(stacked_predictions)
                
        return MetaOptimizer(base_models, meta_model) 