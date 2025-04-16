import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import torch
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
import optuna
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    def __init__(self):
        self.optimizers = {}
        self.schedulers = {}
        self.callbacks = {}
        
    def create_optimizer(self, optimizer_type: str,
                        learning_rate: float = 0.001,
                        **kwargs) -> tf.keras.optimizers.Optimizer:
        """Create a TensorFlow optimizer with advanced features"""
        if optimizer_type == 'adam':
            return optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=kwargs.get('beta_1', 0.9),
                beta_2=kwargs.get('beta_2', 0.999),
                epsilon=kwargs.get('epsilon', 1e-7)
            )
        elif optimizer_type == 'adamw':
            return optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=kwargs.get('weight_decay', 0.01)
            )
        elif optimizer_type == 'radam':
            return optimizers.RAdam(
                learning_rate=learning_rate,
                beta_1=kwargs.get('beta_1', 0.9),
                beta_2=kwargs.get('beta_2', 0.999)
            )
        elif optimizer_type == 'lookahead':
            base_optimizer = optimizers.Adam(learning_rate=learning_rate)
            return optimizers.Lookahead(base_optimizer)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def create_pytorch_optimizer(self, model: torch.nn.Module,
                               optimizer_type: str,
                               learning_rate: float = 0.001,
                               **kwargs) -> optim.Optimizer:
        """Create a PyTorch optimizer with learning rate scheduler"""
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(kwargs.get('beta_1', 0.9), kwargs.get('beta_2', 0.999))
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=kwargs.get('weight_decay', 0.01)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return optimizer, scheduler
        
    def create_callbacks(self, monitor: str = 'val_loss',
                        patience: int = 10,
                        min_delta: float = 0.001) -> List[callbacks.Callback]:
        """Create training callbacks"""
        return [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.1,
                patience=patience // 2,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor=monitor,
                save_best_only=True
            )
        ]
        
    def create_advanced_callbacks(self, log_dir: str = 'logs',
                                monitor: str = 'val_loss',
                                patience: int = 10) -> List[callbacks.Callback]:
        """Create advanced training callbacks"""
        return [
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            callbacks.CSVLogger('training.log'),
            callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * np.cos(epoch / 10 * np.pi)
            ),
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True
            )
        ]
        
    def optimize_learning_rate(self, model: tf.keras.Model,
                             X: np.ndarray, y: np.ndarray,
                             min_lr: float = 1e-6,
                             max_lr: float = 1e-2,
                             num_iterations: int = 100) -> float:
        """Optimize learning rate using range test"""
        lr_finder = callbacks.LearningRateFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            num_iterations=num_iterations
        )
        
        model.fit(
            X, y,
            epochs=1,
            callbacks=[lr_finder],
            verbose=0
        )
        
        return lr_finder.best_lr
        
    def optimize_batch_size(self, model: tf.keras.Model,
                          X: np.ndarray, y: np.ndarray,
                          min_batch_size: int = 16,
                          max_batch_size: int = 256,
                          num_iterations: int = 10) -> int:
        """Optimize batch size using grid search"""
        best_batch_size = min_batch_size
        best_loss = float('inf')
        
        for batch_size in np.linspace(min_batch_size, max_batch_size, num_iterations, dtype=int):
            model.fit(
                X, y,
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )
            
            loss = model.evaluate(X, y, verbose=0)[0]
            if loss < best_loss:
                best_loss = loss
                best_batch_size = batch_size
                
        return best_batch_size
        
    def optimize_architecture(self, input_shape: Tuple[int, int],
                            model_type: str,
                            n_trials: int = 100) -> Dict:
        """Optimize model architecture using Optuna"""
        def objective(trial):
            if model_type == 'transformer':
                num_heads = trial.suggest_int('num_heads', 2, 16)
                num_layers = trial.suggest_int('num_layers', 2, 8)
                d_model = trial.suggest_int('d_model', 128, 1024, step=128)
                model = self.create_transformer_model(input_shape, num_heads, num_layers, d_model)
            elif model_type == 'lstm':
                hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
                num_layers = trial.suggest_int('num_layers', 1, 4)
                model = self.create_attention_lstm(input_shape, hidden_dim, num_layers)
            elif model_type == 'tcn':
                filters = trial.suggest_int('filters', 32, 256, step=32)
                kernel_size = trial.suggest_int('kernel_size', 2, 8)
                num_layers = trial.suggest_int('num_layers', 2, 6)
                model = self.create_tcn(input_shape, filters, kernel_size, num_layers)
                
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(),
                loss='mse',
                metrics=['mae']
            )
            
            # Return validation loss (placeholder)
            return 0.0
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
        
    def optimize_hyperparameters(self, model: tf.keras.Model,
                               X: np.ndarray, y: np.ndarray,
                               param_space: Dict,
                               n_iter: int = 50) -> Dict:
        """Optimize hyperparameters using Bayesian optimization"""
        def objective(**params):
            # Update model with new parameters
            for param, value in params.items():
                if hasattr(model, param):
                    setattr(model, param, value)
                    
            # Train and evaluate model
            model.fit(
                X, y,
                epochs=5,
                verbose=0
            )
            
            return -model.evaluate(X, y, verbose=0)[0]
            
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
        
    def create_ensemble_optimizer(self, models: List[tf.keras.Model],
                                weights: Optional[np.ndarray] = None) -> tf.keras.Model:
        """Create an ensemble model with optimized weights"""
        if weights is None:
            weights = np.ones(len(models)) / len(models)
            
        class EnsembleModel(tf.keras.Model):
            def __init__(self, models, weights):
                super(EnsembleModel, self).__init__()
                self.models = models
                self.weights = tf.Variable(weights, trainable=True)
                
            def call(self, inputs):
                outputs = [model(inputs) for model in self.models]
                weighted_outputs = tf.reduce_sum(
                    [w * o for w, o in zip(self.weights, outputs)],
                    axis=0
                )
                return weighted_outputs
                
        return EnsembleModel(models, weights)
        
    def create_meta_optimizer(self, base_models: List[tf.keras.Model],
                            meta_model: tf.keras.Model) -> tf.keras.Model:
        """Create a meta-optimizer for stacking models"""
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