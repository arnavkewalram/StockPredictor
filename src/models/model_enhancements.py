import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import List, Dict, Tuple, Optional, Union
import optuna
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

class ModelEnhancer:
    def __init__(self):
        self.models = {}
        self.enhancements = {}
        
    def create_adaptive_model(self, input_shape: Tuple[int, int],
                            hidden_dim: int = 64,
                            num_layers: int = 3) -> tf.keras.Model:
        """Create a model with adaptive feature extraction"""
        inputs = layers.Input(shape=input_shape)
        
        # Adaptive feature extraction
        x = layers.Dense(hidden_dim)(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Adaptive layers
        for _ in range(num_layers):
            x = self._adaptive_layer(x, hidden_dim)
            
        # Output layer with uncertainty
        outputs = layers.Dense(2)(x)  # Mean and variance
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
        
    def _adaptive_layer(self, x: tf.Tensor,
                       hidden_dim: int) -> tf.Tensor:
        """Create an adaptive layer with dynamic weights"""
        # Dynamic weight generation
        weights = layers.Dense(hidden_dim)(x)
        weights = layers.Activation('sigmoid')(weights)
        
        # Feature transformation
        x = layers.Dense(hidden_dim)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Apply dynamic weights
        x = layers.Multiply()([x, weights])
        
        return x
        
    def create_uncertainty_estimator(self, model: tf.keras.Model,
                                   X: np.ndarray, y: np.ndarray,
                                   num_samples: int = 100) -> Dict:
        """Create a model that estimates prediction uncertainty"""
        # Monte Carlo dropout
        mc_predictions = []
        for _ in range(num_samples):
            pred = model.predict(X, verbose=0)
            mc_predictions.append(pred)
        mc_predictions = np.array(mc_predictions)
        mc_mean = np.mean(mc_predictions, axis=0)
        mc_std = np.std(mc_predictions, axis=0)
        
        # Deep ensembles
        ensemble_predictions = []
        for _ in range(5):
            model_copy = tf.keras.models.clone_model(model)
            model_copy.set_weights(model.get_weights())
            pred = model_copy.predict(X, verbose=0)
            ensemble_predictions.append(pred)
        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0)
        
        return {
            'monte_carlo': {'mean': mc_mean, 'std': mc_std},
            'ensemble': {'mean': ensemble_mean, 'std': ensemble_std}
        }
        
    def create_self_attention_model(self, input_shape: Tuple[int, int],
                                  num_heads: int = 8,
                                  d_model: int = 512) -> tf.keras.Model:
        """Create a model with self-attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        pos_encoding = self._positional_encoding(input_shape[0], d_model)
        x = layers.Dense(d_model)(inputs)
        x = x + pos_encoding
        
        # Self-attention blocks
        for _ in range(6):
            x = self._attention_block(x, num_heads, d_model)
            
        # Output layer
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
        
    def _attention_block(self, x: tf.Tensor,
                        num_heads: int,
                        d_model: int) -> tf.Tensor:
        """Create a self-attention block"""
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(d_model * 4, activation='relu')(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
        
    def _positional_encoding(self, length: int,
                           d_model: int) -> tf.Tensor:
        """Generate positional encoding"""
        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
        
    def create_dynamic_ensemble(self, models: List[tf.keras.Model],
                              X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """Create a dynamically weighted ensemble"""
        class DynamicEnsemble(tf.keras.Model):
            def __init__(self, models):
                super(DynamicEnsemble, self).__init__()
                self.models = models
                self.weights = tf.Variable(
                    tf.ones(len(models)) / len(models),
                    trainable=True
                )
                
            def call(self, inputs):
                outputs = [model(inputs) for model in self.models]
                weighted_outputs = tf.reduce_sum(
                    [w * o for w, o in zip(self.weights, outputs)],
                    axis=0
                )
                return weighted_outputs
                
        ensemble = DynamicEnsemble(models)
        
        # Optimize weights
        optimizer = optimizers.Adam()
        for _ in range(100):
            with tf.GradientTape() as tape:
                y_pred = ensemble(X)
                loss = tf.keras.losses.mean_squared_error(y, y_pred)
            grads = tape.gradient(loss, ensemble.trainable_variables)
            optimizer.apply_gradients(zip(grads, ensemble.trainable_variables))
            
        return ensemble
        
    def create_adaptive_training(self, model: tf.keras.Model,
                               X: np.ndarray, y: np.ndarray,
                               initial_lr: float = 0.001) -> Tuple[tf.keras.Model, callbacks.Callback]:
        """Create adaptive training with dynamic learning rate and batch size"""
        class AdaptiveTraining(callbacks.Callback):
            def __init__(self, initial_lr):
                self.lr = initial_lr
                self.batch_size = 32
                self.best_loss = float('inf')
                self.patience = 5
                self.wait = 0
                
            def on_epoch_end(self, epoch, logs=None):
                current_loss = logs.get('val_loss')
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.lr *= 0.5
                        self.batch_size = min(self.batch_size * 2, 256)
                        self.wait = 0
                        
        def adaptive_loss(y_true, y_pred):
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
            return mse_loss + 0.01 * l2_loss
            
        model.compile(
            optimizer=optimizers.Adam(learning_rate=initial_lr),
            loss=adaptive_loss
        )
        
        return model, AdaptiveTraining(initial_lr)
        
    def create_hybrid_optimization(self, model: tf.keras.Model,
                                 X: np.ndarray, y: np.ndarray,
                                 n_trials: int = 100) -> Dict:
        """Create hybrid optimization combining multiple techniques"""
        # Architecture optimization
        arch_params = self.optimize_architecture(
            X.shape[1:],
            'transformer',
            n_trials=n_trials // 3
        )
        
        # Learning rate optimization
        best_lr = self.optimize_learning_rate(
            model,
            X, y,
            min_lr=1e-6,
            max_lr=1e-2,
            num_iterations=n_trials // 3
        )
        
        # Batch size optimization
        best_batch_size = self.optimize_batch_size(
            model,
            X, y,
            min_batch_size=16,
            max_batch_size=256,
            num_iterations=n_trials // 3
        )
        
        return {
            'architecture': arch_params,
            'learning_rate': best_lr,
            'batch_size': best_batch_size
        }
        
    def create_uncertainty_quantification(self, model: tf.keras.Model,
                                        X: np.ndarray, y: np.ndarray,
                                        num_samples: int = 100) -> Dict:
        """Quantify prediction uncertainty using multiple methods"""
        # Monte Carlo dropout
        mc_predictions = []
        for _ in range(num_samples):
            pred = model.predict(X, verbose=0)
            mc_predictions.append(pred)
        mc_predictions = np.array(mc_predictions)
        mc_mean = np.mean(mc_predictions, axis=0)
        mc_std = np.std(mc_predictions, axis=0)
        
        # Deep ensembles
        ensemble_predictions = []
        for _ in range(5):
            model_copy = tf.keras.models.clone_model(model)
            model_copy.set_weights(model.get_weights())
            pred = model_copy.predict(X, verbose=0)
            ensemble_predictions.append(pred)
        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0)
        
        # Bayesian neural network
        def bayesian_loss(y_true, y_pred):
            kl_loss = -0.5 * tf.reduce_mean(1 + tf.math.log(tf.square(y_pred)) - tf.square(y_pred))
            return tf.keras.losses.mean_squared_error(y_true, y_pred) + kl_loss
            
        model.compile(
            optimizer=optimizers.Adam(),
            loss=bayesian_loss
        )
        bayesian_pred = model.predict(X, verbose=0)
        
        return {
            'monte_carlo': {'mean': mc_mean, 'std': mc_std},
            'ensemble': {'mean': ensemble_mean, 'std': ensemble_std},
            'bayesian': bayesian_pred
        } 