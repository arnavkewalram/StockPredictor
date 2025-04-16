import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
import optuna
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

class NeuralArchitectures:
    def __init__(self):
        self.models = {}
        self.optimizers = {}
        self.techniques = {}
        
    def create_transformer_model(self, input_shape: Tuple[int, int],
                               num_heads: int = 8,
                               num_layers: int = 6,
                               d_model: int = 512) -> tf.keras.Model:
        """Create a transformer-based model for time series prediction"""
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        pos_encoding = self._positional_encoding(input_shape[0], d_model)
        x = layers.Dense(d_model)(inputs)
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(num_layers):
            x = self._transformer_block(x, num_heads, d_model)
            
        # Output layer
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _transformer_block(self, x: tf.Tensor,
                         num_heads: int,
                         d_model: int) -> tf.Tensor:
        """Create a transformer block with multi-head attention"""
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
        """Generate positional encoding for transformer"""
        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
    
    def create_attention_lstm(self, input_shape: Tuple[int, int],
                            hidden_dim: int = 64,
                            num_layers: int = 2) -> tf.keras.Model:
        """Create an LSTM model with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = inputs
        for _ in range(num_layers):
            x = layers.LSTM(hidden_dim, return_sequences=True)(x)
            
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Softmax(axis=1)(attention)
        x = layers.Multiply()([x, attention])
        
        # Output layer
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_conv_lstm(self, input_shape: Tuple[int, int],
                        filters: int = 64,
                        kernel_size: int = 3,
                        lstm_units: int = 64) -> tf.keras.Model:
        """Create a CNN-LSTM hybrid model"""
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers
        x = layers.Conv1D(filters, kernel_size, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # LSTM layer
        x = layers.LSTM(lstm_units)(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_tcn(self, input_shape: Tuple[int, int],
                  filters: int = 64,
                  kernel_size: int = 3,
                  num_layers: int = 3) -> tf.keras.Model:
        """Create a Temporal Convolutional Network"""
        inputs = layers.Input(shape=input_shape)
        
        # TCN layers
        x = inputs
        for i in range(num_layers):
            dilation_rate = 2 ** i
            x = layers.Conv1D(filters, kernel_size,
                            padding='causal',
                            dilation_rate=dilation_rate)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
        # Output layer
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_hybrid_model(self, input_shape: Tuple[int, int],
                          transformer_params: Dict,
                          lstm_params: Dict,
                          cnn_params: Dict) -> tf.keras.Model:
        """Create a hybrid model combining multiple architectures"""
        inputs = layers.Input(shape=input_shape)
        
        # Transformer branch
        transformer = self.create_transformer_model(input_shape, **transformer_params)
        x1 = transformer(inputs)
        
        # LSTM branch
        lstm = self.create_attention_lstm(input_shape, **lstm_params)
        x2 = lstm(inputs)
        
        # CNN branch
        cnn = self.create_conv_lstm(input_shape, **cnn_params)
        x3 = cnn(inputs)
        
        # Feature fusion
        x = layers.Concatenate()([x1, x2, x3])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_pytorch_model(self, input_shape: Tuple[int, int],
                           hidden_dim: int = 64,
                           num_layers: int = 2) -> nn.Module:
        """Create a PyTorch model with advanced features"""
        class StockPredictor(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):
                super(StockPredictor, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1)
                )
                self.fc = nn.Linear(hidden_dim, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attention_weights = self.attention(lstm_out)
                attention_weights = nn.functional.softmax(attention_weights, dim=1)
                context = torch.sum(lstm_out * attention_weights, dim=1)
                return self.fc(context)
                
        return StockPredictor(input_shape[1], hidden_dim, num_layers)
    
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