import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import List, Dict, Tuple, Optional
import optuna
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class AdvancedArchitectures:
    def __init__(self):
        self.models = {}
        self.optimizers = {}
        
    def create_transformer_model(self, input_shape: Tuple[int, int],
                               num_heads: int = 8,
                               num_layers: int = 4,
                               d_model: int = 64,
                               dff: int = 256,
                               dropout_rate: float = 0.1) -> tf.keras.Model:
        """Create a transformer-based model for time series prediction"""
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        pos_encoding = self._positional_encoding(input_shape[0], d_model)
        x = layers.Dense(d_model)(inputs)
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(num_layers):
            x = self._transformer_block(x, num_heads, d_model, dff, dropout_rate)
            
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dff, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _transformer_block(self, x: tf.Tensor, num_heads: int, d_model: int,
                          dff: int, dropout_rate: float) -> tf.Tensor:
        """Create a transformer block"""
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(dff, activation='relu')(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
    
    def _positional_encoding(self, length: int, d_model: int) -> tf.Tensor:
        """Create positional encoding"""
        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
    
    def create_attention_lstm(self, input_shape: Tuple[int, int],
                            lstm_units: List[int] = [64, 32],
                            attention_units: int = 32,
                            dropout_rate: float = 0.2) -> tf.keras.Model:
        """Create an LSTM model with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = inputs
        for units in lstm_units:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.Dropout(dropout_rate)(x)
            
        # Attention mechanism
        attention = layers.Dense(attention_units, activation='tanh')(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Permute((2, 1))(attention)
        x = layers.Multiply()([x, attention])
        
        # Output layers
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_conv_lstm(self, input_shape: Tuple[int, int],
                        conv_filters: List[int] = [32, 64],
                        lstm_units: List[int] = [64, 32],
                        dropout_rate: float = 0.2) -> tf.keras.Model:
        """Create a CNN-LSTM hybrid model"""
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers
        x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
        for filters in conv_filters:
            x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(dropout_rate)(x)
            
        # LSTM layers
        x = layers.Reshape((x.shape[1], -1))(x)
        for units in lstm_units:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.Dropout(dropout_rate)(x)
            
        # Output layers
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_tcn(self, input_shape: Tuple[int, int],
                  num_filters: int = 64,
                  kernel_size: int = 3,
                  num_layers: int = 4,
                  dropout_rate: float = 0.2) -> tf.keras.Model:
        """Create a Temporal Convolutional Network"""
        inputs = layers.Input(shape=input_shape)
        
        # TCN layers
        x = inputs
        for i in range(num_layers):
            dilation_rate = 2 ** i
            x = layers.Conv1D(
                num_filters,
                kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_hybrid_model(self, input_shape: Tuple[int, int],
                          transformer_params: Dict,
                          lstm_params: Dict,
                          conv_lstm_params: Dict) -> tf.keras.Model:
        """Create a hybrid model combining multiple architectures"""
        inputs = layers.Input(shape=input_shape)
        
        # Transformer branch
        transformer = self.create_transformer_model(input_shape, **transformer_params)
        x1 = transformer(inputs)
        
        # LSTM branch
        lstm = self.create_attention_lstm(input_shape, **lstm_params)
        x2 = lstm(inputs)
        
        # CNN-LSTM branch
        conv_lstm = self.create_conv_lstm(input_shape, **conv_lstm_params)
        x3 = conv_lstm(inputs)
        
        # Combine branches
        x = layers.Concatenate()([x1, x2, x3])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_pytorch_model(self, input_size: int,
                           hidden_sizes: List[int] = [64, 32],
                           dropout_rate: float = 0.2) -> nn.Module:
        """Create a PyTorch model with advanced features"""
        class StockPredictor(nn.Module):
            def __init__(self, input_size, hidden_sizes, dropout_rate):
                super(StockPredictor, self).__init__()
                
                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_sizes[0],
                    num_layers=2,
                    batch_first=True,
                    dropout=dropout_rate,
                    bidirectional=True
                )
                
                # Attention mechanism
                self.attention = nn.Sequential(
                    nn.Linear(hidden_sizes[0] * 2, hidden_sizes[0]),
                    nn.Tanh(),
                    nn.Linear(hidden_sizes[0], 1)
                )
                
                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Linear(hidden_sizes[0] * 2, hidden_sizes[1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_sizes[1], 1)
                )
                
            def forward(self, x):
                # LSTM
                lstm_out, _ = self.lstm(x)
                
                # Attention
                attention_weights = self.attention(lstm_out)
                attention_weights = torch.softmax(attention_weights, dim=1)
                context = torch.sum(attention_weights * lstm_out, dim=1)
                
                # Output
                output = self.fc(context)
                return output
            
        return StockPredictor(input_size, hidden_sizes, dropout_rate)
    
    def optimize_architecture(self, X: np.ndarray, y: np.ndarray,
                            architecture_type: str = 'transformer',
                            n_trials: int = 50) -> Dict:
        """Optimize model architecture using Optuna"""
        def objective(trial):
            if architecture_type == 'transformer':
                params = {
                    'num_heads': trial.suggest_int('num_heads', 2, 16),
                    'num_layers': trial.suggest_int('num_layers', 2, 8),
                    'd_model': trial.suggest_int('d_model', 32, 256),
                    'dff': trial.suggest_int('dff', 128, 512),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5)
                }
                model = self.create_transformer_model(X.shape[1:], **params)
            elif architecture_type == 'attention_lstm':
                params = {
                    'lstm_units': [trial.suggest_int('lstm_units_1', 32, 128),
                                 trial.suggest_int('lstm_units_2', 16, 64)],
                    'attention_units': trial.suggest_int('attention_units', 16, 64),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5)
                }
                model = self.create_attention_lstm(X.shape[1:], **params)
            else:
                raise ValueError(f"Unknown architecture type: {architecture_type}")
                
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
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