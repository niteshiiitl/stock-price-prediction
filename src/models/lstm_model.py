"""
LSTM Model for Stock Price Prediction
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import joblib

class LSTMStockPredictor:
    def __init__(self, sequence_length: int = 60, features: int = 5):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Select features: Open, High, Low, Close, Volume
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in data.columns for col in features):
            raise ValueError(f"Data must contain columns: {features}")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 3])  # Close price index
            
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """Train the LSTM model"""
        X, y = self.prepare_data(data)
        
        if self.model is None:
            self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, data: pd.DataFrame, days_ahead: int = 1) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get last sequence_length days
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        last_sequence = data[features].tail(self.sequence_length)
        
        # Scale the data
        scaled_sequence = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, self.features)
            
            # Make prediction
            pred_scaled = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[3] = pred_scaled  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), self.features))
        dummy_array[:, 3] = predictions
        predictions_unscaled = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        return predictions_unscaled
    
    def save_model(self, filepath: str):
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(f"{filepath}_model.h5")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
    
    def load_model(self, filepath: str):
        """Load model and scaler"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True