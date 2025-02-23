from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class RNNModel:
    """
    RNN model for bearing RUL prediction
    """
    
    def __init__(self, input_shape, learning_rate=0.001):
        """
        Initialize RNN model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        learning_rate : float
            Learning rate for Adam optimizer
        """
        self.model = self._build_model(input_shape, learning_rate)
        
    def _build_model(self, input_shape, learning_rate):
        """
        Build RNN model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        learning_rate : float
            Learning rate for Adam optimizer
            
        Returns:
        --------
        model : Sequential
            Compiled Keras Sequential model
        """
        model = Sequential()
        model.add(SimpleRNN(units=100, activation='relu',
                          return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=50, activation='relu',
                          return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=25, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='mean_squared_error')
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, patience=10):
        """
        Train the RNN model
        
        Parameters:
        -----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets
        X_val : ndarray
            Validation features
        y_val : ndarray
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        history : History
            Training history
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Parameters:
        -----------
        X : ndarray
            Input features
            
        Returns:
        --------
        y_pred : ndarray
            Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : ndarray
            Input features
        y_true : ndarray
            True target values
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        return metrics, y_pred
    