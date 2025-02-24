from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Multiply, Add, Dense, 
                                   Activation, Flatten, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class WaveNetModel:
    """
    WaveNet model for bearing RUL prediction
    """
    
    def __init__(self, input_shape, learning_rate=0.00001):
        """
        Initialize WaveNet model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        learning_rate : float
            Learning rate for Adam optimizer
        """
        self.model = self._build_model(input_shape, learning_rate)
        
    def _residual_block(self, x, dilation_rate):
        """
        Create a residual block with dilated convolutions
        
        Parameters:
        -----------
        x : tensor
            Input tensor
        dilation_rate : int
            Dilation rate for the convolution
        """
        tanh_out = Conv1D(64, kernel_size=3, 
                         dilation_rate=dilation_rate, 
                         padding='causal')(x)
        tanh_out = Activation('tanh')(tanh_out)
        
        sigm_out = Conv1D(64, kernel_size=3, 
                         dilation_rate=dilation_rate, 
                         padding='causal')(x)
        sigm_out = Activation('sigmoid')(sigm_out)
        
        out = Multiply()([tanh_out, sigm_out])
        out = Conv1D(64, kernel_size=1, padding='same')(out)
        out = Dropout(0.3)(out)
        out = Add()([out, x])
        return out
    
    def _build_model(self, input_shape, learning_rate):
        """
        Build WaveNet model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        learning_rate : float
            Learning rate for Adam optimizer
        """
        input_layer = Input(shape=input_shape)
        
        # Initial convolution
        out = Conv1D(64, kernel_size=10, 
                    padding='causal')(input_layer)
        out = Activation('relu')(out)
        out = Dropout(0.3)(out)
        
        # Residual blocks with dilated convolutions
        skip_connections = []
        for i in range(5):
            out = self._residual_block(out, dilation_rate=2**i)
            skip_connections.append(out)
        
        # Combine skip connections
        out = Add()(skip_connections)
        out = Activation('relu')(out)
        
        # Final convolution and dense layers
        out = Conv1D(1, kernel_size=1, activation='relu')(out)
        out = Flatten()(out)
        out = Dropout(0.3)(out)
        out = Dense(1, activation='relu')(out)
        
        model = Model(input_layer, out)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='mean_squared_error')
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=20, batch_size=32, patience=10):
        """
        Train the WaveNet model
        
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
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance
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