import os
import argparse
import time
import numpy as np

from utils.data_processing import prepare_data
from utils.visualization import (
    plot_training_history, plot_validation_results, 
    plot_test_results, save_plot
)
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from models.wavenet_model import WaveNetModel

def train_model(model_type, sequence_length, batch_size, epochs, patience, frame_length, hop_length, filter_window_size):
    """
    Train a model on the bearing RUL prediction task
    
    Parameters:
    -----------
    model_type : str
        Type of model to train ('lstm' or 'rnn' or 'wavenet')
    sequence_length : int
        Length of input sequences
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    patience : int
        Patience for early stopping
    frame_length : int
        Frame length for signal processing
    hop_length : int
        Hop length for signal processing
    filter_window_size : int
        Window size for moving average filter
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    
    # Prepare data
    print(f"Preparing data with sequence length {sequence_length}...")
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data(
        frame_length=frame_length, 
        hop_length=hop_length, 
        sequence_length=sequence_length,
        filter_window_size=filter_window_size
    )
    
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create and train model
    print(f"Creating {model_type.upper()} model...")
    input_shape = (X_train.shape[1], 1)  # (sequence_length, features)
    
    if model_type.lower() == 'lstm':
        model = LSTMModel(input_shape)
    elif model_type.lower() == 'rnn':
        model = RNNModel(input_shape)
    elif model_type.lower() == 'wavenet':
        model = WaveNetModel(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Training {model_type.upper()} model...")
    start_time = time.time()
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, patience=patience
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    val_metrics, y_val_pred = model.evaluate(X_val, y_val)
    
    # Evaluate on test set
    test_metrics, y_test_pred = model.evaluate(X_test, y_test)
    
    
    # Plot and save results
    print("Generating plots...")
    
    # Training history plot
    history_fig = plot_training_history(history)
    save_plot(history_fig, f"results/{model_type}_training_history.png")
    
    # Validation results plot
    val_fig = plot_validation_results(y_val, y_val_pred, val_metrics['mse'])
    save_plot(val_fig, f"results/{model_type}_validation_results.png")
    
    # Test results plot
    test_fig = plot_test_results(y_test, y_test_pred, test_metrics['mse'], start_idx=90)
    save_plot(test_fig, f"results/{model_type}_test_results.png")
    
    return test_metrics, val_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for bearing RUL prediction")
    parser.add_argument('--model', type=str, choices=['lstm', 'rnn', 'wavenet'], required=True,
                        help='Type of model to train (lstm or rnn or wavenet)')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length for time series input')
    parser.add_argument('--frame_length', type=int, default=42000,
                        help='Frame length for signal processing')
    parser.add_argument('--hop_length', type=int, default=21000,
                        help='Hop length for signal processing')
    parser.add_argument('--filter_window_size', type=int, default=11,
                        help='Window size for moving average filter')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    
    args = parser.parse_args()
    
    test_metrics, val_metrics = train_model(
        args.model, args.seq_length, args.batch_size, args.epochs, args.patience, args.frame_length, args.hop_length, args.filter_window_size
    )
    
    print(f"Model training complete.")
    print(f"Validation metrics: MSE={val_metrics['mse']:.6f}, RMSE={val_metrics['rmse']:.6f}, MAE={val_metrics['mae']:.6f}")
    print(f"Test metrics: MSE={test_metrics['mse']:.6f}, RMSE={test_metrics['rmse']:.6f}, MAE={test_metrics['mae']:.6f}")