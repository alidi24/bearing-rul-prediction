import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return plt.gcf()

def plot_validation_results(y_val, y_pred, val_metric):
    """Plot actual vs predicted normalized RUL values for validation set"""
    # Sort by actual RUL in descending order (1 -> 0)
    sorted_indices = np.argsort(y_val)[::-1]
    y_val_sorted = y_val[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_val_sorted, label='Actual RUL')
    plt.plot(y_pred_sorted, label='Predicted RUL')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized RUL')
    plt.title(f"Validation Set: Actual vs Predicted RUL - MSE: {val_metric:.2f}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add horizontal lines at 0 and 1
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.1, 1.1)  # Add some padding to y-axis
    
    plt.tight_layout()
    return plt.gcf()

def plot_test_results(y_test, y_pred, test_metric, start_idx=0):
    """Plot actual vs predicted normalized RUL values for test set"""
    y_test_subset = y_test[start_idx:]
    y_pred_subset = y_pred[start_idx:]
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_subset, label='Actual RUL')
    plt.plot(y_pred_subset, label='Predicted RUL')
    
    # Create percentage-based x-axis ticks
    tick_values = np.linspace(0, len(y_test_subset)-1, 5, dtype=int)
    tick_labels = ['100%', '75%', '50%', '25%', '0%']
    
    plt.xticks(tick_values, tick_labels)
    plt.xlabel('Remaining Life Percentage', fontsize=12)
    plt.ylabel('Normalized RUL', fontsize=12)
    plt.title(f"Test Set: Actual vs Predicted RUL - MSE: {test_metric:.4f}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add horizontal lines at 0 and 1
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.1, 1.1)  # Add some padding to y-axis
    
    plt.tight_layout()
    return plt.gcf()

def save_plot(fig, filename, dpi=300):
    """Save plot to file"""
    fig.savefig(filename, format='png', dpi=dpi, bbox_inches='tight')