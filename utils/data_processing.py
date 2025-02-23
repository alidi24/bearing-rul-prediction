import numpy as np
import librosa
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class SignalProcessor:
    def __init__(self, frame_length: int, hop_length: int):
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def split_and_process(self, examples):
        """Split signals into frames and calculate RMS"""
        frames = []
        rms_values = []
        metadata = []
        
        for signal in examples['signal']:
            # Split into frames
            signal_frames = librosa.util.frame(
                np.array(signal)*100.0,     # Convert from V to m/s² (sensitivity: 10 mV/ms-2)
                frame_length=self.frame_length, 
                hop_length=self.hop_length
            ).T
            
            # Calculate RMS for each frame
            frame_rms = np.sqrt(np.mean(signal_frames**2, axis=1))
            
            frames.extend(signal_frames)
            rms_values.extend(frame_rms)
            metadata.extend([{'defect_size': examples['defect_size'][0]}] * len(frame_rms))
            
        return frames, rms_values, metadata

def smooth_rms(rms_values, window_size=22):
    """Apply moving average filter to smooth RMS values"""
    window = np.ones(window_size) / window_size
    return np.convolve(rms_values, window, mode='valid')



def prepare_data(frame_length=1024, hop_length=512, sequence_length=100):
    """Main function to prepare data
    
    Args:
        frame_length: Length of each frame from the signal
        hop_length: Number of samples between frames
        sequence_length: Length of sequences for RNN/LSTM input
    """
    # Load datasets from Hugging Face (for the original dataset, please refer to 10.17632/h4df4mgrfb.3)
    ds_train = load_dataset("alidi/bearing-run2failure-unsw", split="test1")
    ds_test = load_dataset("alidi/bearing-run2failure-unsw", split="test3")
    
    # Initialize processor
    processor = SignalProcessor(frame_length, hop_length)
    
    # Process training data
    _, rms_train, _ = processor.split_and_process(ds_train)
    rms_train_smooth = smooth_rms(rms_train)
    
    # Process test data
    _, rms_test, _ = processor.split_and_process(ds_test)
    rms_test_smooth = smooth_rms(rms_test)
    
    def create_sequences(data, rul, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            target = rul[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    # Prepare health indicator and RUL
    X_train_raw = rms_train_smooth.reshape(-1, 1)
    X_test_raw = rms_test_smooth.reshape(-1, 1)
    
    # Calculate RUL using linear degradation pattern
    y_train_raw = np.linspace(1, 0, len(X_train_raw))
    y_test_raw = np.linspace(1, 0, len(X_test_raw))

    
    # Create sequences
    X_seq, y_seq = create_sequences(X_train_raw, y_train_raw, sequence_length)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Create test sequences
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, sequence_length)
    
    # Reshape for RNN/LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], sequence_length, 1)
    X_val = X_val.reshape(X_val.shape[0], sequence_length, 1)
    X_test = X_test.reshape(X_test.shape[0], sequence_length, 1)
    
    return X_train, X_val, y_train, y_val, X_test, y_test

if __name__ == "__main__":
    # Example usage
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")