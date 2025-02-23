# Bearing Remaining Useful Life Prediction

This repository contains a deep learning project for predicting the Remaining Useful Life (RUL) of rolling element bearings using time series models. The project implements and compares two deep learning architectures:
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)

## Project Overview

Predicting the remaining useful life of mechanical components is a critical task in predictive maintenance. This project focuses on rolling element bearings, which are commonly used in rotating machinery and prone to degradation over time.

The key highlights of this project:
- Time series modeling of bearing degradation using RMS health indicators
- Implementation of both RNN and LSTM architectures for comparison
- Performance evaluation using MSE metric
- Visualization of predictions against normalized RUL values

## Technical Approach

The project treats the RUL prediction problem as a sequence-to-value regression task. The workflow:
1. Load vibration signals from Hugging Face dataset
2. Process signals into frames and calculate RMS values
3. Apply moving average smoothing to RMS values
4. Create sequences for time series modeling
5. Train models to predict normalized RUL (1.0 -> 0.0)

## Repository Structure

```
bearing-rul-prediction/
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── lstm_model.py      # LSTM model implementation
│   └── rnn_model.py       # RNN model implementation
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── data_processing.py # Data loading and preprocessing
│   └── visualization.py   # Plotting and visualization
├── results/               # Results and figures
│   └── .gitkeep
├── README.md              # This file
├── LICENSE               # License file
└── main.py               # Main script to run the pipeline
```

## Data

The project uses the "alidi/bearing-run2failure-unsw" dataset from Hugging Face, which is a subset of the UNSW bearing run-to-failure dataset. The dataset includes:

### Dataset Characteristics
- Horizontal acceleration (accH) measurements
- Test 1 and Test 3 from the original dataset
- Sampling frequency: 51200 Hz
- Operating conditions:
  - Speed: 6.0 Hz
  - Radial load: 10.5 kN
- Fault characteristics:
  - Defect sizes: 1.0 mm (Test 1) and 0.5 mm (Test 3)
  - Defect type: Ball Pass Frequency Outer race (BPFO)

### Data Structure
Each sample in the dataset contains:
- Signal data (raw acceleration measurements)
- Signal length
- Sampling frequency
- Operating parameters (speed, load)
- Fault information (defect size, type)

### Source
The original dataset is available at: [10.17632/h4df4mgrfb.3](10.17632/h4df4mgrfb.3)


## Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Hugging Face datasets
- NumPy
- Matplotlib
- Librosa
- Scikit-learn

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bearing-rul-prediction.git
   cd bearing-rul-prediction
   ```

2. Create and activate Poetry environment:
   ```bash
   poetry install
   ```

### Usage

To train an LSTM model:
```bash
python main.py --model lstm --seq_length 200 --epochs 20
```

To train an RNN model:
```bash
python main.py --model rnn --seq_length 100 --epochs 50
```

Additional arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--patience`: Patience for early stopping (default: 10)
- `--frame_length`: Frame length for initial signal framing (default: 51200)
- `--hop_length`: Hop length for initial signal framing (default: 12800)

## Data Processing Pipeline

1. Signal Processing:
   - Split raw signals into frames using librosa
   - Calculate RMS for each frame
   - Apply moving average smoothing

2. RUL Calculation:
   - Use linear degradation pattern
   - Normalize to range [1.0, 0.0]
   - 1.0 represents healthy state
   - 0.0 represents failure state

3. Sequence Creation:
   - Create overlapping sequences for time series modeling
   - Split into train/validation/test sets
   - Maintain temporal order of sequences

## Results

The trained models produce several output files in the `results/` directory:
- Training history plots
- Validation results plots
- Test results plots

Plots show:
- Training and validation loss over epochs
- Actual vs predicted normalized RUL values
- Performance comparison between RNN and LSTM models

## License

This project is licensed under the MIT License - see the LICENSE file for details.