# Bearing Remaining Useful Life Prediction

This repository contains a deep learning project for predicting the Remaining Useful Life (RUL) of rolling element bearings using time series models. The project implements and compares two deep learning architectures:
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)

## Project Overview

Predicting the remaining useful life of mechanical components is a critical task in predictive maintenance. This project focuses on rolling element bearings, which are commonly used in rotating machinery and prone to degradation over time.

The key highlights of this project:
- Time series modeling of bearing degradation using health indicators
- Implementation of both RNN and LSTM architectures for comparison
- Performance evaluation using various metrics (MSE, RMSE, MAE)
- Visualization of predictions against actual RUL values

## Technical Approach

The project treats the RUL prediction problem as a sequence-to-value regression task. The models:
1. Take a sequence of health indicator (HI) values as input
2. Process the temporal patterns in the data
3. Output the predicted remaining useful life in million cycles

## Repository Structure

```
bearing-rul-prediction/
├── data/                   # Data directory
│   └── .gitkeep
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── lstm_model.py       # LSTM model implementation
│   └── rnn_model.py        # RNN model implementation
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_processing.py  # Data loading and preprocessing
│   └── visualization.py    # Plotting and visualization
├── notebooks/              # Jupyter notebooks
│   ├── lstm_analysis.ipynb
│   └── rnn_analysis.ipynb
├── results/                # Results and figures
│   └── .gitkeep
├── requirements.txt        # Dependencies
├── README.md               # This file
├── LICENSE                 # License file
└── main.py                 # Main script to run the pipeline
```

## Data

The project uses `.mat` files containing Health Indicator (HI) and RUL values:
- `Training_data.mat`: Used for model training and validation
- `Test_data.mat`: Used for final model evaluation

Data structure:
- Column 0: Health Indicator (HI) values
- Column 1: Remaining Useful Life (RUL) values

> **Note:** You need to place your `.mat` files in the `data/` directory before running the scripts.

## Models

### RNN Model

The Simple RNN implementation uses:
- Multiple RNN layers with decreasing units (100 → 50 → 25)
- Dropout layers (0.2) for regularization
- Final Dense layer for regression output

### LSTM Model

The LSTM implementation uses:
- Multiple LSTM layers with decreasing units (100 → 50 → 25)
- Dropout layers (0.2) for regularization
- Final Dense layer for regression output

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- SciPy
- Scikit-learn

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bearing-rul-prediction.git
   cd bearing-rul-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your data files in the `data/` directory:
   - `Training_data.mat`
   - `Test_data.mat`

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

## Results

The trained models produce several output files in the `results/` directory:
- Trained model files (`.h5`)
- Training history plots
- Validation results plots
- Test results plots


## License

This project is licensed under the MIT License - see the LICENSE file for details.
