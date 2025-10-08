"""LSTM model training script for stock price direction prediction.

WHAT: Trains an LSTM neural network to predict stock price direction (up/down/steady)
WHY: Deep learning models can capture temporal patterns in sequential stock data
HOW: Uses TensorFlow/Keras to build and train a multi-layer LSTM with probability calibration
DATA: Consumes Kaggle S&P 500 historical data, outputs trained model and calibration artifacts

This module implements:
- Sequence-based feature engineering from OHLCV data
- LSTM architecture with 128->64 hidden layers
- Three-class classification (up, down, steady)
- Probability calibration to ensure confidence accuracy
- Train/validation split for reliable evaluation
"""
from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

# Scikit-learn for preprocessing and calibration
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, log_loss

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LSTM training process.

    WHAT: Centralized settings for model architecture and training
    WHY: Makes hyperparameters easy to adjust without code changes
    DATA: All training parameters in one place
    """
    # Data paths
    data_dir: str = "data/training/raw/kaggle_sp500/individual_stocks_5yr/individual_stocks_5yr"
    model_save_dir: str = "pipelines/realtime/models/saved_models"

    # Model architecture
    sequence_length: int = 30  # Days of history to use for prediction
    lstm_layer1_units: int = 128  # First LSTM layer size
    lstm_layer2_units: int = 64   # Second LSTM layer size
    dropout_rate: float = 0.2     # Dropout for regularization

    # Training parameters
    batch_size: int = 64
    epochs: int = 50
    validation_split: float = 0.2  # 20% for validation
    calibration_split: float = 0.1  # 10% for calibration (from train set)
    learning_rate: float = 0.001

    # Class labels
    num_classes: int = 3  # up, down, steady
    price_change_threshold: float = 0.02  # ±2% for up/down classification (if not using ATR)

    # ATR-based dynamic thresholding
    use_atr_threshold: bool = True  # Use ATR-based threshold instead of fixed percentage
    atr_window: int = 14  # ATR calculation window (standard is 14 days)
    atr_multiplier: float = 0.3  # Threshold = ATR * multiplier (0.3 gives best balance)


class LSTMDataProcessor:
    """Processes stock data into sequences for LSTM training.

    WHAT: Converts raw OHLCV data into feature sequences and labels
    WHY: LSTM models require fixed-length sequential input
    HOW: Creates rolling windows of historical data with technical indicators
    DATA: Raw CSV files -> normalized sequence arrays + class labels
    """

    def __init__(self, config: TrainingConfig):
        """Initialize data processor with configuration.

        WHAT: Sets up processor with training parameters
        WHY: Need config to know sequence length and thresholds
        DATA: Stores config reference for processing
        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = []

    def load_all_stock_data(self) -> pd.DataFrame:
        """Load and combine all stock CSV files.

        WHAT: Reads all individual stock files and combines into single DataFrame
        WHY: Need unified dataset for training across multiple stocks
        HOW: Iterates through CSV files, loads each, concatenates
        DATA: Individual CSV files -> combined pandas DataFrame

        Returns:
            Combined DataFrame with all stocks' OHLCV data
        """
        # WHAT: Get list of all CSV files in the data directory
        # WHY: Need to process each stock's historical data
        # HOW: Use pathlib to find all .csv files
        data_path = Path(self.config.data_dir)
        csv_files = list(data_path.glob("*.csv"))

        logger.info(f"Found {len(csv_files)} stock data files")

        # WHAT: Load each CSV and store in list
        # WHY: Pandas concat is more efficient with list than iterative concat
        # DATA: Each CSV -> DataFrame with OHLCV columns
        dataframes = []

        for csv_file in csv_files:
            try:
                # WHAT: Read CSV with date parsing
                # WHY: Need datetime index for time series operations
                # HOW: Use pandas read_csv with parse_dates
                df = pd.read_csv(csv_file, parse_dates=['date'])

                # WHAT: Add ticker symbol from filename
                # WHY: Need to identify which stock each row belongs to
                # HOW: Extract ticker from filename (e.g., "AAPL_data.csv" -> "AAPL")
                ticker = csv_file.stem.replace('_data', '')
                df['ticker'] = ticker

                dataframes.append(df)

            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                continue

        # WHAT: Combine all DataFrames into one
        # WHY: Need unified dataset for training
        # HOW: Pandas concat with ignore_index to create continuous index
        # DATA: List of DataFrames -> single combined DataFrame
        combined_df = pd.concat(dataframes, ignore_index=True)

        # WHAT: Sort by ticker and date
        # WHY: Ensures chronological order within each stock
        # HOW: Sort by ticker first, then date
        combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)

        logger.info(f"Loaded {len(combined_df)} total data points from {len(csv_files)} stocks")

        return combined_df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features.

        WHAT: Computes RSI, MACD, Bollinger Bands, etc. for each stock
        WHY: Technical indicators capture patterns that LSTMs can learn from
        HOW: Group by ticker, calculate indicators, add as new columns
        DATA: OHLCV DataFrame -> DataFrame with added indicator columns

        Args:
            df: DataFrame with OHLCV data and ticker column

        Returns:
            DataFrame with added technical indicator columns
        """
        # WHAT: Create copy to avoid modifying original
        # WHY: Don't want side effects on input data
        df = df.copy()

        # WHAT: Calculate returns (percentage change in closing price)
        # WHY: Returns are more stationary than raw prices for ML
        # HOW: Group by ticker, calculate pct_change for each stock separately
        # DATA: close prices -> percentage change values
        df['returns'] = df.groupby('ticker')['close'].pct_change()

        # WHAT: Calculate RSI (Relative Strength Index)
        # WHY: Measures momentum and overbought/oversold conditions
        # HOW: Rolling window of gains vs losses
        df['rsi'] = df.groupby('ticker')['close'].transform(self._calculate_rsi)

        # WHAT: Calculate moving averages
        # WHY: Smooth out noise, identify trends
        # HOW: Rolling mean over different windows
        df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
        df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())

        # WHAT: Calculate MACD (Moving Average Convergence Divergence)
        # WHY: Shows relationship between two moving averages
        # HOW: EMA(12) - EMA(26)
        ema_12 = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12).mean())
        ema_26 = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=26).mean())
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9).mean())

        # WHAT: Calculate Bollinger Bands
        # WHY: Show volatility and potential reversal points
        # HOW: SMA ± (std * 2)
        df['bb_middle'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
        bb_std = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).std())
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # WHAT: Calculate volume ratio
        # WHY: Volume changes can signal trend strength
        # HOW: Current volume / rolling average volume
        df['volume_ratio'] = df.groupby('ticker')['volume'].transform(
            lambda x: x / x.rolling(10).mean()
        )

        # WHAT: Calculate volatility
        # WHY: Measures price uncertainty
        # HOW: Standard deviation of returns over rolling window
        df['volatility'] = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(10).std()
        )

        return df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator.

        WHAT: Computes Relative Strength Index
        WHY: RSI is a key momentum indicator
        HOW: Average gain / average loss over window period
        DATA: Price series -> RSI values (0-100)

        Args:
            prices: Series of closing prices
            window: RSI calculation window (typically 14)

        Returns:
            Series of RSI values
        """
        # WHAT: Calculate price changes (deltas)
        # WHY: RSI based on gains vs losses
        delta = prices.diff()

        # WHAT: Separate gains and losses
        # WHY: RSI compares average gains to average losses
        # HOW: Clip negatives to 0 for gains, positives to 0 for losses
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        # WHAT: Calculate rolling averages
        # WHY: Need average gain and loss over window
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()

        # WHAT: Calculate RS (Relative Strength) and RSI
        # WHY: RSI formula: 100 - (100 / (1 + RS))
        # HOW: Handle division by zero with where clause
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR) for each stock.

        WHAT: Computes ATR to measure stock volatility
        WHY: ATR provides stock-specific volatility measure for dynamic thresholds
        HOW: Calculate True Range, then rolling average over window period
        DATA: OHLC prices -> ATR values (absolute and percentage)

        Args:
            df: DataFrame with OHLC price data and ticker column

        Returns:
            DataFrame with added 'atr' and 'atr_pct' columns
        """
        # WHAT: Calculate True Range components
        # WHY: TR captures the full price movement range including gaps
        # HOW: Max of three ranges: high-low, |high-prev_close|, |low-prev_close|
        # DATA: OHLC -> three range measures

        # WHAT: High-Low range (intraday range)
        # WHY: Captures normal daily volatility
        df['_high_low'] = df['high'] - df['low']

        # WHAT: High minus previous close
        # WHY: Captures gap-up scenarios
        # HOW: Group by ticker to avoid cross-stock calculations
        df['_high_close'] = abs(df.groupby('ticker')['close'].shift(1) - df['high'])

        # WHAT: Low minus previous close
        # WHY: Captures gap-down scenarios
        df['_low_close'] = abs(df.groupby('ticker')['close'].shift(1) - df['low'])

        # WHAT: True Range is the maximum of the three components
        # WHY: Captures the largest price movement regardless of gaps
        # HOW: Take element-wise max across three columns
        # DATA: Three range components -> single TR value
        df['_true_range'] = df[['_high_low', '_high_close', '_low_close']].max(axis=1)

        # WHAT: Calculate ATR as rolling average of True Range
        # WHY: Smooths out daily volatility into consistent measure
        # HOW: Rolling mean over window period, grouped by ticker
        # DATA: TR -> ATR (smoothed volatility measure)
        df['atr'] = df.groupby('ticker')['_true_range'].transform(
            lambda x: x.rolling(window=self.config.atr_window, min_periods=1).mean()
        )

        # WHAT: Convert ATR to percentage of price
        # WHY: Makes ATR comparable across different price levels
        # HOW: Divide ATR by closing price
        # DATA: ATR (absolute) -> ATR_pct (relative to price)
        df['atr_pct'] = df['atr'] / df['close']

        # WHAT: Clean up temporary columns
        # WHY: Keep DataFrame tidy, only keep final ATR values
        df = df.drop(columns=['_high_low', '_high_close', '_low_close', '_true_range'])

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create classification labels based on future price movement.

        WHAT: Labels each sample as up/down/steady based on next-day price change
        WHY: Need target variable for supervised learning
        HOW: Calculate next-day return, classify based on threshold (fixed or ATR-based)
        DATA: Price data -> class labels (0=down, 1=steady, 2=up)

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added 'label' column
        """
        # WHAT: Calculate next-day price change
        # WHY: Predict tomorrow's direction based on today's features
        # HOW: Shift close prices by -1 to get next day's price
        # DATA: Today's close, tomorrow's close -> percentage change
        df['future_close'] = df.groupby('ticker')['close'].shift(-1)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']

        # WHAT: Determine threshold method (fixed or ATR-based)
        # WHY: ATR-based adapts to each stock's volatility
        # HOW: Check config flag, apply appropriate classification
        if self.config.use_atr_threshold:
            # WHAT: Use ATR-based dynamic threshold
            # WHY: Accounts for stock-specific volatility patterns
            # HOW: Calculate threshold = ATR_pct * multiplier for each row
            # DATA: ATR percentage + multiplier -> dynamic threshold

            logger.info(f"Using ATR-based thresholds (ATR window={self.config.atr_window}, multiplier={self.config.atr_multiplier})")

            # WHAT: Ensure ATR is calculated
            # WHY: Need ATR values for threshold calculation
            if 'atr_pct' not in df.columns:
                raise ValueError("ATR must be calculated before creating labels. Call calculate_atr() first.")

            # WHAT: Calculate dynamic threshold for each row
            # WHY: Each stock/day has unique volatility
            # DATA: ATR_pct * multiplier -> threshold
            df['threshold'] = df['atr_pct'] * self.config.atr_multiplier

            # WHAT: Classify returns using dynamic threshold
            # WHY: More balanced classes across different volatility regimes
            # HOW: Compare return to per-row threshold
            def classify_return_atr(row):
                if pd.isna(row['future_return']) or pd.isna(row['threshold']):
                    return np.nan
                elif row['future_return'] > row['threshold']:
                    return 2  # up
                elif row['future_return'] < -row['threshold']:
                    return 0  # down
                else:
                    return 1  # steady

            df['label'] = df.apply(classify_return_atr, axis=1)

            # WHAT: Log threshold statistics
            # WHY: Understand the range of thresholds being applied
            logger.info(f"ATR threshold statistics:")
            logger.info(f"  Mean threshold: {df['threshold'].mean()*100:.3f}%")
            logger.info(f"  Median threshold: {df['threshold'].median()*100:.3f}%")
            logger.info(f"  Min threshold: {df['threshold'].min()*100:.3f}%")
            logger.info(f"  Max threshold: {df['threshold'].max()*100:.3f}%")

        else:
            # WHAT: Use fixed percentage threshold
            # WHY: Simple, interpretable classification
            # HOW: Apply same threshold to all samples
            # DATA: Continuous return -> discrete class (0, 1, or 2)

            logger.info(f"Using fixed threshold: ±{self.config.price_change_threshold*100}%")

            threshold = self.config.price_change_threshold

            def classify_return_fixed(ret):
                if pd.isna(ret):
                    return np.nan
                elif ret > threshold:
                    return 2  # up
                elif ret < -threshold:
                    return 0  # down
                else:
                    return 1  # steady

            df['label'] = df['future_return'].apply(classify_return_fixed)

        # WHAT: Log class distribution
        # WHY: Check for class imbalance issues
        label_counts = df['label'].value_counts().sort_index()
        total = label_counts.sum()
        logger.info(f"Label distribution:")
        for label in [0, 1, 2]:
            count = label_counts.get(label, 0)
            pct = count / total * 100 if total > 0 else 0
            label_name = ['down', 'steady', 'up'][label]
            logger.info(f"  Class {label} ({label_name:6s}): {count:7d} samples ({pct:5.1f}%)")

        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequential training data for LSTM.

        WHAT: Converts DataFrame into sequence arrays for LSTM input
        WHY: LSTMs require fixed-length sequential input
        HOW: Rolling window approach - create overlapping sequences
        DATA: DataFrame -> (sequences, labels) arrays

        Args:
            df: Processed DataFrame with features and labels

        Returns:
            Tuple of (X_sequences, y_labels) where:
            - X_sequences: shape (n_samples, sequence_length, n_features)
            - y_labels: shape (n_samples,) with class labels
        """
        # WHAT: Define feature columns to use
        # WHY: Need consistent feature set for model
        # HOW: Select all calculated indicators, exclude metadata
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'rsi', 'sma_10', 'sma_20',
            'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower',
            'volume_ratio', 'volatility'
        ]
        self.feature_columns = feature_cols

        # WHAT: Drop rows with NaN values
        # WHY: Can't train on missing data, indicators create NaNs at start
        # HOW: Use dropna with subset of required columns
        required_cols = feature_cols + ['label']
        df_clean = df.dropna(subset=required_cols).copy()

        logger.info(f"After dropping NaN: {len(df_clean)} samples")

        # WHAT: Create sequences for each stock separately
        # WHY: Don't want sequences that span across different stocks
        # HOW: Group by ticker, create sequences within each group
        X_list = []
        y_list = []

        for ticker, group in df_clean.groupby('ticker'):
            # WHAT: Extract features and labels for this stock
            # DATA: DataFrame -> numpy arrays
            features = group[feature_cols].values
            labels = group['label'].values

            # WHAT: Create rolling window sequences
            # WHY: LSTM needs (sequence_length, n_features) input
            # HOW: Iterate with sliding window
            # DATA: Flat arrays -> 3D sequence array
            for i in range(len(features) - self.config.sequence_length):
                # WHAT: Extract sequence of length 30
                # WHY: Each prediction uses 30 days of history
                X_seq = features[i:i + self.config.sequence_length]

                # WHAT: Label is the target for the day after the sequence
                # WHY: Predict next day's direction
                y_label = labels[i + self.config.sequence_length]

                X_list.append(X_seq)
                y_list.append(y_label)

        # WHAT: Convert lists to numpy arrays
        # WHY: TensorFlow requires numpy array inputs
        # DATA: List of arrays -> single stacked array
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        logger.info(f"Created {len(X)} sequences with shape {X.shape}")

        return X, y

    def normalize_sequences(self, X_train: np.ndarray, X_val: np.ndarray,
                          X_cal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize feature sequences using StandardScaler.

        WHAT: Scales features to zero mean and unit variance
        WHY: Neural networks train better with normalized inputs
        HOW: Fit scaler on training data, transform all sets
        DATA: Raw sequences -> normalized sequences

        Args:
            X_train: Training sequences
            X_val: Validation sequences
            X_cal: Calibration sequences

        Returns:
            Tuple of normalized (X_train, X_val, X_cal)
        """
        # WHAT: Reshape for StandardScaler (expects 2D)
        # WHY: Scaler works on 2D arrays, we have 3D sequences
        # HOW: Flatten sequence and feature dimensions, preserving sample dimension
        # DATA: (n_samples, seq_len, features) -> (n_samples, seq_len*features)
        n_samples_train, seq_len, n_features = X_train.shape

        X_train_2d = X_train.reshape(-1, seq_len * n_features)
        X_val_2d = X_val.reshape(-1, seq_len * n_features)
        X_cal_2d = X_cal.reshape(-1, seq_len * n_features)

        # WHAT: Fit scaler on training data only
        # WHY: Prevent data leakage from validation/calibration sets
        # HOW: Use fit_transform on train, transform on others
        X_train_scaled = self.scaler.fit_transform(X_train_2d)
        X_val_scaled = self.scaler.transform(X_val_2d)
        X_cal_scaled = self.scaler.transform(X_cal_2d)

        # WHAT: Reshape back to 3D
        # WHY: LSTM expects (samples, sequence_length, features)
        # DATA: (samples, seq*feat) -> (samples, seq, feat)
        X_train_norm = X_train_scaled.reshape(n_samples_train, seq_len, n_features)
        X_val_norm = X_val_scaled.reshape(-1, seq_len, n_features)
        X_cal_norm = X_cal_scaled.reshape(-1, seq_len, n_features)

        return X_train_norm, X_val_norm, X_cal_norm


class LSTMForecasterModel:
    """LSTM model for stock direction prediction.

    WHAT: Builds and trains LSTM neural network
    WHY: LSTMs excel at learning patterns in sequential data
    HOW: Two LSTM layers (128->64) with dropout, softmax output
    DATA: Sequential features -> probability distribution over classes
    """

    def __init__(self, config: TrainingConfig):
        """Initialize LSTM model builder.

        WHAT: Set up model configuration
        WHY: Need config for architecture parameters
        DATA: Store config reference
        """
        self.config = config
        self.model = None
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM architecture.

        WHAT: Constructs LSTM model with specified architecture
        WHY: Define model structure before training
        HOW: Stack LSTM layers with dropout, add dense output layer
        DATA: Input shape -> compiled Keras model

        Args:
            input_shape: (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        # WHAT: Create sequential model
        # WHY: Stack layers in sequence (LSTM -> LSTM -> Dense)
        # HOW: Use Keras Sequential API
        model = models.Sequential([
            # WHAT: First LSTM layer (128 units)
            # WHY: Learn high-level temporal patterns
            # HOW: return_sequences=True to feed into next LSTM
            # DATA: (batch, seq_len, features) -> (batch, seq_len, 128)
            layers.LSTM(
                self.config.lstm_layer1_units,
                return_sequences=True,
                input_shape=input_shape,
                name='lstm_layer_1'
            ),

            # WHAT: Dropout for regularization
            # WHY: Prevent overfitting by randomly dropping connections
            # HOW: Drop 20% of connections during training
            layers.Dropout(self.config.dropout_rate, name='dropout_1'),

            # WHAT: Second LSTM layer (64 units)
            # WHY: Learn more refined patterns from first layer output
            # HOW: return_sequences=False to output single vector
            # DATA: (batch, seq_len, 128) -> (batch, 64)
            layers.LSTM(
                self.config.lstm_layer2_units,
                return_sequences=False,
                name='lstm_layer_2'
            ),

            # WHAT: Dropout for second layer
            # WHY: Additional regularization before output
            layers.Dropout(self.config.dropout_rate, name='dropout_2'),

            # WHAT: Dense output layer with softmax
            # WHY: Convert LSTM output to class probabilities
            # HOW: 3 units (up/down/steady) with softmax activation
            # DATA: (batch, 64) -> (batch, 3) probabilities summing to 1
            layers.Dense(
                self.config.num_classes,
                activation='softmax',
                name='output_layer'
            )
        ])

        # WHAT: Compile model with optimizer and loss function
        # WHY: Configure training process
        # HOW: Adam optimizer, categorical crossentropy loss, track accuracy
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model on data.

        WHAT: Fits model to training data with validation
        WHY: Learn patterns from historical sequences
        HOW: Mini-batch gradient descent with early stopping
        DATA: Training sequences -> trained model weights

        Args:
            X_train: Training sequences (n_samples, seq_len, features)
            y_train: Training labels (n_samples,)
            X_val: Validation sequences
            y_val: Validation labels

        Returns:
            Dictionary with training history
        """
        # WHAT: Convert labels to one-hot encoding
        # WHY: Categorical crossentropy requires one-hot labels
        # HOW: Use Keras to_categorical utility
        # DATA: [0,1,2] -> [[1,0,0], [0,1,0], [0,0,1]]
        y_train_cat = to_categorical(y_train, num_classes=self.config.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.config.num_classes)

        # WHAT: Set up callbacks for training
        # WHY: Early stopping prevents overfitting, model checkpoint saves best model
        # HOW: Monitor validation loss, stop if no improvement
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        # WHAT: Train model with mini-batch gradient descent
        # WHY: Learn optimal weights to minimize loss
        # HOW: Iterate through batches, update weights, validate each epoch
        # DATA: Sequences + labels -> optimized model weights
        logger.info("Starting model training...")

        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        logger.info("Training completed")

        return self.history.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance.

        WHAT: Computes metrics on test data
        WHY: Assess model quality on unseen data
        HOW: Generate predictions, calculate metrics
        DATA: Test sequences -> performance metrics

        Args:
            X_test: Test sequences
            y_test: True test labels

        Returns:
            Dictionary with evaluation metrics
        """
        # WHAT: Generate predictions
        # WHY: Need predictions to compare against ground truth
        # DATA: Sequences -> probability predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # WHAT: Calculate classification metrics
        # WHY: Understand model performance across classes
        # HOW: Use sklearn metrics
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            y_test, y_pred,
            target_names=['down', 'steady', 'up']
        ))

        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))

        # WHAT: Calculate log loss
        # WHY: Measures quality of probability predictions
        # HOW: Penalizes confident wrong predictions
        y_test_cat = to_categorical(y_test, num_classes=self.config.num_classes)
        logloss = log_loss(y_test_cat, y_pred_proba)

        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'log_loss': logloss
        }


class ProbabilityCalibrator:
    """Calibrates model probabilities to match true frequencies.

    WHAT: Ensures predicted probabilities are well-calibrated
    WHY: Want 70% confidence to mean 70% actual accuracy
    HOW: Uses isotonic regression on calibration set
    DATA: Raw probabilities -> calibrated probabilities
    """

    def __init__(self):
        """Initialize calibrator.

        WHAT: Set up calibration storage
        WHY: Will store calibration curves
        """
        self.calibration_maps = {}

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray,
            class_idx: int) -> None:
        """Fit calibration map for a specific class.

        WHAT: Learn mapping from predicted to calibrated probabilities
        WHY: Model probabilities often poorly calibrated out-of-box
        HOW: Use calibration curve to map predicted -> true frequency
        DATA: True labels + predictions -> calibration mapping

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for class
            class_idx: Which class (0=down, 1=steady, 2=up)
        """
        # WHAT: Create binary labels for this class
        # WHY: Calibration curve needs binary (class vs not-class)
        # DATA: Multi-class labels -> binary labels
        y_binary = (y_true == class_idx).astype(int)

        # WHAT: Calculate calibration curve
        # WHY: Shows how predicted probabilities relate to true frequencies
        # HOW: Bin predictions, calculate actual frequency in each bin
        # DATA: Predictions -> (true_freq, pred_mean) pairs
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_proba, n_bins=10, strategy='quantile'
        )

        # WHAT: Store calibration mapping
        # WHY: Use for transforming future predictions
        # DATA: Arrays of (predicted, actual) points
        self.calibration_maps[class_idx] = {
            'true_freq': fraction_of_positives,
            'pred_mean': mean_predicted_value
        }

        logger.info(f"Calibration for class {class_idx}:")
        logger.info(f"  Predicted: {mean_predicted_value}")
        logger.info(f"  Actual: {fraction_of_positives}")

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities.

        WHAT: Transform predicted probabilities using calibration maps
        WHY: Get better-calibrated confidence estimates
        HOW: Interpolate using calibration curves
        DATA: Raw probabilities -> calibrated probabilities

        Args:
            proba: Raw predicted probabilities (n_samples, n_classes)

        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        calibrated = np.zeros_like(proba)

        for class_idx in range(proba.shape[1]):
            if class_idx not in self.calibration_maps:
                # WHAT: If no calibration available, use raw probabilities
                # WHY: Better than failing completely
                calibrated[:, class_idx] = proba[:, class_idx]
                continue

            # WHAT: Get calibration map for this class
            cal_map = self.calibration_maps[class_idx]

            # WHAT: Interpolate predicted probabilities to calibrated values
            # WHY: Map predicted -> actual frequency
            # HOW: Linear interpolation between calibration points
            calibrated[:, class_idx] = np.interp(
                proba[:, class_idx],
                cal_map['pred_mean'],
                cal_map['true_freq']
            )

        # WHAT: Renormalize to ensure probabilities sum to 1
        # WHY: Calibration may break probability axiom
        # HOW: Divide each by row sum
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums

        return calibrated


def main():
    """Main training pipeline.

    WHAT: Orchestrates entire training process
    WHY: Entry point for model training
    HOW: Load data -> preprocess -> train -> calibrate -> save
    DATA: Raw CSV files -> trained model + artifacts
    """
    # WHAT: Initialize configuration
    # WHY: Centralize all training parameters
    config = TrainingConfig()

    # WHAT: Create save directory if not exists
    # WHY: Need place to store trained model
    # HOW: Use pathlib to create directory
    save_dir = Path(config.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # WHAT: Initialize data processor
    # WHY: Need to load and process training data
    logger.info("=" * 50)
    logger.info("STEP 1: Loading and processing data")
    logger.info("=" * 50)

    processor = LSTMDataProcessor(config)

    # WHAT: Load all stock data
    # DATA: CSV files -> combined DataFrame
    df = processor.load_all_stock_data()

    # WHAT: Calculate technical indicators
    # DATA: OHLCV -> OHLCV + indicators
    df = processor.calculate_technical_indicators(df)

    # WHAT: Calculate ATR if using ATR-based thresholds
    # WHY: Need ATR values before creating labels
    # DATA: OHLC -> ATR (absolute and percentage)
    if config.use_atr_threshold:
        df = processor.calculate_atr(df)
        logger.info("ATR calculated for all stocks")

    # WHAT: Create classification labels
    # DATA: Prices -> class labels (up/down/steady)
    df = processor.create_labels(df)

    # WHAT: Create sequences for LSTM
    # DATA: DataFrame -> (X_sequences, y_labels)
    X, y = processor.create_sequences(df)

    # WHAT: Split into train, validation, and calibration sets
    # WHY: Need separate sets for training, validation, and calibration
    # HOW: First split off validation, then split remaining into train/calibration
    # DATA: Full dataset -> train (70%), val (20%), calibration (10%)
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2: Splitting data")
    logger.info("=" * 50)

    # WHAT: First split: separate validation set
    # WHY: Validation monitors overfitting during training
    X_temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=config.validation_split, random_state=42, stratify=y
    )

    # WHAT: Second split: separate calibration from training
    # WHY: Need held-out set for probability calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=config.calibration_split, random_state=42, stratify=y_temp
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Calibration set: {len(X_cal)} samples")

    # WHAT: Normalize sequences
    # WHY: Neural networks train better with normalized inputs
    # DATA: Raw sequences -> normalized sequences
    X_train_norm, X_val_norm, X_cal_norm = processor.normalize_sequences(
        X_train, X_val, X_cal
    )

    # WHAT: Build and train LSTM model
    # WHY: Learn to predict price direction
    logger.info("\n" + "=" * 50)
    logger.info("STEP 3: Building and training LSTM model")
    logger.info("=" * 50)

    lstm_model = LSTMForecasterModel(config)
    input_shape = (X_train_norm.shape[1], X_train_norm.shape[2])
    lstm_model.build_model(input_shape)

    # WHAT: Train model
    # DATA: Training data -> optimized weights
    history = lstm_model.train(X_train_norm, y_train, X_val_norm, y_val)

    # WHAT: Evaluate on calibration set
    # WHY: Get predictions for calibration
    logger.info("\n" + "=" * 50)
    logger.info("STEP 4: Calibrating probabilities")
    logger.info("=" * 50)

    eval_results = lstm_model.evaluate(X_cal_norm, y_cal)

    # WHAT: Calibrate probabilities
    # WHY: Ensure confidence scores match actual accuracy
    # HOW: Learn mapping from predicted to true frequencies
    calibrator = ProbabilityCalibrator()

    for class_idx in range(config.num_classes):
        calibrator.fit(
            y_cal,
            eval_results['probabilities'][:, class_idx],
            class_idx
        )

    # WHAT: Save model and artifacts
    # WHY: Need to load for inference in ProbabilisticForecaster
    logger.info("\n" + "=" * 50)
    logger.info("STEP 5: Saving model and artifacts")
    logger.info("=" * 50)

    # WHAT: Save Keras model
    # HOW: Use Keras save method
    model_path = save_dir / "lstm_forecaster.h5"
    lstm_model.model.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # WHAT: Save scaler
    # WHY: Need same normalization for inference
    scaler_path = save_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(processor.scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")

    # WHAT: Save calibrator
    # WHY: Need to calibrate predictions during inference
    calibrator_path = save_dir / "calibrator.pkl"
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrator, f)
    logger.info(f"Saved calibrator to {calibrator_path}")

    # WHAT: Save feature columns
    # WHY: Need to know which features and order for inference
    features_path = save_dir / "feature_columns.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(processor.feature_columns, f)
    logger.info(f"Saved feature columns to {features_path}")

    # WHAT: Save config
    # WHY: Need to know sequence length, etc. for inference
    config_path = save_dir / "config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    logger.info(f"Saved config to {config_path}")

    logger.info("\n" + "=" * 50)
    logger.info("Training complete!")
    logger.info("=" * 50)
    logger.info(f"Model artifacts saved to: {save_dir}")
    logger.info(f"\nTo use this model, update ProbabilisticForecaster to load from:")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - Scaler: {scaler_path}")
    logger.info(f"  - Calibrator: {calibrator_path}")


if __name__ == "__main__":
    main()
