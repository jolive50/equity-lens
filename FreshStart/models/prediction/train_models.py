"""Training script for all prediction models (LSTM, GRU, XGBoost)."""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train all prediction models on Kaggle SP500 dataset."""

    def __init__(
        self,
        data_dir: str = "data/raw/kaggle_sp500",
        save_dir: str = "models/prediction/saved_models"
    ):
        """Initialize model trainer.

        Args:
            data_dir: Directory containing Kaggle SP500 CSV files
            save_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.sequence_length = 30
        self.feature_columns = [
            'returns', 'rsi', 'sma_10', 'sma_20', 'macd', 'macd_signal',
            'volume_ratio', 'volatility'
        ]

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load Kaggle SP500 data and prepare for training.

        Returns:
            Combined DataFrame with all stocks

        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Download Kaggle SP500 data first"
            )

        # Load all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        logger.info(f"Loading {len(csv_files)} stock files...")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]

                # Ensure required columns exist
                required = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required):
                    logger.warning(f"Skipping {csv_file}: missing columns")
                    continue

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df['ticker'] = csv_file.stem

                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")

        if not dfs:
            raise ValueError("No valid CSV files loaded")

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total rows from {len(dfs)} stocks")

        return combined_df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # Returns
        df['returns'] = df.groupby('ticker')['close'].pct_change()

        # Simple Moving Averages
        df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
        df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())

        # RSI
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            rs = avg_gains / avg_losses
            return 100 - (100 / (1 + rs))

        df['rsi'] = df.groupby('ticker')['close'].transform(calc_rsi)

        # MACD
        def calc_macd(prices):
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            return ema_12 - ema_26

        df['macd'] = df.groupby('ticker')['close'].transform(calc_macd)
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9).mean())

        # Volume ratio
        df['volume_ratio'] = df.groupby('ticker')['volume'].transform(
            lambda x: x / x.rolling(10).mean()
        )

        # Volatility
        df['volatility'] = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(10).std()
        )

        return df

    def create_labels(self, df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """Create target labels (up/down/neutral).

        Args:
            df: DataFrame with price data
            threshold: Price change threshold for up/down classification

        Returns:
            DataFrame with 'label' column
        """
        df = df.copy()

        # Calculate next day return
        df['next_return'] = df.groupby('ticker')['returns'].shift(-1)

        # Create labels: 0=down, 1=neutral, 2=up
        df['label'] = 1  # default neutral
        df.loc[df['next_return'] > threshold, 'label'] = 2  # up
        df.loc[df['next_return'] < -threshold, 'label'] = 0  # down

        return df

    def prepare_sequences(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/GRU training.

        Args:
            df: DataFrame with features and labels

        Returns:
            Tuple of (X_sequences, y_labels)
        """
        df = df[self.feature_columns + ['label']].dropna()

        sequences = []
        labels = []

        # Create sequences for each stock
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]

            if len(ticker_data) < self.sequence_length + 1:
                continue

            for i in range(len(ticker_data) - self.sequence_length):
                seq = ticker_data.iloc[i:i+self.sequence_length][self.feature_columns].values
                label = ticker_data.iloc[i+self.sequence_length]['label']

                sequences.append(seq)
                labels.append(label)

        X = np.array(sequences)
        y = np.array(labels)

        logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")
        return X, y

    def train_lstm(self, X_train, y_train, X_val, y_val) -> None:
        """Train LSTM model.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
        """
        logger.info("Training LSTM model...")

        # Normalize
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)

        X_train_norm = scaler.fit_transform(X_train_2d)
        X_val_norm = scaler.transform(X_val_2d)

        X_train_norm = X_train_norm.reshape(X_train.shape)
        X_val_norm = X_val_norm.reshape(X_val.shape)

        # Build LSTM
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=50,
            batch_size=64,
            verbose=1
        )

        # Save
        lstm_dir = self.save_dir / "lstm"
        lstm_dir.mkdir(parents=True, exist_ok=True)

        model.save(lstm_dir / "lstm_model.h5")
        with open(lstm_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        logger.info(f"LSTM model saved to {lstm_dir}")

    def train_gru(self, X_train, y_train, X_val, y_val) -> None:
        """Train GRU model.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
        """
        logger.info("Training GRU model...")

        # Normalize
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)

        X_train_norm = scaler.fit_transform(X_train_2d)
        X_val_norm = scaler.transform(X_val_2d)

        X_train_norm = X_train_norm.reshape(X_train.shape)
        X_val_norm = X_val_norm.reshape(X_val.shape)

        # Build GRU
        model = keras.Sequential([
            keras.layers.GRU(128, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            keras.layers.Dropout(0.2),
            keras.layers.GRU(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=50,
            batch_size=64,
            verbose=1
        )

        # Save
        gru_dir = self.save_dir / "gru"
        gru_dir.mkdir(parents=True, exist_ok=True)

        model.save(gru_dir / "gru_model.h5")
        with open(gru_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        logger.info(f"GRU model saved to {gru_dir}")

    def train_xgboost(self, df: pd.DataFrame) -> None:
        """Train XGBoost model.

        Args:
            df: DataFrame with features and labels
        """
        logger.info("Training XGBoost model...")

        # Prepare features (using most recent data point, not sequences)
        feature_cols = self.feature_columns + ['bb_upper', 'bb_lower']
        df_clean = df[feature_cols + ['label']].dropna()

        X = df_clean[feature_cols].values
        y = df_clean['label'].values

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        val_acc = model.score(X_val, y_val)
        logger.info(f"XGBoost validation accuracy: {val_acc:.4f}")

        # Save
        xgb_dir = self.save_dir / "gradient_boost"
        xgb_dir.mkdir(parents=True, exist_ok=True)

        with open(xgb_dir / "xgboost_model.pkl", 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"XGBoost model saved to {xgb_dir}")

    def train_all(self) -> None:
        """Train all models."""
        logger.info("Starting model training pipeline...")

        # Load data
        df = self.load_and_prepare_data()

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Create labels
        df = self.create_labels(df)

        # Train sequence models (LSTM, GRU)
        X, y = self.prepare_sequences(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.train_lstm(X_train, y_train, X_val, y_val)
        self.train_gru(X_train, y_train, X_val, y_val)

        # Train XGBoost (uses single data points, not sequences)
        self.train_xgboost(df)

        logger.info("All models trained successfully!")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()
