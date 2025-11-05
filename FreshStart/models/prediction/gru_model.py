"""GRU model for stock price direction prediction."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from models.prediction.base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class GRUModel(BasePredictionModel):
    """GRU (Gated Recurrent Unit) neural network for stock prediction.

    Alternative to LSTM with potentially faster training.
    Uses sequence of historical price data and technical indicators.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        sequence_length: int = 30,
        weight: float = 1.0
    ):
        """Initialize GRU model.

        Args:
            weights_path: Path to saved model directory
            sequence_length: Number of days of history to use
            weight: Weight for ensemble (default 1.0)

        Raises:
            RuntimeError: If weights_path provided but model files not found
        """
        super().__init__(model_name="GRU")
        self.sequence_length = sequence_length
        self.weight = weight
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'returns', 'rsi', 'sma_10', 'sma_20', 'macd', 'macd_signal',
            'volume_ratio', 'volatility'
        ]

        if weights_path:
            self.load_model(weights_path)

    def load_model(self, model_path: str) -> None:
        """Load pre-trained GRU model.

        Args:
            model_path: Path to saved model directory

        Raises:
            RuntimeError: If model files not found or loading fails
        """
        model_dir = Path(model_path)
        model_file = model_dir / "gru_model.h5"
        scaler_file = model_dir / "scaler.pkl"

        if not model_file.exists():
            raise RuntimeError(
                f"GRU model not found at {model_file}\n"
                f"Train the model first using train_models.py"
            )

        if not scaler_file.exists():
            raise RuntimeError(f"Scaler not found at {scaler_file}")

        logger.info(f"Loading GRU model from {model_file}")
        self.model = keras.models.load_model(model_file)

        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)

        self.is_trained = True
        logger.info("GRU model loaded successfully")

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Predict stock direction using GRU.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            PredictionResult with direction, confidence, probabilities

        Raises:
            RuntimeError: If model not loaded
            ValueError: If insufficient data
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("GRU model not loaded. Call load_model() first.")

        if len(data) < self.sequence_length:
            raise ValueError(
                f"Insufficient data: need {self.sequence_length} rows, got {len(data)}"
            )

        # Calculate technical indicators
        df = self._calculate_technical_indicators(data)

        # Prepare sequence
        df_clean = df[self.feature_columns].dropna()

        if len(df_clean) < self.sequence_length:
            raise ValueError(
                f"Insufficient clean data after indicators: "
                f"need {self.sequence_length}, got {len(df_clean)}"
            )

        # Extract most recent sequence
        sequence = df_clean.iloc[-self.sequence_length:].values

        # Reshape for GRU: (1, sequence_length, n_features)
        sequence = sequence.reshape(1, self.sequence_length, len(self.feature_columns))

        # Normalize
        sequence_2d = sequence.reshape(1, -1)
        sequence_norm = self.scaler.transform(sequence_2d)
        sequence_norm = sequence_norm.reshape(
            1, self.sequence_length, len(self.feature_columns)
        )

        # Predict
        proba = self.model.predict(sequence_norm, verbose=0)

        # Extract probabilities (assuming order: down, neutral, up)
        prob_down = float(proba[0][0])
        prob_neutral = float(proba[0][1])
        prob_up = float(proba[0][2])

        # Determine direction
        if prob_up > prob_down and prob_up > prob_neutral:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_neutral

        return PredictionResult(
            direction=direction,
            confidence=confidence,
            probabilities={
                'up': prob_up,
                'down': prob_down,
                'neutral': prob_neutral
            },
            metadata={
                'model': 'GRU',
                'sequence_length': self.sequence_length,
                'weight': self.weight
            }
        )

    def get_model_info(self) -> dict:
        """Return GRU model metadata."""
        info = super().get_model_info()
        info.update({
            'architecture': 'GRU',
            'sequence_length': self.sequence_length,
            'features': self.feature_columns,
            'weight': self.weight
        })
        return info
