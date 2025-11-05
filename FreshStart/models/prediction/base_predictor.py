"""Base class for all prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


@dataclass
class PredictionResult:
    """Standardized prediction result from any prediction model.

    Attributes:
        direction: Predicted direction ('up', 'down', 'neutral')
        confidence: Confidence score (0.0 to 1.0)
        probabilities: Class probabilities {'up': float, 'down': float, 'neutral': float}
        metadata: Model-specific metadata
    """
    direction: str
    confidence: float
    probabilities: Dict[str, float]
    metadata: Dict[str, Any]


class BasePredictionModel(ABC):
    """Abstract base class for all prediction models.

    Ensures consistent interface across LSTM, GRU, Gradient Boost, and Ensemble.
    All prediction models must inherit from this class.
    """

    def __init__(self, model_name: str):
        """Initialize base prediction model.

        Args:
            model_name: Name of the model (e.g., 'LSTM', 'GRU', 'GradientBoost')
        """
        self.model_name = model_name
        self.is_trained = False

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction on market data.

        Args:
            data: DataFrame with OHLCV data and technical indicators

        Returns:
            PredictionResult with direction, confidence, probabilities

        Raises:
            RuntimeError: If model not trained or data insufficient
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from disk.

        Args:
            model_path: Path to saved model file

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If loading fails
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata.

        Returns:
            Dict with model name, version, architecture details
        """
        return {
            'name': self.model_name,
            'trained': self.is_trained
        }

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # Returns
        df['returns'] = df['Close'].pct_change()

        # Simple Moving Averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()

        # RSI
        delta = df['Close'].diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume ratio
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(10).std()

        return df
