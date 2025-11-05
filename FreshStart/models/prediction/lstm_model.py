"""LSTM model for stock price prediction.

PAM's Component - LSTM Neural Network
Implements Long Short-Term Memory network for time series prediction.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from .base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class LSTMModel(BasePredictionModel):
    """LSTM neural network for stock price direction prediction."""

    def __init__(self, weights_path: Optional[str] = None):
        """Initialize LSTM model.

        Args:
            weights_path: Path to saved model weights (optional)
        """
        self.weights_path = weights_path
        self.model = None
        self.is_trained = False

        if weights_path and Path(weights_path).exists():
            self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        """Load pre-trained model weights."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(weights_path)
            self.is_trained = True
            logger.info(f"Loaded LSTM model from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
            self.is_trained = False

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction using LSTM model.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            PredictionResult with direction and confidence
        """
        if not self.is_trained or self.model is None:
            # Fallback to momentum-based prediction
            return self._momentum_prediction(data)

        try:
            # Extract features and make prediction
            features = self._prepare_features(data)
            probabilities = self.model.predict(features, verbose=0)

            prob_down, prob_neutral, prob_up = probabilities[0]

            # Determine direction
            if prob_up > prob_down and prob_up > prob_neutral:
                direction = "up"
                confidence = float(prob_up)
            elif prob_down > prob_up and prob_down > prob_neutral:
                direction = "down"
                confidence = float(prob_down)
            else:
                direction = "neutral"
                confidence = float(prob_neutral)

            return PredictionResult(
                direction=direction,
                confidence=confidence,
                probabilities={
                    "up": float(prob_up),
                    "down": float(prob_down),
                    "neutral": float(prob_neutral)
                },
                metadata={"model": "LSTM", "trained": self.is_trained}
            )

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return self._momentum_prediction(data)

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for LSTM input."""
        # Simplified feature extraction
        if len(data) < 30:
            raise ValueError("Need at least 30 days of data")

        # Calculate returns and normalize
        returns = data['close'].pct_change().fillna(0).values[-30:]
        volume_norm = (data['volume'] / data['volume'].rolling(10).mean()).fillna(1).values[-30:]

        # Combine features
        features = np.column_stack([returns, volume_norm])
        return features.reshape(1, 30, 2)

    def _momentum_prediction(self, data: pd.DataFrame) -> PredictionResult:
        """Simple momentum-based fallback prediction."""
        if len(data) < 10:
            return PredictionResult(
                direction="neutral",
                confidence=0.5,
                probabilities={"up": 0.33, "down": 0.33, "neutral": 0.34},
                metadata={"model": "LSTM_fallback", "method": "momentum"}
            )

        recent_return = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]

        if recent_return > 0.02:
            return PredictionResult(
                direction="up",
                confidence=0.6,
                probabilities={"up": 0.6, "down": 0.2, "neutral": 0.2},
                metadata={"model": "LSTM_fallback", "momentum": recent_return}
            )
        elif recent_return < -0.02:
            return PredictionResult(
                direction="down",
                confidence=0.6,
                probabilities={"up": 0.2, "down": 0.6, "neutral": 0.2},
                metadata={"model": "LSTM_fallback", "momentum": recent_return}
            )
        else:
            return PredictionResult(
                direction="neutral",
                confidence=0.5,
                probabilities={"up": 0.33, "down": 0.33, "neutral": 0.34},
                metadata={"model": "LSTM_fallback", "momentum": recent_return}
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "LSTM",
            "type": "neural_network",
            "architecture": "Long Short-Term Memory",
            "trained": self.is_trained,
            "weights_path": self.weights_path
        }
