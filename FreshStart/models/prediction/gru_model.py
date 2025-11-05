"""GRU model for stock price prediction.

PAM's Component - GRU Neural Network
Implements Gated Recurrent Unit network, simpler alternative to LSTM.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from .base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class GRUModel(BasePredictionModel):
    """GRU neural network for stock price direction prediction."""

    def __init__(self, weights_path: Optional[str] = None):
        """Initialize GRU model.

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
            logger.info(f"Loaded GRU model from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load GRU model: {e}")
            self.is_trained = False

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction using GRU model.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            PredictionResult with direction and confidence
        """
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(data)

        try:
            features = self._prepare_features(data)
            probabilities = self.model.predict(features, verbose=0)

            prob_down, prob_neutral, prob_up = probabilities[0]

            if prob_up > max(prob_down, prob_neutral):
                direction, confidence = "up", float(prob_up)
            elif prob_down > prob_neutral:
                direction, confidence = "down", float(prob_down)
            else:
                direction, confidence = "neutral", float(prob_neutral)

            return PredictionResult(
                direction=direction,
                confidence=confidence,
                probabilities={
                    "up": float(prob_up),
                    "down": float(prob_down),
                    "neutral": float(prob_neutral)
                },
                metadata={"model": "GRU", "trained": self.is_trained}
            )

        except Exception as e:
            logger.error(f"GRU prediction failed: {e}")
            return self._fallback_prediction(data)

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for GRU input."""
        if len(data) < 30:
            raise ValueError("Need at least 30 days of data")

        returns = data['close'].pct_change().fillna(0).values[-30:]
        volume_norm = (data['volume'] / data['volume'].rolling(10).mean()).fillna(1).values[-30:]
        features = np.column_stack([returns, volume_norm])
        return features.reshape(1, 30, 2)

    def _fallback_prediction(self, data: pd.DataFrame) -> PredictionResult:
        """Fallback momentum-based prediction."""
        if len(data) < 5:
            return PredictionResult(
                direction="neutral",
                confidence=0.5,
                probabilities={"up": 0.33, "down": 0.33, "neutral": 0.34},
                metadata={"model": "GRU_fallback"}
            )

        momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]

        if momentum > 0.01:
            return PredictionResult("up", 0.55, {"up": 0.55, "down": 0.25, "neutral": 0.2}, {"model": "GRU_fallback"})
        elif momentum < -0.01:
            return PredictionResult("down", 0.55, {"up": 0.25, "down": 0.55, "neutral": 0.2}, {"model": "GRU_fallback"})
        else:
            return PredictionResult("neutral", 0.5, {"up": 0.33, "down": 0.33, "neutral": 0.34}, {"model": "GRU_fallback"})

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "GRU",
            "type": "neural_network",
            "architecture": "Gated Recurrent Unit",
            "trained": self.is_trained,
            "weights_path": self.weights_path
        }
