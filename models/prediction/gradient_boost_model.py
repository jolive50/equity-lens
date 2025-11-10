"""Gradient Boosting model for stock price prediction.

PAM's Component - Gradient Boosting Classifier
Non-neural network approach using XGBoost for comparison.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from .base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class GradientBoostModel(BasePredictionModel):
    """Gradient Boosting model for stock price direction prediction."""

    def __init__(self, weights_path: Optional[str] = None):
        """Initialize Gradient Boost model.

        Args:
            weights_path: Path to saved model weights (optional)
        """
        self.weights_path = weights_path
        self.model = None
        self.is_trained = False

        if weights_path and Path(weights_path).exists():
            self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        """Load pre-trained model."""
        try:
            import pickle
            with open(weights_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Loaded Gradient Boost model from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load Gradient Boost model: {e}")
            self.is_trained = False

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction using Gradient Boost model.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            PredictionResult with direction and confidence
        """
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(data)

        try:
            features = self._extract_features(data)
            probabilities = self.model.predict_proba(features)[0]

            prob_down, prob_neutral, prob_up = probabilities

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
                metadata={"model": "GradientBoost", "trained": self.is_trained}
            )

        except Exception as e:
            logger.error(f"Gradient Boost prediction failed: {e}")
            return self._fallback_prediction(data)

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for Gradient Boost."""
        if len(data) < 20:
            raise ValueError("Need at least 20 days of data")

        features = []

        # Price returns
        features.append(data['close'].pct_change(1).iloc[-1])
        features.append(data['close'].pct_change(5).iloc[-1])
        features.append(data['close'].pct_change(10).iloc[-1])

        # Volume
        features.append(data['volume'].iloc[-1] / data['volume'].rolling(10).mean().iloc[-1])

        # Simple moving averages
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        features.append(1.0 if sma_5 > sma_20 else 0.0)

        return np.array([features])

    def _fallback_prediction(self, data: pd.DataFrame) -> PredictionResult:
        """Fallback trend-based prediction."""
        if len(data) < 10:
            return PredictionResult("neutral", 0.5, {"up": 0.33, "down": 0.33, "neutral": 0.34}, {"model": "GB_fallback"})

        trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]

        if trend > 0.03:
            return PredictionResult("up", 0.6, {"up": 0.6, "down": 0.2, "neutral": 0.2}, {"model": "GB_fallback"})
        elif trend < -0.03:
            return PredictionResult("down", 0.6, {"up": 0.2, "down": 0.6, "neutral": 0.2}, {"model": "GB_fallback"})
        else:
            return PredictionResult("neutral", 0.5, {"up": 0.33, "down": 0.33, "neutral": 0.34}, {"model": "GB_fallback"})

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "GradientBoost",
            "type": "gradient_boosting",
            "architecture": "XGBoost Classifier",
            "trained": self.is_trained,
            "weights_path": self.weights_path
        }
