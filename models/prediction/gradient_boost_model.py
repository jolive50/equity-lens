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
        """Extract features for Gradient Boost (Research-Enhanced).

        Uses comprehensive feature set matching training pipeline to maximize
        XGBoost performance (research shows 60-65% accuracy with proper features).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Numpy array shaped (1, n_features)
        """
        if len(data) < 252:  # Need enough for annual returns
            raise ValueError("Need at least 252 days of data for enhanced features")

        df = data.copy()

        # Lagged returns
        returns_1d = df['close'].pct_change(1).iloc[-1]
        returns_5d = df['close'].pct_change(5).iloc[-1]
        returns_10d = df['close'].pct_change(10).iloc[-1]
        returns_20d = df['close'].pct_change(20).iloc[-1]
        returns_60d = df['close'].pct_change(60).iloc[-1]
        returns_252d = df['close'].pct_change(252).iloc[-1] if len(df) >= 252 else 0.0

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = (ema_12 - ema_26).iloc[-1]
        macd_signal = (ema_12 - ema_26).ewm(span=9, adjust=False).mean().iloc[-1]
        macd_diff = macd - macd_signal

        # Bollinger Bands
        bb_ma = df['close'].rolling(20).mean().iloc[-1]
        bb_std = df['close'].rolling(20).std().iloc[-1]
        bb_upper = bb_ma + (2 * bb_std)
        bb_lower = bb_ma - (2 * bb_std)
        bb_width = (bb_upper - bb_lower) / (bb_ma + 1e-10)
        bb_position = (df['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Moving averages
        sma_5 = df['close'].rolling(5).mean().iloc[-1]
        sma_10 = df['close'].rolling(10).mean().iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        sma_200 = df['close'].rolling(200).mean().iloc[-1]
        sma_5_20_cross = (sma_5 / (sma_20 + 1e-10)) - 1
        sma_50_200_cross = (sma_50 / (sma_200 + 1e-10)) - 1

        # Volatility
        volatility_10d = df['close'].pct_change(1).rolling(10).std().iloc[-1]
        volatility_20d = df['close'].pct_change(1).rolling(20).std().iloc[-1]
        volatility_60d = df['close'].pct_change(1).rolling(60).std().iloc[-1]

        # Volume
        volume_ma_10 = df['volume'].rolling(10).mean().iloc[-1]
        volume_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / (volume_ma_10 + 1e-10)
        volume_trend = volume_ma_10 / (volume_ma_20 + 1e-10)

        # Momentum
        momentum_10d = df['close'].iloc[-1] / df['close'].iloc[-11] - 1 if len(df) > 10 else 0.0
        momentum_20d = df['close'].iloc[-1] / df['close'].iloc[-21] - 1 if len(df) > 20 else 0.0

        # Spread
        hl_spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        hl_spread_ma = ((df['high'] - df['low']) / df['close']).rolling(10).mean().iloc[-1]

        # Assemble features in same order as training
        features = [
            returns_1d, returns_5d, returns_10d, returns_20d, returns_60d, returns_252d,
            rsi_14, macd, macd_signal, macd_diff,
            bb_width, bb_position,
            sma_5, sma_10, sma_20, sma_50, sma_200,
            sma_5_20_cross, sma_50_200_cross,
            volatility_10d, volatility_20d, volatility_60d,
            volume_ratio, volume_trend,
            momentum_10d, momentum_20d,
            hl_spread, hl_spread_ma
        ]

        # Replace NaN with 0
        features = [0.0 if pd.isna(f) else f for f in features]

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
