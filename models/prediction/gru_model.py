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
        """Prepare features for GRU input (Research-Enhanced).

        Uses the same comprehensive feature set as LSTM training pipeline.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Numpy array shaped (1, 60, n_features)
        """
        sequence_length = 60  # Research-recommended

        if len(data) < sequence_length + 200:
            raise ValueError(f"Need at least {sequence_length + 200} days of data for enhanced features")

        # Calculate all features (matching preprocessing pipeline)
        df = data.copy()

        # Lagged returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)
        df['returns_60d'] = df['close'].pct_change(60)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_ma + (2 * bb_std)
        bb_lower = bb_ma - (2 * bb_std)
        df['bb_width'] = (bb_upper - bb_lower) / (bb_ma + 1e-10)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_5_20_cross'] = (df['sma_5'] / (df['sma_20'] + 1e-10)) - 1
        df['sma_50_200_cross'] = (df['sma_50'] / (df['close'].rolling(200).mean() + 1e-10)) - 1

        # Volatility
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()

        # Volume
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_10'] + 1e-10)
        df['volume_trend'] = df['volume_ma_10'] / (df['volume_ma_20'] + 1e-10)

        # Momentum
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1

        # Spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_ma'] = df['hl_spread'].rolling(10).mean()

        # Select features matching training (order must match!)
        feature_columns = [
            'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d',
            'rsi_14', 'macd', 'macd_diff',
            'bb_width', 'bb_position',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'sma_5_20_cross', 'sma_50_200_cross',
            'volatility_10d', 'volatility_20d', 'volatility_60d',
            'volume_ratio', 'volume_trend',
            'momentum_10d', 'momentum_20d',
            'hl_spread', 'hl_spread_ma'
        ]

        # Extract last sequence_length timesteps
        df_clean = df[feature_columns].fillna(0)
        sequence = df_clean.iloc[-sequence_length:].values

        return sequence.reshape(1, sequence_length, len(feature_columns))

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
