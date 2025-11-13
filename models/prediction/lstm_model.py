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
        import time
        predict_start = time.time()

        logger.debug(f"            → LSTM predicting (data shape: {data.shape})")

        if not self.is_trained or self.model is None:
            logger.debug(f"            ⚠️  LSTM not trained, using momentum fallback")
            # Fallback to momentum-based prediction
            return self._momentum_prediction(data)

        try:
            # Extract features and make prediction
            feature_start = time.time()
            features = self._prepare_features(data)
            feature_time = time.time() - feature_start
            logger.debug(f"            ✓ Features prepared: shape={features.shape} ({feature_time*1000:.1f}ms)")

            inference_start = time.time()
            probabilities = self.model.predict(features, verbose=0)
            inference_time = time.time() - inference_start

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

            total_time = time.time() - predict_start
            logger.debug(
                f"            ✓ LSTM result: {direction} ({confidence:.2f}) "
                f"[features: {feature_time*1000:.1f}ms, inference: {inference_time*1000:.1f}ms, total: {total_time*1000:.1f}ms]"
            )

            return PredictionResult(
                direction=direction,
                confidence=confidence,
                probabilities={
                    "up": float(prob_up),
                    "down": float(prob_down),
                    "neutral": float(prob_neutral)
                },
                metadata={
                    "model": "LSTM",
                    "trained": self.is_trained,
                    "feature_time_ms": feature_time * 1000,
                    "inference_time_ms": inference_time * 1000,
                    "input_shape": str(features.shape)
                }
            )

        except Exception as e:
            logger.error(f"            ✗ LSTM prediction failed: {e}")
            return self._momentum_prediction(data)

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for LSTM input (Research-Enhanced).

        Uses comprehensive feature set matching training pipeline:
        - Extended lagged returns
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Multiple moving averages
        - Volatility measures
        - Volume features

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Numpy array shaped (1, 60, n_features)
        """
        sequence_length = 60  # Research-recommended

        if len(data) < sequence_length + 200:  # Need extra for 200-day SMA
            raise ValueError(f"Need at least {sequence_length + 200} days of data for enhanced features")

        # Calculate all features (matching preprocessing pipeline)
        df = data.copy()

        # Lagged returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)
        df['returns_60d'] = df['close'].pct_change(60)

        # Technical indicators
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
