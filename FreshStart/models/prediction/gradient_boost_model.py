"""Gradient Boosting model for stock price direction prediction."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from models.prediction.base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class GradientBoostModel(BasePredictionModel):
    """XGBoost gradient boosting classifier for stock prediction.

    Non-neural network approach using gradient boosted decision trees.
    Often faster than LSTM/GRU with competitive accuracy.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        weight: float = 1.0
    ):
        """Initialize Gradient Boost model.

        Args:
            weights_path: Path to saved model directory
            weight: Weight for ensemble (default 1.0)

        Raises:
            RuntimeError: If weights_path provided but model files not found
        """
        super().__init__(model_name="GradientBoost")
        self.weight = weight
        self.model = None
        self.feature_columns = [
            'returns', 'rsi', 'sma_10', 'sma_20', 'macd', 'macd_signal',
            'volume_ratio', 'volatility', 'bb_upper', 'bb_lower'
        ]

        if weights_path:
            self.load_model(weights_path)

    def load_model(self, model_path: str) -> None:
        """Load pre-trained XGBoost model.

        Args:
            model_path: Path to saved model directory

        Raises:
            RuntimeError: If model file not found or loading fails
        """
        model_dir = Path(model_path)
        model_file = model_dir / "xgboost_model.pkl"

        if not model_file.exists():
            raise RuntimeError(
                f"XGBoost model not found at {model_file}\n"
                f"Train the model first using train_models.py"
            )

        logger.info(f"Loading XGBoost model from {model_file}")

        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info("XGBoost model loaded successfully")

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Predict stock direction using XGBoost.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            PredictionResult with direction, confidence, probabilities

        Raises:
            RuntimeError: If model not loaded
            ValueError: If insufficient data
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("XGBoost model not loaded. Call load_model() first.")

        if len(data) < 30:
            raise ValueError(f"Insufficient data: need at least 30 rows, got {len(data)}")

        # Calculate technical indicators
        df = self._calculate_technical_indicators(data)

        # Prepare features
        df_clean = df[self.feature_columns].dropna()

        if df_clean.empty:
            raise ValueError("No clean data after calculating indicators")

        # Use most recent row
        features = df_clean.iloc[-1:].values

        # Predict probabilities
        proba = self.model.predict_proba(features)

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
                'model': 'XGBoost',
                'weight': self.weight,
                'feature_importance': self._get_feature_importance()
            }
        )

    def _get_feature_importance(self) -> dict:
        """Get feature importance from XGBoost model."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance.tolist()))

    def get_model_info(self) -> dict:
        """Return XGBoost model metadata."""
        info = super().get_model_info()
        info.update({
            'architecture': 'XGBoost',
            'features': self.feature_columns,
            'weight': self.weight
        })
        return info
