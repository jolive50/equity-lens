"""Minimal forecaster adapter bridging PredictionAgent with Pam's LSTM/GRU/XGBoost models.

This module provides a lightweight compatibility layer that allows Josh's PredictionAgent
to work with Pam's minimal model implementations, reusing relevant patterns from legacy code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from models.prediction.lstm_model import LSTMModel
from models.prediction.base_predictor import PredictionResult as PamPredictionResult

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from forecaster (matches Josh's agent expectations).

    Wraps Pam's PredictionResult with additional fields for agent compatibility.
    """
    direction: str  # "up", "down", "neutral"
    confidence: float  # 0.0 to 1.0
    daily_probs: List[Dict[str, Any]]  # Probability curve over time
    horizon_95: Dict[str, Any]  # 95% confidence horizon
    feature_importance: Dict[str, float]  # Feature attribution
    model_metadata: Dict[str, Any]  # Model details


class ProbabilisticForecaster:
    """Minimal forecaster using Pam's LSTM model.

    Provides interface expected by Josh's PredictionAgent while delegating
    actual prediction to Pam's LSTMModel implementation.
    """

    def __init__(self, model_type: str = "lstm", model_dir: Optional[str] = None):
        """Initialize forecaster with Pam's model.

        Args:
            model_type: Model to use ("lstm", "gru", "gradient_boosting")
            model_dir: Path to saved models (default: models/prediction/saved_models)
        """
        self.model_type = model_type

        if model_dir is None:
            from pathlib import Path
            model_dir = str(Path(__file__).parent / "saved_models" / model_type)

        # Load Pam's model
        try:
            if model_type == "lstm":
                self.model = LSTMModel(weights_path=model_dir)
            elif model_type == "gru":
                from models.prediction.gru_model import GRUModel
                self.model = GRUModel(weights_path=model_dir)
            elif model_type == "gradient_boosting":
                from models.prediction.gradient_boost_model import GradientBoostModel
                self.model = GradientBoostModel(weights_path=model_dir)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.is_trained = self.model.is_trained
            logger.info(f"Loaded {model_type} model successfully")
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model: {e}")
            logger.info("Will use trend-based fallback")
            self.model = None
            self.is_trained = False

    def predict(self, market_data: List[Dict], fundamentals: Dict[str, float]) -> ForecastResult:
        """Make prediction using Pam's model.

        Args:
            market_data: List of daily price/volume data dictionaries
            fundamentals: Company financial metrics (not used in minimal version)

        Returns:
            ForecastResult matching Josh's agent expectations
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, using trend-based prediction")
            return self._trend_based_prediction(market_data)

        try:
            # Convert market_data to DataFrame for Pam's model
            df = pd.DataFrame(market_data)

            # Ensure required columns with proper names
            df = df.rename(columns={
                'close': 'Close',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            })

            # Call Pam's model
            pam_result: PamPredictionResult = self.model.predict(df)

            # Generate daily probabilities (reused from legacy pattern)
            daily_probs = self._generate_daily_probabilities(
                pam_result.probabilities['up'],
                pam_result.probabilities['down'],
                pam_result.probabilities['neutral']
            )

            # Calculate 95% confidence horizon
            horizon_95 = self._calculate_95_horizon(daily_probs)

            # Extract feature importance (if available)
            feature_importance = pam_result.metadata.get('feature_importance', {})

            return ForecastResult(
                direction=pam_result.direction,
                confidence=pam_result.confidence,
                daily_probs=daily_probs,
                horizon_95=horizon_95,
                feature_importance=feature_importance,
                model_metadata={
                    "model_type": self.model_type,
                    "probabilities": pam_result.probabilities,
                    "calibrated": True
                }
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._trend_based_prediction(market_data)

    def _generate_daily_probabilities(
        self,
        prob_up: float,
        prob_down: float,
        prob_neutral: float
    ) -> List[Dict[str, Any]]:
        """Generate daily probability forecasts with confidence decay.

        Reused from legacy forecaster.py pattern (lines 1375-1416).
        Applies decay factor to show decreasing confidence over time.
        """
        from datetime import datetime, timedelta

        base_date = datetime.utcnow()
        daily_probs = []

        for i in range(1, 31):  # Next 30 days
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            decay_factor = max(0.1, 1.0 - (i * 0.02))

            daily_probs.append({
                "date": date_str,
                "up": max(0.1, min(0.9, prob_up * decay_factor)),
                "down": max(0.1, min(0.9, prob_down * decay_factor)),
                "neutral": max(0.1, min(0.9, prob_neutral * decay_factor))
            })

        return daily_probs

    def _calculate_95_horizon(self, daily_probs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate 95% confidence horizon.

        Reused from legacy forecaster.py pattern (lines 1418-1466).
        Determines how many days we maintain 95%+ confidence.
        """
        max_confidence_days = 0
        first_drop_day = None

        for i, day_prob in enumerate(daily_probs):
            max_prob = max(day_prob["up"], day_prob["down"], day_prob["neutral"])

            if max_prob >= 0.95:
                max_confidence_days = i + 1
            elif first_drop_day is None and max_prob < 0.95:
                first_drop_day = day_prob["date"]

        if daily_probs:
            first_day = daily_probs[0]
            direction = max(["up", "down", "neutral"], key=lambda d: first_day[d])
        else:
            direction = "neutral"

        return {
            "class": direction,
            "start_date": daily_probs[0]["date"] if daily_probs else None,
            "end_date": daily_probs[max_confidence_days - 1]["date"] if max_confidence_days > 0 else None,
            "days": max_confidence_days,
            "drops_below_95_on": first_drop_day
        }

    def _trend_based_prediction(self, market_data: List[Dict]) -> ForecastResult:
        """Fallback prediction using momentum (reused from legacy pattern).

        From legacy forecaster.py lines 1321-1364.
        Uses recent price momentum when ML model unavailable.
        """
        prices = [float(d.get("close", 0)) for d in market_data if d.get("close") is not None]

        if len(prices) < 10:
            raise RuntimeError(
                "Insufficient market history to produce a forecast. "
                "Need at least 10 days of price data."
            )

        # Calculate momentum
        recent_avg = float(np.mean(prices[-5:]))
        older_window = prices[-10:-5] if len(prices) >= 10 else prices[:-5]
        older_avg = float(np.mean(older_window)) if older_window else recent_avg

        if older_avg == 0:
            raise RuntimeError("Invalid price data (zero prices detected)")

        momentum = float((recent_avg - older_avg) / older_avg)
        momentum = float(np.clip(momentum, -0.2, 0.2))

        # Convert momentum to probabilities using softmax
        scores = np.array([momentum, -momentum, 0.0], dtype=float)
        temperature = 0.05
        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        base_probs = exp_scores / exp_scores.sum()

        prob_up, prob_down, prob_neutral = [float(p) for p in base_probs]
        direction_index = int(np.argmax(base_probs))
        direction = ["up", "down", "neutral"][direction_index]

        daily_probs = self._generate_daily_probabilities(prob_up, prob_down, prob_neutral)
        horizon_95 = self._calculate_95_horizon(daily_probs)

        return ForecastResult(
            direction=direction,
            confidence=max(prob_up, prob_down, prob_neutral),
            daily_probs=daily_probs,
            horizon_95=horizon_95,
            feature_importance={"momentum_strength": abs(momentum)},
            model_metadata={
                "model_type": "momentum_fallback",
                "probabilities": {
                    "up": prob_up,
                    "down": prob_down,
                    "neutral": prob_neutral
                },
                "fallback": True
            }
        )


def create_forecaster(model_type: str = "lstm") -> ProbabilisticForecaster:
    """Factory function to create a forecaster.

    Matches legacy interface expected by Josh's PredictionAgent.

    Args:
        model_type: "lstm", "gru", or "gradient_boosting"

    Returns:
        ProbabilisticForecaster instance using Pam's models
    """
    return ProbabilisticForecaster(model_type)
