"""Prediction Agent using PAM's ML models.

JOSH's Component - Prediction Agent
Uses PAM's prediction models (LSTM, GRU, GB, or Ensemble) to forecast price direction.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionAgentResult:
    """Result from PredictionAgent."""
    direction: str  # "up", "down", "neutral"
    confidence: float  # 0.0-1.0
    narrative: str  # English explanation
    probabilities: Dict[str, float]  # {up, down, neutral}
    metadata: Dict[str, Any]  # Model info


class PredictionAgent:
    """Agent that makes price predictions using PAM's models."""

    def __init__(self, model=None):
        """Initialize PredictionAgent.

        Args:
            model: Optional BasePredictionModel instance (LSTM, GRU, GB, or Ensemble)
                   If None, will attempt to load default model
        """
        self.model = model

        if self.model is None:
            self._load_default_model()

    def _load_default_model(self):
        """Load default prediction model (LSTM)."""
        try:
            from models.prediction.lstm_model import LSTMModel
            self.model = LSTMModel()
            logger.info("Loaded default LSTM model")
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
            self.model = None

    def run(
        self,
        ticker: str,
        market_data: List[Dict[str, Any]],
        fundamentals: Dict[str, float]
    ) -> PredictionAgentResult:
        """Make prediction for a stock.

        Args:
            ticker: Stock symbol
            market_data: List of dicts with OHLCV data
            fundamentals: Dict with financial metrics

        Returns:
            PredictionAgentResult with direction, confidence, narrative
        """
        if self.model is None:
            raise RuntimeError("No prediction model loaded")

        # Convert market_data to DataFrame
        df = pd.DataFrame(market_data)

        # Ensure required columns exist
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Market data missing required columns: {required_cols}")

        # Make prediction
        try:
            result = self.model.predict(df)

            # Generate narrative
            narrative = self._generate_narrative(ticker, result, fundamentals)

            return PredictionAgentResult(
                direction=result.direction,
                confidence=result.confidence,
                narrative=narrative,
                probabilities=result.probabilities,
                metadata=result.metadata
            )

        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def _generate_narrative(
        self,
        ticker: str,
        result,
        fundamentals: Dict[str, float]
    ) -> str:
        """Generate English explanation of prediction."""
        model_name = result.metadata.get("model", "Unknown")
        prob_up = result.probabilities["up"]
        prob_down = result.probabilities["down"]

        narrative = (
            f"{ticker} prediction: {result.direction.upper()} with "
            f"{result.confidence:.1%} confidence using {model_name} model. "
            f"Probabilities: up={prob_up:.1%}, down={prob_down:.1%}. "
        )

        # Add fundamental context if available
        pe_ratio = fundamentals.get("pe_ratio", 0)
        if pe_ratio > 0:
            narrative += f"P/E ratio: {pe_ratio:.1f}. "

        return narrative
