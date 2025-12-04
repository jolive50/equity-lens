"""Prediction Agent using PAM's ML models.

JOSH's Component - Prediction Agent
Uses PAM's prediction models (LSTM, GRU, GB, or Ensemble) to forecast price direction.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.prediction_tools import predict_with_model

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

    def __init__(self, model: Optional[Any] = None):
        """Initialize PredictionAgent.

        Args:
            model: Optional BasePredictionModel instance (LSTM, GRU, GB, or Ensemble)
                   Models should be constructed by the caller (decoupled from agent)
        """
        self.model = model

        if self.model is not None:
            logger.info("PredictionAgent initialized with %s", self.model.__class__.__name__)
        else:
            logger.warning("PredictionAgent initialized without a backing model (model must be provided externally)")

    def run(
        self,
        ticker: str,
        market_data: List[Dict[str, Any]],
        fundamentals: Dict[str, float],
    ) -> PredictionAgentResult:
        """Make prediction for a stock."""
        import time

        agent_start = time.time()

        if self.model is None:
            raise RuntimeError("No prediction model loaded")

        logger.info("      PredictionAgent Execution")
        logger.info("         Ticker: %s", ticker)
        logger.info("         Model: %s", self.model.__class__.__name__)
        logger.info("         Market data rows: %d", len(market_data))
        logger.info("         Fundamentals: %d metrics", len(fundamentals))

        try:
            result, predict_time, narrative = predict_with_model(
                self.model, ticker, market_data, fundamentals
            )

            prediction = PredictionAgentResult(
                direction=result.direction,
                confidence=result.confidence,
                narrative=narrative,
                probabilities=result.probabilities,
                metadata=result.metadata,
            )

            total_time = time.time() - agent_start

            logger.info("      Prediction complete")
            logger.info("         Direction: %s", prediction.direction.upper())
            logger.info("         Confidence: %.1f%%", prediction.confidence * 100)
            logger.info(
                "         Probabilities: up=%.1f%% down=%.1f%% neutral=%.1f%%",
                result.probabilities.get("up", 0) * 100,
                result.probabilities.get("down", 0) * 100,
                result.probabilities.get("neutral", 0) * 100,
            )
            logger.info("         Metadata: %s", result.metadata)
            logger.info("         Timing: total=%.3fs model=%.3fs", total_time, predict_time)

            return prediction

        except Exception as exc:
            logger.error("         Prediction failed for %s: %s", ticker, exc)
            raise RuntimeError(f"Prediction failed: {exc}")

