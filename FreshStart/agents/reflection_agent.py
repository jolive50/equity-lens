"""Reflection Agent for quality validation.

JOSH's Component - Reflection Agent
Validates prediction and sentiment results for quality assurance.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Agent that validates other agents' outputs."""

    def __init__(
        self,
        max_data_age_days: int = 3,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0
    ):
        """Initialize ReflectionAgent.

        Args:
            max_data_age_days: Maximum age of data before considered stale
            min_confidence: Minimum valid confidence value
            max_confidence: Maximum valid confidence value
        """
        self.max_data_age_days = max_data_age_days
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        logger.info("ReflectionAgent initialized")

    def run(
        self,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        filing: Dict[str, Any],
        market_data: Any,
        fundamentals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate agent outputs.

        Args:
            prediction: PredictionAgent result
            sentiment: SentimentAgent result
            filing: SmartMoneyAgent result (optional)
            market_data: Historical price data
            fundamentals: Financial metrics

        Returns:
            Dict with validation results
        """
        issues = []
        recommendations = []
        confidence_adjustment = 0.0

        # Check 1: Prediction validity
        if not self._check_prediction_validity(prediction):
            issues.append("Prediction confidence out of valid range")
            recommendations.append("Review prediction model calibration")
            confidence_adjustment -= 0.2

        # Check 2: Sentiment-prediction alignment
        if not self._check_alignment(prediction, sentiment):
            issues.append("Prediction and sentiment show conflicting signals")
            recommendations.append("Verify data sources and model assumptions")
            confidence_adjustment -= 0.1

        # Check 3: Data freshness
        if not self._check_freshness(market_data):
            issues.append("Market data may be stale")
            recommendations.append("Refresh data from API")
            confidence_adjustment -= 0.15

        # Determine validation result
        validation_passed = len(issues) == 0

        return {
            "validation_passed": validation_passed,
            "confidence_adjustment": confidence_adjustment,
            "issues": issues,
            "recommendations": recommendations,
            "checks_performed": {
                "prediction_validity": True,
                "alignment": True,
                "freshness": True
            }
        }

    def _check_prediction_validity(self, prediction: Dict[str, Any]) -> bool:
        """Check if prediction has valid structure."""
        confidence = prediction.get("confidence", 0.0)
        direction = prediction.get("direction", "neutral")

        if not (self.min_confidence <= confidence <= self.max_confidence):
            return False

        if direction not in ["up", "down", "neutral"]:
            return False

        return True

    def _check_alignment(self, prediction: Dict[str, Any], sentiment: Dict[str, Any]) -> bool:
        """Check if prediction and sentiment are aligned."""
        pred_dir = prediction.get("direction", "neutral")
        sent_current = sentiment.get("current", "neutral")

        # Strong misalignment: bullish prediction with negative sentiment
        if pred_dir == "up" and sent_current == "negative":
            return False

        # Strong misalignment: bearish prediction with positive sentiment
        if pred_dir == "down" and sent_current == "positive":
            return False

        return True

    def _check_freshness(self, market_data: Any) -> bool:
        """Check if market data is fresh."""
        try:
            if isinstance(market_data, list) and market_data:
                latest_date_str = market_data[-1].get("date", "")
                if latest_date_str:
                    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
                    age_days = (datetime.now() - latest_date).days

                    if age_days > self.max_data_age_days:
                        return False

            return True
        except Exception as e:
            logger.warning(f"Could not check data freshness: {e}")
            return True  # Assume fresh if we can't check
