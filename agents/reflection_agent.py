"""Reflection Agent for quality validation.

JOSH's Component - Reflection Agent
Validates prediction and sentiment results for quality assurance.
"""
import logging
from typing import Any, Dict, List

from tools.reflection_tools import check_alignment, check_freshness, check_prediction_validity

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Agent that validates other agents' outputs."""

    def __init__(
        self,
        max_data_age_days: int = 3,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
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
        fundamentals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate agent outputs."""
        logger.info("      ReflectionAgent Execution")
        logger.info(
            "         Validating prediction and sentiment alignment: %s (%.1f%%) vs %s (%.2f)",
            prediction.get("direction", "N/A"),
            prediction.get("confidence", 0) * 100,
            sentiment.get("current", "N/A"),
            sentiment.get("score", 0),
        )

        issues: List[str] = []
        recommendations: List[str] = []
        confidence_adjustment = 0.0

        logger.info("         Check 1: Prediction validity")
        check1_pass = check_prediction_validity(prediction, self.min_confidence, self.max_confidence)
        if not check1_pass:
            issues.append("Prediction confidence out of valid range")
            recommendations.append("Review prediction model calibration")
            confidence_adjustment -= 0.2
            logger.warning("            FAILED - Confidence: %.1f%%", prediction.get("confidence", 0) * 100)
        else:
            logger.info(
                "            PASSED - Confidence: %.1f%%, Direction: %s",
                prediction.get("confidence", 0) * 100,
                prediction.get("direction", "N/A"),
            )

        logger.info("         Check 2: Sentiment-prediction alignment")
        check2_pass = check_alignment(prediction, sentiment)
        if not check2_pass:
            issues.append("Prediction and sentiment show conflicting signals")
            recommendations.append("Verify data sources and model assumptions")
            confidence_adjustment -= 0.1
            logger.warning(
                "            FAILED - Prediction: %s, Sentiment: %s",
                prediction.get("direction", "N/A"),
                sentiment.get("current", "N/A"),
            )
        else:
            logger.info(
                "            PASSED - Prediction: %s, Sentiment: %s",
                prediction.get("direction", "N/A"),
                sentiment.get("current", "N/A"),
            )

        logger.info("         Check 3: Data freshness")
        check3_pass = check_freshness(market_data, self.max_data_age_days)
        if not check3_pass:
            issues.append("Market data may be stale")
            recommendations.append("Refresh data from API")
            confidence_adjustment -= 0.15
            logger.warning("            FAILED - Data is stale")
        else:
            logger.info("            PASSED - Data is fresh")

        validation_passed = len(issues) == 0

        logger.info("      Reflection Result")
        logger.info("         Validation: %s", "PASSED" if validation_passed else "FAILED")
        logger.info("         Issues found: %d", len(issues))
        if issues:
            for i, issue in enumerate(issues, 1):
                logger.info("            %d. %s", i, issue)
        logger.info("         Confidence adjustment: %.2f", confidence_adjustment)

        return {
            "validation_passed": validation_passed,
            "confidence_adjustment": confidence_adjustment,
            "issues": issues,
            "recommendations": recommendations,
            "checks_performed": {
                "prediction_validity": True,
                "alignment": True,
                "freshness": True,
            },
        }

