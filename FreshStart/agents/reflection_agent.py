"""Reflection agent for quality assurance and validation."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Quality assurance agent validating other agents' outputs.

    Validates prediction reasonableness, sentiment-prediction alignment,
    data freshness, and fundamentals completeness.
    """

    def __init__(
        self,
        *,
        max_data_age_days: int = 3,
        min_confidence_threshold: float = 0.0,
        max_confidence_threshold: float = 1.0,
        enable_alignment_check: bool = True,
        enable_freshness_check: bool = True,
        enable_completeness_check: bool = True
    ):
        """Initialize reflection agent.

        Args:
            max_data_age_days: Maximum age of market data before stale
            min_confidence_threshold: Minimum valid confidence value
            max_confidence_threshold: Maximum valid confidence value
            enable_alignment_check: Whether to check prediction-sentiment alignment
            enable_freshness_check: Whether to check data freshness
            enable_completeness_check: Whether to check fundamentals completeness
        """
        self.max_data_age_days = max_data_age_days
        self.min_confidence = min_confidence_threshold
        self.max_confidence = max_confidence_threshold
        self.enable_alignment_check = enable_alignment_check
        self.enable_freshness_check = enable_freshness_check
        self.enable_completeness_check = enable_completeness_check

        logger.info(
            f"ReflectionAgent initialized: max_data_age={max_data_age_days}d, "
            f"confidence_range=[{min_confidence_threshold}, {max_confidence_threshold}]"
        )

    def run(
        self,
        *,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        filing: Dict[str, Any],
        market_data: Any,
        fundamentals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate agent outputs and flag quality issues.

        Args:
            prediction: PredictionAgent result
            sentiment: SentimentAgent result
            filing: SmartMoneyAgent result (unused in MVP)
            market_data: Historical price/volume data
            fundamentals: Company financial metrics

        Returns:
            Dict with validation_passed, confidence_adjustment, issues,
            recommendations, checks_performed, checks_passed
        """
        issues: List[str] = []
        recommendations: List[str] = []
        confidence_adjustment: float = 0.0
        checks_performed: Dict[str, bool] = {}
        checks_passed: Dict[str, bool] = {}

        logger.debug(
            f"Running reflection validation on prediction={prediction.get('direction')}, "
            f"sentiment={sentiment.get('current')}"
        )

        # Check 1: Prediction Validity
        checks_performed["prediction_validity"] = True
        if not self._check_prediction_validity(prediction):
            checks_passed["prediction_validity"] = False
            issues.append("Prediction confidence out of valid range or direction invalid")
            recommendations.append("Retrain forecasting model or check feature engineering")
            confidence_adjustment -= 0.2
        else:
            checks_passed["prediction_validity"] = True

        # Check 2: Sentiment-Prediction Alignment
        if self.enable_alignment_check:
            checks_performed["sentiment_alignment"] = True
            if self._check_sentiment_alignment(prediction, sentiment):
                checks_passed["sentiment_alignment"] = False
                issues.append(
                    f"Prediction '{prediction.get('direction')}' conflicts with "
                    f"sentiment '{sentiment.get('current')}'"
                )
                recommendations.append("Review recent news or check sentiment model")
                confidence_adjustment -= 0.1
            else:
                checks_passed["sentiment_alignment"] = True

        # Check 3: Data Freshness
        if self.enable_freshness_check:
            checks_performed["data_freshness"] = True
            freshness_issue = self._check_data_freshness(market_data)
            if freshness_issue:
                checks_passed["data_freshness"] = False
                issues.append(freshness_issue)
                recommendations.append("Refresh market data from data adapter")
                confidence_adjustment -= 0.15
            else:
                checks_passed["data_freshness"] = True

        # Check 4: Fundamentals Completeness
        if self.enable_completeness_check:
            checks_performed["fundamentals_completeness"] = True
            missing_keys = self._check_fundamentals_completeness(fundamentals)
            if missing_keys:
                checks_passed["fundamentals_completeness"] = False
                issues.append(f"Missing critical metrics: {', '.join(missing_keys)}")
                recommendations.append("Update fundamentals or check ticker validity")
                confidence_adjustment -= 0.05
            else:
                checks_passed["fundamentals_completeness"] = True

        validation_passed = len(issues) == 0
        confidence_adjustment = max(-0.25, confidence_adjustment)

        logger.info(
            f"Reflection {'PASSED' if validation_passed else 'FAILED'}: "
            f"{len(issues)} issues, adjustment={confidence_adjustment:.3f}"
        )

        return {
            "validation_passed": validation_passed,
            "confidence_adjustment": confidence_adjustment,
            "issues": issues,
            "recommendations": recommendations,
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
        }

    def _check_prediction_validity(self, prediction: Dict[str, Any]) -> bool:
        """Validate prediction confidence and direction."""
        if not prediction:
            logger.warning("Prediction is missing or empty")
            return False

        confidence = prediction.get("confidence")
        direction = prediction.get("direction")

        if confidence is None or not (
            self.min_confidence <= confidence <= self.max_confidence
        ):
            logger.warning(
                f"Invalid confidence: {confidence} "
                f"(expected [{self.min_confidence}, {self.max_confidence}])"
            )
            return False

        if direction not in ["up", "down", "neutral"]:
            logger.warning(f"Invalid direction: {direction}")
            return False

        return True

    def _check_sentiment_alignment(
        self,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any]
    ) -> bool:
        """Check if prediction direction aligns with sentiment.

        Returns:
            True if misalignment detected (issue), False if aligned (OK)
        """
        pred_direction = prediction.get("direction", "neutral")
        sentiment_current = sentiment.get("current", "neutral")

        if pred_direction == "up" and sentiment_current == "negative":
            logger.debug("Misalignment: prediction='up' but sentiment='negative'")
            return True

        if pred_direction == "down" and sentiment_current == "positive":
            logger.debug("Misalignment: prediction='down' but sentiment='positive'")
            return True

        return False

    def _check_data_freshness(self, market_data: Any) -> Optional[str]:
        """Verify market data is recent.

        Returns:
            Error message if stale, None if fresh
        """
        if not market_data:
            logger.warning("Market data is missing")
            return "Market data is missing"

        if isinstance(market_data, list):
            if len(market_data) == 0:
                logger.warning("Market data list is empty")
                return "Market data is empty"

            latest_record = market_data[-1]
            latest_date_str = latest_record.get("date")

            if not latest_date_str:
                logger.warning("Latest market data missing date field")
                return "Market data missing date information"

        elif isinstance(market_data, dict):
            latest_date_str = market_data.get("date")
            if not latest_date_str:
                logger.warning("Market data dict missing date field")
                return "Market data missing date information"
        else:
            logger.warning(f"Market data has unexpected type: {type(market_data)}")
            return "Market data has invalid format"

        try:
            latest_date = datetime.fromisoformat(latest_date_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse date '{latest_date_str}': {e}")
            return f"Invalid date format: {latest_date_str}"

        days_old = (datetime.now() - latest_date).days

        if days_old > self.max_data_age_days:
            logger.warning(f"Market data is stale: {days_old} days old")
            return f"Market data is {days_old} days old (max: {self.max_data_age_days})"

        return None

    def _check_fundamentals_completeness(
        self,
        fundamentals: Dict[str, Any]
    ) -> List[str]:
        """Check if required fundamental metrics are present.

        Returns:
            List of missing required keys (empty if complete)
        """
        required_keys = [
            "pe_ratio",
            "revenue_growth",
            "ebitda_margin",
        ]

        missing_keys = [key for key in required_keys if key not in fundamentals]

        if missing_keys:
            logger.warning(f"Missing fundamental metrics: {missing_keys}")

        return missing_keys
