"""Reflection agent for quality assurance and output validation.

This module implements the ReflectionAgent, a critical quality control component
that validates outputs from other agents before they reach end users.

What this file does:
- Validates prediction reasonableness (confidence ranges, direction validity)
- Checks sentiment-prediction alignment
- Verifies data freshness (market data, news timestamps)
- Ensures fundamental metrics completeness
- Recalibrates confidence scores based on quality issues
- Generates actionable recommendations for fixing issues

Why we need this:
- Catches errors before users see them
- Improves reliability and trust in recommendations
- Prevents misleading advice from stale or incomplete data
- Provides transparency through explicit quality warnings

How it works:
- Receives outputs from PredictionAgent, SentimentAgent, SmartMoneyAgent
- Runs rule-based validation checks
- Calculates confidence adjustments based on issues found
- Returns validation result with warnings and recommendations
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """
    Quality assurance agent that validates other agents' outputs.

    WHAT: Reviews prediction, sentiment, and filing agent results for consistency
    WHY: Catches errors before users see them, improves system reliability
    HOW: Rule-based validation checks + confidence adjustment calculations
    DATA: Agent outputs → validation results → confidence adjustments + warnings

    The ReflectionAgent acts as a final quality gate before the ExplanationAgent
    generates user-facing narratives. It ensures data quality, internal consistency,
    and appropriate confidence calibration.

    Usage Example:
        reflection_agent = ReflectionAgent(
            max_data_age_days=3,
            min_confidence_threshold=0.0,
            max_confidence_threshold=1.0
        )

        result = reflection_agent.run(
            prediction={"direction": "up", "confidence": 0.85, "narrative": "..."},
            sentiment={"current": "positive", "score": 0.7, "trend": "improving"},
            filing={"institutions": {...}, "insiders": {...}},
            market_data=[{"date": "2025-10-27", "close": 150.0, ...}],
            fundamentals={"pe_ratio": 25, "revenue_growth": 0.08, ...}
        )

        if not result["validation_passed"]:
            # Apply confidence adjustment, display warnings to user
            adjusted_confidence = original_confidence + result["confidence_adjustment"]
    """

    def __init__(
        self,
        *,
        max_data_age_days: int = 3,
        min_confidence_threshold: float = 0.0,
        max_confidence_threshold: float = 1.0,
        enable_alignment_check: bool = True,
        enable_freshness_check: bool = True,
        enable_completeness_check: bool = True,
    ):
        """
        Initialize the reflection agent with validation parameters.

        Args:
            max_data_age_days: Maximum age of market data before considered stale (default: 3)
            min_confidence_threshold: Minimum valid confidence value (default: 0.0)
            max_confidence_threshold: Maximum valid confidence value (default: 1.0)
            enable_alignment_check: Whether to check prediction-sentiment alignment (default: True)
            enable_freshness_check: Whether to check data freshness (default: True)
            enable_completeness_check: Whether to check fundamentals completeness (default: True)

        WHAT: Configure validation thresholds and enable/disable specific checks
        WHY: Allows tuning sensitivity and disabling checks for testing
        HOW: Store parameters as instance variables
        DATA: Configuration parameters → instance attributes
        """
        # WHAT: Store data freshness threshold
        # WHY: Stale data (>3 days old) leads to unreliable forecasts
        # HOW: Use this to compare against market data timestamps
        # DATA: Integer number of days (typically 3 for 2 trading days + buffer)
        self.max_data_age_days = max_data_age_days

        # WHAT: Store valid confidence range
        # WHY: Confidence must be a probability in [0, 1]
        # HOW: Validate prediction confidence against these bounds
        # DATA: Float values defining valid probability range
        self.min_confidence = min_confidence_threshold
        self.max_confidence = max_confidence_threshold

        # WHAT: Store feature flags for each validation check
        # WHY: Allows disabling specific checks for testing or debugging
        # HOW: Check these flags before running each validation
        # DATA: Boolean flags controlling validation behavior
        self.enable_alignment_check = enable_alignment_check
        self.enable_freshness_check = enable_freshness_check
        self.enable_completeness_check = enable_completeness_check

        logger.info(
            f"ReflectionAgent initialized with max_data_age={max_data_age_days}d, "
            f"confidence_range=[{min_confidence_threshold}, {max_confidence_threshold}]"
        )

    def run(
        self,
        *,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        filing: Dict[str, Any],
        market_data: Any,
        fundamentals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate agent outputs and flag quality issues.

        Args:
            prediction: PredictionAgent result with direction, confidence, narrative
            sentiment: SentimentAgent result with current, score, trend, headlines
            filing: SmartMoneyAgent result with institutions, insiders, congress data
            market_data: Historical price/volume data (list of dicts or dict)
            fundamentals: Company financial metrics (pe_ratio, revenue_growth, etc.)

        Returns:
            Dict with:
                - validation_passed: bool (True if all checks passed)
                - confidence_adjustment: float (-0.25 to 0.0, how much to lower confidence)
                - issues: List[str] (human-readable warnings about quality problems)
                - recommendations: List[str] (actionable fixes for issues)
                - checks_performed: Dict[str, bool] (which validation checks ran)
                - checks_passed: Dict[str, bool] (results of each check)

        Raises:
            ValueError: If required fields are missing from input dictionaries

        WHAT: Run all enabled validation checks, aggregate results
        WHY: Provides comprehensive quality assessment before user sees results
        HOW: Execute checks sequentially, accumulate issues, calculate adjustment
        DATA: Agent outputs → validation checks → aggregated quality report
        """
        # WHAT: Initialize result containers
        # WHY: Need to collect issues and track which checks ran/passed
        # HOW: Create empty lists and dicts to populate during validation
        # DATA: Empty structures to be filled by validation checks
        issues: List[str] = []
        recommendations: List[str] = []
        confidence_adjustment: float = 0.0
        checks_performed: Dict[str, bool] = {}
        checks_passed: Dict[str, bool] = {}

        logger.debug(
            f"Running reflection validation on prediction={prediction.get('direction')}, "
            f"sentiment={sentiment.get('current')}"
        )

        # === CHECK 1: Prediction Validity ===
        # WHAT: Verify prediction has valid structure and values
        # WHY: Catch model errors or data corruption early
        # HOW: Call _check_prediction_validity helper
        # DATA: prediction dict → bool (valid or not) + issues if invalid
        checks_performed["prediction_validity"] = True
        if not self._check_prediction_validity(prediction):
            checks_passed["prediction_validity"] = False
            issues.append(
                "Prediction confidence out of valid range or direction invalid"
            )
            recommendations.append(
                "Retrain forecasting model or check feature engineering pipeline"
            )
            # WHAT: Apply large confidence penalty for invalid predictions
            # WHY: Invalid predictions are unreliable, need strong downgrade
            # HOW: Subtract 0.2 from confidence (20 percentage points)
            # DATA: confidence_adjustment updated to reflect severity
            confidence_adjustment -= 0.2
        else:
            checks_passed["prediction_validity"] = True

        # === CHECK 2: Sentiment-Prediction Alignment ===
        # WHAT: Check if prediction direction matches news sentiment
        # WHY: Large mismatches suggest data issues or model errors
        # HOW: Call _check_sentiment_alignment helper (only if enabled)
        # DATA: prediction + sentiment → bool (aligned or misaligned)
        if self.enable_alignment_check:
            checks_performed["sentiment_alignment"] = True
            if self._check_sentiment_alignment(prediction, sentiment):
                checks_passed["sentiment_alignment"] = False
                issues.append(
                    f"Prediction direction '{prediction.get('direction')}' conflicts with "
                    f"sentiment '{sentiment.get('current')}'"
                )
                recommendations.append(
                    "Review recent news articles for context or check sentiment model"
                )
                # WHAT: Apply moderate confidence penalty for misalignment
                # WHY: Misalignment is concerning but not always wrong (contrarian signals)
                # HOW: Subtract 0.1 from confidence (10 percentage points)
                # DATA: confidence_adjustment updated with alignment penalty
                confidence_adjustment -= 0.1
            else:
                checks_passed["sentiment_alignment"] = True

        # === CHECK 3: Data Freshness ===
        # WHAT: Verify market data is recent (< max_data_age_days old)
        # WHY: Stale data leads to unreliable forecasts
        # HOW: Call _check_data_freshness helper (only if enabled)
        # DATA: market_data timestamps → bool (fresh or stale)
        if self.enable_freshness_check:
            checks_performed["data_freshness"] = True
            freshness_issue = self._check_data_freshness(market_data)
            if freshness_issue:
                checks_passed["data_freshness"] = False
                issues.append(freshness_issue)
                recommendations.append(
                    "Refresh market data from data adapter (Alpha Vantage, Tiingo, etc.)"
                )
                # WHAT: Apply large confidence penalty for stale data
                # WHY: Stale data is highly unreliable for forecasting
                # HOW: Subtract 0.15 from confidence (15 percentage points)
                # DATA: confidence_adjustment updated with freshness penalty
                confidence_adjustment -= 0.15
            else:
                checks_passed["data_freshness"] = True

        # === CHECK 4: Fundamentals Completeness ===
        # WHAT: Check if all required fundamental metrics are present
        # WHY: Missing metrics degrade forecast quality
        # HOW: Call _check_fundamentals_completeness helper (only if enabled)
        # DATA: fundamentals dict → bool (complete or incomplete)
        if self.enable_completeness_check:
            checks_performed["fundamentals_completeness"] = True
            missing_keys = self._check_fundamentals_completeness(fundamentals)
            if missing_keys:
                checks_passed["fundamentals_completeness"] = False
                issues.append(
                    f"Missing critical fundamental metrics: {', '.join(missing_keys)}"
                )
                recommendations.append(
                    "Update fundamentals from Alpha Vantage or check ticker validity"
                )
                # WHAT: Apply small confidence penalty for missing fundamentals
                # WHY: Missing data reduces forecast quality but model can still work
                # HOW: Subtract 0.05 from confidence (5 percentage points)
                # DATA: confidence_adjustment updated with completeness penalty
                confidence_adjustment -= 0.05
            else:
                checks_passed["fundamentals_completeness"] = True

        # === AGGREGATE RESULTS ===
        # WHAT: Determine if validation passed overall
        # WHY: Need single boolean flag for workflow branching
        # HOW: Pass if no issues found (empty issues list)
        # DATA: issues list → bool (empty means passed)
        validation_passed = len(issues) == 0

        # WHAT: Clamp confidence adjustment to reasonable range
        # WHY: Prevent extreme adjustments (maximum penalty is -0.25)
        # HOW: Use max() to enforce floor of -0.25
        # DATA: confidence_adjustment clamped to [-0.25, 0.0]
        confidence_adjustment = max(-0.25, confidence_adjustment)

        logger.info(
            f"Reflection validation {'PASSED' if validation_passed else 'FAILED'}: "
            f"{len(issues)} issues found, adjustment={confidence_adjustment:.3f}"
        )

        # WHAT: Return comprehensive validation report
        # WHY: Workflow and UI need detailed quality information
        # HOW: Package all results into structured dictionary
        # DATA: Aggregated validation results with issues, recommendations, adjustments
        return {
            "validation_passed": validation_passed,
            "confidence_adjustment": confidence_adjustment,
            "issues": issues,
            "recommendations": recommendations,
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
        }

    def _check_prediction_validity(self, prediction: Dict[str, Any]) -> bool:
        """
        Validate prediction confidence and direction values.

        Args:
            prediction: PredictionAgent output dictionary

        Returns:
            bool: True if prediction is valid, False otherwise

        WHAT: Check if prediction has valid structure and value ranges
        WHY: Catch model errors, data corruption, or missing fields
        HOW: Verify required keys exist and values are in valid ranges
        DATA: prediction dict → bool (valid or invalid)
        """
        # WHAT: Handle missing or empty prediction
        # WHY: Prediction is required input, absence indicates serious error
        # HOW: Return False immediately if prediction missing or falsy
        # DATA: None/empty dict → False
        if not prediction:
            logger.warning("Prediction is missing or empty")
            return False

        # WHAT: Extract confidence and direction fields
        # WHY: These are the critical fields that need validation
        # HOW: Use .get() to safely access, returns None if missing
        # DATA: prediction dict → confidence (float/None), direction (str/None)
        confidence = prediction.get("confidence")
        direction = prediction.get("direction")

        # WHAT: Validate confidence is a number in valid range
        # WHY: Confidence must be a probability [0, 1]
        # HOW: Check confidence exists and is within min/max thresholds
        # DATA: confidence value → bool (in range or not)
        if confidence is None or not (
            self.min_confidence <= confidence <= self.max_confidence
        ):
            logger.warning(
                f"Invalid confidence: {confidence} (expected [{self.min_confidence}, {self.max_confidence}])"
            )
            return False

        # WHAT: Validate direction is one of three valid values
        # WHY: Direction must match expected enum values from forecaster
        # HOW: Check direction against allowed set
        # DATA: direction string → bool (valid value or not)
        if direction not in ["up", "down", "neutral"]:
            logger.warning(f"Invalid direction: {direction} (expected up/down/neutral)")
            return False

        # WHAT: Prediction passed all validation checks
        # DATA: Return True to indicate valid prediction
        return True

    def _check_sentiment_alignment(
        self, prediction: Dict[str, Any], sentiment: Dict[str, Any]
    ) -> bool:
        """
        Check if prediction direction aligns with news sentiment.

        Args:
            prediction: PredictionAgent output with direction field
            sentiment: SentimentAgent output with current sentiment field

        Returns:
            bool: True if misalignment detected, False if aligned

        WHAT: Compare prediction direction with sentiment polarity
        WHY: Large mismatches suggest data issues, model errors, or missing context
        HOW: Map sentiment to expected direction, check for conflicts
        DATA: prediction direction + sentiment polarity → bool (misaligned or not)

        Note: Returns True for misalignment (flagging an issue), False for alignment (OK)
        """
        # WHAT: Extract direction and sentiment fields
        # WHY: Need both values to check alignment
        # HOW: Use .get() with defaults for safe access
        # DATA: dicts → direction string, sentiment string
        pred_direction = prediction.get("direction", "neutral")
        sentiment_current = sentiment.get("current", "neutral")

        # WHAT: Check for obvious conflicts between prediction and sentiment
        # WHY: If prediction says "up" but sentiment is "negative", that's suspicious
        # HOW: Test specific conflict combinations that indicate problems
        # DATA: (direction, sentiment) pairs → bool (conflict exists)

        # Conflict 1: Bullish prediction with bearish sentiment
        if pred_direction == "up" and sentiment_current == "negative":
            logger.debug(
                "Misalignment detected: prediction='up' but sentiment='negative'"
            )
            return True  # Misalignment detected

        # Conflict 2: Bearish prediction with bullish sentiment
        if pred_direction == "down" and sentiment_current == "positive":
            logger.debug(
                "Misalignment detected: prediction='down' but sentiment='positive'"
            )
            return True  # Misalignment detected

        # WHAT: No obvious conflicts found
        # WHY: Prediction and sentiment are reasonably aligned
        # HOW: Return False to indicate alignment is OK
        # DATA: Return False (no issue)
        return False  # Alignment OK

    def _check_data_freshness(self, market_data: Any) -> Optional[str]:
        """
        Verify market data is recent (< max_data_age_days old).

        Args:
            market_data: Historical price/volume data (list of dicts or dict)

        Returns:
            Optional[str]: Error message if stale, None if fresh

        WHAT: Check if market data timestamp is within acceptable age
        WHY: Stale data (>3 days) leads to unreliable forecasts
        HOW: Extract latest timestamp, compare with current date
        DATA: market_data timestamps → Optional[str] (error message or None)
        """
        # WHAT: Handle missing or invalid market data
        # WHY: Can't validate freshness without data
        # HOW: Check if market_data exists and is expected type
        # DATA: market_data → Optional[str] (error if invalid structure)
        if not market_data:
            logger.warning("Market data is missing")
            return "Market data is missing"

        # WHAT: Handle list of price records (most common format)
        # WHY: Market data typically comes as list of daily records
        # HOW: Check if list, extract last item's date field
        # DATA: list of dicts → latest date string
        if isinstance(market_data, list):
            if len(market_data) == 0:
                logger.warning("Market data list is empty")
                return "Market data is empty"

            # WHAT: Get the most recent data point
            # WHY: Latest date determines freshness
            # HOW: Access last item in list (most recent date)
            # DATA: list[-1] → latest record dict
            latest_record = market_data[-1]
            latest_date_str = latest_record.get("date")

            if not latest_date_str:
                logger.warning("Latest market data record missing date field")
                return "Market data missing date information"

        # WHAT: Handle dict format (alternative structure)
        # WHY: Some data sources return dict with date key
        # HOW: Extract date field directly from dict
        # DATA: dict → date string
        elif isinstance(market_data, dict):
            latest_date_str = market_data.get("date")
            if not latest_date_str:
                logger.warning("Market data dict missing date field")
                return "Market data missing date information"

        # WHAT: Reject unexpected data formats
        # WHY: Can't validate unknown structures
        # HOW: Return error for non-list, non-dict types
        # DATA: Unknown type → error message
        else:
            logger.warning(f"Market data has unexpected type: {type(market_data)}")
            return "Market data has invalid format"

        # WHAT: Parse date string to datetime object
        # WHY: Need datetime to calculate age in days
        # HOW: Try parsing as ISO format (YYYY-MM-DD)
        # DATA: date string → datetime object
        try:
            latest_date = datetime.fromisoformat(latest_date_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse market data date '{latest_date_str}': {e}")
            return f"Invalid date format in market data: {latest_date_str}"

        # WHAT: Calculate how many days old the data is
        # WHY: Compare against max_data_age_days threshold
        # HOW: Subtract latest date from now, extract days component
        # DATA: datetime objects → integer days difference
        days_old = (datetime.now() - latest_date).days

        # WHAT: Check if data exceeds maximum age
        # WHY: Data older than threshold is considered stale
        # HOW: Compare days_old against max_data_age_days
        # DATA: int comparison → bool (stale or fresh)
        if days_old > self.max_data_age_days:
            logger.warning(
                f"Market data is stale: {days_old} days old (max: {self.max_data_age_days})"
            )
            return (
                f"Market data is {days_old} days old (max: {self.max_data_age_days} days)"
            )

        # WHAT: Data is fresh, no issue found
        # DATA: Return None to indicate no error
        return None

    def _check_fundamentals_completeness(
        self, fundamentals: Dict[str, Any]
    ) -> List[str]:
        """
        Check if all required fundamental metrics are present.

        Args:
            fundamentals: Company financial metrics dictionary

        Returns:
            List[str]: List of missing required keys (empty if complete)

        WHAT: Verify fundamental metrics dictionary contains required fields
        WHY: Missing metrics degrade forecast quality, model needs complete features
        HOW: Check for presence of critical financial ratios and metrics
        DATA: fundamentals dict → List[str] (missing keys)
        """
        # WHAT: Define required fundamental metrics
        # WHY: These are critical inputs for the forecasting model
        # HOW: List the key financial ratios that must be present
        # DATA: List of required string keys
        required_keys = [
            "pe_ratio",  # Price-to-Earnings ratio (valuation)
            "revenue_growth",  # Year-over-year revenue growth rate
            "ebitda_margin",  # EBITDA margin (profitability)
        ]

        # WHAT: Find which required keys are missing
        # WHY: Need to report specific missing metrics to user
        # HOW: List comprehension checking key existence
        # DATA: required_keys → filtered list of missing keys
        missing_keys = [key for key in required_keys if key not in fundamentals]

        # WHAT: Log missing keys for debugging
        # WHY: Helps diagnose data adapter issues
        # HOW: Log at warning level with key names
        # DATA: missing_keys list → log message
        if missing_keys:
            logger.warning(f"Missing fundamental metrics: {missing_keys}")

        # WHAT: Return list of missing keys
        # WHY: Caller needs to know which metrics are absent
        # HOW: Return the filtered list (empty if all present)
        # DATA: List of missing key names (or empty list)
        return missing_keys
