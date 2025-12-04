"""Validation helpers shared by the reflection agent."""

from typing import Any, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def check_prediction_validity(
    prediction: Dict[str, Any],
    min_confidence: float,
    max_confidence: float,
) -> bool:
    """Verify prediction direction and confidence are within expected bounds."""
    confidence = prediction.get("confidence", 0.0)
    direction = prediction.get("direction", "neutral")

    if not (min_confidence <= confidence <= max_confidence):
        return False

    if direction not in ["up", "down", "neutral"]:
        return False

    return True


def check_alignment(prediction: Dict[str, Any], sentiment: Dict[str, Any]) -> bool:
    """Check if prediction direction aligns with sentiment signal."""
    pred_dir = prediction.get("direction", "neutral")
    sent_current = sentiment.get("current", "neutral")

    if pred_dir == "up" and sent_current == "negative":
        return False

    if pred_dir == "down" and sent_current == "positive":
        return False

    return True


def check_freshness(market_data: Any, max_data_age_days: int) -> bool:
    """Determine whether market data is fresh enough for reflection checks."""
    try:
        if isinstance(market_data, list) and market_data:
            latest_date_str = market_data[-1].get("date", "")
            if latest_date_str:
                latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
                age_days = (datetime.now() - latest_date).days
                if age_days > max_data_age_days:
                    return False

        return True
    except Exception as exc:
        logger.warning("Could not check data freshness: %s", exc)
        return True

