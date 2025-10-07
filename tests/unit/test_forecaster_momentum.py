"""Tests for the momentum-based fallback in ProbabilisticForecaster."""
import pytest

from pipelines.realtime.models.forecaster import ProbabilisticForecaster


def _make_market_series(start: float, step: float, days: int):
    """Create a list of daily closings following a simple linear trend."""
    # What: Generate deterministic price data so the fallback can compute momentum.
    # Why: Allows us to assert that probabilities depend on real numeric deltas, not constants.
    # How: Create dictionaries mimicking market_data entries with monotonically changing closes.
    # Data: Returns a list of dicts like {"close": price} for each simulated day.
    series = []
    price = start
    for _ in range(days):
        series.append({"close": round(price, 2)})
        price += step
    return series


def test_trend_based_prediction_prefers_upside_for_rising_prices():
    """Momentum fallback should assign higher probability to 'up' for rising prices."""
    forecaster = ProbabilisticForecaster(model_type="gradient_boosting")
    rising_data = _make_market_series(start=100.0, step=1.5, days=20)

    result = forecaster._trend_based_prediction(rising_data)

    probs = result.model_metadata["probabilities"]
    assert result.direction == "up"
    assert probs["up"] > probs["down"]
    assert probs["up"] > probs["neutral"]


def test_trend_based_prediction_requires_minimum_history():
    """Fallback should raise the same error as _default_prediction when data is insufficient."""
    forecaster = ProbabilisticForecaster(model_type="gradient_boosting")
    tiny_history = _make_market_series(start=50.0, step=0.5, days=4)

    with pytest.raises(RuntimeError):
        forecaster._trend_based_prediction(tiny_history)
