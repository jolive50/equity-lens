"""Reusable helpers for prediction-related agent work."""

from typing import Any, Dict, List, Tuple
import time

import pandas as pd

from models.prediction.base_predictor import BasePredictionModel, PredictionResult


def prepare_market_dataframe(market_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw market data into a DataFrame and validate required fields."""
    df = pd.DataFrame(market_data)
    required_cols = ["close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Market data missing required columns: {required_cols}")
    return df


def build_prediction_narrative(
    ticker: str,
    result: PredictionResult,
    fundamentals: Dict[str, float],
) -> str:
    """Create a short English narrative for a prediction result."""
    model_name = result.metadata.get("model", "Unknown")
    prob_up = result.probabilities["up"]
    prob_down = result.probabilities["down"]

    narrative = (
        f"{ticker} prediction: {result.direction.upper()} with "
        f"{result.confidence:.1%} confidence using {model_name} model. "
        f"Probabilities: up={prob_up:.1%}, down={prob_down:.1%}. "
    )

    pe_ratio = fundamentals.get("pe_ratio", 0)
    if pe_ratio > 0:
        narrative += f"P/E ratio: {pe_ratio:.1f}. "

    return narrative


def predict_with_model(
    model: BasePredictionModel,
    ticker: str,
    market_data: List[Dict[str, Any]],
    fundamentals: Dict[str, float],
) -> Tuple[PredictionResult, float, str]:
    """Run a prediction model and build a narrative for agent consumption."""
    df = prepare_market_dataframe(market_data)
    predict_start = time.time()
    result = model.predict(df)
    predict_time = time.time() - predict_start
    narrative = build_prediction_narrative(ticker, result, fundamentals)
    return result, predict_time, narrative

