"""Mock predictor for testing ensemble evaluation.

Simple predictor implementation for testing purposes.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from .base_predictor import BasePredictionModel, PredictionResult


class MockPredictor(BasePredictionModel):
    """Simple mock predictor for testing."""

    def __init__(self, name: str, bias: str = "neutral"):
        """Initialize mock predictor.

        Args:
            name: Name of the predictor
            bias: Direction bias ("up", "down", or "neutral")
        """
        self.name = name
        self.bias = bias

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make a simple prediction based on bias."""
        # Simple logic: use closing price change if available
        if 'close' in data.columns and len(data) > 0:
            # Use some randomness but biased toward our bias
            rand = np.random.random()

            if self.bias == "up":
                probs = {"up": 0.6, "neutral": 0.25, "down": 0.15}
            elif self.bias == "down":
                probs = {"up": 0.15, "neutral": 0.25, "down": 0.6}
            else:
                probs = {"up": 0.33, "neutral": 0.34, "down": 0.33}

            # Add some randomness
            noise = (np.random.random(3) - 0.5) * 0.1
            probs_array = np.array([probs["down"], probs["neutral"], probs["up"]]) + noise
            probs_array = np.maximum(probs_array, 0.01)  # Ensure positive
            probs_array /= probs_array.sum()  # Normalize

            direction = ["down", "neutral", "up"][np.argmax(probs_array)]

            return PredictionResult(
                direction=direction,
                confidence=float(probs_array[np.argmax(probs_array)]),
                probabilities={
                    "up": float(probs_array[2]),
                    "down": float(probs_array[0]),
                    "neutral": float(probs_array[1])
                },
                metadata={"model": self.name, "bias": self.bias}
            )
        else:
            return PredictionResult(
                direction="neutral",
                confidence=0.34,
                probabilities={"up": 0.33, "down": 0.33, "neutral": 0.34},
                metadata={"model": self.name}
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model info."""
        return {
            "name": self.name,
            "type": "mock",
            "bias": self.bias
        }
