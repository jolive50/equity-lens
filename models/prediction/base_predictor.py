"""Base class for all prediction models.

PAM's Component - Base Prediction Model Interface
This defines the standard interface all prediction models must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class PredictionResult:
    """Standardized prediction result from all models."""
    direction: str  # "up", "down", or "neutral"
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # {up, down, neutral} probabilities
    metadata: Dict[str, Any]  # Model-specific information


class BasePredictionModel(ABC):
    """Abstract base class for prediction models.

    All prediction models (LSTM, GRU, Gradient Boost, Ensemble) inherit from this.
    Ensures consistent interface for polymorphic use in agents.
    """

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make a prediction based on market data.

        Args:
            data: DataFrame with columns [date, open, high, low, close, volume]

        Returns:
            PredictionResult with direction, confidence, and probabilities
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, version, architecture).

        Returns:
            Dictionary with model information
        """
        pass
