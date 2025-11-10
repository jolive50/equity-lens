from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Standardized sentiment analysis result."""
    label: str  # "positive", "negative", or "neutral"
    confidence: float  # 0.0-1.0
    probabilities: Dict[str, float]  # {"positive": 0.x, "negative": 0.y, "neutral": 0.z}
    metadata: Dict[str, Any]  # Model-specific metadata


class BaseSentimentModel(ABC):
    """Abstract base class for all sentiment analysis models.

    All sentiment models (FinBERT, RoBERTa, VADER, TextBlob, Alpha Vantage)
    inherit from this class to ensure consistent interface.
    """

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text.

        Args:
            text: Text to analyze (news headline, article content, etc.)

        Returns:
            SentimentResult with label, confidence, probabilities, and metadata

        Raises:
            RuntimeError: If analysis fails
        """
        pass

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts efficiently.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects

        Raises:
            RuntimeError: If batch analysis fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata.

        Returns:
            Dictionary with keys: name, version, type, capabilities
        """
        pass
