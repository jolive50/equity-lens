"""Compatibility adapters for legacy sentiment model API.

Wraps new 5-class BaseSentiment models to work with old analyze() interface.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class LegacySentimentResult:
    """Legacy SentimentResult format for backward compatibility."""
    label: str  # "positive", "negative", or "neutral"
    confidence: float  # 0.0-1.0
    probabilities: Dict[str, float]  # {"positive": 0.x, "negative": 0.y, "neutral": 0.z}
    metadata: Dict[str, Any]  # Model-specific metadata


class LegacyModelAdapter:
    """Adapter that wraps new BaseSentiment models with legacy analyze() API."""

    def __init__(self, new_model):
        """Initialize adapter with a new BaseSentiment model.

        Args:
            new_model: Instance of FinBertSentiment, DeBertaSentiment, RobertaSentiment, etc.
        """
        self.model = new_model
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model."""
        if not self._loaded:
            self.model.load()
            self._loaded = True

    def analyze(self, text: str) -> LegacySentimentResult:
        """Analyze sentiment using legacy API.

        Args:
            text: Text to analyze

        Returns:
            LegacySentimentResult in old format
        """
        self._ensure_loaded()

        # Use new API
        result = self.model.predict_one(title=text)

        # Convert 5-class label to 3-class
        label_map = {
            -2: "negative",  # strongly negative
            -1: "negative",  # negative
            0: "neutral",    # neutral
            1: "positive",   # positive
            2: "positive"    # strongly positive
        }
        label_3class = label_map.get(int(result.label), "neutral")

        # Convert to legacy format
        return LegacySentimentResult(
            label=label_3class,
            confidence=result.confidence,
            probabilities=result.probs,
            metadata={
                "provider": result.provider,
                "score": result.score,
                "label_5class": int(result.label),
                "entropy": result.entropy
            }
        )

    def analyze_batch(self, texts: List[str]) -> List[LegacySentimentResult]:
        """Analyze multiple texts using legacy API.

        Args:
            texts: List of texts to analyze

        Returns:
            List of LegacySentimentResult objects
        """
        self._ensure_loaded()

        # Build items for new API
        items = [{"title": text, "body": "", "id": str(i)} for i, text in enumerate(texts)]
        results = self.model.predict_batch(items, title_key="title", body_key="body", id_key="id")

        # Convert to legacy format
        label_map = {
            -2: "negative",
            -1: "negative",
            0: "neutral",
            1: "positive",
            2: "positive"
        }

        legacy_results = []
        for result in results:
            label_3class = label_map.get(int(result.label), "neutral")
            legacy_results.append(
                LegacySentimentResult(
                    label=label_3class,
                    confidence=result.confidence,
                    probabilities=result.probs,
                    metadata={
                        "provider": result.provider,
                        "score": result.score,
                        "label_5class": int(result.label),
                        "entropy": result.entropy
                    }
                )
            )

        return legacy_results

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata in legacy format.

        Returns:
            Dictionary with keys: name, version, type, capabilities
        """
        return {
            "name": self.model.provider,
            "version": "2.0",
            "type": "transformer" if self.model.provider in ("finbert", "deberta", "roberta") else "hybrid",
            "capabilities": ["sentiment_analysis", "5_class_labels", "confidence_scoring"]
        }
