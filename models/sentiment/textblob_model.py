from typing import Dict, Any, List
from textblob import TextBlob
from .base_sentiment import BaseSentimentModel, SentimentResult


class TextBlobModel(BaseSentimentModel):
    """TextBlob sentiment model.

    Simple pattern-based sentiment analysis.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """

    def __init__(self):
        """Initialize TextBlob sentiment analyzer."""
        pass  # TextBlob requires no initialization

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of single text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with label, confidence, and probabilities

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine label based on polarity
        if polarity > 0.1:
            label = "positive"
            confidence = min(1.0, (polarity + 1.0) / 2.0)
        elif polarity < -0.1:
            label = "negative"
            confidence = min(1.0, (1.0 - polarity) / 2.0)
        else:
            label = "neutral"
            confidence = 1.0 - abs(polarity)

        # Convert polarity to probability distribution
        # Polarity ranges from -1 (negative) to 1 (positive)
        pos_prob = max(0.0, polarity)
        neg_prob = max(0.0, -polarity)
        neu_prob = 1.0 - pos_prob - neg_prob

        # Normalize to sum to 1.0
        total = pos_prob + neg_prob + neu_prob
        if total > 0:
            pos_prob /= total
            neg_prob /= total
            neu_prob /= total

        return SentimentResult(
            label=label,
            confidence=confidence,
            probabilities={
                "positive": float(pos_prob),
                "negative": float(neg_prob),
                "neutral": float(neu_prob)
            },
            metadata={
                "model": "TextBlob",
                "polarity": polarity,
                "subjectivity": subjectivity
            }
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        return [self.analyze(text) for text in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "TextBlob",
            "version": "textblob",
            "type": "pattern-based",
            "capabilities": ["analyze", "analyze_batch"],
            "fine_tuned": False,
            "description": "Pattern-based sentiment analysis with polarity and subjectivity"
        }
