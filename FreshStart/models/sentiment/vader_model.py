from typing import Dict, Any, List
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from .base_sentiment import BaseSentimentModel, SentimentResult


class VADERModel(BaseSentimentModel):
    """VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment model.

    Rule-based sentiment analysis optimized for social media text.
    Fast and lightweight, no training required.
    """

    def __init__(self):
        """Initialize VADER sentiment analyzer.

        Downloads VADER lexicon if not already present.
        """
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        self.analyzer = SentimentIntensityAnalyzer()

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

        scores = self.analyzer.polarity_scores(text)

        # VADER returns: neg, neu, pos, compound
        # compound is overall score from -1 to 1
        compound = scores['compound']

        # Determine label based on compound score
        if compound >= 0.05:
            label = "positive"
            confidence = min(1.0, (compound + 1.0) / 2.0)
        elif compound <= -0.05:
            label = "negative"
            confidence = min(1.0, (1.0 - compound) / 2.0)
        else:
            label = "neutral"
            confidence = 1.0 - abs(compound)

        return SentimentResult(
            label=label,
            confidence=confidence,
            probabilities={
                "positive": float(scores['pos']),
                "negative": float(scores['neg']),
                "neutral": float(scores['neu'])
            },
            metadata={
                "model": "VADER",
                "compound_score": compound,
                "raw_scores": scores
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
            "name": "VADER",
            "version": "nltk.sentiment.vader",
            "type": "rule-based",
            "capabilities": ["analyze", "analyze_batch"],
            "fine_tuned": False,
            "description": "Lexicon and rule-based sentiment analysis"
        }
