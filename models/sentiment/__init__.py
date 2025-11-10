"""Sentiment analysis models module.

Provides 5 sentiment models + ensemble:
- FinBERT: Financial sentiment (fine-tunable)
- RoBERTa: General sentiment (fine-tunable)
- AlphaVantage: API-based sentiment
- VADER: Rule-based sentiment
- TextBlob: Pattern-based sentiment
- SentimentEnsemble: Flexible ensemble of 2+ models
"""

from .base_sentiment import BaseSentimentModel, SentimentResult
from .finbert_model import FinBERTModel
from .roberta_model import RoBERTaModel
from .alpha_vantage_sentiment import AlphaVantageSentiment
from .vader_model import VADERModel
from .textblob_model import TextBlobModel
from .ensemble import SentimentEnsemble

__all__ = [
    "BaseSentimentModel",
    "SentimentResult",
    "FinBERTModel",
    "RoBERTaModel",
    "AlphaVantageSentiment",
    "VADERModel",
    "TextBlobModel",
    "SentimentEnsemble",
]
