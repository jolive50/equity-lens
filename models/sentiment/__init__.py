"""Sentiment analysis models module.

Provides 6 sentiment models + ensemble:
- FinBERT: Financial sentiment (TensorFlow)
- DeBERTa: Advanced sentiment (PyTorch)
- RoBERTa: General sentiment (TensorFlow)
- AlphaVantage: Alpha Vantage API news fetcher
- VADER: Rule-based sentiment (legacy)
- TextBlob: Pattern-based sentiment (legacy)
- SentimentEnsemble: Soft-vote ensemble with 5-class support
"""

# New 5-class sentiment system
from .base_sentiment import (
    BaseSentiment,
    SentimentResult,
    SentimentLabel5,
    label5_to_str,
    str_to_label5,
    normalize_probs,
    DummySentiment
)

# New advanced models
from .finbert_model import FinBertSentiment
from .deberta_model import DeBertaSentiment
from .roberta_model import RobertaSentiment
from .ensemble import SentimentEnsemble

# Alpha Vantage news fetcher
from .alpha_vantage_sentiment import fetch_alpha_vantage_news

# Legacy models (kept for backward compatibility)
try:
    from .vader_model import VADERModel
except ImportError:
    VADERModel = None

try:
    from .textblob_model import TextBlobModel
except ImportError:
    TextBlobModel = None

# Compatibility adapter for legacy API
from .compat import LegacyModelAdapter, LegacySentimentResult


# Backward compatibility aliases - wrap new models with legacy API
class FinBERTModel:
    """Legacy FinBERT wrapper for backward compatibility."""
    def __new__(cls, *args, **kwargs):
        return LegacyModelAdapter(FinBertSentiment(*args, **kwargs))


class RoBERTaModel:
    """Legacy RoBERTa wrapper for backward compatibility."""
    def __new__(cls, *args, **kwargs):
        return LegacyModelAdapter(RobertaSentiment(*args, **kwargs))


AlphaVantageSentiment = fetch_alpha_vantage_news
BaseSentimentModel = BaseSentiment

__all__ = [
    # New API
    "BaseSentiment",
    "SentimentResult",
    "SentimentLabel5",
    "label5_to_str",
    "str_to_label5",
    "normalize_probs",
    "DummySentiment",
    "FinBertSentiment",
    "DeBertaSentiment",
    "RobertaSentiment",
    "SentimentEnsemble",
    "fetch_alpha_vantage_news",
    # Legacy aliases
    "FinBERTModel",
    "RoBERTaModel",
    "AlphaVantageSentiment",
    "BaseSentimentModel",
    "VADERModel",
    "TextBlobModel",
]
