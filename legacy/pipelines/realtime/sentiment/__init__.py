"""Sentiment analysis module for financial news.

WHAT: Provides FinBERT-based sentiment analysis with optional VectorStore integration
WHY: Financial news sentiment drives short-term stock price movements
HOW: Uses fine-tuned FinBERT model for accurate financial sentiment classification
DATA: News articles → FinBERT → sentiment scores + trend + historical context
"""

from .finbert import (
    FinBERTSentimentAnalyzer,
    NewsSentimentProcessor,
    create_sentiment_analyzer
)

__all__ = [
    "FinBERTSentimentAnalyzer",
    "NewsSentimentProcessor",
    "create_sentiment_analyzer"
]
