"""Unit tests for SentimentAgent VectorStore integration."""
import pytest
from unittest.mock import MagicMock
from pipelines.realtime.agents import SentimentAgent

class StubAnalyzer:
    def process_news_articles(self, articles):
        return {
            "current": "positive",
            "score": 0.8,
            "trend": "improving",
            "headlines": ["Test headline"],
        }

def test_sentiment_agent_stores_news_in_vector_store(monkeypatch):
    monkeypatch.setattr(
        "pipelines.realtime.agents.create_sentiment_analyzer",
        lambda: StubAnalyzer(),
    )
    mock_vector_store = MagicMock()
    agent = SentimentAgent(vector_store=mock_vector_store)
    news_items = [{
        "headline": "Earnings beat expectations",
        "content": "Company reports strong growth",
        "date": "2025-10-28",
        "source": "Financial Times",
    }]
    agent.run(ticker="AAPL", news_data=news_items)
    mock_vector_store.add_news_articles.assert_called_once()
    args, kwargs = mock_vector_store.add_news_articles.call_args
    assert kwargs["ticker"] == "AAPL"
    assert "sentiment" in kwargs["articles"][0]

def test_sentiment_agent_finds_related_news(monkeypatch):
    monkeypatch.setattr(
        "pipelines.realtime.agents.create_sentiment_analyzer",
        lambda: StubAnalyzer(),
    )
    mock_vector_store = MagicMock()
    mock_vector_store.search_similar_news.return_value = [
        {"headline": "Old headline", "sentiment": 0.6},
        {"headline": "Older headline", "sentiment": 0.7},
    ]
    agent = SentimentAgent(vector_store=mock_vector_store)
    related = agent.find_related_news(query="Earnings", ticker="AAPL")
    mock_vector_store.search_similar_news.assert_called_once()
    assert len(related) == 2
    assert related[0]["headline"] == "Old headline"

def test_sentiment_agent_blends_historical_sentiment(monkeypatch):
    monkeypatch.setattr(
        "pipelines.realtime.agents.create_sentiment_analyzer",
        lambda: StubAnalyzer(),
    )
    mock_vector_store = MagicMock()
    mock_vector_store.add_news_articles.return_value = 1
    mock_vector_store.search_similar_news.return_value = [
        {"headline": "Old headline", "sentiment": 0.6},
        {"headline": "Older headline", "sentiment": 0.8},
    ]
    agent = SentimentAgent(vector_store=mock_vector_store)
    news_items = [{
        "headline": "Earnings beat expectations",
        "content": "Company reports strong growth",
        "date": "2025-10-28",
        "source": "Financial Times",
    }]
    result = agent.run(ticker="AAPL", news_data=news_items)
    # Blended score should be average of FinBERT (0.8) and historical (0.7)
    assert abs(result.score - 0.75) < 0.01
