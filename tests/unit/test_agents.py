"""Unit tests for deterministic agent behaviours (no LLM fallbacks)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from pipelines.realtime.agents import (
    ExplanationAgent,
    PredictionAgent,
    SentimentAgent,
    SmartMoneyAgent,
)
from pipelines.realtime.models.forecaster import ForecastResult


def _stub_llm(tag: str) -> RunnableLambda:
    """Return a deterministic runnable that echoes payloads with a tag."""
    return RunnableLambda(lambda payload: f"[{tag}] {payload}")


def test_prediction_agent_uses_ml_forecaster(monkeypatch):
    """PredictionAgent must rely entirely on the gradient boosting forecaster."""
    agent = PredictionAgent()

    stub_result = ForecastResult(
        direction="up",
        confidence=0.87,
        daily_probs=[{"day": 1, "up": 0.8, "down": 0.1, "neutral": 0.1}],
        horizon_95={"days": 3},
        feature_importance={"momentum": 0.6, "volume_trend": 0.4},
        model_metadata={"probabilities": {"up": 0.87, "down": 0.05, "neutral": 0.08}},
    )
    agent.forecaster = MagicMock()
    agent.forecaster.predict.return_value = stub_result

    market_snapshot = [{"close": 150.0, "volume": 1_000_000} for _ in range(60)]
    fundamentals = {"pe_ratio": 25.4, "revenue_growth": 0.08}

    result = agent.run(ticker="AAPL", market_data=market_snapshot, fundamentals=fundamentals)

    agent.forecaster.predict.assert_called_once()
    assert result.direction == "up"
    assert result.confidence == pytest.approx(0.87)
    assert "Probability distribution" in result.narrative


def test_prediction_agent_raises_when_forecaster_fails(monkeypatch):
    """If the ML forecaster raises, the agent must propagate a RuntimeError."""
    agent = PredictionAgent()
    agent.forecaster = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="Gradient boosting forecaster failed"):
        agent.run(ticker="MSFT", market_data=[{"close": 120, "volume": 5_000}], fundamentals={})


def test_sentiment_agent_uses_finbert(monkeypatch):
    """SentimentAgent should call FinBERT and surface its structured response."""

    class StubAnalyzer:
        def process_news_articles(self, articles):
            return {
                "current": "positive",
                "score": 0.78,
                "trend": "improving",
                "headlines": ["Stub headline"],
            }

    monkeypatch.setattr(
        "pipelines.realtime.agents.create_sentiment_analyzer",
        lambda: StubAnalyzer(),
    )

    agent = SentimentAgent()
    news_items = [{
        "title": "Earnings beat expectations",
        "content": "Company reports strong growth",
        "timestamp": "2024-01-24T10:00:00Z",
        "source": "Financial Times",
    }]

    result = agent.run(ticker="AAPL", news_data=news_items)

    assert result.current == "positive"
    assert result.trend == "improving"
    assert result.headlines == ["Stub headline"]


def test_sentiment_agent_raises_when_finbert_missing(monkeypatch):
    """Instantiating SentimentAgent without FinBERT assets should fail fast."""

    def _raise():
        raise ImportError("no finbert")

    monkeypatch.setattr(
        "pipelines.realtime.agents.create_sentiment_analyzer",
        _raise,
    )

    with pytest.raises(RuntimeError, match="FinBERT dependencies are missing"):
        SentimentAgent()


def test_smart_money_agent_uses_data_service(monkeypatch):
    """SmartMoneyAgent must delegate to the configured data service."""

    class StubSmartMoneyService:
        def get_institutional_summary(self, ticker: str):
            return {"summary": f"{ticker} institutional"}

        def get_insider_summary(self, ticker: str):
            return {"summary": f"{ticker} insiders"}

        def get_congressional_summary(self, ticker: str):
            return {"summary": f"{ticker} congress"}

    monkeypatch.setattr(
        "pipelines.realtime.agents.create_smart_money_service",
        lambda api_keys: StubSmartMoneyService(),
    )
    monkeypatch.setattr(
        "pipelines.realtime.agents.get_available_api_keys",
        lambda *args: {"alpha_vantage": "x"},
    )

    agent = SmartMoneyAgent()
    result = agent.run(ticker="TSLA")

    assert result.institutions["summary"] == "TSLA institutional"
    assert result.insiders["summary"] == "TSLA insiders"
    assert result.congress["summary"] == "TSLA congress"


def test_smart_money_agent_raises_on_service_failure(monkeypatch):
    """If the underlying service raises, the agent should bubble up the error."""

    class FailingService:
        def get_institutional_summary(self, ticker: str):
            raise RuntimeError("service down")

    monkeypatch.setattr(
        "pipelines.realtime.agents.create_smart_money_service",
        lambda api_keys: FailingService(),
    )
    monkeypatch.setattr(
        "pipelines.realtime.agents.get_available_api_keys",
        lambda *args: {"alpha_vantage": "x"},
    )

    agent = SmartMoneyAgent()
    with pytest.raises(RuntimeError, match="Smart money service failed"):
        agent.run(ticker="TSLA")


def test_explanation_agent_llm_output():
    """ExplanationAgent still relies on LLM generation for narratives."""
    llm = _stub_llm("explanation")
    agent = ExplanationAgent(llm)

    narrative = agent.run(
        ticker="AAPL",
        prediction={"direction": "up", "confidence": 0.8},
        sentiment={"current": "positive", "score": 0.7},
        smart_money={"institutions": {"summary": "Net buying"}},
        user_tier="basic",
        confidence_level="high",
    )

    assert isinstance(narrative, str)
    assert "[explanation]" in narrative
