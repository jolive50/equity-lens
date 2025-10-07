"""Unit tests covering agent fallback behaviour with deterministic stub LLMs."""
import os

import pytest
from langchain_core.runnables import RunnableLambda

from pipelines.realtime.agents import (
    PredictionAgent,
    SentimentAgent,
    SmartMoneyAgent,
    ExplanationAgent,
    PredictionResult,
    SentimentResult,
    SmartMoneyResult,
    build_openai_llm,
)


def _stub_llm(tag: str) -> RunnableLambda:
    """Return a deterministic LangChain runnable that mimics an LLM response."""
    # What: Provide a lightweight callable to satisfy the agent interfaces during tests.
    # Why: Tests must remain offline-friendly without inventing synthetic financial data.
    # How: Wrap a lambda that echoes the prompt with a tag so assertions can inspect output.
    # Data: Accepts runnable input dictionaries and emits a tagged string.
    return RunnableLambda(lambda payload: f"[{tag}] {payload}")


def test_prediction_agent_llm_only_returns_structured_result():
    """PredictionAgent should emit a structured PredictionResult when only LLM is available."""
    llm = _stub_llm("prediction")
    agent = PredictionAgent(llm, use_ml_model=False)

    market_snapshot = [{"close": 150.0, "volume": 1_000_000}]
    fundamentals = {"pe_ratio": 25.4, "revenue_growth": 0.08}

    result = agent.run(ticker="AAPL", market_data=market_snapshot, fundamentals=fundamentals)

    assert isinstance(result, PredictionResult)
    assert result.direction in {"up", "down", "neutral"}
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.narrative, str)


def test_sentiment_agent_llm_only_reports_summary():
    """SentimentAgent should summarise news when FinBERT is disabled."""
    llm = _stub_llm("sentiment")
    agent = SentimentAgent(llm, use_finbert=False)

    news_items = [{
        "title": "Earnings beat expectations",
        "content": "Company reports strong growth",
        "sentiment_score": 0.7,
        "source": "Financial Times",
        "timestamp": "2024-01-24T10:00:00Z",
    }]

    result = agent.run(ticker="AAPL", news_data=news_items)

    assert isinstance(result, SentimentResult)
    assert result.current in {"positive", "neutral", "negative"}
    assert isinstance(result.headlines, list)


def test_smart_money_agent_without_data_service_returns_placeholders():
    """SmartMoneyAgent should deliver a SmartMoneyResult even without live APIs."""
    llm = _stub_llm("smart-money")
    agent = SmartMoneyAgent(llm, use_data_service=False)

    result = agent.run(ticker="AAPL")

    assert isinstance(result, SmartMoneyResult)
    assert isinstance(result.institutions, dict)
    assert isinstance(result.insiders, dict)
    assert isinstance(result.congress, dict)


def test_explanation_agent_llm_output():
    """ExplanationAgent should compose a narrative using the stub LLM."""
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
    assert "AAPL" in narrative or "up" in narrative


def test_build_openai_llm_requires_api_key(monkeypatch):
    """Ensure build_openai_llm raises when OPENAI_API_KEY is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_openai_llm("gpt-4o-mini")


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Real OpenAI API key required for integration smoke test",
)
def test_build_openai_llm_with_real_key():
    """Ensure build_openai_llm constructs the LangChain client when a key is present."""
    llm = build_openai_llm("gpt-4o-mini")
    # What: Invoke with a minimal payload to confirm the object behaves like a Runnable.
    # Why: Avoid hitting the network while still validating interface conformance.
    # How: Send a short message and expect a response string (LangChain handles the network call).
    # Data: The call may hit the real API; we keep payload tiny to minimize cost.
    preview = llm.invoke({"messages": [{"content": "Ping"}]})
    assert preview is not None
