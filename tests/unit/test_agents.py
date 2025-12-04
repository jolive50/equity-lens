import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.prediction_agent import PredictionAgent, PredictionAgentResult
from models.prediction.base_predictor import PredictionResult


@pytest.fixture
def sample_market_data():
    """Sample market data for agent testing."""
    return [
        {"date": "2024-11-01", "open": 175.0, "high": 178.0, "low": 174.0, "close": 177.0, "volume": 50000000},
        {"date": "2024-11-02", "open": 177.0, "high": 180.0, "low": 176.0, "close": 179.0, "volume": 52000000},
        {"date": "2024-11-03", "open": 179.0, "high": 182.0, "low": 178.0, "close": 181.0, "volume": 48000000},
    ]


@pytest.fixture
def sample_fundamentals():
    """Sample fundamental data."""
    return {
        "pe_ratio": 25.5,
        "forward_pe": 22.3,
        "market_cap": 2500000000000,
        "revenue_growth": 0.12,
        "profit_margin": 0.25,
        "beta": 1.2
    }


@pytest.fixture
def mock_prediction_model():
    """Mock prediction model for testing."""
    model = Mock()
    model.predict.return_value = PredictionResult(
        direction="up",
        confidence=0.75,
        probabilities={"up": 0.75, "down": 0.15, "neutral": 0.10},
        metadata={"model": "MockModel"}
    )
    model.get_model_info.return_value = {"name": "MockModel", "type": "test"}
    return model


class TestPredictionAgent:
    """Test suite for PredictionAgent (Josh's component)."""

    def test_prediction_agent_initialization_with_model(self, mock_prediction_model):
        """Test initializing PredictionAgent with a model."""
        agent = PredictionAgent(model=mock_prediction_model)
        assert agent.model is not None
        assert agent.model == mock_prediction_model

    def test_prediction_agent_initialization_without_model(self):
        """Test initializing PredictionAgent without a model."""
        agent = PredictionAgent()
        assert agent.model is None

    def test_prediction_agent_run(self, mock_prediction_model, sample_market_data, sample_fundamentals):
        """Test running PredictionAgent."""
        agent = PredictionAgent(model=mock_prediction_model)
        result = agent.run("AAPL", sample_market_data, sample_fundamentals)

        assert isinstance(result, PredictionAgentResult)
        assert result.direction in ["up", "down", "neutral"]
        assert 0 <= result.confidence <= 1
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 0

    def test_prediction_agent_result_fields(self, mock_prediction_model, sample_market_data, sample_fundamentals):
        """Test that PredictionAgent result has all required fields."""
        agent = PredictionAgent(model=mock_prediction_model)
        result = agent.run("AAPL", sample_market_data, sample_fundamentals)

        assert hasattr(result, 'direction')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'narrative')
        assert hasattr(result, 'probabilities')
        assert hasattr(result, 'metadata')

    def test_prediction_agent_calls_model(self, mock_prediction_model, sample_market_data, sample_fundamentals):
        """Test that PredictionAgent calls the model's predict method."""
        agent = PredictionAgent(model=mock_prediction_model)
        agent.run("AAPL", sample_market_data, sample_fundamentals)

        mock_prediction_model.predict.assert_called_once()


class TestSentimentAgent:
    """Test suite for SentimentAgent (Josh's component)."""

    def test_sentiment_agent_placeholder(self):
        """Placeholder for SentimentAgent tests."""
        pytest.skip("Will implement when testing sentiment agent integration")


class TestReflectionAgent:
    """Test suite for ReflectionAgent (Josh's component)."""

    def test_reflection_agent_placeholder(self):
        """Placeholder for ReflectionAgent tests."""
        pytest.skip("Will implement when testing reflection agent")

    def test_reflection_agent_validates_predictions(self):
        """Test that ReflectionAgent validates prediction results."""
        pytest.skip("Will implement when testing reflection agent validation")

    def test_reflection_agent_compares_to_actual(self):
        """Test that ReflectionAgent can compare predictions to actual results."""
        pytest.skip("Will implement when testing reflection agent comparison")


class TestExplanationAgent:
    """Test suite for ExplanationAgent (Josh's component)."""

    def test_explanation_agent_placeholder(self):
        """Placeholder for ExplanationAgent tests."""
        pytest.skip("Will implement when testing explanation agent")

    def test_explanation_agent_uses_vector_store(self):
        """Test that ExplanationAgent uses ChromaDB for context."""
        pytest.skip("Will implement when testing vector store integration")

    def test_explanation_agent_generates_narrative(self):
        """Test that ExplanationAgent generates natural language explanations."""
        pytest.skip("Will implement when testing explanation generation")


class TestAgentInteractions:
    """Test interactions between multiple agents."""

    def test_prediction_and_sentiment_agents_together(self):
        """Test using both prediction and sentiment agents."""
        pytest.skip("Will implement for integration testing")

    def test_full_agent_pipeline(self):
        """Test complete agent pipeline: predict -> sentiment -> reflect -> explain."""
        pytest.skip("Will implement for integration testing")


class TestAgentErrorHandling:
    """Test error handling in agents."""

    def test_prediction_agent_handles_empty_data(self, mock_prediction_model, sample_fundamentals):
        """Test PredictionAgent handles empty market data."""
        agent = PredictionAgent(model=mock_prediction_model)

        with pytest.raises(Exception):
            agent.run("AAPL", [], sample_fundamentals)

    def test_prediction_agent_handles_invalid_ticker(self, mock_prediction_model, sample_market_data, sample_fundamentals):
        """Test PredictionAgent handles invalid ticker."""
        agent = PredictionAgent(model=mock_prediction_model)

        result = agent.run("", sample_market_data, sample_fundamentals)
        assert result is not None

    def test_prediction_agent_handles_model_failure(self, sample_market_data, sample_fundamentals):
        """Test PredictionAgent handles model failures gracefully."""
        failing_model = Mock()
        failing_model.predict.side_effect = Exception("Model failed")

        agent = PredictionAgent(model=failing_model)

        with pytest.raises(Exception):
            agent.run("AAPL", sample_market_data, sample_fundamentals)


class TestAgentConfiguration:
    """Test agent configuration and model selection."""

    def test_prediction_agent_accepts_different_models(self):
        """Test that PredictionAgent can work with different model types."""
        pytest.skip("Will implement when testing model configuration system")

    def test_agents_use_config_for_model_selection(self):
        """Test that agents use config.py for model selection."""
        pytest.skip("Will implement when testing configuration system")
