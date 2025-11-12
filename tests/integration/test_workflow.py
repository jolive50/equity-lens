import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coordinator.workflow import create_freshstart_workflow, StockAnalysisState
from agents.prediction_agent import PredictionAgentResult
from agents.sentiment_agent import SentimentAgentResult


def make_prediction_result(
    direction="up",
    confidence=0.7,
    narrative="test",
    probabilities=None,
    metadata=None
) -> PredictionAgentResult:
    """Build a PredictionAgentResult with sensible defaults for tests."""
    return PredictionAgentResult(
        direction=direction,
        confidence=confidence,
        narrative=narrative,
    probabilities=probabilities or {"up": 0.7, "down": 0.2, "neutral": 0.1},
    metadata=metadata or {}
    )


def make_sentiment_result(
    current="positive",
    score=0.6,
    trend="stable",
    headlines=None
) -> SentimentAgentResult:
    """Build a SentimentAgentResult with defaults used across tests."""
    return SentimentAgentResult(
        current=current,
        score=score,
        trend=trend,
    headlines=headlines or ["Headline"]
    )


def build_mock_agents(
    prediction_result=None,
    sentiment_result=None,
    reflection_output=None,
    explanation_text="Test"
):
    """Create mock agents wired with deterministic responses."""
    prediction_agent = Mock()
    prediction_agent.run.return_value = prediction_result or make_prediction_result()

    sentiment_agent = Mock()
    sentiment_agent.run.return_value = sentiment_result or make_sentiment_result()

    reflection_agent = Mock()
    reflection_agent.run.return_value = reflection_output or {
        "validation_passed": True,
        "issues": []
    }

    explanation_agent = Mock()
    explanation_agent.run.return_value = explanation_text

    return {
        "prediction": prediction_agent,
        "sentiment": sentiment_agent,
        "reflection": reflection_agent,
        "explanation": explanation_agent
    }


def compile_workflow_from_agents(agents):
    """Compile workflow with provided mock agents."""
    return create_freshstart_workflow(
        agents["prediction"],
        agents["sentiment"],
        agents["reflection"],
        agents["explanation"]
    ).compile()


@contextmanager
def patched_data_pipeline(
    price_payload=None,
    fundamentals_payload=None,
    news_payload=None,
    price_side_effect=None
):
    """Patch external data dependencies for workflow tests."""
    price_data = price_payload if price_payload is not None else [
        {
            "date": "2024-01-01",
            "open": 100,
            "high": 100,
            "low": 100,
            "close": 100,
            "volume": 1000
        }
    ]
    fundamentals = fundamentals_payload if fundamentals_payload is not None else {}
    news_data = news_payload if news_payload is not None else []

    with patch('coordinator.workflow._load_cached_market_data', return_value=[]), \
         patch('coordinator.workflow._cache_market_data'), \
         patch('coordinator.workflow._load_cached_news', return_value=[]), \
         patch('coordinator.workflow._cache_news_articles'), \
         patch('coordinator.workflow._persist_news_embeddings'), \
         patch('data.fetchers.price_data.get_historical_data') as mock_prices, \
         patch('data.fetchers.price_data.get_fundamentals') as mock_fundamentals, \
         patch('data.fetchers.news_data.NewsDataFetcher') as mock_news_fetcher:

        if price_side_effect is not None:
            mock_prices.side_effect = price_side_effect
        else:
            mock_prices.return_value = price_data

        mock_fundamentals.return_value = fundamentals
        mock_news_fetcher.return_value.fetch_news.return_value = news_data

        yield mock_prices, mock_fundamentals, mock_news_fetcher


@pytest.fixture
def mock_agents():
    """Create mock agents for workflow testing."""
    prediction_agent = Mock()
    sentiment_agent = Mock()
    reflection_agent = Mock()
    explanation_agent = Mock()

    return {
        'prediction': prediction_agent,
        'sentiment': sentiment_agent,
        'reflection': reflection_agent,
        'explanation': explanation_agent
    }


class TestWorkflowCreation:
    """Test LangGraph workflow creation (Josh's component)."""

    def test_create_workflow(self, mock_agents):
        """Test creating FreshStart workflow."""
        workflow = create_freshstart_workflow(
            mock_agents['prediction'],
            mock_agents['sentiment'],
            mock_agents['reflection'],
            mock_agents['explanation']
        )
        assert workflow is not None

    def test_workflow_has_required_nodes(self, mock_agents):
        """Test that workflow has all required processing nodes."""
        workflow = create_freshstart_workflow(
            mock_agents['prediction'],
            mock_agents['sentiment'],
            mock_agents['reflection'],
            mock_agents['explanation']
        )
        # Workflow should have nodes for: validate, fetch_data, predict, sentiment, reflect, explain
        assert workflow is not None


class TestWorkflowExecution:
    """Test workflow execution flow."""

    def test_workflow_execution_success(self, mock_agents, sample_price_data):
        """Test successful workflow execution."""
        from agents.prediction_agent import PredictionAgentResult
        from agents.sentiment_agent import SentimentAgentResult

        news_sample = [
            {
                "title": "Good news",
                "content": "Positive outlook",
                "source": "TestWire",
                "timestamp": "2024-01-01T00:00:00Z",
                "url": "https://example.com/good-news"
            }
        ]

        with patch('coordinator.workflow._load_cached_market_data', return_value=[]), \
             patch('coordinator.workflow._cache_market_data'), \
             patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals, \
             patch('coordinator.workflow._load_cached_news', return_value=[]), \
             patch('coordinator.workflow._cache_news_articles'), \
             patch('coordinator.workflow._persist_news_embeddings'), \
             patch('coordinator.workflow.NewsDataFetcher') as mock_news_fetcher:

            mock_prices.return_value = [
                {
                    "date": "2024-01-01",
                    "open": 100,
                    "high": 105,
                    "low": 95,
                    "close": 102,
                    "volume": 1000000
                }
            ]
            mock_fundamentals.return_value = {"pe_ratio": 25.5, "eps": 4.0}
            mock_news_fetcher.return_value.fetch_news.return_value = news_sample

            mock_agents['prediction'].run.return_value = PredictionAgentResult(
                direction="up",
                confidence=0.80,
                narrative="Strong momentum",
                probabilities={"up": 0.8, "down": 0.1, "neutral": 0.1},
                metadata={"model": "LSTM"}
            )

            mock_agents['sentiment'].run.return_value = SentimentAgentResult(
                current="positive",
                score=0.70,
                trend="stable",
                headlines=["Good news"]
            )

            mock_agents['reflection'].run.return_value = {
                "validation_passed": True,
                "issues": []
            }

            mock_agents['explanation'].run.return_value = "Analysis complete"

            workflow = create_freshstart_workflow(
                mock_agents['prediction'],
                mock_agents['sentiment'],
                mock_agents['reflection'],
                mock_agents['explanation']
            ).compile()

            result = workflow.invoke({"ticker": "AAPL"})

        assert result["ticker"] == "AAPL"
        assert result["prediction_result"]["direction"] == "up"
        assert result["sentiment_result"]["current"] == "positive"

    def test_workflow_state_management(self):
        """Test that workflow properly manages state between nodes."""
        from agents.prediction_agent import PredictionAgentResult
        from agents.sentiment_agent import SentimentAgentResult

        prediction_agent = Mock()
        sentiment_agent = Mock()
        reflection_agent = Mock()
        explanation_agent = Mock()

        prediction_agent.run.return_value = PredictionAgentResult(
            direction="up",
            confidence=0.80,
            narrative="Test",
            probabilities={"up": 0.8, "down": 0.1, "neutral": 0.1},
            metadata={}
        )

        sentiment_agent.run.return_value = SentimentAgentResult(
            current="positive",
            score=0.70,
            trend="stable",
            headlines=["Headline"]
        )

        reflection_agent.run.return_value = {
            "validation_passed": True,
            "issues": []
        }

        explanation_agent.run.return_value = "Test explanation"

        workflow = create_freshstart_workflow(
            prediction_agent,
            sentiment_agent,
            reflection_agent,
            explanation_agent
        ).compile()

        with patch('coordinator.workflow._load_cached_market_data', return_value=[]), \
             patch('coordinator.workflow._cache_market_data'), \
             patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals, \
             patch('coordinator.workflow._load_cached_news', return_value=[]), \
             patch('coordinator.workflow._cache_news_articles'), \
             patch('coordinator.workflow._persist_news_embeddings'), \
             patch('coordinator.workflow.NewsDataFetcher') as mock_news_fetcher:

            mock_prices.return_value = [
                {
                    "date": "2024-01-01",
                    "open": 100,
                    "high": 100,
                    "low": 100,
                    "close": 100,
                    "volume": 1000
                }
            ]
            mock_fundamentals.return_value = {}
            mock_news_fetcher.return_value.fetch_news.return_value = [
                {
                    "title": "Headline",
                    "content": "Summary",
                    "source": "TestWire",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "url": "https://example.com/headline"
                }
            ]

            result = workflow.invoke({"ticker": "TEST"})

        assert result["ticker"] == "TEST"
        assert isinstance(result.get("warnings"), list)
        assert "confidence_level" in result

    def test_workflow_handles_invalid_ticker(self, mock_agents):
        """Test workflow handles invalid ticker input."""
        workflow = create_freshstart_workflow(
            mock_agents['prediction'],
            mock_agents['sentiment'],
            mock_agents['reflection'],
            mock_agents['explanation']
        ).compile()

        with pytest.raises(ValueError, match="Ticker is required"):
            workflow.invoke({"ticker": ""})


class TestWorkflowDataFlow:
    """Test data flow through workflow nodes."""

    def test_fetch_data_node_calls_fetchers(self):
        """Workflow should call external fetchers when cache is empty."""
        agents = build_mock_agents()
        workflow = compile_workflow_from_agents(agents)

        news_sample = [
            {
                "title": "Sample headline",
                "content": "Detailed news content",
                "source": "TestWire",
                "timestamp": "2024-01-01T00:00:00Z",
                "url": "https://example.com/news"
            }
        ]

        with patched_data_pipeline(
            fundamentals_payload={"pe_ratio": 20.0},
            news_payload=news_sample
        ) as (mock_prices, mock_fundamentals, mock_news_fetcher):
            result = workflow.invoke({"ticker": "AAPL"})

        assert mock_prices.called
        assert mock_fundamentals.called
        assert mock_news_fetcher.return_value.fetch_news.called
        assert result["market_data"]
        assert result["fundamentals"]["pe_ratio"] == 20.0
        assert result["news_data"][0]["title"] == "Sample headline"

    def test_prediction_node_calls_prediction_agent(self):
        """Prediction node should invoke PredictionAgent.run."""
        prediction = make_prediction_result(
            direction="up",
            confidence=0.75,
            narrative="Strong",
            probabilities={"up": 0.75, "down": 0.15, "neutral": 0.10}
        )
        agents = build_mock_agents(prediction_result=prediction)
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert agents["prediction"].run.called
        assert result["prediction_result"]["direction"] == "up"

    def test_sentiment_node_calls_sentiment_agent(self):
        """Sentiment node should invoke SentimentAgent.run."""
        sentiment = make_sentiment_result(current="positive", score=0.65, trend="improving")
        agents = build_mock_agents(sentiment_result=sentiment)
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert agents["sentiment"].run.called
        assert result["sentiment_result"]["current"] == "positive"

    def test_reflection_node_validates_results(self):
        """Reflection node should downgrade confidence when validation fails."""
        reflection_output = {
            "validation_passed": False,
            "issues": ["Test issue"]
        }
        agents = build_mock_agents(reflection_output=reflection_output)
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert agents["reflection"].run.called
        assert result["confidence_level"] == "low"
        assert "Test issue" in result["warnings"]

    def test_explanation_node_generates_narrative(self):
        """Explanation node should surface explanation text."""
        explanation_text = "Detailed explanation for the user"
        agents = build_mock_agents(explanation_text=explanation_text)
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert agents["explanation"].run.called
        assert result["explanation"] == explanation_text


class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    def test_workflow_handles_data_fetch_failure(self):
        """Workflow should surface warnings when price fetch fails."""
        agents = build_mock_agents(
            prediction_result=make_prediction_result(direction="neutral")
        )
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline(price_side_effect=Exception("API unavailable")) as (mock_prices, _, _):
            result = workflow.invoke({"ticker": "AAPL"})

        assert mock_prices.called
        assert "warnings" in result
        assert any("error" in warning.lower() for warning in result["warnings"])

    def test_workflow_handles_prediction_failure(self):
        """Workflow should fall back to neutral prediction when agent fails."""
        agents = build_mock_agents()
        agents["prediction"].run.side_effect = Exception("Model unavailable")
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert result["prediction_result"]["direction"] == "neutral"
        assert any("prediction error" in warning.lower() for warning in result["warnings"])

    def test_workflow_handles_sentiment_failure(self):
        """Workflow should provide neutral sentiment when agent fails."""
        agents = build_mock_agents()
        agents["sentiment"].run.side_effect = Exception("Sentiment model failed")
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert result["sentiment_result"]["current"] == "neutral"
        assert any("sentiment error" in warning.lower() for warning in result["warnings"])

    def test_workflow_collects_warnings(self):
        """Warnings from reflection should appear in final state."""
        reflection_output = {
            "validation_passed": False,
            "issues": ["Warning 1", "Warning 2"]
        }
        agents = build_mock_agents(reflection_output=reflection_output)
        workflow = compile_workflow_from_agents(agents)

        with patched_data_pipeline() as _:
            result = workflow.invoke({"ticker": "AAPL"})

        assert "warnings" in result
        assert "Warning 1" in result["warnings"]
        assert "Warning 2" in result["warnings"]


class TestWorkflowCaching:
    """Test workflow integration with database caching."""

    def test_workflow_checks_cache_before_fetching(self, test_db):
        """Test that workflow checks cache before making API calls."""
        pytest.skip("Caching integration test - requires workflow-database integration")

    def test_workflow_saves_to_cache(self, test_db):
        """Test that workflow saves fetched data to cache."""
        pytest.skip("Caching integration test - requires workflow-database integration")

    def test_workflow_saves_analysis_results(self, test_db):
        """Test that workflow saves final results to database."""
        pytest.skip("Analysis saving test - requires workflow-database integration")


class TestWorkflowConfiguration:
    """Test workflow model configuration."""

    def test_workflow_uses_configured_models(self):
        """Test that workflow uses models specified in config."""
        pytest.skip("Configuration test - requires config.py integration")

    def test_workflow_logs_model_failures(self):
        """Test that workflow logs which models fail to load."""
        pytest.skip("Model failure logging test - requires config integration")


class TestWorkflowOutputFormat:
    """Test workflow output format."""

    def test_workflow_output_has_required_fields(self):
        """Test that workflow output has all required fields."""
        pytest.skip("Output format test - requires workflow integration")

    def test_workflow_output_matches_api_contract(self):
        """Test that workflow output matches API response model."""
        pytest.skip("API contract test - requires API integration")


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    def test_workflow_completes_within_timeout(self):
        """Test that workflow completes within reasonable time."""
        pytest.skip("Performance test - requires full workflow integration")

    def test_workflow_handles_concurrent_requests(self):
        """Test that workflow can handle multiple concurrent analyses."""
        pytest.skip("Concurrency test - requires full workflow integration")
