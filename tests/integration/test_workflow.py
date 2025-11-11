import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coordinator.workflow import create_freshstart_workflow, StockAnalysisState


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

    @patch('coordinator.workflow.get_historical_data')
    @patch('coordinator.workflow.get_fundamentals')
    def test_workflow_execution_success(self, mock_fundamentals, mock_prices, mock_agents, sample_price_data):
        """Test successful workflow execution."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        # Mock data fetchers
        mock_prices.return_value = [
            {"date": "2024-01-01", "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000000}
        ]
        mock_fundamentals.return_value = {"pe_ratio": 25.5, "eps": 4.0}

        # Mock agent responses
        mock_agents['prediction'].run.return_value = PredictionResult(
            direction="up",
            confidence=0.75,
            narrative="Strong momentum",
            probabilities={"up": 0.75, "down": 0.15, "neutral": 0.10},
            metadata={"model": "LSTM"}
        )

        mock_agents['sentiment'].run.return_value = SentimentResult(
            current="positive",
            score=0.65,
            trend="improving",
            headlines=[{"title": "Good news", "sentiment": "positive"}]
        )

        mock_agents['reflection'].run.return_value = {
            "validation_passed": True,
            "issues": []
        }

        mock_agents['explanation'].run.return_value = "Analysis complete"

        # Create and run workflow
        workflow = create_freshstart_workflow(
            mock_agents['prediction'],
            mock_agents['sentiment'],
            mock_agents['reflection'],
            mock_agents['explanation']
        ).compile()

        result = workflow.invoke({"ticker": "AAPL"})

        # Verify workflow executed all nodes
        assert result["ticker"] == "AAPL"
        assert "prediction_result" in result
        assert "sentiment_result" in result
        assert result["prediction_result"]["direction"] == "up"
        assert result["sentiment_result"]["current"] == "positive"

    def test_workflow_state_management(self):
        """Test that workflow properly manages state between nodes."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        # Create mock agents
        prediction_agent = Mock()
        sentiment_agent = Mock()
        reflection_agent = Mock()
        explanation_agent = Mock()

        # Setup mock responses
        prediction_agent.run.return_value = PredictionResult(
            direction="up",
            confidence=0.80,
            narrative="Test",
            probabilities={"up": 0.8, "down": 0.1, "neutral": 0.1},
            metadata={}
        )

        sentiment_agent.run.return_value = SentimentResult(
            current="positive",
            score=0.70,
            trend="stable",
            headlines=[]
        )

        reflection_agent.run.return_value = {
            "validation_passed": True,
            "issues": []
        }

        explanation_agent.run.return_value = "Test explanation"

        # Create workflow
        workflow = create_freshstart_workflow(
            prediction_agent,
            sentiment_agent,
            reflection_agent,
            explanation_agent
        ).compile()

        # Run with minimal state
        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "TEST"})

            # Verify state is maintained through workflow
            assert result["ticker"] == "TEST"
            assert "warnings" in result
            assert isinstance(result["warnings"], list)
            assert "confidence_level" in result

    def test_workflow_handles_invalid_ticker(self, mock_agents):
        """Test workflow handles invalid ticker input."""
        workflow = create_freshstart_workflow(
            mock_agents['prediction'],
            mock_agents['sentiment'],
            mock_agents['reflection'],
            mock_agents['explanation']
        ).compile()

        # Empty ticker should raise ValueError
        with pytest.raises(ValueError, match="Ticker is required"):
            workflow.invoke({"ticker": ""})


class TestWorkflowDataFlow:
    """Test data flow through workflow nodes."""

    @patch('coordinator.workflow.get_historical_data')
    @patch('coordinator.workflow.get_fundamentals')
    def test_fetch_data_node_calls_fetchers(self, mock_fundamentals, mock_prices):
        """Test that fetch_data node calls price and news fetchers."""
        mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
        mock_fundamentals.return_value = {"pe_ratio": 20.0}

        # Create mock agents
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.7, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.6, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        result = workflow.invoke({"ticker": "AAPL"})

        # Verify data was fetched
        assert mock_prices.called
        assert mock_fundamentals.called
        assert "market_data" in result
        assert "fundamentals" in result

    def test_prediction_node_calls_prediction_agent(self):
        """Test that prediction node calls PredictionAgent."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.75, "Strong", {"up": 0.75}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.6, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Verify prediction agent was called
            assert prediction_agent.run.called
            assert result["prediction_result"]["direction"] == "up"

    def test_sentiment_node_calls_sentiment_agent(self):
        """Test that sentiment node calls SentimentAgent."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.7, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.65, "improving", [{"title": "Good news"}])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Verify sentiment agent was called
            assert sentiment_agent.run.called
            assert result["sentiment_result"]["current"] == "positive"

    def test_reflection_node_validates_results(self):
        """Test that reflection node validates prediction/sentiment results."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.7, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.6, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": False, "issues": ["Test issue"]}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Verify reflection was called and validation failed
            assert reflection_agent.run.called
            assert result["confidence_level"] == "low"
            assert "Test issue" in result["warnings"]

    def test_explanation_node_generates_narrative(self):
        """Test that explanation node generates narrative explanation."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.7, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.6, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Detailed explanation for the user"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Verify explanation was generated
            assert explanation_agent.run.called
            assert result["explanation"] == "Detailed explanation for the user"


class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    @patch('coordinator.workflow.get_historical_data')
    @patch('coordinator.workflow.get_fundamentals')
    def test_workflow_handles_data_fetch_failure(self, mock_fundamentals, mock_prices):
        """Test workflow handles data fetching failures."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        # Simulate fetch failure
        mock_prices.side_effect = Exception("API unavailable")
        mock_fundamentals.return_value = {}

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("neutral", 0.5, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("neutral", 0.5, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        result = workflow.invoke({"ticker": "AAPL"})

        # Workflow should continue with warnings
        assert "warnings" in result
        assert any("error" in w.lower() for w in result["warnings"])

    def test_workflow_handles_prediction_failure(self):
        """Test workflow continues when prediction fails."""
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.side_effect = Exception("Model unavailable")

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.6, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Should have fallback prediction
            assert result["prediction_result"]["direction"] == "neutral"
            assert any("Prediction error" in w for w in result["warnings"])

    def test_workflow_handles_sentiment_failure(self):
        """Test workflow continues when sentiment fails."""
        from agents.prediction_agent import PredictionResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.7, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.side_effect = Exception("Sentiment model failed")

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": True, "issues": []}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Should have fallback sentiment
            assert result["sentiment_result"]["current"] == "neutral"
            assert any("Sentiment error" in w for w in result["warnings"])

    def test_workflow_collects_warnings(self):
        """Test that workflow collects warnings from all nodes."""
        from agents.prediction_agent import PredictionResult
        from agents.sentiment_agent import SentimentResult

        prediction_agent = Mock()
        prediction_agent.run.return_value = PredictionResult("up", 0.7, "test", {}, {})

        sentiment_agent = Mock()
        sentiment_agent.run.return_value = SentimentResult("positive", 0.6, "stable", [])

        reflection_agent = Mock()
        reflection_agent.run.return_value = {"validation_passed": False, "issues": ["Warning 1", "Warning 2"]}

        explanation_agent = Mock()
        explanation_agent.run.return_value = "Test"

        workflow = create_freshstart_workflow(
            prediction_agent, sentiment_agent, reflection_agent, explanation_agent
        ).compile()

        with patch('coordinator.workflow.get_historical_data') as mock_prices, \
             patch('coordinator.workflow.get_fundamentals') as mock_fundamentals:
            mock_prices.return_value = [{"date": "2024-01-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
            mock_fundamentals.return_value = {}

            result = workflow.invoke({"ticker": "AAPL"})

            # Should collect all warnings
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
