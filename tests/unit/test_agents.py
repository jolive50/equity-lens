"""Unit tests for StockSense agents."""
import pytest
from unittest.mock import Mock, patch
from pipelines.realtime.agents import (
    PredictionAgent,
    SentimentAgent,
    SmartMoneyAgent,
    ExplanationAgent,
    PredictionResult,
    SentimentResult,
    SmartMoneyResult,
    build_mock_llm,
    build_openai_llm,
)


class TestPredictionAgent:
    """Test cases for PredictionAgent."""
    
    def test_prediction_agent_initialization(self):
        """Test PredictionAgent initialization."""
        llm = build_mock_llm("test")
        agent = PredictionAgent(llm)
        assert agent is not None
        assert agent.use_ml_model is True
    
    def test_prediction_agent_without_ml(self):
        """Test PredictionAgent without ML model."""
        llm = build_mock_llm("test")
        agent = PredictionAgent(llm, use_ml_model=False)
        assert agent.use_ml_model is False
        assert agent.forecaster is None
    
    def test_prediction_agent_run_with_mock_data(self):
        """Test PredictionAgent run method with mock data."""
        llm = build_mock_llm("test")
        agent = PredictionAgent(llm, use_ml_model=False)
        
        market_data = {"price": 150.0, "volume": 1000000}
        fundamentals = {"revenue_growth": 0.08, "pe_ratio": 25.5}
        
        result = agent.run(
            ticker="AAPL",
            market_data=market_data,
            fundamentals=fundamentals
        )
        
        assert isinstance(result, PredictionResult)
        assert result.direction in ["up", "down", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.narrative, str)


class TestSentimentAgent:
    """Test cases for SentimentAgent."""
    
    def test_sentiment_agent_initialization(self):
        """Test SentimentAgent initialization."""
        llm = build_mock_llm("test")
        agent = SentimentAgent(llm)
        assert agent is not None
        assert agent.use_finbert is True
    
    def test_sentiment_agent_without_finbert(self):
        """Test SentimentAgent without FinBERT."""
        llm = build_mock_llm("test")
        agent = SentimentAgent(llm, use_finbert=False)
        assert agent.use_finbert is False
        assert agent.sentiment_analyzer is None
    
    def test_sentiment_agent_run_with_mock_data(self):
        """Test SentimentAgent run method with mock data."""
        llm = build_mock_llm("test")
        agent = SentimentAgent(llm, use_finbert=False)
        
        news_data = [
            {
                "title": "Strong earnings report",
                "content": "Company reports better than expected results",
                "sentiment_score": 0.8,
                "source": "Financial Times",
                "timestamp": "2024-01-24T10:00:00Z"
            }
        ]
        
        result = agent.run(ticker="AAPL", news_data=news_data)
        
        assert isinstance(result, SentimentResult)
        assert result.current in ["positive", "neutral", "negative"]
        assert 0.0 <= result.score <= 1.0
        assert result.trend in ["improving", "stable", "declining"]
        assert isinstance(result.headlines, list)


class TestSmartMoneyAgent:
    """Test cases for SmartMoneyAgent."""
    
    def test_smart_money_agent_initialization(self):
        """Test SmartMoneyAgent initialization."""
        llm = build_mock_llm("test")
        agent = SmartMoneyAgent(llm)
        assert agent is not None
        assert agent.use_data_service is True
    
    def test_smart_money_agent_without_data_service(self):
        """Test SmartMoneyAgent without data service."""
        llm = build_mock_llm("test")
        agent = SmartMoneyAgent(llm, use_data_service=False)
        assert agent.use_data_service is False
        assert agent.smart_money_service is None
    
    def test_smart_money_agent_run(self):
        """Test SmartMoneyAgent run method."""
        llm = build_mock_llm("test")
        agent = SmartMoneyAgent(llm, use_data_service=False)
        
        result = agent.run(ticker="AAPL")
        
        assert isinstance(result, SmartMoneyResult)
        assert isinstance(result.institutions, dict)
        assert isinstance(result.insiders, dict)
        assert isinstance(result.congress, dict)


class TestExplanationAgent:
    """Test cases for ExplanationAgent."""
    
    def test_explanation_agent_initialization(self):
        """Test ExplanationAgent initialization."""
        llm = build_mock_llm("test")
        agent = ExplanationAgent(llm)
        assert agent is not None
    
    def test_explanation_agent_run(self):
        """Test ExplanationAgent run method."""
        llm = build_mock_llm("test")
        agent = ExplanationAgent(llm)
        
        prediction = {"direction": "up", "confidence": 0.85}
        sentiment = {"current": "positive", "score": 0.7}
        smart_money = {"institutions": {"summary": "Net buying"}}
        
        result = agent.run(
            ticker="AAPL",
            prediction=prediction,
            sentiment=sentiment,
            smart_money=smart_money,
            user_tier="basic",
            confidence_level="high"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestLLMBuilders:
    """Test cases for LLM builder functions."""
    
    def test_build_mock_llm(self):
        """Test build_mock_llm function."""
        llm = build_mock_llm("test")
        assert llm is not None
        
        # Test that it returns a mock response
        result = llm.invoke({"messages": [{"content": "test prompt"}]})
        assert isinstance(result, str)
        assert "test" in result
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_build_openai_llm_with_key(self):
        """Test build_openai_llm with API key."""
        with patch('pipelines.realtime.agents.ChatOpenAI') as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            llm = build_openai_llm("gpt-4o-mini")
            
            assert llm is not None
            mock_chat.assert_called_once()
    
    def test_build_openai_llm_without_key(self):
        """Test build_openai_llm without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                build_openai_llm("gpt-4o-mini")


class TestAgentIntegration:
    """Integration tests for agents working together."""
    
    def test_agent_workflow_simulation(self):
        """Test a simulated workflow with all agents."""
        llm = build_mock_llm("integration-test")
        
        # Initialize all agents
        prediction_agent = PredictionAgent(llm, use_ml_model=False)
        sentiment_agent = SentimentAgent(llm, use_finbert=False)
        smart_money_agent = SmartMoneyAgent(llm, use_data_service=False)
        explanation_agent = ExplanationAgent(llm)
        
        # Simulate workflow data
        market_data = {"price": 150.0, "volume": 1000000}
        fundamentals = {"revenue_growth": 0.08, "pe_ratio": 25.5}
        news_data = [{"title": "Test news", "content": "Test content"}]
        
        # Run agents
        prediction_result = prediction_agent.run(
            ticker="AAPL",
            market_data=market_data,
            fundamentals=fundamentals
        )
        
        sentiment_result = sentiment_agent.run(
            ticker="AAPL",
            news_data=news_data
        )
        
        smart_money_result = smart_money_agent.run(ticker="AAPL")
        
        explanation = explanation_agent.run(
            ticker="AAPL",
            prediction=prediction_result.__dict__,
            sentiment=sentiment_result.__dict__,
            smart_money=smart_money_result.__dict__,
            user_tier="basic",
            confidence_level="high"
        )
        
        # Verify results
        assert isinstance(prediction_result, PredictionResult)
        assert isinstance(sentiment_result, SentimentResult)
        assert isinstance(smart_money_result, SmartMoneyResult)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
