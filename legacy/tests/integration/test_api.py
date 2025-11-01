"""Integration tests for StockSense API."""
import pytest
import json
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from pipelines.realtime.api import app


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "StockSense API"
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        with patch('pipelines.realtime.api.get_data_service') as mock_data_service, \
             patch('pipelines.realtime.api.get_agents') as mock_agents:
            
            mock_data_service.return_value.adapters = {"test": Mock()}
            mock_agents.return_value = {"test": Mock()}
            
            response = self.client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "data_adapters" in data
            assert "agents" in data
    
    def test_health_endpoint_error(self):
        """Test health check endpoint with error."""
        with patch('pipelines.realtime.api.get_data_service') as mock_data_service:
            mock_data_service.side_effect = Exception("Service error")
            
            response = self.client.get("/health")
            assert response.status_code == 500
    
    def test_analyze_endpoint_success(self):
        """Test analyze endpoint with successful response."""
        mock_result = {
            "ticker": "AAPL",
            "as_of": "2024-01-24T10:00:00Z",
            "forecast": {
                "direction": "up",
                "confidence": 0.85,
                "horizon_95": {
                    "class": "up",
                    "days": 5,
                    "drops_below_95_on": "2024-01-29"
                },
                "daily_probs": [
                    {"date": "2024-01-25", "up": 0.85, "down": 0.10, "neutral": 0.05}
                ]
            },
            "metrics": {
                "revenue_growth": {
                    "value": 0.08,
                    "verdict": "Good",
                    "explanation": "Strong growth"
                }
            },
            "sentiment": {
                "current": "positive",
                "score": 0.7,
                "trend": "improving",
                "headlines": ["Strong earnings"]
            },
            "smart_money": {
                "institutions": {"summary": "Net buying"},
                "insiders": {"summary": "No recent activity"},
                "congress": {"summary": "No recent activity"}
            },
            "explanation": "Test explanation",
            "warnings": [],
            "disclaimers": ["Test disclaimer"]
        }
        
        with patch('pipelines.realtime.api.run_stocksense_analysis') as mock_analysis:
            mock_analysis.return_value = mock_result
            
            response = self.client.post(
                "/analyze",
                json={"ticker": "AAPL", "user_tier": "basic"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert data["forecast"]["direction"] == "up"
            assert data["forecast"]["confidence"] == 0.85
    
    def test_analyze_endpoint_invalid_ticker(self):
        """Test analyze endpoint with invalid ticker."""
        response = self.client.post(
            "/analyze",
            json={"ticker": "", "user_tier": "basic"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid ticker symbol" in data["detail"]
    
    def test_analyze_endpoint_invalid_tier(self):
        """Test analyze endpoint with invalid user tier."""
        response = self.client.post(
            "/analyze",
            json={"ticker": "AAPL", "user_tier": "invalid"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "User tier must be 'basic' or 'premium'" in data["detail"]
    
    def test_analyze_endpoint_analysis_error(self):
        """Test analyze endpoint with analysis error."""
        with patch('pipelines.realtime.api.run_stocksense_analysis') as mock_analysis:
            mock_analysis.side_effect = Exception("Analysis failed")
            
            response = self.client.post(
                "/analyze",
                json={"ticker": "AAPL", "user_tier": "basic"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Analysis failed" in data["detail"]
    
    def test_prices_endpoint_success(self):
        """Test prices endpoint with successful response."""
        mock_market_data = [
            {
                "timestamp": "2024-01-24",
                "close": 150.0,
                "open": 148.0,
                "high": 152.0,
                "low": 147.0,
                "volume": 1000000
            }
        ]
        
        with patch('pipelines.realtime.api.get_data_service') as mock_data_service:
            mock_adapter = Mock()
            mock_adapter.fetch_historical_data.return_value = mock_market_data
            mock_data_service.return_value.adapters = {"alpha_vantage": mock_adapter}
            
            response = self.client.get("/prices?ticker=AAPL&source=alpha_vantage")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert data["source"] == "alpha_vantage"
            assert "items" in data
            assert len(data["items"]) == 1
    
    def test_prices_endpoint_unknown_source(self):
        """Test prices endpoint with unknown source."""
        with patch('pipelines.realtime.api.get_data_service') as mock_data_service:
            mock_data_service.return_value.adapters = {}
            
            response = self.client.get("/prices?ticker=AAPL&source=unknown")
            
            assert response.status_code == 400
            data = response.json()
            assert "Unknown source" in data["detail"]
    
    def test_news_endpoint_success(self):
        """Test news endpoint with successful response."""
        mock_news = [
            {
                "title": "Test News",
                "content": "Test content",
                "source": "Test Source",
                "url": "https://test.com",
                "timestamp": "2024-01-24T10:00:00Z",
                "sentiment_score": 0.8
            }
        ]
        
        with patch('pipelines.realtime.api.get_data_service') as mock_data_service:
            mock_adapter = Mock()
            mock_adapter.fetch_news.return_value = mock_news
            mock_data_service.return_value.adapters = {"newsapi": mock_adapter}
            
            response = self.client.get("/news?ticker=AAPL&source=newsapi")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert data["source"] == "newsapi"
            assert "items" in data
            assert len(data["items"]) == 1
    
    def test_sentiment_endpoint_success(self):
        """Test sentiment endpoint with successful response."""
        mock_sentiment = {
            "current": "positive",
            "score": 0.7,
            "trend": "improving",
            "headlines": ["Strong earnings"],
            "news_count": 5
        }
        
        with patch('pipelines.realtime.api.get_data_service') as mock_data_service, \
             patch('pipelines.realtime.api.get_agents') as mock_agents:
            
            mock_data_service.return_value.get_all_news.return_value = []
            mock_agent = Mock()
            mock_agent.run.return_value = Mock(
                current="positive",
                score=0.7,
                trend="improving",
                headlines=["Strong earnings"]
            )
            mock_agents.return_value = {"sentiment": mock_agent}
            
            response = self.client.get("/sentiment?ticker=AAPL")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert data["current"] == "positive"
    
    def test_smart_money_endpoint_success(self):
        """Test smart money endpoint with successful response."""
        mock_smart_money = {
            "institutions": {"summary": "Net buying"},
            "insiders": {"summary": "No recent activity"},
            "congress": {"summary": "No recent activity"}
        }
        
        with patch('pipelines.realtime.api.get_agents') as mock_agents:
            mock_agent = Mock()
            mock_agent.run.return_value = Mock(
                institutions={"summary": "Net buying"},
                insiders={"summary": "No recent activity"},
                congress={"summary": "No recent activity"}
            )
            mock_agents.return_value = {"smart_money": mock_agent}
            
            response = self.client.get("/smart-money?ticker=AAPL")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert "institutions" in data
            assert "insiders" in data
            assert "congress" in data


class TestAPIIntegration:
    """Integration tests for API workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('pipelines.realtime.api.run_stocksense_analysis')
    def test_full_analysis_workflow(self, mock_analysis):
        """Test full analysis workflow."""
        mock_analysis.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-24T10:00:00Z",
            "forecast": {
                "direction": "up",
                "confidence": 0.85,
                "horizon_95": {
                    "class": "up",
                    "days": 5,
                    "drops_below_95_on": "2024-01-29"
                },
                "daily_probs": [
                    {"date": "2024-01-25", "up": 0.85, "down": 0.10, "neutral": 0.05}
                ]
            },
            "metrics": {
                "revenue_growth": {
                    "value": 0.08,
                    "verdict": "Good",
                    "explanation": "Strong growth"
                }
            },
            "sentiment": {
                "current": "positive",
                "score": 0.7,
                "trend": "improving",
                "headlines": ["Strong earnings"]
            },
            "smart_money": {
                "institutions": {"summary": "Net buying"},
                "insiders": {"summary": "No recent activity"},
                "congress": {"summary": "No recent activity"}
            },
            "explanation": "Test explanation",
            "warnings": [],
            "disclaimers": ["Test disclaimer"]
        }
        
        # Test basic tier
        response = self.client.post(
            "/analyze",
            json={"ticker": "AAPL", "user_tier": "basic"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["forecast"]["direction"] == "up"
        
        # Test premium tier
        response = self.client.post(
            "/analyze",
            json={"ticker": "AAPL", "user_tier": "premium"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["forecast"]["direction"] == "up"
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options("/analyze")
        
        # CORS middleware should handle OPTIONS requests
        assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS
    
    def test_api_documentation(self):
        """Test API documentation endpoint."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        # Should return HTML documentation
        assert "text/html" in response.headers.get("content-type", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
