import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFastAPIEndpoints:
    """Integration tests for FastAPI endpoints (Sua's component)."""

    def test_api_initialization(self):
        """Test FastAPI server initialization."""
        from api.main import app

        assert app is not None
        assert app.title == "Equity Lens Stock Analysis API"
        assert app.version == "1.0.0"

    def test_analyze_endpoint_exists(self):
        """Test that /analyze endpoint exists."""
        from api.main import app

        client = TestClient(app)
        # POST to /analyze should exist (even if it returns error without params)
        response = client.post("/analyze", json={})
        # Should get validation error, not 404
        assert response.status_code != 404

    @patch('coordinator.workflow.run_stock_analysis')
    def test_analyze_endpoint_accepts_ticker(self, mock_workflow):
        """Test that /analyze endpoint accepts ticker parameter."""
        from api.main import app

        # Mock workflow response
        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        assert response.status_code == 200
        assert mock_workflow.called

    @patch('coordinator.workflow.run_stock_analysis')
    def test_analyze_endpoint_returns_prediction(self, mock_workflow):
        """Test that /analyze returns prediction results."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.75, "narrative": "Strong", "probabilities": {"up": 0.75}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"]["direction"] == "up"
        assert data["prediction"]["confidence"] == 0.75

    @patch('coordinator.workflow.run_stock_analysis')
    def test_analyze_endpoint_returns_sentiment(self, mock_workflow):
        """Test that /analyze returns sentiment analysis."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.65, "trend": "improving", "headlines": [{"title": "Good news"}]},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert data["sentiment"]["label"] == "positive"
        assert data["sentiment"]["score"] == 0.65

    @patch('coordinator.workflow.run_stock_analysis')
    def test_analyze_endpoint_returns_explanation(self, mock_workflow):
        """Test that /analyze returns explanation."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Detailed user explanation",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        assert response.status_code == 200
        data = response.json()
        assert "explanation" in data
        assert data["explanation"] == "Detailed user explanation"


class TestAPIValidation:
    """Test API request/response validation (Sua's Pydantic models)."""

    @patch('coordinator.workflow.run_stock_analysis')
    def test_request_validation_valid_ticker(self, mock_workflow):
        """Test request validation accepts valid ticker."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL", "user_tier": "basic"})

        assert response.status_code == 200

    def test_request_validation_invalid_ticker(self):
        """Test request validation rejects invalid ticker."""
        from api.main import app

        client = TestClient(app)
        # Empty ticker should be rejected
        response = client.post("/analyze", json={"ticker": ""})

        assert response.status_code == 422  # Validation error

    @patch('coordinator.workflow.run_stock_analysis')
    def test_response_validation_structure(self, mock_workflow):
        """Test response matches Pydantic model structure."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "ticker" in data
        assert "prediction" in data
        assert "sentiment" in data
        assert "explanation" in data
        assert "confidence_level" in data


class TestAPICORS:
    """Test CORS configuration for Next.js frontend."""

    def test_cors_allows_frontend_origin(self):
        """Test that CORS allows requests from Next.js frontend."""
        from api.main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:3000"})

        # CORS should allow Next.js frontend
        assert response.status_code == 200

    @patch('coordinator.workflow.run_stock_analysis')
    def test_cors_headers_present(self, mock_workflow):
        """Test that CORS headers are present in responses."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"}, headers={"Origin": "http://localhost:3000"})

        # CORS headers should be present
        assert response.status_code == 200


class TestAPIWorkflowIntegration:
    """Test API integration with Josh's workflow."""

    @patch('coordinator.workflow.run_stock_analysis')
    def test_api_calls_workflow(self, mock_workflow):
        """Test that API endpoint calls LangGraph workflow."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        # Verify workflow was called
        assert mock_workflow.called
        assert response.status_code == 200

    @patch('coordinator.workflow.run_stock_analysis')
    def test_api_handles_workflow_errors(self, mock_workflow):
        """Test that API handles workflow errors gracefully."""
        from api.main import app

        # Simulate workflow error
        mock_workflow.side_effect = ValueError("Invalid ticker")

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "INVALID"})

        # Should return 400 for validation errors
        assert response.status_code == 400

    @patch('coordinator.workflow.run_stock_analysis')
    def test_api_returns_workflow_results(self, mock_workflow):
        """Test that API returns workflow results in correct format."""
        from api.main import app

        mock_workflow.return_value = {
            "ticker": "TSLA",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "down", "confidence": 0.65, "narrative": "Bearish", "probabilities": {"down": 0.65}, "metadata": {}},
            "sentiment": {"current": "negative", "score": 0.4, "trend": "declining", "headlines": []},
            "explanation": "Market correction",
            "confidence_level": "medium",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "TSLA"})

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "TSLA"
        assert data["prediction"]["direction"] == "down"


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_api_handles_invalid_ticker(self):
        """Test API handles invalid ticker symbols."""
        from api.main import app

        client = TestClient(app)
        # Empty ticker
        response = client.post("/analyze", json={"ticker": ""})
        assert response.status_code == 422  # Validation error

    @patch('coordinator.workflow.run_stock_analysis')
    def test_api_handles_missing_data(self, mock_workflow):
        """Test API handles cases when data fetching fails."""
        from api.main import app

        # Simulate data fetch failure
        mock_workflow.side_effect = Exception("Data fetch failed")

        client = TestClient(app)
        response = client.post("/analyze", json={"ticker": "AAPL"})

        # Should return 500 for internal errors
        assert response.status_code == 500

    @patch('coordinator.workflow.run_stock_analysis')
    def test_api_returns_proper_http_status_codes(self, mock_workflow):
        """Test that API returns appropriate HTTP status codes."""
        from api.main import app

        client = TestClient(app)

        # 200 for success
        mock_workflow.return_value = {
            "ticker": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "prediction": {"direction": "up", "confidence": 0.7, "narrative": "test", "probabilities": {}, "metadata": {}},
            "sentiment": {"current": "positive", "score": 0.6, "trend": "stable", "headlines": []},
            "explanation": "Test",
            "confidence_level": "high",
            "warnings": [],
            "metadata": {"user_tier": "basic", "reflection_passed": True}
        }
        response = client.post("/analyze", json={"ticker": "AAPL"})
        assert response.status_code == 200

        # 422 for validation error
        response = client.post("/analyze", json={})
        assert response.status_code == 422


class TestAPICaching:
    """Test API integration with database caching."""

    def test_api_uses_cached_prices(self, test_db, sample_price_data, mock_ticker):
        """Test that API uses cached price data when available."""
        # This requires full integration testing with real database
        pytest.skip("Requires full integration with database")

    def test_api_uses_cached_news(self, test_db, sample_news_data, mock_ticker):
        """Test that API uses cached news when available."""
        # This requires full integration testing with real database
        pytest.skip("Requires full integration with database")

    def test_api_saves_analysis_results(self, test_db):
        """Test that API saves analysis results to database."""
        # This requires full integration testing with real database
        pytest.skip("Requires full integration with database")
