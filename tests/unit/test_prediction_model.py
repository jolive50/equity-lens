import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.prediction.base_predictor import BasePredictionModel, PredictionResult
from models.prediction.lstm_model import LSTMModel


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    return pd.DataFrame({
        'open': np.random.uniform(100, 110, 60),
        'high': np.random.uniform(105, 115, 60),
        'low': np.random.uniform(95, 105, 60),
        'close': np.random.uniform(100, 110, 60),
        'volume': np.random.randint(1000000, 5000000, 60)
    }, index=dates)


@pytest.fixture
def uptrend_data():
    """Create uptrend market data."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    close_prices = [100 + i*0.5 for i in range(60)]
    return pd.DataFrame({
        'open': [p - 1 for p in close_prices],
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'close': close_prices,
        'volume': [1000000] * 60
    }, index=dates)


@pytest.fixture
def downtrend_data():
    """Create downtrend market data."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    close_prices = [130 - i*0.5 for i in range(60)]
    return pd.DataFrame({
        'open': [p + 1 for p in close_prices],
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'close': close_prices,
        'volume': [1000000] * 60
    }, index=dates)


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult."""
        result = PredictionResult(
            direction="up",
            confidence=0.75,
            probabilities={"up": 0.75, "down": 0.15, "neutral": 0.10},
            metadata={"model": "LSTM"}
        )
        assert result.direction == "up"
        assert result.confidence == 0.75
        assert "up" in result.probabilities
        assert result.metadata["model"] == "LSTM"


class TestLSTMModel:
    """Test suite for LSTM prediction model."""

    def test_lstm_initialization_without_weights(self):
        """Test LSTM model initialization without pre-trained weights."""
        model = LSTMModel()
        assert model.model is None
        assert model.is_trained is False

    def test_lstm_initialization_with_nonexistent_weights(self):
        """Test LSTM initialization with non-existent weights path."""
        model = LSTMModel(weights_path="/nonexistent/path/model.h5")
        assert model.is_trained is False

    def test_lstm_predict_without_training(self, sample_market_data):
        """Test LSTM prediction falls back to momentum when not trained."""
        model = LSTMModel()
        result = model.predict(sample_market_data)

        assert isinstance(result, PredictionResult)
        assert result.direction in ["up", "down", "neutral"]
        assert 0 <= result.confidence <= 1
        assert result.metadata["model"] == "LSTM_fallback"

    def test_lstm_predict_uptrend(self, uptrend_data):
        """Test LSTM momentum fallback on uptrend data."""
        model = LSTMModel()
        result = model.predict(uptrend_data)

        assert isinstance(result, PredictionResult)
        assert result.direction == "up"
        assert result.confidence > 0.5

    def test_lstm_predict_downtrend(self, downtrend_data):
        """Test LSTM momentum fallback on downtrend data."""
        model = LSTMModel()
        result = model.predict(downtrend_data)

        assert isinstance(result, PredictionResult)
        assert result.direction == "down"
        assert result.confidence > 0.5

    def test_lstm_get_model_info(self):
        """Test getting LSTM model information."""
        model = LSTMModel()
        info = model.get_model_info()

        assert "name" in info
        assert info["name"] == "LSTM"
        assert "type" in info
        assert "trained" in info

    @patch('tensorflow.keras.models.load_model')
    def test_lstm_with_mock_trained_model(self, mock_load, sample_market_data):
        """Test LSTM with mocked trained model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.2, 0.1, 0.7]])
        mock_load.return_value = mock_model

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            model = LSTMModel(weights_path=f.name)

            if model.is_trained:
                result = model.predict(sample_market_data)
                assert result.direction == "up"
                assert result.confidence == 0.7

    def test_lstm_probabilities_sum_to_one(self, sample_market_data):
        """Test that prediction probabilities sum to approximately 1."""
        model = LSTMModel()
        result = model.predict(sample_market_data)

        prob_sum = sum(result.probabilities.values())
        assert 0.99 <= prob_sum <= 1.01


class TestPredictionEnsemble:
    """Test suite for prediction ensemble (when implemented by Pam)."""

    def test_ensemble_placeholder(self):
        """Placeholder for ensemble tests."""
        pytest.skip("Prediction ensemble testing will be added when Pam completes ensemble.py")


class TestGRUModel:
    """Test suite for GRU model (Pam's component)."""

    def test_gru_placeholder(self):
        """Placeholder for GRU model tests."""
        pytest.skip("GRU model testing will be added when Pam completes implementation")


class TestGradientBoostModel:
    """Test suite for Gradient Boost model (Pam's component)."""

    def test_gradient_boost_placeholder(self):
        """Placeholder for Gradient Boost tests."""
        pytest.skip("Gradient Boost testing will be added when Pam completes implementation")


class TestPredictionModelInterface:
    """Test that all prediction models follow the base interface."""

    def test_lstm_implements_base_interface(self):
        """Test that LSTMModel properly implements BasePredictionModel."""
        model = LSTMModel()
        assert isinstance(model, BasePredictionModel)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_model_info')

    def test_prediction_result_required_fields(self, sample_market_data):
        """Test that all predictions return required fields."""
        model = LSTMModel()
        result = model.predict(sample_market_data)

        assert hasattr(result, 'direction')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'probabilities')
        assert hasattr(result, 'metadata')

        assert all(key in result.probabilities for key in ['up', 'down', 'neutral'])
