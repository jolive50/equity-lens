import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.sentiment.base_sentiment import BaseSentimentModel, SentimentResult
from models.sentiment.vader_model import VADERModel
from models.sentiment.textblob_model import TextBlobModel


@pytest.fixture
def positive_text_samples():
    """Sample positive sentiment texts."""
    return [
        "Apple reports record earnings, stock surges to new highs",
        "Incredible growth in iPhone sales exceeds all expectations",
        "Strong fundamentals and excellent quarterly performance",
        "Analysts raise price targets following outstanding results"
    ]


@pytest.fixture
def negative_text_samples():
    """Sample negative sentiment texts."""
    return [
        "Apple faces regulatory scrutiny, shares decline sharply",
        "Supply chain disruptions threaten production targets",
        "Disappointing earnings miss analyst expectations",
        "Market concerns grow over increasing competition"
    ]


@pytest.fixture
def neutral_text_samples():
    """Sample neutral sentiment texts."""
    return [
        "Apple announces quarterly dividend payment",
        "The company will hold earnings call on Thursday",
        "Stock price remains stable in sideways trading",
        "Analysts maintain hold rating on the stock"
    ]


class TestSentimentResult:
    """Test SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test creating a SentimentResult."""
        result = SentimentResult(
            label="positive",
            confidence=0.85,
            probabilities={"positive": 0.85, "negative": 0.05, "neutral": 0.10},
            metadata={"model": "TestModel"}
        )
        assert result.label == "positive"
        assert result.confidence == 0.85
        assert result.probabilities["positive"] == 0.85
        assert result.metadata["model"] == "TestModel"


class TestVADERModel:
    """Test VADER sentiment model (Tae's component)."""

    def test_vader_initialization(self):
        """Test VADER model initialization."""
        model = VADERModel()
        assert model.analyzer is not None

    def test_vader_positive_sentiment(self, positive_text_samples):
        """Test VADER on positive texts."""
        model = VADERModel()
        for text in positive_text_samples:
            result = model.analyze(text)
            assert isinstance(result, SentimentResult)
            assert result.label in ["positive", "negative", "neutral"]
            assert 0 <= result.confidence <= 1

    def test_vader_negative_sentiment(self, negative_text_samples):
        """Test VADER on negative texts."""
        model = VADERModel()
        for text in negative_text_samples:
            result = model.analyze(text)
            assert isinstance(result, SentimentResult)
            assert result.label in ["positive", "negative", "neutral"]
            assert 0 <= result.confidence <= 1

    def test_vader_neutral_sentiment(self, neutral_text_samples):
        """Test VADER on neutral texts."""
        model = VADERModel()
        for text in neutral_text_samples:
            result = model.analyze(text)
            assert isinstance(result, SentimentResult)
            assert result.label in ["positive", "negative", "neutral"]
            assert 0 <= result.confidence <= 1

    def test_vader_compound_score(self):
        """Test that VADER returns compound sentiment score."""
        model = VADERModel()
        result = model.analyze("This stock is great!")
        assert "compound_score" in result.metadata
        assert isinstance(result.metadata["compound_score"], float)

    def test_vader_empty_text_raises_error(self):
        """Test that VADER raises error on empty text."""
        model = VADERModel()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            model.analyze("")

    def test_vader_batch_analysis(self, positive_text_samples):
        """Test VADER batch analysis."""
        model = VADERModel()
        results = model.analyze_batch(positive_text_samples)
        assert len(results) == len(positive_text_samples)
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_vader_get_model_info(self):
        """Test VADER get_model_info method."""
        model = VADERModel()
        info = model.get_model_info()
        assert info["name"] == "VADER"
        assert info["type"] == "rule-based"
        assert "capabilities" in info


class TestTextBlobModel:
    """Test TextBlob sentiment model (Tae's component)."""

    def test_textblob_initialization(self):
        """Test TextBlob model initialization."""
        model = TextBlobModel()
        assert model is not None

    def test_textblob_positive_sentiment(self, positive_text_samples):
        """Test TextBlob on positive texts."""
        model = TextBlobModel()
        for text in positive_text_samples:
            result = model.analyze(text)
            assert isinstance(result, SentimentResult)
            assert result.label in ["positive", "negative", "neutral"]
            assert 0 <= result.confidence <= 1

    def test_textblob_negative_sentiment(self, negative_text_samples):
        """Test TextBlob on negative texts."""
        model = TextBlobModel()
        for text in negative_text_samples:
            result = model.analyze(text)
            assert isinstance(result, SentimentResult)
            assert result.label in ["positive", "negative", "neutral"]
            assert 0 <= result.confidence <= 1

    def test_textblob_polarity_score(self):
        """Test that TextBlob returns polarity scores."""
        model = TextBlobModel()
        result = model.analyze("This is amazing!")
        assert "polarity" in result.metadata
        assert "subjectivity" in result.metadata
        assert isinstance(result.metadata["polarity"], float)

    def test_textblob_empty_text_raises_error(self):
        """Test that TextBlob raises error on empty text."""
        model = TextBlobModel()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            model.analyze("")

    def test_textblob_batch_analysis(self, positive_text_samples):
        """Test TextBlob batch analysis."""
        model = TextBlobModel()
        results = model.analyze_batch(positive_text_samples)
        assert len(results) == len(positive_text_samples)
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_textblob_get_model_info(self):
        """Test TextBlob get_model_info method."""
        model = TextBlobModel()
        info = model.get_model_info()
        assert info["name"] == "TextBlob"
        assert info["type"] == "pattern-based"
        assert "capabilities" in info


class TestSentimentModelInterface:
    """Test that all sentiment models follow base interface."""

    def test_vader_implements_base_interface(self):
        """Test that VADER implements BaseSentimentModel."""
        model = VADERModel()
        assert isinstance(model, BaseSentimentModel)
        assert hasattr(model, "analyze")
        assert hasattr(model, "analyze_batch")
        assert hasattr(model, "get_model_info")

    def test_textblob_implements_base_interface(self):
        """Test that TextBlob implements BaseSentimentModel."""
        model = TextBlobModel()
        assert isinstance(model, BaseSentimentModel)
        assert hasattr(model, "analyze")
        assert hasattr(model, "analyze_batch")
        assert hasattr(model, "get_model_info")

    def test_all_models_return_sentiment_result(self):
        """Test that all models return SentimentResult dataclass."""
        text = "Test sentiment analysis"

        vader = VADERModel()
        vader_result = vader.analyze(text)
        assert isinstance(vader_result, SentimentResult)

        textblob = TextBlobModel()
        textblob_result = textblob.analyze(text)
        assert isinstance(textblob_result, SentimentResult)

    def test_sentiment_result_has_required_fields(self):
        """Test SentimentResult has label, confidence, probabilities, and metadata."""
        model = VADERModel()
        result = model.analyze("Test text")
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "probabilities")
        assert hasattr(result, "metadata")
        assert result.label in ["positive", "negative", "neutral"]
        assert 0 <= result.confidence <= 1
        assert "positive" in result.probabilities
        assert "negative" in result.probabilities
        assert "neutral" in result.probabilities


class TestSentimentScoreValidation:
    """Test sentiment score validation across all models."""

    def test_vader_scores_within_valid_range(self):
        """Test that VADER sentiment scores are valid."""
        model = VADERModel()
        result = model.analyze("Great stock performance!")
        assert 0 <= result.confidence <= 1
        for prob in result.probabilities.values():
            assert 0 <= prob <= 1

    def test_textblob_scores_within_valid_range(self):
        """Test that TextBlob sentiment scores are valid."""
        model = TextBlobModel()
        result = model.analyze("Poor earnings report")
        assert 0 <= result.confidence <= 1
        for prob in result.probabilities.values():
            assert 0 <= prob <= 1

    def test_probabilities_sum_to_one(self):
        """Test that probability distributions sum to approximately 1.0."""
        models = [VADERModel(), TextBlobModel()]
        for model in models:
            result = model.analyze("Test sentiment")
            prob_sum = sum(result.probabilities.values())
            assert 0.99 <= prob_sum <= 1.01
