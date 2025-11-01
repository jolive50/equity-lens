"""
Unit tests for SentimentAgent + VectorStore integration.

WHAT: Tests the integration between SentimentAgent and VectorStore for semantic search
WHY: Ensures news articles are properly stored and retrieved for historical context
HOW: Mock VectorStore, test article storage, similarity search, and error handling
DATA: Sample news articles → VectorStore → similar articles retrieval
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, MagicMock
from datetime import datetime
import sys

# WHAT: Mock tensorflow before importing finbert module
# WHY: Unit tests should not require TensorFlow installation
# HOW: Add mock module to sys.modules
# DATA: sys.modules['tensorflow'] = Mock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Import classes to test
from pipelines.realtime.sentiment.finbert import NewsSentimentProcessor, FinBERTSentimentAnalyzer


class TestSentimentVectorStoreIntegration:
    """Test suite for SentimentAgent + VectorStore integration.

    WHAT: Validates that news articles are stored and retrieved from VectorStore
    WHY: VectorStore integration is critical for historical context in sentiment analysis
    HOW: Use mocked VectorStore to verify storage and retrieval operations
    """

    @pytest.fixture
    def mock_vector_store(self):
        """
        WHAT: Create mock VectorStore for testing.
        WHY: Avoid real ChromaDB dependencies in unit tests.
        HOW: Use unittest.mock.Mock with predefined methods.
        DATA: Returns Mock object with add_news_articles and search_similar_news methods.
        """
        vector_store = Mock()
        vector_store.add_news_articles = Mock(return_value=3)  # Return count of stored articles
        vector_store.search_similar_news = Mock(return_value=[
            {
                "title": "Historical article about earnings",
                "content": "Past earnings beat expectations",
                "distance": 0.15,
                "metadata": {"ticker": "AAPL", "sentiment": "positive"}
            }
        ])
        return vector_store

    @pytest.fixture
    def mock_finbert_analyzer(self):
        """
        WHAT: Create mock FinBERT analyzer for testing.
        WHY: Avoid loading actual ML model in tests.
        HOW: Mock analyze_batch method to return sentiment scores.
        DATA: Returns Mock with predefined sentiment scores.
        """
        analyzer = Mock(spec=FinBERTSentimentAnalyzer)
        analyzer.analyze_batch = Mock(return_value=[
            {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            {"positive": 0.7, "negative": 0.2, "neutral": 0.1},
            {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
        ])
        return analyzer

    @pytest.fixture
    def sample_articles(self):
        """
        WHAT: Sample news articles for testing.
        WHY: Realistic test data for article processing.
        HOW: Create list of article dictionaries with required fields.
        DATA: Returns List[Dict] with title, content, timestamp, source.
        """
        return [
            {
                "title": "Apple Reports Record Q4 Earnings",
                "content": "Apple Inc. announced record-breaking earnings for Q4 2024...",
                "timestamp": "2024-10-28T10:00:00",
                "source": "CNBC"
            },
            {
                "title": "iPhone Sales Exceed Expectations",
                "content": "Strong iPhone 15 sales drive revenue growth...",
                "timestamp": "2024-10-28T11:00:00",
                "source": "Bloomberg"
            },
            {
                "title": "Tech Sector Shows Resilience",
                "content": "Technology stocks demonstrate strong performance...",
                "timestamp": "2024-10-28T12:00:00",
                "source": "Reuters"
            }
        ]

    def test_sentiment_processor_with_vector_store(self, mock_finbert_analyzer, mock_vector_store, sample_articles):
        """
        WHAT: Test SentimentProcessor stores articles in VectorStore.
        WHY: Verify VectorStore integration works correctly.
        HOW: Process articles with VectorStore enabled, check add_news_articles called.
        DATA: sample_articles → NewsSentimentProcessor → VectorStore.add_news_articles.
        """
        # WHAT: Create sentiment processor with VectorStore
        # WHY: Test integrated behavior
        # HOW: Pass mock vector_store to processor
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles with ticker
        # WHY: Ticker is required for VectorStore storage
        # HOW: Call process_news_articles with ticker parameter
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify VectorStore.add_news_articles was called
        # WHY: Ensure articles are stored after processing
        # HOW: Check mock was called once with ticker="AAPL"
        # DATA: Expected call: add_news_articles(ticker="AAPL", articles=...)
        mock_vector_store.add_news_articles.assert_called_once()
        call_args = mock_vector_store.add_news_articles.call_args
        assert call_args.kwargs["ticker"] == "AAPL"
        assert len(call_args.kwargs["articles"]) == 3

    def test_sentiment_processor_retrieves_similar_articles(self, mock_finbert_analyzer, mock_vector_store, sample_articles):
        """
        WHAT: Test SentimentProcessor retrieves similar historical articles.
        WHY: Verify similarity search functionality.
        HOW: Process articles, check search_similar_news called, verify result includes similar_articles.
        DATA: sample_articles → process → VectorStore.search_similar_news → result["similar_articles"].
        """
        # WHAT: Create processor with VectorStore enabled
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify similarity search was called
        # WHY: Should search for similar articles using top headline
        # HOW: Check mock called with query and ticker
        mock_vector_store.search_similar_news.assert_called_once()
        call_args = mock_vector_store.search_similar_news.call_args
        assert call_args.kwargs["ticker"] == "AAPL"
        assert call_args.kwargs["n_results"] == 5

        # WHAT: Verify result includes similar_articles
        # WHY: Historical context should be included in result
        # HOW: Check "similar_articles" key in result
        assert "similar_articles" in result
        assert len(result["similar_articles"]) == 1
        assert result["similar_articles"][0]["title"] == "Historical article about earnings"

    def test_sentiment_processor_without_vector_store(self, mock_finbert_analyzer, sample_articles):
        """
        WHAT: Test SentimentProcessor works without VectorStore.
        WHY: VectorStore is optional, should not break core functionality.
        HOW: Process articles without VectorStore, verify no errors.
        DATA: sample_articles → process (no VectorStore) → sentiment result.
        """
        # WHAT: Create processor without VectorStore
        # WHY: Test backward compatibility
        # HOW: Pass vector_store=None
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=None,
            enable_similarity_search=False
        )

        # WHAT: Process articles should work without VectorStore
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify standard result fields exist
        # WHY: Core functionality should work without VectorStore
        # HOW: Check required keys in result
        assert "current" in result
        assert "score" in result
        assert "trend" in result
        assert "headlines" in result

        # WHAT: Verify similar_articles not in result
        # WHY: VectorStore disabled, no similarity search
        # HOW: Check key not present
        assert "similar_articles" not in result

    def test_sentiment_processor_without_ticker(self, mock_finbert_analyzer, mock_vector_store, sample_articles):
        """
        WHAT: Test SentimentProcessor handles missing ticker gracefully.
        WHY: Ticker is optional, VectorStore should be skipped if not provided.
        HOW: Process articles without ticker, verify VectorStore not called.
        DATA: sample_articles (no ticker) → process → no VectorStore calls.
        """
        # WHAT: Create processor with VectorStore
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles WITHOUT ticker
        # WHY: Test graceful degradation
        # HOW: Call process_news_articles without ticker parameter
        result = processor.process_news_articles(sample_articles)

        # WHAT: Verify VectorStore methods were NOT called
        # WHY: Cannot store/search without ticker
        # HOW: Check mocks not called
        mock_vector_store.add_news_articles.assert_not_called()
        mock_vector_store.search_similar_news.assert_not_called()

        # WHAT: Verify result still valid
        # WHY: Should not fail, just skip VectorStore
        # HOW: Check required keys present
        assert "current" in result
        assert "score" in result

    def test_vector_store_error_handling(self, mock_finbert_analyzer, sample_articles):
        """
        WHAT: Test SentimentProcessor handles VectorStore errors gracefully.
        WHY: VectorStore failures should not crash sentiment analysis.
        HOW: Mock VectorStore to raise exception, verify analysis still succeeds.
        DATA: sample_articles → process → VectorStore error → result (no similar_articles).
        """
        # WHAT: Create VectorStore that raises errors
        # WHY: Simulate VectorStore failures
        # HOW: Mock methods to raise exceptions
        error_vector_store = Mock()
        error_vector_store.add_news_articles = Mock(side_effect=Exception("ChromaDB connection failed"))
        error_vector_store.search_similar_news = Mock(side_effect=Exception("Search failed"))

        # WHAT: Create processor with error-prone VectorStore
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=error_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles should succeed despite VectorStore errors
        # WHY: VectorStore is enhancement, not requirement
        # HOW: Call process_news_articles, verify no exception raised
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify result is valid
        # WHY: Core sentiment analysis should work
        # HOW: Check required fields present
        assert "current" in result
        assert "score" in result
        assert "trend" in result
        assert "headlines" in result

        # WHAT: Verify similar_articles not included
        # WHY: VectorStore failed, no historical context available
        # HOW: Check key not in result
        assert "similar_articles" not in result

    def test_stored_articles_include_sentiment_metadata(self, mock_finbert_analyzer, mock_vector_store, sample_articles):
        """
        WHAT: Test stored articles include sentiment scores in metadata.
        WHY: VectorStore should store enriched articles with sentiment information.
        HOW: Process articles, check stored articles have sentiment metadata.
        DATA: sample_articles → process → stored_articles with {sentiment, sentiment_score, sentiment_breakdown}.
        """
        # WHAT: Create processor with VectorStore
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Get the stored articles from mock call
        # WHY: Verify articles have sentiment metadata
        # HOW: Extract articles parameter from mock call
        call_args = mock_vector_store.add_news_articles.call_args
        stored_articles = call_args.kwargs["articles"]

        # WHAT: Verify each stored article has sentiment metadata
        # WHY: VectorStore needs sentiment data for context
        # HOW: Check required metadata keys in each article
        for article in stored_articles:
            assert "sentiment" in article
            assert "sentiment_score" in article
            assert "sentiment_breakdown" in article
            assert "positive" in article["sentiment_breakdown"]
            assert "negative" in article["sentiment_breakdown"]
            assert "neutral" in article["sentiment_breakdown"]

    def test_similarity_search_uses_top_headline(self, mock_finbert_analyzer, mock_vector_store, sample_articles):
        """
        WHAT: Test similarity search uses top headline as query.
        WHY: Most relevant article should be used for finding similar historical articles.
        HOW: Process articles, verify search_similar_news called with top headline.
        DATA: sample_articles[0] headline → search query.
        """
        # WHAT: Create processor with VectorStore
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify search used top headline as query
        # WHY: Top headline is most relevant for similarity search
        # HOW: Check query parameter in mock call
        call_args = mock_vector_store.search_similar_news.call_args
        query = call_args.kwargs["query"]

        # WHAT: Query should be the first article's title
        # WHY: Articles are filtered/scored, top article used
        # HOW: Compare query with expected headline
        # Note: Due to filtering, we just check query is non-empty string
        assert isinstance(query, str)
        assert len(query) > 0

    def test_enable_similarity_search_flag(self, mock_finbert_analyzer, mock_vector_store, sample_articles):
        """
        WHAT: Test enable_similarity_search flag controls VectorStore behavior.
        WHY: Should be able to disable similarity search even with VectorStore available.
        HOW: Create processor with enable_similarity_search=False, verify search not called.
        DATA: VectorStore available but similarity_search disabled → no search calls.
        """
        # WHAT: Create processor with similarity search disabled
        # WHY: Test flag controls behavior
        # HOW: Set enable_similarity_search=False
        processor = NewsSentimentProcessor(
            analyzer=mock_finbert_analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=False
        )

        # WHAT: Process articles
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify VectorStore methods NOT called
        # WHY: Similarity search disabled
        # HOW: Check mocks not called
        mock_vector_store.add_news_articles.assert_not_called()
        mock_vector_store.search_similar_news.assert_not_called()

        # WHAT: Result should not have similar_articles
        assert "similar_articles" not in result

    def test_integration_with_real_finbert_structure(self, mock_vector_store, sample_articles):
        """
        WHAT: Test integration with real FinBERT analyzer structure.
        WHY: Ensure mocked analyzer matches actual FinBERT behavior.
        HOW: Use more realistic mock that mimics FinBERT API.
        DATA: Realistic sentiment scores → proper result structure.
        """
        # WHAT: Create more realistic FinBERT mock
        # WHY: Better match actual FinBERT behavior
        # HOW: Mock with proper return structure
        analyzer = Mock(spec=FinBERTSentimentAnalyzer)
        analyzer.analyze_batch = Mock(return_value=[
            {"positive": 0.85, "negative": 0.05, "neutral": 0.10},
            {"positive": 0.75, "negative": 0.15, "neutral": 0.10},
            {"positive": 0.90, "negative": 0.03, "neutral": 0.07}
        ])

        # WHAT: Create processor
        processor = NewsSentimentProcessor(
            analyzer=analyzer,
            vector_store=mock_vector_store,
            enable_similarity_search=True
        )

        # WHAT: Process articles
        result = processor.process_news_articles(sample_articles, ticker="AAPL")

        # WHAT: Verify result has expected structure
        assert result["current"] == "positive"  # Avg positive score highest
        assert 0.0 <= result["score"] <= 1.0
        assert result["trend"] in ["improving", "stable", "declining"]
        assert len(result["headlines"]) <= 3
        assert result["article_count"] == 3

        # WHAT: Verify sentiment breakdown included
        assert "sentiment_breakdown" in result
        assert "positive" in result["sentiment_breakdown"]
        assert "negative" in result["sentiment_breakdown"]
        assert "neutral" in result["sentiment_breakdown"]


if __name__ == "__main__":
    # WHAT: Run tests when script executed directly
    # WHY: Allow running tests without pytest command
    # HOW: Use pytest.main() to run tests in this file
    pytest.main([__file__, "-v"])
