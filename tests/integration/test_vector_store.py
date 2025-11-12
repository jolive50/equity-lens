import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVectorStoreIntegration:
    """Integration tests for ChromaDB vector store (Tae's component)."""

    def test_vector_store_initialization(self):
        """Test initializing ChromaDB vector store."""
        import tempfile
        import os
        from storage.vector_store import NewsVectorStore

        # Create temp directory for test
        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = NewsVectorStore(persist_directory=temp_dir)
            assert vector_store is not None
            assert vector_store.collection is not None
        finally:
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_vector_store_add_documents(self):
        """Test adding news documents to vector store."""
        import tempfile
        import os
        from storage.vector_store import NewsVectorStore

        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = NewsVectorStore(persist_directory=temp_dir)

            articles = [
                {
                    "title": "Company reports strong earnings",
                    "content": "The company exceeded expectations",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "source": "Financial News",
                    "sentiment": "positive",
                    "sentiment_score": 0.85
                }
            ]

            count = vector_store.add_news_articles("AAPL", articles)
            assert count == 1
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_vector_store_semantic_search(self):
        """Test semantic search for similar documents."""
        import tempfile
        import os
        from storage.vector_store import NewsVectorStore

        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = NewsVectorStore(persist_directory=temp_dir)

            articles = [
                {
                    "title": "Strong earnings report",
                    "content": "Company profits exceeded expectations",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "source": "News",
                    "sentiment": "positive",
                    "sentiment_score": 0.8
                }
            ]

            vector_store.add_news_articles("AAPL", articles)

            # Search for similar articles
            results = vector_store.search_similar_news("earnings report", ticker="AAPL")

            assert len(results) > 0
            assert "Strong earnings report" in results[0]["title"]
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_vector_store_search_by_ticker(self):
        """Test searching documents filtered by ticker."""
        import tempfile
        import os
        from storage.vector_store import NewsVectorStore

        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = NewsVectorStore(persist_directory=temp_dir)

            # Add articles for different tickers
            vector_store.add_news_articles("AAPL", [{
                "title": "Apple news",
                "content": "Apple earnings",
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "News",
                "sentiment": "positive",
                "sentiment_score": 0.8
            }])

            vector_store.add_news_articles("TSLA", [{
                "title": "Tesla news",
                "content": "Tesla production",
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "News",
                "sentiment": "neutral",
                "sentiment_score": 0.5
            }])

            # Search filtered by ticker
            results = vector_store.search_similar_news("news", ticker="AAPL")

            assert len(results) > 0
            assert all(r["ticker"] == "AAPL" for r in results)
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_vector_store_search_similarity_scores(self):
        """Test that search returns similarity scores."""
        import tempfile
        import os
        from storage.vector_store import NewsVectorStore

        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = NewsVectorStore(persist_directory=temp_dir)

            articles = [{
                "title": "Earnings exceed expectations",
                "content": "Strong quarterly performance",
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "News",
                "sentiment": "positive",
                "sentiment_score": 0.8
            }]

            vector_store.add_news_articles("AAPL", articles)

            results = vector_store.search_similar_news("earnings report", ticker="AAPL")

            assert len(results) > 0
            assert "similarity" in results[0]
            assert 0 <= results[0]["similarity"] <= 1
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_vector_store_handles_empty_collection(self):
        """Test vector store behavior with empty collection."""
        import tempfile
        import os
        from storage.vector_store import NewsVectorStore

        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = NewsVectorStore(persist_directory=temp_dir)

            # Search empty collection
            results = vector_store.search_similar_news("test query", ticker="AAPL")

            assert results == []
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestVectorStoreWithNewsCache:
    """Test integration between vector store and news cache."""

    def test_populate_vector_store_from_news_cache(self, test_db, sample_news_data, mock_ticker):
        """Test populating vector store from cached news."""
        pytest.skip("Vector store not yet implemented by Tae")

    def test_news_embedding_quality(self):
        """Test that news embeddings capture semantic meaning."""
        pytest.skip("Vector store not yet implemented by Tae")


class TestExplanationAgentVectorStoreIntegration:
    """Test ExplanationAgent using vector store for context retrieval."""

    def test_explanation_agent_retrieves_relevant_news(self):
        """Test that ExplanationAgent retrieves relevant news context."""
        pytest.skip("Vector store and ExplanationAgent integration not yet complete")

    def test_explanation_uses_retrieved_context(self):
        """Test that generated explanations use retrieved news context."""
        pytest.skip("Vector store and ExplanationAgent integration not yet complete")

    def test_explanation_limits_context_window(self):
        """Test that ExplanationAgent limits number of retrieved documents."""
        pytest.skip("Vector store and ExplanationAgent integration not yet complete")
