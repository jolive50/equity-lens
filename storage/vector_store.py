import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from datetime import datetime
import os


class NewsVectorStore:
    """ChromaDB vector store for news article embeddings.

    Enables semantic search over historical news articles.
    """

    def __init__(
        self,
        persist_directory: str = "./db/chroma",
        collection_name: str = "news_articles"
    ):
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use

        Raises:
            RuntimeError: If ChromaDB initialization fails
        """
        try:
            os.makedirs(persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Financial news articles with sentiment"}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

    def add_news_articles(
        self,
        ticker: str,
        articles: List[Dict[str, Any]]
    ) -> int:
        """Add news articles to vector store.

        Args:
            ticker: Stock ticker symbol
            articles: List of article dictionaries with keys:
                     title, content, timestamp, source, sentiment, sentiment_score

        Returns:
            Number of articles added

        Raises:
            ValueError: If articles list is empty
            RuntimeError: If addition fails
        """
        if not articles:
            raise ValueError("Articles list cannot be empty")

        try:
            documents = []
            metadatas = []
            ids = []

            for i, article in enumerate(articles):
                # Combine title and content for embedding
                text = f"{article.get('title', '')} {article.get('content', '')}"
                documents.append(text)

                # Store metadata
                metadata = {
                    "ticker": ticker,
                    "title": article.get("title", "")[:500],  # Limit length
                    "source": article.get("source", "unknown"),
                    "timestamp": article.get("timestamp", datetime.now().isoformat()),
                    "sentiment": article.get("sentiment", "neutral"),
                    "sentiment_score": float(article.get("sentiment_score", 0.0))
                }

                # Add sentiment breakdown if available
                if "sentiment_breakdown" in article:
                    breakdown = article["sentiment_breakdown"]
                    metadata["positive"] = float(breakdown.get("positive", 0.0))
                    metadata["negative"] = float(breakdown.get("negative", 0.0))
                    metadata["neutral"] = float(breakdown.get("neutral", 0.0))

                metadatas.append(metadata)

                # Generate unique ID
                timestamp_str = article.get("timestamp", datetime.now().isoformat())
                article_id = f"{ticker}_{timestamp_str}_{i}"
                ids.append(article_id)

            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            return len(articles)

        except Exception as e:
            raise RuntimeError(f"Failed to add articles to vector store: {e}") from e

    def search_similar_news(
        self,
        query: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
        sentiment_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar news articles using semantic search.

        Args:
            query: Search query (e.g., headline or topic)
            ticker: Optional ticker to filter results
            n_results: Number of results to return
            sentiment_filter: Optional sentiment filter ("positive", "negative", "neutral")

        Returns:
            List of similar articles with metadata and similarity scores

        Raises:
            RuntimeError: If search fails
        """
        try:
            # Build where filter
            where = {}
            if ticker:
                where["ticker"] = ticker
            if sentiment_filter:
                where["sentiment"] = sentiment_filter

            # Query vector store
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )

            # Format results
            similar_articles = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    similar_articles.append({
                        "content": doc,
                        "title": metadata.get("title", ""),
                        "ticker": metadata.get("ticker", ""),
                        "source": metadata.get("source", "unknown"),
                        "timestamp": metadata.get("timestamp", ""),
                        "sentiment": metadata.get("sentiment", "neutral"),
                        "sentiment_score": metadata.get("sentiment_score", 0.0),
                        "distance": float(distance),
                        "similarity": 1.0 - float(distance)  # Convert distance to similarity
                    })

            return similar_articles

        except Exception as e:
            raise RuntimeError(f"Failed to search vector store: {e}") from e

    def get_ticker_statistics(self, ticker: str) -> Dict[str, Any]:
        """Get statistics for a specific ticker's news coverage.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with statistics: total_articles, avg_sentiment, etc.

        Raises:
            RuntimeError: If statistics retrieval fails
        """
        try:
            # Get all articles for ticker
            results = self.collection.get(
                where={"ticker": ticker}
            )

            if not results["metadatas"]:
                return {
                    "ticker": ticker,
                    "total_articles": 0,
                    "avg_sentiment_score": 0.0,
                    "sentiment_distribution": {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0
                    }
                }

            # Calculate statistics
            total = len(results["metadatas"])
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            sentiment_scores = []

            for metadata in results["metadatas"]:
                sentiment = metadata.get("sentiment", "neutral")
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                sentiment_scores.append(metadata.get("sentiment_score", 0.0))

            avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            return {
                "ticker": ticker,
                "total_articles": total,
                "avg_sentiment_score": avg_score,
                "sentiment_distribution": sentiment_counts
            }

        except Exception as e:
            raise RuntimeError(f"Failed to get ticker statistics: {e}") from e

    def delete_old_articles(self, days: int = 30) -> int:
        """Delete articles older than specified days.

        Args:
            days: Delete articles older than this many days

        Returns:
            Number of articles deleted

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()

            # Get all articles
            all_results = self.collection.get()

            # Filter old articles
            old_ids = []
            for i, metadata in enumerate(all_results["metadatas"]):
                timestamp = metadata.get("timestamp", "")
                if timestamp < cutoff_str:
                    old_ids.append(all_results["ids"][i])

            # Delete old articles
            if old_ids:
                self.collection.delete(ids=old_ids)

            return len(old_ids)

        except Exception as e:
            raise RuntimeError(f"Failed to delete old articles: {e}") from e

    def clear_collection(self) -> None:
        """Clear all articles from the collection.

        Raises:
            RuntimeError: If clear operation fails
        """
        try:
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description": "Financial news articles with sentiment"}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to clear collection: {e}") from e
