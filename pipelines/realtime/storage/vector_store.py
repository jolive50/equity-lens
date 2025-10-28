"""ChromaDB-based vector storage for semantic search on news and analyses.

This module provides vector storage capabilities for StockSense, enabling:
- Semantic similarity search on news articles
- Historical pattern matching for predictions
- Similar filing retrieval for smart money analysis
- Context augmentation for explanation generation

What this file does:
- Wraps ChromaDB client with StockSense-specific collections
- Manages embeddings for news articles, analyses, and filings
- Provides semantic search with configurable similarity thresholds
- Handles collection lifecycle (create, populate, query, reset)

Why we need this:
- Enable "find similar news" for context augmentation
- Detect recurring patterns in historical analyses
- Improve explanation quality with relevant historical examples
- Support RAG (Retrieval-Augmented Generation) workflows

How it works:
- ChromaDB stores document embeddings as vectors
- Uses sentence-transformers for text embedding (all-MiniLM-L6-v2)
- Cosine similarity for finding related documents
- Persistent storage in local directory (data/chroma/)
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector storage for semantic search across StockSense data.

    WHAT: Manages vector embeddings for news articles, analyses, and filings
    WHY: Enables semantic similarity search for context augmentation
    HOW: Wraps ChromaDB client with typed collections and embedding functions
    DATA: Text documents → embeddings → similarity search → relevant documents

    The VectorStore provides three primary collections:
    - "news": Financial news articles with ticker tags
    - "analyses": Historical stock analysis results
    - "filings": SEC filing summaries (13F, insider trades, etc.)

    Usage Example:
        store = VectorStore(persist_directory="data/chroma")

        # Add news articles
        store.add_news_articles(
            ticker="AAPL",
            articles=[
                {"headline": "Apple beats earnings", "content": "...", "date": "2025-10-28"},
                {"headline": "iPhone sales strong", "content": "...", "date": "2025-10-27"},
            ]
        )

        # Find similar news
        similar = store.search_similar_news(
            query="earnings beat expectations",
            ticker="AAPL",
            n_results=5
        )
    """

    def __init__(
        self,
        *,
        persist_directory: str = "data/chroma",
        collection_metadata: Optional[Dict[str, Any]] = None,
        reset_on_init: bool = False,
    ):
        """
        Initialize ChromaDB vector store with persistent storage.

        Args:
            persist_directory: Path to store ChromaDB data (default: "data/chroma")
            collection_metadata: Optional metadata for collections
            reset_on_init: If True, delete existing data and start fresh

        WHAT: Set up ChromaDB client and create/connect to collections
        WHY: Need persistent vector storage that survives restarts
        HOW: Initialize ChromaDB client with local persistence settings
        DATA: Configuration → ChromaDB client → collection handles
        """
        # WHAT: Ensure persist directory exists
        # WHY: ChromaDB needs a valid directory for storage
        # HOW: Create directory path if it doesn't exist
        # DATA: String path → Path object → mkdir with parents
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # WHAT: Initialize ChromaDB client with persistence
        # WHY: Want embeddings to survive process restarts
        # HOW: Use PersistentClient for file-based storage
        # DATA: ChromaDB client configured for persistent mode
        logger.info(f"Initializing ChromaDB vector store at {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # WHAT: Handle reset flag (delete existing data)
        # WHY: Useful for testing or fresh starts
        # HOW: Delete all collections if reset requested
        # DATA: Collection names → delete each collection
        if reset_on_init:
            logger.warning("Resetting vector store - deleting all collections")
            self._reset_all_collections()

        # WHAT: Store collection metadata for later use
        # WHY: May want to tag collections with creation date, version, etc.
        # HOW: Store dict for merging into collection metadata
        # DATA: User-provided metadata or empty dict
        self.collection_metadata = collection_metadata or {}

        # WHAT: Initialize collection handles
        # WHY: Keep references to collections for fast access
        # HOW: Get or create each collection type
        # DATA: Collection names → ChromaDB collection objects
        self.news_collection = self._get_or_create_collection("news")
        self.analyses_collection = self._get_or_create_collection("analyses")
        self.filings_collection = self._get_or_create_collection("filings")

        logger.info("ChromaDB vector store initialized successfully")

    def _reset_all_collections(self) -> None:
        """
        Delete all collections in the vector store.

        WHAT: Remove all stored embeddings and documents
        WHY: Clean slate for testing or reindexing
        HOW: List all collections and delete each one
        DATA: ChromaDB client → collection list → delete operations
        """
        # WHAT: Get list of all existing collections
        # WHY: Need to know what to delete
        # HOW: Query ChromaDB for collection names
        # DATA: ChromaDB API → list of collection objects
        collections = self.client.list_collections()

        # WHAT: Delete each collection
        # WHY: Reset operation should remove all data
        # HOW: Call delete_collection for each name
        # DATA: Collection objects → delete by name
        for collection in collections:
            logger.debug(f"Deleting collection: {collection.name}")
            self.client.delete_collection(name=collection.name)

    def _get_or_create_collection(self, name: str) -> Any:
        """
        Get existing collection or create new one.

        Args:
            name: Collection name (e.g., "news", "analyses", "filings")

        Returns:
            ChromaDB collection object

        WHAT: Ensure a named collection exists
        WHY: Collections must exist before adding documents
        HOW: Try to get existing, create if not found
        DATA: Collection name → ChromaDB collection handle
        """
        # WHAT: Merge user metadata with collection-specific metadata
        # WHY: Want to tag collections with creation time and purpose
        # HOW: Combine default metadata with user-provided metadata
        # DATA: Dicts merged into single metadata payload
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "collection_type": name,
            **self.collection_metadata,
        }

        # WHAT: Get or create the collection
        # WHY: ChromaDB handles idempotency (won't error if exists)
        # HOW: Use get_or_create_collection API
        # DATA: Name + metadata → collection object
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )

        logger.debug(f"Collection '{name}' ready (count: {collection.count()})")
        return collection

    def add_news_articles(
        self,
        *,
        ticker: str,
        articles: List[Dict[str, Any]],
    ) -> int:
        """
        Add news articles to vector store with embeddings.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            articles: List of article dicts with keys: headline, content, date, source

        Returns:
            int: Number of articles successfully added

        WHAT: Store news articles as embeddings for semantic search
        WHY: Enable finding similar news when analyzing a stock
        HOW: Generate embeddings from article text, store with metadata
        DATA: Article dicts → embeddings → ChromaDB news collection

        Example article format:
            {
                "headline": "Apple reports record revenue",
                "content": "Apple Inc. announced quarterly results...",
                "date": "2025-10-28",
                "source": "Reuters",
                "url": "https://...",  # optional
                "sentiment": 0.75,     # optional
            }
        """
        # WHAT: Validate input articles list
        # WHY: Need at least one article to process
        # HOW: Check list length, return 0 if empty
        # DATA: articles list → count check
        if not articles:
            logger.warning(f"No articles provided for ticker {ticker}")
            return 0

        # WHAT: Prepare data structures for ChromaDB batch insert
        # WHY: ChromaDB expects parallel lists (ids, documents, metadatas)
        # HOW: Build lists by iterating articles
        # DATA: articles → (ids, documents, metadatas) tuples
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # WHAT: Process each article into ChromaDB format
        # WHY: Need to extract text for embedding and preserve metadata
        # HOW: Combine headline + content as document, store rest as metadata
        # DATA: article dict → id string, document text, metadata dict
        for idx, article in enumerate(articles):
            # WHAT: Generate unique ID for this article
            # WHY: ChromaDB requires unique IDs for deduplication
            # HOW: Combine ticker, date, and index for uniqueness
            # DATA: ticker + date + idx → unique string ID
            article_id = f"{ticker}_{article.get('date', 'unknown')}_{idx}"
            ids.append(article_id)

            # WHAT: Combine headline and content for embedding
            # WHY: Both contain semantic information for search
            # HOW: Concatenate with separator, handle missing fields
            # DATA: headline + content → single document string
            headline = article.get("headline", "")
            content = article.get("content", "")
            document = f"{headline}\n\n{content}".strip()
            documents.append(document)

            # WHAT: Store article metadata for retrieval
            # WHY: Need to return this info with search results
            # HOW: Extract relevant fields, add ticker tag
            # DATA: article dict → filtered metadata dict
            metadata = {
                "ticker": ticker,
                "date": article.get("date", "unknown"),
                "source": article.get("source", "unknown"),
                "headline": headline,
                "url": article.get("url", ""),
                "sentiment": article.get("sentiment", 0.0),
                "added_at": datetime.utcnow().isoformat(),
            }
            metadatas.append(metadata)

        # WHAT: Batch insert articles into ChromaDB
        # WHY: More efficient than individual inserts
        # HOW: Use add() method with parallel lists
        # DATA: (ids, documents, metadatas) → ChromaDB storage
        try:
            self.news_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                f"Added {len(articles)} news articles for {ticker} to vector store"
            )
            return len(articles)
        except Exception as e:
            logger.error(f"Failed to add articles to vector store: {e}")
            return 0

    def search_similar_news(
        self,
        *,
        query: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
        max_distance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles semantically similar to query.

        Args:
            query: Search query text (e.g., "earnings beat expectations")
            ticker: Optional ticker filter (only return news for this stock)
            n_results: Maximum number of results to return (default: 5)
            max_distance: Optional maximum cosine distance threshold (0-2, lower is more similar)

        Returns:
            List of article dicts with similarity scores

        WHAT: Find news articles similar to a text query
        WHY: Context augmentation for analysis (find related news)
        HOW: Embed query, compute cosine similarity, return top-k
        DATA: Query text → embedding → similarity scores → ranked articles

        Example result:
            [
                {
                    "headline": "Apple reports record revenue",
                    "ticker": "AAPL",
                    "date": "2025-10-28",
                    "source": "Reuters",
                    "distance": 0.23,  # Lower is more similar
                    "content": "..."
                },
                ...
            ]
        """
        # WHAT: Build where filter for ticker if provided
        # WHY: User may want to scope search to specific stock
        # HOW: Create ChromaDB where clause for metadata filtering
        # DATA: ticker string → where dict or None
        where = {"ticker": ticker} if ticker else None

        # WHAT: Query ChromaDB for similar documents
        # WHY: Core semantic search operation
        # HOW: Embed query text, compute similarity to stored embeddings
        # DATA: Query text → embedding → similarity computation → results
        try:
            results = self.news_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            return []

        # WHAT: Check if results are empty
        # WHY: ChromaDB returns empty structure if no matches
        # HOW: Check if results lists are populated
        # DATA: results dict → boolean check
        if not results["ids"] or not results["ids"][0]:
            logger.debug(f"No similar news found for query: {query[:50]}...")
            return []

        # WHAT: Format results into user-friendly dicts
        # WHY: API should return consistent structured data
        # HOW: Zip parallel lists from ChromaDB into dicts
        # DATA: ChromaDB result arrays → list of article dicts
        formatted_results = []
        for i in range(len(results["ids"][0])):
            # WHAT: Extract data from parallel arrays
            # WHY: ChromaDB returns data in separate lists
            # HOW: Index into each list at position i
            # DATA: Array indices → individual result values
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]

            # WHAT: Apply max_distance filter if specified
            # WHY: User may want to exclude distant matches
            # HOW: Skip results with distance > threshold
            # DATA: distance float → comparison → include/exclude
            if max_distance is not None and distance > max_distance:
                continue

            # WHAT: Build result dictionary
            # WHY: Consistent return format for API consumers
            # HOW: Merge metadata with distance and document
            # DATA: metadata + distance + document → result dict
            formatted_results.append({
                **metadata,  # ticker, date, source, headline, etc.
                "distance": distance,
                "content": document,
            })

        logger.debug(
            f"Found {len(formatted_results)} similar articles "
            f"(ticker={ticker}, max_distance={max_distance})"
        )
        return formatted_results

    def add_analysis(
        self,
        *,
        ticker: str,
        analysis_text: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Add a stock analysis to vector store for pattern matching.

        Args:
            ticker: Stock ticker symbol
            analysis_text: Full analysis narrative text
            metadata: Dict with keys: date, prediction, confidence, sentiment, etc.

        Returns:
            bool: True if successfully added

        WHAT: Store completed analysis for historical pattern matching
        WHY: Enable finding similar past analyses when forecasting
        HOW: Embed analysis narrative, store with prediction metadata
        DATA: Analysis text + metadata → embedding → analyses collection

        Example metadata:
            {
                "date": "2025-10-28",
                "prediction": "up",
                "confidence": 0.85,
                "sentiment": "positive",
                "actual_outcome": "up",  # optional, for backtesting
            }
        """
        # WHAT: Generate unique ID for analysis
        # WHY: ChromaDB requires unique IDs
        # HOW: Combine ticker and timestamp for uniqueness
        # DATA: ticker + timestamp → unique analysis ID
        analysis_id = f"{ticker}_{datetime.utcnow().isoformat()}"

        # WHAT: Merge provided metadata with standard fields
        # WHY: Want consistent metadata across all analyses
        # HOW: Add ticker and added_at to user metadata
        # DATA: user metadata + standard fields → complete metadata dict
        full_metadata = {
            "ticker": ticker,
            "added_at": datetime.utcnow().isoformat(),
            **metadata,
        }

        # WHAT: Add analysis to ChromaDB
        # WHY: Store for later similarity search
        # HOW: Call add() with analysis text as document
        # DATA: (id, text, metadata) → ChromaDB analyses collection
        try:
            self.analyses_collection.add(
                ids=[analysis_id],
                documents=[analysis_text],
                metadatas=[full_metadata],
            )
            logger.info(f"Added analysis for {ticker} to vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to add analysis to vector store: {e}")
            return False

    def search_similar_analyses(
        self,
        *,
        query: str,
        ticker: Optional[str] = None,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find historical analyses similar to current situation.

        Args:
            query: Description of current situation (e.g., "bullish sentiment with high volume")
            ticker: Optional ticker filter
            n_results: Maximum number of results (default: 3)

        Returns:
            List of similar analysis dicts with metadata

        WHAT: Retrieve past analyses that match current conditions
        WHY: Learn from historical patterns (what happened last time in similar situation?)
        HOW: Semantic search on analysis narratives
        DATA: Query text → embedding → similar analyses with outcomes
        """
        # WHAT: Build ticker filter if provided
        where = {"ticker": ticker} if ticker else None

        # WHAT: Query analyses collection
        # WHY: Find historically similar situations
        # HOW: Embed query, compute similarity to past analyses
        # DATA: Query → embedding → similarity search → results
        try:
            results = self.analyses_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error(f"Failed to query analyses: {e}")
            return []

        # WHAT: Check for empty results
        if not results["ids"] or not results["ids"][0]:
            logger.debug(f"No similar analyses found for query: {query[:50]}...")
            return []

        # WHAT: Format results into dicts
        # WHY: Consistent API return format
        # HOW: Zip parallel arrays from ChromaDB
        # DATA: ChromaDB arrays → list of analysis dicts
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                **results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "analysis_text": results["documents"][0][i],
            })

        logger.debug(f"Found {len(formatted_results)} similar analyses")
        return formatted_results

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get document counts for all collections.

        Returns:
            Dict with collection names and document counts

        WHAT: Report how many documents stored in each collection
        WHY: Monitoring and debugging (know what's in the vector store)
        HOW: Query count() for each collection
        DATA: Collection objects → counts → stats dict
        """
        # WHAT: Query document counts from ChromaDB
        # WHY: Need visibility into what's stored
        # HOW: Call count() on each collection handle
        # DATA: Collection objects → integer counts
        stats = {
            "news": self.news_collection.count(),
            "analyses": self.analyses_collection.count(),
            "filings": self.filings_collection.count(),
        }

        logger.debug(f"Vector store stats: {stats}")
        return stats

    def reset_collection(self, collection_name: str) -> bool:
        """
        Delete and recreate a specific collection.

        Args:
            collection_name: Name of collection to reset ("news", "analyses", "filings")

        Returns:
            bool: True if successfully reset

        WHAT: Clear all data from a collection
        WHY: Useful for reindexing or cleaning test data
        HOW: Delete collection, recreate empty one
        DATA: Collection name → delete → recreate → update handle
        """
        # WHAT: Validate collection name
        # WHY: Only allow resetting known collections
        # HOW: Check against allowed list
        # DATA: collection_name → membership test
        if collection_name not in ["news", "analyses", "filings"]:
            logger.error(f"Invalid collection name: {collection_name}")
            return False

        # WHAT: Delete existing collection
        # WHY: Need to remove all documents
        # HOW: Call delete_collection on client
        # DATA: Collection name → ChromaDB delete operation
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")
            # Continue anyway - collection might not exist

        # WHAT: Recreate collection and update handle
        # WHY: Need fresh empty collection
        # HOW: Call _get_or_create_collection, update instance attribute
        # DATA: Collection name → new collection object → instance attribute
        new_collection = self._get_or_create_collection(collection_name)

        # WHAT: Update instance attribute to point to new collection
        # WHY: Keep references valid after reset
        # HOW: Use setattr to update the attribute dynamically
        # DATA: Collection name → attribute name → new collection object
        if collection_name == "news":
            self.news_collection = new_collection
        elif collection_name == "analyses":
            self.analyses_collection = new_collection
        elif collection_name == "filings":
            self.filings_collection = new_collection

        logger.info(f"Reset collection: {collection_name}")
        return True
