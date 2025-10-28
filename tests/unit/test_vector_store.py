"""Unit tests for VectorStore ChromaDB integration.

This test suite validates the VectorStore's ability to:
- Initialize ChromaDB with persistent storage
- Add news articles with embeddings
- Search for semantically similar news
- Add and retrieve historical analyses
- Reset collections
- Report collection statistics

WHAT: Comprehensive test coverage for all VectorStore functionality
WHY: Ensures vector storage works correctly for semantic search
HOW: Pytest-based unit tests with temporary directories and fixtures
DATA: Sample news/analyses → vector store → search results → assertions
"""
import tempfile
from pathlib import Path

import pytest

from pipelines.realtime.storage.vector_store import VectorStore


# ===== FIXTURES =====


@pytest.fixture
def temp_chroma_dir():
    """
    Create temporary directory for ChromaDB storage.

    WHAT: Provides isolated storage for each test
    WHY: Tests shouldn't interfere with each other or production data
    HOW: Use tempfile to create temporary directory
    DATA: Temporary directory path that gets cleaned up after test
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_chroma_dir):
    """
    Create VectorStore instance with temporary storage.

    WHAT: Provides fresh VectorStore for each test
    WHY: Tests need clean state without cross-contamination
    HOW: Initialize VectorStore with temp directory
    DATA: VectorStore object ready for testing
    """
    return VectorStore(persist_directory=temp_chroma_dir)


@pytest.fixture
def sample_news_articles():
    """
    Sample news articles for testing.

    WHAT: Realistic news article data for testing vector operations
    WHY: Need consistent test data for reproducibility
    HOW: Return list of article dicts with varied content
    DATA: List of news article dictionaries
    """
    return [
        {
            "headline": "Apple reports record Q4 earnings",
            "content": "Apple Inc. announced record fourth quarter earnings today, "
                      "beating analyst expectations on strong iPhone sales.",
            "date": "2025-10-28",
            "source": "Reuters",
            "sentiment": 0.8,
        },
        {
            "headline": "Apple stock rises on earnings beat",
            "content": "Shares of Apple rose 5% in after-hours trading following "
                      "the company's better-than-expected earnings report.",
            "date": "2025-10-28",
            "source": "Bloomberg",
            "sentiment": 0.75,
        },
        {
            "headline": "Tech sector shows mixed results",
            "content": "The technology sector showed mixed performance this quarter, "
                      "with some companies beating expectations while others missed targets.",
            "date": "2025-10-27",
            "source": "CNBC",
            "sentiment": 0.0,
        },
    ]


@pytest.fixture
def sample_analysis():
    """
    Sample stock analysis for testing.

    WHAT: Realistic analysis data for testing pattern matching
    WHY: Need consistent test data for analysis storage
    HOW: Return tuple of (text, metadata)
    DATA: Analysis narrative + structured metadata
    """
    text = (
        "Strong bullish sentiment with positive earnings surprise. "
        "Technical indicators show upward momentum with increasing volume. "
        "Institutional buying detected in recent 13F filings."
    )
    metadata = {
        "date": "2025-10-28",
        "prediction": "up",
        "confidence": 0.85,
        "sentiment": "positive",
    }
    return text, metadata


# ===== INITIALIZATION TESTS =====


def test_vector_store_initialization(temp_chroma_dir):
    """
    Test VectorStore initializes correctly with persistent storage.

    WHAT: Verify VectorStore creates ChromaDB client and collections
    WHY: Initialization must work before any other operations
    HOW: Create VectorStore, check collections exist
    DATA: temp_dir → VectorStore → collection handles
    """
    # WHAT: Initialize VectorStore with temp directory
    # WHY: Should create client and three collections
    # HOW: Call VectorStore constructor
    # DATA: temp_dir path → VectorStore object
    store = VectorStore(persist_directory=temp_chroma_dir)

    # WHAT: Verify collections were created
    # WHY: All three collections should exist after init
    # HOW: Check collection attributes are not None
    # DATA: Collection handles should be initialized
    assert store.news_collection is not None
    assert store.analyses_collection is not None
    assert store.filings_collection is not None

    # WHAT: Verify persist directory was created
    # WHY: ChromaDB needs valid directory
    # HOW: Check Path exists
    # DATA: Directory should exist on filesystem
    assert Path(temp_chroma_dir).exists()


def test_vector_store_reset_on_init(temp_chroma_dir):
    """
    Test reset_on_init flag clears existing data.

    WHAT: Verify reset_on_init deletes pre-existing collections
    WHY: Testing requires ability to start with clean slate
    HOW: Create store, add data, recreate with reset=True, verify empty
    DATA: Populated store → reset → empty collections
    """
    # WHAT: Create initial store and add some data
    # WHY: Need existing data to verify reset works
    store = VectorStore(persist_directory=temp_chroma_dir)
    store.add_news_articles(
        ticker="AAPL",
        articles=[{"headline": "Test", "content": "Test content", "date": "2025-10-28"}]
    )

    # WHAT: Verify data was added
    # WHY: Confirm starting point before reset
    stats = store.get_collection_stats()
    assert stats["news"] == 1

    # WHAT: Create new store with reset flag
    # WHY: Should delete existing data
    # HOW: Initialize with reset_on_init=True
    # DATA: Same directory, reset flag → clean collections
    store2 = VectorStore(persist_directory=temp_chroma_dir, reset_on_init=True)

    # WHAT: Verify collections are empty
    # WHY: Reset should have deleted all data
    stats2 = store2.get_collection_stats()
    assert stats2["news"] == 0


# ===== NEWS ARTICLE TESTS =====


def test_add_news_articles_success(vector_store, sample_news_articles):
    """
    Test adding news articles to vector store.

    WHAT: Verify articles are stored with embeddings
    WHY: Core functionality for news semantic search
    HOW: Add articles, check return count and collection count
    DATA: Sample articles → vector store → count verification
    """
    # WHAT: Add news articles for AAPL
    # WHY: Test batch insertion
    # HOW: Call add_news_articles with fixture data
    # DATA: 3 articles → return count
    count = vector_store.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles
    )

    # WHAT: Verify all articles were added
    # WHY: Return value should match input count
    # HOW: Assert return value equals 3
    # DATA: count == 3
    assert count == 3

    # WHAT: Verify collection count matches
    # WHY: ChromaDB should have stored all 3 documents
    # HOW: Get stats, check news count
    # DATA: Collection stats should show 3 news items
    stats = vector_store.get_collection_stats()
    assert stats["news"] == 3


def test_add_news_articles_empty_list(vector_store):
    """
    Test adding empty article list returns 0.

    WHAT: Verify graceful handling of empty input
    WHY: Edge case - should not error, just return 0
    HOW: Call add_news_articles with empty list
    DATA: Empty list → return 0
    """
    # WHAT: Add empty articles list
    # WHY: Should handle gracefully
    count = vector_store.add_news_articles(
        ticker="AAPL",
        articles=[]
    )

    # WHAT: Verify 0 articles added
    # WHY: Empty input should result in 0 additions
    assert count == 0

    # WHAT: Verify collection is still empty
    stats = vector_store.get_collection_stats()
    assert stats["news"] == 0


def test_search_similar_news_finds_matches(vector_store, sample_news_articles):
    """
    Test semantic search finds similar news articles.

    WHAT: Verify similar news can be retrieved by query
    WHY: Core semantic search functionality
    HOW: Add articles, search with related query, verify results
    DATA: Articles → embeddings → query → similar results
    """
    # WHAT: Add sample news articles
    # WHY: Need data in store before searching
    vector_store.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles
    )

    # WHAT: Search for earnings-related news
    # WHY: Test semantic similarity (not keyword matching)
    # HOW: Query with earnings-related text
    # DATA: Query text → embedding → similar documents
    results = vector_store.search_similar_news(
        query="company reports strong financial results",
        ticker="AAPL",
        n_results=2
    )

    # WHAT: Verify results were found
    # WHY: Should find earnings-related articles
    # HOW: Check results list is not empty
    # DATA: results list should have entries
    assert len(results) > 0

    # WHAT: Verify results have required fields
    # WHY: API contract - should return structured data
    # HOW: Check first result has expected keys
    # DATA: Result dict should have headline, ticker, distance, etc.
    first_result = results[0]
    assert "headline" in first_result
    assert "ticker" in first_result
    assert "distance" in first_result
    assert "content" in first_result
    assert first_result["ticker"] == "AAPL"


def test_search_similar_news_ticker_filter(vector_store, sample_news_articles):
    """
    Test ticker filter limits search results.

    WHAT: Verify ticker parameter filters results to specific stock
    WHY: Users want to search within specific ticker's news
    HOW: Add articles for multiple tickers, search with filter
    DATA: Multi-ticker articles → filtered search → single ticker results
    """
    # WHAT: Add articles for AAPL
    vector_store.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles[:2]
    )

    # WHAT: Add articles for MSFT
    # WHY: Need multi-ticker data to test filter
    vector_store.add_news_articles(
        ticker="MSFT",
        articles=[{
            "headline": "Microsoft cloud revenue grows",
            "content": "Microsoft reported strong Azure cloud growth this quarter.",
            "date": "2025-10-28",
            "source": "WSJ",
        }]
    )

    # WHAT: Search with AAPL filter
    # WHY: Should only return AAPL news
    # HOW: Pass ticker="AAPL" parameter
    # DATA: Query → filtered search → AAPL results only
    results = vector_store.search_similar_news(
        query="technology company earnings",
        ticker="AAPL",
        n_results=5
    )

    # WHAT: Verify all results are for AAPL
    # WHY: Ticker filter should exclude MSFT
    # HOW: Check ticker field in each result
    # DATA: All results should have ticker="AAPL"
    assert len(results) > 0
    for result in results:
        assert result["ticker"] == "AAPL"


def test_search_similar_news_max_distance_filter(vector_store, sample_news_articles):
    """
    Test max_distance parameter filters distant matches.

    WHAT: Verify max_distance excludes low-similarity results
    WHY: Users may want only high-quality matches
    HOW: Search with strict max_distance, verify filtering
    DATA: Query → similarity scores → distance threshold → filtered results
    """
    # WHAT: Add sample articles
    vector_store.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles
    )

    # WHAT: Search with very strict distance threshold
    # WHY: Should exclude most/all results
    # HOW: Set max_distance very low (only exact matches)
    # DATA: Query with max_distance=0.1 → few/no results
    results = vector_store.search_similar_news(
        query="unrelated topic about weather",
        ticker="AAPL",
        n_results=5,
        max_distance=0.3  # Strict threshold
    )

    # WHAT: Verify results respect distance threshold
    # WHY: All returned results should be below threshold
    # HOW: Check distance field on each result
    # DATA: result["distance"] <= max_distance
    for result in results:
        assert result["distance"] <= 0.3


def test_search_similar_news_no_matches(vector_store):
    """
    Test search with no matching documents returns empty list.

    WHAT: Verify empty collection search returns []
    WHY: Graceful handling of empty results
    HOW: Search before adding any articles
    DATA: Empty collection → search → empty results
    """
    # WHAT: Search empty collection
    # WHY: Should return empty list, not error
    results = vector_store.search_similar_news(
        query="test query",
        ticker="AAPL",
        n_results=5
    )

    # WHAT: Verify empty results
    # WHY: No data in collection means no matches
    assert len(results) == 0


# ===== ANALYSIS TESTS =====


def test_add_analysis_success(vector_store, sample_analysis):
    """
    Test adding stock analysis to vector store.

    WHAT: Verify analysis text and metadata are stored
    WHY: Core functionality for analysis pattern matching
    HOW: Add analysis, verify return value and collection count
    DATA: Analysis text + metadata → vector store → success
    """
    # WHAT: Add sample analysis
    # WHY: Test analysis storage
    text, metadata = sample_analysis
    success = vector_store.add_analysis(
        ticker="AAPL",
        analysis_text=text,
        metadata=metadata
    )

    # WHAT: Verify success return
    # WHY: Should return True on successful add
    assert success is True

    # WHAT: Verify collection count increased
    # WHY: ChromaDB should have stored the analysis
    stats = vector_store.get_collection_stats()
    assert stats["analyses"] == 1


def test_search_similar_analyses(vector_store, sample_analysis):
    """
    Test searching for similar historical analyses.

    WHAT: Verify similar analyses can be retrieved by query
    WHY: Pattern matching functionality for predictions
    HOW: Add analysis, search with similar query, verify results
    DATA: Analysis → embedding → query → similar matches
    """
    # WHAT: Add sample analysis
    text, metadata = sample_analysis
    vector_store.add_analysis(
        ticker="AAPL",
        analysis_text=text,
        metadata=metadata
    )

    # WHAT: Search for similar situation
    # WHY: Test semantic matching on analysis narratives
    # HOW: Query with related description
    # DATA: Query → embedding → similarity → results
    results = vector_store.search_similar_analyses(
        query="bullish momentum with strong institutional buying",
        ticker="AAPL",
        n_results=5
    )

    # WHAT: Verify results found
    # WHY: Should match the stored analysis
    assert len(results) > 0

    # WHAT: Verify result structure
    # WHY: Should return analysis with metadata
    first_result = results[0]
    assert "ticker" in first_result
    assert "prediction" in first_result
    assert "confidence" in first_result
    assert "analysis_text" in first_result
    assert "distance" in first_result


# ===== COLLECTION MANAGEMENT TESTS =====


def test_get_collection_stats(vector_store, sample_news_articles, sample_analysis):
    """
    Test collection statistics reporting.

    WHAT: Verify get_collection_stats returns correct counts
    WHY: Monitoring functionality for debugging
    HOW: Add data to multiple collections, check stats
    DATA: Mixed data → stats query → counts
    """
    # WHAT: Add news articles
    vector_store.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles
    )

    # WHAT: Add analysis
    text, metadata = sample_analysis
    vector_store.add_analysis(
        ticker="AAPL",
        analysis_text=text,
        metadata=metadata
    )

    # WHAT: Get collection stats
    # WHY: Should show counts for each collection
    stats = vector_store.get_collection_stats()

    # WHAT: Verify counts are correct
    # WHY: Stats should match what we added
    assert stats["news"] == 3
    assert stats["analyses"] == 1
    assert stats["filings"] == 0  # Haven't added any filings


def test_reset_collection(vector_store, sample_news_articles):
    """
    Test resetting a specific collection.

    WHAT: Verify reset_collection clears target collection only
    WHY: Need ability to clear specific collection for reindexing
    HOW: Add data to multiple collections, reset one, verify counts
    DATA: Populated collections → reset one → verify isolation
    """
    # WHAT: Add data to news collection
    vector_store.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles
    )

    # WHAT: Add data to analyses collection
    vector_store.add_analysis(
        ticker="AAPL",
        analysis_text="Test analysis",
        metadata={"date": "2025-10-28", "prediction": "up", "confidence": 0.8}
    )

    # WHAT: Verify both collections have data
    stats_before = vector_store.get_collection_stats()
    assert stats_before["news"] == 3
    assert stats_before["analyses"] == 1

    # WHAT: Reset news collection only
    # WHY: Should clear news, leave analyses intact
    success = vector_store.reset_collection("news")
    assert success is True

    # WHAT: Verify news is empty, analyses unchanged
    # WHY: Reset should be isolated to target collection
    stats_after = vector_store.get_collection_stats()
    assert stats_after["news"] == 0
    assert stats_after["analyses"] == 1  # Unchanged


def test_reset_collection_invalid_name(vector_store):
    """
    Test reset_collection rejects invalid collection names.

    WHAT: Verify invalid collection name returns False
    WHY: Should not allow resetting arbitrary collections
    HOW: Try to reset non-existent collection
    DATA: Invalid name → False return
    """
    # WHAT: Try to reset invalid collection
    # WHY: Should reject and return False
    success = vector_store.reset_collection("invalid_collection")

    # WHAT: Verify failure
    # WHY: Invalid names should not be accepted
    assert success is False


# ===== EDGE CASES AND ERROR HANDLING =====


def test_add_articles_with_missing_fields(vector_store):
    """
    Test articles with missing optional fields are handled gracefully.

    WHAT: Verify articles with minimal fields still work
    WHY: Not all fields are required
    HOW: Add article with only headline, verify storage
    DATA: Minimal article → vector store → success
    """
    # WHAT: Article with only required fields
    # WHY: Optional fields should not be required
    minimal_article = {
        "headline": "Test headline",
        "content": "Test content",
        # Missing: date, source, url, sentiment
    }

    # WHAT: Add minimal article
    # WHY: Should handle missing optional fields
    count = vector_store.add_news_articles(
        ticker="TEST",
        articles=[minimal_article]
    )

    # WHAT: Verify success
    # WHY: Missing optional fields shouldn't cause failure
    assert count == 1

    # WHAT: Verify can search for it
    # WHY: Should be retrievable despite missing fields
    results = vector_store.search_similar_news(
        query="test",
        ticker="TEST",
        n_results=1
    )

    assert len(results) == 1
    assert results[0]["headline"] == "Test headline"


def test_persistence_across_instances(temp_chroma_dir, sample_news_articles):
    """
    Test data persists across VectorStore instances.

    WHAT: Verify data survives VectorStore recreation
    WHY: ChromaDB should provide persistent storage
    HOW: Create store, add data, recreate store, verify data still there
    DATA: Store 1 → add data → close → Store 2 → data still present
    """
    # WHAT: Create first store and add data
    store1 = VectorStore(persist_directory=temp_chroma_dir)
    store1.add_news_articles(
        ticker="AAPL",
        articles=sample_news_articles
    )

    # WHAT: Get stats from first store
    stats1 = store1.get_collection_stats()
    assert stats1["news"] == 3

    # WHAT: Create second store with same directory
    # WHY: Should load existing data
    store2 = VectorStore(persist_directory=temp_chroma_dir)

    # WHAT: Verify data persisted
    # WHY: Should have same data as store1
    stats2 = store2.get_collection_stats()
    assert stats2["news"] == 3

    # WHAT: Verify can search persisted data
    # WHY: Embeddings should still be searchable
    results = store2.search_similar_news(
        query="earnings report",
        ticker="AAPL",
        n_results=5
    )

    assert len(results) > 0
