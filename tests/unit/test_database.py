import pytest
import pandas as pd
from datetime import datetime, timedelta


def test_database_initialization(test_db):
    """Test that database initializes with correct schema."""
    conn = test_db._get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    assert 'price_cache' in tables
    assert 'news_cache' in tables
    assert 'analysis_results' in tables
    conn.close()


def test_cache_prices(test_db, sample_price_data, mock_ticker):
    """Test caching price data."""
    inserted = test_db.cache_prices(mock_ticker, sample_price_data)
    assert inserted > 0
    assert inserted == len(sample_price_data)


def test_get_cached_prices(test_db, sample_price_data, mock_ticker):
    """Test retrieving cached price data."""
    test_db.cache_prices(mock_ticker, sample_price_data)
    cached_data = test_db.get_cached_prices(mock_ticker)

    assert not cached_data.empty
    assert len(cached_data) == len(sample_price_data)
    assert 'close' in cached_data.columns
    assert 'volume' in cached_data.columns


def test_get_cached_prices_with_date_range(test_db, sample_price_data, mock_ticker):
    """Test retrieving cached prices with date filtering."""
    test_db.cache_prices(mock_ticker, sample_price_data)

    start_date = "2024-10-20"
    end_date = "2024-10-25"
    cached_data = test_db.get_cached_prices(mock_ticker, start_date, end_date)

    assert not cached_data.empty
    assert len(cached_data) <= len(sample_price_data)


def test_cache_prices_empty_dataframe(test_db, mock_ticker):
    """Test caching empty price data."""
    empty_df = pd.DataFrame()
    inserted = test_db.cache_prices(mock_ticker, empty_df)
    assert inserted == 0


def test_cache_news(test_db, sample_news_data, mock_ticker):
    """Test caching news articles."""
    inserted = test_db.cache_news(mock_ticker, sample_news_data)
    assert inserted > 0
    assert inserted == len(sample_news_data)


def test_get_cached_news(test_db, sample_news_data, mock_ticker):
    """Test retrieving cached news articles."""
    test_db.cache_news(mock_ticker, sample_news_data)
    cached_news = test_db.get_cached_news(mock_ticker)

    assert len(cached_news) > 0
    assert cached_news[0]['title'] is not None
    assert 'sentiment_score' in cached_news[0]


def test_get_cached_news_with_limit(test_db, sample_news_data, mock_ticker):
    """Test retrieving limited number of news articles."""
    test_db.cache_news(mock_ticker, sample_news_data)
    limit = 3
    cached_news = test_db.get_cached_news(mock_ticker, limit=limit)

    assert len(cached_news) <= limit


def test_get_cached_news_max_age_filter(test_db, sample_news_data, mock_ticker):
    """Test that cached news respects max_age_minutes filter."""
    test_db.cache_news(mock_ticker, sample_news_data)

    # Mark cached rows as old
    cutoff = (datetime.now() - timedelta(minutes=90)).isoformat()
    with test_db._get_connection() as conn:
        conn.execute(
            "UPDATE news_cache SET fetched_at = ? WHERE ticker = ?",
            (cutoff, mock_ticker)
        )
        conn.commit()

    cached_news = test_db.get_cached_news(mock_ticker, max_age_minutes=30)
    assert len(cached_news) == 0


def test_cache_news_empty_list(test_db, mock_ticker):
    """Test caching empty news list."""
    inserted = test_db.cache_news(mock_ticker, [])
    assert inserted == 0


def test_save_analysis(test_db, sample_analysis_result, mock_ticker):
    """Test saving analysis results."""
    analysis_id = test_db.save_analysis(mock_ticker, sample_analysis_result)
    assert analysis_id > 0


def test_get_analysis_history(test_db, sample_analysis_result, mock_ticker):
    """Test retrieving analysis history."""
    test_db.save_analysis(mock_ticker, sample_analysis_result)
    test_db.save_analysis(mock_ticker, sample_analysis_result)

    history = test_db.get_analysis_history(mock_ticker)
    assert len(history) == 2
    assert history[0]['ticker'] == mock_ticker
    assert history[0]['prediction_direction'] == 'up'


def test_get_analysis_history_with_limit(test_db, sample_analysis_result, mock_ticker):
    """Test retrieving limited analysis history."""
    for _ in range(5):
        test_db.save_analysis(mock_ticker, sample_analysis_result)

    limit = 2
    history = test_db.get_analysis_history(mock_ticker, limit=limit)
    assert len(history) == limit


def test_clear_cache_single_table(test_db, sample_price_data, mock_ticker):
    """Test clearing specific cache table."""
    test_db.cache_prices(mock_ticker, sample_price_data)
    test_db.clear_cache('price_cache')

    cached_data = test_db.get_cached_prices(mock_ticker)
    assert cached_data.empty


def test_clear_cache_all_tables(test_db, sample_price_data, sample_news_data, mock_ticker):
    """Test clearing all cache tables."""
    test_db.cache_prices(mock_ticker, sample_price_data)
    test_db.cache_news(mock_ticker, sample_news_data)
    test_db.clear_cache()

    cached_prices = test_db.get_cached_prices(mock_ticker)
    cached_news = test_db.get_cached_news(mock_ticker)

    assert cached_prices.empty
    assert len(cached_news) == 0


def test_price_cache_unique_constraint(test_db, sample_price_data, mock_ticker):
    """Test that duplicate prices are handled correctly."""
    inserted_first = test_db.cache_prices(mock_ticker, sample_price_data)
    inserted_second = test_db.cache_prices(mock_ticker, sample_price_data)

    assert inserted_first == inserted_second
    cached_data = test_db.get_cached_prices(mock_ticker)
    assert len(cached_data) == len(sample_price_data)


def test_news_cache_unique_constraint(test_db, sample_news_data, mock_ticker):
    """Test that duplicate news are handled correctly."""
    inserted_first = test_db.cache_news(mock_ticker, sample_news_data)
    inserted_second = test_db.cache_news(mock_ticker, sample_news_data)

    assert inserted_first == inserted_second
    cached_news = test_db.get_cached_news(mock_ticker)
    assert len(cached_news) == len(sample_news_data)
