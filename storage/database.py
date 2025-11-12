import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger("freshstart.storage")


class Database:
    def __init__(self, db_path: str = "db/freshstart.db"):
        self.db_path = db_path
        self._connection_pool = []
        self._init_database()

    def _init_database(self):
        schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
        if not schema_path.exists():
            raise RuntimeError(f"Schema file not found: {schema_path}")

        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())

        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def cache_prices(self, ticker: str, price_data: pd.DataFrame) -> int:
        if price_data.empty:
            logger.warning(f"Attempted to cache empty price data for {ticker}")
            return 0

        logger.info(f"      📀 DATABASE: Caching price data for {ticker}")
        logger.info(f"         → Rows to insert: {len(price_data)}")
        logger.info(f"         → Date range: {price_data.index.min()} to {price_data.index.max()}")

        # Log sample of first row
        first_row = price_data.iloc[0]
        logger.info(f"         → Sample data (first row):")
        logger.info(f"            Date: {price_data.index[0]}")
        logger.info(f"            Open: ${first_row.get('Open', 0):.2f}")
        logger.info(f"            High: ${first_row.get('High', 0):.2f}")
        logger.info(f"            Low: ${first_row.get('Low', 0):.2f}")
        logger.info(f"            Close: ${first_row.get('Close', 0):.2f}")
        logger.info(f"            Volume: {int(first_row.get('Volume', 0)):,}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
            errors = 0
            for idx, row in price_data.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO price_cache
                        (ticker, date, open, high, low, close, volume, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker,
                        str(idx),
                        float(row.get('Open', 0)),
                        float(row.get('High', 0)),
                        float(row.get('Low', 0)),
                        float(row.get('Close', 0)),
                        int(row.get('Volume', 0)),
                        datetime.now().isoformat()
                    ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert price row for {ticker} on {idx}: {e}")
                    errors += 1
            conn.commit()

        logger.info(f"      ✅ DATABASE: Cached {inserted} price rows for {ticker} ({errors} errors)")
        return inserted

    def is_price_cache_fresh(self, ticker: str, max_age_hours: int = 1) -> bool:
        """Check if cached price data is fresh enough.

        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age in hours (default 1 hour)

        Returns:
            True if cache exists and is fresh, False otherwise
        """
        query = """
            SELECT MAX(fetched_at) as last_fetch
            FROM price_cache
            WHERE ticker = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            row = cursor.fetchone()

        if not row or not row['last_fetch']:
            logger.debug(f"No cache found for {ticker} prices")
            return False

        last_fetch = datetime.fromisoformat(row['last_fetch'])
        age = datetime.now() - last_fetch
        is_fresh = age < timedelta(hours=max_age_hours)

        if is_fresh:
            logger.debug(f"Price cache for {ticker} is fresh (age: {age.total_seconds():.0f}s)")
        else:
            logger.info(f"Price cache for {ticker} is stale (age: {age.total_seconds():.0f}s, max: {max_age_hours}h)")

        return is_fresh

    def get_cached_prices(self, ticker: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM price_cache WHERE ticker = ?"
        params: List[Any] = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            logger.info(f"Cache MISS for {ticker} prices (no cached data)")
            return pd.DataFrame()

        logger.info(f"Cache HIT for {ticker} prices ({len(df)} rows found)")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def cache_news(self, ticker: str, news_articles: List[Dict[str, Any]]) -> int:
        if not news_articles:
            logger.warning(f"Attempted to cache empty news data for {ticker}")
            return 0

        logger.info(f"      📀 DATABASE: Caching news articles for {ticker}")
        logger.info(f"         → Articles to insert: {len(news_articles)}")

        # Log sample of first article
        first_article = news_articles[0]
        logger.info(f"         → Sample article (first):")
        logger.info(f"            Title: {first_article.get('title', 'N/A')[:60]}...")
        logger.info(f"            Source: {first_article.get('source', 'N/A')}")
        logger.info(f"            Published: {first_article.get('published_at', 'N/A')}")
        logger.info(f"            Sentiment: {first_article.get('sentiment_label', 'N/A')} ({first_article.get('sentiment_score', 0):.2f})")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
            errors = 0
            for article in news_articles:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO news_cache
                        (ticker, title, url, published_at, source, summary,
                         sentiment_label, sentiment_score, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker,
                        article.get('title', ''),
                        article.get('url', ''),
                        article.get('published_at', ''),
                        article.get('source', ''),
                        article.get('summary', ''),
                        article.get('sentiment_label'),
                        article.get('sentiment_score'),
                        datetime.now().isoformat()
                    ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert news article for {ticker}: {e}")
                    errors += 1
            conn.commit()

        logger.info(f"      ✅ DATABASE: Cached {inserted} news articles for {ticker} ({errors} errors)")
        return inserted

    def is_news_cache_fresh(self, ticker: str, max_age_hours: int = 1) -> bool:
        """Check if cached news data is fresh enough."""
        query = """
            SELECT MAX(fetched_at) as last_fetch
            FROM news_cache
            WHERE ticker = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            row = cursor.fetchone()

        if not row or not row['last_fetch']:
            logger.debug(f"No cache found for {ticker} news")
            return False

        last_fetch = datetime.fromisoformat(row['last_fetch'])
        age = datetime.now() - last_fetch
        is_fresh = age < timedelta(hours=max_age_hours)

        if is_fresh:
            logger.debug(f"News cache for {ticker} is fresh (age: {age.total_seconds():.0f}s)")
        else:
            logger.info(
                f"News cache for {ticker} is stale (age: {age.total_seconds():.0f}s, max: {max_age_hours}h)"
            )

        return is_fresh

    def get_cached_news(
        self,
        ticker: str,
        limit: Optional[int] = 50,
        max_age_minutes: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT title, url, published_at, source, summary,
                   sentiment_label, sentiment_score, fetched_at
            FROM news_cache
            WHERE ticker = ?
        """
        params: List[Any] = [ticker]

        if max_age_minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
            query += " AND fetched_at >= ?"
            params.append(cutoff.isoformat())

        query += " ORDER BY published_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        result = [dict(row) for row in rows]
        if result:
            logger.info(f"Cache HIT for {ticker} news ({len(result)} articles found)")
        else:
            logger.info(f"Cache MISS for {ticker} news (no cached data)")

        return result

    def save_analysis(self, ticker: str, analysis_data: Dict[str, Any]) -> int:
        logger.info(f"\n      📀 DATABASE: Saving analysis result for {ticker}")
        logger.info(f"         → Prediction Direction: {analysis_data.get('prediction_direction')}")
        logger.info(f"         → Prediction Confidence: {analysis_data.get('prediction_confidence'):.1%}")
        logger.info(f"         → Prediction Model: {analysis_data.get('prediction_model')}")
        logger.info(f"         → Sentiment Label: {analysis_data.get('sentiment_label')}")
        logger.info(f"         → Sentiment Score: {analysis_data.get('sentiment_score'):.2f}")
        logger.info(f"         → Sentiment Model: {analysis_data.get('sentiment_model')}")
        logger.info(f"         → Reflection Warnings: {analysis_data.get('reflection_warnings')}")
        logger.info(f"         → Explanation Length: {len(analysis_data.get('explanation', ''))} chars")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_results
                (ticker, prediction_direction, prediction_confidence, prediction_model,
                 sentiment_score, sentiment_label, sentiment_model,
                 explanation, reflection_warnings, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                analysis_data.get('prediction_direction'),
                analysis_data.get('prediction_confidence'),
                analysis_data.get('prediction_model'),
                analysis_data.get('sentiment_score'),
                analysis_data.get('sentiment_label'),
                analysis_data.get('sentiment_model'),
                analysis_data.get('explanation'),
                analysis_data.get('reflection_warnings'),
                json.dumps(analysis_data.get('raw_data', {}))
            ))
            conn.commit()
            row_id = cursor.lastrowid

        logger.info(f"      ✅ DATABASE: Analysis saved with ID={row_id}")

        if row_id is None:
            raise RuntimeError("Failed to persist analysis result")
        return int(row_id)

    def get_analysis_history(self, ticker: str, limit: Optional[int] = 10) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM analysis_results
            WHERE ticker = ?
            ORDER BY analysis_date DESC
        """
        params: List[Any] = [ticker]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            result = dict(row)
            if result.get('raw_data'):
                try:
                    result['raw_data'] = json.loads(result['raw_data'])
                except:
                    pass
            results.append(result)
        return results

    def clear_cache(self, table: Optional[str] = None):
        tables = [table] if table else ['price_cache', 'news_cache', 'analysis_results']
        with self._get_connection() as conn:
            for tbl in tables:
                conn.execute(f"DELETE FROM {tbl}")
            conn.commit()

    def close(self):
        """Close all open database connections."""
        for conn in self._connection_pool:
            try:
                conn.close()
            except:
                pass
        self._connection_pool.clear()
        logger.debug(f"Database connections closed for {self.db_path}")
