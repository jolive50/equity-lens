import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd


class Database:
    def __init__(self, db_path: str = "freshstart.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
        if not schema_path.exists():
            raise RuntimeError(f"Schema file not found: {schema_path}")

        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def cache_prices(self, ticker: str, price_data: pd.DataFrame) -> int:
        if price_data.empty:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
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
                    continue
            conn.commit()
        return inserted

    def get_cached_prices(self, ticker: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM price_cache WHERE ticker = ?"
        params = [ticker]

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
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def cache_news(self, ticker: str, news_articles: List[Dict[str, Any]]) -> int:
        if not news_articles:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
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
                except Exception:
                    continue
            conn.commit()
        return inserted

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
        params = [ticker]

        if max_age_minutes is not None:
            cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
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

        return [dict(row) for row in rows]

    def save_analysis(self, ticker: str, analysis_data: Dict[str, Any]) -> int:
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
            return cursor.lastrowid

    def get_analysis_history(self, ticker: str, limit: Optional[int] = 10) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM analysis_results
            WHERE ticker = ?
            ORDER BY analysis_date DESC
        """
        params = [ticker]

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
