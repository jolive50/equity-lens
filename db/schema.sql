-- Equity Lens MVP Database Schema
-- Byeol's database design for caching and analysis storage

-- Price cache table - stores all fetched price data to avoid redundant API calls
CREATE TABLE IF NOT EXISTS price_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Index for fast ticker-based lookups
CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_cache(ticker, date DESC);

-- News cache table - stores all fetched news articles to avoid redundant API calls
CREATE TABLE IF NOT EXISTS news_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    published_at TEXT,
    source TEXT,
    summary TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, url)
);

-- Index for fast ticker and date-based news retrieval
CREATE INDEX IF NOT EXISTS idx_news_ticker_published ON news_cache(ticker, published_at DESC);

-- Analysis results table - stores complete analysis results for user history
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_direction TEXT,
    prediction_confidence REAL,
    prediction_model TEXT,
    sentiment_score REAL,
    sentiment_label TEXT,
    sentiment_model TEXT,
    explanation TEXT,
    reflection_warnings TEXT,
    raw_data TEXT
);

-- Index for fast ticker-based analysis history retrieval
CREATE INDEX IF NOT EXISTS idx_analysis_ticker_date ON analysis_results(ticker, analysis_date DESC);
