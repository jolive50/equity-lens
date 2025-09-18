# Free Market & Sentiment Data Sources

This playbook summarizes how the project can operationalize freely accessible market and
news data providers alongside the FinBERT sentiment model. Each subsection links a
source to the scaffolding modules created under `pipelines/` so that implementation can
start once credentials are available.

## Historical Market Data Providers

### Yahoo Finance (`yfinance`)
- **Module:** `pipelines.batch.historical.yahoo_finance`
- **Usage Plan:** Schedule batch jobs that hydrate the `YahooFinanceConfig` with lists of tickers
  and rolling date windows. Persisted files land under `data/batch/raw/market/yahoo_finance/`
  for downstream feature engineering.

### Plotly Sample OHLC Archives (MIT Licensed)
- **Type:** Static CSV exports of Apple (AAPL) OHLCV data prepared for the Plotly open-data
  gallery.
- **Access URL:**
  `https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv`
- **Usage Plan:** Mirror the file into `data/batch/raw/market/yahoo_finance/free_download/` via
  `curl` or `wget` for immediate experimentation without requiring API credentials. The dataset
  includes Bollinger Band features that can jump-start feature-store prototyping.

### Plotly Multi-Ticker Sample (MIT Licensed)
- **Type:** Multivariate CSV spanning Microsoft, IBM, Starbucks, Apple, and S&P 500 daily closes.
- **Access URL:** `https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv`
- **Usage Plan:** Store alongside other Yahoo Finance snapshots to exercise multivariate feature
  engineering or to back quickstart notebooks when network access is constrained.

### Alpha Vantage
- **Module:** `pipelines.batch.historical.alpha_vantage`
- **Usage Plan:** Rotate through tickers while respecting the free-tier throttling window
  (5 API calls/minute). The scaffold is pre-wired for retries, checkpointing, and storage in
  `data/batch/raw/market/alpha_vantage/`.

### Tiingo
- **Module:** `pipelines.batch.historical.tiingo`
- **Usage Plan:** Run nightly EOD jobs limited to the 500-call free tier. Configure the
  `TiingoConfig` with session retry behavior and date bounds, then land results in
  `data/batch/raw/market/tiingo/`.

### IEX Cloud
- **Module:** `pipelines.batch.historical.iex_cloud`
- **Usage Plan:** Track message counts against the 50,000 message quota. The placeholder
  function highlights the need for checkpointed persistence and rate accounting so that
  orchestrators (Airflow/Prefect) can resume safely.

### Nasdaq Data Link
- **Module:** `pipelines.batch.historical.nasdaq_data_link`
- **Usage Plan:** Configure dataset codes for free tables (e.g., Wiki EOD, FRED macro
  releases). Implementers should add schema harmonization before writing into
  `data/batch/raw/market/nasdaq_data_link/`.

## Realtime & News Sentiment Sources

### Yahoo Finance Headlines
- **Module:** `pipelines.realtime.news.yahoo_finance`
- **Usage Plan:** Poll the `Ticker.news` endpoint via `yfinance` on 5-minute intervals
  to populate `data/realtime/landing/news/yahoo_finance/` for nearline processing.

### NewsAPI
- **Module:** `pipelines.realtime.news.newsapi`
- **Usage Plan:** Issue keyword-expanded queries tied to S&P 500 tickers. Store responses in
  `data/realtime/landing/news/newsapi/` while enforcing the 100-requests-per-day cap
  through scheduler controls.

### Finnhub.io
- **Module:** `pipelines.realtime.news.finnhub`
- **Usage Plan:** Cycle through symbol lists with 60-call/minute pacing. Persist articles to
  `data/realtime/landing/news/finnhub/` for immediate sentiment scoring.

### Reddit (WallStreetBets, etc.)
- **Module:** `pipelines.realtime.news.social_reddit`
- **Usage Plan:** Stream subreddit posts/comments via PRAW or Pushshift clones. Normalize
  text and stash events in `data/realtime/landing/social/reddit/` for FinBERT inference.

### Twitter/X (FinTwit)
- **Module:** `pipelines.realtime.news.social_twitter`
- **Usage Plan:** Consume filtered streams or search endpoints using third-party wrappers.
  Outputs persist under `data/realtime/landing/social/twitter/` with reconnect logic to
  respect rate limits.

## Sentiment Modeling

### FinBERT
- **Module:** `pipelines.realtime.sentiment.finbert`
- **Usage Plan:** Load `ProsusAI/finbert` via Hugging Face Transformers, batch inputs from
  the realtime landing zones, and write scored outputs into
  `data/realtime/feature_store/sentiment/finbert/`.

## Next Implementation Steps
1. Secure API keys for Alpha Vantage, Tiingo, Finnhub, NewsAPI, and social wrappers.
2. Implement the placeholder functions with proper retry, logging, and validation logic.
3. Wire the modules into orchestrated jobs (Airflow/Prefect) and sentiment pipelines.
4. Add unit tests that mock provider responses to maintain coverage without hitting live APIs.
