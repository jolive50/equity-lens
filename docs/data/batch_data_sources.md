# Batch Data Sources and Ingestion Plan

## 1. Market Microstructure & Pricing Data
- **Primary Providers:** Polygon.io, Alpha Vantage, Twelve Data (equities, ETFs), Tiingo (equities + news), Quandl (macro & alternative datasets).
- **Content:** OHLCV bars (1-min to daily), corporate actions, dividends, splits.
- **Acquisition Method:** REST API with pagination + bulk download endpoints; schedule nightly ingestion jobs via Airflow.
- **Storage Target:** `data/batch/raw/market/` (Parquet). Post-transformations stored in `data/batch/processed/features/`.
- **Use Cases:** Feature generation (returns, volatility, technical indicators), backtesting, training autoregressive price models.

## 2. Fundamentals & Financial Statements
- **Primary Providers:** Financial Modeling Prep, Intrinio, SEC EDGAR bulk filings, Refinitiv (if licensed).
- **Content:** Income statements, balance sheets, cash flow statements, key ratios, earnings transcripts.
- **Acquisition Method:** Combination of API pulls (quarterly) and SEC bulk ZIP downloads processed with Python ETL jobs.
- **Storage Target:** `data/batch/raw/fundamentals/` with schema validation using Pydantic models.
- **Use Cases:** Enrich factor models, build valuation signals, provide context to explanation agents.

## 3. Alternative & Sentiment Corpora
- **Primary Providers:** RavenPack, GDELT, Bloomberg ESG, FRED (macro), Kaggle competition datasets.
- **Content:** News sentiment scores, ESG indicators, macroeconomic releases, event metadata.
- **Acquisition Method:** Licensed data feeds (SFTP/API) plus curated public CSV downloads via scripted jobs.
- **Storage Target:** `data/batch/raw/alt_data/` partitioned by source and date.
- **Use Cases:** Augment sentiment agent, event-driven features, risk monitoring.

## 4. Labeling & Ground-Truth Datasets
- **Sources:** Historical analyst recommendations, price targets, curated internal annotations, simulated labels from strategy rules.
- **Acquisition Method:** Internal exports, manual labeling workflows, and generation scripts stored under `pipelines/batch/labelling/` (to be implemented).
- **Storage Target:** `data/batch/processed/labels/` with versioning tracked via DVC or LakeFS.
- **Use Cases:** Supervised fine-tuning, evaluation benchmarks, RLHF reward models.

## 5. Data Governance & Quality Controls
- Validate schema and null thresholds with Great Expectations suites committed alongside ingestion code.
- Track dataset lineage using DataHub and maintain retention policies aligned with licensing agreements.
- Implement checksum verification for bulk downloads; log ingestion metrics to observability stack.

## 6. Next Steps
1. Prioritize provider contracts and obtain API keys/test datasets.
2. Prototype ingestion notebooks for top-priority sources (pricing + news) and baseline storage conventions.
3. Define feature engineering jobs in `pipelines/batch/` to standardize parquet schema across sources.
