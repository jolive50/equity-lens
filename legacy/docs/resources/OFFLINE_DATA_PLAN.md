# Offline Modeling Data Requirements

This checklist summarizes the real-world datasets needed to train and validate the `ProbabilisticForecaster` without relying on synthetic values.

## 1. Historical Price Candles
- **Format**: CSV
- **Columns**: `date`, `open`, `high`, `low`, `close`, `volume`
- **Frequency**: Daily candles covering at least 3 years per ticker
- **Source Suggestions**:
  - Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED`
  - Tiingo End-of-Day Equity Prices
  - Nasdaq Data Link Quandl `WIKI` archive

## 2. Fundamental Snapshots
- **Format**: CSV or Parquet
- **Columns**: `ticker`, `as_of_date`, `revenue`, `net_income`, `ebitda`, `total_debt`, `shareholders_equity`, `market_cap`, `roe`, `pe_ratio`
- **Frequency**: Quarterly statements aligned to price history
- **Source Suggestions**:
  - SEC EDGAR filings ingested via financial modeling APIs
  - Alpha Vantage fundamental endpoints (premium plan)

## 3. Label Construction
- **Format**: CSV
- **Columns**: `date`, `ticker`, `target_return_5d`, `target_direction`
- **Computation**: Forward 5-day log return, mapped to `up`, `down`, `neutral` thresholds (+/-1%)
- **Note**: Generate locally after downloading price candles.

## 4. Storage Layout
- Place raw files under `data/training/raw/` using subfolders per ticker.
- Store derived features and labels under `data/training/features/` and `data/training/labels/` respectively.
- Maintain a `README.md` inside each folder documenting extraction timestamps and API quota usage.

Collecting these CSV datasets before training ensures the forecasting pipeline operates entirely on verifiable, real-world market data.
