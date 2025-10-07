"""Download stock fundamental data from Kaggle and other sources.

This script downloads comprehensive fundamental financial metrics for stocks:
- P/E Ratio (Price to Earnings)
- EBITDA, EBIT, EBT
- Operating Cash Flow, Free Cash Flow
- Discounted Cash Flow
- Earnings Per Share (EPS)
- EV/EBITDA, EV/Sales
- Enterprise Value, Market Cap
- Price to Sales, Price to Book
- Return on Assets (ROA), Return on Equity (ROE), Return on Invested Capital (ROIC)
- Growth metrics
- And more!

College-Level Explanation:
Fundamental data = company's financial health (like a report card)
Price data = what people pay for the stock (market sentiment)
Both together = better predictions (fundamentals + sentiment)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FundamentalsDownloader:
    """Downloads and manages fundamental financial data for stocks.

    What this does: Fetches P/E, EBITDA, cash flow, and other financial metrics
    Why: Fundamental data is crucial for stock valuation and prediction
    How: Downloads from Kaggle datasets with comprehensive financial data

    College-Level Analogy:
    Price data tells you what the stock costs today.
    Fundamental data tells you what the company is actually worth.
    Like looking at a house's price vs. its foundation, plumbing, and condition.
    """

    # Kaggle datasets with fundamental data
    FUNDAMENTAL_DATASETS = {
        "sp500_financials": {
            "id": "paytonfisher/sp-500-companies-with-financial-information",
            "description": "S&P 500 companies with financial metrics",
            "metrics": ["Market Cap", "P/E Ratio", "EPS", "Dividend Yield"]
        },
        "nyse_fundamentals": {
            "id": "dgawlik/nyse",
            "description": "NYSE fundamentals with quarterly/annual data",
            "metrics": ["Revenue", "EBITDA", "ROA", "ROE", "Debt/Equity"]
        },
        "sec_financials": {
            "id": "cnic92/200-financial-indicators-of-us-stocks-20142018",
            "description": "200+ financial indicators from SEC filings",
            "metrics": ["Cash Flow", "Operating Income", "Total Assets", "many more"]
        }
    }

    def __init__(self, output_dir: Path = Path("data/fundamentals")):
        """Initialize downloader with output directory.

        Args:
            output_dir: Where to save downloaded fundamental data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def download_sp500_financials(self) -> Path:
        """Download S&P 500 companies with financial information.

        What: Basic financial metrics for S&P 500 companies
        Why: Good starting point, covers major stocks
        How: Downloads CSV with key financial ratios

        Returns:
            Path to downloaded data

        Metrics included:
        - Symbol, Company Name
        - Sector, Industry
        - Market Cap
        - P/E Ratio (Price to Earnings)
        - EPS (Earnings Per Share)
        - Dividend Yield
        - 52 Week Price Range
        - Revenue, EBITDA
        """
        dataset_id = self.FUNDAMENTAL_DATASETS["sp500_financials"]["id"]
        logger.info(f"Downloading S&P 500 financial data: {dataset_id}")

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            output_path = self.output_dir / "sp500_financials"
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading to: {output_path}")
            api.dataset_download_files(
                dataset_id,
                path=str(output_path),
                unzip=True
            )

            csv_files = list(output_path.glob("*.csv"))
            logger.info(f"✓ Downloaded {len(csv_files)} files")

            for csv_file in csv_files:
                logger.info(f"  - {csv_file.name}")

            return output_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def download_nyse_fundamentals(self) -> Path:
        """Download NYSE fundamental data (quarterly/annual financials).

        What: Detailed quarterly and annual financial statements
        Why: Time-series fundamental data for trend analysis
        How: Downloads comprehensive financial history

        Returns:
            Path to downloaded data

        Metrics included:
        - Income Statement: Revenue, Operating Income, Net Income, EPS
        - Balance Sheet: Total Assets, Total Liabilities, Stockholders Equity
        - Cash Flow: Operating CF, Investing CF, Financing CF, Free CF
        - Ratios: ROA, ROE, Profit Margin, Debt/Equity
        - Growth: Revenue Growth, Earnings Growth
        """
        dataset_id = self.FUNDAMENTAL_DATASETS["nyse_fundamentals"]["id"]
        logger.info(f"Downloading NYSE fundamental data: {dataset_id}")

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            output_path = self.output_dir / "nyse_fundamentals"
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading to: {output_path}")
            api.dataset_download_files(
                dataset_id,
                path=str(output_path),
                unzip=True
            )

            csv_files = list(output_path.glob("*.csv"))
            logger.info(f"✓ Downloaded {len(csv_files)} files")

            return output_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def download_sec_indicators(self) -> Path:
        """Download 200+ financial indicators from SEC filings.

        What: Comprehensive fundamental metrics extracted from SEC 10-K/10-Q filings
        Why: Most detailed fundamental data available
        How: Pre-processed SEC EDGAR filing data

        Returns:
            Path to downloaded data

        Metrics included (200+ indicators):
        - Valuation: P/E, P/B, P/S, EV/EBITDA, EV/Sales, PEG Ratio
        - Profitability: ROA, ROE, ROIC, Gross Margin, Operating Margin, Net Margin
        - Liquidity: Current Ratio, Quick Ratio, Cash Ratio
        - Leverage: Debt/Equity, Debt/Assets, Interest Coverage
        - Efficiency: Asset Turnover, Inventory Turnover, Receivables Turnover
        - Growth: Revenue Growth, Earnings Growth, EPS Growth
        - Cash Flow: Operating CF, Free CF, CF/Sales
        - Per Share: EPS, Book Value/Share, Sales/Share, CF/Share
        - And 150+ more!
        """
        dataset_id = self.FUNDAMENTAL_DATASETS["sec_financials"]["id"]
        logger.info(f"Downloading SEC financial indicators: {dataset_id}")
        logger.info("⚠ This is a large dataset (~200MB+)")

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            output_path = self.output_dir / "sec_indicators"
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading to: {output_path}")
            api.dataset_download_files(
                dataset_id,
                path=str(output_path),
                unzip=True
            )

            csv_files = list(output_path.glob("*.csv"))
            logger.info(f"✓ Downloaded {len(csv_files)} files")

            return output_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def merge_fundamentals_with_prices(
        self,
        fundamentals_csv: Path,
        prices_dir: Path,
        output_csv: Path
    ) -> pd.DataFrame:
        """Merge fundamental data with price data for training.

        What: Combines financial metrics with historical prices
        Why: ML models need both fundamentals and price history
        How: Joins on ticker symbol and date

        Args:
            fundamentals_csv: Path to fundamentals CSV
            prices_dir: Directory with price CSVs (one per ticker)
            output_csv: Where to save merged data

        Returns:
            Merged DataFrame

        Example merged row:
        {
            "ticker": "AAPL",
            "date": "2023-01-15",
            "close": 150.23,
            "volume": 1000000,
            "pe_ratio": 25.3,
            "eps": 6.02,
            "market_cap": 2500000000000,
            "ebitda": 120000000000,
            ...
        }
        """
        logger.info("Merging fundamental data with price data...")

        # Load fundamentals
        logger.info(f"Loading fundamentals from {fundamentals_csv}")
        fundamentals = pd.read_csv(fundamentals_csv)
        logger.info(f"  Loaded {len(fundamentals)} rows, {len(fundamentals.columns)} columns")

        # List available price files
        price_files = list(prices_dir.glob("*.csv"))
        logger.info(f"Found {len(price_files)} price files")

        merged_data = []

        for price_file in price_files:
            ticker = price_file.stem  # Filename without .csv
            logger.info(f"Processing {ticker}...")

            # Load price data
            prices = pd.read_csv(price_file)

            # Filter fundamentals for this ticker
            ticker_fundamentals = fundamentals[
                fundamentals['Symbol'] == ticker
            ] if 'Symbol' in fundamentals.columns else fundamentals[
                fundamentals['Ticker'] == ticker
            ] if 'Ticker' in fundamentals.columns else pd.DataFrame()

            if ticker_fundamentals.empty:
                logger.warning(f"  No fundamental data for {ticker}")
                continue

            # Simple merge (assumes fundamentals are point-in-time, not historical)
            # For more sophisticated merging, would need dates in both datasets
            for _, price_row in prices.iterrows():
                merged_row = price_row.to_dict()
                # Add fundamental metrics
                for col in ticker_fundamentals.columns:
                    if col not in merged_row:
                        merged_row[col] = ticker_fundamentals[col].iloc[0]
                merged_data.append(merged_row)

        # Convert to DataFrame
        merged_df = pd.DataFrame(merged_data)
        logger.info(f"Merged data: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        # Save
        merged_df.to_csv(output_csv, index=False)
        logger.info(f"✓ Saved to {output_csv}")

        return merged_df


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download fundamental financial data for stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download S&P 500 financial data
  python download_fundamentals.py --dataset sp500

  # Download NYSE fundamentals (quarterly/annual)
  python download_fundamentals.py --dataset nyse

  # Download comprehensive SEC indicators (200+ metrics)
  python download_fundamentals.py --dataset sec

  # Download all available fundamental datasets
  python download_fundamentals.py --all

  # Merge fundamentals with price data
  python download_fundamentals.py --dataset sp500 --merge-with-prices data/training/prepared
        """
    )

    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--dataset',
        choices=['sp500', 'nyse', 'sec'],
        help='Which fundamental dataset to download'
    )
    dataset_group.add_argument(
        '--all',
        action='store_true',
        help='Download all available fundamental datasets'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/fundamentals'),
        help='Directory to save fundamental data'
    )

    # Merge option
    parser.add_argument(
        '--merge-with-prices',
        type=Path,
        help='Path to price data directory to merge with fundamentals'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("Fundamental Data Downloader")
    logger.info("=" * 60)

    # Initialize downloader
    downloader = FundamentalsDownloader(output_dir=args.output_dir)

    # Download requested datasets
    try:
        if args.all:
            logger.info("Downloading all fundamental datasets...")
            downloader.download_sp500_financials()
            downloader.download_nyse_fundamentals()
            downloader.download_sec_indicators()

        elif args.dataset == 'sp500':
            output_path = downloader.download_sp500_financials()

            # Merge if requested
            if args.merge_with_prices:
                # Find the main CSV file
                csv_files = list(output_path.glob("*.csv"))
                if csv_files:
                    merged_output = args.output_dir / "merged_sp500_with_prices.csv"
                    downloader.merge_fundamentals_with_prices(
                        fundamentals_csv=csv_files[0],
                        prices_dir=args.merge_with_prices,
                        output_csv=merged_output
                    )

        elif args.dataset == 'nyse':
            downloader.download_nyse_fundamentals()

        elif args.dataset == 'sec':
            downloader.download_sec_indicators()

        logger.info("=" * 60)
        logger.info("✓ Download Complete!")
        logger.info("=" * 60)
        logger.info(f"Data location: {args.output_dir}")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Review downloaded fundamental data")
        logger.info("2. Merge with price data using --merge-with-prices")
        logger.info("3. Update build_training_dataset.py to include fundamental features")
        logger.info("4. Retrain model with enhanced features")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
