"""Download stock market data from Kaggle for model training.

This script downloads historical stock price data from Kaggle datasets
for training the StockSense forecasting model. Designed with clear
explanations for college-level understanding.

What this script does:
- Authenticates with Kaggle API using credentials
- Downloads S&P 500 historical price datasets
- Extracts and organizes data into training directories
- Validates downloaded data quality

Why we need this:
- Kaggle provides large, free historical stock datasets
- Perfect for training ML models (years of data)
- Pre-cleaned and ready to use
- No rate limits like APIs

How it works:
1. Authenticate with Kaggle API
2. Download specified dataset (S&P 500 stocks)
3. Extract CSV files to data/training/raw/
4. Validate data quality
5. Report download statistics

Usage:
    python scripts/download_kaggle_data.py

Requirements:
    - pip install kaggle
    - Kaggle API credentials in ~/.kaggle/kaggle.json
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleDataDownloader:
    """Downloads and manages stock market data from Kaggle.

    What this does: Handles complete Kaggle dataset download process
    Why: Centralizes Kaggle data fetching with validation
    How: Uses Kaggle API to download datasets, extracts to organized structure

    College-Level Analogy:
    Think of this as a download manager specifically for financial datasets.
    It knows where to find good data (Kaggle), how to get it (API),
    and where to put it (organized directories).
    """

    # Popular Kaggle datasets for stock data
    # What: Dataset identifiers on Kaggle
    # Why: These are well-maintained, comprehensive datasets
    # How: Use dataset ID to download via API
    DATASETS = {
        "sp500": "camnugent/sandp500",  # S&P 500 historical data
        "us_stocks": "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",  # Comprehensive US stocks
        "stock_fundamentals": "dgawlik/nyse"  # NYSE fundamentals
    }

    def __init__(self, output_dir: Path = Path("data/training/raw")):
        """Initialize Kaggle downloader with output directory.

        Args:
            output_dir: Where to save downloaded data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check Kaggle credentials
        # What: Verify API key is configured
        # Why: Need credentials to download from Kaggle
        # How: Check for kaggle.json file
        self._verify_kaggle_credentials()

    def _verify_kaggle_credentials(self) -> None:
        """Verify Kaggle API credentials are configured.

        What: Checks for kaggle.json with API credentials
        Why: Can't download without authentication
        How: Looks in standard Kaggle config locations

        Raises:
            RuntimeError: If credentials not found or invalid
        """
        # Check standard Kaggle credential locations
        # What: Look for kaggle.json in expected places
        # Why: Kaggle API looks here automatically
        # How: Check Windows and Unix paths
        kaggle_config_paths = [
            Path.home() / ".kaggle" / "kaggle.json",  # Unix/Mac
            Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json",  # Windows
        ]

        found = False
        for path in kaggle_config_paths:
            if path.exists():
                logger.info(f"✓ Found Kaggle credentials: {path}")
                found = True
                break

        if not found:
            raise RuntimeError(
                "Kaggle credentials not found!\n"
                "Create ~/.kaggle/kaggle.json with your API key:\n"
                '{"username":"your_username","key":"your_api_key"}\n'
                "Get API key from: https://www.kaggle.com/settings"
            )

    def download_sp500_data(self) -> Path:
        """Download S&P 500 historical stock data from Kaggle.

        What: Downloads comprehensive S&P 500 dataset
        Why: S&P 500 is the benchmark index, good training data
        How: Uses Kaggle API to download and extract

        Returns:
            Path to extracted data directory

        Dataset Details:
        - Dataset: camnugent/sandp500
        - Contains: Historical prices for all S&P 500 stocks
        - Format: Individual CSV files per stock
        - Data: Date, Open, High, Low, Close, Volume
        - History: Multiple years of daily data
        """
        dataset_id = self.DATASETS["sp500"]
        logger.info(f"Downloading S&P 500 data from Kaggle: {dataset_id}")
        logger.info("This may take a few minutes (~100MB download)...")

        try:
            # Import Kaggle API
            # What: Load Kaggle Python library
            # Why: Need API client to download datasets
            # How: Dynamic import with error handling
            from kaggle.api.kaggle_api_extended import KaggleApi

            # Initialize and authenticate API
            # What: Create Kaggle API client
            # Why: Need authenticated client for downloads
            # How: API reads credentials from kaggle.json
            api = KaggleApi()
            api.authenticate()
            logger.info("✓ Authenticated with Kaggle")

            # Download dataset
            # What: Fetch dataset files from Kaggle
            # Why: Get the raw CSV files
            # How: API downloads and extracts automatically
            download_path = self.output_dir / "kaggle_sp500"
            download_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading to: {download_path}")
            api.dataset_download_files(
                dataset_id,
                path=str(download_path),
                unzip=True  # Automatically extract ZIP files
            )

            logger.info("✓ Download complete")

            # List downloaded files
            # What: Show what we got
            # Why: Verify download succeeded
            # How: Count CSV files in directory
            csv_files = list(download_path.glob("**/*.csv"))
            logger.info(f"✓ Found {len(csv_files)} CSV files")

            # Show some example files
            if csv_files:
                logger.info("Sample files:")
                for csv_file in csv_files[:5]:
                    file_size = csv_file.stat().st_size / 1024  # KB
                    logger.info(f"  - {csv_file.name} ({file_size:.1f} KB)")
                if len(csv_files) > 5:
                    logger.info(f"  ... and {len(csv_files) - 5} more")

            return download_path

        except ImportError:
            raise RuntimeError(
                "Kaggle library not installed.\n"
                "Install with: pip install kaggle"
            )
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise RuntimeError(f"Kaggle download failed: {e}") from e

    def download_comprehensive_us_stocks(self) -> Path:
        """Download comprehensive US stock data (larger dataset).

        What: Downloads extensive historical data for US stocks and ETFs
        Why: More comprehensive than just S&P 500
        How: Uses Kaggle API for bulk download

        Returns:
            Path to extracted data directory

        Warning: This is a larger download (~500MB+)
        """
        dataset_id = self.DATASETS["us_stocks"]
        logger.info(f"Downloading comprehensive US stocks data: {dataset_id}")
        logger.info("⚠ Large download (~500MB+), may take several minutes...")

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            download_path = self.output_dir / "kaggle_us_stocks"
            download_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading to: {download_path}")
            api.dataset_download_files(
                dataset_id,
                path=str(download_path),
                unzip=True
            )

            logger.info("✓ Download complete")

            csv_files = list(download_path.glob("**/*.csv"))
            logger.info(f"✓ Found {len(csv_files)} CSV files")

            return download_path

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise RuntimeError(f"Kaggle download failed: {e}") from e

    def validate_data(self, data_dir: Path) -> Dict[str, any]:
        """Validate downloaded data quality.

        What: Checks CSV files for completeness and format
        Why: Ensure data is usable for training
        How: Checks file counts, sizes, required columns

        Args:
            data_dir: Directory containing downloaded CSV files

        Returns:
            Dictionary with validation statistics
        """
        logger.info("Validating downloaded data...")

        csv_files = list(data_dir.glob("**/*.csv"))

        if not csv_files:
            logger.error("No CSV files found!")
            return {"valid": False, "error": "No CSV files"}

        # Check a sample file for required columns
        # What: Verify CSV has expected structure
        # Why: Need specific columns for training
        # How: Read first file, check headers
        import pandas as pd

        sample_file = csv_files[0]
        logger.info(f"Checking sample file: {sample_file.name}")

        try:
            df = pd.read_csv(sample_file, nrows=5)  # Read just first 5 rows
            columns = df.columns.tolist()
            logger.info(f"Columns: {columns}")

            # Check for required columns (case-insensitive)
            # What: Ensure we have price and volume data
            # Why: These are minimum requirements for training
            # How: Look for close price and volume columns
            columns_lower = [col.lower() for col in columns]
            has_price = any('close' in col or 'price' in col for col in columns_lower)
            has_volume = 'volume' in columns_lower
            has_date = any('date' in col or 'time' in col for col in columns_lower)

            if not all([has_price, has_volume, has_date]):
                logger.warning("Sample file missing required columns")
                logger.warning(f"Has price: {has_price}, Has volume: {has_volume}, Has date: {has_date}")

        except Exception as e:
            logger.warning(f"Could not validate sample file: {e}")

        # Calculate statistics
        # What: Summarize downloaded data
        # Why: Show user what they got
        # How: Count files, calculate total size
        total_size = sum(f.stat().st_size for f in csv_files)
        total_size_mb = total_size / (1024 * 1024)

        stats = {
            "valid": True,
            "total_files": len(csv_files),
            "total_size_mb": round(total_size_mb, 2),
            "data_directory": str(data_dir)
        }

        logger.info("=" * 60)
        logger.info("Validation Summary:")
        logger.info(f"  Total CSV files: {stats['total_files']}")
        logger.info(f"  Total size: {stats['total_size_mb']} MB")
        logger.info(f"  Location: {stats['data_directory']}")
        logger.info("=" * 60)

        return stats


def main():
    """Main entry point for Kaggle data download script."""
    logger.info("=" * 60)
    logger.info("Kaggle Stock Data Downloader")
    logger.info("=" * 60)

    try:
        # Initialize downloader
        downloader = KaggleDataDownloader()

        # Download S&P 500 data (recommended for training)
        logger.info("\nDownloading S&P 500 dataset...")
        data_dir = downloader.download_sp500_data()

        # Validate downloaded data
        stats = downloader.validate_data(data_dir)

        if stats["valid"]:
            logger.info("\n" + "=" * 60)
            logger.info("✓ Download Complete!")
            logger.info("=" * 60)
            logger.info(f"Data location: {data_dir}")
            logger.info(f"Total files: {stats['total_files']}")
            logger.info(f"Total size: {stats['total_size_mb']} MB")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Review the downloaded CSV files")
            logger.info("2. Run: python scripts/build_training_dataset.py --ticker-file <ticker_list>")
            logger.info("3. Or use specific tickers: python scripts/build_training_dataset.py --tickers AAPL MSFT GOOGL")
            logger.info("=" * 60)

            return 0
        else:
            logger.error("Validation failed!")
            return 1

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
