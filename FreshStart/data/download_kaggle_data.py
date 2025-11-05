"""Download SP500 stock data from Kaggle for model training.

PAM's Component - Kaggle Data Download
Downloads the 'andrewmvd/sp-500-stocks' dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
Contains: Historical OHLCV data for all S&P 500 stocks

Usage:
    python -m data.download_kaggle_data
"""
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_sp500_data(output_dir: str = "data/raw") -> bool:
    """
    Download SP500 stock data from Kaggle.

    Args:
        output_dir: Directory to save downloaded data

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info("Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        dataset_name = "andrewmvd/sp-500-stocks"
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info("This may take a few minutes...")

        # Download and extract dataset
        api.dataset_download_files(
            dataset_name,
            path=str(output_path),
            unzip=True
        )

        # Check what was downloaded
        csv_files = list(output_path.glob("*.csv"))
        logger.info(f"\n✓ Download complete!")
        logger.info(f"✓ Found {len(csv_files)} CSV files")

        # List key files
        expected_files = ["sp500_companies.csv", "sp500_index.csv", "sp500_stocks.csv"]
        for filename in expected_files:
            file_path = output_path / filename
            if file_path.exists():
                logger.info(f"  ✓ {filename}")
            else:
                logger.warning(f"  ✗ {filename} not found")

        # Check for individual stock files
        stock_files = [f for f in csv_files if f.name not in expected_files]
        if stock_files:
            logger.info(f"  ✓ {len(stock_files)} individual stock CSV files")

        logger.info(f"\nData saved to: {output_path.absolute()}")
        return True

    except ImportError:
        logger.error("Kaggle library not installed!")
        logger.error("Install with: pip install kaggle")
        return False

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Ensure ~/.kaggle/kaggle.json exists with valid credentials")
        logger.error("2. Get API key from: https://www.kaggle.com/settings")
        logger.error("3. Accept dataset terms: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks")
        return False


def verify_data(data_dir: str = "data/raw") -> bool:
    """
    Verify downloaded data is complete.

    Args:
        data_dir: Directory containing downloaded data

    Returns:
        True if data is valid
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return False

    # Check for required files
    required_files = ["sp500_companies.csv", "sp500_index.csv", "sp500_stocks.csv"]
    missing_files = []

    for filename in required_files:
        if not (data_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    # Count stock CSV files
    csv_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in csv_files if f.name not in required_files]

    logger.info(f"✓ Data verification passed")
    logger.info(f"  - {len(required_files)} metadata files")
    logger.info(f"  - {len(stock_files)} individual stock files")

    return True


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("FreshStart - Kaggle SP500 Data Download")
    logger.info("Dataset: andrewmvd/sp-500-stocks")
    logger.info("=" * 60)

    # Download data
    success = download_sp500_data()

    if not success:
        logger.error("\n✗ Download failed")
        return 1

    # Verify data
    if not verify_data():
        logger.error("\n✗ Data verification failed")
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("✓ Setup complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run: python -m data.preprocess_data")
    logger.info("2. Run: python -m models.prediction.train_models")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
