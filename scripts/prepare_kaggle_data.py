"""Prepare Kaggle data for training by extracting specific tickers.

This script processes the consolidated Kaggle CSV (all_stocks_5yr.csv) and
extracts data for specific tickers, preparing it for the build_training_dataset.py
script. Includes clear explanations for college-level understanding.

What this script does:
- Reads the consolidated all_stocks_5yr.csv file
- Filters data for specified tickers
- Saves individual ticker CSV files
- Creates a ticker list file for batch processing

Why we need this:
- Kaggle data is in one big file (all stocks combined)
- Training script expects data organized by ticker
- Need to select which stocks to train on

How it works:
1. Load consolidated CSV using pandas
2. Filter for desired tickers (or top N by data quality)
3. Save individual ticker CSVs
4. Generate ticker list for build_training_dataset.py

Usage:
    # Extract specific tickers
    python scripts/prepare_kaggle_data.py --tickers AAPL MSFT GOOGL

    # Extract top 50 tickers by data completeness
    python scripts/prepare_kaggle_data.py --top 50

    # Extract all available tickers
    python scripts/prepare_kaggle_data.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_kaggle_data(kaggle_csv_path: Path) -> pd.DataFrame:
    """Load consolidated Kaggle CSV file.

    Args:
        kaggle_csv_path: Path to all_stocks_5yr.csv

    Returns:
        DataFrame with all stock data
    """
    logger.info(f"Loading Kaggle data from {kaggle_csv_path}...")
    df = pd.read_csv(kaggle_csv_path)
    logger.info(f"✓ Loaded {len(df)} rows, {df['Name'].nunique()} unique tickers")
    return df


def get_available_tickers(df: pd.DataFrame, min_rows: int = 500) -> List[str]:
    """Get list of available tickers with sufficient data.

    Args:
        df: DataFrame with stock data
        min_rows: Minimum number of data points required

    Returns:
        List of ticker symbols with enough data
    """
    # Count rows per ticker
    ticker_counts = df['Name'].value_counts()

    # Filter tickers with enough data
    good_tickers = ticker_counts[ticker_counts >= min_rows].index.tolist()

    logger.info(f"Found {len(good_tickers)} tickers with {min_rows}+ data points")
    return good_tickers


def extract_tickers(
    df: pd.DataFrame,
    tickers: List[str],
    output_dir: Path
) -> int:
    """Extract data for specific tickers and save individual CSV files.

    Args:
        df: DataFrame with all stock data
        tickers: List of tickers to extract
        output_dir: Where to save individual ticker CSVs

    Returns:
        Number of tickers successfully extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    for ticker in tickers:
        # Filter data for this ticker
        ticker_data = df[df['Name'] == ticker].copy()

        if ticker_data.empty:
            logger.warning(f"No data found for {ticker}")
            continue

        # Sort by date
        ticker_data = ticker_data.sort_values('date')

        # Save to CSV
        output_file = output_dir / f"{ticker}.csv"
        ticker_data.to_csv(output_file, index=False)

        logger.info(f"✓ {ticker}: {len(ticker_data)} rows → {output_file.name}")
        extracted_count += 1

    return extracted_count


def create_ticker_list_file(tickers: List[str], output_path: Path) -> None:
    """Create a text file with list of tickers (one per line).

    Args:
        tickers: List of ticker symbols
        output_path: Where to save the ticker list
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ticker in sorted(tickers):
            f.write(f"{ticker}\n")

    logger.info(f"✓ Saved ticker list to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare Kaggle stock data for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract specific tickers
  python prepare_kaggle_data.py --tickers AAPL MSFT GOOGL AMZN TSLA

  # Extract top 50 tickers by data completeness
  python prepare_kaggle_data.py --top 50

  # Extract all available tickers (may be slow)
  python prepare_kaggle_data.py --all

  # Specify custom input/output paths
  python prepare_kaggle_data.py --tickers AAPL --input data/custom.csv --output data/prepared
        """
    )

    # Input options
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/training/raw/kaggle_sp500/all_stocks_5yr.csv'),
        help='Path to consolidated Kaggle CSV'
    )

    # Ticker selection (mutually exclusive)
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to extract'
    )
    ticker_group.add_argument(
        '--top',
        type=int,
        help='Extract top N tickers by data completeness'
    )
    ticker_group.add_argument(
        '--all',
        action='store_true',
        help='Extract all available tickers'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training/prepared'),
        help='Directory to save individual ticker CSVs'
    )
    parser.add_argument(
        '--ticker-list',
        type=Path,
        default=Path('data/training/ticker_list.txt'),
        help='Path to save ticker list file'
    )
    parser.add_argument(
        '--min-rows',
        type=int,
        default=500,
        help='Minimum rows required per ticker (default: 500)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("Kaggle Data Preparation")
    logger.info("=" * 60)

    # Check if input file exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.error("Run download_kaggle_data.py first!")
        return 1

    # Load data
    try:
        df = load_kaggle_data(args.input)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Determine which tickers to extract
    if args.tickers:
        # Specific tickers requested
        tickers = [t.upper() for t in args.tickers]
        logger.info(f"Extracting {len(tickers)} specified tickers")

    elif args.top:
        # Top N by data completeness
        available = get_available_tickers(df, min_rows=args.min_rows)
        tickers = available[:args.top]
        logger.info(f"Extracting top {args.top} tickers by data completeness")

    else:  # args.all
        # All available tickers
        tickers = get_available_tickers(df, min_rows=args.min_rows)
        logger.info(f"Extracting all {len(tickers)} available tickers")

    # Extract ticker data
    logger.info("=" * 60)
    logger.info("Extracting ticker data...")
    logger.info("=" * 60)

    try:
        extracted_count = extract_tickers(df, tickers, args.output_dir)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1

    if extracted_count == 0:
        logger.error("No tickers were extracted!")
        return 1

    # Create ticker list file
    try:
        successfully_extracted = [
            t for t in tickers
            if (args.output_dir / f"{t}.csv").exists()
        ]
        create_ticker_list_file(successfully_extracted, args.ticker_list)
    except Exception as e:
        logger.error(f"Failed to create ticker list: {e}")
        return 1

    # Success summary
    logger.info("=" * 60)
    logger.info("✓ Data Preparation Complete!")
    logger.info("=" * 60)
    logger.info(f"Extracted tickers: {extracted_count}/{len(tickers)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Ticker list: {args.ticker_list}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info(f"1. Review extracted data in {args.output_dir}")
    logger.info(f"2. Run: python scripts/build_training_dataset.py --ticker-file {args.ticker_list}")
    logger.info("3. Then: python scripts/train_forecaster.py")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
