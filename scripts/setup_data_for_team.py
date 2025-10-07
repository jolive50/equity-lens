#!/usr/bin/env python
"""Quick setup script for team members to download all necessary data.

This script automates the data download process for new team members.
Just run it once and you'll have everything you need!

Usage:
    python scripts/setup_data_for_team.py

College-Level Explanation:
This is a convenience script that runs all the data download steps
in the right order. Think of it as an installer that gets your
development environment ready automatically.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured.

    Returns:
        True if credentials exist, False otherwise
    """
    kaggle_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json" if os.name == 'nt' else None
    ]

    for path in kaggle_paths:
        if path and path.exists():
            logger.info(f"✓ Found Kaggle credentials at {path}")
            return True

    return False


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status.

    Args:
        cmd: Command to run as list
        description: What this command does

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} - FAILED")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main setup process."""
    import os

    logger.info("="*60)
    logger.info("StockSense Data Setup for Team Members")
    logger.info("="*60)
    logger.info("\nThis script will download all necessary data for training.")
    logger.info("Total download size: ~100-150 MB")
    logger.info("Estimated time: 5-10 minutes\n")

    # Check prerequisites
    logger.info("Step 1: Checking prerequisites...")

    # Check Kaggle credentials
    if not check_kaggle_credentials():
        logger.error("✗ Kaggle credentials not found!")
        logger.error("\nPlease set up Kaggle credentials first:")
        logger.error("1. Go to https://www.kaggle.com/settings")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. Download kaggle.json")
        logger.error("4. Move to ~/.kaggle/kaggle.json")
        logger.error("   On Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
        logger.error("   On Mac/Linux: ~/.kaggle/kaggle.json")
        logger.error("5. Run this script again")
        return 1

    # Check if running from correct directory
    if not Path("scripts/download_kaggle_data.py").exists():
        logger.error("✗ Please run this script from the project root directory:")
        logger.error("  cd path/to/capstone")
        logger.error("  python scripts/setup_data_for_team.py")
        return 1

    logger.info("✓ Prerequisites OK\n")

    # Ask user what to download
    logger.info("What would you like to download?")
    logger.info("1. Minimum (Price data for 8 stocks) - ~20 MB, fastest")
    logger.info("2. Recommended (Price + fundamentals for 50 stocks) - ~80 MB")
    logger.info("3. Complete (Everything) - ~150 MB")

    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or "2"

    steps = []

    if choice == "1":
        # Minimum setup
        steps = [
            (["python", "scripts/download_kaggle_data.py"],
             "Downloading S&P 500 price data"),
            (["python", "scripts/prepare_kaggle_data.py", "--tickers", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AMD", "INTC", "NFLX"],
             "Preparing 8 tech stocks"),
        ]
    elif choice == "2":
        # Recommended setup
        steps = [
            (["python", "scripts/download_kaggle_data.py"],
             "Downloading S&P 500 price data"),
            (["python", "scripts/prepare_kaggle_data.py", "--top", "50"],
             "Preparing top 50 stocks"),
            (["python", "scripts/download_fundamentals.py", "--dataset", "sp500"],
             "Downloading fundamental data"),
        ]
    elif choice == "3":
        # Complete setup
        steps = [
            (["python", "scripts/download_kaggle_data.py"],
             "Downloading S&P 500 price data"),
            (["python", "scripts/prepare_kaggle_data.py", "--all"],
             "Preparing all available stocks"),
            (["python", "scripts/download_fundamentals.py", "--dataset", "sp500"],
             "Downloading S&P 500 fundamentals"),
            (["python", "scripts/download_fundamentals.py", "--dataset", "sec"],
             "Downloading SEC indicators (225 metrics)"),
        ]
    else:
        logger.error("Invalid choice. Exiting.")
        return 1

    # Execute steps
    success_count = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
        else:
            logger.error("\nSetup failed. Please check errors above.")
            logger.error("You can run individual scripts manually:")
            logger.error("  python scripts/download_kaggle_data.py")
            logger.error("  python scripts/prepare_kaggle_data.py --tickers AAPL MSFT")
            logger.error("  python scripts/download_fundamentals.py --dataset sp500")
            return 1

    # Success!
    logger.info("\n" + "="*60)
    logger.info("✓ SETUP COMPLETE!")
    logger.info("="*60)
    logger.info(f"Successfully completed {success_count}/{len(steps)} steps")
    logger.info("\nData downloaded to:")
    logger.info("  - data/training/raw/          (Raw Kaggle data)")
    logger.info("  - data/training/prepared/     (Individual stock CSVs)")

    if choice in ["2", "3"]:
        logger.info("  - data/fundamentals/          (Financial metrics)")

    logger.info("\nNext steps:")
    logger.info("1. Review downloaded data:")
    logger.info("   ls data/training/prepared/")
    logger.info("\n2. Build training dataset:")
    logger.info("   python scripts/build_training_dataset.py --ticker-file data/training/ticker_list.txt")
    logger.info("\n3. Train the model:")
    logger.info("   python scripts/train_forecaster.py")
    logger.info("\n4. Download FinBERT (sentiment analysis):")
    logger.info("   python scripts/download_finbert.py")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
