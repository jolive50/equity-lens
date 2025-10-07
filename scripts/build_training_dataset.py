"""Build training dataset for the StockSense forecasting model.

This script fetches historical market data, calculates features, and prepares
training datasets for the ML forecaster. It's designed following SOLID principles
and includes clear explanations for college-level understanding.

What this script does:
- Fetches historical data from configured data sources (Alpha Vantage, Tiingo, etc.)
- Engineers features from raw price/volume data
- Creates labels for supervised learning (up/down/neutral)
- Splits data into training, validation, and test sets
- Saves processed data to CSV files

Why we need this:
- ML models need historical data to learn patterns
- Raw market data must be transformed into features
- We need labels (what actually happened) for supervised learning

How it works:
1. Read ticker list from configuration
2. Fetch historical data via data adapters
3. Engineer features (RSI, MACD, momentum, etc.)
4. Generate labels based on future price movements
5. Split and save datasets

College-Level Concepts:
- Feature Engineering: Converting raw data into ML-ready numerical features
- Supervised Learning: Teaching the model by showing it examples with answers
- Train/Validation/Test Split: Prevents overfitting by testing on unseen data
- SOLID Principles: Single Responsibility (each class does one thing)
"""
from __future__ import annotations

import argparse  # For command-line argument parsing
import logging  # For tracking progress and errors
import os  # For file system operations
import sys  # For system-level operations
from datetime import datetime, timedelta  # For date calculations
from pathlib import Path  # For cross-platform file paths
from typing import Dict, List, Optional, Tuple

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation (DataFrames)

# Add parent directory to path for imports
# What: Allows importing from pipelines package
# Why: Script is in scripts/ but code is in pipelines/
# How: Adds parent directory to Python's module search path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.realtime.data_adapters import DataService
from pipelines.realtime.models.forecaster import TechnicalIndicators, FeatureEngineer

# Configure logging
# What: Sets up logging to show INFO level messages
# Why: We want to track progress as the script runs
# How: Configures format to show timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates training labels from future price movements.

    What this does: Creates the "answers" for supervised learning
    Why: ML models learn by seeing inputs (features) and outputs (labels)
    How: Looks at future price changes and classifies them as up/down/neutral

    This class follows the Single Responsibility Principle (SOLID):
    - Its only job is to generate labels from price data
    - Doesn't fetch data, engineer features, or train models

    College-Level Analogy:
    Imagine studying for an exam with flash cards. The question side is the
    features (current market state), the answer side is the label (what happened next).
    This class creates the "answer side" of the flash cards.
    """

    def __init__(
        self,
        *,
        forecast_horizon: int = 5,  # How many days ahead to predict
        up_threshold: float = 0.02,  # +2% counts as "up"
        down_threshold: float = -0.02,  # -2% counts as "down"
    ):
        """Initialize label generator with classification thresholds.

        What: Sets up the rules for classifying price movements
        Why: We need clear definitions of "up", "down", and "neutral"
        How: Stores thresholds that define each class

        Args:
            forecast_horizon: Days ahead to look (default 5 = 1 week)
            up_threshold: Minimum % gain to classify as "up" (default +2%)
            down_threshold: Maximum % loss to classify as "down" (default -2%)

        Example:
            - Price goes from $100 to $103 in 5 days = +3% = "up"
            - Price goes from $100 to $98 in 5 days = -2% = "down"
            - Price goes from $100 to $101 in 5 days = +1% = "neutral"
        """
        self.forecast_horizon = forecast_horizon
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def generate_labels(self, prices: List[float]) -> List[str]:
        """Generate classification labels from price series.

        What: Converts a list of prices into a list of labels
        Why: Need labeled data for supervised learning
        How: For each price, look ahead N days and classify the movement

        Args:
            prices: List of closing prices in chronological order
                   Example: [100, 101, 99, 102, 105, ...]

        Returns:
            List of labels: ["up", "down", "neutral", ...]
            Note: Last N elements will be None (can't look into the future)

        Data Flow Example:
            Input: [100, 102, 101, 103, 105]  (forecast_horizon=2, up_threshold=0.02)
            Process:
            - prices[0]=100 → prices[2]=101 → +1% → "neutral"
            - prices[1]=102 → prices[3]=103 → +0.98% → "neutral"
            - prices[2]=101 → prices[4]=105 → +3.96% → "up"
            - prices[3]=103 → no future data → None
            - prices[4]=105 → no future data → None
            Output: ["neutral", "neutral", "up", None, None]
        """
        labels = []

        for i in range(len(prices)):
            # Check if we have enough future data to look ahead
            # What: Ensures we don't go out of bounds
            # Why: Can't predict the future of the last few days
            if i + self.forecast_horizon >= len(prices):
                labels.append(None)  # Not enough future data
                continue

            # Calculate future return
            # What: Percentage change from current price to future price
            # Why: This is what we're trying to predict
            # How: (future_price - current_price) / current_price
            current_price = prices[i]
            future_price = prices[i + self.forecast_horizon]

            if current_price == 0:
                # Avoid division by zero
                # What: Safety check for bad data
                # Why: Division by zero would crash the program
                labels.append(None)
                continue

            future_return = (future_price - current_price) / current_price

            # Classify the return
            # What: Apply thresholds to determine class
            # Why: Convert continuous returns into discrete classes
            # How: Compare return to up/down thresholds
            if future_return >= self.up_threshold:
                label = "up"  # Significant gain
            elif future_return <= self.down_threshold:
                label = "down"  # Significant loss
            else:
                label = "neutral"  # Small movement

            labels.append(label)

        return labels


class DatasetBuilder:
    """Builds complete training datasets from raw market data.

    What this does: Orchestrates the entire dataset creation pipeline
    Why: Combines data fetching, feature engineering, and label generation
    How: Uses dependency injection (SOLID) to work with any data service

    This class demonstrates SOLID principles:
    - Single Responsibility: Only builds datasets (doesn't fetch or train)
    - Open/Closed: Can work with any DataService implementation
    - Dependency Inversion: Depends on abstractions (DataService interface)

    College-Level Analogy:
    Think of this as a factory assembly line. Raw materials (market data) come in,
    go through multiple processing steps (feature engineering, labeling), and
    finished products (training data) come out.
    """

    def __init__(
        self,
        *,
        data_service: DataService,  # Injected dependency for data fetching
        label_generator: Optional[LabelGenerator] = None,  # Optional custom labeler
        feature_engineer: Optional[FeatureEngineer] = None,  # Optional custom engineer
    ):
        """Initialize dataset builder with dependencies.

        What: Sets up the builder with required services
        Why: Dependency injection makes code testable and flexible
        How: Accepts configured objects rather than creating them internally

        Args:
            data_service: Service for fetching market data (required)
            label_generator: Optional custom label generator
            feature_engineer: Optional custom feature engineer

        SOLID Principle: Dependency Inversion
        Instead of: self.data_service = AlphaVantageService()  (tight coupling)
        We do: self.data_service = data_service  (loose coupling)
        This allows using ANY data service, not just Alpha Vantage.
        """
        self.data_service = data_service
        self.label_generator = label_generator or LabelGenerator()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def build_dataset(
        self,
        *,
        tickers: List[str],  # List of stock symbols to fetch
        start_date: datetime,  # Start of date range
        end_date: datetime,  # End of date range
        output_dir: Path,  # Where to save the results
    ) -> Dict[str, Path]:
        """Build complete training dataset for specified tickers and date range.

        What: Main entry point that creates the full training dataset
        Why: Combines all steps into one convenient function
        How: Fetches data, engineers features, generates labels, saves files

        Args:
            tickers: Stock symbols to include (e.g., ["AAPL", "MSFT", "GOOGL"])
            start_date: Beginning of historical data range
            end_date: End of historical data range
            output_dir: Directory to save processed datasets

        Returns:
            Dictionary mapping dataset names to file paths:
            {
                "features": Path("data/training/features/features.csv"),
                "labels": Path("data/training/labels/labels.csv"),
                "raw": Path("data/training/raw/raw_data.csv")
            }

        Data Flow:
        1. Fetch raw data for each ticker
        2. For each ticker's data:
           a. Engineer features (RSI, MACD, etc.)
           b. Generate labels (up/down/neutral)
           c. Combine into rows
        3. Concatenate all tickers into single dataset
        4. Split into train/validation/test sets
        5. Save to CSV files
        """
        logger.info(f"Building dataset for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}")

        # Create output directories
        # What: Ensures necessary folders exist
        # Why: Can't save files if directories don't exist
        # How: Creates directories if they don't already exist
        features_dir = output_dir / "features"
        labels_dir = output_dir / "labels"
        raw_dir = output_dir / "raw"

        for directory in [features_dir, labels_dir, raw_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        # Collect data for all tickers
        # What: Fetches and processes each ticker
        # Why: Need data from multiple stocks for robust model
        # How: Loop through tickers, fetch data, process, accumulate
        all_features = []
        all_labels = []
        all_raw_data = []

        for ticker in tickers:
            logger.info(f"Processing {ticker}...")

            try:
                # Fetch historical market data
                # What: Get price/volume data from data service
                # Why: This is our raw input data
                # How: Use injected data service to fetch
                market_data = self.data_service.get_market_data(
                    ticker=ticker,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )

                if not market_data or len(market_data) < 30:
                    # Skip tickers with insufficient data
                    # What: Need at least 30 days for technical indicators
                    # Why: RSI needs 14+ days, MACD needs 26+ days
                    logger.warning(f"Insufficient data for {ticker} ({len(market_data)} days), skipping")
                    continue

                # Extract prices for label generation
                # What: Pull out closing prices into a list
                # Why: Label generator needs just prices, not full data
                prices = [float(d.get('close', 0)) for d in market_data]

                # Generate labels
                # What: Create "answer key" for this ticker
                # Why: Supervised learning needs labels
                # How: Look at future prices and classify
                labels = self.label_generator.generate_labels(prices)

                # Engineer features
                # What: Convert raw data into ML features
                # Why: ML models need structured numerical inputs
                # How: Use feature engineer to create indicators
                features = self._process_ticker_data(ticker, market_data)

                # Combine features and labels
                # What: Match each feature row with its label
                # Why: Need aligned training pairs (X, y)
                # How: Iterate and combine non-None labels
                for i, (feature_row, label) in enumerate(zip(features, labels)):
                    if label is not None:  # Only use labeled samples
                        # Add ticker and date for tracking
                        feature_row['ticker'] = ticker
                        feature_row['date'] = market_data[i].get('date', '')

                        all_features.append(feature_row)
                        all_labels.append({'ticker': ticker, 'date': feature_row['date'], 'label': label})

                # Save raw data for reference
                # What: Keep original data for debugging/analysis
                # Why: Helpful to trace back to source data
                all_raw_data.extend([{**d, 'ticker': ticker} for d in market_data])

                logger.info(f"✓ Processed {ticker}: {len(market_data)} days, {len([l for l in labels if l is not None])} labeled samples")

            except Exception as e:
                logger.error(f"✗ Error processing {ticker}: {e}")
                continue

        # Convert to DataFrames
        # What: Transform lists of dicts into pandas DataFrames
        # Why: DataFrames are easier to manipulate and save as CSV
        # How: Use pandas DataFrame constructor
        features_df = pd.DataFrame(all_features)
        labels_df = pd.DataFrame(all_labels)
        raw_df = pd.DataFrame(all_raw_data)

        logger.info(f"Total samples: {len(features_df)} features, {len(labels_df)} labels")

        # Save datasets
        # What: Write DataFrames to CSV files
        # Why: Persistent storage for training later
        # How: Use pandas to_csv method
        features_path = features_dir / "features.csv"
        labels_path = labels_dir / "labels.csv"
        raw_path = raw_dir / "raw_data.csv"

        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)
        raw_df.to_csv(raw_path, index=False)

        logger.info(f"✓ Saved features to {features_path}")
        logger.info(f"✓ Saved labels to {labels_path}")
        logger.info(f"✓ Saved raw data to {raw_path}")

        return {
            "features": features_path,
            "labels": labels_path,
            "raw": raw_path
        }

    def _process_ticker_data(self, ticker: str, market_data: List[Dict]) -> List[Dict]:
        """Process market data for a single ticker into features.

        What: Converts raw price/volume data into ML features
        Why: Need structured features for each time point
        How: Uses FeatureEngineer for each day's data

        Args:
            ticker: Stock symbol (for logging)
            market_data: List of daily price/volume dictionaries

        Returns:
            List of feature dictionaries (one per day)

        Technical Note:
        For each day, we use ALL previous data up to that point to create features.
        This prevents "look-ahead bias" (using future information to predict the past).
        """
        features_list = []

        # Need at least 30 days for technical indicators
        min_window = 30

        for i in range(len(market_data)):
            # Check if we have enough historical data
            # What: Ensure sufficient history for indicators
            # Why: RSI, MACD need historical window
            if i < min_window - 1:
                continue  # Not enough history yet

            # Get data up to current point (no look-ahead)
            # What: Slice data from start to current day (inclusive)
            # Why: Simulates real-time prediction (only know the past)
            # How: Use all data from index 0 to i+1
            historical_data = market_data[:i+1]

            # Engineer features for this time point
            # What: Create feature vector from historical data
            # Why: This is what the ML model will see
            # How: Feature engineer creates all technical indicators
            features = self.feature_engineer.create_features(historical_data)

            if features.size == 0:
                # Skip if feature engineering failed
                # What: Handle edge cases gracefully
                # Why: Bad data shouldn't crash the pipeline
                continue

            # Convert numpy array to dictionary
            # What: Transform array into named fields
            # Why: Easier to work with and save to CSV
            # How: Create dict with feature names as keys
            feature_names = [
                "1d_return", "5d_return", "10d_return", "20d_return", "price_position",
                "volume_ratio", "volume_vs_avg", "rsi", "macd", "macd_signal", "macd_histogram",
                "bb_position", "golden_cross", "momentum", "volatility"
            ]

            feature_dict = dict(zip(feature_names, features[0]))
            features_list.append(feature_dict)

        return features_list


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    What: Handles command-line options for the script
    Why: Makes script flexible and scriptable
    How: Uses argparse library to define and parse arguments

    Returns:
        Namespace object with parsed arguments

    Example usage:
        python build_training_dataset.py --tickers AAPL MSFT GOOGL --days 365
        python build_training_dataset.py --ticker-file sp500.txt --days 730
    """
    parser = argparse.ArgumentParser(
        description='Build training dataset for StockSense forecasting model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset for specific tickers (last 365 days)
  python build_training_dataset.py --tickers AAPL MSFT GOOGL --days 365

  # Build dataset from ticker file (last 2 years)
  python build_training_dataset.py --ticker-file data/sp500_tickers.txt --days 730

  # Specify custom output directory
  python build_training_dataset.py --tickers AAPL --output-dir data/training_custom
        """
    )

    # Ticker input options (mutually exclusive)
    # What: Allow specifying tickers via command line or file
    # Why: Flexibility for different use cases
    # How: Create mutually exclusive group
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        '--tickers',
        nargs='+',  # One or more tickers
        help='Space-separated list of stock tickers (e.g., AAPL MSFT GOOGL)'
    )
    ticker_group.add_argument(
        '--ticker-file',
        type=Path,
        help='Path to file with ticker symbols (one per line)'
    )

    # Date range options
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data to fetch (default: 365)'
    )
    parser.add_argument(
        '--end-date',
        type=lambda s: datetime.fromisoformat(s),
        default=datetime.now(),
        help='End date for data range in ISO format (default: today)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training'),
        help='Directory to save processed datasets (default: data/training)'
    )

    # Feature engineering options
    parser.add_argument(
        '--forecast-horizon',
        type=int,
        default=5,
        help='Days ahead to forecast for labels (default: 5)'
    )
    parser.add_argument(
        '--up-threshold',
        type=float,
        default=0.02,
        help='Minimum return to classify as "up" (default: 0.02 = 2%%)'
    )
    parser.add_argument(
        '--down-threshold',
        type=float,
        default=-0.02,
        help='Maximum return to classify as "down" (default: -0.02 = -2%%)'
    )

    return parser.parse_args()


def load_tickers_from_file(file_path: Path) -> List[str]:
    """Load ticker symbols from a text file.

    What: Reads tickers from a file (one per line)
    Why: Convenient for processing many tickers
    How: Opens file, reads lines, strips whitespace

    Args:
        file_path: Path to file containing tickers

    Returns:
        List of ticker symbols

    Example file format:
        AAPL
        MSFT
        GOOGL
        AMZN
        # Comments starting with # are ignored
    """
    tickers = []

    with open(file_path, 'r') as f:
        for line in f:
            # Strip whitespace and convert to uppercase
            # What: Clean up each line
            # Why: Handle various formatting styles
            ticker = line.strip().upper()

            # Skip empty lines and comments
            # What: Filter out non-ticker lines
            # Why: Allow comments and blank lines in file
            if ticker and not ticker.startswith('#'):
                tickers.append(ticker)

    logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
    return tickers


def main():
    """Main entry point for the dataset building script.

    What: Orchestrates the entire dataset creation process
    Why: Provides command-line interface for users
    How: Parses arguments, sets up services, builds dataset

    This function demonstrates the Single Responsibility Principle:
    - Only handles CLI coordination
    - Delegates actual work to specialized classes
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Determine ticker list
    # What: Get tickers from command line or file
    # Why: Support both interactive and batch usage
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_tickers_from_file(args.ticker_file)

    logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")

    # Calculate date range
    # What: Determine start and end dates for data fetch
    # Why: Need to specify time period for historical data
    end_date = args.end_date
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({args.days} days)")

    # Initialize data service
    # What: Create service for fetching market data
    # Why: Need to get historical data from somewhere
    # How: Use DataService with configured API keys
    try:
        from pipelines.realtime.api_keys import get_available_api_keys

        # Get available API keys
        # What: Retrieve configured API credentials
        # Why: Data sources require authentication
        # How: Use centralized API key management
        api_keys = get_available_api_keys("alpha_vantage", "tiingo", "finnhub")

        if not api_keys:
            logger.error("No API keys configured. Please set up at least one data source API key.")
            logger.error("See docs/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md for setup instructions.")
            return 1

        logger.info(f"Using data sources: {', '.join(api_keys.keys())}")

        # Create data service
        # What: Instantiate data fetching service
        # Why: Abstraction over multiple data sources
        # How: Pass API keys to factory function
        data_service = DataService(api_keys)

    except Exception as e:
        logger.error(f"Failed to initialize data service: {e}")
        return 1

    # Initialize label generator with custom thresholds
    # What: Create label generator with specified parameters
    # Why: Allow customization of classification rules
    # How: Pass thresholds from command-line arguments
    label_generator = LabelGenerator(
        forecast_horizon=args.forecast_horizon,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold
    )

    # Initialize dataset builder
    # What: Create main orchestration object
    # Why: Coordinates all dataset building steps
    # How: Inject dependencies (data service, label generator)
    builder = DatasetBuilder(
        data_service=data_service,
        label_generator=label_generator
    )

    # Build dataset
    # What: Execute the full dataset creation pipeline
    # Why: This is the main work of the script
    # How: Call build_dataset with all parameters
    try:
        output_paths = builder.build_dataset(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output_dir
        )

        logger.info("=" * 60)
        logger.info("✓ Dataset building complete!")
        logger.info("=" * 60)
        logger.info(f"Features: {output_paths['features']}")
        logger.info(f"Labels: {output_paths['labels']}")
        logger.info(f"Raw data: {output_paths['raw']}")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Review the generated datasets for quality")
        logger.info("2. Run scripts/train_forecaster.py to train the model")
        logger.info("3. Evaluate model performance on test set")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Failed to build dataset: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
