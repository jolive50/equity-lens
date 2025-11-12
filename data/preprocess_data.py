"""Preprocess Kaggle SP500 data for model training.

PAM's Component - Data Preprocessing Pipeline
Prepares downloaded Kaggle data for LSTM, GRU, and Gradient Boost training.

Usage:
    python -m data.preprocess_data
"""
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses SP500 stock data for ML training."""

    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """Initialize preprocessor.

        Args:
            raw_data_dir: Directory with downloaded Kaggle data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.aggregate_file = self.raw_data_dir / "sp500_stocks.csv"
        self._aggregate_df: Optional[pd.DataFrame] = None

    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """Load individual stock CSV file.

        Args:
            ticker: Stock symbol

        Returns:
            DataFrame with OHLCV data
        """
        csv_file = self.raw_data_dir / f"{ticker}.csv"

        if csv_file.exists():
            df = pd.read_csv(csv_file)
        elif self.aggregate_file.exists():
            df = self._load_from_aggregate(ticker)
        else:
            raise FileNotFoundError(f"Stock file not found: {csv_file}")

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def _load_from_aggregate(self, ticker: str) -> pd.DataFrame:
        """Load stock rows from aggregated Kaggle CSV."""
        aggregate_df = self._load_aggregate_dataset()
        ticker_upper = ticker.upper()
        df = aggregate_df[aggregate_df['symbol'] == ticker_upper].copy()

        if df.empty:
            raise FileNotFoundError(
                f"{ticker_upper} not found in {self.aggregate_file}"
            )

        return df

    def _load_aggregate_dataset(self) -> pd.DataFrame:
        """Load and cache the aggregated Kaggle dataset."""
        if self._aggregate_df is None:
            if not self.aggregate_file.exists():
                raise FileNotFoundError(
                    f"Aggregate dataset missing: {self.aggregate_file}"
                )

            df = pd.read_csv(self.aggregate_file)
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
            )

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            for col in ['adj_close', 'close', 'high', 'low', 'open', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].str.upper()

            df = df.dropna(subset=['symbol', 'date'])
            self._aggregate_df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        return self._aggregate_df

    def _get_available_tickers(self) -> List[str]:
        """Return a list of tickers based on available files or aggregate CSV."""
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        tickers = [
            f.stem for f in csv_files
            if f.name not in ['sp500_companies.csv', 'sp500_index.csv', 'sp500_stocks.csv']
        ]

        if tickers:
            return sorted(set(tickers))

        if self.aggregate_file.exists():
            aggregate_df = self._load_aggregate_dataset()
            return sorted(aggregate_df['symbol'].unique().tolist())

        return []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models (Research-Enhanced).

        Based on research showing these features significantly improve accuracy:
        - Lagged returns at multiple scales (1d, 5d, 10d, 20d, 60d, 252d)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Extended moving averages (5, 10, 20, 50, 200-day)
        - Volatility measures

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Extended lagged returns (research recommendation)
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)
        df['returns_60d'] = df['close'].pct_change(60)

        # Long-term momentum (annual)
        if len(df) >= 252:
            df['returns_252d'] = df['close'].pct_change(252)
        else:
            df['returns_252d'] = 0.0

        # Extended moving averages (research shows 50 and 200-day are critical)
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # Moving average crossovers (strong signals)
        df['sma_5_20_cross'] = (df['sma_5'] / df['sma_20']) - 1
        df['sma_50_200_cross'] = (df['sma_50'] / df['sma_200']) - 1

        # RSI (Relative Strength Index) - 14-day standard
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger Bands (20-day, 2 std dev)
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_ma + (2 * bb_std)
        df['bb_lower'] = bb_ma - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Enhanced volatility measures
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()

        # Volume features
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_10'] + 1e-10)
        df['volume_trend'] = df['volume_ma_10'] / (df['volume_ma_20'] + 1e-10)

        # Price momentum indicators
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1

        # High-Low spread (volatility proxy)
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_ma'] = df['hl_spread'].rolling(10).mean()

        # Target: Next day direction
        df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)

        # Multi-class target (±0.5% threshold per research)
        next_return = df['close'].pct_change(1).shift(-1)
        df['target_multiclass'] = 1  # neutral
        df.loc[next_return > 0.005, 'target_multiclass'] = 2  # up (>0.5%)
        df.loc[next_return < -0.005, 'target_multiclass'] = 0  # down (<-0.5%)

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        features: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/GRU training (Research-Enhanced).

        Research shows 60-day lookback window is optimal for capturing
        both short-term patterns and longer-term trends.

        Args:
            df: DataFrame with features
            sequence_length: Number of time steps (default 60 per research)
            features: List of feature column names

        Returns:
            (X, y) arrays for training
        """
        if features is None:
            features = [
                # Lagged returns (research-recommended)
                'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d',
                # Technical indicators (critical per research)
                'rsi_14', 'macd', 'macd_diff',
                # Bollinger Bands
                'bb_width', 'bb_position',
                # Moving averages
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'sma_5_20_cross', 'sma_50_200_cross',
                # Volatility
                'volatility_10d', 'volatility_20d', 'volatility_60d',
                # Volume
                'volume_ratio', 'volume_trend',
                # Momentum
                'momentum_10d', 'momentum_20d',
                # Spread
                'hl_spread', 'hl_spread_ma'
            ]

        # Remove NaN rows
        df_clean = df.dropna(subset=features + ['target_multiclass'])

        if len(df_clean) < sequence_length + 1:
            raise ValueError(f"Not enough data: need {sequence_length + 1}, have {len(df_clean)}")

        X_list = []
        y_list = []

        for i in range(len(df_clean) - sequence_length):
            # Get sequence
            sequence = df_clean.iloc[i:i + sequence_length][features].values
            target = df_clean.iloc[i + sequence_length]['target_multiclass']

            X_list.append(sequence)
            y_list.append(target)

        return np.array(X_list), np.array(y_list)

    def create_tabular_features(
        self,
        df: pd.DataFrame,
        features: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create tabular features for Gradient Boost (Research-Enhanced).

        Research shows XGBoost performs best with comprehensive feature set
        including all technical indicators.

        Args:
            df: DataFrame with features
            features: List of feature column names

        Returns:
            (X, y) arrays for training
        """
        if features is None:
            features = [
                # Lagged returns (research-critical)
                'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_60d', 'returns_252d',
                # Technical indicators
                'rsi_14', 'macd', 'macd_signal', 'macd_diff',
                # Bollinger Bands
                'bb_width', 'bb_position',
                # Moving averages
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'sma_5_20_cross', 'sma_50_200_cross',
                # Volatility
                'volatility_10d', 'volatility_20d', 'volatility_60d',
                # Volume
                'volume_ratio', 'volume_trend',
                # Momentum
                'momentum_10d', 'momentum_20d',
                # Spread
                'hl_spread', 'hl_spread_ma'
            ]

        df_clean = df.dropna(subset=features + ['target_multiclass'])

        X = df_clean[features].values
        y = df_clean['target_multiclass'].values

        return X, y

    def process_all_stocks(
        self,
        max_stocks: int = 50,
        sequence_length: int = 60
    ) -> Dict[str, Any]:
        """Process multiple stocks and combine data (Research-Enhanced).

        Research recommends 60-day sequences for optimal temporal pattern capture.

        Args:
            max_stocks: Maximum number of stocks to process
            sequence_length: Sequence length for LSTM/GRU (default 60 per research)

        Returns:
            Dictionary with processed data
        """
        logger.info(f"Processing up to {max_stocks} stocks with {sequence_length}-day sequences...")

        tickers = self._get_available_tickers()

        if not tickers:
            raise FileNotFoundError("No stock CSV files found")

        # Limit number of stocks
        tickers = tickers[:max_stocks]
        logger.info(f"Processing {len(tickers)} stocks")

        X_lstm_all = []
        y_lstm_all = []
        X_gb_all = []
        y_gb_all = []

        successful_stocks = []

        for ticker in tickers:
            try:
                # Load and preprocess
                df = self.load_stock_data(ticker)
                df = self.engineer_features(df)

                # Create LSTM sequences
                X_lstm, y_lstm = self.create_sequences(df, sequence_length)
                X_lstm_all.append(X_lstm)
                y_lstm_all.append(y_lstm)

                # Create Gradient Boost features
                X_gb, y_gb = self.create_tabular_features(df)
                X_gb_all.append(X_gb)
                y_gb_all.append(y_gb)

                successful_stocks.append(ticker)
                logger.info(f"  ✓ {ticker}: {len(X_lstm)} sequences, {len(X_gb)} samples")

            except Exception as e:
                logger.warning(f"  ✗ {ticker}: {e}")
                continue

        if not successful_stocks:
            raise ValueError("No stocks processed successfully")

        # Combine all data
        X_lstm_combined = np.vstack(X_lstm_all)
        y_lstm_combined = np.concatenate(y_lstm_all)
        X_gb_combined = np.vstack(X_gb_all)
        y_gb_combined = np.concatenate(y_gb_all)

        logger.info(f"\n✓ Combined data:")
        logger.info(f"  - LSTM/GRU: {X_lstm_combined.shape} sequences")
        logger.info(f"  - Gradient Boost: {X_gb_combined.shape} samples")

        # Save processed data
        processed_data = {
            'X_lstm': X_lstm_combined,
            'y_lstm': y_lstm_combined,
            'X_gb': X_gb_combined,
            'y_gb': y_gb_combined,
            'tickers': successful_stocks,
            'sequence_length': sequence_length
        }

        # Save to disk
        output_file = self.processed_data_dir / "training_data.npz"
        np.savez(
            output_file,
            X_lstm=X_lstm_combined,
            y_lstm=y_lstm_combined,
            X_gb=X_gb_combined,
            y_gb=y_gb_combined,
            tickers=successful_stocks
        )

        logger.info(f"\n✓ Saved processed data to: {output_file}")

        return processed_data

    def create_train_val_test_split(
        self,
        data: Dict[str, Any],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """Split data into train, validation, and test sets.

        Args:
            data: Dictionary with processed data
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Dictionary with split data
        """
        X_lstm = data['X_lstm']
        y_lstm = data['y_lstm']
        X_gb = data['X_gb']
        y_gb = data['y_gb']

        # Calculate split indices
        n_lstm = len(X_lstm)
        train_end_lstm = int(n_lstm * train_ratio)
        val_end_lstm = int(n_lstm * (train_ratio + val_ratio))

        n_gb = len(X_gb)
        train_end_gb = int(n_gb * train_ratio)
        val_end_gb = int(n_gb * (train_ratio + val_ratio))

        splits = {
            # LSTM/GRU splits
            'X_lstm_train': X_lstm[:train_end_lstm],
            'y_lstm_train': y_lstm[:train_end_lstm],
            'X_lstm_val': X_lstm[train_end_lstm:val_end_lstm],
            'y_lstm_val': y_lstm[train_end_lstm:val_end_lstm],
            'X_lstm_test': X_lstm[val_end_lstm:],
            'y_lstm_test': y_lstm[val_end_lstm:],

            # Gradient Boost splits
            'X_gb_train': X_gb[:train_end_gb],
            'y_gb_train': y_gb[:train_end_gb],
            'X_gb_val': X_gb[train_end_gb:val_end_gb],
            'y_gb_val': y_gb[train_end_gb:val_end_gb],
            'X_gb_test': X_gb[val_end_gb:],
            'y_gb_test': y_gb[val_end_gb:],
        }

        logger.info(f"\nData splits:")
        logger.info(f"  LSTM/GRU Train: {len(splits['X_lstm_train'])}")
        logger.info(f"  LSTM/GRU Val:   {len(splits['X_lstm_val'])}")
        logger.info(f"  LSTM/GRU Test:  {len(splits['X_lstm_test'])}")
        logger.info(f"  GB Train: {len(splits['X_gb_train'])}")
        logger.info(f"  GB Val:   {len(splits['X_gb_val'])}")
        logger.info(f"  GB Test:  {len(splits['X_gb_test'])}")

        # Save splits
        output_file = self.processed_data_dir / "training_splits.npz"
        np.savez(output_file, **splits)
        logger.info(f"\n✓ Saved splits to: {output_file}")

        return splits


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("FreshStart - Data Preprocessing")
    logger.info("=" * 60)

    preprocessor = DataPreprocessor()

    try:
        # Process stocks
        data = preprocessor.process_all_stocks(max_stocks=50, sequence_length=30)

        # Create splits
        splits = preprocessor.create_train_val_test_split(data)

        logger.info("\n" + "=" * 60)
        logger.info("✓ Preprocessing complete!")
        logger.info("=" * 60)
        logger.info("\nNext step:")
        logger.info("Run: python -m models.prediction.train_models")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n✗ Data not found: {e}")
        logger.error("\nRun first: python -m data.download_kaggle_data")
        return 1

    except Exception as e:
        logger.error(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
