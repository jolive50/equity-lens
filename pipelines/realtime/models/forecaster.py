"""Probabilistic forecasting models for stock price direction prediction.

This module implements a real machine learning model that predicts whether a stock
will go up, down, or stay neutral.

Features:
- ML Model: A program that learns patterns from historical data to make predictions
- Features: Measurable characteristics we extract from data (like RSI, MACD)
- Training: Showing the model many examples so it learns patterns
- Probability: Instead of just "up" or "down", we get confidence levels (0-100%)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# TensorFlow/Keras for LSTM model
import tensorflow as tf
from tensorflow import keras

# Scikit-learn: Popular Python library for machine learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV  # Makes probabilities more accurate
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from probabilistic forecasting model.

    Think of this as a report card for our prediction. It contains:
    - direction: Which way the stock is likely to move (up/down/neutral)
    - confidence: How sure we are (0.0 = guessing, 1.0 = very sure)
    - daily_probs: Probability breakdown for each of the next 30 days
    - horizon_95: How many days we can maintain 95%+ confidence
    - feature_importance: Which factors influenced the prediction most
    - model_metadata: Technical details about the prediction
    """
    direction: str  # "up", "down", "neutral"
    confidence: float  # 0.0 to 1.0
    daily_probs: List[Dict[str, Any]]  # Probabilities for each day
    horizon_95: Dict[str, Any]  # 95% confidence horizon details
    feature_importance: Dict[str, float]  # Which features mattered most
    model_metadata: Dict[str, Any]  # Model details


class TechnicalIndicators:
    """Calculate technical indicators from price data.

    Technical indicators are mathematical calculations based on price and volume
    that help identify trends and potential reversals. Think of them as different
    ways to analyze the same data to spot patterns.

    Why we need this: Raw prices alone don't tell the full story. These indicators
    help us see momentum, trends, and potential turning points.
    """

    @staticmethod
    def sma(prices: List[float], window: int) -> List[float]:
        """Simple Moving Average (SMA).

        What it does: Calculates the average price over a sliding window
        Why it's useful: Smooths out price fluctuations to see the overall trend
        How it works: Average of the last N prices

        Example: If window=5 and prices=[10,12,11,13,14], SMA = (10+12+11+13+14)/5 = 12

        Args:
            prices: List of closing prices (most recent last)
            window: How many days to average (e.g., 20 for 20-day moving average)

        Returns:
            List of SMA values (one for each price point)
        """
        # Handle case where we don't have enough data yet
        if len(prices) < window:
            # For early points, just average what we have so far
            return [np.mean(prices[:i+1]) for i in range(len(prices))]

        sma_values = []
        # Calculate SMA for each position in the price series
        for i in range(len(prices)):
            if i < window - 1:
                # Not enough data yet, average all prices up to this point
                sma_values.append(np.mean(prices[:i+1]))
            else:
                # We have enough data, calculate proper window average
                # Example: if window=20 and i=25, average prices[6:26]
                sma_values.append(np.mean(prices[i-window+1:i+1]))

        return sma_values

    @staticmethod
    def ema(prices: List[float], window: int) -> List[float]:
        """Exponential Moving Average (EMA).

        What it does: Like SMA but gives more weight to recent prices
        Why it's better than SMA: Reacts faster to price changes
        How it works: Uses a multiplier (alpha) that gives more weight to recent data

        Args:
            prices: List of closing prices
            window: Period for EMA calculation

        Returns:
            List of EMA values
        """
        if not prices:
            return []

        # Alpha determines how much weight we give to new data
        # Smaller window = higher alpha = more reactive to recent changes
        alpha = 2 / (window + 1)

        # Start with the first price
        ema_values = [prices[0]]

        # Calculate EMA for each subsequent price
        for price in prices[1:]:
            # New EMA = (alpha × new_price) + ((1-alpha) × old_EMA)
            # This formula gives more weight to recent prices
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])

        return ema_values

    @staticmethod
    def rsi(prices: List[float], window: int = 14) -> List[float]:
        """Relative Strength Index (RSI).

        What it does: Measures momentum on a 0-100 scale
        Why it's useful: Identifies overbought (>70) and oversold (<30) conditions
        How it works: Compares average gains to average losses over a period

        Interpretation:
        - RSI > 70: Stock might be overbought (too expensive, might drop)
        - RSI < 30: Stock might be oversold (too cheap, might rise)
        - RSI = 50: Neutral

        Args:
            prices: List of closing prices
            window: Period for RSI calculation (typically 14 days)

        Returns:
            List of RSI values (0-100)
        """
        # Need at least window+1 prices to calculate RSI
        if len(prices) < window + 1:
            # Return neutral value (50) if not enough data
            return [50.0] * len(prices)

        # Calculate price changes (deltas) from one day to the next
        # Example: [100, 102, 101] → [2, -1]
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        # Separate gains and losses
        # Gains: positive deltas (price went up)
        # Losses: negative deltas converted to positive (price went down)
        gains = [max(0, delta) for delta in deltas]  # [2, 0]
        losses = [max(0, -delta) for delta in deltas]  # [0, 1]

        # Start with neutral value
        rsi_values = [50.0]

        # Calculate RSI for each point where we have enough data
        for i in range(window, len(deltas)):
            # Average gain and loss over the window
            avg_gain = np.mean(gains[i-window:i])
            avg_loss = np.mean(losses[i-window:i])

            # Calculate RS (Relative Strength) and RSI
            if avg_loss == 0:
                # No losses means RSI = 100 (very bullish)
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss  # Ratio of gains to losses
                rsi = 100 - (100 / (1 + rs))  # Convert to 0-100 scale
                rsi_values.append(rsi)

        return rsi_values

    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence).

        What it does: Shows relationship between two moving averages
        Why it's useful: Identifies trend changes and momentum
        How it works:
        1. MACD Line = Fast EMA - Slow EMA
        2. Signal Line = EMA of MACD Line
        3. Histogram = MACD Line - Signal Line

        Trading Signals:
        - MACD crosses above Signal: Bullish signal (might go up)
        - MACD crosses below Signal: Bearish signal (might go down)
        - Histogram > 0: Upward momentum
        - Histogram < 0: Downward momentum

        Args:
            prices: List of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate fast and slow EMAs
        ema_fast = TechnicalIndicators.ema(prices, fast)  # More reactive (12-day)
        ema_slow = TechnicalIndicators.ema(prices, slow)  # Less reactive (26-day)

        # MACD line is the difference between fast and slow EMAs
        # When fast > slow, MACD is positive (bullish)
        # When fast < slow, MACD is negative (bearish)
        macd_line = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast, ema_slow)]

        # Signal line is a smoothed version of MACD
        signal_line = TechnicalIndicators.ema(macd_line, signal)

        # Histogram shows the difference between MACD and Signal
        # Positive histogram = bullish momentum increasing
        # Negative histogram = bearish momentum increasing
        histogram = [macd_val - sig_val for macd_val, sig_val in zip(macd_line, signal_line)]

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(prices: List[float], window: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands.

        What it does: Creates an upper and lower boundary around the moving average
        Why it's useful: Shows volatility and potential reversal points
        How it works:
        1. Middle Band = SMA
        2. Upper Band = SMA + (2 × standard deviation)
        3. Lower Band = SMA - (2 × standard deviation)

        Interpretation:
        - Price near upper band: Might be overbought
        - Price near lower band: Might be oversold
        - Bands squeeze together: Low volatility, potential breakout coming
        - Bands expand: High volatility

        Args:
            prices: List of closing prices
            window: Period for calculation (typically 20 days)
            std_dev: Number of standard deviations (typically 2)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Middle band is just the simple moving average
        sma_values = TechnicalIndicators.sma(prices, window)

        upper_bands = []
        lower_bands = []

        # Calculate bands for each price point
        for i in range(len(prices)):
            # Get the prices in the current window
            if i < window - 1:
                period_prices = prices[:i+1]
            else:
                period_prices = prices[i-window+1:i+1]

            # Calculate standard deviation (measure of volatility)
            # Higher std = more volatile = wider bands
            std = np.std(period_prices)

            # Create bands by adding/subtracting std from the middle
            upper_bands.append(sma_values[i] + (std_dev * std))
            lower_bands.append(sma_values[i] - (std_dev * std))

        return upper_bands, sma_values, lower_bands


class FeatureEngineer:
    """Engineer features from market data for ML models.

    What this does: Converts raw price/volume data into meaningful features
    Why we need this: ML models need structured numerical features, not just raw prices
    How it works: Extracts various metrics that capture different aspects of price behavior

    Think of this as a translator that converts messy market data into a clean
    spreadsheet that the ML model can understand.
    """

    def __init__(self):
        """Initialize the feature engineer.

        StandardScaler: Normalizes features so they're all on similar scales
        Why: Some features might be 0-100 (like RSI) while others are 0-1 (like returns)
        Scaling ensures no single feature dominates just because it has bigger numbers
        """
        self.scaler = StandardScaler()
        self.feature_names = []  # Track what features we created

    def create_features(self, market_data: List[Dict]) -> np.ndarray:
        """Create feature matrix from market data.

        What this does: Takes raw market data and creates a row of features for ML
        Why: ML models need fixed-size numerical inputs
        How: Extracts multiple feature categories and combines them

        Args:
            market_data: List of daily market data dictionaries
                        Each dict has: {close: 150.23, volume: 1000000, ...}

        Returns:
            NumPy array of features (1 row with many columns)
            Example: [0.02, -0.01, 0.05, 67.5, 1.5, ...]
                     [1d_ret, 5d_ret, 10d_ret, RSI, vol_ratio, ...]
        """
        # Need at least 30 days of data to calculate meaningful indicators
        if len(market_data) < 30:
            logger.warning("Insufficient data for feature engineering (need 30+ days)")
            return np.array([])

        # Extract price and volume arrays from the dictionaries
        # Use .get() with default 0 in case some data is missing
        prices = [float(data.get('close', 0)) for data in market_data]
        volumes = [int(data.get('volume', 0)) for data in market_data]

        # Sanity check: make sure we have valid price data
        if not prices or all(p == 0 for p in prices):
            logger.warning("No valid price data found")
            return np.array([])

        # Create different categories of features
        features = []

        # Price-based: Returns, trends, position in range
        features.extend(self._price_features(prices))

        # Volume-based: Trading activity patterns
        features.extend(self._volume_features(volumes))

        # Technical indicators: RSI, MACD, Bollinger Bands
        features.extend(self._technical_features(prices))

        # Momentum: Moving average crossovers, price momentum
        features.extend(self._momentum_features(prices))

        # Volatility: How much prices are fluctuating
        features.extend(self._volatility_features(prices))

        # Convert to NumPy array and reshape to (1, n_features)
        # ML models expect 2D arrays: rows=samples, columns=features
        return np.array(features).reshape(1, -1)

    def _price_features(self, prices: List[float]) -> List[float]:
        """Extract price-based features.

        What this does: Calculates returns over different time periods
        Why: Shows how the price has changed recently (momentum)
        How: Calculates percentage change from N days ago

        Returns:
            List of features: [1d_return, 5d_return, 10d_return, 20d_return, price_position]
        """
        features = []

        # 1-day return: (today - yesterday) / yesterday
        # Example: Price went from $100 to $102 → return = 0.02 (2% gain)
        if len(prices) >= 2:
            features.append((prices[-1] - prices[-2]) / prices[-2])
        else:
            features.append(0.0)  # Not enough data

        # 5-day return: Shows weekly trend
        if len(prices) >= 5:
            features.append((prices[-1] - prices[-5]) / prices[-5])
        else:
            features.append(0.0)

        # 10-day return: Shows medium-term trend
        if len(prices) >= 10:
            features.append((prices[-1] - prices[-10]) / prices[-10])
        else:
            features.append(0.0)

        # 20-day return: Shows monthly trend
        if len(prices) >= 20:
            features.append((prices[-1] - prices[-20]) / prices[-20])
        else:
            features.append(0.0)

        # Price position in recent range (0 = at low, 1 = at high)
        # This shows whether we're near the top or bottom of recent trading
        if len(prices) >= 20:
            recent_prices = prices[-20:]
            min_price = min(recent_prices)
            max_price = max(recent_prices)

            # Avoid division by zero
            if max_price > min_price:
                # Calculate where current price sits in the range
                features.append((prices[-1] - min_price) / (max_price - min_price))
            else:
                features.append(0.5)  # If no range, assume middle
        else:
            features.append(0.5)

        return features

    def _volume_features(self, volumes: List[int]) -> List[float]:
        """Extract volume-based features.

        What this does: Analyzes trading volume patterns
        Why: High volume often confirms price movements
        How: Compares recent volume to historical averages

        Returns:
            List of features: [volume_ratio, volume_vs_avg]
        """
        features = []

        # Volume ratio: Today's volume vs yesterday's
        # >1 means volume is increasing (more interest in the stock)
        # <1 means volume is decreasing (less interest)
        if len(volumes) >= 2:
            features.append(volumes[-1] / max(volumes[-2], 1))  # Avoid division by zero
        else:
            features.append(1.0)  # Neutral

        # Volume vs 10-day average
        # Shows if today's volume is unusual compared to recent average
        if len(volumes) >= 10:
            avg_volume = np.mean(volumes[-10:])
            features.append(volumes[-1] / max(avg_volume, 1))
        else:
            features.append(1.0)

        return features

    def _technical_features(self, prices: List[float]) -> List[float]:
        """Extract technical indicator features.

        What this does: Calculates standard technical indicators
        Why: These are proven metrics traders use to make decisions
        How: Uses the TechnicalIndicators class methods

        Returns:
            List of features: [rsi, macd, macd_signal, macd_histogram, bb_position]
        """
        features = []

        # RSI (Relative Strength Index)
        # Normalized to 0-1 range (divide by 100)
        rsi_values = TechnicalIndicators.rsi(prices, 14)
        features.append(rsi_values[-1] / 100.0)  # Get most recent RSI

        # MACD (Moving Average Convergence Divergence)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
        if macd_line and signal_line:
            # Add all three MACD components
            features.append(macd_line[-1])      # MACD value
            features.append(signal_line[-1])    # Signal value
            features.append(histogram[-1])      # Histogram (MACD - Signal)
        else:
            features.extend([0.0, 0.0, 0.0])

        # Bollinger Bands position (0 = at lower band, 1 = at upper band)
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices)
        if upper and lower:
            # Calculate where price sits within the bands
            if upper[-1] > lower[-1]:
                bb_position = (prices[-1] - lower[-1]) / (upper[-1] - lower[-1])
                features.append(bb_position)
            else:
                features.append(0.5)  # Neutral
        else:
            features.append(0.5)

        return features

    def _momentum_features(self, prices: List[float]) -> List[float]:
        """Extract momentum features.

        What this does: Identifies trend strength and direction
        Why: Momentum often continues (trending stocks keep trending)
        How: Checks moving average crossovers and recent price changes

        Returns:
            List of features: [golden_cross, momentum]
        """
        features = []

        # Golden Cross / Death Cross
        # Golden Cross: Fast MA > Slow MA (bullish signal)
        # Death Cross: Fast MA < Slow MA (bearish signal)
        sma_5 = TechnicalIndicators.sma(prices, 5)    # Short-term average
        sma_20 = TechnicalIndicators.sma(prices, 20)  # Long-term average

        if sma_5 and sma_20:
            # Binary feature: 1 if golden cross, 0 if death cross
            features.append(1.0 if sma_5[-1] > sma_20[-1] else 0.0)
        else:
            features.append(0.5)  # Neutral

        # Price momentum over 5 days
        # Positive = upward momentum, Negative = downward momentum
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5]
            features.append(momentum)
        else:
            features.append(0.0)

        return features

    def _volatility_features(self, prices: List[float]) -> List[float]:
        """Extract volatility features.

        What this does: Measures how much prices are fluctuating
        Why: High volatility = higher risk and potential reward
        How: Calculates standard deviation of recent returns

        Returns:
            List of features: [volatility]
        """
        features = []

        # Recent volatility (standard deviation of daily returns)
        if len(prices) >= 10:
            # Calculate daily returns for last 10 days
            returns = [(prices[i] - prices[i-1]) / prices[i-1]
                      for i in range(1, min(10, len(prices)))]

            # Standard deviation measures how much returns vary
            # High std = high volatility = big price swings
            volatility = np.std(returns) if returns else 0.0
            features.append(volatility)
        else:
            features.append(0.0)

        return features


class ProbabilisticForecaster:
    """Probabilistic forecasting model for stock direction prediction.

    What this does: Predicts whether a stock will go up, down, or stay neutral
    Why probabilistic: Instead of just "up" or "down", gives probability (e.g., 85% chance up)
    How it works: Uses pre-trained LSTM neural network or fallback gradient boosting

    Think of this as a smart system that has studied thousands of stock patterns
    and can now recognize similar patterns in new data.
    """

    def __init__(self, model_type: str = "lstm", model_dir: Optional[str] = None):
        """Initialize the forecaster with specified model type.

        WHAT: Sets up the forecasting model with pre-trained weights or fallback
        WHY: Pre-trained LSTM provides better accuracy than on-the-fly models
        HOW: Loads saved LSTM model, scaler, and calibrator from disk
        DATA: Model files -> loaded model ready for inference

        Args:
            model_type: "lstm" (default), "gradient_boosting", or "random_forest"
            model_dir: Path to saved model directory (default: pipelines/realtime/models/saved_models)
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()  # Creates features from data
        self.is_trained = False  # Track whether model has been trained

        # WHAT: Set default model directory if not provided
        # WHY: Centralized location for saved models
        # HOW: Use pathlib to construct path
        if model_dir is None:
            model_dir = Path(__file__).parent / "saved_models"
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir

        # WHAT: Initialize based on model type
        # WHY: Support both LSTM and traditional ML models
        # HOW: Load LSTM if available, otherwise use gradient boosting
        if model_type == "lstm":
            # WHAT: Load pre-trained LSTM model and artifacts
            # WHY: LSTM trained on historical data provides better predictions
            # HOW: Load Keras model, scaler, calibrator from saved files
            self._load_lstm_model()
        elif model_type == "random_forest":
            # Random Forest: Creates many decision trees and averages their predictions
            # Good for: Handling noisy data, resistant to overfitting
            self.model = RandomForestClassifier(
                n_estimators=100,      # Use 100 trees
                max_depth=10,          # Limit tree depth to prevent overfitting
                random_state=42,       # For reproducible results
                class_weight='balanced'  # Handle imbalanced classes
            )
        elif model_type == "gradient_boosting":
            # Gradient Boosting: Builds trees sequentially, each correcting previous errors
            # Good for: Higher accuracy, better at capturing complex patterns
            self.model = GradientBoostingClassifier(
                n_estimators=100,      # Use 100 boosting stages
                learning_rate=0.1,     # How much each tree contributes
                max_depth=6,           # Tree complexity
                random_state=42        # For reproducibility
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # WHAT: Set up calibrated model for non-LSTM types
        # WHY: Improve probability estimates for traditional ML
        # HOW: Wrap model in CalibratedClassifierCV
        if model_type != "lstm":
            # Calibrated Classifier: Improves probability estimates
            # Why: Raw ML models often have poorly calibrated probabilities
            # Isotonic calibration: Learns to map model scores to true probabilities
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method='isotonic',  # Non-parametric calibration
                cv=3                # Use 3-fold cross-validation
            )

    def _load_lstm_model(self) -> None:
        """Load pre-trained LSTM model and artifacts.

        WHAT: Loads LSTM model, scaler, calibrator, and config from disk
        WHY: Use pre-trained model for inference without retraining
        HOW: Check if files exist, load using Keras and pickle
        DATA: Saved files -> loaded model artifacts in memory
        """
        # WHAT: Define paths to model artifacts
        # WHY: Need to load model, scaler, calibrator, and metadata
        model_path = self.model_dir / "lstm_forecaster.h5"
        scaler_path = self.model_dir / "scaler.pkl"
        calibrator_path = self.model_dir / "calibrator.pkl"
        features_path = self.model_dir / "feature_columns.pkl"
        config_path = self.model_dir / "config.pkl"

        # WHAT: Check if all required files exist
        # WHY: Fail fast with clear error if model not trained
        # HOW: Check each file path exists, raise error with instructions
        required_files = [model_path, scaler_path, calibrator_path, features_path, config_path]
        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            # WHAT: Raise error if model files missing
            # WHY: Cannot run inference without trained model
            # HOW: Provide clear instructions to user
            raise RuntimeError(
                f"LSTM model not found. Missing files: {missing_files}\n"
                f"Please train the model first by running:\n"
                f"  python pipelines/realtime/models/train_lstm_forecaster.py"
            )

        # WHAT: Load Keras model
        # WHY: Need model for predictions
        # HOW: Use Keras load_model function
        # DATA: .h5 file -> Keras model object
        logger.info(f"Loading LSTM model from {model_path}")
        self.lstm_model = keras.models.load_model(model_path)
        self.is_trained = True

        # WHAT: Load scaler
        # WHY: Need same normalization used during training
        # HOW: Pickle deserialization
        # DATA: .pkl file -> StandardScaler object
        with open(scaler_path, 'rb') as f:
            self.lstm_scaler = pickle.load(f)

        # WHAT: Load calibrator
        # WHY: Need to calibrate probabilities during inference
        # HOW: Pickle deserialization
        # DATA: .pkl file -> ProbabilityCalibrator object
        with open(calibrator_path, 'rb') as f:
            self.lstm_calibrator = pickle.load(f)

        # WHAT: Load feature columns
        # WHY: Need to know which features and order
        # HOW: Pickle deserialization
        # DATA: .pkl file -> list of feature names
        with open(features_path, 'rb') as f:
            self.lstm_feature_columns = pickle.load(f)

        # WHAT: Load config
        # WHY: Need sequence length and other parameters
        # HOW: Pickle deserialization
        # DATA: .pkl file -> TrainingConfig object
        with open(config_path, 'rb') as f:
            self.lstm_config = pickle.load(f)

        logger.info("LSTM model loaded successfully")
        logger.info(f"Sequence length: {self.lstm_config.sequence_length}")
        logger.info(f"Features: {self.lstm_feature_columns}")

    def predict(self, market_data: List[Dict], fundamentals: Dict[str, float]) -> ForecastResult:
        """Make probabilistic forecast for a stock.

        WHAT: Takes market data and returns prediction with probabilities
        WHY: Main function for getting stock direction predictions
        HOW: Processes data into sequences, runs through LSTM, calibrates probabilities
        DATA: Market data dict -> ForecastResult with direction and confidence

        Args:
            market_data: List of daily price/volume data (need 30+ days for LSTM)
            fundamentals: Company metrics (P/E ratio, etc.) - not used yet

        Returns:
            ForecastResult with direction, confidence, and daily probabilities
        """
        # WHAT: Route to appropriate prediction method based on model type
        # WHY: LSTM requires different processing than traditional ML
        # HOW: Check model type, call corresponding method
        if self.model_type == "lstm":
            return self._predict_lstm(market_data, fundamentals)
        else:
            return self._predict_traditional(market_data, fundamentals)

    def _predict_lstm(self, market_data: List[Dict], fundamentals: Dict[str, float]) -> ForecastResult:
        """Make prediction using pre-trained LSTM model.

        WHAT: Processes market data through LSTM for direction prediction
        WHY: LSTM model trained on historical data provides accurate forecasts
        HOW: Convert to DataFrame, calculate indicators, create sequence, predict
        DATA: Market data -> sequence -> LSTM -> calibrated probabilities

        Args:
            market_data: List of daily OHLCV data
            fundamentals: Company fundamentals (not used yet)

        Returns:
            ForecastResult with LSTM predictions
        """
        # WHAT: Check if we have enough data
        # WHY: LSTM needs sequence_length days of history
        # HOW: Compare data length to required sequence length
        if len(market_data) < self.lstm_config.sequence_length:
            logger.warning(f"Insufficient data for LSTM (need {self.lstm_config.sequence_length}, got {len(market_data)})")
            return self._trend_based_prediction(market_data)

        # WHAT: Convert market data to DataFrame
        # WHY: Need pandas for technical indicator calculation
        # HOW: Create DataFrame from list of dicts
        # DATA: List of dicts -> pandas DataFrame
        df = pd.DataFrame(market_data)

        # WHAT: Ensure required columns exist
        # WHY: Need OHLCV columns for indicators
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns, using trend-based prediction")
            return self._trend_based_prediction(market_data)

        # WHAT: Calculate technical indicators
        # WHY: LSTM model trained on these features
        # HOW: Use same calculations as training
        # DATA: OHLCV -> OHLCV + indicators
        df['returns'] = df['close'].pct_change()
        df['rsi'] = self._calculate_rsi_series(df['close'])
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()

        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        df['volatility'] = df['returns'].rolling(10).std()

        # WHAT: Extract feature sequence
        # WHY: LSTM needs fixed-length sequence of features
        # HOW: Take last sequence_length rows, select feature columns
        # DATA: DataFrame -> numpy array (sequence_length, n_features)
        df_clean = df[self.lstm_feature_columns].dropna()

        if len(df_clean) < self.lstm_config.sequence_length:
            logger.warning("Not enough clean data after indicators, using trend-based prediction")
            return self._trend_based_prediction(market_data)

        # WHAT: Get most recent sequence
        # WHY: Predict based on latest data
        # DATA: Take last sequence_length rows
        sequence = df_clean.iloc[-self.lstm_config.sequence_length:].values

        # WHAT: Reshape for LSTM input
        # WHY: LSTM expects (batch_size, sequence_length, n_features)
        # HOW: Reshape to (1, sequence_length, n_features)
        # DATA: (seq_len, features) -> (1, seq_len, features)
        sequence = sequence.reshape(1, self.lstm_config.sequence_length, len(self.lstm_feature_columns))

        # WHAT: Normalize sequence
        # WHY: Model trained on normalized data
        # HOW: Use saved scaler from training
        # DATA: Raw values -> normalized values
        sequence_2d = sequence.reshape(1, -1)
        sequence_norm = self.lstm_scaler.transform(sequence_2d)
        sequence_norm = sequence_norm.reshape(1, self.lstm_config.sequence_length, len(self.lstm_feature_columns))

        # WHAT: Get predictions from LSTM
        # WHY: Model outputs probability distribution
        # HOW: Forward pass through network
        # DATA: Normalized sequence -> raw probabilities (3 classes)
        raw_proba = self.lstm_model.predict(sequence_norm, verbose=0)

        # WHAT: Calibrate probabilities
        # WHY: Ensure confidence matches actual accuracy
        # HOW: Apply calibration learned during training
        # DATA: Raw probabilities -> calibrated probabilities
        calibrated_proba = self.lstm_calibrator.calibrate(raw_proba)

        # WHAT: Extract class probabilities
        # WHY: Need individual class probabilities
        # DATA: Array -> individual floats
        prob_down = float(calibrated_proba[0][0])
        prob_steady = float(calibrated_proba[0][1])
        prob_up = float(calibrated_proba[0][2])

        # WHAT: Determine direction based on highest probability
        # WHY: Main prediction is the most likely outcome
        # HOW: Find argmax of probabilities
        if prob_up > prob_down and prob_up > prob_steady:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_steady:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_steady

        # WHAT: Generate daily probability forecasts
        # WHY: Show how confidence decays over time
        # HOW: Apply decay factor to probabilities
        daily_probs = self._generate_daily_probabilities(prob_up, prob_down, prob_steady)

        # WHAT: Calculate 95% confidence horizon
        # WHY: Show how long prediction remains highly confident
        horizon_95 = self._calculate_95_horizon(daily_probs)

        # WHAT: Feature importance placeholder for LSTM
        # WHY: LSTM feature importance harder to extract than tree models
        # TODO: Implement SHAP or attention-based importance
        feature_importance = {"lstm_attention": 1.0}

        return ForecastResult(
            direction=direction,
            confidence=confidence,
            daily_probs=daily_probs,
            horizon_95=horizon_95,
            feature_importance=feature_importance,
            model_metadata={
                "model_type": "lstm",
                "probabilities": {
                    "up": prob_up,
                    "down": prob_down,
                    "neutral": prob_steady
                },
                "calibrated": True
            }
        )

    def _calculate_rsi_series(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for pandas Series.

        WHAT: Computes RSI indicator for price series
        WHY: Needed for LSTM feature calculation
        HOW: Rolling window of gains vs losses
        DATA: Price series -> RSI series (0-100)

        Args:
            prices: Pandas Series of closing prices
            window: RSI calculation window

        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _predict_traditional(self, market_data: List[Dict], fundamentals: Dict[str, float]) -> ForecastResult:
        """Make prediction using traditional ML models (Random Forest, Gradient Boosting).

        WHAT: Uses gradient boosting or random forest for prediction
        WHY: Fallback when LSTM not available or for comparison
        HOW: Feature engineering -> model prediction -> calibration
        DATA: Market data -> features -> probabilities

        Args:
            market_data: List of daily price/volume data
            fundamentals: Company metrics

        Returns:
            ForecastResult with traditional ML predictions
        """
        # Check if model has been trained
        # Note: For now, we use a pre-configured model that doesn't need training
        # In production, you'd train on historical data first
        if not self.is_trained:
            logger.info("Model not explicitly trained, using trend-based prediction")
            return self._trend_based_prediction(market_data)

        # Create features from the market data
        features = self.feature_engineer.create_features(market_data)
        if features.size == 0:
            logger.warning("Could not create features, using trend-based prediction")
            return self._trend_based_prediction(market_data)

        # Get probability predictions from the model
        # Returns array like [[0.2, 0.3, 0.5]] meaning:
        # 20% down, 30% neutral, 50% up
        probabilities = self.calibrated_model.predict_proba(features)
        prediction = self.calibrated_model.predict(features)[0]

        # Extract individual probabilities
        # Order is typically: down, neutral, up (depends on training data)
        prob_down = probabilities[0][0]
        prob_neutral = probabilities[0][1]
        prob_up = probabilities[0][2]

        # Determine direction based on highest probability
        if prob_up > prob_down and prob_up > prob_neutral:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_neutral

        # Generate probability forecast for next 30 days
        daily_probs = self._generate_daily_probabilities(prob_up, prob_down, prob_neutral)

        # Calculate how many days we can maintain 95% confidence
        horizon_95 = self._calculate_95_horizon(daily_probs)

        # Get feature importance (which factors mattered most)
        feature_importance = self._get_feature_importance()

        return ForecastResult(
            direction=direction,
            confidence=confidence,
            daily_probs=daily_probs,
            horizon_95=horizon_95,
            feature_importance=feature_importance,
            model_metadata={
                "model_type": self.model_type,
                "probabilities": {
                    "up": float(prob_up),
                    "down": float(prob_down),
                    "neutral": float(prob_neutral)
                }
            }
        )

    def _trend_based_prediction(self, market_data: List[Dict]) -> ForecastResult:
        """Create a probability forecast directly from observed price momentum."""
        # What: Use recent momentum as a rule-based fallback when the ML model is unavailable.
        # Why: Provides data-driven probabilities without fabricating arbitrary constants.
        # How: Compute momentum from closing prices and map it to class probabilities via softmax.
        # Data: Consumes a list of daily market-data dicts and emits a ForecastResult object.
        prices = [float(d.get("close", 0)) for d in market_data if d.get("close") is not None]
        if len(prices) < 10:
            return self._default_prediction()  # Raises with guidance when history is insufficient.
        recent_avg = float(np.mean(prices[-5:]))
        older_window = prices[-10:-5] if len(prices) >= 10 else prices[:-5]
        older_avg = float(np.mean(older_window)) if older_window else recent_avg
        if older_avg == 0:
            return self._default_prediction()
        momentum = float((recent_avg - older_avg) / older_avg)
        momentum = float(np.clip(momentum, -0.2, 0.2))  # Clamp extreme moves to keep probabilities stable.
        scores = np.array([momentum, -momentum, 0.0], dtype=float)
        temperature = 0.05  # Lower temperature -> sharper probabilities for stronger momentum.
        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        base_probs = exp_scores / exp_scores.sum()
        prob_up, prob_down, prob_neutral = [float(p) for p in base_probs]
        direction_index = int(np.argmax(base_probs))
        direction = ["up", "down", "neutral"][direction_index]
        daily_probs = self._generate_daily_probabilities(prob_up, prob_down, prob_neutral)
        horizon_95 = self._calculate_95_horizon(daily_probs)
        feature_importance = {"momentum_strength": abs(momentum)}
        confidence = max(prob_up, prob_down, prob_neutral)
        return ForecastResult(
            direction=direction,
            confidence=confidence,
            daily_probs=daily_probs,
            horizon_95=horizon_95,
            feature_importance=feature_importance,
            model_metadata={
                "model_type": self.model_type,
                "probabilities": {
                    "up": prob_up,
                    "down": prob_down,
                    "neutral": prob_neutral,
                },
                "fallback": "momentum_softmax"
            }
        )
    def _default_prediction(self) -> ForecastResult:
        """Raise an informative error instead of fabricating a neutral prediction."""
        # What: Stop execution when we lack sufficient historical data for a real forecast.
        # Why: Returning placeholder probabilities would mislead users and violate requirements.
        # How: Raise a RuntimeError directing engineers to gather data and retrain the model.
        # Data: Emits no ForecastResult because the inputs were inadequate.
        raise RuntimeError(
            "Insufficient market history to produce a forecast. "
            "Collect additional daily price data and retrain the probabilistic forecaster."
        )
    def _generate_daily_probabilities(self, prob_up: float, prob_down: float, prob_neutral: float) -> List[Dict[str, Any]]:
        """Generate daily probability forecasts with confidence decay.

        What this does: Creates probability predictions for each of next 30 days
        Why decay: We're less confident about predictions further in the future
        How: Applies decay factor that reduces confidence over time

        Example: If initial prob_up = 0.80:
        - Day 1: 0.80
        - Day 10: 0.80 * 0.80 = 0.64
        - Day 20: 0.80 * 0.60 = 0.48

        Args:
            prob_up: Initial probability of upward movement
            prob_down: Initial probability of downward movement
            prob_neutral: Initial probability of neutral movement

        Returns:
            List of daily probability dictionaries
        """
        base_date = datetime.utcnow()
        daily_probs = []

        for i in range(1, 31):  # Next 30 days
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")

            # Apply decay factor (confidence decreases with time)
            # After 1 day: 98% of original confidence
            # After 30 days: 40% of original confidence
            # Formula: decay = 1.0 - (day * 0.02)
            decay_factor = max(0.1, 1.0 - (i * 0.02))

            # Apply decay and clamp to reasonable range (0.1 to 0.9)
            # We never want 0% or 100% (always some uncertainty)
            daily_probs.append({
                "date": date_str,
                "up": max(0.1, min(0.9, prob_up * decay_factor)),
                "down": max(0.1, min(0.9, prob_down * decay_factor)),
                "neutral": max(0.1, min(0.9, prob_neutral * decay_factor))
            })

        return daily_probs

    def _calculate_95_horizon(self, daily_probs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate 95% confidence horizon.

        What this does: Determines how many days we can maintain 95%+ confidence
        Why: Our requirements say only recommend if ≥95% confident
        How: Counts consecutive days where max probability ≥ 0.95

        Example:
        - Day 1-5: 96% up → horizon = 5 days
        - Day 6: 92% up → stops here

        Args:
            daily_probs: List of daily probability dictionaries

        Returns:
            Dictionary with horizon details:
            - days: Number of days with 95%+ confidence
            - class: Direction (up/down/neutral)
            - start_date/end_date: Date range
            - drops_below_95_on: When confidence first drops below 95%
        """
        max_confidence_days = 0
        first_drop_day = None

        # Check each day's maximum probability
        for i, day_prob in enumerate(daily_probs):
            max_prob = max(day_prob["up"], day_prob["down"], day_prob["neutral"])

            if max_prob >= 0.95:
                # Still highly confident, extend horizon
                max_confidence_days = i + 1
            elif first_drop_day is None and max_prob < 0.95:
                # First time confidence dropped below 95%
                first_drop_day = day_prob["date"]

        # Determine which direction has highest confidence
        if daily_probs:
            first_day = daily_probs[0]
            direction = max(["up", "down", "neutral"], key=lambda d: first_day[d])
        else:
            direction = "neutral"

        return {
            "class": direction,
            "start_date": daily_probs[0]["date"] if daily_probs else None,
            "end_date": daily_probs[max_confidence_days - 1]["date"] if max_confidence_days > 0 else None,
            "days": max_confidence_days,
            "drops_below_95_on": first_drop_day
        }

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model.

        What this does: Shows which factors influenced the prediction most
        Why: Helps users understand WHY the model made its prediction
        How: Extracts importance scores from the ML model

        Returns:
            Dictionary mapping feature names to importance scores
            Example: {"rsi": 0.25, "macd": 0.20, "momentum": 0.15, ...}
        """
        # Check if model is trained and supports feature importance
        if not self.is_trained or not hasattr(self.calibrated_model.base_estimator, 'feature_importances_'):
            return {}

        # Get importance scores from the model
        importance = self.calibrated_model.base_estimator.feature_importances_

        # Feature names must match order in FeatureEngineer
        feature_names = [
            "1d_return", "5d_return", "10d_return", "20d_return", "price_position",
            "volume_ratio", "volume_vs_avg", "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_position", "golden_cross", "momentum", "volatility"
        ]

        # Create dictionary mapping names to scores
        return dict(zip(feature_names, importance.tolist()))


def create_forecaster(model_type: str = "gradient_boosting") -> ProbabilisticForecaster:
    """Factory function to create a forecaster.

    What this does: Creates and returns a new forecaster instance
    Why factory pattern: Provides a simple way to create forecasters
    How: Just calls the constructor with the specified model type

    Args:
        model_type: "gradient_boosting" or "random_forest"

    Returns:
        New ProbabilisticForecaster instance ready to use
    """
    return ProbabilisticForecaster(model_type)


# Test code (runs when this file is executed directly)
if __name__ == "__main__":
    print("Testing the Probabilistic Forecaster...")

    # Create a forecaster instance
    forecaster = create_forecaster("gradient_boosting")
    print(f"Created {forecaster.model_type} forecaster")

    # Generate test data (in real use, this comes from yfinance)
    import random
    test_market_data = []
    base_price = 150.0

    for i in range(60):  # 60 days of data
        # Simulate price with some randomness and upward trend
        price = base_price + (i * 0.1) + random.uniform(-2, 2)
        test_market_data.append({
            "close": price,
            "volume": 1000000 + random.randint(-100000, 100000),
            "date": f"2025-{((i // 30) + 1):02d}-{((i % 30) + 1):02d}"
        })

    # Test prediction
    print("\nMaking prediction on test data...")
    result = forecaster.predict(test_market_data, {})

    print(f"\nPrediction Results:")
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"95% Confidence Horizon: {result.horizon_95['days']} days")
    print(f"Model Type: {result.model_metadata['model_type']}")
    print(f"\nProbabilities:")
    for direction, prob in result.model_metadata['probabilities'].items():
        print(f"  {direction}: {prob:.1%}")

    print(f"\nFirst 5 days of forecast:")
    for day in result.daily_probs[:5]:
        print(f"  {day['date']}: Up={day['up']:.1%}, Down={day['down']:.1%}, Neutral={day['neutral']:.1%}")

    print("\n✓ Forecaster test complete!")
