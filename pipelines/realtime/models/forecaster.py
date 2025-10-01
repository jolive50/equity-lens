"""Probabilistic forecasting models for stock price direction prediction."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from probabilistic forecasting model."""
    direction: str  # "up", "down", "neutral"
    confidence: float
    daily_probs: List[Dict[str, Any]]
    horizon_95: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_metadata: Dict[str, Any]


class TechnicalIndicators:
    """Calculate technical indicators from price data."""
    
    @staticmethod
    def sma(prices: List[float], window: int) -> List[float]:
        """Simple Moving Average."""
        if len(prices) < window:
            return [np.mean(prices[:i+1]) for i in range(len(prices))]
        
        sma_values = []
        for i in range(len(prices)):
            if i < window - 1:
                sma_values.append(np.mean(prices[:i+1]))
            else:
                sma_values.append(np.mean(prices[i-window+1:i+1]))
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], window: int) -> List[float]:
        """Exponential Moving Average."""
        if not prices:
            return []
        
        alpha = 2 / (window + 1)
        ema_values = [prices[0]]
        
        for price in prices[1:]:
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], window: int = 14) -> List[float]:
        """Relative Strength Index."""
        if len(prices) < window + 1:
            return [50.0] * len(prices)
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        rsi_values = [50.0]  # First value
        
        for i in range(window, len(deltas)):
            avg_gain = np.mean(gains[i-window:i])
            avg_loss = np.mean(losses[i-window:i])
            
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD indicator."""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast, ema_slow)]
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = [macd_val - sig_val for macd_val, sig_val in zip(macd_line, signal_line)]
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: List[float], window: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands."""
        sma_values = TechnicalIndicators.sma(prices, window)
        
        upper_bands = []
        lower_bands = []
        
        for i in range(len(prices)):
            if i < window - 1:
                period_prices = prices[:i+1]
            else:
                period_prices = prices[i-window+1:i+1]
            
            std = np.std(period_prices)
            upper_bands.append(sma_values[i] + (std_dev * std))
            lower_bands.append(sma_values[i] - (std_dev * std))
        
        return upper_bands, sma_values, lower_bands


class FeatureEngineer:
    """Engineer features from market data for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, market_data: List[Dict]) -> np.ndarray:
        """Create feature matrix from market data."""
        if len(market_data) < 30:
            logger.warning("Insufficient data for feature engineering")
            return np.array([])
        
        # Extract price and volume data
        prices = [float(data.get('close', 0)) for data in market_data]
        volumes = [int(data.get('volume', 0)) for data in market_data]
        
        if not prices or all(p == 0 for p in prices):
            return np.array([])
        
        features = []
        
        # Price-based features
        features.extend(self._price_features(prices))
        
        # Volume-based features
        features.extend(self._volume_features(volumes))
        
        # Technical indicators
        features.extend(self._technical_features(prices))
        
        # Momentum features
        features.extend(self._momentum_features(prices))
        
        # Volatility features
        features.extend(self._volatility_features(prices))
        
        return np.array(features).reshape(1, -1)
    
    def _price_features(self, prices: List[float]) -> List[float]:
        """Extract price-based features."""
        features = []
        
        # Recent price changes
        if len(prices) >= 2:
            features.append((prices[-1] - prices[-2]) / prices[-2])  # 1-day return
        else:
            features.append(0.0)
        
        if len(prices) >= 5:
            features.append((prices[-1] - prices[-5]) / prices[-5])  # 5-day return
        else:
            features.append(0.0)
        
        if len(prices) >= 10:
            features.append((prices[-1] - prices[-10]) / prices[-10])  # 10-day return
        else:
            features.append(0.0)
        
        if len(prices) >= 20:
            features.append((prices[-1] - prices[-20]) / prices[-20])  # 20-day return
        else:
            features.append(0.0)
        
        # Price position relative to recent range
        if len(prices) >= 20:
            recent_prices = prices[-20:]
            min_price = min(recent_prices)
            max_price = max(recent_prices)
            if max_price > min_price:
                features.append((prices[-1] - min_price) / (max_price - min_price))
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        return features
    
    def _volume_features(self, volumes: List[int]) -> List[float]:
        """Extract volume-based features."""
        features = []
        
        if len(volumes) >= 2:
            features.append(volumes[-1] / max(volumes[-2], 1))  # Volume ratio
        else:
            features.append(1.0)
        
        if len(volumes) >= 10:
            avg_volume = np.mean(volumes[-10:])
            features.append(volumes[-1] / max(avg_volume, 1))  # Volume vs average
        else:
            features.append(1.0)
        
        return features
    
    def _technical_features(self, prices: List[float]) -> List[float]:
        """Extract technical indicator features."""
        features = []
        
        # RSI
        rsi_values = TechnicalIndicators.rsi(prices, 14)
        features.append(rsi_values[-1] / 100.0)  # Normalize to 0-1
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
        if macd_line and signal_line:
            features.append(macd_line[-1])
            features.append(signal_line[-1])
            features.append(histogram[-1])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Bollinger Bands position
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices)
        if upper and lower:
            if upper[-1] > lower[-1]:
                features.append((prices[-1] - lower[-1]) / (upper[-1] - lower[-1]))
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        return features
    
    def _momentum_features(self, prices: List[float]) -> List[float]:
        """Extract momentum features."""
        features = []
        
        # Moving average crossovers
        sma_5 = TechnicalIndicators.sma(prices, 5)
        sma_20 = TechnicalIndicators.sma(prices, 20)
        
        if sma_5 and sma_20:
            features.append(1.0 if sma_5[-1] > sma_20[-1] else 0.0)  # Golden cross
        else:
            features.append(0.5)
        
        # Price momentum
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5]
            features.append(momentum)
        else:
            features.append(0.0)
        
        return features
    
    def _volatility_features(self, prices: List[float]) -> List[float]:
        """Extract volatility features."""
        features = []
        
        # Recent volatility
        if len(prices) >= 10:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, min(10, len(prices)))]
            volatility = np.std(returns) if returns else 0.0
            features.append(volatility)
        else:
            features.append(0.0)
        
        return features


class ProbabilisticForecaster:
    """Probabilistic forecasting model for stock direction prediction."""
    
    def __init__(self, model_type: str = "gradient_boosting"):
        """Initialize the forecaster with specified model type."""
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Wrap with calibration for better probability estimates
        self.calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
    
    def train(self, training_data: List[Dict], labels: List[str]) -> Dict[str, float]:
        """Train the forecasting model."""
        if len(training_data) < 50:
            logger.warning("Insufficient training data")
            return {"accuracy": 0.0, "brier_score": 1.0}
        
        # Create features
        X = []
        y = []
        
        for i, data in enumerate(training_data):
            features = self.feature_engineer.create_features(data.get('market_data', []))
            if features.size > 0:
                X.append(features.flatten())
                y.append(labels[i])
        
        if len(X) < 10:
            logger.warning("Insufficient valid training samples")
            return {"accuracy": 0.0, "brier_score": 1.0}
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.calibrated_model.fit(X, y)
        self.is_trained = True
        
        # Evaluate
        predictions = self.calibrated_model.predict(X)
        probabilities = self.calibrated_model.predict_proba(X)
        
        accuracy = np.mean(predictions == y)
        
        # Calculate Brier score
        label_to_idx = {"down": 0, "neutral": 1, "up": 2}
        y_binary = np.array([label_to_idx[label] for label in y])
        brier_score = brier_score_loss(y_binary, probabilities[:, 2])  # Using 'up' class
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}, Brier score: {brier_score:.3f}")
        
        return {"accuracy": accuracy, "brier_score": brier_score}
    
    def predict(self, market_data: List[Dict], fundamentals: Dict[str, float]) -> ForecastResult:
        """Make probabilistic forecast."""
        if not self.is_trained:
            logger.warning("Model not trained, returning neutral prediction")
            return self._default_prediction()
        
        # Create features
        features = self.feature_engineer.create_features(market_data)
        if features.size == 0:
            logger.warning("Could not create features, returning neutral prediction")
            return self._default_prediction()
        
        # Get predictions
        probabilities = self.calibrated_model.predict_proba(features)
        prediction = self.calibrated_model.predict(features)[0]
        
        # Map probabilities to direction
        prob_up = probabilities[0][2]  # Assuming order: down, neutral, up
        prob_down = probabilities[0][0]
        prob_neutral = probabilities[0][1]
        
        # Determine direction and confidence
        if prob_up > prob_down and prob_up > prob_neutral:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_neutral
        
        # Generate daily probabilities for next 30 days
        daily_probs = self._generate_daily_probabilities(prob_up, prob_down, prob_neutral)
        
        # Calculate 95% confidence horizon
        horizon_95 = self._calculate_95_horizon(daily_probs)
        
        # Feature importance
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
                    "up": prob_up,
                    "down": prob_down,
                    "neutral": prob_neutral
                }
            }
        )
    
    def _default_prediction(self) -> ForecastResult:
        """Return default neutral prediction."""
        base_date = datetime.utcnow()
        daily_probs = []
        
        for i in range(1, 31):
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_probs.append({
                "date": date_str,
                "up": 0.33,
                "down": 0.33,
                "neutral": 0.34
            })
        
        return ForecastResult(
            direction="neutral",
            confidence=0.34,
            daily_probs=daily_probs,
            horizon_95={"days": 0, "class": "neutral"},
            feature_importance={},
            model_metadata={"model_type": "default"}
        )
    
    def _generate_daily_probabilities(self, prob_up: float, prob_down: float, prob_neutral: float) -> List[Dict[str, Any]]:
        """Generate daily probabilities with decay."""
        base_date = datetime.utcnow()
        daily_probs = []
        
        for i in range(1, 31):
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Apply decay factor
            decay_factor = max(0.1, 1.0 - (i * 0.02))
            
            daily_probs.append({
                "date": date_str,
                "up": max(0.1, min(0.9, prob_up * decay_factor)),
                "down": max(0.1, min(0.9, prob_down * decay_factor)),
                "neutral": max(0.1, min(0.9, prob_neutral * decay_factor))
            })
        
        return daily_probs
    
    def _calculate_95_horizon(self, daily_probs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate 95% confidence horizon."""
        max_confidence_days = 0
        first_drop_day = None
        
        for i, day_prob in enumerate(daily_probs):
            max_prob = max(day_prob["up"], day_prob["down"], day_prob["neutral"])
            if max_prob >= 0.95:
                max_confidence_days = i + 1
            elif first_drop_day is None and max_prob < 0.95:
                first_drop_day = day_prob["date"]
        
        # Find the direction with highest confidence
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
        """Get feature importance from trained model."""
        if not self.is_trained or not hasattr(self.calibrated_model.base_estimator, 'feature_importances_'):
            return {}
        
        importance = self.calibrated_model.base_estimator.feature_importances_
        feature_names = [
            "1d_return", "5d_return", "10d_return", "20d_return", "price_position",
            "volume_ratio", "volume_vs_avg", "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_position", "golden_cross", "momentum", "volatility"
        ]
        
        return dict(zip(feature_names, importance.tolist()))


def create_forecaster(model_type: str = "gradient_boosting") -> ProbabilisticForecaster:
    """Factory function to create a forecaster."""
    return ProbabilisticForecaster(model_type)


if __name__ == "__main__":
    # Test the forecaster
    forecaster = create_forecaster()
    
    # Mock training data
    training_data = []
    labels = []
    
    for i in range(100):
        # Generate mock market data
        base_price = 100 + i * 0.1
        market_data = []
        for j in range(30):
            price = base_price + np.random.normal(0, 1)
            market_data.append({
                "close": price,
                "volume": 1000000 + np.random.randint(-100000, 100000),
                "timestamp": f"2024-01-{j+1:02d}"
            })
        
        training_data.append({"market_data": market_data})
        
        # Generate mock labels
        if i < 30:
            labels.append("down")
        elif i < 70:
            labels.append("neutral")
        else:
            labels.append("up")
    
    # Train model
    metrics = forecaster.train(training_data, labels)
    print(f"Training metrics: {metrics}")
    
    # Test prediction
    test_data = [{"market_data": training_data[0]["market_data"]}]
    result = forecaster.predict(test_data[0]["market_data"], {})
    
    print(f"Prediction: {result.direction}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"95% Horizon: {result.horizon_95}")
