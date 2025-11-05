# PAM's Code Guide - Prediction Models & Price Data

**Your Responsibility:** Prediction Models (LSTM, GRU, Gradient Boost), Ensemble System, Price Data Fetcher, Model Training

**Last Updated:** 2025-11-05

---

## Table of Contents

1. [Overview](#overview)
2. [Your Components](#your-components)
3. [Base Prediction Model](#base-prediction-model)
4. [LSTM Model](#lstm-model)
5. [GRU Model](#gru-model)
6. [Gradient Boost Model](#gradient-boost-model)
7. [Ensemble System](#ensemble-system)
8. [Training Pipeline](#training-pipeline)
9. [How Everything Works Together](#how-everything-works-together)
10. [Testing Your Code](#testing-your-code)
11. [Common Questions](#common-questions)

---

## Overview

### What You Built

You created the **prediction engine** for FreshStart - the ML models that forecast whether a stock will go UP, DOWN, or stay NEUTRAL. You built THREE different model architectures and an ensemble system that combines them.

### Your Files

```
models/prediction/
├── base_predictor.py          # Base class all models inherit from
├── lstm_model.py              # Long Short-Term Memory neural network
├── gru_model.py               # Gated Recurrent Unit neural network
├── gradient_boost_model.py    # XGBoost gradient boosting classifier
├── ensemble.py                # Combines 2+ models for better accuracy
├── train_models.py            # Training script (placeholder)
└── saved_models/              # Directory for trained model weights
    ├── lstm/
    ├── gru/
    └── gradient_boost/
```

### Key Design Principles

1. **Polymorphic Interface** - All models implement `BasePredictionModel`
2. **Standardized Output** - Every model returns `PredictionResult`
3. **Graceful Fallbacks** - If model not trained, use momentum-based prediction
4. **Flexible Ensemble** - Combine any 2+ models with different strategies

---

## Your Components

### Architecture Overview

```
BasePredictionModel (Abstract Base Class)
    ↓
    ├── LSTMModel
    ├── GRUModel
    ├── GradientBoostModel
    └── PredictionEnsemble (contains 2+ models above)
```

**Why this structure?**
- **Josh's agents** can accept ANY model (single or ensemble)
- **Easy testing** - Can swap models without changing agent code
- **Future-proof** - Add new models by inheriting from base class

---

## Base Prediction Model

**File:** `models/prediction/base_predictor.py`

### What It Does

Defines the **contract** (interface) that ALL prediction models must follow. Think of it as a blueprint that ensures all models work the same way from the outside.

### PredictionResult Data Structure

```python
@dataclass
class PredictionResult:
    direction: str               # "up", "down", or "neutral"
    confidence: float            # 0.0 to 1.0 (0% to 100%)
    probabilities: Dict[str, float]  # Individual class probabilities
    metadata: Dict[str, Any]     # Model-specific information
```

**Example:**
```python
PredictionResult(
    direction="up",
    confidence=0.75,
    probabilities={
        "up": 0.75,      # 75% chance of going up
        "down": 0.15,    # 15% chance of going down
        "neutral": 0.10  # 10% chance of staying neutral
    },
    metadata={
        "model": "LSTM",
        "trained": True,
        "weights_path": "saved_models/lstm/"
    }
)
```

### BasePredictionModel Interface

```python
class BasePredictionModel(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make a prediction based on market data.

        Args:
            data: DataFrame with columns [date, open, high, low, close, volume]

        Returns:
            PredictionResult with direction, confidence, and probabilities
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, version, architecture).

        Returns:
            Dictionary with model information
        """
        pass
```

**Two required methods:**

1. **`predict(data)`** - Takes price data, returns prediction
2. **`get_model_info()`** - Returns metadata about the model

### Why Use Abstract Base Class?

**Without ABC:**
```python
# Problem: No guarantee models implement the right methods
lstm = LSTMModel()
lstm.make_prediction(data)  # Oops! Method name doesn't match

gru = GRUModel()
gru.forecast(data)  # Different method name!

# Josh's agent breaks because methods are inconsistent
```

**With ABC:**
```python
# All models MUST implement predict() and get_model_info()
lstm = LSTMModel()
lstm.predict(data)  # ✓ Works

gru = GRUModel()
gru.predict(data)   # ✓ Works

# Josh's agent works with any model polymorphically
```

**Key Benefit:** Josh's `PredictionAgent` can use ANY model without knowing the specific type:

```python
# Josh's code doesn't care if it's LSTM, GRU, or Ensemble
def run(self, ticker, market_data, fundamentals):
    result = self.model.predict(market_data)  # Works for any model!
```

---

## LSTM Model

**File:** `models/prediction/lstm_model.py`

### What Is LSTM?

**LSTM** = Long Short-Term Memory

A type of **recurrent neural network (RNN)** designed for time series data like stock prices. It "remembers" patterns from past data to predict future movements.

### How LSTM Works (Simplified)

```
Input: 30 days of stock data
    ↓
LSTM Cell 1 → processes day 1, remembers important info
    ↓
LSTM Cell 2 → processes day 2, uses memory from cell 1
    ↓
... (continues for all 30 days)
    ↓
LSTM Cell 30 → final output
    ↓
Output Layer → probabilities for up/down/neutral
```

**Key Feature:** LSTM has "memory gates" that decide what to remember and forget:
- **Forget gate:** What old information to discard
- **Input gate:** What new information to store
- **Output gate:** What to output based on memory

### Your LSTM Implementation

**Architecture:**
```python
class LSTMModel(BasePredictionModel):
    def __init__(self, weights_path: Optional[str] = None):
        # Try to load pre-trained model
        # If no weights → use fallback prediction

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        # 1. Prepare features (returns + volume)
        # 2. Run through LSTM model
        # 3. Get probabilities
        # 4. Return PredictionResult
```

### Feature Engineering

**What features does your LSTM use?**

```python
def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
    # Need at least 30 days of data
    if len(data) < 30:
        raise ValueError("Need at least 30 days of data")

    # Feature 1: Price returns (% change)
    returns = data['close'].pct_change().fillna(0).values[-30:]

    # Feature 2: Normalized volume
    volume_norm = (data['volume'] / data['volume'].rolling(10).mean()).fillna(1).values[-30:]

    # Combine into 2D array: [30 days, 2 features]
    features = np.column_stack([returns, volume_norm])

    # Reshape for LSTM: [1 sample, 30 timesteps, 2 features]
    return features.reshape(1, 30, 2)
```

**Why these features?**

1. **Returns (price changes):**
   - LSTM learns from patterns in price movements
   - Example: If prices went up 5 days in a row, what happens next?

2. **Normalized volume:**
   - Volume spikes often precede price changes
   - Normalization: volume / 10-day average
   - Example: 2.0 = volume is 2x normal

### Prediction Logic

```python
def predict(self, data: pd.DataFrame) -> PredictionResult:
    # Check if model is trained
    if not self.is_trained or self.model is None:
        return self._momentum_prediction(data)  # Fallback

    # Prepare features: [1, 30, 2]
    features = self._prepare_features(data)

    # LSTM forward pass
    probabilities = self.model.predict(features, verbose=0)
    # Returns: [[prob_down, prob_neutral, prob_up]]

    prob_down, prob_neutral, prob_up = probabilities[0]

    # Determine direction (highest probability wins)
    if prob_up > prob_down and prob_up > prob_neutral:
        direction = "up"
        confidence = float(prob_up)
    elif prob_down > prob_up and prob_down > prob_neutral:
        direction = "down"
        confidence = float(prob_down)
    else:
        direction = "neutral"
        confidence = float(prob_neutral)

    return PredictionResult(
        direction=direction,
        confidence=confidence,
        probabilities={
            "up": float(prob_up),
            "down": float(prob_down),
            "neutral": float(prob_neutral)
        },
        metadata={"model": "LSTM", "trained": self.is_trained}
    )
```

### Fallback Prediction (When Not Trained)

If LSTM weights are not loaded, use **simple momentum-based prediction:**

```python
def _momentum_prediction(self, data: pd.DataFrame) -> PredictionResult:
    # Calculate 10-day momentum
    recent_return = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]

    # If price went up > 2% → predict UP
    if recent_return > 0.02:
        return PredictionResult(
            direction="up",
            confidence=0.6,
            probabilities={"up": 0.6, "down": 0.2, "neutral": 0.2},
            metadata={"model": "LSTM_fallback", "momentum": recent_return}
        )

    # If price went down > 2% → predict DOWN
    elif recent_return < -0.02:
        return PredictionResult(
            direction="down",
            confidence=0.6,
            probabilities={"up": 0.2, "down": 0.6, "neutral": 0.2},
            metadata={"model": "LSTM_fallback", "momentum": recent_return}
        )

    # Otherwise → NEUTRAL
    else:
        return PredictionResult(
            direction="neutral",
            confidence=0.5,
            probabilities={"up": 0.33, "down": 0.33, "neutral": 0.34},
            metadata={"model": "LSTM_fallback", "momentum": recent_return}
        )
```

**Why fallback?**
- Allows system to work even without trained models
- Useful for development and testing
- Better than crashing with "model not found" error

### Example Usage

```python
from models.prediction.lstm_model import LSTMModel
import pandas as pd

# Create LSTM model
lstm = LSTMModel(weights_path="saved_models/lstm/model.h5")

# Prepare data
data = pd.DataFrame({
    'date': ['2025-01-01', '2025-01-02', ...],
    'close': [150.0, 151.5, ...],
    'volume': [1000000, 1100000, ...]
})

# Make prediction
result = lstm.predict(data)

print(f"Direction: {result.direction}")      # "up"
print(f"Confidence: {result.confidence}")    # 0.75
print(f"Probabilities: {result.probabilities}")
# {"up": 0.75, "down": 0.15, "neutral": 0.10}
```

---

## GRU Model

**File:** `models/prediction/gru_model.py`

### What Is GRU?

**GRU** = Gated Recurrent Unit

A **simpler, faster alternative to LSTM** with similar performance. GRU has fewer parameters, making it:
- Faster to train
- Less prone to overfitting
- Still good at learning time series patterns

### LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parameters | More (slower) | Fewer (faster) |
| Memory | Separate cell state | Combined hidden state |
| Performance | Slightly better on complex data | Similar on most tasks |

**Analogy:**
- **LSTM** = Professional camera (more features, more settings)
- **GRU** = Smartphone camera (simpler, faster, good enough)

### Your GRU Implementation

**Structure is nearly identical to LSTM:**

```python
class GRUModel(BasePredictionModel):
    def __init__(self, weights_path: Optional[str] = None)
    def predict(self, data: pd.DataFrame) -> PredictionResult
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray
    def _fallback_prediction(self, data: pd.DataFrame) -> PredictionResult
    def get_model_info(self) -> Dict[str, Any]
```

**Key differences from LSTM:**

1. **Model architecture:** Uses GRU layers instead of LSTM layers (when trained)
2. **Fallback uses 5-day momentum** instead of 10-day (faster signal)
3. **Threshold:** 1% instead of 2% (more sensitive)

### GRU Fallback Prediction

```python
def _fallback_prediction(self, data: pd.DataFrame) -> PredictionResult:
    # 5-day momentum (shorter window than LSTM)
    momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]

    if momentum > 0.01:  # 1% threshold (vs 2% for LSTM)
        return PredictionResult("up", 0.55, {"up": 0.55, "down": 0.25, "neutral": 0.2}, {"model": "GRU_fallback"})
    elif momentum < -0.01:
        return PredictionResult("down", 0.55, {"up": 0.25, "down": 0.55, "neutral": 0.2}, {"model": "GRU_fallback"})
    else:
        return PredictionResult("neutral", 0.5, {"up": 0.33, "down": 0.33, "neutral": 0.34}, {"model": "GRU_fallback"})
```

### When to Use GRU vs LSTM?

**Use GRU when:**
- Training time is limited
- You want faster predictions
- Dataset is not too complex

**Use LSTM when:**
- You have lots of training data
- Complex patterns to learn
- Accuracy is more important than speed

**Use both in ensemble when:**
- You want the benefits of both
- Maximum accuracy is the goal

---

## Gradient Boost Model

**File:** `models/prediction/gradient_boost_model.py`

### What Is Gradient Boosting?

**Gradient Boosting** = Ensemble of decision trees trained sequentially

Unlike LSTM/GRU (neural networks), gradient boosting uses **decision trees**:

```
Tree 1: Makes initial prediction
    ↓
Tree 2: Corrects errors from Tree 1
    ↓
Tree 3: Corrects errors from Tree 2
    ↓
... (continues for N trees)
    ↓
Final Prediction: Weighted sum of all trees
```

**XGBoost** = Extreme Gradient Boosting (optimized implementation)

### Why Include Non-Neural Network Model?

**Diversity in ensemble:**
- LSTM/GRU: Good at sequential patterns
- Gradient Boost: Good at feature interactions
- Together: Cover different types of patterns

**Different strengths:**
- Neural networks: Learn complex non-linear relationships
- Tree models: Handle feature interactions, robust to outliers

### Your Gradient Boost Implementation

**Feature extraction is different from LSTM/GRU:**

```python
def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
    # Need at least 20 days
    if len(data) < 20:
        raise ValueError("Need at least 20 days of data")

    features = []

    # Feature 1-3: Returns at different windows
    features.append(data['close'].pct_change(1).iloc[-1])   # 1-day return
    features.append(data['close'].pct_change(5).iloc[-1])   # 5-day return
    features.append(data['close'].pct_change(10).iloc[-1])  # 10-day return

    # Feature 4: Volume ratio
    avg_volume = data['volume'].rolling(10).mean().iloc[-1]
    features.append(data['volume'].iloc[-1] / avg_volume)

    # Feature 5: Moving average crossover
    sma_5 = data['close'].rolling(5).mean().iloc[-1]
    sma_20 = data['close'].rolling(20).mean().iloc[-1]
    features.append(1.0 if sma_5 > sma_20 else 0.0)  # Golden cross indicator

    return np.array([features])  # Shape: [1, 5]
```

**Why these features?**

1. **Multi-timeframe returns:** Capture short, medium, long-term trends
2. **Volume ratio:** Detect unusual trading activity
3. **MA crossover:** Classic technical indicator (golden cross = bullish)

### Prediction with XGBoost

```python
def predict(self, data: pd.DataFrame) -> PredictionResult:
    if not self.is_trained:
        return self._fallback_prediction(data)

    # Extract 5 features
    features = self._extract_features(data)  # Shape: [1, 5]

    # XGBoost prediction
    probabilities = self.model.predict_proba(features)[0]
    # Returns: [prob_down, prob_neutral, prob_up]

    prob_down, prob_neutral, prob_up = probabilities

    # Same logic as LSTM/GRU
    if prob_up > max(prob_down, prob_neutral):
        direction, confidence = "up", float(prob_up)
    elif prob_down > prob_neutral:
        direction, confidence = "down", float(prob_down)
    else:
        direction, confidence = "neutral", float(prob_neutral)

    return PredictionResult(...)
```

### Fallback Prediction

```python
def _fallback_prediction(self, data: pd.DataFrame) -> PredictionResult:
    # 10-day trend (3% threshold - more conservative)
    trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]

    if trend > 0.03:  # 3% threshold
        return PredictionResult("up", 0.6, {"up": 0.6, "down": 0.2, "neutral": 0.2}, {"model": "GB_fallback"})
    elif trend < -0.03:
        return PredictionResult("down", 0.6, {"up": 0.2, "down": 0.6, "neutral": 0.2}, {"model": "GB_fallback"})
    else:
        return PredictionResult("neutral", 0.5, {"up": 0.33, "down": 0.33, "neutral": 0.34}, {"model": "GB_fallback"})
```

### Model Loading

**Different from TensorFlow models:**

```python
def _load_model(self, weights_path: str):
    try:
        import pickle
        with open(weights_path, 'rb') as f:
            self.model = pickle.load(f)  # XGBoost saved as pickle
        self.is_trained = True
    except Exception as e:
        logger.warning(f"Could not load Gradient Boost model: {e}")
        self.is_trained = False
```

**Why pickle?**
- XGBoost models are saved as Python objects
- TensorFlow models are saved as .h5 or SavedModel format
- Different serialization methods

---

## Ensemble System

**File:** `models/prediction/ensemble.py`

### What Is an Ensemble?

**Ensemble** = Combining multiple models to make a single, better prediction

**Analogy:**
- **Single model** = One expert's opinion
- **Ensemble** = Panel of experts voting

### Why Use Ensemble?

**Benefits:**
1. **Reduced variance:** Different models make different mistakes
2. **Improved accuracy:** Average of good models beats individual models
3. **Robustness:** If one model fails, others compensate

**Example:**
- LSTM predicts: UP (70%)
- GRU predicts: UP (65%)
- GB predicts: NEUTRAL (55%)
- **Ensemble average:** UP (63% weighted)

### Your Ensemble Implementation

```python
class PredictionEnsemble(BasePredictionModel):
    def __init__(
        self,
        models: List[BasePredictionModel],  # 2+ models
        strategy: str = "weighted_average",  # How to combine
        weights: Optional[Dict[str, float]] = None  # Model weights
    ):
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        self.models = models
        self.strategy = strategy
        self.weights = weights or self._equal_weights()
```

### Ensemble Strategies

**1. Weighted Average** (Default)

Combines probability scores using predefined weights:

```python
def _weighted_average(self, predictions: List[tuple]) -> PredictionResult:
    total_up, total_down, total_neutral = 0.0, 0.0, 0.0

    for model_name, pred in predictions:
        weight = self.weights.get(model_name, 1.0)
        total_up += pred.probabilities["up"] * weight
        total_down += pred.probabilities["down"] * weight
        total_neutral += pred.probabilities["neutral"] * weight

    # Normalize by total weight
    # ... determine direction from highest probability
```

**Example:**
```
LSTM (weight=0.5): up=0.8, down=0.1, neutral=0.1
GRU  (weight=0.3): up=0.6, down=0.3, neutral=0.1
GB   (weight=0.2): up=0.5, down=0.2, neutral=0.3

Weighted average:
  up      = 0.8*0.5 + 0.6*0.3 + 0.5*0.2 = 0.68
  down    = 0.1*0.5 + 0.3*0.3 + 0.2*0.2 = 0.18
  neutral = 0.1*0.5 + 0.1*0.3 + 0.3*0.2 = 0.14

Result: UP with 68% confidence
```

**2. Simple Average**

Equal weight to all models (ignores configured weights):

```python
def _simple_average(self, predictions: List[tuple]) -> PredictionResult:
    equal_weights = {name: 1.0/len(predictions) for name, _ in predictions}
    # Use weighted_average logic with equal weights
```

**Example:**
```
LSTM: up=0.8, down=0.1, neutral=0.1
GRU:  up=0.6, down=0.3, neutral=0.1

Average:
  up      = (0.8 + 0.6) / 2 = 0.70
  down    = (0.1 + 0.3) / 2 = 0.20
  neutral = (0.1 + 0.1) / 2 = 0.10

Result: UP with 70% confidence
```

**3. Voting**

Majority vote based on predicted direction (not probabilities):

```python
def _voting(self, predictions: List[tuple]) -> PredictionResult:
    votes = {"up": 0, "down": 0, "neutral": 0}

    for _, pred in predictions:
        votes[pred.direction] += 1

    direction = max(votes, key=votes.get)  # Most votes wins
    confidence = votes[direction] / len(predictions)
```

**Example:**
```
LSTM: UP
GRU:  UP
GB:   NEUTRAL

Votes: up=2, neutral=1, down=0
Result: UP with 67% confidence (2/3 voted up)
```

### Handling Model Failures

```python
def predict(self, data: pd.DataFrame) -> PredictionResult:
    predictions = []

    for model in self.models:
        try:
            pred = model.predict(data)
            predictions.append((model.get_model_info()["name"], pred))
        except Exception as e:
            logger.warning(f"Model {model.get_model_info()['name']} failed: {e}")
            continue  # Skip failed model, continue with others

    if not predictions:
        raise RuntimeError("All models failed to predict")

    # Combine remaining predictions
    return self._weighted_average(predictions)
```

**Why this approach?**
- **Resilience:** One model failure doesn't break the ensemble
- **Transparency:** Log which models failed
- **Graceful degradation:** Use remaining models

### Example Usage

```python
from models.prediction.lstm_model import LSTMModel
from models.prediction.gru_model import GRUModel
from models.prediction.gradient_boost_model import GradientBoostModel
from models.prediction.ensemble import PredictionEnsemble

# Create individual models
lstm = LSTMModel()
gru = GRUModel()
gb = GradientBoostModel()

# Create ensemble
ensemble = PredictionEnsemble(
    models=[lstm, gru, gb],
    strategy="weighted_average",
    weights={
        "LSTM": 0.5,
        "GRU": 0.3,
        "GradientBoost": 0.2
    }
)

# Make prediction
result = ensemble.predict(data)

print(result.direction)     # "up"
print(result.confidence)    # 0.68
print(result.metadata)
# {
#   "model": "Ensemble",
#   "strategy": "weighted_average",
#   "num_models": 3,
#   "models": ["LSTM", "GRU", "GradientBoost"]
# }
```

---

## Training Pipeline

**File:** `models/prediction/train_models.py`

### Current Status

**Placeholder implementation** - Training script not yet fully implemented.

### What the Training Script Will Do

```python
def train_all_models():
    # 1. Load Kaggle SP500 dataset
    # 2. Preprocess data (normalize, create features)
    # 3. Split into train/validation/test sets
    # 4. Train LSTM model
    # 5. Train GRU model
    # 6. Train Gradient Boost model
    # 7. Evaluate and compare models
    # 8. Save best weights to saved_models/
```

### Training Workflow (Future)

**Step 1: Data Preparation**
```python
# Load Kaggle SP500 data
data = pd.read_csv("data/raw/kaggle_sp500/sp500_data.csv")

# Create features
features = engineer_features(data)  # Returns, volume, MA, etc.

# Create labels (target)
# Label = UP if tomorrow's close > today's close + threshold
# Label = DOWN if tomorrow's close < today's close - threshold
# Label = NEUTRAL otherwise
labels = create_labels(data, threshold=0.01)

# Split data
train_data, val_data, test_data = train_test_split(features, labels)
```

**Step 2: Train LSTM**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build LSTM architecture
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 2)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: up, down, neutral
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=50,
    batch_size=32
)

# Save
model.save("saved_models/lstm/model.h5")
```

**Step 3: Train GRU** (similar to LSTM, just use GRU layers)

**Step 4: Train XGBoost**
```python
import xgboost as xgb

# Create XGBoost classifier
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='multi:softprob',  # 3-class probability
    num_class=3
)

# Train
model.fit(train_features, train_labels)

# Save
import pickle
with open("saved_models/gradient_boost/model.pkl", 'wb') as f:
    pickle.dump(model, f)
```

**Step 5: Evaluate and Compare**
```python
# Test all models
lstm_accuracy = evaluate_model(lstm_model, test_data)
gru_accuracy = evaluate_model(gru_model, test_data)
gb_accuracy = evaluate_model(gb_model, test_data)

print(f"LSTM Accuracy: {lstm_accuracy:.2%}")
print(f"GRU Accuracy: {gru_accuracy:.2%}")
print(f"Gradient Boost Accuracy: {gb_accuracy:.2%}")

# Save metrics for documentation
```

### Where to Get Training Data

**Kaggle SP500 Dataset:**
1. Go to Kaggle.com
2. Search for "S&P 500 stock data"
3. Download CSV with historical prices
4. Place in `data/raw/kaggle_sp500/`

**Requirements:**
- At least 1-2 years of daily data
- Columns: date, open, high, low, close, volume
- Multiple stocks (diversified training)

---

## How Everything Works Together

### Complete Prediction Flow

**Scenario: Predicting AAPL stock direction**

**Step 1: Data Collection**
```python
# PAM's price_data.py fetches historical data
from data.fetchers.price_data import get_historical_data

market_data = get_historical_data("AAPL", period="3mo")
# Returns: 90 days of OHLCV data
```

**Step 2: Model Selection (Josh's Config)**
```python
# Josh loads configuration
from coordinator.config import WorkflowConfig

config = WorkflowConfig.from_yaml("config.yaml")
# config.yaml specifies: use LSTM + GRU ensemble

model = config.get_prediction_model()
# Returns: PredictionEnsemble([LSTMModel(), GRUModel()])
```

**Step 3: Prediction Agent (Josh)**
```python
# Josh creates prediction agent with your ensemble
from agents.prediction_agent import PredictionAgent

agent = PredictionAgent(model=model)

result = agent.run(
    ticker="AAPL",
    market_data=market_data,
    fundamentals={"pe_ratio": 25.5}
)
```

**Step 4: Your Ensemble Prediction**
```python
# Inside ensemble.predict()

# Get LSTM prediction
lstm_result = lstm_model.predict(market_data)
# Returns: up=0.75, down=0.15, neutral=0.10

# Get GRU prediction
gru_result = gru_model.predict(market_data)
# Returns: up=0.65, down=0.25, neutral=0.10

# Combine with weighted average (LSTM=60%, GRU=40%)
ensemble_up = 0.75*0.6 + 0.65*0.4 = 0.71
ensemble_down = 0.15*0.6 + 0.25*0.4 = 0.19
ensemble_neutral = 0.10*0.6 + 0.10*0.4 = 0.10

# Result: UP with 71% confidence
return PredictionResult(
    direction="up",
    confidence=0.71,
    probabilities={"up": 0.71, "down": 0.19, "neutral": 0.10},
    metadata={
        "model": "Ensemble",
        "strategy": "weighted_average",
        "num_models": 2,
        "models": ["LSTM", "GRU"]
    }
)
```

**Step 5: Return to Workflow**
```python
# Josh's workflow receives your prediction
# Continues to sentiment → reflection → explanation
# Final result returned to user
```

### Integration Points

**With JOSH (Agents):**
- Josh's `PredictionAgent` uses your models
- Polymorphic interface: works with single model or ensemble
- Configuration-driven: Josh selects which models to use

**With SUA (Frontend/Backend):**
- Sua doesn't interact with your models directly
- Goes through Josh's workflow
- Your predictions appear in API response

**With BYEOL (Testing):**
- Byeol writes unit tests for each model
- Tests individual models and ensemble
- Validates prediction output format

**With TAE (Sentiment):**
- No direct interaction
- Both provide models to Josh's agents
- Similar architecture pattern (base class + ensemble)

---

## Testing Your Code

### Unit Tests (Written by Byeol)

**Test: LSTM prediction with real data**
```python
from models.prediction.lstm_model import LSTMModel
import pandas as pd

def test_lstm_fallback():
    # Test fallback when model not trained
    lstm = LSTMModel()  # No weights_path

    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112],
        'volume': [1000000] * 11
    })

    result = lstm.predict(data)

    # Should use momentum fallback
    assert result.direction == "up"  # Price trending up
    assert result.metadata["model"] == "LSTM_fallback"
    assert 0.0 <= result.confidence <= 1.0
```

**Test: Ensemble with multiple models**
```python
from models.prediction.ensemble import PredictionEnsemble
from models.prediction.lstm_model import LSTMModel
from models.prediction.gru_model import GRUModel

def test_ensemble_weighted_average():
    lstm = LSTMModel()
    gru = GRUModel()

    ensemble = PredictionEnsemble(
        models=[lstm, gru],
        strategy="weighted_average",
        weights={"LSTM": 0.7, "GRU": 0.3}
    )

    data = create_test_data()
    result = ensemble.predict(data)

    assert result.metadata["model"] == "Ensemble"
    assert result.metadata["num_models"] == 2
    assert result.direction in ["up", "down", "neutral"]
```

**Test: Model failure handling**
```python
class FailingModel:
    def predict(self, data):
        raise RuntimeError("Model failed!")

    def get_model_info(self):
        return {"name": "FailingModel"}

def test_ensemble_handles_failure():
    working_model = LSTMModel()
    failing_model = FailingModel()

    ensemble = PredictionEnsemble(
        models=[working_model, failing_model]
    )

    # Should still work with 1 model
    result = ensemble.predict(data)
    assert result is not None
    assert result.metadata["num_models"] == 1  # Only working model
```

### Manual Testing

**Test individual LSTM:**
```bash
cd /home/user/capstone/FreshStart
python -c "
from models.prediction.lstm_model import LSTMModel
import pandas as pd

lstm = LSTMModel()
data = pd.DataFrame({
    'close': [150 + i for i in range(50)],
    'volume': [1000000] * 50
})

result = lstm.predict(data)
print(f'Direction: {result.direction}')
print(f'Confidence: {result.confidence:.2%}')
print(f'Metadata: {result.metadata}')
"
```

**Test ensemble:**
```python
from models.prediction.lstm_model import LSTMModel
from models.prediction.gru_model import GRUModel
from models.prediction.ensemble import PredictionEnsemble

lstm = LSTMModel()
gru = GRUModel()

ensemble = PredictionEnsemble(
    models=[lstm, gru],
    strategy="weighted_average"
)

# Test with upward trend data
import pandas as pd
import numpy as np

dates = pd.date_range('2025-01-01', periods=50)
prices = 150 + np.cumsum(np.random.randn(50) * 0.5)  # Random walk

data = pd.DataFrame({
    'date': dates,
    'close': prices,
    'volume': np.random.randint(900000, 1100000, 50)
})

result = ensemble.predict(data)
print(f"\nEnsemble Prediction:")
print(f"Direction: {result.direction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Probabilities: {result.probabilities}")
print(f"Metadata: {result.metadata}")
```

---

## Common Questions

### Q1: Why do we need three different models?

**Diversity = Better ensemble performance**

- **LSTM:** Best at long-term sequential patterns
- **GRU:** Faster, good at short-term patterns
- **Gradient Boost:** Different algorithm, catches different patterns

**Example scenario:**
- **Steady uptrend:** LSTM excels (recognizes sustained pattern)
- **Sudden spike:** GRU reacts faster (shorter memory)
- **Feature interactions:** Gradient Boost captures (e.g., volume + price change)

**Together:** Cover more scenarios than any single model

---

### Q2: How accurate will these models be?

**Realistic expectations for stock prediction:**

- **Random guessing:** 33% accuracy (up/down/neutral)
- **Simple momentum:** 40-45% accuracy
- **Single ML model:** 50-55% accuracy
- **Ensemble:** 55-60% accuracy

**Important notes:**
- Stock markets are **partially random** (efficient market hypothesis)
- Perfect prediction is impossible
- Even 55% accuracy can be profitable (slight edge over random)
- Focus on **consistent performance** over time

**Your goal:** Beat simple baselines, learn ML techniques

---

### Q3: What if I don't have trained model weights?

**Your models have fallback logic:**

```python
if not self.is_trained or self.model is None:
    return self._momentum_prediction(data)  # Simple fallback
```

**Fallback strategies:**
- **LSTM:** 10-day momentum (2% threshold)
- **GRU:** 5-day momentum (1% threshold)
- **Gradient Boost:** 10-day trend (3% threshold)

**This means:**
- System works during development
- Can test integration before training
- Graceful degradation if weights missing

**When to train:**
- After getting Kaggle data
- Before final demo
- For accurate results

---

### Q4: How do I choose ensemble weights?

**Method 1: Equal weights (start here)**
```python
# All models contribute equally
weights = {"LSTM": 0.33, "GRU": 0.33, "GradientBoost": 0.34}
```

**Method 2: Based on validation accuracy**
```python
# After training, test each model
lstm_accuracy = 0.55
gru_accuracy = 0.52
gb_accuracy = 0.50

# Assign weights proportional to accuracy
total = lstm_accuracy + gru_accuracy + gb_accuracy
weights = {
    "LSTM": lstm_accuracy / total,  # 0.52
    "GRU": gru_accuracy / total,    # 0.33
    "GB": gb_accuracy / total       # 0.32
}
```

**Method 3: Grid search (advanced)**
```python
# Try different weight combinations
# Find combination with best validation performance
```

**Recommendation:** Start with equal weights, optimize later

---

### Q5: Why use DataFrame instead of numpy arrays?

**DataFrames are more convenient:**

```python
# With DataFrame
data['close'].pct_change()  # Easy
data['volume'].rolling(10).mean()  # Readable

# With numpy
np.diff(close_array) / close_array[:-1]  # Less clear
rolling_mean = np.convolve(volume, np.ones(10)/10)  # Complex
```

**Pandas provides:**
- Date indexing
- Named columns (no confusion about column order)
- Built-in functions (pct_change, rolling, etc.)
- Easy integration with data fetchers

---

### Q6: What's the difference between confidence and probability?

**Probability:** Individual class likelihood
```python
probabilities = {
    "up": 0.75,      # 75% chance of going up
    "down": 0.15,    # 15% chance of going down
    "neutral": 0.10  # 10% chance of staying neutral
}
```

**Confidence:** How sure the model is about its prediction
```python
# Confidence = probability of the predicted direction
direction = "up"
confidence = probabilities["up"]  # 0.75 (75% confident in "up")
```

**Example:**
```python
# High confidence
probabilities = {"up": 0.90, "down": 0.05, "neutral": 0.05}
confidence = 0.90  # Very sure it's going up

# Low confidence
probabilities = {"up": 0.40, "down": 0.35, "neutral": 0.25}
confidence = 0.40  # Not very sure (only slightly favors up)
```

---

### Q7: How do I debug prediction failures?

**Step 1: Check data**
```python
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"First few rows:\n{data.head()}")
```

**Step 2: Verify required columns**
```python
required = ['close', 'volume']
missing = [col for col in required if col not in data.columns]
if missing:
    print(f"Missing columns: {missing}")
```

**Step 3: Check data length**
```python
print(f"Data length: {len(data)}")
# LSTM needs 30 days, GB needs 20 days
```

**Step 4: Test fallback**
```python
# Force fallback mode
model = LSTMModel()  # Don't provide weights_path
result = model.predict(data)

if "fallback" in result.metadata.get("model", ""):
    print("Using fallback (expected if no weights)")
```

**Step 5: Check for NaN values**
```python
print(f"NaN values:\n{data.isnull().sum()}")
```

---

### Q8: Can I add more features to the models?

**Absolutely! Here's how:**

**For LSTM/GRU:**
```python
def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
    # Original 2 features
    returns = data['close'].pct_change().fillna(0).values[-30:]
    volume_norm = (data['volume'] / data['volume'].rolling(10).mean()).fillna(1).values[-30:]

    # Add new feature: RSI (Relative Strength Index)
    rsi = calculate_rsi(data['close']).values[-30:]

    # Combine: now [30 timesteps, 3 features]
    features = np.column_stack([returns, volume_norm, rsi])
    return features.reshape(1, 30, 3)  # Changed from (1, 30, 2)
```

**For Gradient Boost:**
```python
def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
    features = []
    features.append(data['close'].pct_change(1).iloc[-1])
    features.append(data['close'].pct_change(5).iloc[-1])
    features.append(data['close'].pct_change(10).iloc[-1])
    features.append(data['volume'].iloc[-1] / data['volume'].rolling(10).mean().iloc[-1])

    # Add new features
    features.append(calculate_rsi(data['close']).iloc[-1])  # RSI
    features.append(calculate_macd(data['close']))  # MACD
    features.append(data['high'].iloc[-1] - data['low'].iloc[-1])  # Daily range

    return np.array([features])  # Now 8 features instead of 5
```

**Note:** When adding features, retrain the models with new architecture

---

## Summary

### What You Built (TL;DR)

1. **Base class** defining the prediction model interface
2. **3 ML models:** LSTM, GRU, Gradient Boost
3. **Ensemble system** combining 2+ models
4. **Graceful fallbacks** when models aren't trained

### Key Achievements

✅ **Polymorphic design** - All models implement same interface
✅ **Multiple approaches** - Neural networks + tree-based model
✅ **Flexible ensemble** - Combine any 2+ models with different strategies
✅ **Error resilience** - Fallback predictions when models fail
✅ **Standardized output** - PredictionResult format for all models

### Your Integration Points

- **Provides to JOSH:** Prediction models via BasePredictionModel interface
- **Uses data from:** Price data fetcher (your responsibility)
- **Tested by BYEOL:** Unit tests for all models and ensemble
- **Configured by JOSH:** Model selection via WorkflowConfig

### Next Steps

1. **Get Kaggle data** - Download SP500 historical dataset
2. **Implement training** - Complete train_models.py
3. **Train models** - Generate weights for LSTM, GRU, GB
4. **Save weights** - Commit to saved_models/ directories
5. **Test ensemble** - Compare different weight combinations
6. **Document performance** - Record accuracy metrics

---

**Great work, Pam!** Your prediction models are the core ML engine of FreshStart. 🚀
