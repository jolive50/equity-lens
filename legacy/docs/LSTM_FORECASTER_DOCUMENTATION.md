# LSTM Forecaster Documentation

## Overview

The LSTM (Long Short-Term Memory) Forecaster is a deep learning model for stock price direction prediction. It predicts whether a stock will go **up**, **down**, or remain **steady** with calibrated probability estimates.

### Key Features

- **LSTM Architecture**: Two-layer LSTM (128→64 units) with dropout regularization
- **Three-Class Classification**: Predicts up, down, or steady price movements
- **Probability Calibration**: Ensures confidence scores match actual accuracy (e.g., 70% confidence = 70% actual success rate)
- **Technical Indicators**: Uses 16 engineered features including RSI, MACD, Bollinger Bands
- **Sequence-Based Learning**: Learns from 30-day historical patterns

## Architecture

### Model Structure

```
Input: (batch_size, 30 timesteps, 16 features)
    ↓
LSTM Layer 1: 128 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 64 units, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Output: 3 units (up/down/steady), softmax activation
    ↓
Output: (batch_size, 3 probabilities)
```

### Input Features (16 total)

1. **Price Data**: open, high, low, close, volume
2. **Returns**: Percentage change in closing price
3. **Technical Indicators**:
   - RSI (Relative Strength Index)
   - SMA 10-day and 20-day
   - MACD and MACD signal
   - Bollinger Bands (upper, middle, lower)
   - Volume ratio
   - Volatility

### Output Classes

#### ATR-Based Dynamic Thresholds (Default) ⭐

The model uses **Average True Range (ATR)** to create stock-specific, volatility-adjusted thresholds:

- **Class 0 (Down)**: Price decreases > (ATR × 0.3)
- **Class 1 (Steady)**: Price changes within ±(ATR × 0.3)
- **Class 2 (Up)**: Price increases > (ATR × 0.3)

**Why ATR-based?**
- Fixed threshold results in 90% of samples being "steady" (severe class imbalance)
- ATR adapts to each stock's volatility (volatile stocks need larger moves to classify as up/down)
- **Result**: Much more balanced classes (≈29% down, 39% steady, 32% up)

**Example**:
- Low volatility stock (ATR = 1%): Threshold = 0.3%
- High volatility stock (ATR = 3%): Threshold = 0.9%

#### Fixed Threshold (Optional)

Can be enabled by setting `use_atr_threshold = False`:
- **Class 0 (Down)**: Price decreases > 2%
- **Class 1 (Steady)**: Price changes between -2% and +2%
- **Class 2 (Up)**: Price increases > 2%

## Training Process

### Data Pipeline

1. **Data Loading**: Load all CSV files from Kaggle S&P 500 dataset
2. **Indicator Calculation**: Compute technical indicators for each stock
3. **ATR Calculation**: Compute Average True Range for volatility-adjusted thresholds
4. **Label Creation**: Classify next-day returns into up/down/steady using ATR-based thresholds
5. **Sequence Creation**: Create 30-day rolling windows
6. **Train/Val/Cal Split**: 70% train, 20% validation, 10% calibration
7. **Normalization**: StandardScaler on feature sequences
8. **Training**: 50 epochs with early stopping
9. **Calibration**: Isotonic regression for probability calibration

### ATR-Based Labeling

**What is ATR?**
- Measures a stock's volatility
- Calculated as 14-day rolling average of True Range
- True Range = max(high-low, |high-prev_close|, |low-prev_close|)

**How it works:**
1. Calculate ATR for each stock
2. Convert to percentage: `ATR_pct = ATR / close_price`
3. Set dynamic threshold: `threshold = ATR_pct × 0.3`
4. Classify: up if return > threshold, down if return < -threshold, else steady

**Example:**
```
Stock A (low volatility):
  ATR = $1.50, Price = $100
  ATR% = 1.5%
  Threshold = 1.5% × 0.3 = 0.45%

Stock B (high volatility):
  ATR = $4.50, Price = $150
  ATR% = 3.0%
  Threshold = 3.0% × 0.3 = 0.9%
```

### Training Configuration

```python
# Architecture
sequence_length = 30          # Days of history
lstm_layer1_units = 128       # First LSTM layer
lstm_layer2_units = 64        # Second LSTM layer
dropout_rate = 0.2            # Dropout for regularization

# Training
batch_size = 64
epochs = 50
validation_split = 0.2        # 20% for validation
calibration_split = 0.1       # 10% for calibration
learning_rate = 0.001

# Classification
num_classes = 3               # up, down, steady
price_change_threshold = 0.02 # ±2% for classification (if not using ATR)

# ATR-based dynamic thresholding (recommended)
use_atr_threshold = True      # Use ATR-based threshold
atr_window = 14               # ATR calculation window (standard)
atr_multiplier = 0.3          # Threshold = ATR * multiplier
```

### Running Training

```bash
# Train the LSTM model
python pipelines/realtime/models/train_lstm_forecaster.py
```

**Output files** (saved to `pipelines/realtime/models/saved_models/`):
- `lstm_forecaster.h5` - Trained Keras model
- `scaler.pkl` - Feature normalization scaler
- `calibrator.pkl` - Probability calibrator
- `feature_columns.pkl` - Feature names and order
- `config.pkl` - Training configuration

## Using the Forecaster

### Basic Usage

```python
from pipelines.realtime.models.forecaster import ProbabilisticForecaster

# Initialize forecaster (automatically loads pre-trained LSTM)
forecaster = ProbabilisticForecaster(model_type="lstm")

# Prepare market data (need 30+ days)
market_data = [
    {
        "date": "2025-01-01",
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 151.0,
        "volume": 1000000
    },
    # ... more days (need at least 30)
]

# Make prediction
result = forecaster.predict(market_data, fundamentals={})

# Access results
print(f"Direction: {result.direction}")           # "up", "down", or "neutral"
print(f"Confidence: {result.confidence:.1%}")     # e.g., 75.3%
print(f"Probabilities:")
print(f"  Up: {result.model_metadata['probabilities']['up']:.1%}")
print(f"  Down: {result.model_metadata['probabilities']['down']:.1%}")
print(f"  Neutral: {result.model_metadata['probabilities']['neutral']:.1%}")
```

### Custom Model Directory

```python
# Load model from custom location
forecaster = ProbabilisticForecaster(
    model_type="lstm",
    model_dir="/path/to/saved_models"
)
```

### Fallback to Traditional Models

```python
# Use gradient boosting instead (no pre-training required)
forecaster = ProbabilisticForecaster(model_type="gradient_boosting")

# Use random forest
forecaster = ProbabilisticForecaster(model_type="random_forest")
```

## Probability Calibration

### Why Calibration?

Raw neural network outputs are often poorly calibrated. A model might output 90% confidence but only be correct 70% of the time. Calibration fixes this.

### How It Works

1. **Training**: Model trained on 70% of data
2. **Validation**: 20% used for early stopping
3. **Calibration**: 10% held out specifically for calibration
4. **Calibration Curve**: Learn mapping from predicted → actual frequency
5. **Isotonic Regression**: Non-parametric calibration method

### Calibration Goal

**Target**: 70% confidence = 70% actual accuracy

**Implementation**:
- Bin predictions by confidence level
- Calculate actual accuracy in each bin
- Learn transformation to match predicted → actual
- Apply transformation to all future predictions

### Verifying Calibration

```python
# After training, check calibration metrics
# The calibrator stores calibration curves for each class
calibrator = forecaster.lstm_calibrator

for class_idx in range(3):
    cal_map = calibrator.calibration_maps[class_idx]
    print(f"Class {class_idx} calibration:")
    print(f"  Predicted: {cal_map['pred_mean']}")
    print(f"  Actual: {cal_map['true_freq']}")
```

## Model Performance

### Expected Metrics

- **Accuracy**: 50-65% (3-class problem, 33% is random)
- **Log Loss**: < 1.0 (lower is better)
- **Calibration Error**: < 0.05 (5% maximum deviation)

### Evaluation

```python
from pipelines.realtime.models.train_lstm_forecaster import LSTMForecasterModel

# Load trained model
model = LSTMForecasterModel(config)
model.model = keras.models.load_model("saved_models/lstm_forecaster.h5")

# Evaluate on test data
results = model.evaluate(X_test, y_test)

print(f"Log Loss: {results['log_loss']:.3f}")
print("Predictions:", results['predictions'])
print("Probabilities:", results['probabilities'])
```

## Data Requirements

### Minimum Data

- **Training**: 500+ stocks with 1000+ days each (Kaggle S&P 500 dataset)
- **Inference**: 30+ days of OHLCV data for a single prediction

### Data Format

**CSV Files** (training):
```csv
date,open,high,low,close,volume,Name
2013-02-08,67.71,68.40,66.89,67.85,158168416,AAPL
2013-02-11,68.07,69.28,67.61,68.56,129029425,AAPL
...
```

**Dictionary List** (inference):
```python
[
    {
        "date": "2025-01-01",
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 151.0,
        "volume": 1000000
    },
    ...
]
```

## Error Handling

### Model Not Found

```python
try:
    forecaster = ProbabilisticForecaster(model_type="lstm")
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: LSTM model not found. Please train the model first by running:
    #   python pipelines/realtime/models/train_lstm_forecaster.py
```

### Insufficient Data

```python
# Less than 30 days of data
market_data = [...]  # Only 10 days

result = forecaster.predict(market_data, {})
# Falls back to trend-based prediction with warning
```

### Missing Columns

```python
# Missing required OHLCV columns
market_data = [{"close": 150.0}]  # Missing open, high, low, volume

result = forecaster.predict(market_data, {})
# Falls back to trend-based prediction with warning
```

## Advanced Usage

### Batch Predictions

```python
# Predict for multiple stocks
stocks = ["AAPL", "GOOGL", "MSFT"]
results = {}

for ticker in stocks:
    market_data = get_market_data(ticker)  # Your data fetching function
    results[ticker] = forecaster.predict(market_data, {})

# Compare predictions
for ticker, result in results.items():
    print(f"{ticker}: {result.direction} ({result.confidence:.1%})")
```

### Custom Threshold

To change the price movement threshold (default ±2%):

```python
# Modify TrainingConfig in train_lstm_forecaster.py
config = TrainingConfig()
config.price_change_threshold = 0.01  # ±1% threshold
```

Then retrain the model.

### Feature Importance

```python
result = forecaster.predict(market_data, {})

# Note: LSTM feature importance is complex
# Current implementation provides placeholder
# TODO: Implement SHAP or attention-based importance
print(result.feature_importance)
```

## Troubleshooting

### Issue: Model training is slow

**Solution**: Reduce dataset size or epochs
```python
config = TrainingConfig()
config.epochs = 25  # Reduce from 50
config.batch_size = 128  # Increase batch size
```

### Issue: Poor calibration (confidence doesn't match accuracy)

**Solution**: Increase calibration set size
```python
config.calibration_split = 0.15  # Increase from 0.1
```

### Issue: Model overfitting

**Solution**: Increase dropout or reduce model complexity
```python
config.dropout_rate = 0.3  # Increase from 0.2
config.lstm_layer1_units = 64  # Reduce from 128
config.lstm_layer2_units = 32  # Reduce from 64
```

### Issue: Memory errors during training

**Solution**: Reduce batch size or sequence length
```python
config.batch_size = 32  # Reduce from 64
config.sequence_length = 20  # Reduce from 30
```

## Testing

### Unit Tests

```bash
# Run all LSTM forecaster tests
pytest tests/unit/test_lstm_forecaster.py -v

# Run specific test
pytest tests/unit/test_lstm_forecaster.py::TestLSTMForecasterModel::test_build_model_architecture -v
```

### Test Coverage

Tests cover:
- ✅ Configuration initialization
- ✅ Data loading and preprocessing
- ✅ Technical indicator calculation
- ✅ Label creation (up/down/steady)
- ✅ Sequence generation
- ✅ Normalization
- ✅ Model architecture
- ✅ Training process
- ✅ Probability calibration
- ✅ Model loading
- ✅ Prediction pipeline
- ✅ Error handling

## Implementation Checklist

When integrating the LSTM forecaster:

- [ ] Train model using Kaggle S&P 500 data
- [ ] Verify model files saved to `saved_models/` directory
- [ ] Test model loading with `ProbabilisticForecaster(model_type="lstm")`
- [ ] Validate predictions have calibrated probabilities
- [ ] Run unit tests to ensure functionality
- [ ] Document any custom configurations
- [ ] Set up monitoring for prediction accuracy
- [ ] Implement retraining schedule (monthly/quarterly)

## References

### Papers & Resources

- **LSTM Networks**: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Probability Calibration**: Zadrozny & Elkan, "Transforming Classifier Scores into Accurate Multiclass Probability Estimates" (2002)
- **Technical Indicators**: Murphy, "Technical Analysis of the Financial Markets" (1999)

### Code Structure

```
pipelines/realtime/models/
├── train_lstm_forecaster.py    # Training script
├── forecaster.py                # ProbabilisticForecaster class
└── saved_models/                # Model artifacts
    ├── lstm_forecaster.h5
    ├── scaler.pkl
    ├── calibrator.pkl
    ├── feature_columns.pkl
    └── config.pkl

tests/unit/
└── test_lstm_forecaster.py      # Unit tests

docs/
└── LSTM_FORECASTER_DOCUMENTATION.md  # This file
```

## Changelog

### Version 1.0 (2025-10-07)
- Initial implementation
- Two-layer LSTM (128→64) architecture
- Probability calibration with isotonic regression
- 16 technical indicator features
- Three-class classification (up/down/steady)
- Integration with ProbabilisticForecaster
- Comprehensive test coverage

## Future Enhancements

### Planned Features

1. **Attention Mechanism**: Add attention layer to identify important time steps
2. **Multi-Horizon Prediction**: Predict 1-day, 5-day, 10-day ahead
3. **Uncertainty Quantification**: Bayesian LSTM for epistemic uncertainty
4. **Feature Importance**: SHAP values for LSTM explanations
5. **Online Learning**: Incremental updates without full retraining
6. **Ensemble Methods**: Combine LSTM with gradient boosting
7. **Multi-Asset Prediction**: Predict multiple correlated stocks jointly

### Research Directions

- **Transformers**: Replace LSTM with attention-based architecture
- **Graph Neural Networks**: Model stock relationships
- **Reinforcement Learning**: Optimize for trading returns, not just accuracy
- **Alternative Data**: Incorporate news sentiment, social media

## Support

For issues or questions:

1. Check error messages for troubleshooting steps
2. Review test cases for usage examples
3. Consult inline code documentation (WHAT/WHY/HOW/DATA comments)
4. Open issue on project repository

---

**Last Updated**: October 7, 2025
**Author**: StockSense Development Team
**Model Version**: 1.0
