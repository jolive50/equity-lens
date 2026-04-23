# Model Training Guide

This guide explains how to train the Equity Lens prediction models using best practices.

## Overview

The training scripts have been split into individual files for each model, following ML engineering best practices:

```
models/prediction/
├── train_lstm.py          # LSTM model training
├── train_gru.py           # GRU model training
├── train_xgboost.py       # XGBoost model training
├── train_all_models.py    # Master script to train all models
└── TRAINING_GUIDE.md      # This file
```

## Benefits of This Structure

✅ **Memory Efficiency** - Train one model at a time with proper cleanup
✅ **Better Organization** - Each model has its own configuration
✅ **Easier Debugging** - Isolate issues to specific models
✅ **Flexibility** - Train only the models you need
✅ **Proper Checkpointing** - Save best models with hyperparameters
✅ **Early Stopping** - Prevent overfitting automatically
✅ **Comprehensive Metrics** - Financial metrics + ML metrics

## Quick Start

### Train All Models

```bash
# Train all three models sequentially
python -m models.prediction.train_all_models
```

### Train Individual Models

```bash
# Train only LSTM
python -m models.prediction.train_lstm

# Train only GRU
python -m models.prediction.train_gru

# Train only XGBoost
python -m models.prediction.train_xgboost
```

### Train Specific Models

```bash
# Train only LSTM and GRU
python -m models.prediction.train_all_models --models lstm gru

# Train only XGBoost (skip grid search for faster training)
python -m models.prediction.train_xgboost --no-gridsearch
```

## Features

### 1. Automatic Best Model Checkpointing

Each training script automatically saves:
- **Best model** (based on validation accuracy)
- **Training history** (CSV format)
- **Hyperparameters** (JSON format)
- **Training metrics** (accuracy, loss, confusion matrix)
- **Financial metrics** (expected value, win rate, payoff ratio)

### 2. Early Stopping

Prevents overfitting by:
- Monitoring validation loss
- Stopping after 5 epochs without improvement
- Restoring best weights automatically

### 3. Learning Rate Scheduling

LSTM and GRU use Cosine Decay with Restarts:
- Starts high for fast initial learning
- Gradually decreases for fine-tuning
- Periodic restarts to escape local minima

### 4. Memory Management

The master script (`train_all_models.py`) includes:
- Keras session cleanup between models
- Garbage collection forcing
- Error handling to prevent crashes

### 5. Hyperparameter Tuning

**LSTM/GRU:** Default hyperparameters optimized from research
- 3-layer architecture (128 → 64 → 32 units)
- 30% dropout for regularization
- Batch normalization between layers

**XGBoost:** Optional GridSearchCV
- 3-fold cross-validation
- Tests 200+ parameter combinations
- Automatically selects best parameters

## Output Files

After training, each model saves:

```
models/prediction/saved_models/
├── lstm/
│   ├── best_model.keras           # Best LSTM model
│   ├── training_history.csv       # Training metrics per epoch
│   └── training_results.json      # Full results + hyperparameters
├── gru/
│   ├── best_model.keras           # Best GRU model
│   ├── training_history.csv       # Training metrics per epoch
│   └── training_results.json      # Full results + hyperparameters
└── xgboost/
    ├── best_model.json            # Best XGBoost model (JSON format)
    ├── best_model.pkl             # Best XGBoost model (Pickle format)
    └── training_results.json      # Full results + hyperparameters
```

Combined report:
```
models/prediction/training_report.txt  # Comparison of all models
```

## Training Results Structure

Each `training_results.json` contains:

```json
{
  "hyperparameters": {
    "units_layer1": 128,
    "dropout_rate": 0.3,
    ...
  },
  "training_time_hours": 1.5,
  "epochs_completed": 23,
  "train_accuracy": 0.4123,
  "val_accuracy": 0.4067,
  "classification_report": {...},
  "confusion_matrix": [[...], [...], [...]],
  "financial_metrics": {
    "expected_value_per_trade": 0.0234,
    "win_rate": 48.52,
    "payoff_asymmetry": 1.23,
    "total_trades": 43521
  }
}
```

## Financial Metrics Explained

These metrics evaluate profitability of a simulated trading strategy:

- **Expected Value/Trade**: Average profit per trade (%)
  - Positive = profitable strategy
  - Negative = losing strategy

- **Win Rate**: Percentage of profitable trades
  - Target: >50% for simple strategies

- **Payoff Asymmetry**: Avg profit / Avg loss ratio
  - >1.0 = winners bigger than losers (good!)
  - <1.0 = losers bigger than winners (bad)

- **Total Trades**: Number of trades executed
  - More trades = more statistical confidence

## Customizing Hyperparameters

### LSTM/GRU

Edit hyperparameters in the training call:

```python
from models.prediction.train_lstm import train_lstm_model

hyperparameters = {
    'units_layer1': 256,      # Increase model capacity
    'units_layer2': 128,
    'units_layer3': 64,
    'dropout_rate': 0.4,      # More regularization
    'learning_rate': 0.0005,  # Slower learning
    'batch_size': 32,         # Smaller batches
    'max_epochs': 200,        # More training
    'early_stopping_patience': 10
}

results = train_lstm_model(
    X_train, y_train,
    X_val, y_val,
    hyperparameters=hyperparameters
)
```

### XGBoost

Edit the parameter grid in `train_xgboost.py`:

```python
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Add more options
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    ...
}
```

Or provide custom hyperparameters without grid search:

```python
hyperparameters = {
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    ...
}

results = train_xgboost_model(
    X_train, y_train,
    X_val, y_val,
    use_gridsearch=False,
    hyperparameters=hyperparameters
)
```

## Troubleshooting

### Out of Memory (OOM)

If training gets killed:

1. **Train models individually** (not all at once)
2. **Reduce batch size** in hyperparameters
3. **Reduce dataset size** in preprocessing
4. **Close other programs** to free RAM

### Training Too Slow

1. **Skip XGBoost grid search**: Use `--no-gridsearch`
2. **Reduce max_epochs**: Set lower in hyperparameters
3. **Use GPU**: Ensure CUDA is available for Keras models
4. **Reduce dataset**: Process fewer stocks in preprocessing

### Models Not Improving

1. **Check data quality**: Ensure preprocessing completed correctly
2. **Adjust learning rate**: Try lower/higher values
3. **Increase model capacity**: More units in layers
4. **Decrease regularization**: Lower dropout rate
5. **More epochs**: Increase max_epochs

## Best Practices Checklist

✅ Always train on preprocessed data first: `python -m data.preprocess_data`
✅ Monitor validation accuracy (not just training accuracy)
✅ Save best models (done automatically)
✅ Use early stopping to prevent overfitting
✅ Evaluate on test set after training (not during)
✅ Compare multiple models before deployment
✅ Check financial metrics (not just accuracy)
✅ Version control hyperparameters and results

## Next Steps

After training:

1. **Review training report**: `models/prediction/training_report.txt`
2. **Test on test set**: Create test script
3. **Compare models**: Choose best based on financial metrics
4. **Deploy best model**: Use in coordinator workflow
5. **Monitor performance**: Track real predictions

## Example Workflow

```bash
# 1. Preprocess data
python -m data.preprocess_data

# 2. Train all models (takes several hours)
python -m models.prediction.train_all_models

# 3. Review results
cat models/prediction/training_report.txt

# 4. If needed, retrain specific model with custom params
python -m models.prediction.train_lstm  # with custom hyperparameters

# 5. Test on test set
python -m models.prediction.test_models  # (create this script)

# 6. Deploy best model
python -m coordinator.workflow
```

## Support

For issues or questions:
- Check error messages and tracebacks
- Review this guide
- Check training logs in saved_models directories
- Verify data preprocessing completed successfully
