# FreshStart MVP - Research-Enhanced Training Guide

## Overview

This guide documents the implementation of research-backed enhancements for stock price prediction models based on the paper **"Predicting Next-Day S&P 500 Stock Movements: Best ML/DL Models"**.

**Research Source**: `docs/research/Predicting Next-Day S&P 500 Stock Movements_ Best ML_DL Models.pdf`

## Training Time Budget

**Total: 11-13 hours** across all models, GridSearch, and hyperparameter tuning

**Allocation**:
- XGBoost + GridSearchCV: ~4.5 hours (highest priority per research)
- LSTM (Enhanced): ~3.5 hours
- GRU (Enhanced): ~2.5 hours
- Data Preprocessing: ~30 minutes

## Research-Backed Enhancements Implemented

### 1. Feature Engineering (Research-Critical)

**Research Finding**: Feature quality matters more than model complexity. Well-engineered features with XGBoost match or beat deep neural networks.

**Implemented Features** (based on research recommendations):

#### Lagged Returns (Multi-Scale Momentum)
- `returns_1d`: 1-day return
- `returns_5d`: 5-day return
- `returns_10d`: 10-day return
- `returns_20d`: 20-day return
- `returns_60d`: 60-day return
- `returns_252d`: Annual return

**Research Citation**: "Form feature vectors of 13 lagged relative returns (1-day, 5-day, 10-day, ... up to 1-year)"

#### Technical Indicators
**RSI (Relative Strength Index)**
- 14-day RSI standard
- Research shows RSI is top-3 most important feature

**MACD (Moving Average Convergence Divergence)**
- MACD line, signal line, and MACD diff
- Critical for detecting momentum shifts

**Bollinger Bands**
- Band width (volatility measure)
- Band position (price position relative to bands)
- Research: "Bollinger Bands capture volatility regime changes"

#### Extended Moving Averages
- SMA 5, 10, 20, 50, 200-day
- 5/20 crossover signal
- 50/200 crossover signal (Golden/Death Cross)

**Research Citation**: "50 and 200-day moving averages are critical for long-term trend detection"

#### Volatility Measures
- 10-day, 20-day, 60-day rolling volatility
- Research: "Volatility regime can change model behavior; excluding high-volatility stocks from trading can hurt Sharpe ratio despite prediction accuracy"

#### Volume Features
- Volume ratio (current vs 10-day MA)
- Volume trend (10-day vs 20-day MA)

**Files Modified**:
- `data/preprocess_data.py`: `engineer_features()` method
- `models/prediction/lstm_model.py`: `_prepare_features()` method
- `models/prediction/gru_model.py`: `_prepare_features()` method
- `models/prediction/gradient_boost_model.py`: `_extract_features()` method

---

### 2. Sequence Length: 30 → 60 Days

**Research Finding**: "60-day lookback window is optimal for capturing both short-term patterns and longer-term trends"

**Implementation**:
- Changed default `sequence_length` from 30 to 60 in `preprocess_data.py`
- LSTM/GRU models now use 60-timestep sequences
- Better temporal pattern capture

**Research Citation**: "A typical preprocessing step is to create a fixed-length window of recent daily returns as the input sequence (for example, the past 60 trading days)"

**Files Modified**:
- `data/preprocess_data.py`: `create_sequences()` default changed to 60
- `models/prediction/lstm_model.py`: Uses 60-day sequences
- `models/prediction/gru_model.py`: Uses 60-day sequences

---

### 3. Enhanced LSTM Architecture

**Research Finding**: "LSTM achieved ~52-53% directional accuracy and Sharpe ~5.8, significantly outperforming memory-free models"

**Baseline Architecture** (Original):
```
LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(3)
```

**Enhanced Architecture** (Research-Based):
```
LSTM(128) → BatchNorm → Dropout(0.3) →
LSTM(64) → BatchNorm → Dropout(0.3) →
LSTM(32) → BatchNorm → Dropout(0.2) →
Dense(16) → Dropout(0.2) → Dense(3)
```

**Key Improvements**:
1. **Deeper network**: 3 LSTM layers instead of 2
2. **More units**: 128 → 64 → 32 (captures more complex patterns)
3. **Batch Normalization**: Training stability
4. **Learning rate schedule**: Exponential decay for better convergence
5. **Early stopping**: Patience=5, prevents overfitting
6. **Model checkpointing**: Saves best model based on val_accuracy
7. **ReduceLROnPlateau**: Adaptive learning rate

**Research Citation**: "A 3-layer LSTM with a few hundred units can be trained on years of daily data in minutes to hours – within the 12h constraint"

**Training Time**: ~3.5 hours (with early stopping)

**Files Modified**:
- `models/prediction/train_models.py`: `train_lstm_model()` function

---

### 4. Enhanced GRU Architecture

**Research Finding**: "GRU is faster than LSTM while maintaining similar performance, effective at capturing recent trends"

**Enhanced Architecture** (Same as LSTM but with GRU layers):
```
GRU(128) → BatchNorm → Dropout(0.3) →
GRU(64) → BatchNorm → Dropout(0.3) →
GRU(32) → BatchNorm → Dropout(0.2) →
Dense(16) → Dropout(0.2) → Dense(3)
```

**Advantages over LSTM**:
- Faster training (fewer parameters)
- Good for capturing recent trends
- Lower memory footprint

**Training Time**: ~2.5 hours (faster than LSTM)

**Files Modified**:
- `models/prediction/train_models.py`: `train_gru_model()` function

---

### 5. XGBoost with GridSearchCV

**Research Finding**: "XGBoost is a top performer (51-52% accuracy baseline, up to 60-65% with proper hyperparameter tuning)"

**Baseline**: Fixed hyperparameters
**Enhanced**: Comprehensive GridSearchCV

**Hyperparameter Search Space**:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}
```

**Total Combinations**: 3 × 4 × 3 × 3 × 3 × 3 × 3 = **2,916 combinations** with 3-fold CV

**Features**:
- 3-fold cross-validation
- Parallel processing (`n_jobs=-1`)
- `tree_method='hist'` for faster training
- Feature importance analysis (top 15 features logged)

**Research Citation**: "XGBoost with proper hyperparameter tuning can achieve 60-65% accuracy"

**Training Time**: ~4.5 hours (with GridSearch) or ~1 hour (without GridSearch)

**Files Modified**:
- `models/prediction/train_models.py`: `train_gradient_boost_model()` function

---

## Expected Performance Benchmarks

Based on research findings and academic benchmarks:

### LSTM
- **Directional Accuracy**: 52-53% (research benchmark)
- **Sharpe Ratio**: ~5.8 (before transaction costs)
- **Daily Return**: ~0.46% (research: Fischer & Krauss 2018)

### XGBoost (with GridSearch)
- **Accuracy**: 60-65% (with proper features + tuning)
- **Baseline**: 51-52% (without tuning)

### Ensemble (All 3 Models)
- **Expected**: ~53-55% accuracy
- **Research**: "Ensemble of RF, boosted trees, and neural net outperformed any single model"

### Key Metrics to Track
1. **Accuracy** (per class: down/neutral/up)
2. **Precision** (minimize false positives)
3. **Recall** (capture true signals)
4. **F1-Score** (balanced metric)
5. **Classification Report** (per-class breakdown)

---

## Training Pipeline

### Step 1: Download Data
```bash
python -m data.download_kaggle_data
```

**Note**: Requires Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Step 2: Preprocess Data
```bash
python -m data.preprocess_data
```

**Output**: `data/processed/training_splits.npz`

**Features Generated**:
- LSTM/GRU: 25 features × 60 timesteps
- XGBoost: 28 features (tabular)

### Step 3: Train All Models
```bash
python -m models.prediction.train_models
```

**This will**:
1. Train LSTM (3.5 hours)
2. Train GRU (2.5 hours)
3. Train XGBoost with GridSearch (4.5 hours)
4. Generate `training_report.txt`

**Total Time**: ~11 hours

---

## Model Files Structure

```
models/prediction/saved_models/
├── lstm/
│   ├── lstm_model.keras           # Final trained model
│   └── checkpoint.keras            # Best checkpoint
├── gru/
│   ├── gru_model.keras
│   └── checkpoint.keras
└── gradient_boost/
    ├── gb_model.json               # XGBoost native format
    └── gb_model.pkl                # Pickle format (compatibility)
```

---

## Research Citations

### Primary Research Paper
- **Title**: "Predicting Next-Day S&P 500 Stock Movements: Best ML/DL Models"
- **Location**: `docs/research/Predicting Next-Day S&P 500 Stock Movements_ Best ML_DL Models.pdf`

### Key Studies Referenced
1. **Fischer & Krauss (2018)**: LSTM on S&P 500 (1992-2015), 52-53% accuracy, Sharpe ~5.8
2. **Htun et al. (2024)**: LSTM vs RF vs SVM on S&P 500 (2017-2022), LSTM best performer
3. **FinBERT Study (2024)**: FinBERT sentiment improved AUC by 12.6%, P&L by 26%
4. **Kaggle Two Sigma (2019)**: ExtraTrees ensemble, focus on risk control

---

## Troubleshooting

### Issue: "Not enough data"
**Solution**: Ensure you have at least 260 days of data (60 sequence + 200 for SMA_200)

### Issue: GridSearch taking too long
**Solution**: Set `use_gridsearch=False` in `train_gradient_boost_model()`

### Issue: Out of memory
**Solution**: Reduce `batch_size` in LSTM/GRU training or reduce `max_stocks` in preprocessing

### Issue: Poor accuracy
**Check**:
1. Data quality (sufficient history?)
2. Feature engineering (all features calculated?)
3. Hyperparameters (GridSearch completed?)
4. Class imbalance (check target distribution)

---

## Next Steps After Training

1. **Evaluate on test set**: Load models and evaluate on held-out test data
2. **Backtest**: Simulate trading strategy with transaction costs
3. **Monitor performance**: Track accuracy degradation over time (regime changes)
4. **Retrain periodically**: Models need updates as market patterns change

---

## Contact

For questions about this implementation:
- Check `docs/team/PAM_CODE_GUIDE.md` for detailed explanations
- See `ROLE_DIVISION.md` for team responsibilities

---

**Last Updated**: 2025-11-12
**Implemented By**: Research-enhanced pipeline per PDF recommendations
**Training Budget**: 11-13 hours total
