# Trained Model Weights

This directory contains trained weights for all prediction models.

## Directory Structure

```
saved_models/
├── lstm/
│   └── lstm_model.keras          # LSTM model weights
├── gru/
│   └── gru_model.keras            # GRU model weights
└── gradient_boost/
    ├── gb_model.json              # XGBoost model (JSON format)
    └── gb_model.pkl               # XGBoost model (pickle format)
```

## Training Models

To train and generate model weights:

```bash
# 1. Download Kaggle data
python -m data.download_kaggle_data

# 2. Preprocess data
python -m data.preprocess_data

# 3. Train all models
python -m models.prediction.train_models
```

## Using Trained Models

```python
from models.prediction.lstm_model import LSTMModel
from models.prediction.gru_model import GRUModel
from models.prediction.gradient_boost_model import GradientBoostModel

# Load trained models
lstm = LSTMModel("models/prediction/saved_models/lstm/lstm_model.keras")
gru = GRUModel("models/prediction/saved_models/gru/gru_model.keras")
gb = GradientBoostModel("models/prediction/saved_models/gradient_boost/gb_model.pkl")

# Make predictions
result = lstm.predict(data)
```

## Model Performance

After training, check `training_report.txt` in the parent directory for performance metrics.

Typical validation accuracies:
- LSTM: 55-62%
- GRU: 54-60%
- Gradient Boost: 60-68%

## Git & Model Weights

**Note**: Model weight files (.keras, .pkl, .json) should be committed to git so team members can use them without retraining.

Current `.gitignore` may exclude large model files. If models don't appear in git:
1. Check `.gitignore` for patterns excluding model files
2. Use `git add -f` to force-add specific model files if needed
3. Or configure Git LFS for large files

## File Sizes

Expected file sizes after training:
- LSTM: ~2-5 MB
- GRU: ~1.5-4 MB
- Gradient Boost: ~500KB-2MB

Total: ~8-12 MB for all models
