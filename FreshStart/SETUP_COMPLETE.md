# FreshStart Setup Complete

## What Was Created

This setup created minimal working code for **PAM** and **JOSH** team members.

### PAM's Components (Prediction Models & Price Data)

#### Data Fetching
- `data/fetchers/price_data.py` - Real-time price data using yfinance
  - `get_historical_data()` - Fetch OHLCV data
  - `get_fundamentals()` - Fetch financial metrics

#### Prediction Models
- `models/prediction/base_predictor.py` - Base class defining standard interface
  - `BasePredictionModel` abstract class
  - `PredictionResult` dataclass

- `models/prediction/lstm_model.py` - LSTM neural network
  - Loads pre-trained weights if available
  - Falls back to momentum-based prediction

- `models/prediction/gru_model.py` - GRU neural network
  - Alternative to LSTM
  - Simplified architecture

- `models/prediction/gradient_boost_model.py` - XGBoost classifier
  - Non-neural network approach
  - Feature engineering for price/volume data

- `models/prediction/ensemble.py` - Ensemble system
  - Combines 2+ models
  - Strategies: weighted_average, simple_average, voting
  - Configurable model weights

- `models/prediction/train_models.py` - Training script placeholder
  - To be implemented after Kaggle data download

### JOSH's Components (Agents & Orchestration)

#### Agents
- `agents/prediction_agent.py` - Uses PAM's models for predictions
  - Wraps prediction models
  - Generates narrative explanations

- `agents/sentiment_agent.py` - Analyzes news sentiment
  - Placeholder for TAE's sentiment models
  - Simple keyword-based fallback

- `agents/reflection_agent.py` - Quality validation
  - Validates prediction/sentiment alignment
  - Checks data freshness
  - Adjusts confidence based on issues

- `agents/explanation_agent.py` - Generates explanations
  - Template-based for now
  - Ready for OpenAI LLM integration

#### Orchestration
- `coordinator/config.py` - Configuration system
  - Runtime model selection
  - Ensemble strategy configuration
  - Loads from YAML or uses defaults

- `coordinator/workflow.py` - LangGraph workflow
  - Simplified linear workflow
  - Nodes: validate → fetch_data → predict → sentiment → reflect → explain
  - Handles errors gracefully

### Configuration Files
- `secrets.env` - API keys for all services
- `config.example.yaml` - Example configuration file
- `~/.kaggle/kaggle.json` - Kaggle credentials (configured)

## Quick Start

### Test the Workflow

```bash
cd /home/user/capstone/FreshStart

# Test the workflow with a stock symbol
python -m coordinator.workflow
```

This will analyze AAPL stock using:
1. Real yfinance data
2. LSTM model (momentum fallback)
3. Sentiment analysis (placeholder)
4. Quality validation
5. Explanation generation

### Customize Configuration

1. Copy example config:
```bash
cp config.example.yaml config.yaml
```

2. Edit config.yaml to enable multiple models:
```yaml
prediction_models:
  - LSTM
  - GRU
  - GradientBoost

use_ensemble: true
ensemble_strategy: weighted_average
```

3. Run with custom config:
```python
from coordinator.workflow import run_stock_analysis
from coordinator.config import WorkflowConfig

config = WorkflowConfig.from_yaml("config.yaml")
result = run_stock_analysis("AAPL", config=config)
```

## Next Steps

### For PAM:
1. **Download Kaggle SP500 data**:
   ```bash
   pip install kaggle
   # Credentials already configured in ~/.kaggle/kaggle.json
   ```

2. **Implement training script** in `models/prediction/train_models.py`:
   - Load Kaggle data
   - Train LSTM, GRU, Gradient Boost models
   - Save weights to `models/prediction/saved_models/`

3. **Test individual models**:
   ```python
   from models.prediction.lstm_model import LSTMModel
   import pandas as pd

   model = LSTMModel()
   # Test with real data
   ```

### For JOSH:
1. **Test agents individually**:
   ```python
   from agents.prediction_agent import PredictionAgent
   from data.fetchers.price_data import get_historical_data, get_fundamentals

   agent = PredictionAgent()
   data = get_historical_data("AAPL")
   fundamentals = get_fundamentals("AAPL")
   result = agent.run("AAPL", data, fundamentals)
   print(result.narrative)
   ```

2. **Integrate with TAE's sentiment models** when ready:
   - Update `agents/sentiment_agent.py` to use real models
   - Replace keyword-based fallback

3. **Add OpenAI LLM** to ExplanationAgent:
   ```python
   from langchain_openai import ChatOpenAI

   llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
   explanation_agent = ExplanationAgent(llm=llm)
   ```

4. **Extend workflow** as needed:
   - Add smart money agent (premium features)
   - Add parallel execution for multiple tickers
   - Add caching with BYEOL's database

## Dependencies to Install

```bash
# Core dependencies
pip install yfinance pandas numpy tensorflow xgboost

# LangChain/LangGraph
pip install langchain langchain-openai langgraph

# Configuration
pip install pyyaml

# Optional: Kaggle data
pip install kaggle
```

## Testing

```bash
# Test price data fetcher
python -c "from data.fetchers.price_data import get_historical_data; print(len(get_historical_data('AAPL')))"

# Test LSTM model
python -c "from models.prediction.lstm_model import LSTMModel; m = LSTMModel(); print(m.get_model_info())"

# Test ensemble
python -c "from models.prediction.ensemble import PredictionEnsemble; from models.prediction.lstm_model import LSTMModel; from models.prediction.gru_model import GRUModel; e = PredictionEnsemble([LSTMModel(), GRUModel()]); print(e.get_model_info())"

# Test full workflow
python -m coordinator.workflow
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FreshStart Workflow                      │
└─────────────────────────────────────────────────────────────┘

Input: ticker, user_tier
  │
  ├─> validate_input
  │
  ├─> fetch_data (PAM's price_data.py)
  │     ├─> get_historical_data()
  │     └─> get_fundamentals()
  │
  ├─> predict (JOSH's PredictionAgent → PAM's Models)
  │     ├─> LSTM / GRU / GradientBoost
  │     └─> Optional: Ensemble
  │
  ├─> sentiment (JOSH's SentimentAgent → TAE's Models)
  │     └─> Placeholder: keyword-based
  │
  ├─> reflect (JOSH's ReflectionAgent)
  │     ├─> Validate prediction
  │     ├─> Check alignment
  │     └─> Assess data freshness
  │
  └─> explain (JOSH's ExplanationAgent)
        └─> Generate narrative

Output: {prediction, sentiment, explanation, confidence, warnings}
```

## Status

✅ **Complete**: Basic structure and minimal working code
⚠️  **Pending**: Model training (needs Kaggle data)
⚠️  **Pending**: TAE's sentiment models integration
⚠️  **Pending**: SUA's API and frontend
⚠️  **Pending**: BYEOL's database and tests

## File Structure

```
FreshStart/
├── secrets.env (API keys)
├── config.example.yaml (example configuration)
├── data/
│   └── fetchers/
│       └── price_data.py (PAM - yfinance wrapper)
├── models/
│   └── prediction/ (PAM - all prediction models)
│       ├── base_predictor.py
│       ├── lstm_model.py
│       ├── gru_model.py
│       ├── gradient_boost_model.py
│       ├── ensemble.py
│       └── train_models.py
├── agents/ (JOSH - all agents)
│   ├── prediction_agent.py
│   ├── sentiment_agent.py
│   ├── reflection_agent.py
│   └── explanation_agent.py
└── coordinator/ (JOSH - orchestration)
    ├── config.py
    └── workflow.py
```

---

**Last Updated**: 2025-11-05
**Created By**: Claude (AI Assistant)
**For**: PAM and JOSH (FreshStart Capstone Team)
