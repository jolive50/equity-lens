# Josh's Code Guide - Agent Implementation

## Overview

This guide explains the FreshStart agent and coordinator implementation extracted from the legacy system. The implementation follows CLAUDE.md guidelines with lightweight, minimal code and comprehensive documentation here.

**Total Lines Implemented:** ~730 lines (61% reduction from legacy ~1,880 lines)

---

## What Was Implemented

### 1. Agents (`agents/` folder)

Four specialized agents for stock analysis:

1. **PredictionAgent** (`agents/prediction_agent.py`)
   - Uses Pam's LSTM model from `models/prediction/`
   - Generates directional forecasts (up/down/neutral)
   - Optional GPT explanations with SHAP feature importance
   - Returns `PredictionResult` with confidence scores

2. **SentimentAgent** (`agents/sentiment_agent.py`)
   - Uses Tae's FinBERT model from `models/sentiment/`
   - Analyzes news articles for financial sentiment
   - Integrates with Tae's ChromaDB VectorStore
   - Returns `SentimentResult` with sentiment scores and headlines

3. **ReflectionAgent** (`agents/reflection_agent.py`)
   - Pure rule-based validation (no external dependencies)
   - Checks prediction validity, sentiment alignment, data freshness
   - Adjusts confidence scores based on quality issues
   - Returns validation results with warnings and recommendations

4. **ExplanationAgent** (`agents/explanation_agent.py`)
   - Uses OpenAI LLM to generate plain English explanations
   - Synthesizes prediction, sentiment, and smart money data
   - Customizable prompt templates
   - Returns user-friendly narrative (3-4 paragraphs)

### 2. Coordinator (`coordinator/` folder)

Orchestration and configuration:

1. **workflow.py** - Simplified LangGraph workflow
   - Linear pipeline: validate → fetch_data → predict → sentiment → reflect → explain
   - Single-ticker analysis only (no multi-company coordination)
   - Error handling with graceful degradation
   - Exports `create_stocksense_workflow()` and `run_stocksense_analysis()`

2. **config.py** - Model configuration system
   - PyYAML-based configuration loader
   - Model selection (single or ensemble)
   - Ensemble strategies and weights
   - Validation helpers

### 3. Configuration (`config.yaml`)

Example configuration file showing:
- Single-model setup (LSTM + FinBERT)
- Ensemble configuration examples (for future expansion)
- Model weights and strategies
- Comments explaining all options

---

## How It Works

### Agent Data Flow

```
User Request
    ↓
Workflow (LangGraph)
    ↓
PredictionAgent → Uses Pam's LSTM → PredictionResult
    ↓
SentimentAgent → Uses Tae's FinBERT → SentimentResult
    ↓
ReflectionAgent → Validates outputs → Confidence adjustment
    ↓
ExplanationAgent → Uses OpenAI GPT → Plain English explanation
    ↓
Final Response
```

### Workflow Pipeline

The simplified workflow follows a linear pipeline:

1. **validate_input** - Validates ticker and sets defaults
2. **collect_data** - Validates presence of market data, news, fundamentals
3. **run_prediction** - Calls PredictionAgent with LSTM model
4. **run_sentiment** - Calls SentimentAgent with FinBERT
5. **run_reflection** - Validates outputs, adjusts confidence
6. **build_explanation** - Generates plain English explanation

### Key Simplifications from Legacy

**Removed:**
- Multi-company coordination (CoordinationAgent, HistoricalAnalysisAgent, SentimentAnalysisAgent)
- SmartMoneyAgent (not in MVP scope)
- Complex state management for multiple tickers
- SP500 data service batch processing
- Excessive WHAT/WHY/HOW comments

**Kept:**
- Core single-ticker analysis path
- ML-powered predictions (LSTM)
- FinBERT sentiment analysis
- Quality validation (ReflectionAgent)
- Plain English explanations (ExplanationAgent)

---

## Integration Points

### Josh → Pam (Prediction Models)

**Interface:** `BasePredictionModel` with `predict(data, fundamentals) -> PredictionResult`

**Usage:**
```python
from models.prediction.forecaster import create_forecaster

forecaster = create_forecaster("lstm")
result = forecaster.predict(market_data, fundamentals)
```

**What PredictionAgent expects:**
- `result.direction` - "up", "down", or "neutral"
- `result.confidence` - float between 0.0 and 1.0
- `result.feature_importance` - Optional SHAP scores
- `result.daily_probs` - Optional probability curve
- `result.horizon_95` - Optional 95% confidence horizon

### Josh → Tae (Sentiment Models)

**Interface:** `BaseSentimentModel` with `analyze(text) -> SentimentResult`

**Usage:**
```python
from models.sentiment.finbert import create_sentiment_analyzer

analyzer = create_sentiment_analyzer(
    vector_store=vector_store,
    enable_similarity_search=True
)
result = analyzer.process_news_articles(news_data, ticker=ticker)
```

**What SentimentAgent expects:**
- `result["current"]` - "positive", "neutral", or "negative"
- `result["score"]` - float between 0.0 and 1.0
- `result["trend"]` - "improving", "stable", or "declining"
- `result["headlines"]` - List of top 3 headlines

### Josh → Tae (ChromaDB VectorStore)

**Interface:** `VectorStore` with `search_similar(query, ticker, n_results)`

**Usage:**
```python
from storage.vector_store import VectorStore

vector_store = VectorStore()
sentiment_agent = SentimentAgent(vector_store=vector_store)
```

**What SentimentAgent does:**
- Passes `vector_store` to `create_sentiment_analyzer()`
- FinBERT stores articles in VectorStore automatically
- FinBERT uses similarity search for context augmentation

### Josh → Sua (API Integration)

**Interface:** `run_stocksense_analysis()` function

**Usage:**
```python
from coordinator import run_stocksense_analysis
from agents import (
    PredictionAgent, SentimentAgent,
    ReflectionAgent, ExplanationAgent, build_openai_llm
)

# Initialize agents
llm = build_openai_llm()
prediction_agent = PredictionAgent(llm=llm, use_gpt_explanations=True)
sentiment_agent = SentimentAgent(vector_store=vector_store)
reflection_agent = ReflectionAgent()
explanation_agent = ExplanationAgent(llm=llm)

# Run analysis
result = run_stocksense_analysis(
    ticker="AAPL",
    user_tier="premium",
    market_data=market_data,
    news_data=news_data,
    fundamentals=fundamentals,
    prediction_agent=prediction_agent,
    sentiment_agent=sentiment_agent,
    reflection_agent=reflection_agent,
    explanation_agent=explanation_agent
)
```

**Response format:**
```python
{
    "ticker": "AAPL",
    "direction": "up",
    "confidence": 0.85,
    "confidence_level": "high",
    "narrative": "AAPL forecast: UP with 85% confidence...",
    "sentiment": {
        "current": "positive",
        "score": 0.78,
        "trend": "improving",
        "headlines": [...]
    },
    "explanation": "Based on our analysis, AAPL shows strong...",
    "warnings": [],
    "daily_probs": [...],
    "horizon_95": {...}
}
```

### Josh → Byeol (Testing)

**Test files to create:**
- `tests/unit/test_prediction_agent.py` - Test prediction with mock forecaster
- `tests/unit/test_sentiment_agent.py` - Test sentiment with mock analyzer
- `tests/unit/test_reflection_agent.py` - Test validation logic
- `tests/unit/test_explanation_agent.py` - Test LLM chain with mock LLM
- `tests/integration/test_workflow.py` - Test full pipeline end-to-end
- `tests/integration/test_config.py` - Test configuration loading

**Sample test fixtures:**
```python
# Sample market data
market_data = [
    {"date": "2025-10-27", "close": 150.0, "volume": 1000000},
    {"date": "2025-10-28", "close": 152.0, "volume": 1100000},
]

# Sample fundamentals
fundamentals = {
    "pe_ratio": 25.0,
    "revenue_growth": 0.08,
    "ebitda_margin": 0.35
}

# Sample news
news_data = [
    {
        "title": "Apple beats earnings expectations",
        "content": "Apple reported strong quarterly results...",
        "published_at": "2025-10-27"
    }
]
```

---

## Usage Examples

### Example 1: Basic Single-Ticker Analysis

```python
from coordinator import run_stocksense_analysis
from agents import (
    PredictionAgent, SentimentAgent,
    ReflectionAgent, ExplanationAgent, build_openai_llm
)
from storage.vector_store import VectorStore

# Initialize components
llm = build_openai_llm()
vector_store = VectorStore()

prediction_agent = PredictionAgent(llm=llm)
sentiment_agent = SentimentAgent(vector_store=vector_store)
reflection_agent = ReflectionAgent()
explanation_agent = ExplanationAgent(llm=llm)

# Prepare data (fetched from Pam's and Tae's data fetchers)
ticker = "AAPL"
market_data = [...]  # From data/fetchers/price_data.py
news_data = [...]    # From data/fetchers/news_data.py
fundamentals = {...} # From data/fetchers/price_data.py

# Run analysis
result = run_stocksense_analysis(
    ticker=ticker,
    user_tier="premium",
    market_data=market_data,
    news_data=news_data,
    fundamentals=fundamentals,
    prediction_agent=prediction_agent,
    sentiment_agent=sentiment_agent,
    reflection_agent=reflection_agent,
    explanation_agent=explanation_agent
)

print(f"{ticker}: {result['direction']} ({result['confidence']:.0%} confidence)")
print(f"Sentiment: {result['sentiment']['current']}")
print(f"\n{result['explanation']}")
```

### Example 2: Using ModelConfig for Experimentation

```python
from coordinator import ModelConfig

# Load configuration
config = ModelConfig("config.yaml")

# Check which models to use
prediction_models = config.get_prediction_models()  # ["lstm"]
sentiment_models = config.get_sentiment_models()    # ["finbert"]

print(f"Using prediction models: {prediction_models}")
print(f"Using sentiment models: {sentiment_models}")

# Validate configuration
if not config.validate():
    print("Configuration invalid!")
```

### Example 3: Custom Workflow with Direct StateGraph

```python
from coordinator.workflow import create_stocksense_workflow
from agents import PredictionAgent, SentimentAgent, ReflectionAgent, ExplanationAgent

# Create agents
prediction_agent = PredictionAgent()
sentiment_agent = SentimentAgent()
reflection_agent = ReflectionAgent()
explanation_agent = ExplanationAgent(llm=build_openai_llm())

# Build workflow
workflow = create_stocksense_workflow(
    prediction_agent=prediction_agent,
    sentiment_agent=sentiment_agent,
    reflection_agent=reflection_agent,
    explanation_agent=explanation_agent
).compile()

# Invoke with custom state
result_state = workflow.invoke({
    "ticker": "MSFT",
    "user_tier": "basic",
    "market_data": [...],
    "news_data": [...],
    "fundamentals": {...},
    "warnings": []
})

print(result_state["explanation"])
```

---

## Configuration System

### config.yaml Structure

```yaml
prediction:
  mode: "single"              # or "ensemble"
  models: ["lstm"]            # or ["lstm", "gru", "gradient_boost"]
  strategy: "weighted_average" # or "voting", "stacking"
  weights:
    lstm: 1.0

sentiment:
  mode: "single"              # or "ensemble"
  models: ["finbert"]         # or ["finbert", "roberta", "vader"]
  strategy: "weighted_average"
  weights:
    finbert: 1.0
```

### Using ModelConfig

```python
from coordinator import ModelConfig

# Load config
config = ModelConfig("config.yaml")

# Get model lists
prediction_models = config.get_prediction_models()
sentiment_models = config.get_sentiment_models()

# Get ensemble settings
pred_mode = config.get_prediction_mode()  # "single" or "ensemble"
pred_weights = config.get_model_weights("prediction")  # {"lstm": 1.0}

# Validate
if config.validate():
    print("Configuration valid")
```

---

## Error Handling

### Agent-Level Error Handling

Each agent handles errors gracefully:

1. **PredictionAgent** - Returns neutral prediction if forecaster fails
2. **SentimentAgent** - Returns neutral sentiment if FinBERT fails
3. **ReflectionAgent** - Catches validation errors, adds warnings
4. **ExplanationAgent** - Returns basic explanation if LLM fails

### Workflow-Level Error Handling

The workflow catches errors at each node:

```python
try:
    prediction = prediction_agent.run(...)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    state["prediction_result"] = {"direction": "neutral", "confidence": 0.0}
    state["warnings"].append(f"Prediction error: {e}")
```

This ensures the workflow continues even if one agent fails.

---

## Logging

All agents and workflow nodes use Python's `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

# Example log statements
logger.info("Running prediction for AAPL")
logger.warning("No news data available")
logger.error("Forecasting failed", exc_info=True)
```

Configure logging in your application:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Testing Guidelines

### Unit Testing

Test each agent independently with mock dependencies:

```python
import pytest
from unittest.mock import Mock
from agents import PredictionAgent

def test_prediction_agent():
    # Mock forecaster
    mock_forecaster = Mock()
    mock_forecaster.predict.return_value = Mock(
        direction="up",
        confidence=0.85,
        narrative="Test narrative"
    )

    # Test agent
    agent = PredictionAgent()
    agent.forecaster = mock_forecaster

    result = agent.run(
        ticker="AAPL",
        market_data=[...],
        fundamentals={...}
    )

    assert result.direction == "up"
    assert result.confidence == 0.85
```

### Integration Testing

Test the full workflow end-to-end:

```python
def test_workflow_integration():
    # Initialize real agents
    agents = initialize_agents()

    # Run workflow
    result = run_stocksense_analysis(
        ticker="AAPL",
        user_tier="premium",
        market_data=sample_market_data,
        news_data=sample_news_data,
        fundamentals=sample_fundamentals,
        **agents
    )

    # Verify results
    assert result["ticker"] == "AAPL"
    assert result["direction"] in ["up", "down", "neutral"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["explanation"]) > 0
```

---

## Troubleshooting

### Common Issues

1. **ImportError: models.prediction.forecaster not found**
   - Ensure Pam's prediction models are implemented
   - Check PYTHONPATH includes project root

2. **ImportError: models.sentiment.finbert not found**
   - Ensure Tae's sentiment models are implemented
   - Install TensorFlow and transformers

3. **ValueError: OPENAI_API_KEY not found**
   - Add OPENAI_API_KEY to secrets.env
   - Load environment variables before running

4. **RuntimeError: create_forecaster returned None**
   - Check LSTM model weights exist
   - Verify model initialization in Pam's code

5. **FileNotFoundError: config.yaml not found**
   - Create config.yaml in project root
   - Or specify custom path: `ModelConfig("path/to/config.yaml")`

---

## Future Enhancements

### Planned for Post-MVP

1. **Ensemble Support**
   - Add GRU and Gradient Boost models (Pam)
   - Add RoBERTa, VADER, DistilBERT (Tae)
   - Implement weighted averaging and voting strategies

2. **SmartMoneyAgent**
   - Track institutional investors
   - Analyze insider trading
   - Monitor congressional stock trades

3. **Multi-Ticker Analysis**
   - Analyze multiple stocks simultaneously
   - Cross-stock correlation analysis
   - Sector-wide sentiment trends

4. **Advanced Features**
   - SHAP explanations visualization
   - Real-time streaming updates
   - Historical backtest mode

---

## Summary

**What was built:**
- 4 specialized agents (~400 lines total)
- Simplified workflow orchestration (~200 lines)
- Model configuration system (~100 lines)
- Utility functions and documentation (~30 lines)

**Total: ~730 lines** (vs. ~1,880 legacy lines = 61% reduction)

**Key principles followed:**
- ✅ Lightweight, minimal code
- ✅ No mock/placeholder data
- ✅ Type hints on all functions
- ✅ Error handling with specific exceptions
- ✅ Real ML models (LSTM, FinBERT)
- ✅ Configuration-driven architecture
- ✅ Clear integration points

**Next steps:**
1. Coordinate with Pam to finalize prediction model interface
2. Coordinate with Tae to finalize sentiment model interface
3. Work with Byeol to create comprehensive test suite
4. Integrate with Sua's FastAPI backend

---

**Last Updated:** 2025-11-04
