# Josh's Implementation Plan - Minimal Legacy Code Extraction

## Overview
This document outlines the **minimal** pieces of legacy code needed to implement Josh's responsibilities per `FreshStart/docs/ROLE_DIVISION.md`.

---

## Josh's Scope (from ROLE_DIVISION.md)

### Responsibilities:
1. **Agents** (`agents/` folder):
   - `prediction_agent.py` - Prediction analysis using Pam's models
   - `sentiment_agent.py` - Sentiment analysis using Tae's models
   - `reflection_agent.py` - Quality validation and confidence adjustment
   - `explanation_agent.py` - Natural language explanations with LLM

2. **Coordinator** (`coordinator/` folder):
   - `workflow.py` - Simplified LangGraph workflow orchestration
   - `config.py` - Model configuration system (PyYAML-based)

### Estimated LOC: ~820 lines

---

## Legacy Code Extraction Map

### 1. From `legacy/pipelines/realtime/agents.py`

#### Extract: PredictionAgent (lines 107-320)
**What:** ML-powered prediction agent using LSTM forecaster
**Lines:** ~200 lines
**Key features:**
- Uses `create_forecaster("lstm")` from models
- Supports GPT explanations with SHAP feature importance
- Returns PredictionResult with direction, confidence, narrative
- Includes daily probability curves and 95% confidence horizons

**Minimal extraction:**
```python
# Data structures (lines 61-75)
@dataclass
class PredictionResult:
    direction: Literal["up", "down", "neutral"]
    confidence: float
    narrative: str
    daily_probs: Optional[List[Dict[str, Any]]] = None
    horizon_95: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None

# PredictionAgent class (lines 107-320)
class PredictionAgent:
    def __init__(self, llm=None, *, use_ml_model=True, use_gpt_explanations=False):
        # Initialize LSTM forecaster
        from .models.forecaster import create_forecaster
        self.forecaster = create_forecaster("lstm")
        self.llm = llm
        self.use_gpt_explanations = use_gpt_explanations

    def run(self, *, ticker, market_data, fundamentals):
        # Use ML model to predict
        # Build narrative (basic or GPT-powered)
        # Return PredictionResult
```

**Dependencies:**
- `models/prediction/` (Pam's models) via `create_forecaster()`
- Optional: OpenAI LLM for GPT explanations

**Remove from legacy:**
- Excessive WHAT/WHY/HOW comments (violates CLAUDE.md lightweight code policy)
- Keep only essential docstrings

---

#### Extract: SentimentAgent (lines 325-458)
**What:** FinBERT-powered sentiment analysis with VectorStore integration
**Lines:** ~130 lines
**Key features:**
- Uses `create_sentiment_analyzer()` from sentiment module
- VectorStore integration for semantic search
- Returns SentimentResult with current, score, trend, headlines

**Minimal extraction:**
```python
# Data structure (lines 78-89)
@dataclass
class SentimentResult:
    current: Literal["positive", "neutral", "negative"]
    score: float
    trend: Literal["improving", "stable", "declining"]
    headlines: List[str]

# SentimentAgent class (lines 325-458)
class SentimentAgent:
    def __init__(self, llm=None, *, use_finbert=True, vector_store=None):
        from .sentiment.finbert import create_sentiment_analyzer
        self.sentiment_analyzer = create_sentiment_analyzer(
            vector_store=vector_store,
            enable_similarity_search=True
        )
        self.vector_store = vector_store

    def run(self, *, ticker, news_data):
        # Use FinBERT to analyze news sentiment
        # Store articles in VectorStore if available
        # Return SentimentResult
```

**Dependencies:**
- `models/sentiment/` (Tae's models) via `create_sentiment_analyzer()`
- `storage/vector_store.py` (Tae's ChromaDB integration)

**Remove from legacy:**
- SmartMoneyAgent support (not in Josh's scope)
- Multi-company coordination logic (too complex for MVP)

---

#### Extract: ExplanationAgent (lines 503-601)
**What:** LLM-powered plain English explanation generator
**Lines:** ~100 lines
**Key features:**
- Uses OpenAI LLM to synthesize analysis into narrative
- Customizable prompt templates
- Returns plain text explanation

**Minimal extraction:**
```python
# ExplanationAgent class (lines 503-601)
class ExplanationAgent:
    def __init__(self, llm, *, prompt=None):
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are a financial educator explaining stock analysis...
            Create explanation for {ticker} based on:
            - Prediction: {prediction}
            - Sentiment: {sentiment}
            - Smart Money: {smart_money}
            - User Tier: {user_tier}
            - Confidence Level: {confidence_level}
            """
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, ticker, prediction, sentiment, smart_money, user_tier, confidence_level):
        return self._chain.invoke({...}).strip()
```

**Dependencies:**
- OpenAI LLM (via `build_openai_llm()`)
- LangChain ChatPromptTemplate, StrOutputParser

---

#### Extract: Helper Functions (lines 607-638)
**What:** LLM builder utility
**Lines:** ~30 lines

**Minimal extraction:**
```python
def build_openai_llm(model="gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    from .api_keys import get_api_key

    api_key_str = get_api_key("openai", required=True)
    if api_key_str is None:
        raise ValueError("OPENAI_API_KEY is required but not found.")

    return ChatOpenAI(
        model=model,
        api_key=api_key_str,
        temperature=0.1
    )
```

**Dependencies:**
- `api_keys.py` (should already exist in FreshStart)

---

### 2. From `legacy/pipelines/realtime/reflection.py`

#### Extract: ReflectionAgent (entire file)
**What:** Quality assurance agent validating other agents' outputs
**Lines:** ~540 lines (but includes extensive comments)
**Actual code:** ~150 lines

**Key features:**
- Validates prediction reasonableness (confidence, direction)
- Checks sentiment-prediction alignment
- Verifies data freshness (market data timestamps)
- Checks fundamentals completeness
- Returns validation result with confidence adjustments and warnings

**Minimal extraction:**
```python
class ReflectionAgent:
    def __init__(self, *, max_data_age_days=3, min_confidence_threshold=0.0,
                 max_confidence_threshold=1.0, enable_alignment_check=True,
                 enable_freshness_check=True, enable_completeness_check=True):
        self.max_data_age_days = max_data_age_days
        self.min_confidence = min_confidence_threshold
        self.max_confidence = max_confidence_threshold
        # ... store flags

    def run(self, *, prediction, sentiment, filing, market_data, fundamentals):
        # Run validation checks
        # Calculate confidence adjustment
        # Return {validation_passed, confidence_adjustment, issues, recommendations}
```

**Dependencies:**
- Standard library only (datetime, typing)
- No external dependencies!

**Remove from legacy:**
- Excessive WHAT/WHY/HOW comments throughout the file
- Keep only essential docstrings per CLAUDE.md guidelines

---

### 3. From `legacy/pipelines/realtime/langgraph_workflow.py`

#### Extract: Simplified Workflow (lines 23-653)
**What:** LangGraph state machine orchestrating the analysis pipeline
**Lines to extract:** ~200 lines (heavily simplified from 850+)

**Simplifications for MVP:**
- Remove multi-company coordination (CoordinationAgent, HistoricalAnalysisAgent, SentimentAnalysisAgent)
- Remove SmartMoneyAgent (not in Josh's scope per ROLE_DIVISION.md)
- Keep only single-ticker analysis path
- Simplified state management

**Minimal extraction:**
```python
# Simplified State (lines 23-50 → simplified to ~30 lines)
class StockAnalysisState(TypedDict, total=False):
    ticker: str
    user_tier: Literal["basic", "premium"]
    market_data: Optional[Dict]
    news_data: List[Dict]
    fundamentals: Dict[str, float]
    prediction_result: Optional[Dict]
    sentiment_result: Optional[Dict]
    reflection_result: Optional[Dict]
    explanation: Optional[str]
    confidence_level: Literal["high", "medium", "low"]
    daily_probs: List[Dict]
    horizon_95: Optional[Dict]
    warnings: List[str]

# Workflow builder (lines 52-517 → simplified to ~100 lines)
def create_stocksense_workflow(
    prediction_agent, sentiment_agent, reflection_agent, explanation_agent
):
    builder = StateGraph(StockAnalysisState)

    # Define nodes
    builder.add_node("validate", validate_input)
    builder.add_node("fetch_data", collect_data)
    builder.add_node("predict", run_prediction)
    builder.add_node("sentiment", run_sentiment)
    builder.add_node("reflect", run_reflection)
    builder.add_node("explain", build_explanation)

    # Define edges (linear pipeline)
    builder.set_entry_point("validate")
    builder.add_edge("validate", "fetch_data")
    builder.add_edge("fetch_data", "predict")
    builder.add_edge("predict", "sentiment")
    builder.add_edge("sentiment", "reflect")
    builder.add_edge("reflect", "explain")
    builder.set_finish_point("explain")

    return builder

# Execution helper (lines 588-653 → ~65 lines)
def run_stocksense_analysis(*, ticker, user_tier, ...agents):
    workflow = create_stocksense_workflow(...agents).compile()
    result_state = workflow.invoke({"ticker": ticker, "user_tier": user_tier})
    return format_response(result_state)
```

**Dependencies:**
- LangGraph (StateGraph)
- All four agents (prediction, sentiment, reflection, explanation)
- Data fetchers (Pam's price_data, Tae's news_data)

**Remove from legacy:**
- Multi-company analysis (comprehensive_data, comprehensive_news_data, etc.)
- Coordination agent and working agents
- SmartMoneyAgent
- Complex SP500 data service logic

---

### 4. NEW FILE: `coordinator/config.py`

**What:** Model configuration system for runtime model selection
**Lines:** ~100 lines (NEW CODE, not from legacy)

**Purpose:**
- Load configuration from YAML files
- Select which prediction models to use (LSTM only, LSTM+GRU, all 3)
- Select which sentiment models to use (FinBERT only, ensemble, etc.)
- Set ensemble strategies and weights
- Enable easy experimentation with different model combinations

**Implementation outline:**
```python
import yaml
from typing import Dict, List, Any
from pathlib import Path

class ModelConfig:
    """Configuration for model selection and ensemble strategies."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_prediction_models(self) -> List[str]:
        """Returns list of prediction models to use."""
        return self.config.get("prediction", {}).get("models", ["lstm"])

    def get_sentiment_models(self) -> List[str]:
        """Returns list of sentiment models to use."""
        return self.config.get("sentiment", {}).get("models", ["finbert"])

    def get_ensemble_strategy(self, model_type: str) -> str:
        """Returns ensemble strategy (weighted_average, voting, etc.)."""
        return self.config.get(model_type, {}).get("strategy", "weighted_average")

    def get_model_weights(self, model_type: str) -> Dict[str, float]:
        """Returns model weights for ensemble."""
        return self.config.get(model_type, {}).get("weights", {})

# Example config.yaml structure:
"""
prediction:
  mode: "single"  # or "ensemble"
  models: ["lstm"]  # or ["lstm", "gru", "gradient_boost"]
  strategy: "weighted_average"
  weights:
    lstm: 1.0

sentiment:
  mode: "single"  # or "ensemble"
  models: ["finbert"]  # or ["finbert", "roberta", "vader"]
  strategy: "weighted_average"
  weights:
    finbert: 1.0
"""
```

**Dependencies:**
- PyYAML (already in requirements.txt)

---

## Total Line Count Breakdown

| Component | Legacy Source | Lines (Legacy) | Lines (Minimal) | Status |
|-----------|---------------|----------------|-----------------|--------|
| PredictionAgent | agents.py:107-320 | ~200 | ~100 | Extract + simplify |
| SentimentAgent | agents.py:325-458 | ~130 | ~80 | Extract + simplify |
| ExplanationAgent | agents.py:503-601 | ~100 | ~60 | Extract + simplify |
| Helper functions | agents.py:607-638 | ~30 | ~20 | Extract as-is |
| Data structures | agents.py:61-90 | ~30 | ~20 | Extract as-is |
| ReflectionAgent | reflection.py | ~540 | ~150 | Extract, remove verbose comments |
| Workflow | langgraph_workflow.py | ~850 | ~200 | Heavy simplification |
| config.py | NEW | 0 | ~100 | Create from scratch |
| **TOTAL** | | **~1,880** | **~730** | **61% reduction** |

**Josh's target:** 820 lines
**Our minimal extraction:** ~730 lines
**Buffer remaining:** 90 lines for adjustments

---

## Implementation Steps

### Step 1: Create Agent Files (agents/ folder)

1. **agents/__init__.py**
   - Export all four agent classes
   - Export data structures (PredictionResult, SentimentResult)
   - Export helper functions (build_openai_llm)

2. **agents/prediction_agent.py**
   - Extract PredictionAgent from legacy agents.py (lines 107-320)
   - Remove excessive comments, keep only essential docstrings
   - Keep GPT explanation feature (useful for Josh's scope)
   - Import from Pam's `models/prediction/` via factory function

3. **agents/sentiment_agent.py**
   - Extract SentimentAgent from legacy agents.py (lines 325-458)
   - Remove excessive comments
   - Keep VectorStore integration (Tae's responsibility)
   - Import from Tae's `models/sentiment/` via factory function

4. **agents/reflection_agent.py**
   - Extract ReflectionAgent from legacy reflection.py
   - **Critical:** Remove all WHAT/WHY/HOW comments (violates CLAUDE.md)
   - Keep only class docstring and method docstrings (1-2 sentences each)
   - Pure rule-based validation, no external dependencies

5. **agents/explanation_agent.py**
   - Extract ExplanationAgent from legacy agents.py (lines 503-601)
   - Remove excessive comments
   - Uses OpenAI LLM for natural language generation
   - Customizable prompt templates

### Step 2: Create Coordinator Files (coordinator/ folder)

1. **coordinator/__init__.py**
   - Export workflow functions
   - Export ModelConfig class

2. **coordinator/workflow.py**
   - Extract simplified workflow from legacy langgraph_workflow.py
   - Remove multi-company coordination logic
   - Remove SmartMoneyAgent integration
   - Keep linear pipeline: validate → fetch_data → predict → sentiment → reflect → explain
   - Simplify state management (remove comprehensive_* fields)

3. **coordinator/config.py** (NEW)
   - Create PyYAML-based configuration loader
   - Model selection logic
   - Ensemble strategy configuration
   - Validation helpers

4. **Example config.yaml** (in FreshStart root)
   - Provide example configuration file
   - Document all configuration options
   - Show single model and ensemble examples

### Step 3: Update Documentation

1. **docs/team/JOSH_CODE_GUIDE.md**
   - Update with all implementation details
   - Explain what each agent does
   - How to use the configuration system
   - Integration instructions with Pam's and Tae's components
   - Testing guidance

---

## Integration Points

### Josh ← Pam (Prediction Models)
**Interface:** `BasePredictionModel` with `predict(data) -> PredictionResult`
**Josh's usage:** PredictionAgent calls `create_forecaster("lstm")` which returns a model implementing BasePredictionModel

### Josh ← Tae (Sentiment Models)
**Interface:** `BaseSentimentModel` with `analyze(text) -> SentimentResult`
**Josh's usage:** SentimentAgent calls `create_sentiment_analyzer()` which returns a model implementing BaseSentimentModel

### Josh ← Tae (ChromaDB Vector Store)
**Interface:** `VectorStore` with `search_similar(query, ticker, n_results)`
**Josh's usage:** SentimentAgent passes `vector_store` to sentiment analyzer for semantic search

### Josh → Sua (API Integration)
**Interface:** `run_stocksense_analysis(ticker, user_tier, ...agents) -> Dict`
**Sua's usage:** FastAPI endpoint calls Josh's workflow function and returns formatted response

### Josh → Byeol (Testing)
**Interface:** All agent classes and workflow functions
**Byeol's usage:** Unit tests for each agent, integration tests for workflow

---

## Dependencies Summary

### External Packages (from requirements.txt):
- `langgraph` - Workflow orchestration
- `langchain-core` - Prompt templates, output parsers
- `langchain-openai` - OpenAI LLM integration
- `pyyaml` - Configuration file loading

### Internal Dependencies:
- `models/prediction/` - Pam's LSTM model (via `create_forecaster()`)
- `models/sentiment/` - Tae's FinBERT model (via `create_sentiment_analyzer()`)
- `storage/vector_store.py` - Tae's ChromaDB integration
- `data/fetchers/price_data.py` - Pam's price data fetcher
- `data/fetchers/news_data.py` - Tae's news data fetcher

---

## What Josh Does NOT Need from Legacy

1. **SmartMoneyAgent** - Not in Josh's scope per ROLE_DIVISION.md
2. **CoordinationAgent** - Too complex for MVP, designed for multi-company analysis
3. **HistoricalAnalysisAgent** - Too complex for MVP
4. **SentimentAnalysisAgent** (multi-company version) - Too complex for MVP
5. **SP500 data service** - Overly complex batch processing
6. **Multi-ticker state management** - Simplify to single ticker for MVP
7. **Complex coordination logic** - Linear pipeline is sufficient for MVP

---

## Adherence to CLAUDE.md Guidelines

### ✅ Lightweight, Minimal Code
- Removed extensive WHAT/WHY/HOW comments from legacy
- Keep only essential docstrings (1-2 sentences)
- Type hints on all functions
- Minimal error handling with specific exceptions

### ✅ No Placeholder/Mock Data
- All agents connect to real models (Pam's, Tae's)
- Real ML models, real API calls
- No TODO comments, no stub implementations

### ✅ Configuration-Driven Architecture
- Model selection via config.yaml (not hardcoded)
- Runtime flexibility for experimentation
- Easy A/B testing of different model combinations

### ✅ Proper Dependencies
- Only approved tech stack (Python, LangChain, LangGraph)
- No PyTorch, no cloud services
- Uses configured APIs from secrets.env

---

## Testing Strategy (for Byeol)

### Unit Tests:
- `test_prediction_agent.py` - Test prediction with mock forecaster
- `test_sentiment_agent.py` - Test sentiment with mock analyzer
- `test_reflection_agent.py` - Test validation logic with fixture data
- `test_explanation_agent.py` - Test LLM chain with mock LLM

### Integration Tests:
- `test_workflow.py` - Test full LangGraph pipeline end-to-end
- `test_config.py` - Test configuration loading and validation

### Fixtures:
- Sample market data (from Pam)
- Sample news articles (from Tae)
- Expected agent outputs

---

## Risk Mitigation

### Risk 1: Pam's models not ready yet
**Mitigation:** PredictionAgent gracefully handles missing forecaster
**Fallback:** Raise clear error message, log warning

### Risk 2: Tae's models not ready yet
**Mitigation:** SentimentAgent gracefully handles missing analyzer
**Fallback:** Raise clear error message, log warning

### Risk 3: Configuration file missing
**Mitigation:** Provide sane defaults (single LSTM, single FinBERT)
**Fallback:** Create example config.yaml with documentation

### Risk 4: OpenAI API key missing
**Mitigation:** ExplanationAgent requires OpenAI key, raises clear error
**Fallback:** User must add OPENAI_API_KEY to secrets.env

---

## Success Criteria

### Code Quality:
- ✅ All files follow CLAUDE.md guidelines (lightweight, minimal code)
- ✅ Type hints on all parameters and returns
- ✅ Error handling with specific exceptions
- ✅ No mock/placeholder/TODO code

### Functionality:
- ✅ Prediction agent uses Pam's models correctly
- ✅ Sentiment agent uses Tae's models correctly
- ✅ Reflection agent validates outputs correctly
- ✅ Explanation agent generates clear narratives
- ✅ Workflow orchestrates all agents in correct order
- ✅ Configuration system loads and validates settings

### Integration:
- ✅ Integrates with Pam's prediction models via BasePredictionModel interface
- ✅ Integrates with Tae's sentiment models via BaseSentimentModel interface
- ✅ Integrates with Tae's ChromaDB via VectorStore interface
- ✅ Sua can call workflow function from FastAPI endpoints
- ✅ Byeol can write unit and integration tests

### Documentation:
- ✅ JOSH_CODE_GUIDE.md updated with all implementation details
- ✅ Example config.yaml provided with documentation
- ✅ Integration instructions for all team members

---

## Timeline Estimate

1. **Extract agents/** (4 files) - 2 hours
2. **Extract coordinator/workflow.py** - 1 hour
3. **Create coordinator/config.py** - 1 hour
4. **Update JOSH_CODE_GUIDE.md** - 1 hour
5. **Testing and validation** - 1 hour

**Total:** ~6 hours of focused work

---

## Conclusion

This plan extracts **~730 lines** of minimal, essential code from the legacy system while staying well within Josh's **820 line budget**. The extraction focuses on:

1. **Core agent functionality** - Prediction, sentiment, reflection, explanation
2. **Simplified workflow** - Linear pipeline, no complex coordination
3. **Configuration-driven design** - Easy experimentation without code changes
4. **Clean integration points** - Clear interfaces with Pam's and Tae's components

The implementation prioritizes:
- **Lightweight code** (per CLAUDE.md guidelines)
- **Real functionality** (no mocks or placeholders)
- **Proper architecture** (SOLID principles, dependency injection)
- **Easy testing** (clear interfaces, minimal dependencies)

Josh can now implement these components independently while coordinating with Pam (for model interfaces) and Tae (for sentiment and vector store interfaces).
