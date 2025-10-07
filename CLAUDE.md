# CLAUDE.md - Claude AI Agent Guidelines for StockSense

## Project Overview

StockSense is a machine-learning-powered stock analysis platform that provides explainable forecasts, sentiment analysis, and smart money tracking for retail investors. This project strictly avoids LLM fallbacks for core ML functionality and follows enterprise software engineering standards.

## Core Engineering Principles

### 1. Software Engineering Best Practices

#### Object-Oriented Programming (OOP)
- **Encapsulation**: Group related data and behavior into cohesive classes
- **Abstraction**: Hide implementation details behind clean interfaces
- **Inheritance**: Use base classes for shared functionality (e.g., `BaseDataAdapter`)
- **Polymorphism**: Allow interchangeable implementations via interfaces

#### SOLID Principles
- **Single Responsibility**: Each class has one reason to change
  - Example: `FinBERTAnalyzer` only handles sentiment analysis, not data fetching
- **Open/Closed**: Open for extension, closed for modification
  - Example: New data adapters extend `BaseDataAdapter` without modifying existing code
- **Liskov Substitution**: Subtypes must be substitutable for base types
  - Example: Any `BaseDataAdapter` implementation works with `DataService`
- **Interface Segregation**: Clients shouldn't depend on unused methods
  - Example: Separate interfaces for read vs. write operations
- **Dependency Inversion**: Depend on abstractions, not concrete implementations
  - Example: Inject adapters via constructor, not hardcoded instantiation

### 2. Code Documentation Requirements

#### Inline Code Comments - Mandatory Format
Every function, class, and non-trivial code block MUST include comments explaining:

```python
# WHAT: Brief description of what this code does
# HOW: Explain the approach/algorithm used
# WHY: Justify design decisions and edge case handling
# DATA: Document input parameters, return values, and data transformations
def example_function(input_data: Dict[str, Any]) -> ProcessedResult:
    """
    High-level function summary.

    Args:
        input_data: Dictionary containing raw market data with keys:
                    - 'ticker': Stock symbol (str)
                    - 'prices': List of OHLCV tuples
                    - 'timestamp': Unix epoch (int)

    Returns:
        ProcessedResult object with:
            - normalized_prices: numpy array scaled to [0,1]
            - metadata: Dict with processing timestamp and stats

    Raises:
        ValueError: If ticker is invalid or prices list is empty
        DataError: If timestamp is in future or prices contain NaN
    """
    # WHAT: Validate input data structure before processing
    # WHY: Fail fast with clear error messages rather than cryptic failures downstream
    # HOW: Check required keys and data types using explicit validation
    if not isinstance(input_data, dict) or 'ticker' not in input_data:
        raise ValueError("input_data must be dict with 'ticker' key")

    # WHAT: Extract and normalize price data
    # HOW: Convert list of tuples to numpy array, then apply min-max scaling
    # WHY: Neural networks require normalized inputs in [0,1] range for stable training
    # DATA: Transform [(open, high, low, close, volume), ...] -> normalized array
    prices = np.array(input_data['prices'])
    normalized = (prices - prices.min()) / (prices.max() - prices.min())

    return ProcessedResult(normalized_prices=normalized, metadata={...})
```

#### Documentation Location
- **All documentation** goes in the `docs/` folder
- Keep docs synchronized with code changes
- Update relevant docs when modifying functionality
- Document architecture decisions in `docs/resources/`

### 3. No Mock/Synthetic/Placeholder Data

#### Prohibited Practices
- ❌ Returning hardcoded "example" data from functions
- ❌ Using placeholder values like `TODO: implement actual logic`
- ❌ Mock implementations that don't connect to real data sources
- ❌ Pseudo-code in production files

#### Required Practices
- ✅ Connect to actual APIs (Alpha Vantage, Tiingo, Finnhub, SEC EDGAR)
- ✅ Use real ML models (FinBERT, trained forecasters)
- ✅ Implement complete error handling with specific exceptions
- ✅ Return actual computed results or fail explicitly

```python
# ❌ BAD - Mock placeholder
def get_sentiment(text: str) -> float:
    # TODO: Implement FinBERT integration
    return 0.5  # placeholder neutral sentiment

# ✅ GOOD - Real implementation with error handling
def get_sentiment(text: str) -> float:
    """
    Analyze sentiment using FinBERT model.

    WHAT: Computes sentiment score [-1, 1] for financial text
    HOW: Tokenizes text, runs through FinBERT, returns logit-normalized score
    WHY: FinBERT is fine-tuned on financial language, outperforms general BERT
    DATA: Input text (str) -> sentiment score (float) where -1=bearish, 1=bullish
    """
    # WHAT: Load FinBERT model from local cache
    # WHY: Fail fast if model not available, no LLM fallback allowed
    # HOW: Check model path exists, raise descriptive error if missing
    if not self.model_path.exists():
        raise RuntimeError(
            f"FinBERT model not found at {self.model_path}. "
            f"Run 'python scripts/download_finbert.py' to download."
        )

    # WHAT: Tokenize input text and run inference
    # HOW: Use FinBERT tokenizer, pad to max_length, run forward pass
    # DATA: text (str) -> input_ids (torch.Tensor) -> logits (3-class output)
    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = self.model(**inputs)

    # WHAT: Convert 3-class probabilities to continuous sentiment score
    # HOW: Apply softmax, compute weighted average: (positive - negative)
    # WHY: Continuous score more useful than discrete labels for downstream agents
    # DATA: logits [batch, 3] -> probs [positive, neutral, negative] -> score [-1, 1]
    probs = torch.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    sentiment_score = probs[2] - probs[0]  # bullish - bearish

    return float(sentiment_score)
```

### 4. Testing Requirements

#### Test Coverage Mandates
- ✅ **Every function** you create must have corresponding tests
- ✅ **Every class** needs unit tests for all public methods
- ✅ **Every API endpoint** requires integration tests
- ✅ **Every data transformation** needs validation tests

#### Test Structure
```
tests/
├── unit/                    # Fast, isolated tests (no external dependencies)
│   ├── test_agents.py      # Test agent logic with mocked data services
│   ├── test_forecaster.py  # Test model inference with fixture data
│   └── test_adapters.py    # Test adapter logic with recorded API responses
├── integration/             # Tests with real external systems
│   ├── test_api_endpoints.py
│   └── test_workflow.py
└── fixtures/                # Recorded API responses, sample datasets
    ├── alpha_vantage_response.json
    └── sample_prices.csv
```

#### Test Example
```python
# tests/unit/test_sentiment.py
import pytest
from pathlib import Path
from pipelines.realtime.sentiment.finbert import FinBERTAnalyzer

class TestFinBERTAnalyzer:
    """
    Test suite for FinBERT sentiment analysis.

    WHAT: Validates sentiment scoring accuracy and error handling
    HOW: Uses recorded test data and mocked model for fast execution
    WHY: Ensures sentiment agent fails correctly when model unavailable
    """

    @pytest.fixture
    def analyzer(self, tmp_path):
        """
        WHAT: Create FinBERT analyzer instance with temporary model path.
        WHY: Isolated testing environment, doesn't affect production models.
        DATA: Returns configured FinBERTAnalyzer instance.
        """
        return FinBERTAnalyzer(model_path=tmp_path / "finbert")

    def test_missing_model_raises_error(self, analyzer):
        """
        WHAT: Verify analyzer fails fast when model files missing.
        HOW: Call get_sentiment() without downloading model first.
        WHY: Must not silently fail or return placeholder data.
        DATA: Input text -> expect RuntimeError with installation instructions.
        """
        with pytest.raises(RuntimeError, match="FinBERT model not found"):
            analyzer.get_sentiment("Tesla stock surges on earnings beat")

    def test_bullish_sentiment_detection(self, analyzer_with_model):
        """
        WHAT: Verify positive sentiment scores for bullish financial text.
        HOW: Pass known bullish phrases, assert score > 0.5.
        WHY: Validate model correctly interprets financial language.
        DATA: Bullish text -> sentiment score in [0.5, 1.0] range.
        """
        text = "Exceptional revenue growth exceeded analyst expectations"
        score = analyzer_with_model.get_sentiment(text)

        # WHAT: Assert score indicates bullish sentiment
        # WHY: Score > 0.5 means model weighted positive class higher
        # DATA: Expected score range [0.5, 1.0] for strongly bullish text
        assert score > 0.5, f"Expected bullish score, got {score}"
        assert score <= 1.0, f"Score must be in [-1, 1], got {score}"
```

### 5. LLM Usage Policy - Critical Rules

#### Where LLMs ARE Allowed
- ✅ **Explanation Agent**: Generating natural language summaries for users
  - Example: Converting forecasts + sentiment into readable insights
- ✅ **OpenAI API integration**: When explicitly configured for text generation
- ✅ **LangGraph orchestration**: Using LLM for workflow coordination

#### Where LLMs ARE FORBIDDEN
- ❌ **Fallback for ML models**: Never use LLM when forecaster/FinBERT fails
- ❌ **Data substitution**: Don't generate synthetic data via prompts
- ❌ **Missing functionality**: Don't prompt LLM to "simulate" unimplemented features
- ❌ **Error recovery**: Don't catch exceptions and ask LLM to "figure it out"

```python
# ❌ FORBIDDEN - LLM fallback for missing ML model
def predict_price(ticker: str) -> float:
    try:
        return self.forecaster.predict(ticker)
    except ModelNotFoundError:
        # WRONG: Using LLM to guess prediction
        prompt = f"Predict tomorrow's stock price for {ticker}"
        return float(llm.invoke(prompt))

# ✅ CORRECT - Fail fast with actionable error
def predict_price(ticker: str) -> float:
    """
    WHAT: Generate price forecast using trained gradient boosting model.
    WHY: Deterministic ML predictions required, LLM guesses forbidden.
    DATA: ticker (str) -> predicted_price (float)
    """
    # WHAT: Check if forecaster model loaded successfully
    # WHY: Fail immediately with setup instructions if model missing
    # HOW: Raise descriptive RuntimeError, don't fall back to LLM
    if self.forecaster is None:
        raise RuntimeError(
            "Forecaster model not loaded. Train model with: "
            "python scripts/train_forecaster.py"
        )

    # WHAT: Run inference on trained model
    # HOW: Extract features from ticker data, pass to forecaster.predict()
    # DATA: ticker -> features [volatility, momentum, ...] -> predicted_price
    return self.forecaster.predict(ticker)
```

### 6. Project Direction Validation

#### When to Challenge User Requests
Alert the user if their request contradicts core project principles:

```
⚠️ PROJECT DIRECTION CONFLICT DETECTED

Request: "Add an LLM fallback when FinBERT fails"

This conflicts with:
- docs/resources/ACTION_PLAN_NO_LLM_FALLBACK.md (lines 6-18)
- Current architecture requirement: fail fast on missing ML assets

Before implementing, please confirm:
1. Should we modify the no-fallback policy?
2. Is this a temporary workaround or permanent change?
3. Does this align with the project's ML-first approach?

Current project direction: All forecasting/sentiment done by trained models,
LLMs only for explanation generation. Suggest alternatives:
- Option A: Improve model availability checks on startup
- Option B: Add better error messages for missing models
- Option C: Create monitoring alerts for model staleness
```

#### Indicators of Direction Conflicts
- Request to add mock/placeholder data in production code
- Request to use LLM for forecasting or sentiment (should use ML models)
- Request to remove tests or reduce coverage
- Request to violate SOLID principles (tight coupling, hardcoded dependencies)
- Request to skip documentation updates

### 7. Data Flow Documentation

Every component that transforms data must document:

```python
class DataTransformer:
    """
    WHAT: Converts raw API responses to feature-engineered training data.
    WHY: ML models require normalized, time-aligned features.

    DATA FLOW:
        Input:  API response JSON with nested price/volume/fundamentals
        Step 1: Parse JSON -> validated Pydantic models (PriceData, Fundamentals)
        Step 2: Time-align series -> pandas DataFrame with datetime index
        Step 3: Compute features -> add moving averages, RSI, volatility
        Step 4: Normalize -> scale features to [0,1] using MinMaxScaler
        Output: numpy array [n_samples, n_features] ready for model.fit()

    DEPENDENCIES:
        - pandas: Time series alignment and feature computation
        - numpy: Array operations and normalization
        - sklearn.preprocessing: MinMaxScaler for feature scaling
    """
```

## File Organization Standards

### Production Code
```
pipelines/realtime/
├── agents.py              # LangGraph agent implementations
├── api.py                 # FastAPI endpoints and validation
├── langgraph_workflow.py  # Multi-agent orchestration
├── repository.py          # Data persistence layer (JSON)
├── models/
│   └── forecaster.py      # ML model inference
├── sentiment/
│   └── finbert.py         # FinBERT sentiment analysis
├── smart_money/
│   └── sec_edgar.py       # SEC filing data integration
└── data_adapters/
    ├── base.py            # Abstract base adapter
    ├── alpha_vantage.py   # Alpha Vantage API client
    └── tiingo.py          # Tiingo API client
```

### Documentation
```
docs/
├── IMPLEMENTATION_STATUS.md        # Current progress tracker
├── DATA_INVENTORY.md               # Available datasets and metrics
├── resources/
│   ├── TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md
│   ├── ACTION_PLAN_NO_LLM_FALLBACK.md
│   └── INLINE_COMMENTS_PROGRESS.md
```

### Tests
```
tests/
├── unit/                  # Fast isolated tests
├── integration/           # Real API/system tests
└── fixtures/              # Test data and recorded responses
```

## Code Review Checklist

Before considering any code complete, verify:

- [ ] Inline comments explain WHAT, HOW, WHY, DATA for every function
- [ ] No mock/placeholder/TODO code in production files
- [ ] All new functions have corresponding tests
- [ ] Tests achieve >80% coverage for new code
- [ ] No LLM fallbacks for ML functionality (forecasting, sentiment)
- [ ] Follows SOLID principles (check for tight coupling, god classes)
- [ ] Documentation updated in `docs/` folder
- [ ] Type hints on all function parameters and return values
- [ ] Error handling raises specific exceptions with actionable messages
- [ ] No hardcoded paths, API keys, or configuration (use env vars)

## Common Patterns

### Adapter Pattern for Data Sources
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDataAdapter(ABC):
    """
    WHAT: Abstract interface for market data providers.
    WHY: Allows swapping data sources without changing agent logic.
    HOW: Subclasses implement fetch methods for their specific API.
    DATA: All adapters return standardized MarketData objects.
    """

    @abstractmethod
    def fetch_prices(self, ticker: str, start_date: str, end_date: str) -> MarketData:
        """
        WHAT: Retrieve OHLCV price data for date range.
        DATA: Returns MarketData with lists of prices, volumes, timestamps.
        """
        pass

    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        """
        WHAT: Get company financial metrics (PE ratio, EPS, etc.).
        DATA: Returns Fundamentals object with validated numeric fields.
        """
        pass

class AlphaVantageAdapter(BaseDataAdapter):
    """
    WHAT: Alpha Vantage API client implementing BaseDataAdapter.
    HOW: Makes HTTP requests to Alpha Vantage endpoints, parses JSON.
    WHY: Alpha Vantage provides free tier with good historical coverage.
    """

    def __init__(self, api_key: str):
        """
        WHAT: Initialize adapter with API credentials.
        WHY: Dependency injection allows testing with different keys.
        DATA: api_key (str) stored as private attribute.
        """
        # WHAT: Validate API key format before storing
        # WHY: Fail fast if key obviously invalid (wrong length, format)
        # HOW: Check key matches expected pattern for this provider
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid Alpha Vantage API key")
        self._api_key = api_key
        self._base_url = "https://www.alphavantage.co/query"

    def fetch_prices(self, ticker: str, start_date: str, end_date: str) -> MarketData:
        """
        WHAT: Download OHLCV data from Alpha Vantage TIME_SERIES_DAILY.
        HOW: Construct request URL, parse JSON, convert to MarketData.
        WHY: Standardizes different API response formats into common structure.
        DATA: ticker, date range -> HTTP GET -> JSON -> MarketData object.
        """
        # WHAT: Build API request with required parameters
        # DATA: ticker, API key, function name -> query string
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self._api_key,
            "outputsize": "full"
        }

        # WHAT: Execute HTTP request with retry logic
        # WHY: APIs can have transient failures, retry improves reliability
        # HOW: Use requests library with exponential backoff
        response = self._request_with_retry(params)

        # WHAT: Parse JSON and extract time series data
        # HOW: Navigate nested JSON structure, validate date range
        # DATA: JSON {"Time Series (Daily)": {"2024-01-01": {...}}} -> list of prices
        time_series = response.json().get("Time Series (Daily)", {})
        prices = self._parse_time_series(time_series, start_date, end_date)

        return MarketData(ticker=ticker, prices=prices)
```

### Repository Pattern for Persistence
```python
class JSONRepository:
    """
    WHAT: JSON file-based persistence for watchlists, history, alerts.
    WHY: Simple storage without database overhead, sufficient for MVP.
    HOW: Read/write JSON files with file locking for concurrency safety.
    DATA: Python objects <-> JSON files in data/ directory.
    """

    def save_watchlist(self, user_id: str, tickers: List[str]) -> None:
        """
        WHAT: Persist user's watchlist to JSON file.
        HOW: Serialize tickers list to JSON with atomic write.
        WHY: File locking prevents corruption from concurrent writes.
        DATA: tickers (List[str]) -> JSON array in data/watchlists/{user_id}.json.
        """
        # Implementation with file locking
        pass
```

## Summary

This document defines the engineering standards for StockSense development. When contributing code:

1. **Follow SOLID principles** - Keep classes focused, depend on abstractions
2. **Document thoroughly** - WHAT, HOW, WHY, DATA for every function
3. **Test everything** - Unit + integration tests for all new code
4. **No placeholders** - Only real implementations, never mock data
5. **No LLM fallbacks** - ML models only, fail fast if unavailable
6. **Update docs** - Keep `docs/` synchronized with code changes
7. **Challenge conflicts** - Alert user if request contradicts project direction

These rules ensure maintainable, reliable, production-grade code that meets enterprise standards.
