# AGENT.md - AI Agent Implementation Guidelines

## Purpose

This document provides implementation-level instructions for AI agents (LangGraph agents, autonomous code generators, and development assistants) working on the StockSense project. These guidelines ensure consistency, quality, and alignment with project architecture.

## Agent Types in StockSense

### 1. LangGraph Analysis Agents (Production Code)
These agents run in production to analyze stocks:
- **ForecastAgent**: Price prediction using trained ML models
- **SentimentAgent**: FinBERT-powered news sentiment analysis
- **SmartMoneyAgent**: SEC EDGAR institutional activity tracking
- **ExplanationAgent**: Natural language insight generation (LLM-powered)
- **AlertAgent**: Risk threshold monitoring and notifications

### 2. Development Agents (This Agent)
AI assistants that write, refactor, and test code for this project.

## Development Agent Responsibilities

### Task Planning
When assigned a task:

1. **Assess Scope**
   ```
   WHAT: What is the user asking for?
   WHERE: Which files/modules are affected?
   WHY: Does this align with project architecture?
   HOW: What's the implementation approach?
   DATA: What data structures are involved?
   ```

2. **Check for Conflicts**
   - Review [docs/resources/ACTION_PLAN_NO_LLM_FALLBACK.md](docs/resources/ACTION_PLAN_NO_LLM_FALLBACK.md)
   - Verify request doesn't violate SOLID principles
   - Confirm no mock/placeholder data requested
   - Ensure LLM usage appropriate (explanation only, not fallback)

3. **Plan Implementation**
   - Break complex tasks into steps
   - Identify affected files and dependencies
   - Determine required tests
   - Note documentation updates needed

### Code Generation Standards

#### Every Function Must Include

```python
def function_name(param: Type) -> ReturnType:
    """
    Brief one-line summary of function purpose.

    Detailed description of functionality, use cases, and important behavior.

    Args:
        param: Description of parameter including:
               - Expected data structure/format
               - Valid value ranges or constraints
               - Example values

    Returns:
        Description of return value including:
        - Data structure and shape
        - Example return values
        - Conditions for different return values

    Raises:
        ExceptionType: When and why this exception occurs
        AnotherException: Specific error conditions

    Example:
        >>> result = function_name("input")
        >>> print(result)
        expected_output
    """
    # WHAT: Describe what this code block does at high level
    # WHY: Explain design decision or necessity of this approach
    # HOW: Describe algorithm or technique used
    # DATA: Document data transformation (input -> intermediate -> output)

    # Actual implementation here
    pass
```

#### Inline Comment Requirements

**Every code block** needs at least one comment explaining WHAT/HOW/WHY/DATA:

```python
# WHAT: Parse API response JSON into validated Pydantic models
# WHY: Ensure data types match expectations before passing to ML model
# HOW: Iterate through nested JSON, extract fields, validate with Pydantic
# DATA: JSON dict -> List[PriceData] objects with float prices, datetime timestamps
parsed_data = [
    PriceData(
        timestamp=datetime.fromisoformat(item["date"]),
        price=float(item["close"]),
        volume=int(item["volume"])
    )
    for item in response.json()["prices"]
]
```

**Before each significant logic block**:
```python
# WHAT: Calculate exponential moving average for trend detection
# WHY: EMA reacts faster to price changes than simple moving average
# HOW: Apply exponential smoothing with alpha=0.1 (10-period equivalent)
# DATA: prices [n_samples] -> ema [n_samples] where ema[i] = alpha*price[i] + (1-alpha)*ema[i-1]
ema = []
for i, price in enumerate(prices):
    if i == 0:
        ema.append(price)
    else:
        ema.append(0.1 * price + 0.9 * ema[-1])
```

**After complex transformations**:
```python
normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
# DATA: features transformed from original scale to z-scores (mean=0, std=1)
# WHY: Prevents features with larger magnitudes from dominating model training
```

### SOLID Principles in Practice

#### Single Responsibility Principle
Each class has **one reason to change**:

```python
# ❌ BAD - Class doing too much
class StockAnalyzer:
    def fetch_data(self, ticker: str) -> Dict:
        """Fetches data from API"""
        pass

    def analyze_sentiment(self, text: str) -> float:
        """Runs FinBERT analysis"""
        pass

    def generate_forecast(self, data: Dict) -> float:
        """Trains and predicts with ML model"""
        pass

    def save_to_database(self, result: Dict) -> None:
        """Persists results"""
        pass

# ✅ GOOD - Separated responsibilities
class DataFetcher:
    """
    WHAT: Retrieves market data from external APIs.
    WHY: Single responsibility - data acquisition only.
    """
    def fetch(self, ticker: str) -> MarketData:
        pass

class SentimentAnalyzer:
    """
    WHAT: Analyzes financial text sentiment using FinBERT.
    WHY: Single responsibility - sentiment scoring only.
    """
    def analyze(self, text: str) -> SentimentScore:
        pass

class Forecaster:
    """
    WHAT: Generates price predictions using trained ML model.
    WHY: Single responsibility - forecasting only.
    """
    def predict(self, features: np.ndarray) -> Prediction:
        pass

class ResultRepository:
    """
    WHAT: Persists analysis results to storage.
    WHY: Single responsibility - data persistence only.
    """
    def save(self, result: AnalysisResult) -> None:
        pass
```

#### Open/Closed Principle
Open for extension, closed for modification:

```python
# WHAT: Base adapter that new data sources can extend without modifying existing code
# WHY: Adding new data providers shouldn't require changing agent logic
# HOW: Define abstract interface, implement in subclasses
class BaseDataAdapter(ABC):
    """
    WHAT: Abstract interface for market data providers.
    WHY: Allows extending with new providers without modifying dependent code.
    HOW: Subclasses implement fetch methods for specific APIs.
    """

    @abstractmethod
    def fetch_prices(self, ticker: str) -> MarketData:
        """
        WHAT: Retrieve OHLCV price data.
        DATA: ticker (str) -> MarketData object with prices, volumes, timestamps.
        """
        pass

# WHAT: Concrete implementation for Alpha Vantage API
# WHY: Extends BaseDataAdapter without modifying it (Open/Closed Principle)
class AlphaVantageAdapter(BaseDataAdapter):
    def fetch_prices(self, ticker: str) -> MarketData:
        # Implementation for Alpha Vantage
        pass

# WHAT: Concrete implementation for Tiingo API
# WHY: Can add new provider without changing BaseDataAdapter or AlphaVantageAdapter
class TiingoAdapter(BaseDataAdapter):
    def fetch_prices(self, ticker: str) -> MarketData:
        # Implementation for Tiingo
        pass
```

#### Liskov Substitution Principle
Subtypes must be substitutable for base types:

```python
# WHAT: Service that works with any BaseDataAdapter implementation
# WHY: Can swap adapters without breaking functionality (Liskov Substitution)
# HOW: Depends on abstract interface, not concrete implementations
class DataService:
    def __init__(self, adapter: BaseDataAdapter):
        """
        WHAT: Initialize service with data adapter.
        WHY: Dependency injection enables testing and provider swapping.
        DATA: adapter must implement BaseDataAdapter interface.
        """
        # WHAT: Validate adapter implements required interface
        # WHY: Fail fast if invalid adapter provided
        # HOW: Check adapter is instance of BaseDataAdapter
        if not isinstance(adapter, BaseDataAdapter):
            raise TypeError("adapter must inherit from BaseDataAdapter")
        self._adapter = adapter

    def get_market_data(self, ticker: str) -> MarketData:
        """
        WHAT: Fetch market data using configured adapter.
        WHY: Works with any adapter that follows BaseDataAdapter interface.
        DATA: ticker (str) -> delegates to adapter.fetch_prices() -> MarketData.
        """
        # WHAT: Delegate to adapter's fetch_prices method
        # HOW: Call abstract method implemented by concrete adapter
        # DATA: ticker -> adapter API call -> MarketData object
        return self._adapter.fetch_prices(ticker)

# WHAT: Usage example showing adapter substitutability
# WHY: Can use AlphaVantage or Tiingo interchangeably (Liskov Substitution)
alpha_service = DataService(AlphaVantageAdapter(api_key="..."))
tiingo_service = DataService(TiingoAdapter(api_key="..."))

# Both work identically because both implement BaseDataAdapter contract
data1 = alpha_service.get_market_data("AAPL")
data2 = tiingo_service.get_market_data("AAPL")
```

#### Interface Segregation Principle
Clients shouldn't depend on unused methods:

```python
# ❌ BAD - Fat interface forces implementations to stub unused methods
class DataAdapter(ABC):
    @abstractmethod
    def fetch_prices(self, ticker: str) -> MarketData:
        pass

    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        pass

    @abstractmethod
    def fetch_news(self, ticker: str) -> List[Article]:
        pass

    @abstractmethod
    def fetch_options(self, ticker: str) -> OptionsChain:
        pass

# ✅ GOOD - Segregated interfaces, implement only what you need
class PriceProvider(ABC):
    """WHAT: Interface for price data only."""
    @abstractmethod
    def fetch_prices(self, ticker: str) -> MarketData:
        pass

class FundamentalsProvider(ABC):
    """WHAT: Interface for fundamental data only."""
    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        pass

class NewsProvider(ABC):
    """WHAT: Interface for news data only."""
    @abstractmethod
    def fetch_news(self, ticker: str) -> List[Article]:
        pass

# WHAT: Adapter implements only interfaces it actually supports
# WHY: No need to stub unimplemented methods (Interface Segregation)
class AlphaVantageAdapter(PriceProvider, FundamentalsProvider):
    """
    WHAT: Alpha Vantage adapter supporting prices and fundamentals.
    WHY: Doesn't implement NewsProvider because Alpha Vantage lacks news API.
    """
    def fetch_prices(self, ticker: str) -> MarketData:
        # Real implementation
        pass

    def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        # Real implementation
        pass
```

#### Dependency Inversion Principle
Depend on abstractions, not concrete implementations:

```python
# ❌ BAD - Direct dependency on concrete implementation
class ForecastAgent:
    def __init__(self):
        # WHAT: Hardcoded dependency on specific adapter
        # WHY BAD: Can't swap providers, can't test with mock, tight coupling
        self.data_source = AlphaVantageAdapter(api_key=os.getenv("ALPHA_KEY"))

    def run(self, ticker: str) -> Forecast:
        data = self.data_source.fetch_prices(ticker)
        return self._predict(data)

# ✅ GOOD - Dependency injection with abstract interface
class ForecastAgent:
    def __init__(self, data_adapter: BaseDataAdapter):
        """
        WHAT: Initialize agent with injected data adapter.
        WHY: Depends on abstraction (BaseDataAdapter), not concrete class.
        HOW: Accept any object implementing BaseDataAdapter interface.
        DATA: data_adapter must implement fetch_prices() method.
        """
        # WHAT: Store injected dependency for later use
        # WHY: Enables dependency inversion - high-level agent doesn't depend on low-level adapter
        # HOW: Accept interface, not concrete implementation
        self._data_adapter = data_adapter

    def run(self, ticker: str) -> Forecast:
        """
        WHAT: Generate forecast using injected data adapter.
        WHY: Works with any adapter implementation (Alpha Vantage, Tiingo, test mock).
        DATA: ticker -> adapter.fetch_prices() -> self._predict() -> Forecast.
        """
        # WHAT: Fetch data via abstract interface
        # HOW: Call method defined in BaseDataAdapter, implemented by concrete adapter
        # DATA: ticker (str) -> MarketData object with prices
        data = self._data_adapter.fetch_prices(ticker)

        # WHAT: Generate prediction from fetched data
        # DATA: MarketData -> ML model inference -> Forecast object
        return self._predict(data)

# WHAT: Usage example showing dependency injection
# WHY: Can inject different implementations without changing ForecastAgent
alpha_agent = ForecastAgent(data_adapter=AlphaVantageAdapter("key1"))
tiingo_agent = ForecastAgent(data_adapter=TiingoAdapter("key2"))
test_agent = ForecastAgent(data_adapter=MockAdapter())  # For testing
```

### Testing Requirements

#### Test Structure
For every module `foo.py`, create `tests/unit/test_foo.py`:

```python
# tests/unit/test_forecaster.py
import pytest
import numpy as np
from pathlib import Path
from pipelines.realtime.models.forecaster import Forecaster

class TestForecaster:
    """
    Test suite for ML forecasting model.

    WHAT: Validates model loading, inference, and error handling.
    HOW: Uses pytest fixtures for setup, recorded data for reproducibility.
    WHY: Ensures forecaster fails gracefully when model missing (no LLM fallback).
    """

    @pytest.fixture
    def sample_features(self) -> np.ndarray:
        """
        WHAT: Generate sample feature array for testing.
        WHY: Reproducible test data without external dependencies.
        DATA: Returns [10, 5] array (10 samples, 5 features).
        """
        # WHAT: Create synthetic features with known properties
        # HOW: Use numpy random with fixed seed for reproducibility
        # DATA: 10 samples x 5 features (volatility, momentum, volume, RSI, MA)
        np.random.seed(42)
        return np.random.randn(10, 5)

    @pytest.fixture
    def forecaster_with_model(self, tmp_path: Path) -> Forecaster:
        """
        WHAT: Create forecaster instance with trained model.
        WHY: Tests require valid model file for inference tests.
        HOW: Copy fixture model to temp directory, initialize forecaster.
        DATA: Returns configured Forecaster instance pointing to test model.
        """
        # WHAT: Setup test model directory
        # WHY: Isolated test environment, doesn't affect production models
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # WHAT: Copy pre-trained test model to temp location
        # HOW: Use shutil to copy from fixtures directory
        # DATA: fixtures/test_model.pkl -> tmp_path/model/forecaster.pkl
        import shutil
        fixture_model = Path(__file__).parent.parent / "fixtures" / "test_model.pkl"
        shutil.copy(fixture_model, model_dir / "forecaster.pkl")

        return Forecaster(model_path=model_dir / "forecaster.pkl")

    def test_missing_model_raises_error(self, tmp_path: Path):
        """
        WHAT: Verify forecaster fails fast when model file doesn't exist.
        HOW: Initialize forecaster with non-existent path, call predict().
        WHY: Must not return placeholder data or fall back to LLM.
        DATA: Expect RuntimeError with training instructions.
        """
        # WHAT: Create forecaster pointing to non-existent model
        # WHY: Simulate production scenario where model not trained yet
        forecaster = Forecaster(model_path=tmp_path / "nonexistent.pkl")

        # WHAT: Attempt prediction with missing model
        # WHY: Should raise descriptive error, not silently fail
        # DATA: Expect RuntimeError matching pattern "model not found"
        with pytest.raises(RuntimeError, match="model not found"):
            forecaster.predict(np.random.randn(1, 5))

    def test_prediction_output_shape(self, forecaster_with_model, sample_features):
        """
        WHAT: Verify model returns correct output shape.
        HOW: Pass sample features, check output dimensions.
        WHY: Ensures model inference pipeline working correctly.
        DATA: Input [10, 5] -> Output [10] (one prediction per sample).
        """
        # WHAT: Run inference on sample features
        # HOW: Call forecaster.predict() with known input shape
        # DATA: sample_features [10, 5] -> predictions [10]
        predictions = forecaster_with_model.predict(sample_features)

        # WHAT: Validate output shape matches expectations
        # WHY: Downstream agents expect 1D array of predictions
        # DATA: predictions.shape should be (10,) for 10 input samples
        assert predictions.shape == (10,), f"Expected (10,), got {predictions.shape}"

    def test_prediction_value_ranges(self, forecaster_with_model, sample_features):
        """
        WHAT: Verify predictions fall within reasonable ranges.
        HOW: Run inference, check outputs are finite and realistic.
        WHY: Catch numerical instability or model corruption.
        DATA: Predictions should be finite floats in reasonable price range.
        """
        # WHAT: Generate predictions from model
        predictions = forecaster_with_model.predict(sample_features)

        # WHAT: Verify no NaN or infinite values
        # WHY: Invalid predictions would crash downstream agents
        # HOW: Check numpy isfinite on all outputs
        assert np.all(np.isfinite(predictions)), "Predictions contain NaN or inf"

        # WHAT: Verify predictions in realistic range for stock prices
        # WHY: Catch model errors that produce nonsensical outputs
        # DATA: Stock prices typically in [0.01, 10000] range
        assert np.all(predictions > 0), "Predictions must be positive"
        assert np.all(predictions < 10000), "Predictions unrealistically high"

    @pytest.mark.integration
    def test_end_to_end_forecast(self, forecaster_with_model):
        """
        WHAT: Integration test with real feature engineering pipeline.
        HOW: Load recorded market data, engineer features, run forecast.
        WHY: Validates entire pipeline from raw data to prediction.
        DATA: Recorded AAPL data -> features -> forecast -> validated output.
        """
        # WHAT: Load recorded market data from fixtures
        # WHY: Reproducible test without live API calls
        # HOW: Read CSV fixture with pandas
        # DATA: fixtures/aapl_2024_01.csv -> DataFrame with OHLCV columns
        import pandas as pd
        fixture_data = Path(__file__).parent.parent / "fixtures" / "aapl_2024_01.csv"
        df = pd.read_csv(fixture_data)

        # WHAT: Engineer features from raw price data
        # HOW: Call feature engineering pipeline
        # DATA: OHLCV DataFrame -> feature matrix [n_samples, n_features]
        from pipelines.realtime.models.features import engineer_features
        features = engineer_features(df)

        # WHAT: Generate forecast using engineered features
        # DATA: features -> model.predict() -> price predictions
        predictions = forecaster_with_model.predict(features)

        # WHAT: Validate forecast output
        # WHY: Ensure end-to-end pipeline produces valid results
        assert len(predictions) == len(df), "Prediction count mismatch"
        assert np.all(np.isfinite(predictions)), "Pipeline produced invalid outputs"
```

#### Test Coverage Requirements
- **Unit tests**: Every public method in every class
- **Integration tests**: API endpoints, workflow execution, data pipelines
- **Fixtures**: Recorded API responses, sample datasets in `tests/fixtures/`
- **Coverage**: Minimum 80% line coverage, aim for 90%+

### Error Handling Standards

#### Fail Fast with Actionable Messages

```python
# ❌ BAD - Generic error, no guidance
def load_model(path: str):
    if not os.path.exists(path):
        raise Exception("Model not found")

# ✅ GOOD - Specific error with instructions
def load_model(path: str):
    """
    WHAT: Load trained forecasting model from disk.
    WHY: Fail immediately if model missing (no LLM fallback allowed).
    DATA: path (str) -> loaded model object.
    """
    # WHAT: Check if model file exists at specified path
    # WHY: Fail fast with actionable error message, not generic exception
    # HOW: Use pathlib for cross-platform path checking
    model_path = Path(path)
    if not model_path.exists():
        raise RuntimeError(
            f"Forecasting model not found at {path}.\n"
            f"Train the model first by running:\n"
            f"  python scripts/train_forecaster.py\n"
            f"Or specify a different model path via MODEL_PATH environment variable."
        )

    # WHAT: Attempt to deserialize model file
    # WHY: Catch corruption or version mismatch issues
    # DATA: pickle file -> deserialized sklearn model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except (pickle.PickleError, EOFError) as e:
        raise RuntimeError(
            f"Failed to load model from {path}. File may be corrupted.\n"
            f"Retrain the model with: python scripts/train_forecaster.py\n"
            f"Original error: {e}"
        )

    return model
```

#### No Silent Failures

```python
# ❌ BAD - Silently returns placeholder on error
def get_sentiment(text: str) -> float:
    try:
        return finbert.analyze(text)
    except Exception:
        return 0.0  # "neutral" fallback

# ✅ GOOD - Propagates error with context
def get_sentiment(text: str) -> float:
    """
    WHAT: Compute sentiment score using FinBERT model.
    WHY: Must fail loudly if FinBERT unavailable (no placeholder data).
    DATA: text (str) -> sentiment_score (float) in [-1, 1].
    """
    try:
        # WHAT: Run FinBERT inference on input text
        # DATA: text -> tokenized inputs -> model logits -> softmax probs -> score
        return self.finbert.analyze(text)
    except ModelNotFoundError as e:
        # WHAT: Re-raise with context about how to fix
        # WHY: Developer needs actionable guidance, not silent failure
        raise RuntimeError(
            f"FinBERT model not loaded. Download with:\n"
            f"  python scripts/download_finbert.py\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        # WHAT: Catch unexpected errors but don't hide them
        # WHY: Need to investigate root cause, not mask with placeholder
        raise RuntimeError(
            f"FinBERT inference failed on input text (length={len(text)}).\n"
            f"Error: {e}"
        ) from e
```

### Documentation Update Protocol

When modifying code, update corresponding documentation:

```
Modified File                          -> Update Documentation
======================================================================
pipelines/realtime/agents.py          -> docs/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md (Agent Architecture)
pipelines/realtime/models/forecaster.py -> docs/resources/ML_MODELS.md (if exists)
scripts/train_forecaster.py           -> docs/IMPLEMENTATION_STATUS.md (Training Pipeline)
pipelines/realtime/api.py              -> README.md (API Endpoints section)
New data source added                  -> docs/DATA_INVENTORY.md
```

#### Documentation Comment Template

When updating docs, add a comment block:

```markdown
<!-- Updated: 2024-01-15 by Agent -->
<!-- Change: Added ForecastAgent.predict() method documentation -->
<!-- Reason: New forecasting endpoint added to API -->

## Forecast Agent

The `ForecastAgent` generates probabilistic price predictions...
```

### LLM Usage Decision Tree

When writing code that might use an LLM, follow this decision tree:

```
Is this for generating natural language explanations?
├─ YES → ✅ OK to use OpenAI API (ExplanationAgent)
│         Example: Converting forecast + sentiment into readable insight
│
└─ NO → Is this for data analysis or prediction?
        ├─ YES → ❌ FORBIDDEN - Use ML model instead
        │         Example: Use FinBERT for sentiment, not GPT prompt
        │
        └─ NO → Is this a fallback for a failed operation?
                ├─ YES → ❌ FORBIDDEN - Fail fast with error
                │         Example: Don't prompt LLM if model missing
                │
                └─ NO → Is this for workflow orchestration (LangGraph)?
                        ├─ YES → ✅ OK to use LLM for routing/coordination
                        │
                        └─ NO → ❌ Default to FORBIDDEN unless explicitly approved
```

### Common Anti-Patterns to Avoid

#### ❌ Mock/Placeholder Data
```python
# WRONG
def get_fundamentals(ticker: str) -> Fundamentals:
    # TODO: Implement API call
    return Fundamentals(pe_ratio=15.0, eps=2.5)  # placeholder
```

#### ❌ LLM Fallback
```python
# WRONG
def predict_price(ticker: str) -> float:
    try:
        return self.model.predict(ticker)
    except ModelNotFoundError:
        prompt = f"Predict {ticker} price tomorrow"
        return float(llm.invoke(prompt))
```

#### ❌ God Class
```python
# WRONG - Does too many things
class StockAgent:
    def fetch_data(self): pass
    def train_model(self): pass
    def predict(self): pass
    def generate_explanation(self): pass
    def send_alert(self): pass
    def save_to_db(self): pass
```

#### ❌ Hardcoded Dependencies
```python
# WRONG - Can't swap implementation or test
class ForecastAgent:
    def __init__(self):
        self.data = AlphaVantageAdapter("hardcoded_key")
```

#### ❌ Missing Error Context
```python
# WRONG - Unhelpful error message
raise Exception("Failed")
```

#### ❌ Silent Failure
```python
# WRONG - Hides problems
try:
    result = risky_operation()
except Exception:
    result = default_value  # User won't know something failed
```

## Agent Workflow

### Standard Task Execution Flow

1. **Understand Request**
   - Read user's task description
   - Identify affected files/modules
   - Check for conflicts with project architecture

2. **Check Documentation**
   - Review relevant docs in `docs/`
   - Understand existing patterns
   - Note required updates

3. **Plan Implementation**
   - Break into discrete steps
   - Identify dependencies
   - Determine test strategy

4. **Implement with Comments**
   - Write code with inline WHAT/HOW/WHY/DATA comments
   - Follow SOLID principles
   - Use real implementations (no mocks)

5. **Write Tests**
   - Create unit tests for new functions
   - Add integration tests if needed
   - Ensure >80% coverage

6. **Update Documentation**
   - Modify relevant docs in `docs/`
   - Update README if public API changed
   - Add code examples if helpful

7. **Verify Quality**
   - Run tests: `pytest`
   - Check types: `mypy pipelines/`
   - Format code: `black pipelines/`
   - Lint: `ruff check pipelines/`

## Summary

As a development agent working on StockSense:

1. **Always explain code** with WHAT/HOW/WHY/DATA comments
2. **Follow SOLID** principles for maintainable architecture
3. **Test everything** you create (unit + integration)
4. **No placeholders** - only real, working implementations
5. **No LLM fallbacks** - fail fast with actionable errors
6. **Update docs** in `docs/` folder when changing code
7. **Challenge conflicts** - alert user if request violates architecture

These guidelines ensure generated code meets production quality standards and integrates seamlessly with the existing codebase.
