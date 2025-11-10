# FreshStart Test Suite

**Owner:** Byeol
**Last Updated:** 2025-11-10

## Overview

This directory contains the complete test suite for the FreshStart MVP, including unit tests, integration tests, and end-to-end tests. The test suite ensures quality assurance across all components developed by team members.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── fixtures/                   # Test data files
│   ├── sample_prices.json      # 30 days of realistic AAPL price data
│   ├── sample_news.json        # 10 realistic news articles with sentiment
│   └── expected_outputs.json   # Expected test results
├── unit/                       # Unit tests for individual components
│   ├── test_database.py        # Database operations (~20 tests)
│   ├── test_data_fetchers.py   # Price and news fetchers (~10 tests)
│   ├── test_prediction_model.py # Prediction models (~15 tests)
│   ├── test_sentiment_models.py # Sentiment models (~25 tests, placeholders)
│   └── test_agents.py          # Agent functionality (~15 tests)
├── integration/                # Integration tests
│   ├── test_vector_store.py    # ChromaDB integration (placeholders)
│   ├── test_api_endpoints.py   # FastAPI endpoints (placeholders)
│   └── test_workflow.py        # LangGraph workflow (placeholders)
└── e2e/                        # End-to-end tests
    └── test_full_analysis.py   # Full user workflows (placeholders)
```

## Running Tests

### Run All Tests
```bash
cd /home/user/capstone/FreshStart
pytest tests/ -v
```

### Run Specific Test Categories

**Unit Tests Only:**
```bash
pytest tests/unit/ -v
```

**Integration Tests Only:**
```bash
pytest tests/integration/ -v
```

**End-to-End Tests Only:**
```bash
pytest tests/e2e/ -v
```

**Specific Test File:**
```bash
pytest tests/unit/test_database.py -v
```

**Specific Test Function:**
```bash
pytest tests/unit/test_database.py::test_cache_prices -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

Coverage report will be generated in `htmlcov/index.html`.

### Skip Placeholder Tests

Many tests are placeholders waiting for other team members to complete their components:

```bash
pytest tests/ -v --ignore-skipped
```

## Test Categories

### Unit Tests (tests/unit/)

**test_database.py** - Tests Byeol's database implementation:
- Database initialization and schema
- Price caching and retrieval
- News caching and retrieval
- Analysis results storage
- Cache clearing and uniqueness constraints

**test_data_fetchers.py** - Tests Pam's and Tae's data fetchers:
- Price data fetching (Pam - IMPLEMENTED)
- News data fetching (Tae - PLACEHOLDER)
- API error handling
- Data format validation

**test_prediction_model.py** - Tests Pam's prediction models:
- LSTM model (PARTIAL - tests momentum fallback)
- GRU model (PLACEHOLDER)
- Gradient Boost model (PLACEHOLDER)
- Prediction ensemble (PLACEHOLDER)
- Base interface compliance

**test_sentiment_models.py** - Tests Tae's sentiment models:
- FinBERT model (PLACEHOLDER)
- RoBERTa model (PLACEHOLDER)
- VADER model (PLACEHOLDER)
- TextBlob model (PLACEHOLDER)
- Alpha Vantage API (PLACEHOLDER)
- Sentiment ensemble (PLACEHOLDER)

**test_agents.py** - Tests Josh's agents:
- PredictionAgent (PARTIAL)
- SentimentAgent (PLACEHOLDER)
- ReflectionAgent (PLACEHOLDER)
- ExplanationAgent (PLACEHOLDER)

### Integration Tests (tests/integration/)

**test_vector_store.py** - Tests Tae's ChromaDB integration:
- Vector store initialization (PLACEHOLDER)
- Document storage and retrieval (PLACEHOLDER)
- Semantic search (PLACEHOLDER)
- Integration with ExplanationAgent (PLACEHOLDER)

**test_api_endpoints.py** - Tests Sua's FastAPI backend:
- /analyze endpoint (PLACEHOLDER)
- Request/response validation (PLACEHOLDER)
- CORS configuration (PLACEHOLDER)
- Workflow integration (PLACEHOLDER)

**test_workflow.py** - Tests Josh's LangGraph workflow:
- Workflow creation (PARTIAL)
- State management (PLACEHOLDER)
- Data flow through nodes (PLACEHOLDER)
- Error handling (PLACEHOLDER)

### End-to-End Tests (tests/e2e/)

**test_full_analysis.py** - Tests complete user workflows:
- Frontend → API → Workflow → Results (PLACEHOLDER)
- Caching benefits (PLACEHOLDER)
- Error scenarios (PLACEHOLDER)
- Multiple stock analyses (PLACEHOLDER)

## Test Fixtures

### conftest.py Fixtures

**test_db** - Temporary SQLite database for testing
```python
def test_example(test_db):
    test_db.cache_prices("AAPL", price_data)
```

**sample_price_data** - Pandas DataFrame with 30 days of price data
```python
def test_example(sample_price_data):
    assert len(sample_price_data) == 30
```

**sample_news_data** - List of 10 news articles with sentiment
```python
def test_example(sample_news_data):
    assert len(sample_news_data) == 10
```

**sample_analysis_result** - Complete analysis result structure
**mock_ticker** - Default test ticker ("AAPL")
**expected_outputs** - Expected results for validation

## Writing New Tests

### Test Naming Convention
- Test files: `test_<component>.py`
- Test classes: `Test<Component>` (e.g., `TestDatabase`)
- Test functions: `test_<action>_<scenario>` (e.g., `test_cache_prices_with_empty_dataframe`)

### Example Unit Test
```python
def test_cache_prices(test_db, sample_price_data, mock_ticker):
    """Test caching price data."""
    inserted = test_db.cache_prices(mock_ticker, sample_price_data)
    assert inserted > 0
    assert inserted == len(sample_price_data)
```

### Example Integration Test
```python
@patch('coordinator.workflow.get_historical_data')
def test_workflow_fetches_data(mock_get_data, sample_price_data):
    """Test workflow calls data fetchers."""
    mock_get_data.return_value = sample_price_data
    # Test workflow execution
```

## Test Coverage Goals

**Current Status:**
- Database: ~90% coverage ✅
- Data Fetchers: ~60% coverage (waiting for news fetcher)
- Prediction Models: ~40% coverage (waiting for trained models)
- Sentiment Models: 0% coverage (waiting for Tae)
- Agents: ~30% coverage (partial implementation)
- Integration: ~10% coverage (placeholders)
- E2E: 0% coverage (placeholders)

**Target Coverage:** >80% across all modules

## Continuous Integration

Tests should be run:
1. Before committing code
2. Before creating pull requests
3. During CI/CD pipeline (if configured)

## Debugging Failed Tests

**Run with verbose output:**
```bash
pytest tests/unit/test_database.py -vv
```

**Run with print statements:**
```bash
pytest tests/unit/test_database.py -s
```

**Run and drop into debugger on failure:**
```bash
pytest tests/unit/test_database.py --pdb
```

**Run specific test that's failing:**
```bash
pytest tests/unit/test_database.py::test_cache_prices -vv
```

## Updating Tests

When team members complete their components:

1. **Tae completes sentiment models:**
   - Remove `pytest.skip()` from `test_sentiment_models.py`
   - Add actual assertions based on implemented models

2. **Pam completes trained models:**
   - Update `test_prediction_model.py` with real model tests
   - Add ensemble tests when ensemble is implemented

3. **Josh completes agents:**
   - Remove placeholders from `test_agents.py`
   - Add integration tests for workflow

4. **Sua completes API:**
   - Implement `test_api_endpoints.py` tests
   - Add E2E tests with real API calls

## Test Data Philosophy

**No Mock/Synthetic Data in Production:**
- Test fixtures use realistic data (actual AAPL prices, real news articles)
- Unit tests mock external APIs to avoid rate limits
- Integration tests may use real APIs sparingly
- E2E tests use cached data when possible

## Contact

**Questions about tests?** Ask Byeol
**Component not tested?** Coordinate with component owner
**Test failing?** Check component implementation first, then test logic

## References

- Pytest Documentation: https://docs.pytest.org/
- Test Coverage: https://coverage.readthedocs.io/
- ROLE_DIVISION.md: Team member responsibilities
