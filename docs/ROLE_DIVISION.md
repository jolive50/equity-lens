# FreshStart MVP - Role Division & Responsibility Assignment

## Team Members
- **Tae**: Sentiment Models (FinBERT, RoBERTa, Alpha Vantage API Sentiment, TextBlob, VADER), SentimentEnsemble, ChromaDB, News Data Fetcher
- **Pam**: Prediction Models (LSTM, GRU, Gradient Boost), PredictionEnsemble, Price Data Fetcher, Kaggle SP500 Training Data
- **Sua**: Next.js Frontend, FastAPI Backend
- **Byeol**: SQLite Database, All Testing (Unit, Integration, E2E)
- **Josh**: LangChain Agents, Coordinator Workflow, Configuration System (model selection/loading)
- **Documentation**: SHARED - All team members contribute

---

## Directory Structure with Assignments

```
.
├── README.md                           # [SHARED] - Quick start guide
├── requirements.txt                    # [SHARED] - Dependencies (including ChromaDB, LangChain)
│
├── data/
│   ├── raw/                            # [PAM] - Kaggle SP500 dataset for LSTM training
│   ├── processed/                      # [SHARED] - Cached/processed data
│   └── fetchers/
│       ├── __init__.py                 # [SHARED]
│       ├── price_data.py               # [PAM] - yfinance wrapper for real-time data (~80 lines)
│       └── news_data.py                # [TAE] - Alpha Vantage news API (~100 lines)
│
├── models/
│   ├── __init__.py                     # [SHARED]
│   ├── prediction/                     # [PAM]
│   │   ├── __init__.py                 # [PAM]
│   │   ├── base_predictor.py           # [PAM] - Base class for all prediction models (~80 lines)
│   │   ├── lstm_model.py               # [PAM] - LSTM model implementation (~200 lines)
│   │   ├── gru_model.py                # [PAM] - GRU model implementation (~180 lines)
│   │   ├── gradient_boost_model.py     # [PAM] - Gradient boosting model (~150 lines)
│   │   ├── ensemble.py                 # [PAM] - Ensemble 2+ models or use individually (~150 lines)
│   │   ├── train_models.py             # [PAM] - Training script for all models (~300 lines)
│   │   └── saved_models/               # [PAM] - Trained weights for all models
│   │       ├── lstm/                   # [PAM] - LSTM weights
│   │       ├── gru/                    # [PAM] - GRU weights
│   │       └── gradient_boost/         # [PAM] - GB weights
│   │
│   └── sentiment/                      # [TAE]
│       ├── __init__.py                 # [TAE]
│       ├── base_sentiment.py           # [TAE] - Base class for all sentiment models (~60 lines)
│       ├── finbert_model.py            # [TAE] - FinBERT financial sentiment (~120 lines)
│       ├── roberta_model.py            # [TAE] - RoBERTa sentiment model (~100 lines)
│       ├── alpha_vantage_sentiment.py  # [TAE] - Alpha Vantage sentiment API (~80 lines)
│       ├── vader_model.py              # [TAE] - NLTK VADER wrapper (~40 lines)
│       ├── textblob_model.py           # [TAE] - TextBlob wrapper (~30 lines)
│       └── ensemble.py                 # [TAE] - Ensemble 2+ models or use individually (~150 lines)
│
├── agents/                             # [JOSH]
│   ├── __init__.py                     # [JOSH]
│   ├── prediction_agent.py             # [JOSH] - Simplified from current agents.py (~100 lines)
│   ├── sentiment_agent.py              # [JOSH] - Simplified from current agents.py (~120 lines)
│   ├── reflection_agent.py             # [JOSH] - Extract from reflection.py (~150 lines)
│   └── explanation_agent.py            # [JOSH] - Simplified from current agents.py (~100 lines)
│
├── coordinator/                        # [JOSH]
│   ├── __init__.py                     # [JOSH]
│   ├── workflow.py                     # [JOSH] - Simplified LangGraph workflow (~150 lines)
│   └── config.py                       # [JOSH] - Model configuration system (~100 lines)
│
├── storage/
│   ├── __init__.py                     # [SHARED]
│   ├── vector_store.py                 # [TAE] - Simplified ChromaDB integration (~120 lines)
│   └── database.py                     # [BYEOL] - SQLite schema + basic queries (~150 lines)
│
├── api/                                # [SUA]
│   ├── __init__.py                     # [SUA]
│   ├── main.py                         # [SUA] - FastAPI server (~150 lines)
│   └── models.py                       # [SUA] - Pydantic models for request/response (~80 lines)
│
├── frontend/                           # [SUA]
│   ├── README.md                       # [SUA]
│   ├── package.json                    # [SUA] - Next.js dependencies
│   ├── next.config.js                  # [SUA] - Next.js configuration
│   ├── tsconfig.json                   # [SUA] - TypeScript configuration
│   └── src/
│       ├── app/
│       │   ├── layout.tsx              # [SUA] - Root layout (~20 lines)
│       │   ├── page.tsx                # [SUA] - Main page (~60 lines)
│       │   └── globals.css             # [SUA] - Global styles (~200 lines)
│       └── components/
│           ├── AnalysisForm.tsx        # [SUA] - Input form (~40 lines)
│           └── ResultsDisplay.tsx      # [SUA] - Results display (~80 lines)
│
├── tests/                              # [BYEOL] - All testing
│   ├── __init__.py                     # [BYEOL]
│   ├── unit/                           # [BYEOL] - Unit tests for all components
│   │   ├── __init__.py                 # [BYEOL]
│   │   ├── test_prediction_model.py    # [BYEOL] - Test Pam's LSTM (~100 lines)
│   │   ├── test_sentiment_models.py    # [BYEOL] - Test Tae's sentiment (~150 lines)
│   │   ├── test_agents.py              # [BYEOL] - Test Josh's agents (~200 lines)
│   │   ├── test_data_fetchers.py       # [BYEOL] - Test price/news fetchers (~100 lines)
│   │   └── test_database.py            # [BYEOL] - Test database operations (~80 lines)
│   ├── integration/                    # [BYEOL] - Integration tests
│   │   ├── __init__.py                 # [BYEOL]
│   │   ├── test_workflow.py            # [BYEOL] - Test full LangGraph workflow (~150 lines)
│   │   ├── test_api_endpoints.py       # [BYEOL] - Test FastAPI endpoints (~100 lines)
│   │   └── test_vector_store.py        # [BYEOL] - Test ChromaDB integration (~80 lines)
│   ├── e2e/                            # [BYEOL] - End-to-end tests
│   │   ├── __init__.py                 # [BYEOL]
│   │   └── test_full_analysis.py       # [BYEOL] - Full user workflow test (~120 lines)
│   ├── fixtures/                       # [BYEOL] - Test data (real fetched data for testing)
│   │   ├── sample_prices.json          # [BYEOL] - Real price data samples
│   │   ├── sample_news.json            # [BYEOL] - Real news article samples
│   │   └── expected_outputs.json       # [BYEOL] - Expected test results
│   ├── conftest.py                     # [BYEOL] - Pytest configuration and fixtures (~100 lines)
│   └── README.md                       # [BYEOL] - Testing documentation
│
├── docs/                               # [SHARED] - All documentation
│   ├── ROLE_DIVISION.md                # [SHARED] - This file (team responsibilities)
│
└── db/                                 # [BYEOL]
    └── schema.sql                      # [BYEOL] - SQLite table definitions (~50 lines)
```

---

## Detailed Responsibility Breakdown

### TAE (Sentiment & News)
**Total Estimated LOC: ~690 lines**

#### Primary Responsibilities:

1. **News Data Fetcher** (`data/fetchers/news_data.py`)
   - Fetch news from Alpha Vantage NEWS_SENTIMENT endpoint
   - Parse and structure news articles
   - Handle API rate limits and errors

2. **Base Sentiment Model** (`models/sentiment/base_sentiment.py`)
   - Abstract base class defining sentiment model interface
   - Standardized methods: `analyze(text) -> SentimentResult`
   - Ensures all sentiment models follow same contract
   - Enables polymorphic use in ensemble

3. **FinBERT Sentiment Model** (`models/sentiment/finbert_model.py`)
   - Implement FinBERT for financial text sentiment analysis
   - Inherit from `BaseSentimentModel`
   - Return standardized `SentimentResult` format
   - Handle model loading and inference

4. **RoBERTa Sentiment Model** (`models/sentiment/roberta_model.py`)
   - Implement RoBERTa sentiment analysis model
   - Inherit from `BaseSentimentModel`
   - Return standardized `SentimentResult` format
   - Handle model inference and preprocessing

5. **Alpha Vantage Sentiment API Wrapper** (`models/sentiment/alpha_vantage_sentiment.py`)
   - **NOTE**: Alpha Vantage is an API (not an ML model) that provides sentiment analysis
   - Integrate Alpha Vantage sentiment API
   - Inherit from `BaseSentimentModel` for consistent interface
   - Extract sentiment scores from API responses
   - Handle API rate limits

6. **VADER Sentiment Model** (`models/sentiment/vader_model.py`)
   - Implement NLTK VADER wrapper for rule-based sentiment
   - Inherit from `BaseSentimentModel`
   - Return standardized `SentimentResult` format

7. **TextBlob Sentiment Model** (`models/sentiment/textblob_model.py`)
   - Implement TextBlob wrapper for lexicon-based sentiment
   - Inherit from `BaseSentimentModel`
   - Return standardized `SentimentResult` format

8. **Sentiment Ensemble** (`models/sentiment/ensemble.py`)
   - **Tae implements SentimentEnsemble class**
   - **FLEXIBLE ARCHITECTURE**: Accepts 2+ sentiment models as input
   - Supports multiple ensemble strategies: weighted voting, averaging, majority vote
   - Can use any combination of models (e.g., just FinBERT + RoBERTa, or all 5)
   - Returns aggregated sentiment with per-model breakdown
   - Configuration-driven model selection
   - **NOTE**: Do not begin ensemble work until at least 2 sentiment models are complete and working

9. **ChromaDB Vector Store** (`storage/vector_store.py`)
   - Simplified ChromaDB wrapper for news embeddings
   - Store and retrieve news articles using semantic search
   - Enable explanation agent to find relevant context

#### Key Design Features:
- **Individual Model Usage**: Each sentiment model can be used standalone via `BaseSentimentModel` interface
- **Flexible Ensemble**: Ensemble accepts list of models at runtime, not hardcoded to 5 models
- **Configurable**: Model selection controlled via configuration file or runtime parameters

#### Dependencies:
- Works with: Josh (agents will use these models individually or as ensemble)
- Provides data to: SentimentAgent via coordinator workflow

---

### PAM (Prediction & Price Data)
**Total Estimated LOC: ~980 lines**

#### Primary Responsibilities:

1. **Kaggle SP500 Training Data** (`data/raw/`)
   - Download and organize Kaggle SP500 historical dataset (Pam chooses specific dataset)
   - Prefer datasets with more dates, rows, and features (can trim down as needed)
   - Preprocess data for training all prediction models
   - Document data sources and preprocessing steps
   - Create train/validation/test splits

2. **Price Data Fetcher** (`data/fetchers/price_data.py`)
   - Fetch real-time price data using yfinance
   - Return standardized DataFrame format
   - Handle invalid tickers and API errors

3. **Base Prediction Model** (`models/prediction/base_predictor.py`)
   - Abstract base class defining prediction model interface
   - Standardized methods: `predict(data) -> PredictionResult`
   - Ensures all prediction models follow same contract
   - Enables polymorphic use in ensemble

4. **LSTM Model** (`models/prediction/lstm_model.py`)
   - Implement LSTM neural network for stock prediction
   - Inherit from `BasePredictionModel`
   - Return standardized `PredictionResult` format
   - Design architecture (input features, hidden layers, output)

5. **GRU Model** (`models/prediction/gru_model.py`)
   - Implement GRU (Gated Recurrent Unit) neural network
   - Inherit from `BasePredictionModel`
   - Alternative to LSTM with potentially faster training
   - Return standardized `PredictionResult` format

6. **Gradient Boosting Model** (`models/prediction/gradient_boost_model.py`)
   - Implement gradient boosting (XGBoost or LightGBM - Pam's choice) for prediction
   - Inherit from `BasePredictionModel`
   - Non-neural network approach for comparison
   - Return standardized `PredictionResult` format

7. **Prediction Ensemble** (`models/prediction/ensemble.py`)
   - **Pam implements PredictionEnsemble class**
   - **FLEXIBLE ARCHITECTURE**: Accepts 2+ prediction models as input
   - Supports multiple ensemble strategies: weighted averaging, voting, stacking
   - Can use any combination of models (e.g., just LSTM + GRU, or all models)
   - Returns aggregated prediction with per-model breakdown
   - Configuration-driven model selection
   - **NOTE**: Do not begin ensemble work until at least 2 prediction models are complete and working

8. **Model Training Script** (`models/prediction/train_models.py`)
   - Train all prediction models on Kaggle SP500 dataset
   - Implement training pipeline with validation for each model
   - Save trained weights to respective `saved_models/` directories
   - Compare model performance and log metrics

9. **Model Weights Management** (`models/prediction/saved_models/`)
   - Organize trained model files by type (LSTM, GRU, Gradient Boost)
   - **Commit model weights to git** for other team members to use
   - Document model versioning and performance metrics
   - Store model configurations and hyperparameters

#### Key Design Features:
- **Individual Model Usage**: Each prediction model can be used standalone via `BasePredictionModel` interface
- **Flexible Ensemble**: Ensemble accepts list of models at runtime, not hardcoded to specific models
- **Configurable**: Model selection controlled via configuration file or runtime parameters
- **Performance Tracking**: All models log metrics for comparison and selection

#### Dependencies:
- Works with: Josh (PredictionAgent will use models individually or as ensemble)
- Provides data to: PredictionAgent via coordinator workflow

---

### SUA (Next.js Frontend & FastAPI Backend)
**Total Estimated LOC: ~600 lines**

#### Primary Responsibilities:

**Backend (FastAPI):**
1. **FastAPI Server** (`api/main.py`)
   - Create single `/analyze` endpoint
   - Handle CORS for Next.js frontend access
   - Integrate with LangGraph workflow
   - Error handling and HTTP responses

2. **API Models** (`api/models.py`)
   - Define Pydantic request models (AnalysisRequest)
   - Define Pydantic response models (AnalysisResponse)
   - Input validation schemas

**Frontend (Next.js):**
3. **Next.js App Layout** (`frontend/src/app/layout.tsx`)
   - Root layout component with Next.js 13+ App Router
   - Metadata configuration
   - Global app structure

4. **Main Page** (`frontend/src/app/page.tsx`)
   - Main analysis page with React state management
   - API integration with FastAPI backend
   - Loading and error states

5. **Analysis Form Component** (`frontend/src/components/AnalysisForm.tsx`)
   - Ticker input form with validation
   - Form submission handling
   - User input sanitization

6. **Results Display Component** (`frontend/src/components/ResultsDisplay.tsx`)
   - Display LSTM prediction results
   - Show ensemble sentiment analysis
   - Render explanation text
   - Optional reflection warnings

7. **Global Styles** (`frontend/src/app/globals.css`)
   - Professional Tailwind CSS or custom styling
   - Responsive layout for mobile/desktop
   - Color coding for predictions (up/down/neutral)
   - Loading spinner animations

#### Dependencies:
- Works with: Josh (coordinator workflow), All team members (integrates all components)
- Consumes: All backend services

---

### BYEOL (SQLite Database & Testing)
**Total Estimated LOC: ~1,600 lines**

#### Primary Responsibilities:

**1. Database Management**
1. **Database Schema** (`db/schema.sql`)
   - Design tables for price cache, news cache, analysis results
   - Create indexes for performance
   - Document schema structure

2. **Database Wrapper** (`storage/database.py`)
   - SQLite connection management
   - CRUD operations for caching
   - Query methods for retrieving cached data
   - Save and retrieve analysis results

**Key Tables:**
- `price_cache`: **Store ALL fetched price data** to avoid wasting API calls refetching
- `news_cache`: **Store ALL fetched news articles** to avoid wasting API calls refetching
- `analysis_results`: Persist completed analyses for history

**Cache Strategy:**
- Store everything we fetch (no expiration/TTL for now)
- Don't worry about stale data in MVP phase
- Focus on avoiding redundant API calls

**2. Complete Testing Suite**
1. **Unit Tests** (`tests/unit/`)
   - `test_prediction_model.py`: Test Pam's LSTM model with fixture data (~100 lines)
   - `test_sentiment_models.py`: Test all of Tae's sentiment models (FinBERT, RoBERTa, Alpha Vantage, VADER, TextBlob, ensemble) (~150 lines)
   - `test_agents.py`: Test all of Josh's agents (prediction, sentiment, reflection, explanation) (~200 lines)
   - `test_data_fetchers.py`: Test price and news data fetchers with mocked API responses (~100 lines)
   - `test_database.py`: Test SQLite operations, caching, CRUD operations (~80 lines)

2. **Integration Tests** (`tests/integration/`)
   - `test_workflow.py`: Test full LangGraph workflow end-to-end (~150 lines)
   - `test_api_endpoints.py`: Test FastAPI endpoints with real coordinator (~100 lines)
   - `test_vector_store.py`: Test ChromaDB integration and semantic search (~80 lines)

3. **End-to-End Tests** (`tests/e2e/`)
   - `test_full_analysis.py`: Simulate full user workflow from Next.js frontend to results (~120 lines)

4. **Test Infrastructure** (`tests/`)
   - `conftest.py`: Pytest fixtures, test database setup (~100 lines)
   - `fixtures/`: Real sample data files for testing (JSON files with actual fetched price/news/expected outputs - no mock data)
   - `README.md`: Testing documentation with instructions for running tests

#### Dependencies:
- Works with: ALL team members (tests all components)
- Tests code from: Tae, Pam, Sua, Josh
- Provides: Database infrastructure and quality assurance

---

### JOSH (Agents & Orchestration)
**Total Estimated LOC: ~820 lines**

#### Primary Responsibilities:

1. **Prediction Agent** (`agents/prediction_agent.py`)
   - Simplify from current agents.py
   - **FLEXIBLE MODEL INTERFACE**: Accept any `BasePredictionModel` (single or ensemble)
   - Configure to use individual models or ensemble at runtime
   - Return structured prediction results with model configuration metadata

2. **Sentiment Agent** (`agents/sentiment_agent.py`)
   - Simplify from current agents.py
   - **FLEXIBLE MODEL INTERFACE**: Accept any `BaseSentimentModel` (single or ensemble)
   - Configure to use individual models or ensemble at runtime
   - Return structured sentiment results with individual model breakdown (if ensemble)

3. **Reflection Agent** (`agents/reflection_agent.py`)
   - Extract from existing reflection.py
   - **Primary Purpose**: Compare predictions to real results the next day to update models
   - Reward models when predictions are correct (increase confidence/weight)
   - Punish models when predictions are wrong (decrease confidence/weight)
   - Validate prediction and sentiment alignment
   - Check data freshness and quality
   - Validate ensemble consistency (if multiple models used)

4. **Explanation Agent** (`agents/explanation_agent.py`)
   - Generate natural language explanations
   - Use ChromaDB for semantic context retrieval
   - LLM integration for text generation (OpenAI API)
   - **Note**: OpenAI API usage is for private development/prototype only (team use only)
   - Explain model selection reasoning when ensemble is used

5. **LangGraph Workflow** (`coordinator/workflow.py`)
   - Simplified linear workflow (vs complex state graph)
   - Define nodes: fetch_data → predict → sentiment → reflect → explain
   - State management between agents
   - Error handling and flow control
   - **Model configuration management**: Load and initialize selected models

6. **Model Configuration System** (`coordinator/config.py` - NEW)
   - **Josh implements config loading logic using PyYAML**
   - Configuration file or runtime parameters for model selection
   - Define which prediction models to use (e.g., LSTM only, LSTM+GRU, all 3)
   - Define which sentiment models to use (e.g., FinBERT only, FinBERT+RoBERTa, all 5)
   - Set ensemble strategies and model weights
   - Enable easy experimentation with different model combinations

#### Workflow Flow:
```
User Request (ticker) via Next.js Frontend
    ↓
FastAPI Backend (Sua)
    ↓
fetch_data (calls Tae's news_data + Pam's price_data)
    - If API down/rate limit: Return error message explaining why (out of our control)
    - If code error: Fix and retry
    ↓
predict (calls Pam's LSTM model via PredictionAgent)
    - At least 1 prediction model must work
    - Console log which models fail to load for debugging
    ↓
sentiment (calls Tae's sentiment models via SentimentAgent)
    - At least 1 sentiment model must work
    - Console log which models fail to load for debugging
    ↓
reflect (validates results via ReflectionAgent)
    ↓
explain (generates explanation via ExplanationAgent + ChromaDB)
    ↓
Return to FastAPI → Next.js Frontend
```

#### Key Design Features:
- **Configuration-Driven**: Model selection happens at initialization, not hardcoded
- **Polymorphic Design**: Agents work with any model implementing the base interface
- **Runtime Flexibility**: Can switch between single models and ensembles without code changes
- **Experimentation Support**: Easy A/B testing of different model combinations

#### Dependencies:
- Works with: ALL team members (orchestrates entire system)
- Integrates: Tae's flexible sentiment system (1+ models), Pam's flexible prediction system (1+ models), Byeol's database, Sua's FastAPI + Next.js
- Configuration: Manages which models are active for each analysis request

---

## Integration Points

### Critical Handoffs:

1. **Pam → Josh (Prediction Models)**
   - **Individual Model Interface**: All prediction models inherit from `BasePredictionModel`
   - **Standard Interface**: `predict(data) -> PredictionResult`
   - **Return Format**: `PredictionResult` schema TBD (Pam will define exact format when implementing)
   - **Ensemble Interface**: `PredictionEnsemble` accepts list of `BasePredictionModel` instances
   - **Configuration**: PredictionAgent can use single model or ensemble based on config
   - **Flexibility**: Josh can configure to use LSTM only, LSTM+GRU, or any combination
   - **Note**: Josh will wait for Pam to provide models before finalizing agent implementation

2. **Tae → Josh (Sentiment Models)**
   - **Individual Model Interface**: All sentiment models inherit from `BaseSentimentModel`
   - **Standard Interface**: `analyze(text) -> SentimentResult`
   - **Return Format**: `SentimentResult` schema TBD (Tae will define exact format when implementing)
   - **Ensemble Interface**: `SentimentEnsemble` accepts list of `BaseSentimentModel` instances
   - **Configuration**: SentimentAgent can use single model or ensemble based on config
   - **Flexibility**: Josh can configure to use FinBERT only, FinBERT+RoBERTa, or any combination
   - **Ensemble Return**: When using ensemble, includes breakdown with per-model results
   - **Note**: Josh will wait for Tae to provide models before finalizing agent implementation

3. **Josh → Sua**
   - Workflow final state must match API response model
   - Ensure all required fields populated
   - Include model configuration info (which models were used)

4. **Byeol → Tae & Pam**
   - Database methods for caching must align with fetcher needs
   - Provide methods: `cache_prices()`, `get_cached_prices()`, `cache_news()`, `get_cached_news()`
   - Cache individual model results for performance comparison

5. **Tae → Josh (Explanation Agent)**
   - ChromaDB vector_store interface for semantic search
   - Method: `search_similar(query, ticker, n_results)`

### Model Configuration Examples:

**Example 1: Use single LSTM model**
```python
# In PredictionAgent configuration
prediction_model = LSTMModel(weights_path="saved_models/lstm/")
```

**Example 2: Use LSTM + GRU ensemble**
```python
# In PredictionAgent configuration
models = [
    LSTMModel(weights_path="saved_models/lstm/"),
    GRUModel(weights_path="saved_models/gru/")
]
prediction_model = PredictionEnsemble(models=models, strategy="weighted_average")
```

**Example 3: Use all 3 prediction models**
```python
# In PredictionAgent configuration
models = [
    LSTMModel(weights_path="saved_models/lstm/", weight=0.4),
    GRUModel(weights_path="saved_models/gru/", weight=0.4),
    GradientBoostModel(weights_path="saved_models/gb/", weight=0.2)
]
prediction_model = PredictionEnsemble(models=models, strategy="weighted_average")
```

**Example 4: Use single FinBERT model**
```python
# In SentimentAgent configuration
sentiment_model = FinBERTModel()
```

**Example 5: Use FinBERT + RoBERTa + VADER ensemble**
```python
# In SentimentAgent configuration
models = [
    FinBERTModel(weight=0.5),
    RoBERTaModel(weight=0.3),
    VADERModel(weight=0.2)
]
sentiment_model = SentimentEnsemble(models=models, strategy="weighted_average")
```

---

## Estimated Line Counts by Team Member

| Team Member | Components | Estimated LOC | Percentage |
|-------------|------------|---------------|------------|
| **Tae**     | Base class + 5 Sentiment models + Ensemble, News fetcher, ChromaDB | ~690 lines | 15% |
| **Pam**     | Base class + 3 Prediction models + Ensemble, Price fetcher, Training scripts | ~980 lines | 21% |
| **Sua**     | Next.js Frontend + FastAPI Backend | ~600 lines | 13% |
| **Byeol**   | SQLite database + All Testing (now includes ensemble testing) | ~1,600 lines | 34% |
| **Josh**    | Agents + LangGraph Coordinator + Model Configuration System | ~820 lines | 17% |

**Total: ~4,690 lines** (includes comprehensive test suite + ensemble architecture)

---

## Development Workflow

### Recommended Development Approach:

**Foundation Work** (Start together):
- Byeol: Database schema and test infrastructure
- Tae: ChromaDB setup and news data fetcher
- Pam: Price data fetcher and Kaggle SP500 data preparation
- Sua: Can start with frontend mockups/UI design
- Josh: Can design workflow architecture

**Model Development** (Work independently):
- Pam: Train prediction models (LSTM, GRU, Gradient Boost) as ready
- Tae: Implement sentiment models (FinBERT, RoBERTa, etc.) as ready
- **Ensemble Note**: Don't start ensemble work until at least 2 models are complete and working
- Byeol: Write tests for completed components as they become available

**Integration Work** (Coordinate as components become ready):
- Josh: Build agents and workflow when models are available (at least 1 prediction + 1 sentiment model)
- Josh: Console log which models fail to load for debugging
- Sua: Build FastAPI backend when workflow is available
- Sua: Build Next.js frontend (can work in parallel with backend)
- Byeol: Integration and E2E tests when full system is available

**Flexibility Notes:**
- Team members can work in parallel where possible
- Don't wait for everything to be perfect before integrating
- Focus on getting at least 1 working model of each type first
- Ensemble is a "plus" feature - get single models working first

---

## Testing Responsibilities

### Unit Testing (BYEOL)
- Write unit tests for all components developed by team members
- Test Pam's prediction model (LSTM)
- Test Tae's sentiment models (FinBERT, VADER, TextBlob, ensemble)
- Test Josh's agents (prediction, sentiment, reflection, explanation)
- Test Sua's API endpoints and Pydantic models
- Test data fetchers (price and news)
- Test database operations

### Integration Testing (BYEOL)
- Test full LangGraph workflow integration
- Test API endpoints with real coordinator
- Test ChromaDB vector store integration
- Test interactions between components

### End-to-End Testing (BYEOL)
- Test complete user workflows from frontend to backend
- Simulate real user scenarios
- Verify system works correctly as a whole

### Test Infrastructure (BYEOL)
- Set up pytest configuration and fixtures
- Create mock data and API responses
- Maintain test fixtures (sample prices, news, expected outputs)
- Document testing procedures in TESTING_GUIDE.md
- Ensure >80% test coverage across all modules

### Quality Assurance (BYEOL)
- Run test suite before all team integrations
- Identify bugs and report to respective team members
- Verify fixes with regression tests
- Maintain test documentation

### Collaboration on Testing
While Byeol writes and maintains all tests, team members should:
- Provide example data for their components
- Explain expected behavior of their code
- Fix bugs identified by tests
- Review test cases for their components

---

## Communication Protocols

### Daily Sync:
- Share progress on assigned components
- Identify blockers or dependencies
- Coordinate integration points

### Code Review:
- Cross-review within integration pairs
- Josh reviews all agent integrations
- Sua reviews API contracts

### Integration Meetings:
- **Weekly check-ins**: Share progress, coordinate integrations, identify blockers
- **When models are ready**: Model interfaces review + Byeol presents unit test results
- **When workflow is integrated**: Full workflow integration test + Byeol runs integration test suite
- **When frontend/backend connected**: Frontend-backend integration test + Byeol runs E2E test suite
- **Before major integrations**: Byeol runs relevant test suite and reports results

---

## Questions or Issues?

If you have questions about responsibilities or need clarification on interfaces:
1. Check this document first
2. Discuss in team channel
3. Update this document with decisions

**Last Updated**: 2025-11-01

**Key Technologies:**
- **Tae**: FinBERT, RoBERTa, Alpha Vantage API, TextBlob, VADER (5 sentiment models) + SentimentEnsemble + ChromaDB (vector store)
- **Pam**: LSTM, GRU, Gradient Boosting (XGBoost/LightGBM) trained on Kaggle SP500 dataset + PredictionEnsemble
- **Sua**: Next.js (frontend) + FastAPI (backend)
- **Byeol**: SQLite + pytest (comprehensive testing including ensemble scenarios)
- **Josh**: LangChain + LangGraph (agent orchestration) + Model Configuration System + PyYAML (config loading)
- **Documentation**: SHARED responsibility across all team members

---

## Ensemble Architecture Design Summary

### Key Design Decisions

#### 1. **Flexible Model Selection (2+ Models)**
Both sentiment and prediction systems support using:
- **Single model**: For simplicity, speed, or when one model performs best
- **Subset ensemble**: 2-3 models for balanced performance and speed
- **Full ensemble**: All available models for maximum accuracy

#### 2. **Base Class Pattern**
- All models inherit from abstract base classes (`BaseSentimentModel`, `BasePredictionModel`)
- Ensures consistent interfaces across all models
- Enables polymorphic usage in agents
- Makes testing easier (can mock base class)

#### 3. **Configuration-Driven Architecture**
Models are selected via configuration, not hardcoded:
```python
# config.yaml example
prediction:
  mode: "ensemble"  # or "single"
  models: ["lstm", "gru"]  # or ["lstm", "gru", "gradient_boost"]
  strategy: "weighted_average"
  weights:
    lstm: 0.5
    gru: 0.5

sentiment:
  mode: "ensemble"  # or "single"
  models: ["finbert", "roberta", "vader"]  # or all 5
  strategy: "weighted_average"
  weights:
    finbert: 0.5
    roberta: 0.3
    vader: 0.2
```

#### 4. **Ensemble Strategies**
Multiple combination strategies available:
- **Weighted Average**: Combine scores with predefined weights
- **Simple Average**: Equal weight to all models
- **Voting**: Majority vote for classification
- **Stacking**: Meta-model learns optimal combination (advanced)

#### 5. **Benefits of This Architecture**

**For Development:**
- **Tae** can develop sentiment models independently and test them individually
- **Pam** can develop prediction models independently and test them individually
- **Josh** can experiment with different combinations without changing model code
- **Byeol** can test each model in isolation and in combination

**For Performance:**
- Compare individual model performance vs ensemble
- Identify which models contribute most to accuracy
- Remove underperforming models easily
- A/B test different ensemble strategies

**For Deployment:**
- Start with single best-performing model for speed
- Add ensemble when accuracy is more important than latency
- Dynamically adjust based on user preferences or ticker characteristics
- Fall back to single model if ensemble fails

**For Maintenance:**
- Add new models without modifying existing code
- Replace underperforming models easily
- Update model weights based on production metrics
- Experiment with new architectures safely

#### 6. **Team Workflow**

**Phase 1: Individual Development**
- Tae develops each sentiment model independently
- Pam develops each prediction model independently
- Each model tested in isolation

**Phase 2: Integration**
- Josh creates configuration system
- Models integrated into agents via base class interface
- Byeol tests each model individually

**Phase 3: Ensemble Development**
- Tae creates SentimentEnsemble
- Pam creates PredictionEnsemble
- Ensembles tested with various model combinations

**Phase 4: Optimization**
- Compare single model vs ensemble performance
- Tune ensemble weights and strategies
- Document optimal configurations for different use cases

### Example Usage Scenarios

**Scenario 1: Development/Testing**
- Use single fastest model (e.g., VADER for sentiment, Gradient Boost for prediction)
- Fast iteration and debugging

**Scenario 2: Accuracy Priority**
- Use full ensemble (all sentiment models, all prediction models)
- Maximum confidence and accuracy

**Scenario 3: Balanced Performance**
- Use 2-3 best models in ensemble
- Good balance of accuracy and speed

**Scenario 4: A/B Testing**
- Run analysis with different configurations
- Compare results to determine optimal setup

### Technical Implementation Notes

**Base Class Interface (Pam - Prediction Models):**
```python
class BasePredictionModel(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Predict price direction for given data."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata (name, version, architecture)."""
        pass
```

**Base Class Interface (Tae - Sentiment Models):**
```python
class BaseSentimentModel(ABC):
    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata (name, version)."""
        pass
```

**Ensemble Interface (Both):**
```python
class PredictionEnsemble(BasePredictionModel):
    def __init__(self, models: List[BasePredictionModel], strategy: str, weights: dict = None):
        self.models = models
        self.strategy = strategy
        self.weights = weights or {m.get_model_info()['name']: 1.0/len(models) for m in models}

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        # Get predictions from all models
        predictions = [model.predict(data) for model in self.models]
        # Combine using strategy
        return self._combine_predictions(predictions)
```

This architecture provides maximum flexibility while maintaining clean separation of concerns and enabling independent development by team members.
