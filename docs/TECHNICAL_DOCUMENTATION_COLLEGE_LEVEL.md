# StockSense: Technical Documentation for College Students
**AI-Powered Stock Analysis Platform - Complete Architecture Guide**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components Explained](#core-components-explained)
4. [Data Flow & Processing](#data-flow--processing)
5. [Software Engineering Principles Applied](#software-engineering-principles-applied)
6. [API Endpoints Reference](#api-endpoints-reference)
7. [Current Status & Next Steps](#current-status--next-steps)

---

## Project Overview

### What is StockSense?

StockSense is a web application that helps everyday people (not just expert investors) understand stock market trends. The goal is to make stock analysis accessible by:

1. **Predicting stock movements** (up/down/neutral) using AI and machine learning
2. **Analyzing news sentiment** to understand public opinion about stocks
3. **Explaining predictions in simple English** so anyone can understand
4. **Tracking "smart money"** (what big investors like hedge funds are doing)

### The Problem We're Solving

Many retail investors avoid the stock market because they don't understand it. Studies show:
- 40% of non-investors say they "don't know how" to invest
- 70% would invest more with better education
- 47% of U.S. adults use AI chatbots like ChatGPT for financial advice

**Our Solution**: Create a user-friendly platform that combines AI analysis with plain-language explanations to make stock analysis accessible to everyone.

### Key Features by User Type

#### Anonymous Users (Free)
- Search any stock ticker
- See basic prediction (up/down/neutral)
- View limited news sentiment summary

#### Registered Users (Free Account)
- All anonymous features +
- Save search history
- Create watchlists of favorite stocks
- Compare multiple stocks side-by-side

#### Premium Users (Paid Subscription)
- All registered features +
- AI assistant for conversational analysis
- Full watchlist/portfolio analysis
- Alerts when predictions change
- Historical prediction accuracy tracking
- Export data to CSV
- "Smart money" insights (institutional trading)
- Detailed plain-language explanations (up to 3 key factors)

#### Administrators
- Manage user accounts
- Update AI models and data sources
- Monitor system performance and usage statistics

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Frontend      │ (Next.js/React - User Interface)
│   (Browser)     │
└────────┬────────┘
         │ HTTP Requests
         ↓
┌─────────────────┐
│   API Layer     │ (FastAPI - Web Server)
│   api.py        │
└────────┬────────┘
         │
         ↓
┌──────────────────────────────────────────────┐
│         LangGraph Workflow Orchestrator      │
│         (Coordinates AI Agents)              │
└──────┬─────────────────────────────────┬─────┘
       │                                 │
       ↓                                 ↓
┌─────────────────┐            ┌─────────────────┐
│ Working Agents  │            │  Legacy Agents  │
│ - Historical    │            │  - Prediction   │
│ - Sentiment     │            │  - Sentiment    │
└────────┬────────┘            │  - SmartMoney   │
         │                     │  - Explanation  │
         ↓                     └────────┬────────┘
┌─────────────────┐                    │
│ Coordination    │                    │
│    Agent        │                    │
└────────┬────────┘                    │
         │                             │
         └──────────┬──────────────────┘
                    ↓
         ┌──────────────────────┐
         │   Data Services      │
         │  - S&P 500 Service   │
         │  - Data Adapters     │
         │  - News Aggregation  │
         └──────────┬───────────┘
                    │
                    ↓
         ┌──────────────────────┐
         │  External APIs       │
         │  - Yahoo Finance     │
         │  - Alpha Vantage     │
         │  - Finnhub           │
         │  - NewsAPI           │
         └──────────────────────┘
```

### Technology Stack

**Backend (Python)**
- **FastAPI**: Modern web framework for building APIs
- **LangChain**: Framework for building AI agent applications
- **LangGraph**: Orchestrates multi-agent workflows
- **OpenAI GPT**: Large language model for natural language understanding
- **yfinance**: Free Yahoo Finance data access
- **Alpha Vantage**: Financial data API
- **FinBERT**: Specialized AI model for financial sentiment analysis

**Frontend (TypeScript/JavaScript)**
- **Next.js**: React framework for web applications
- **React**: UI component library
- **TypeScript**: Type-safe JavaScript
- **Radix UI**: Accessible component library

**Data Storage**
- **JSON files**: Simple file-based storage for watchlists, history, alerts
- **PostgreSQL**: Future database for production (not yet implemented)

---

## Core Components Explained

Let's break down each major file and explain what it does, why it exists, and how it works.

### 1. API Layer (`api.py`)

**Responsibility**: This is the "front desk" of your application. It receives requests from users' web browsers and coordinates responses.

**What it does**:
- Defines HTTP endpoints (URLs) that the frontend can call
- Validates user input (is the ticker symbol valid? is the user tier correct?)
- Calls the appropriate AI agents and data services
- Formats and returns responses

**Key Methods**:

```python
@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
```
- **Purpose**: Main endpoint for analyzing a single stock
- **Receives**: `{ "ticker": "AAPL", "user_tier": "basic" }`
- **Returns**: Complete analysis including prediction, sentiment, explanation
- **Why**: This is the primary function users interact with

```python
@app.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest):
```
- **Purpose**: Analyze multiple stocks at once (for watchlists)
- **Receives**: `{ "tickers": ["AAPL", "MSFT"], "user_tier": "premium" }`
- **Returns**: Summary predictions for each ticker
- **Why**: Premium users want to see their whole portfolio at once

```python
@app.get("/watchlist")
async def get_watchlist(user_id: str):
```
- **Purpose**: Get a user's saved watchlist
- **Receives**: User ID (who is asking?)
- **Returns**: List of ticker symbols they're watching
- **Why**: Users want to track specific stocks over time

**Data Flow Through API**:
1. User browser sends HTTP request → API endpoint
2. API validates input (check ticker format, user permissions)
3. API calls `get_data_service()` to fetch market data
4. API calls `get_agents()` to get AI agent instances
5. API runs `run_stocksense_analysis()` workflow
6. API formats result and sends back to browser

---

### 2. AI Agents (`agents.py`)

**Responsibility**: These are specialized AI "workers" that each handle a specific task.

Think of agents like a team at a research firm:
- **Prediction Agent**: The quant analyst who crunches numbers
- **Sentiment Agent**: The news analyst who reads financial news
- **Smart Money Agent**: The insider who tracks big investors
- **Explanation Agent**: The communicator who translates everything to plain English
- **Coordination Agent**: The manager who synthesizes everyone's input
- **Historical Analysis Agent**: The data scientist who analyzes trends
- **Sentiment Analysis Agent**: The social media monitor

#### PredictionAgent

**What it does**: Predicts if a stock will go up, down, or stay neutral.

**How it works**:
1. Receives market data (prices, volumes) and fundamentals (P/E ratio, revenue growth)
2. First tries to use a machine learning model (if available)
3. Falls back to LLM-based analysis if ML model fails
4. Returns: direction ("up"/"down"/"neutral"), confidence (0.0-1.0), and narrative explanation

**Why we need it**: This is the core value proposition - helping users know if a stock might go up or down.

**Key Method**:
```python
def run(self, *, ticker: str, market_data: Dict, fundamentals: Dict) -> PredictionResult
```
- **Receives**: Ticker symbol, recent price data, company fundamentals
- **Returns**: `PredictionResult(direction="up", confidence=0.85, narrative="...")`
- **Logic**:
  - If ML model available → use gradient boosting forecaster
  - Else → send data to GPT-4 with specialized prompt
  - Parse response to extract prediction

#### SentimentAgent

**What it does**: Analyzes news articles and social media to gauge market sentiment.

**How it works**:
1. Receives a list of news articles about the stock
2. First tries FinBERT (specialized financial sentiment AI)
3. Falls back to GPT-based analysis if FinBERT unavailable
4. Returns: current sentiment (positive/neutral/negative), score (0-1), trend (improving/stable/declining), top headlines

**Why we need it**: News and sentiment often drive short-term price movements more than fundamentals.

**Key Method**:
```python
def run(self, *, ticker: str, news_data: List[Dict]) -> SentimentResult
```
- **Receives**: List of news articles with titles, content, timestamps
- **Returns**: `SentimentResult(current="positive", score=0.75, trend="improving", headlines=[...])`
- **Logic**:
  - If FinBERT available → use specialized model for financial text
  - Else → ask GPT-4 to analyze sentiment
  - Extract most relevant headlines

#### SmartMoneyAgent

**What it does**: Tracks what institutional investors, company insiders, and Congress members are doing with the stock.

**How it works**:
1. Attempts to fetch institutional ownership data
2. Checks insider trading (CEOs, CFOs buying/selling)
3. Looks up congressional trading disclosures
4. Returns structured data about each category

**Why we need it**: "Smart money" often knows things before the public. If Warren Buffett is buying, that's a signal.

**Key Method**:
```python
def run(self, *, ticker: str) -> SmartMoneyResult
```
- **Receives**: Ticker symbol
- **Returns**: `SmartMoneyResult(institutions={...}, insiders={...}, congress={...})`
- **Logic**:
  - Try to get real data from smart money service
  - Fallback to LLM-generated analysis if unavailable
  - Return structured summaries for each category

#### ExplanationAgent

**What it does**: Translates all the technical analysis into plain English that anyone can understand.

**How it works**:
1. Receives results from all other agents
2. Considers the user's tier (basic vs premium)
3. Generates a narrative that explains:
   - The prediction and why
   - What sentiment indicates
   - What smart money is doing (premium only)
   - Risk warnings
4. Returns formatted plain-language explanation

**Why we need it**: Raw data and predictions aren't useful if users don't understand them.

**Key Method**:
```python
def run(self, *, ticker, prediction, sentiment, smart_money, user_tier, confidence_level) -> str
```
- **Receives**: Results from all other agents + user context
- **Returns**: Plain English narrative explanation
- **Logic**:
  - Craft prompt telling GPT to explain in simple terms
  - For premium users, include more factors
  - Always include risk disclaimers

#### CoordinationAgent

**What it does**: Synthesizes results from multiple working agents (Historical + Sentiment) and determines overall confidence.

**How it works**:
1. Receives results from Historical Analysis Agent and Sentiment Analysis Agent
2. Calculates weighted overall confidence (60% historical, 40% sentiment)
3. Checks if confidence meets the 95% threshold
4. If threshold not met, returns "INSUFFICIENT_CONFIDENCE"
5. If threshold met, synthesizes insights and generates recommendations

**Why we need it**: This ensures we only make recommendations when we're very confident (95%+ threshold from requirements).

**Key Method**:
```python
def coordinate(self, *, tickers: List[str], working_agent_results: Dict) -> Dict
```
- **Receives**: Results from Historical and Sentiment working agents
- **Returns**: Coordination summary with confidence assessment
- **Logic**:
  - Calculate overall_confidence = (historical * 0.6) + (sentiment * 0.4)
  - If confidence < 95% → return "INSUFFICIENT_CONFIDENCE"
  - Else → synthesize analysis and return recommendations

#### HistoricalAnalysisAgent

**What it does**: Analyzes historical market data and fundamental metrics across multiple S&P 500 companies.

**How it works**:
1. Receives comprehensive data for multiple tickers (market data + fundamentals)
2. Analyzes metrics like revenue growth, EBITDA margin, P/E ratio, ROE, price momentum
3. Uses both LLM analysis and quantitative calculations
4. Identifies top performers and companies with concerns
5. Returns analysis with confidence score

**Why we need it**: This is one of the "working agents" that does deep analysis before the coordination agent makes final decisions.

**Key Method**:
```python
def run(self, *, comprehensive_data: Dict[str, Dict], market_data: Dict) -> Dict
```
- **Receives**: Market and fundamental data for multiple companies
- **Returns**: Analysis with overall trend, technical/fundamental scores, confidence
- **Logic**:
  - Extract metrics: revenue growth, ROE, P/E ratios, momentum
  - Calculate sector-wide averages
  - Determine trends: positive if revenue growth > 5% and ROE > 15%
  - Combine LLM insights with quantitative analysis (60% quant, 40% LLM)

#### SentimentAnalysisAgent

**What it does**: Analyzes sentiment from news across multiple S&P 500 companies to identify sector-wide trends.

**How it works**:
1. Receives comprehensive news data for multiple tickers
2. Uses FinBERT to analyze sentiment if available
3. Identifies positive drivers and negative concerns
4. Returns sector-wide sentiment summary with confidence

**Why we need it**: This is the second "working agent" that analyzes news/sentiment before coordination.

**Key Method**:
```python
def run(self, *, comprehensive_news_data: Dict[str, List[Dict]], market_context: Dict) -> Dict
```
- **Receives**: News articles for multiple companies
- **Returns**: Overall sentiment, score, trend, key drivers/concerns
- **Logic**:
  - If FinBERT available → analyze sector-wide sentiment
  - Calculate convergence score (how aligned is sentiment across companies?)
  - Extract top headlines and identify themes
  - Return confidence based on sentiment alignment

---

### 3. LangGraph Workflow (`langgraph_workflow.py`)

**Responsibility**: This is the "orchestra conductor" that coordinates all the AI agents in the right order.

**What it does**:
- Defines the workflow: which agent runs when, in what order
- Manages state (data that flows between agents)
- Decides which path to take based on confidence levels
- Handles both single-ticker and multi-ticker analysis

**How it works**:

The workflow is like a factory assembly line:

```
Input (ticker)
  ↓
Validate Input → Check ticker is valid
  ↓
Collect S&P 500 Data → Get market data, fundamentals
  ↓
Collect News Data → Get news articles
  ↓
[PARALLEL EXECUTION]
  ├→ Historical Working Agent → Analyze trends
  └→ Sentiment Working Agent → Analyze news
  ↓
Coordination Agent → Synthesize results, check 95% threshold
  ↓
Prediction Agent (Legacy) → Generate prediction
  ↓
Sentiment Agent (Legacy) → Analyze sentiment
  ↓
Smart Money Agent → Track institutional activity (premium only)
  ↓
Explanation Agent → Generate plain English explanation
  ↓
Output (final analysis)
```

**State Object**:
The workflow passes a `StockAnalysisState` object between nodes. Think of it as a clipboard that each agent writes on:

```python
class StockAnalysisState(TypedDict):
    ticker: str                    # Which stock we're analyzing
    user_tier: str                 # basic/premium (what features user gets)
    market_data: Dict              # Price, volume data
    news_data: List[Dict]          # News articles
    fundamentals: Dict             # P/E ratio, revenue growth, etc.
    prediction_result: Dict        # What prediction agent found
    sentiment_result: Dict         # What sentiment agent found
    smart_money_data: Dict         # What institutional investors are doing
    explanation: str               # Plain English explanation
    confidence_level: str          # high/medium/low
    working_agent_results: Dict    # Results from working agents
    coordination_summary: str      # Synthesis from coordination agent
    warnings: List[str]            # Any warnings to show user
```

**Key Functions**:

```python
def create_stocksense_workflow(...) -> StateGraph
```
- **Purpose**: Build the workflow graph
- **How**: Adds nodes (agents) and edges (connections between them)
- **Returns**: A compiled workflow ready to execute

```python
def run_stocksense_analysis(...) -> Dict
```
- **Purpose**: Execute the workflow for a single ticker
- **How**: Invokes the compiled workflow with initial state
- **Returns**: Formatted response with all analysis results

```python
def run_enhanced_stocksense_analysis(...) -> Dict
```
- **Purpose**: Execute enhanced workflow for multiple S&P 500 tickers
- **How**: Uses coordination agent to synthesize multi-ticker analysis
- **Returns**: Comprehensive analysis across multiple companies

**Why this approach?**

The workflow pattern (LangGraph) provides:
1. **Modularity**: Easy to add/remove agents
2. **Visibility**: Can visualize the workflow graph
3. **Testability**: Can test each node independently
4. **Error handling**: If one agent fails, others can continue
5. **Flexibility**: Can change the order or add conditional branches

---

### 4. Data Adapters (`data_adapters.py`)

**Responsibility**: Fetch data from external APIs in a consistent, swappable way.

**Design Pattern**: **Strategy Pattern** - different data sources, same interface.

**Why we need this**: We want to use multiple data sources (Yahoo Finance, Alpha Vantage, Finnhub, NewsAPI) but don't want the rest of our code to know which one we're using.

**Key Classes**:

#### MarketData (Data Class)
```python
@dataclass
class MarketData:
    symbol: str
    price: float
    volume: int
    open_price: float
    high: float
    low: float
    close: float
    timestamp: str
    source: str
```
- **Purpose**: Standard format for market data from any source
- **Why**: Different APIs return data in different formats. This normalizes it.

#### DataAdapter (Abstract Base Class)
```python
class DataAdapter(ABC):
    def fetch_market_data(self, symbol: str) -> Optional[MarketData]
    def fetch_historical_data(self, symbol: str, days: int) -> List[MarketData]
    def fetch_news(self, symbol: str, limit: int) -> List[NewsItem]
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]
```
- **Purpose**: Define the contract that all data sources must follow
- **Why**: This allows us to swap data sources without changing other code (Open/Closed Principle)

#### AlphaVantageAdapter
**What it does**: Fetches data from Alpha Vantage API

**How it works**:
1. Stores API key and base URL
2. Implements each method using Alpha Vantage's specific endpoints
3. Converts Alpha Vantage's JSON format to our standard `MarketData` format
4. Handles errors (API limits, network failures)

**Example**:
```python
def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
    data = self._make_request("", {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol
    })
    # Convert to MarketData format
    return MarketData(
        symbol=symbol,
        price=float(quote["05. price"]),
        volume=int(quote["06. volume"]),
        # ... etc
    )
```

#### TiingoAdapter, FinnhubAdapter, NewsAPIAdapter
Similar to AlphaVantageAdapter but for their respective APIs.

#### DataAdapterFactory (Factory Pattern)
```python
class DataAdapterFactory:
    @staticmethod
    def create_adapter(adapter_type: str, api_key: str) -> DataAdapter:
        adapters = {
            "alpha_vantage": AlphaVantageAdapter,
            "tiingo": TiingoAdapter,
            "finnhub": FinnhubAdapter,
            "newsapi": NewsAPIAdapter,
        }
        return adapters[adapter_type](api_key)
```
- **Purpose**: Create the right adapter based on a string name
- **Why**: Easier than `if/elif` chains. Adding new sources is just one line.

#### DataService (Facade Pattern)
```python
class DataService:
    def __init__(self, adapters: Dict[str, DataAdapter]):
        self.adapters = adapters

    def get_all_news(self, symbol: str) -> List[NewsItem]:
        # Fetch from ALL news sources and combine
        all_news = []
        for source, adapter in self.adapters.items():
            news = adapter.fetch_news(symbol)
            all_news.extend(news)
        return unique_news  # Remove duplicates
```
- **Purpose**: Simplify data fetching for the rest of the application
- **Why**: Instead of calling multiple adapters, just call `DataService` and it handles everything

---

### 5. S&P 500 Data Service (`sp500_data_service.py`)

**Responsibility**: Specialized service for fetching and processing data for multiple S&P 500 companies.

**Why separate from DataService?** This handles bulk operations and S&P 500-specific logic (rate limiting, batch processing, sector analysis).

**Key Classes**:

#### TickerListProvider (Single Responsibility Principle)
```python
class TickerListProvider:
    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", ...]

    @staticmethod
    def get_sp500_tickers(count: int = 5) -> List[str]:
        return TickerListProvider.DEFAULT_TICKERS[:count]
```
- **Purpose**: Provide validated lists of S&P 500 tickers
- **Why**: Separates ticker management from data fetching

#### DataProcessingValidator
```python
class DataProcessingValidator:
    @staticmethod
    def process_ticker_data(ticker, historical, fundamentals, news):
        # Validate each component
        validated_historical = DataValidationService.validate_historical_data(historical)
        validated_fundamentals = DataValidationService.validate_fundamentals(fundamentals)
        return processed_data
```
- **Purpose**: Validate and process data quality
- **Why**: Ensures data is clean before analysis (garbage in, garbage out)

#### SP500DataService (Main Service)
```python
class SP500DataService:
    def __init__(self, data_provider, validator, ticker_provider):
        self.data_provider = data_provider  # Dependency Injection
        self.validator = validator
        self.ticker_provider = ticker_provider

    def get_single_ticker_data(self, ticker: str) -> Dict:
        # 1. Validate ticker
        # 2. Fetch historical, fundamentals, news
        # 3. Validate and process
        # 4. Return comprehensive data

    def get_multiple_tickers_data(self, tickers: List[str]) -> Dict[str, Dict]:
        # Process multiple tickers with rate limiting
        # Returns dict mapping ticker → data

    def get_sector_analysis(self, tickers: List[str]) -> Dict:
        # Calculate sector-wide metrics
        # Identify strong/weak performers
        # Return confidence and trend
```

**How it works**:

1. **Single Ticker**:
   ```
   User requests "AAPL"
   → Validate ticker format
   → Fetch historical data (3 months of prices)
   → Fetch fundamentals (P/E, revenue growth, etc.)
   → Fetch news (recent articles)
   → Validate all data
   → Return comprehensive package
   ```

2. **Multiple Tickers**:
   ```
   User requests ["AAPL", "MSFT", "GOOGL"]
   → Validate each ticker
   → For each ticker:
      → Fetch data (with 0.5s delay between requests for rate limiting)
      → Validate
      → Add to results
   → Return all results
   ```

3. **Sector Analysis**:
   ```
   User requests sector analysis for tech stocks
   → Fetch data for all tickers
   → Calculate averages:
      - Average revenue growth
      - Average P/E ratio
      - Average ROE
   → Classify companies:
      - Strong performers: revenue growth > 10% AND ROE > 15%
      - Weak performers: revenue growth < -5% OR ROE < 5%
   → Calculate confidence based on:
      - Sample size
      - Metric consistency
      - Percentage of strong performers
   → Return analysis
   ```

---

### 6. Data Sources (`data_sources.py`)

**Responsibility**: Abstract interface for financial data providers with concrete implementations.

**Key Classes**:

#### APIRateLimiter
```python
class APIRateLimiter:
    def __init__(self, requests_per_minute: int = 12):
        self.requests_per_minute = requests_per_minute
        self.last_request_times = []

    def check_and_wait(self):
        # If we've hit the limit, sleep until we can make another request
```
- **Purpose**: Prevent hitting API rate limits
- **How**: Tracks request timestamps, sleeps if needed
- **Why**: Free APIs have limits (e.g., 500 calls/day)

#### FinancialDataProvider (Abstract Interface)
```python
class FinancialDataProvider(ABC):
    @abstractmethod
    def get_historical_data(self, ticker: str) -> List[Dict]

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Dict[str, float]

    @abstractmethod
    def get_news_data(self, ticker: str) -> List[Dict]
```
- **Purpose**: Define interface for all data providers
- **Why**: Allows swapping between Yahoo Finance, Alpha Vantage, etc.

#### YahooFinanceProvider (Concrete Implementation)
```python
class YahooFinanceProvider(FinancialDataProvider):
    def get_historical_data(self, ticker: str) -> List[Dict]:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        return [convert to standard format]

    def get_fundamentals(self, ticker: str) -> Dict:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "revenue_growth": info.get("revenueGrowth"),
            "pe_ratio": info.get("trailingPE"),
            # ... etc
        }
```
- **Purpose**: Fetch real data from Yahoo Finance (free!)
- **How**: Uses `yfinance` library to access Yahoo's API
- **Why**: Yahoo Finance provides comprehensive free data

#### DataValidationService
```python
class DataValidationService:
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        # Check format: 1-10 uppercase letters

    @staticmethod
    def validate_historical_data(data: List[Dict]) -> List[Dict]:
        # Remove invalid data points
        # Ensure required fields exist

    @staticmethod
    def validate_fundamentals(fundamentals: Dict) -> Dict:
        # Check metrics are reasonable
        # Fill in missing values with defaults
```
- **Purpose**: Ensure data quality
- **Why**: APIs sometimes return incomplete or invalid data

---

### 7. Repository Layer (`repository.py`)

**Responsibility**: Persist user data (watchlists, history, alerts) to disk.

**Design Pattern**: **Repository Pattern** - abstract data storage so we can swap between JSON files, SQL database, etc.

**Key Classes**:

#### JsonFileRepository
```python
class JsonFileRepository:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._lock = threading.Lock()  # Thread-safe

    def get_user_data(self, user_id: str) -> Any:
        data = self._read_all()
        return data.get(user_id)

    def set_user_data(self, user_id: str, value: Any):
        data = self._read_all()
        data[user_id] = value
        self._write_all(data)
```
- **Purpose**: Read/write user data from JSON files
- **How**: Stores as `{"user_id": {...data...}}`
- **Why**: Simple, no database needed for MVP
- **Thread-safe**: Uses locks so multiple requests don't corrupt data

#### WatchlistRepository
```python
class WatchlistRepository:
    def get(self, user_id: str) -> List[str]:
        # Get user's watchlist

    def add(self, user_id: str, ticker: str) -> List[str]:
        # Add ticker to watchlist

    def remove(self, user_id: str, ticker: str) -> List[str]:
        # Remove ticker from watchlist
```
- **Purpose**: Manage user watchlists
- **Data**: Just a list of ticker symbols per user
- **Example**: `{"demo": ["AAPL", "MSFT", "GOOGL"]}`

#### HistoryRepository
```python
class HistoryRepository:
    def list(self, user_id: str, limit: int = 100) -> List[Dict]:
        # Get user's analysis history

    def append(self, user_id: str, entry: Dict):
        # Add new analysis to history
```
- **Purpose**: Track user's past stock analyses
- **Data**: List of compact entries: `{"ticker": "AAPL", "date": "...", "direction": "up", "confidence": 0.85}`

#### AlertsRepository
```python
@dataclass
class AlertRule:
    ticker: str
    condition: str  # "prob_down_gte", "prob_up_gte", "confidence_gte"
    threshold: float  # 0.0 to 1.0

class AlertsRepository:
    def upsert(self, user_id: str, rule: AlertRule):
        # Add or update alert rule

    def delete(self, user_id: str, ticker: str, condition: str):
        # Remove alert rule
```
- **Purpose**: Store user alert preferences
- **Example**: Alert me if AAPL probability of going down >= 70%

---

## Data Flow & Processing

### Complete Request Flow: Analyzing a Stock

Let's trace what happens when a user searches for "AAPL":

```
1. Frontend (Browser)
   User types "AAPL" and clicks "Analyze"
   ↓
   HTTP POST to /analyze
   Body: { "ticker": "AAPL", "user_tier": "premium" }

2. API Layer (api.py)
   ↓
   @app.post("/analyze")
   ↓
   Validate ticker: Is "AAPL" valid?
   ↓
   Get services:
   - get_data_service() → DataService instance
   - get_agents() → Dict of AI agents
   ↓
   Call: run_stocksense_analysis(ticker="AAPL", user_tier="premium", ...)

3. LangGraph Workflow (langgraph_workflow.py)
   ↓
   Node 1: validate_input
   - Check ticker is provided
   - Set default user_tier if missing
   ↓
   Node 2: collect_sp500_data
   - Call SP500DataService.get_single_ticker_data("AAPL")
   - Fetch historical prices (3 months)
   - Fetch fundamentals (P/E, revenue growth, ROE, etc.)
   - Store in state["market_data"] and state["fundamentals"]
   ↓
   Node 3: collect_comprehensive_news
   - Call SP500DataService.get_news_data("AAPL")
   - Fetch news from Yahoo Finance
   - Store in state["news_data"]
   ↓
   [PARALLEL EXECUTION]
   Node 4a: run_historical_working_agent
   - HistoricalAnalysisAgent analyzes market data + fundamentals
   - Calculate technical scores, fundamental scores
   - Identify trends
   - Store in state["working_agent_results"]["historical"]
   ↓
   Node 4b: run_sentiment_working_agent
   - SentimentAnalysisAgent analyzes news data
   - Use FinBERT to score sentiment
   - Extract key drivers and concerns
   - Store in state["working_agent_results"]["sentiment"]
   ↓
   Node 5: run_coordination_agent
   - CoordinationAgent synthesizes working agent results
   - Calculate overall_confidence = (historical * 0.6) + (sentiment * 0.4)
   - Check if confidence >= 0.95
   - If yes: state["confidence_level"] = "high"
   - If no: state["warnings"].append("Below 95% threshold")
   - Store in state["coordination_summary"]
   ↓
   Node 6: run_prediction (Legacy)
   - PredictionAgent generates probabilistic forecast
   - Returns direction, confidence, narrative
   - Generate daily probabilities for next 30 days
   - Calculate 95% confidence horizon
   - Store in state["prediction_result"]
   ↓
   Node 7: run_sentiment (Legacy)
   - SentimentAgent analyzes news (again, for compatibility)
   - Store in state["sentiment_result"]
   ↓
   Node 8: run_smart_money (Premium only)
   - SmartMoneyAgent tracks institutional activity
   - Fetch institutional ownership, insider trades, congressional disclosures
   - Store in state["smart_money_data"]
   ↓
   Node 9: build_explanation
   - ExplanationAgent creates plain English summary
   - Input: all previous results
   - Output: "AAPL shows strong upward momentum with 85% confidence..."
   - Store in state["explanation"]
   ↓
   Return final state

4. API Layer (api.py)
   ↓
   Format response:
   {
     "ticker": "AAPL",
     "as_of": "2025-01-24T10:30:00Z",
     "forecast": {
       "direction": "up",
       "confidence": 0.85,
       "horizon_95": {...},
       "daily_probs": [...]
     },
     "metrics": {...fundamentals...},
     "sentiment": {...},
     "smart_money": {...},
     "explanation": "...",
     "warnings": [],
     "disclaimers": [...]
   }
   ↓
   Send HTTP response to frontend

5. Frontend (Browser)
   ↓
   Display results to user:
   - Show direction (up/down/neutral) with visual indicator
   - Display confidence percentage
   - Show charts of daily probabilities
   - Display explanation text
   - Show sentiment indicators
   - Premium: show smart money insights
```

### Data Transformations

Throughout this flow, data gets transformed multiple times:

**Raw API Data → Standardized Format → Validated Data → Analysis Results → User-Friendly Presentation**

Example for AAPL:

```
1. Raw Yahoo Finance API:
{
  "Close": [150.23, 151.45, 149.87, ...],
  "Volume": [89234567, 92345678, ...],
  "trailingPE": 28.5,
  "revenueGrowth": 0.08,
  ...
}

2. Standardized (MarketData):
[
  {
    "date": "2025-01-01",
    "close": 150.23,
    "volume": 89234567,
    "open": 149.50,
    ...
  },
  ...
]

3. Validated:
- Remove any null values
- Ensure prices > 0
- Check volume is reasonable
- Fill missing fundamentals with defaults

4. Analysis Results:
{
  "direction": "up",
  "confidence": 0.85,
  "sentiment": "positive",
  "score": 0.72,
  ...
}

5. User-Friendly:
"Based on our analysis, AAPL is likely to move upward with 85% confidence.
Recent news sentiment is strongly positive, driven by strong earnings reports
and new product announcements. Institutional investors have increased their
positions by 3% this quarter."
```

---

## Software Engineering Principles Applied

### SOLID Principles

#### 1. Single Responsibility Principle (SRP)
**Definition**: Each class should have only ONE reason to change.

**Examples**:
- **TickerListProvider**: Only responsible for managing S&P 500 ticker lists
- **APIRateLimiter**: Only responsible for rate limiting API calls
- **DataValidationService**: Only responsible for validating data
- **WatchlistRepository**: Only responsible for watchlist persistence

**Why it matters**: If we need to change how we store watchlists, we only modify `WatchlistRepository`. The rest of the code doesn't change.

#### 2. Open/Closed Principle (OCP)
**Definition**: Classes should be open for extension but closed for modification.

**Example**:
```python
class DataAdapter(ABC):  # Abstract base class
    @abstractmethod
    def fetch_market_data(self, symbol: str):
        pass

class AlphaVantageAdapter(DataAdapter):  # Extension
    def fetch_market_data(self, symbol: str):
        # Alpha Vantage implementation

class TiingoAdapter(DataAdapter):  # Another extension
    def fetch_market_data(self, symbol: str):
        # Tiingo implementation
```

**Why it matters**: We can add new data sources (e.g., `BloombergAdapter`) without modifying existing code. Just create a new class that implements the `DataAdapter` interface.

#### 3. Liskov Substitution Principle (LSP)
**Definition**: Objects of a superclass should be replaceable with objects of a subclass.

**Example**:
```python
def get_data(adapter: DataAdapter, symbol: str):
    return adapter.fetch_market_data(symbol)

# Can use ANY adapter
get_data(AlphaVantageAdapter(api_key), "AAPL")  # Works
get_data(TiingoAdapter(api_key), "AAPL")        # Also works
get_data(FinnhubAdapter(api_key), "AAPL")       # Also works
```

**Why it matters**: Code that uses `DataAdapter` doesn't need to know which specific adapter it's using. They're all interchangeable.

#### 4. Interface Segregation Principle (ISP)
**Definition**: Don't force clients to depend on interfaces they don't use.

**Example**:
```python
# BAD: One giant interface
class DataProvider(ABC):
    @abstractmethod
    def fetch_prices(self): pass
    @abstractmethod
    def fetch_options(self): pass  # Not all providers have options
    @abstractmethod
    def fetch_crypto(self): pass   # Not all providers have crypto
    @abstractmethod
    def fetch_forex(self): pass    # Not all providers have forex

# GOOD: Separate interfaces
class PriceDataProvider(ABC):
    @abstractmethod
    def fetch_prices(self): pass

class OptionsDataProvider(ABC):
    @abstractmethod
    def fetch_options(self): pass
```

**Why it matters**: NewsAPI doesn't provide market data, only news. It shouldn't be forced to implement `fetch_market_data()`.

#### 5. Dependency Inversion Principle (DIP)
**Definition**: Depend on abstractions, not concrete implementations.

**Example**:
```python
# BAD: Depends on concrete class
class SP500Service:
    def __init__(self):
        self.provider = YahooFinanceProvider()  # Hard-coded!

# GOOD: Depends on abstraction
class SP500Service:
    def __init__(self, provider: FinancialDataProvider):
        self.provider = provider  # Injected!

# Usage
service = SP500Service(provider=YahooFinanceProvider())
# Or
service = SP500Service(provider=AlphaVantageProvider())
```

**Why it matters**: Easy to swap providers, easy to test (can inject mock provider).

### Design Patterns Used

#### 1. Strategy Pattern
**Where**: Data adapters (`DataAdapter` and subclasses)
**Purpose**: Swap data sources without changing code
**Benefit**: Can easily switch from Alpha Vantage to Tiingo

#### 2. Factory Pattern
**Where**: `DataAdapterFactory.create_adapter()`
**Purpose**: Create objects without specifying exact class
**Benefit**: Centralized object creation, easy to extend

#### 3. Facade Pattern
**Where**: `DataService` class
**Purpose**: Simple interface to complex subsystem
**Benefit**: Instead of calling multiple adapters, just call `DataService`

#### 4. Repository Pattern
**Where**: `WatchlistRepository`, `HistoryRepository`, `AlertsRepository`
**Purpose**: Abstract data persistence
**Benefit**: Can swap from JSON files to SQL database without changing business logic

#### 5. Dependency Injection
**Where**: Throughout the codebase (e.g., `SP500DataService.__init__`)
**Purpose**: Pass dependencies rather than creating them
**Benefit**: Easier testing, more flexible

### Object-Oriented Programming (OOP) Concepts

#### Encapsulation
**Definition**: Hide internal details, expose only what's necessary.

**Example**:
```python
class APIRateLimiter:
    def __init__(self, requests_per_minute: int):
        self._last_request_times = []  # Private (by convention with _)

    def check_and_wait(self):  # Public
        # Internal implementation hidden
```

**Why it matters**: Users of `APIRateLimiter` don't need to know HOW it works, just that it works.

#### Inheritance
**Definition**: Create specialized versions of general classes.

**Example**:
```python
class DataAdapter(ABC):  # Base class
    def _make_request(self, endpoint, params):
        # Common request logic

class AlphaVantageAdapter(DataAdapter):  # Inherits from DataAdapter
    def fetch_market_data(self, symbol):
        return self._make_request(...)  # Uses inherited method
```

**Why it matters**: Shared code (like error handling in `_make_request`) is written once, reused everywhere.

#### Polymorphism
**Definition**: Different classes can be used interchangeably if they implement the same interface.

**Example**:
```python
adapters = [
    AlphaVantageAdapter(api_key),
    TiingoAdapter(api_key),
    FinnhubAdapter(api_key)
]

for adapter in adapters:
    data = adapter.fetch_market_data("AAPL")  # Same method, different implementations
```

**Why it matters**: Write code once that works with any adapter.

#### Abstraction
**Definition**: Focus on WHAT something does, not HOW it does it.

**Example**:
```python
class FinancialDataProvider(ABC):
    @abstractmethod
    def get_historical_data(self, ticker: str) -> List[Dict]:
        """Fetch historical price data."""
        pass  # Subclasses implement HOW
```

**Why it matters**: Define contracts without worrying about implementation details.

---

## API Endpoints Reference

### Core Analysis Endpoints

#### `POST /analyze`
**Purpose**: Analyze a single stock

**Request**:
```json
{
  "ticker": "AAPL",
  "user_tier": "premium"
}
```

**Response**:
```json
{
  "ticker": "AAPL",
  "as_of": "2025-01-24T10:30:00Z",
  "forecast": {
    "direction": "up",
    "confidence": 0.85,
    "horizon_95": {
      "class": "up",
      "days": 14,
      "start_date": "2025-01-25",
      "end_date": "2025-02-08"
    },
    "daily_probs": [
      {"date": "2025-01-25", "up": 0.85, "down": 0.10, "neutral": 0.05},
      ...
    ]
  },
  "metrics": {
    "revenue_growth": {"value": 0.08, "verdict": "Good"},
    "ebitda_margin": {"value": 0.25, "verdict": "Good"},
    ...
  },
  "sentiment": {
    "current": "positive",
    "score": 0.72,
    "trend": "improving",
    "headlines": [...]
  },
  "smart_money": {
    "institutions": {...},
    "insiders": {...},
    "congress": {...}
  },
  "explanation": "Based on our analysis, AAPL shows strong upward momentum...",
  "warnings": [],
  "disclaimers": [...]
}
```

#### `POST /analyze/batch`
**Purpose**: Analyze multiple stocks (for watchlists)

**Request**:
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "user_tier": "premium"
}
```

**Response**:
```json
{
  "user_tier": "premium",
  "count": 3,
  "items": [
    {
      "ticker": "AAPL",
      "as_of": "2025-01-24T10:30:00Z",
      "direction": "up",
      "confidence": 0.85,
      "horizon_days": 14,
      "score": 0.72,
      "sentiment": {...}
    },
    ...
  ]
}
```

### Data Endpoints

#### `GET /prices?ticker=AAPL&source=alpha_vantage&limit=200`
**Purpose**: Get historical price data

**Response**:
```json
{
  "ticker": "AAPL",
  "source": "alpha_vantage",
  "count": 200,
  "items": [
    {
      "date": "2025-01-24",
      "close": 150.23,
      "open": 149.50,
      "high": 151.00,
      "low": 149.00,
      "volume": 89234567
    },
    ...
  ]
}
```

#### `GET /news?ticker=AAPL&source=newsapi&limit=100`
**Purpose**: Get news articles

**Response**:
```json
{
  "ticker": "AAPL",
  "source": "newsapi",
  "count": 100,
  "items": [
    {
      "title": "Apple Reports Record Q4 Earnings",
      "content": "...",
      "source": "Reuters",
      "url": "https://...",
      "timestamp": "2025-01-24T09:00:00Z",
      "sentiment_score": 0.75
    },
    ...
  ]
}
```

#### `GET /sentiment?ticker=AAPL&limit=200`
**Purpose**: Get sentiment analysis

**Response**:
```json
{
  "ticker": "AAPL",
  "current": "positive",
  "score": 0.72,
  "trend": "improving",
  "headlines": [...],
  "news_count": 200
}
```

#### `GET /smart-money?ticker=AAPL&user_tier=premium`
**Purpose**: Get smart money data (premium only)

**Response**:
```json
{
  "ticker": "AAPL",
  "institutions": {
    "summary": "Institutional ownership increased by 3% this quarter..."
  },
  "insiders": {
    "summary": "CEO purchased 10,000 shares on 2025-01-15..."
  },
  "congress": {
    "summary": "5 congressional trades disclosed this month..."
  }
}
```

### User Data Endpoints

#### `GET /watchlist?user_id=demo`
**Purpose**: Get user's watchlist

**Response**:
```json
{
  "user_id": "demo",
  "tickers": ["AAPL", "MSFT", "GOOGL"]
}
```

#### `POST /watchlist?user_id=demo&ticker=TSLA`
**Purpose**: Add ticker to watchlist

**Response**:
```json
{
  "user_id": "demo",
  "tickers": ["AAPL", "MSFT", "GOOGL", "TSLA"]
}
```

#### `DELETE /watchlist?user_id=demo&ticker=TSLA`
**Purpose**: Remove ticker from watchlist

**Response**:
```json
{
  "user_id": "demo",
  "tickers": ["AAPL", "MSFT", "GOOGL"]
}
```

#### `GET /history?user_id=demo&limit=100`
**Purpose**: Get analysis history

**Response**:
```json
{
  "user_id": "demo",
  "count": 100,
  "items": [
    {
      "ticker": "AAPL",
      "as_of": "2025-01-24T10:30:00Z",
      "direction": "up",
      "confidence": 0.85,
      "score": 0.72
    },
    ...
  ]
}
```

### Alert Endpoints

#### `GET /alerts?user_id=demo`
**Purpose**: List user's alert rules

**Response**:
```json
{
  "user_id": "demo",
  "rules": [
    {
      "ticker": "AAPL",
      "condition": "prob_down_gte",
      "threshold": 0.7
    }
  ]
}
```

#### `POST /alerts?user_id=demo&ticker=AAPL&condition=prob_down_gte&threshold=0.7`
**Purpose**: Create or update alert rule

**Response**:
```json
{
  "user_id": "demo",
  "rules": [...]
}
```

#### `DELETE /alerts?user_id=demo&ticker=AAPL&condition=prob_down_gte`
**Purpose**: Delete alert rule

**Response**:
```json
{
  "user_id": "demo",
  "rules": [...]
}
```

### Export Endpoints

#### `GET /export/predictions.csv?tickers=AAPL,MSFT&max_rows=5000&user_tier=premium`
**Purpose**: Export predictions to CSV (premium only)

**Response**: CSV file
```csv
ticker,date,up,down,neutral,direction,confidence,horizon_days
AAPL,2025-01-25,0.8500,0.1000,0.0500,up,0.8500,14
AAPL,2025-01-26,0.8400,0.1050,0.0550,up,0.8500,14
...
```

### Admin Endpoints

#### `GET /admin/usage`
**Purpose**: Get usage statistics

**Response**:
```json
{
  "active_users": 1,
  "recent_predictions": 150,
  "top_tickers": [
    ["AAPL", 45],
    ["MSFT", 32],
    ["GOOGL", 28],
    ...
  ]
}
```

#### `POST /admin/models/deploy?model_name=v2_forecaster`
**Purpose**: Deploy new model version

**Response**:
```json
{
  "status": "ok",
  "message": "Deployment initiated for v2_forecaster"
}
```

### Health & Info Endpoints

#### `GET /`
**Purpose**: API information

**Response**:
```json
{
  "message": "StockSense API",
  "version": "1.0.0",
  "description": "Layperson-friendly stock insights with AI-powered forecasting",
  "endpoints": {...}
}
```

#### `GET /health`
**Purpose**: Health check

**Response**:
```json
{
  "status": "healthy",
  "data_adapters": 4,
  "agents": 4,
  "timestamp": "2025-01-24T10:30:00Z"
}
```

---

## Current Status & Next Steps

### What's Working ✅

1. **Data Collection**:
   - Yahoo Finance integration (free, real-time data)
   - Alpha Vantage integration (backup data source)
   - News fetching from multiple sources
   - S&P 500 company data retrieval

2. **AI Agents**:
   - Prediction agent (with ML model fallback)
   - Sentiment analysis (with FinBERT support)
   - Smart money tracking
   - Plain-language explanations
   - Coordination agent (95% confidence threshold)
   - Historical analysis agent
   - Sentiment analysis agent

3. **API Endpoints**:
   - Single stock analysis
   - Batch analysis for watchlists
   - Watchlist management
   - History tracking
   - Alert management
   - CSV export

4. **Software Architecture**:
   - SOLID principles applied throughout
   - Design patterns (Strategy, Factory, Repository, Facade)
   - Dependency injection for testability
   - Clean separation of concerns

### What Needs Work 🚧

1. **Machine Learning Model**:
   - Current: Placeholder ML forecaster
   - Needed: Trained gradient boosting or LSTM model
   - Action: Collect historical data, train model, backtest
   - Reference: `pipelines/realtime/models/forecaster.py` (needs implementation)

2. **FinBERT Integration**:
   - Current: Fallback to GPT for sentiment
   - Needed: Fully integrated FinBERT model
   - Action: Download FinBERT model, implement sentiment pipeline
   - Reference: `pipelines/realtime/sentiment/finbert.py` (needs completion)

3. **Smart Money Data Sources**:
   - Current: LLM-generated placeholders
   - Needed: Real SEC filings, insider trading data, congressional disclosures
   - Action: Integrate SEC EDGAR API, insider trading APIs
   - Reference: `pipelines/realtime/smart_money/` (needs creation)

4. **Database Migration**:
   - Current: JSON file storage
   - Needed: PostgreSQL for production
   - Action: Create database schema, migrate repositories
   - Benefit: Better performance, concurrent access

5. **Frontend Integration**:
   - Current: Backend API complete
   - Needed: Connect Next.js frontend to backend
   - Action: Implement API client, create UI components
   - Reference: `frontend/` directory

6. **User Authentication**:
   - Current: Simple user_id parameter
   - Needed: JWT-based auth, session management
   - Action: Implement auth middleware, user management
   - Security: Hash passwords, secure sessions

7. **Testing**:
   - Current: Some unit tests
   - Needed: Comprehensive test suite
   - Action: Write tests for all components
   - Target: >80% code coverage

8. **Documentation**:
   - Current: Code comments, this document
   - Needed: API documentation (Swagger/OpenAPI)
   - Action: Add docstrings, generate API docs
   - Tool: FastAPI auto-generates docs at `/docs`

9. **Performance Optimization**:
   - Current: Sequential processing
   - Needed: Caching, async processing
   - Action: Add Redis cache, implement async workflows
   - Benefit: Faster response times

10. **Monitoring & Logging**:
    - Current: Basic logging
    - Needed: Structured logging, metrics, alerts
    - Action: Add Prometheus metrics, ELK stack
    - Benefit: Production observability

### Recommended Next Steps (Priority Order)

#### Phase 1: Core Functionality (Weeks 1-2)
1. **Train ML Forecasting Model**
   - Collect 5 years of S&P 500 data
   - Feature engineering (technical indicators: RSI, MACD, Bollinger Bands)
   - Train gradient boosting model
   - Backtest and validate
   - Integrate into `PredictionAgent`

2. **Complete FinBERT Integration**
   - Download pre-trained FinBERT model
   - Implement `finbert.py` sentiment analysis
   - Test on real news articles
   - Benchmark against GPT fallback

3. **Frontend-Backend Connection**
   - Create API client in Next.js
   - Build stock search component
   - Display prediction results
   - Add watchlist UI

#### Phase 2: Enhanced Features (Weeks 3-4)
4. **Smart Money Data Integration**
   - Research SEC EDGAR API
   - Implement insider trading data fetch
   - Add congressional trading (use existing public datasets)
   - Test with multiple tickers

5. **User Authentication**
   - Choose auth library (e.g., Auth0, Firebase, or custom JWT)
   - Implement registration/login endpoints
   - Add auth middleware to protected endpoints
   - Update frontend with auth UI

6. **Database Migration**
   - Design PostgreSQL schema
   - Create migration scripts
   - Update repositories to use SQLAlchemy
   - Test data persistence

#### Phase 3: Production Readiness (Weeks 5-6)
7. **Testing & Quality Assurance**
   - Write unit tests for all agents
   - Integration tests for workflows
   - End-to-end tests for critical paths
   - Performance testing (load testing)

8. **Performance Optimization**
   - Add Redis caching for market data
   - Implement async processing for batch analysis
   - Optimize database queries
   - Add CDN for frontend assets

9. **Monitoring & Deployment**
   - Set up structured logging
   - Add Prometheus metrics
   - Configure alerts (error rates, latency)
   - Deploy to cloud (AWS/GCP/Azure)

10. **Documentation & Polish**
    - Complete API documentation
    - Write user guide
    - Create admin dashboard
    - Final UI/UX polish

### Gaps & Improvements

#### Architecture Improvements

1. **Event-Driven Architecture**
   - **Current**: Synchronous workflow
   - **Improvement**: Use message queue (e.g., RabbitMQ, Kafka)
   - **Benefit**: Better scalability, async processing
   - **Example**: User requests analysis → publish to queue → worker processes → notify user

2. **Microservices Separation**
   - **Current**: Monolithic backend
   - **Improvement**: Split into services (prediction service, sentiment service, data service)
   - **Benefit**: Independent scaling, fault isolation
   - **Complexity**: Requires service discovery, API gateway

3. **Circuit Breaker Pattern**
   - **Current**: Direct API calls, failures cascade
   - **Improvement**: Implement circuit breakers for external APIs
   - **Benefit**: Graceful degradation when APIs fail
   - **Tool**: Use `pybreaker` library

#### Code Quality Improvements

1. **Type Safety**
   - **Current**: Some type hints
   - **Improvement**: Full type coverage with mypy
   - **Benefit**: Catch errors at development time
   - **Action**: Add type hints to all functions

2. **Error Handling**
   - **Current**: Try/except blocks
   - **Improvement**: Custom exception hierarchy
   - **Benefit**: Better error messages, easier debugging
   - **Example**:
     ```python
     class StockSenseError(Exception): pass
     class DataFetchError(StockSenseError): pass
     class ValidationError(StockSenseError): pass
     ```

3. **Configuration Management**
   - **Current**: Environment variables
   - **Improvement**: Centralized config with validation
   - **Benefit**: Easier to manage, type-safe config
   - **Tool**: Use `pydantic` for config validation

4. **Logging Standardization**
   - **Current**: Basic logging
   - **Improvement**: Structured logging (JSON format)
   - **Benefit**: Better log aggregation, searchability
   - **Tool**: Use `structlog` library

#### Feature Enhancements

1. **Backtesting Framework**
   - **Purpose**: Test predictions against historical data
   - **Benefit**: Measure model accuracy over time
   - **Implementation**: Store predictions, compare to actual outcomes

2. **Portfolio Analysis**
   - **Purpose**: Analyze entire portfolio, not just individual stocks
   - **Benefit**: Understand diversification, risk exposure
   - **Implementation**: Calculate portfolio metrics (Sharpe ratio, beta, etc.)

3. **Sector Comparison**
   - **Purpose**: Compare stock performance within sectors
   - **Benefit**: Identify relative winners/losers
   - **Implementation**: Extend `get_sector_analysis()` method

4. **Risk Metrics**
   - **Purpose**: Show volatility, max drawdown, VaR
   - **Benefit**: Help users understand risk
   - **Implementation**: Calculate from historical prices

5. **Explainability Dashboard**
   - **Purpose**: Show which features influenced prediction
   - **Benefit**: Trust and transparency
   - **Tool**: SHAP values from ML model

#### Data Source Improvements

1. **Add More Free Sources**
   - IEX Cloud (50,000 messages/month free)
   - Nasdaq Data Link (some free datasets)
   - Financial Modeling Prep (free tier)

2. **Data Quality Monitoring**
   - Track data freshness
   - Detect anomalies (e.g., sudden price spikes)
   - Alert when data quality degrades

3. **Historical Data Pipeline**
   - **Current**: Fetch on-demand
   - **Improvement**: Pre-fetch and cache daily
   - **Benefit**: Faster responses, less API usage
   - **Tool**: Scheduled jobs (cron or Airflow)

#### Security Enhancements

1. **Rate Limiting**
   - **Current**: None on API endpoints
   - **Improvement**: Add per-user rate limits
   - **Benefit**: Prevent abuse
   - **Tool**: `slowapi` library

2. **Input Sanitization**
   - **Current**: Basic validation
   - **Improvement**: Comprehensive input validation
   - **Benefit**: Prevent injection attacks
   - **Tool**: Pydantic models

3. **API Key Management**
   - **Current**: Environment variables
   - **Improvement**: Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
   - **Benefit**: Secure, rotatable secrets

4. **HTTPS Only**
   - **Current**: HTTP in development
   - **Improvement**: Enforce HTTPS in production
   - **Benefit**: Encrypted communication
   - **Tool**: Let's Encrypt for free SSL

### Learning Resources

To continue developing this project, study these topics:

1. **LangChain/LangGraph**:
   - Official docs: https://docs.langchain.com/
   - LangGraph tutorial: https://langchain-ai.github.io/langgraph/

2. **Machine Learning for Finance**:
   - "Machine Learning for Algorithmic Trading" by Stefan Jansen
   - Kaggle competitions on stock prediction

3. **FastAPI**:
   - Official tutorial: https://fastapi.tiangolo.com/tutorial/
   - Real Python FastAPI guide

4. **Design Patterns**:
   - "Design Patterns" by Gang of Four
   - Refactoring.Guru: https://refactoring.guru/design-patterns

5. **Financial Data**:
   - Alpha Vantage docs: https://www.alphavantage.co/documentation/
   - yfinance guide: https://pypi.org/project/yfinance/

6. **Testing**:
   - pytest documentation: https://docs.pytest.org/
   - "Test-Driven Development with Python" by Harry Percival

---

## Glossary of Terms

**API (Application Programming Interface)**: A way for programs to talk to each other. Like a waiter taking your order to the kitchen.

**Agent**: In AI, a specialized program that performs a specific task. Like having different experts on a team.

**Confidence**: How sure the AI is about its prediction. 95% confidence = very sure, 50% = just guessing.

**Dependency Injection**: Passing objects to a class instead of creating them inside. Makes testing easier.

**FinBERT**: A specialized AI model trained on financial text. Better at understanding stock news than general AI.

**Fundamentals**: Company metrics like revenue, profit, debt. Indicate long-term health.

**LangChain**: Framework for building AI applications that use large language models.

**LangGraph**: Tool for orchestrating multiple AI agents in a workflow.

**ML (Machine Learning)**: Teaching computers to learn patterns from data instead of programming rules explicitly.

**OOP (Object-Oriented Programming)**: Programming style using objects and classes. Like building with Lego blocks.

**Repository Pattern**: Abstracting data storage so you can swap databases easily.

**Sentiment**: Public opinion about a stock based on news and social media.

**SOLID**: Five principles for clean, maintainable code (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion).

**Strategy Pattern**: Design pattern for swapping algorithms (like changing payment methods: cash vs credit card).

**Technical Indicators**: Calculated metrics from price/volume data (RSI, MACD, etc.). Indicate short-term trends.

**Ticker**: Stock symbol (e.g., AAPL for Apple, MSFT for Microsoft).

**Workflow**: A series of steps executed in order. Like a recipe.

---

## Summary

**StockSense** is an AI-powered platform that makes stock analysis accessible to everyone by:

1. **Collecting data** from free sources (Yahoo Finance, Alpha Vantage, news APIs)
2. **Analyzing** with specialized AI agents (prediction, sentiment, smart money)
3. **Coordinating** via LangGraph workflow (95% confidence threshold)
4. **Explaining** in plain English for everyday investors

The system follows **SOLID principles** and uses proven **design patterns** (Strategy, Factory, Repository) for maintainable, extensible code. The architecture separates concerns cleanly:
- **API layer** handles HTTP requests
- **Agents** perform specialized analysis
- **Data adapters** abstract external APIs
- **Services** orchestrate complex operations
- **Repositories** persist user data

**Current status**: Backend core is complete, ML models and frontend integration are in progress.

**Next steps**: Train forecasting model, complete FinBERT integration, connect frontend, add authentication, deploy to production.

---

**This documentation is your roadmap to understanding and extending StockSense. Use it as a reference as you work on the project!**
