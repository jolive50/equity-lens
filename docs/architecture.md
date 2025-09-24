# StockSense System Architecture

## Overview

StockSense is a production-ready web application that provides layperson-friendly stock insights using AI-powered forecasting, sentiment analysis, and smart money tracking. The system integrates LangChain and LangGraph for orchestrated multi-agent workflows.

## System Design

### Core Components

1. **Frontend (Next.js/React)**
   - Single-page application with responsive design
   - Real-time stock analysis dashboard
   - Accessibility-compliant (WCAG AA)

2. **Backend Services**
   - FastAPI-based microservices architecture
   - LangGraph workflow orchestration
   - Multi-agent AI system (Prediction, Sentiment, Explanation, Smart Money)

3. **Data Layer**
   - Swappable data adapters (Alpha Vantage, Tiingo, Finnhub, NewsAPI)
   - Feature store for ML features
   - PostgreSQL for structured data
   - Object storage for raw documents

4. **ML Pipeline**
   - Probabilistic forecasting with 95% confidence horizon
   - FinBERT sentiment analysis
   - Feature engineering and model registry

### Data Flow Architecture

```
User Input (Ticker) 
    ↓
API Gateway (FastAPI)
    ↓
LangGraph Workflow Orchestrator
    ↓
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Data Agents   │ Prediction Agent│ Sentiment Agent│ Smart Money Agent│
│                 │                 │                 │                 │
│ • Market Data   │ • ML Models     │ • FinBERT       │ • Institutions  │
│ • Fundamentals  │ • Probabilities │ • News Analysis │ • Insiders     │
│ • News Sources  │ • Confidence    │ • Social Media  │ • Congress     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
    ↓
Explanation Agent (Plain English)
    ↓
Response Assembly
    ↓
Frontend Display
```

## API Contracts

### Core Endpoint: `/api/analyze`

**Request:**
```json
{
  "ticker": "AAPL",
  "user_tier": "basic|premium"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "as_of": "2025-01-24T10:30:00Z",
  "forecast": {
    "direction": "up|down|neutral",
    "confidence": 0.96,
    "horizon_95": {
      "days": 9,
      "end_date": "2025-02-02",
      "drops_below_95_on": "2025-02-03"
    },
    "daily_probs": [
      {
        "date": "2025-01-25",
        "up": 0.71,
        "down": 0.18,
        "neutral": 0.11
      }
    ]
  },
  "metrics": {
    "revenue_growth": {
      "value": 0.08,
      "verdict": "Good",
      "explanation": "Revenue growth is above sector average and 5-year median."
    },
    "ebitda_margin": {
      "value": 0.28,
      "verdict": "Good", 
      "explanation": "EBITDA margin is strong and improving over recent quarters."
    }
  },
  "sentiment": {
    "current": "positive",
    "score": 0.72,
    "trend": "improving",
    "headlines": [
      "Apple reports strong Q4 earnings",
      "New product launch drives optimism",
      "Analyst upgrades price target"
    ]
  },
  "smart_money": {
    "institutions": {
      "ownership_change": 0.02,
      "top_buyers": ["Vanguard", "BlackRock"],
      "summary": "Institutional ownership increased 2% last quarter"
    },
    "insiders": {
      "recent_activity": "2 buys, 1 sell",
      "net_position": "positive",
      "summary": "Insider activity shows confidence with net buying"
    },
    "congress": {
      "recent_trades": [],
      "summary": "No recent congressional trading activity"
    }
  },
  "disclaimers": [
    "This is informational only, not investment advice",
    "Data sources: Alpha Vantage, NewsAPI, SEC filings",
    "Last updated: 2025-01-24T10:30:00Z"
  ]
}
```

### Supporting Endpoints

- `/api/prices` - Historical price data
- `/api/news` - News sentiment data  
- `/api/sentiment` - Sentiment analysis results
- `/api/smart-money` - Institutional and insider data

## Data Adapters (OOP Design)

### Abstract Base Classes

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: int
    timestamp: str
    # Additional OHLCV fields

@dataclass  
class NewsItem:
    title: str
    content: str
    sentiment_score: float
    source: str
    timestamp: str

class DataAdapter(ABC):
    """Abstract base class for all data adapters."""
    
    @abstractmethod
    def fetch_market_data(self, symbol: str) -> MarketData:
        pass
        
    @abstractmethod
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        pass

class AlphaVantageAdapter(DataAdapter):
    """Alpha Vantage API implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def fetch_market_data(self, symbol: str) -> MarketData:
        # Implementation
        pass
        
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        # Implementation  
        pass

class TiingoAdapter(DataAdapter):
    """Tiingo API implementation."""
    # Similar structure
```

## LangGraph Workflow

### State Definition

```python
from typing import TypedDict, Literal

class StockAnalysisState(TypedDict):
    ticker: str
    market_data: Optional[MarketData]
    news_data: List[NewsItem]
    fundamentals: Dict[str, float]
    prediction_result: Optional[PredictionResult]
    sentiment_result: Optional[SentimentResult]
    smart_money_data: Optional[SmartMoneyData]
    explanation: Optional[str]
    confidence_level: Literal["high", "medium", "low"]
    user_tier: Literal["basic", "premium"]
```

### Workflow Nodes

1. **Data Collection Node**
   - Parallel data fetching from multiple sources
   - Data validation and normalization

2. **Prediction Node** 
   - ML model inference
   - Confidence calculation
   - 95% horizon determination

3. **Sentiment Analysis Node**
   - FinBERT processing
   - News aggregation
   - Social media sentiment

4. **Smart Money Node**
   - Institutional holdings
   - Insider trading
   - Congressional disclosures

5. **Explanation Node**
   - Plain English generation
   - Tier-based detail level
   - Risk warnings

## Database Schema

### Core Tables

```sql
-- Market data
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,2),
    volume BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE,
    source VARCHAR(50)
);

-- Predictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    direction VARCHAR(10),
    confidence DECIMAL(3,2),
    horizon_days INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sentiment scores
CREATE TABLE sentiment_scores (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    positive DECIMAL(3,2),
    neutral DECIMAL(3,2), 
    negative DECIMAL(3,2),
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Smart money data
CREATE TABLE institutional_holdings (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    institution_name VARCHAR(100),
    shares BIGINT,
    change_shares BIGINT,
    filing_date DATE
);
```

## Security & Compliance

### Data Protection
- API key management via environment variables
- Rate limiting and request validation
- Input sanitization and SQL injection prevention

### Financial Disclaimers
- Prominent "informational only" warnings
- No investment advice language
- Data source transparency
- Confidence level indicators

### Accessibility
- WCAG AA compliance
- Screen reader support
- Keyboard navigation
- Color contrast ratios

## Performance Requirements

### Response Times
- Basic users: 95% of requests < 2 seconds
- Premium users: 95% of requests < 1 second
- Dashboard load: < 3 seconds

### Data Freshness
- Free users: Data within 24 hours
- Premium users: Data within 1 minute

### Availability
- 95% uptime during trading hours
- Graceful degradation for API failures
- Cached fallbacks for critical data

## Deployment Architecture

### Infrastructure
- Containerized services (Docker)
- Kubernetes orchestration
- Load balancing and auto-scaling
- CI/CD pipeline with automated testing

### Monitoring
- Application performance monitoring
- Error tracking and alerting
- Data freshness monitoring
- User experience metrics

### Environment Configuration
```bash
# API Keys
ALPHA_VANTAGE_API_KEY=TKL8YS43GMNA1BYD
TIINGO_API_KEY=01163c25587928b9b80c53f37d946a76460a1f2f
FINNHUB_API_KEY=d3a2f59r01qli8jccd0gd3a2f59r01qli8jccd10
NEWSAPI_KEY=6fb3385bef034fcea2afabf02a2365fe

# Database
DATABASE_URL=postgresql://user:pass@localhost/stocksense

# LangChain
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langchain_key
```

## Next Steps

1. **Phase 1**: Core forecasting and sentiment analysis
2. **Phase 2**: Smart money tracking implementation  
3. **Phase 3**: Advanced features and optimization
4. **Phase 4**: Mobile app and additional data sources

## Limitations & Future Enhancements

### Current Limitations
- Limited to US markets initially
- Basic technical analysis features
- No real-time streaming data
- Limited historical backtesting

### Future Enhancements
- Options data integration
- Alternative data sources (satellite, social)
- Portfolio optimization features
- Advanced risk metrics
- Multi-market support
- Real-time streaming updates
