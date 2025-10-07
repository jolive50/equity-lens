# 🚀 How to Run StockSense - Complete Guide

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Start backend (Terminal 1)
python -m pipelines.realtime.api

# 3. Start frontend (Terminal 2)
cd frontend && npm run dev

# 4. Open browser
# Go to: http://localhost:3000
```

---

## Detailed Instructions

### Step 1: Verify Prerequisites

Open a terminal and check you have everything installed:

```bash
python --version   # Should be 3.11+
node --version     # Should be 18+
npm --version      # Comes with Node
```

### Step 2: Install Backend Dependencies

```bash
# Make sure you're in the project root directory
cd capstone  # or wherever you cloned the repo

# Install Python packages
pip install -r requirements.txt
```

**This will install:**
- FastAPI (web framework)
- LangChain & LangGraph (AI agents)
- PyTorch & Transformers (FinBERT)
- yfinance, pandas (data processing)
- And 40+ other packages

**Note:** Installing PyTorch might take 5-10 minutes!

### Step 3: Install Frontend Dependencies

```bash
# Navigate to frontend folder
cd frontend

# Install Node packages
npm install

# Go back to root
cd ..
```

**This will install:**
- Next.js & React
- Radix UI components
- Recharts (charts/graphs)
- TypeScript tooling

### Step 4: Verify Environment Files

#### Backend Environment (.env)

Check that `.env` exists in the root directory:

```bash
# Windows
dir .env

# Mac/Linux
ls -la .env
```

It should contain your API keys. The file is already set up!

#### Frontend Environment (frontend/.env.local)

Check that frontend has its config:

```bash
# Windows
dir frontend\.env.local

# Mac/Linux
ls -la frontend/.env.local
```

It should contain: `NEXT_PUBLIC_BACKEND_API_BASE=http://localhost:8000`

---

## Running the Application

### You Need TWO Terminals Running

#### Terminal 1: Backend API Server

```bash
# Make sure you're in the project root
python -m pipelines.realtime.api
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**What this does:**
- Starts FastAPI web server on port 8000
- Loads AI agents (Prediction, Sentiment, Explanation, Smart Money)
- Connects to data sources (Yahoo Finance, NewsAPI, etc.)
- Makes API available at http://localhost:8000

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### Terminal 2: Frontend Development Server

Open a NEW terminal window:

```bash
# Navigate to frontend
cd frontend

# Start Next.js
npm run dev
```

**Expected Output:**
```
  ▲ Next.js 15.5.4
  - Local:        http://localhost:3000
  - Network:      http://192.168.x.x:3000

  ✓ Ready in 2.5s
```

**What this does:**
- Starts Next.js development server on port 3000
- Compiles TypeScript and React components
- Enables hot-reloading (auto-refresh on code changes)
- Serves the web application

### Access the Application

Open your browser and go to:
```
http://localhost:3000
```

You should see the StockSense homepage!

---

## Testing the Application

### Test 1: Basic Stock Analysis

1. On the homepage, enter a stock ticker: **AAPL**
2. Click **"Analyze"**
3. Wait 5-10 seconds (it's fetching real data!)

**You should see:**
- Forecast direction (Up/Down/Neutral)
- Confidence percentage
- 95% confidence horizon (how many days the prediction is valid)
- Daily probability charts
- Key metrics (revenue growth, P/E ratio, etc.)
- News sentiment analysis
- Plain English explanation

### Test 2: User Tiers

Try different user tiers to see feature differences:

**As Basic User:**
- Can analyze any stock
- Sees basic predictions and sentiment
- Cannot save watchlists or history

**As Registered User:**
- All Basic features
- Can save favorite stocks to watchlist
- Sees search history
- Can batch analyze watchlist

**As Premium User:**
- All Registered features
- Smart money tracking (institutions, insiders, Congress)
- Alert system (get notified on price changes)
- Historical accuracy tracking
- CSV export capability

### Test 3: Batch Analysis (Registered/Premium)

1. Switch to "Registered" or "Premium" tier
2. Add stocks to watchlist: AAPL, MSFT, GOOGL
3. Click "Batch Predict"
4. See analysis for all stocks in a table

### Test 4: CSV Export (Premium Only)

1. Switch to "Premium" tier
2. Add stocks to watchlist
3. Click "Export CSV"
4. Downloads predictions.csv with detailed probability data

---

## Understanding the Data Flow

### What Happens When You Analyze a Stock:

```
1. Frontend: User enters "AAPL" and clicks Analyze
   ↓
2. Frontend sends: POST /api/analyze { "ticker": "AAPL", "user_tier": "basic" }
   ↓
3. Backend API receives request
   ↓
4. LangGraph Workflow Executes:

   A. Data Collection Phase
      • YahooFinanceProvider connects to Yahoo Finance API
      • Fetches last 3 months of daily prices (OHLCV)
      • Gets fundamental metrics (P/E, revenue growth, margins)
      • Retrieves recent news headlines

   B. Analysis Phase
      • PredictionAgent: Analyzes price patterns
        - Uses technical indicators
        - Considers fundamentals
        - Generates Up/Down/Neutral prediction
        - Calculates confidence score

      • SentimentAgent: Analyzes news
        - Feeds headlines to FinBERT model
        - FinBERT outputs: positive/negative/neutral
        - Aggregates sentiment across all articles
        - Identifies trend (improving/stable/declining)

      • SmartMoneyAgent (Premium only):
        - Fetches institutional ownership changes
        - Gets insider trading activity
        - Retrieves congressional disclosures

      • ExplanationAgent:
        - Takes prediction + sentiment + smart money
        - Uses GPT-4 to write plain English summary
        - Explains the "why" behind predictions

   C. Response Assembly
      • Formats all results into JSON
      • Adds warnings if confidence < 95%
      • Includes disclaimers (not investment advice)

   ↓
5. Frontend receives JSON response
   ↓
6. React components render:
   • Charts (Recharts library)
   • Metrics cards (Radix UI)
   • Formatted text
   • Color-coded indicators
```

### Data Sources Used:

**Market Data (Free):**
- Yahoo Finance: Historical prices, fundamentals
  - No API key required
  - Unlimited requests (rate-limited)
  - 3 months of daily data

**News Data (Free Tier):**
- Yahoo Finance News: Company news
- NewsAPI: General financial news (100 req/day)
- Finnhub: Financial news (60 req/min)

**AI Models:**
- FinBERT: Financial sentiment (runs locally)
- GPT-4o-mini: Explanations (OpenAI API)

---

## Troubleshooting

### Problem: Backend won't start

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch==2.5.1 transformers==4.36.2
```

---

**Error:** `Address already in use (port 8000)`

**Solution:**
```bash
# Windows - Find and kill process
netstat -ano | findstr :8000
# Note the PID, then kill it in Task Manager

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

---

### Problem: Frontend shows "Failed to fetch"

**Checklist:**
1. ✓ Backend running on port 8000?
2. ✓ `frontend/.env.local` has correct API URL?
3. ✓ No firewall blocking localhost connections?

**Solution:**
```bash
# Restart both servers
# Terminal 1:
python -m pipelines.realtime.api

# Terminal 2:
cd frontend && npm run dev
```

---

### Problem: "No API key found"

**Solution:**
1. Check `.env` exists in project root
2. Verify it has all API keys
3. Restart backend (it loads .env on startup)

---

### Problem: Rate limit errors

**Free Tier Limits:**
- Alpha Vantage: 500 calls/day
- NewsAPI: 100 requests/day
- Finnhub: 60 calls/minute
- Yahoo Finance: Rate-limited (wait if errors)

**Solution:**
- Wait a few minutes
- Use different tickers to spread requests
- Consider upgrading API plans if needed

---

### Problem: FinBERT is slow (first use)

**This is normal!** First time using FinBERT:

1. Downloads model from HuggingFace (~400 MB)
2. Loads neural network into memory
3. Takes 30-60 seconds

**Subsequent uses are much faster** (model is cached)

---

## Performance Optimization

### Speed Up Analysis

**Use LLM instead of FinBERT:**
```python
# In agents.py, line 138
sentiment_agent = SentimentAgent(llm, use_finbert=False)
```
Faster, but less accurate for financial sentiment.

**Reduce historical data period:**
```python
# In data_sources.py, line 68
hist = stock.history(period="1mo")  # Instead of "3mo"
```

### Reduce Memory Usage

**Use CPU instead of GPU for FinBERT:**
Already configured! FinBERT auto-detects and uses CPU on most systems.

---

## Advanced Usage

### API Examples

#### Analyze a stock programmatically:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "user_tier": "premium"}'
```

#### Get just price data:

```bash
curl "http://localhost:8000/prices?ticker=AAPL&limit=30"
```

#### Get sentiment only:

```bash
curl "http://localhost:8000/sentiment?ticker=AAPL"
```

### Python SDK Usage

```python
from pipelines.realtime.sp500_data_service import get_sp500_data_service

# Get data service
service = get_sp500_data_service()

# Fetch data
data = service.get_single_ticker_data("AAPL")

# Access components
prices = data['market_data']
fundamentals = data['fundamentals']
news = data['news_data']

print(f"Latest close: ${prices[-1]['close']}")
print(f"P/E ratio: {fundamentals['pe_ratio']}")
print(f"News count: {len(news)}")
```

---

## File Structure Reference

```
StockSense/
├── .env                          # API keys (DO NOT COMMIT!)
├── .env.example                  # Template for API keys
├── requirements.txt              # Python dependencies
├── test_data_fetch.py           # Test script
├── collect_training_data.py     # Data collection
├── SETUP.md                     # Detailed setup guide
├── RUNNING_INSTRUCTIONS.md      # This file
│
├── pipelines/
│   └── realtime/
│       ├── api.py               # Main FastAPI app
│       ├── agents.py            # AI agents
│       ├── langgraph_workflow.py # Agent orchestration
│       ├── data_sources.py      # Data providers
│       ├── sp500_data_service.py # S&P 500 service
│       ├── sentiment/
│       │   └── finbert.py       # FinBERT analyzer
│       └── models/
│           └── forecaster.py    # ML model
│
└── frontend/
    ├── .env.local              # Frontend config (DO NOT COMMIT!)
    ├── package.json            # Node dependencies
    └── src/
        ├── app/
        │   ├── page.tsx        # Homepage
        │   └── api/            # API routes (proxy)
        └── components/         # UI components
```

---

## Next Steps

### 1. Collect Training Data

```bash
python collect_training_data.py
```

This collects 2 years of historical data for training ML models.

### 2. Explore the API

Go to: http://localhost:8000/docs

Interactive Swagger UI where you can:
- Test all API endpoints
- See request/response schemas
- Try different parameters

### 3. Customize Agents

Edit `pipelines/realtime/agents.py` to:
- Change prompt templates
- Adjust confidence thresholds
- Add new analysis criteria

### 4. Add More Tickers

Edit `pipelines/realtime/sp500_data_service.py`:
- Update `TickerListProvider.DEFAULT_TICKERS` to add or remove tracked symbols
- Extend `DataQualitySummary` if you need additional quality metrics surfaced to the frontend

---

## Support

If something doesn't work:

1. Check this guide's Troubleshooting section
2. Review terminal logs for error messages
3. Check browser console (F12 → Console tab)
4. Verify `.env` has all required API keys
5. Make sure both servers are running

---

## Quick Command Reference

```bash
# Start backend
python -m pipelines.realtime.api

# Start frontend
cd frontend && npm run dev

# Run tests
python test_data_fetch.py

# Collect training data
python collect_training_data.py

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Check API docs
# http://localhost:8000/docs
```

---

**Happy analyzing! 📈**
