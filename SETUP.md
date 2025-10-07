# 🚀 StockSense Setup Guide

This guide will walk you through setting up and running the StockSense AI stock analysis platform from scratch.

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running the Application](#running-the-application)
5. [Testing the System](#testing-the-system)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you start, make sure you have these installed on your computer:

### Required Software
- **Python 3.11 or higher** - Download from [python.org](https://www.python.org/downloads/)
- **Node.js 18 or higher** - Download from [nodejs.org](https://nodejs.org/)
- **Git** - Download from [git-scm.com](https://git-scm.com/)

### Check Your Installations
Open a terminal and run these commands to verify:

```bash
python --version    # Should show 3.11 or higher
node --version      # Should show 18 or higher
npm --version       # Comes with Node.js
git --version       # Should show any recent version
```

---

## Quick Start

If you just want to get the app running quickly:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd capstone

# 2. Install Python dependencies
python -m pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..

# 4. The .env file is already set up with API keys!

# 5. Start the backend (in one terminal)
python -m pipelines.realtime.api

# 6. Start the frontend (in another terminal)
cd frontend
npm run dev

# 7. Open your browser to http://localhost:3000
```

---

## Detailed Setup

### Step 1: Clone the Repository

```bash
# Open your terminal and navigate to where you want the project
cd ~/Documents  # or wherever you keep your projects

# Clone the repository
git clone <your-repo-url>
cd capstone
```

### Step 2: Set Up Python Environment

It's good practice to use a virtual environment to keep dependencies isolated:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Your terminal prompt should now show (.venv) at the beginning
```

### Step 3: Install Python Dependencies

```bash
# Make sure you're in the project root and your virtual environment is activated
pip install -r requirements.txt

# This installs:
# - FastAPI (our backend web framework)
# - LangChain & LangGraph (AI agent orchestration)
# - PyTorch & Transformers (for FinBERT sentiment analysis)
# - yfinance, alpha-vantage (real stock market data)
# - pandas, numpy (data processing)
# - and many more...
```

**Note:** Installing PyTorch might take a few minutes as it's a large package.

### Step 4: Set Up Frontend

```bash
# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

# This installs:
# - Next.js (React framework)
# - Radix UI (UI components)
# - Recharts (for charts and graphs)
# - TypeScript (type safety)

# Go back to the root directory
cd ..
```

### Step 5: Environment Configuration

**Good news!** The `.env` file with your API keys is already set up in the root directory.

To verify it exists:
```bash
# On Windows:
dir .env

# On macOS/Linux:
ls -la .env
```

You should see the file listed. It contains your API keys for:
- Alpha Vantage (stock data)
- Tiingo (historical data)
- Finnhub (financial news)
- NewsAPI (news articles)
- OpenAI (AI models)
- LangChain (monitoring)

**IMPORTANT:** The `.env` file should NEVER be committed to git (it's in .gitignore).

### Step 6: Verify Frontend Environment

Check that the frontend environment file exists:

```bash
# On Windows:
dir frontend\.env.local

# On macOS/Linux:
ls -la frontend/.env.local
```

This file tells the frontend where to find the backend API (http://localhost:8000).

---

## Running the Application

You need to run TWO servers: the backend API and the frontend web app.

### Terminal 1: Start the Backend API

```bash
# Make sure you're in the project root with virtual environment activated
python -m pipelines.realtime.api

# You should see output like:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**What's happening:**
- FastAPI starts a web server on port 8000
- The API endpoints become available at http://localhost:8000
- You can see the interactive API docs at http://localhost:8000/docs

### Terminal 2: Start the Frontend

Open a NEW terminal window/tab:

```bash
# Navigate to the frontend directory
cd frontend

# Start the Next.js development server
npm run dev

# You should see output like:
#   ▲ Next.js 15.5.4
#   - Local:        http://localhost:3000
#   - Ready in 2.3s
```

**What's happening:**
- Next.js starts a development server on port 3000
- The website compiles and becomes available
- Hot-reloading is enabled (changes refresh automatically)

### Access the Application

Open your web browser and go to:
```
http://localhost:3000
```

You should see the StockSense homepage!

---

## Testing the System

### Test 1: Basic Data Fetching

Let's make sure we can fetch real stock data:

```bash
# In a new terminal, activate your virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Run the data fetch test
python -c "
from pipelines.realtime.sp500_data_service import get_sp500_data_service

# Create the data service
service = get_sp500_data_service()

# Fetch data for Apple
print('Fetching AAPL data...')
data = service.get_single_ticker_data('AAPL')

print(f'✓ Successfully fetched {len(data[\"market_data\"])} days of market data')
print(f'✓ Got {len(data[\"fundamentals\"])} fundamental metrics')
print(f'✓ Found {len(data[\"news_data\"])} news articles')
print('✓ Data fetching works!')
"
```

**Expected output:**
```
Fetching AAPL data...
✓ Successfully fetched 63 days of market data
✓ Got 15 fundamental metrics
✓ Found 10 news articles
✓ Data fetching works!
```

### Test 2: FinBERT Sentiment Analysis

Test that the AI sentiment analyzer works:

```bash
python -c "
from pipelines.realtime.sentiment.finbert import create_sentiment_analyzer

# Create analyzer
analyzer = create_sentiment_analyzer()

# Test article
test_article = {
    'title': 'Apple reports record earnings, stock surges',
    'content': 'Apple Inc. exceeded analyst expectations with strong iPhone sales.'
}

# Analyze sentiment
result = analyzer.process_news_articles([test_article])

print(f'✓ Sentiment: {result[\"current\"]}')
print(f'✓ Score: {result[\"score\"]:.2%}')
print(f'✓ Trend: {result[\"trend\"]}')
print('✓ FinBERT sentiment analysis works!')
"
```

**Expected output:**
```
✓ Sentiment: positive
✓ Score: 85%
✓ Trend: improving
✓ FinBERT sentiment analysis works!
```

### Test 3: Full Analysis via API

Test the complete stock analysis workflow:

```bash
# Make sure your backend is running, then:
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "user_tier": "premium"}'
```

You should get a JSON response with forecast, sentiment, smart money data, and explanations.

### Test 4: Frontend UI Test

1. Go to http://localhost:3000
2. Enter a stock ticker (e.g., "AAPL")
3. Click "Analyze"
4. You should see:
   - Forecast with confidence percentage
   - Charts showing daily probabilities
   - Key metrics (revenue growth, profit margins, etc.)
   - News sentiment analysis
   - Plain English explanation

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch==2.5.1 transformers==4.36.2
```

### Problem: Backend won't start - "Address already in use"

**Solution:** Something else is using port 8000
```bash
# On Windows:
netstat -ano | findstr :8000
# Find the PID and kill it in Task Manager

# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

### Problem: Frontend shows "Failed to fetch"

**Solutions:**
1. Make sure the backend is running on port 8000
2. Check `frontend/.env.local` has: `NEXT_PUBLIC_BACKEND_API_BASE=http://localhost:8000`
3. Restart both servers

### Problem: "No API key found"

**Solution:**
1. Check that `.env` exists in the project root
2. Verify it contains your API keys
3. Restart the backend server to load new environment variables

### Problem: Rate limit errors

**Solution:** Free tier API limits:
- Alpha Vantage: 500 calls/day
- NewsAPI: 100 requests/day
- Finnhub: 60 calls/minute

Wait a bit or use different tickers to spread out requests.

### Problem: FinBERT is slow

**Explanation:** FinBERT is a large neural network model. The first time you use it:
1. It downloads the model (can be 400+ MB)
2. It loads the model into memory
3. This can take 30-60 seconds

Subsequent requests are much faster!

---

## Understanding the Data Flow

Here's how data flows through the system when you analyze a stock:

```
1. User enters "AAPL" in frontend
   └─> Frontend sends POST /api/analyze {"ticker": "AAPL"}

2. Frontend proxy forwards to backend
   └─> POST http://localhost:8000/analyze

3. Backend API (api.py) receives request
   └─> Calls run_stocksense_analysis()

4. LangGraph Workflow starts:

   a. Validation Agent
      └─> Checks ticker is valid

   b. Data Collection
      └─> YahooFinanceProvider.get_historical_data("AAPL")
      └─> Gets last 3 months of price data from Yahoo Finance API
      └─> YahooFinanceProvider.get_fundamentals("AAPL")
      └─> Gets P/E ratio, revenue growth, margins, etc.
      └─> YahooFinanceProvider.get_news_data("AAPL")
      └─> Gets recent news headlines

   c. Prediction Agent
      └─> Takes market data + fundamentals
      └─> Runs ML model to predict Up/Down/Neutral
      └─> Calculates confidence score
      └─> Generates 30-day probability forecast

   d. Sentiment Agent
      └─> Takes news articles
      └─> Runs FinBERT on each headline
      └─> FinBERT returns: positive, negative, or neutral
      └─> Aggregates scores and identifies trend

   e. Smart Money Agent (Premium only)
      └─> Fetches institutional ownership data
      └─> Gets insider trading activity
      └─> Retrieves congressional disclosures

   f. Explanation Agent
      └─> Takes prediction + sentiment + smart money
      └─> Uses GPT-4 to write plain English summary
      └─> Explains the "why" behind the prediction

5. Response flows back:
   └─> Backend returns JSON with all results
   └─> Frontend receives and renders:
       - Charts (Recharts library)
       - Metrics cards (Radix UI)
       - Explanations (formatted text)
```

---

## Next Steps

Now that you have the system running:

1. **Try different stocks:** MSFT, GOOGL, TSLA, etc.
2. **Switch user tiers:** See how features change (Basic → Registered → Premium)
3. **Add to watchlist:** Use the watchlist feature (Registered+)
4. **Batch analysis:** Analyze multiple stocks at once (Premium)
5. **Export data:** Download CSV of predictions (Premium)

---

## Development Tips

### Hot Reloading

Both servers support hot reloading:
- **Backend:** FastAPI auto-reloads when you change Python files
- **Frontend:** Next.js auto-refreshes when you change TypeScript/React files

### Debugging

Enable debug mode for more detailed logs:

```python
# In your code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### API Documentation

FastAPI provides interactive docs:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test API endpoints directly from your browser!

---

## Architecture Overview

```
StockSense/
├── pipelines/
│   ├── realtime/              # Real-time analysis pipeline
│   │   ├── api.py            # FastAPI application (main backend entry)
│   │   ├── agents.py         # AI agents (Prediction, Sentiment, etc.)
│   │   ├── langgraph_workflow.py  # Agent orchestration with LangGraph
│   │   ├── data_sources.py   # Data provider abstractions (Strategy pattern)
│   │   ├── sp500_data_service.py  # S&P 500 data service
│   │   ├── sentiment/
│   │   │   └── finbert.py    # FinBERT sentiment analyzer
│   │   └── models/
│   │       └── forecaster.py # ML prediction model
│   └── batch/                 # Batch data ingestion
│       └── historical/        # Historical data fetchers
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx      # Main homepage
│       │   └── api/          # Next.js API routes (proxy to backend)
│       └── components/       # Reusable UI components
├── tests/                     # Test suite
├── .env                      # Backend environment (API keys) - NOT in git
├── .env.example              # Template for .env
└── frontend/.env.local       # Frontend environment - NOT in git
```

---

## Support

If you run into issues:

1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review the logs in your terminal
3. Check the browser console (F12) for frontend errors
4. Verify your `.env` file has all required API keys

---

**You're all set! Happy analyzing! 📈**
