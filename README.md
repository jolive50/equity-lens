# StockSense

StockSense is an end-to-end platform that turns raw market information into clear, plain-language insights for retail investors. The system combines real-time data ingestion, LangGraph-powered multi-agent analysis, and a Next.js front-end to deliver explainable forecasts with confidence bands, sentiment context, and smart-money signals.

## Core Capabilities

- **Probabilistic forecasts** with calibrated 95% confidence horizons
- **Sentiment intelligence** using FinBERT over live news streams, with semantic search and historical context augmentation via VectorStore
- **Smart money tracing** across institutional filings, insider trades, and congressional disclosures
- **Tier-aware explanations** so basic users see essentials while premium users receive deeper metrics and recommended actions
- **Pluggable data adapters** (Alpha Vantage, Tiingo, Finnhub, Yahoo Finance) that follow SOLID/OOP principles for maintainability

## Quick Start

### Prerequisites

- **Python 3.11+** ([download](https://www.python.org/))
- **Node.js 18+** ([download](https://nodejs.org/))

### Automated Setup & Launch

**Windows:**
```bash
startup.bat
```

**macOS/Linux:**
```bash
./startup.sh
```

The startup script will:
1. Check for Python and Node.js
2. Create a virtual environment (if needed)
3. Install all dependencies (Python and Node.js)
4. Launch both backend and frontend servers
5. Open new terminal windows for each service

Once started, open your browser to:
- **Frontend:** http://localhost:3000
- **API Documentation:** http://localhost:8000/docs

### Manual Setup (if preferred)

<details>
<summary>Click to expand manual setup instructions</summary>

#### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env  # Then edit .env with your API keys
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

#### 3. Run the Services

**Terminal 1 - Backend:**
```bash
python -m pipelines.realtime.api
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

</details>

## Architecture

| Layer      | Technology                                      | Responsibility                               |
| ---------- | ------------------------------------------------ | --------------------------------------------- |
| Frontend   | Next.js (App Router), TypeScript, Radix UI       | User experience, dashboards, client auth      |
| API        | FastAPI, Pydantic, LangGraph workflow            | Request validation, agent orchestration       |
| Agents     | LangGraph-driven multi-agent prediction pipeline, VectorStore integration | Forecasting, sentiment (with semantic search & historical context), explanation, alerts   |
| Data       | Adapter + repository pattern                     | Vendor integrations and durable storage       |
| ML Assets  | Forecaster models and training utilities         | Time-series forecasting with risk metrics     |

## Project Structure

```
capstone/
├── data/                 # Training data and fundamentals (not in git)
├── docs/                 # Architecture, design notes, data inventory
├── frontend/             # Next.js client application
├── model/                # Forecasting models and training utilities
├── pipelines/            # FastAPI service, LangGraph workflow, data adapters
├── scripts/              # Operational helpers (data collection, test runner)
├── tests/                # Pytest suite (unit + integration)
├── startup.bat           # Automated Windows startup script
├── startup.sh            # Automated macOS/Linux startup script
├── requirements.txt      # Backend dependencies
└── pytest.ini            # Pytest configuration
```

## Testing & Quality

```bash
# Run all quality checks (formatting, linting, type checks, tests)
python scripts/run_tests.py

# Run only pytest
pytest

# Frontend tests
cd frontend && npm test
```

Coverage thresholds are defined in [pytest.ini](pytest.ini).

## Development Workflows

### Data Collection & Model Training

For team members who need to set up training data:

```bash
# Download historical price data (Kaggle S&P 500 dataset)
python scripts/download_kaggle_data.py

# Prepare specific stocks for training
python scripts/prepare_kaggle_data.py --tickers AAPL MSFT GOOGL

# Download fundamental financial data
python scripts/download_fundamentals.py --dataset sp500

# Build training features
python scripts/build_training_dataset.py --ticker-file data/training/ticker_list.txt

# Train the forecasting model
python scripts/train_forecaster.py

# Download FinBERT sentiment model (requires Hugging Face token)
python scripts/download_finbert.py --token hf_your_token_here
```

**Note:** Training data is excluded from git to keep the repository lightweight. All data can be reproduced using the scripts above.

### Working with the API

The FastAPI backend exposes several endpoints:

- `POST /analysis` - Get multi-agent analysis for a ticker
- `GET /watchlist` - Retrieve user's watchlist
- `POST /watchlist` - Add ticker to watchlist
- `GET /history/{ticker}` - Historical analysis results
- `GET /alerts` - Active alerts for user
- `POST /export` - Export analysis as PDF/JSON

Visit http://localhost:8000/docs for interactive API documentation.

## Environment Configuration

### Backend (.env)

Required API keys for full functionality:

| Key                       | Purpose                                   | Get it from                              |
| ------------------------- | ----------------------------------------- | ---------------------------------------- |
| `ALPHA_VANTAGE_API_KEY`  | Historical price data                     | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `TIINGO_API_KEY`         | Alternate price and fundamentals feed     | [tiingo.com](https://www.tiingo.com/) |
| `FINNHUB_API_KEY`        | Real-time quotes and sentiment endpoints  | [finnhub.io](https://finnhub.io/) |
| `NEWSAPI_API_KEY`        | News ingestion for FinBERT                | [newsapi.org](https://newsapi.org/) |
| `OPENAI_API_KEY`         | Explanation agent (LLM narratives)        | [platform.openai.com](https://platform.openai.com/) |

Optional keys can remain blank; adapters gracefully skip providers that are not configured.

### Frontend (frontend/.env.local)

```bash
NEXT_PUBLIC_BACKEND_API_BASE=http://localhost:3000
```

Adjust if you proxy the API elsewhere.

## Documentation

- **[Implementation Status](docs/IMPLEMENTATION_STATUS.md)** - Current progress and roadmap
- **[Data Inventory](docs/DATA_INVENTORY.md)** - Complete list of available datasets and metrics
- **[Technical Documentation](docs/resources/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md)** - Architecture deep-dive
- **SentimentAgent & VectorStore Integration:**
   - SentimentAgent now stores all processed news articles in VectorStore for semantic search and historical context.
   - Semantic similarity search enables finding related news for a ticker using embeddings.
   - Sentiment analysis is enhanced by blending current and historical sentiment scores from similar news articles.
   - See `pipelines/realtime/agents.py` and `pipelines/realtime/storage/vector_store.py` for implementation details.

## Contributing

1. Fork and clone the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Run `python scripts/run_tests.py` before pushing
4. Open a pull request describing the change and any follow-up tasks

## Troubleshooting

### Startup Scripts

**"Python/Node.js not found"**
- Ensure Python 3.11+ and Node.js 18+ are installed and in your PATH
- Run `python --version` and `node --version` to verify

**"Failed to install dependencies"**
- Check your internet connection
- Try running `pip install -r requirements.txt` manually with verbose output
- For frontend: `cd frontend && npm install --verbose`

**"API calls failing with 401/403"**
- Verify the relevant API key exists in `.env`
- Check that the key is valid and hasn't expired

**"Frontend cannot reach backend"**
- Confirm backend is running on port 8000
- Check `NEXT_PUBLIC_BACKEND_API_BASE` in `frontend/.env.local`
- Ensure CORS is enabled (it is by default in development)

### Data & Models

**"Kaggle credentials not found"**
1. Get your API key from https://www.kaggle.com/settings
2. Save `kaggle.json` to:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (macOS/Linux)

**"FinBERT model download slow/stalled"**
- First download can take time (~440MB)
- Subsequent runs use local cache
- Check disk space and internet connection

## License & Support

StockSense is released under the MIT license. For support, open a GitHub issue or check the documentation in the [docs/](docs/) directory.

---

**Quick Links:**
- [Frontend README](frontend/README.md)
- [Data Directory Info](data/README.md)
- [Run Tests](scripts/run_tests.py)
