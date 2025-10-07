# Running StockSense Locally

This guide walks through everything you need to run the full StockSense stack on a
developer workstation.

## Quick Start Checklist

1. Create and activate a Python 3.11 virtual environment.
2. Install backend dependencies with `pip install -r requirements.txt`.
3. Install frontend dependencies with `npm install` inside `frontend/`.
4. Copy `.env.example` to `.env` and fill in API keys (Alpha Vantage, Tiingo, Finnhub, NewsAPI, OpenAI).
5. Copy `frontend/.env.local.example` to `frontend/.env.local` if you need to override defaults.
6. Start the FastAPI backend: `python -m pipelines.realtime.api`.
7. Start the Next.js frontend: `cd frontend && npm run dev`.
8. Visit `http://localhost:3000` for the UI and `http://localhost:8000/docs` for the API.

## Detailed Walkthrough

### 1. Verify Prerequisites

Open a terminal and confirm the required runtimes are available:

```bash
python --version   # 3.11 or newer
node --version     # 18 or newer
npm --version
git --version
```

### 2. Set Up the Backend Environment

```bash
python -m venv .venv
# Activate the virtual environment
#   Windows PowerShell: .\.venv\Scripts\Activate.ps1
#   macOS/Linux:       source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file at the project root (copy `.env.example` to start) and
populate it with your API credentials. At minimum you will need API keys for
Alpha Vantage, Tiingo, Finnhub, NewsAPI, and OpenAI.

### 3. Set Up the Frontend Environment

```bash
cd frontend
npm install
# Optional: copy .env.local.example to .env.local and adjust NEXT_PUBLIC_BACKEND_API_BASE
cd ..
```

By default the frontend expects the backend at `http://localhost:8000`.

### 4. Launch the Services

Use two terminals (or two panes in your terminal multiplexer):

**Terminal 1 - Backend**

```bash
python -m pipelines.realtime.api
```

You should see Uvicorn logs indicating the server is listening on port 8000.

**Terminal 2 - Frontend**

```bash
cd frontend
npm run dev
```

Next.js will compile and expose the UI at `http://localhost:3000`.

### 5. Smoke Test the Stack

1. Navigate to `http://localhost:3000`.
2. Submit a ticker (e.g. `AAPL`) and verify the dashboard loads.
3. Check API docs at `http://localhost:8000/docs` and invoke `/analysis` with a valid ticker.

## Helpful Commands

```bash
# Run the backend test suite, linting, and type checks
python scripts/run_tests.py

# Collect historical market data for offline training
python scripts/collect_training_data.py

# Run only pytest (inside the virtual environment)
pytest

# Run frontend tests
cd frontend && npm test
```

## Project Layout (abridged)

```
capstone/
|-- data/                        # Local data dumps (not committed)
|-- docs/                        # Architecture and progress documentation
|-- frontend/                    # Next.js application
|   |-- src/app/api/             # API route proxies for browser calls
|   |-- src/components/          # Reusable UI components
|   `-- package.json
|-- pipelines/
|   `-- realtime/
|       |-- api.py               # FastAPI entry point
|       |-- agents.py            # LangGraph-compatible agents
|       |-- langgraph_workflow.py# Orchestrated analysis workflow
|       |-- data_adapters.py     # External data integrations
|       |-- sentiment/finbert.py # FinBERT sentiment analyser
|       `-- models/forecaster.py # Probabilistic forecaster
|-- scripts/
|   |-- collect_training_data.py # Batch data collection helper
|   `-- run_tests.py             # Convenience test runner
|-- tests/                       # Pytest suites
|-- README.md
|-- SETUP.md
`-- requirements.txt
```

## Troubleshooting Tips

- Double-check `.env` and `frontend/.env.local` when the API returns 401/403 errors.
- PyTorch installs can take a few minutes; re-run `pip install -r requirements.txt` if the process is interrupted.
- Use `python -m pipelines.realtime.api --reload` during development if you want autoreload semantics.
- If the frontend cannot reach the backend, confirm both servers are running and that the base URL matches in `frontend/.env.local`.

## Where to Go Next

- Review `docs/resources/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md` for a deep dive into the architecture.
- Explore `pipelines/realtime/agents.py` to customise prompts, thresholds, or agent behaviour.
- Extend `scripts/collect_training_data.py` if you need additional historical features for model training.
