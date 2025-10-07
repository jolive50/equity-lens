# StockSense

StockSense is an end-to-end platform that turns raw market information into
clear, plain-language insights for retail investors.  The system combines
real-time data ingestion, LangGraph-powered multi-agent analysis, and a
Next.js front-end to deliver explainable forecasts with confidence bands,
sentiment context, and smart-money signals.

## Core Capabilities

- **Probabilistic forecasts** with calibrated 95 % confidence horizons.
- **Sentiment intelligence** using FinBERT over live news streams.
- **Smart money tracing** across institutional filings, insider trades, and
  congressional disclosures.
- **Tier-aware explanations** so basic users see essentials while premium users
  receive deeper metrics and recommended actions.
- **Pluggable data adapters** (Alpha Vantage, Tiingo, Finnhub, Yahoo Finance)
  that follow SOLID/OOP principles for maintainability.

## Architecture Overview

| Layer      | Technology                                      | Responsibility                               |
| ---------- | ------------------------------------------------ | --------------------------------------------- |
| Frontend   | Next.js (App Router), TypeScript, Radix UI       | User experience, dashboards, client auth      |
| API        | FastAPI, Pydantic, LangGraph workflow            | Request validation, agent orchestration       |
| Agents     | LangChain/LangGraph powered prediction pipeline  | Forecasting, sentiment, explanation, alerts   |
| Data       | Adapter + repository pattern                     | Vendor integrations and durable storage       |
| ML Assets  | Forecaster models and training utilities         | Time-series forecasting with risk metrics     |

## Getting Started

### 1. Prerequisites

- Python 3.11 or newer
- Node.js 18 or newer
- (Optional) PostgreSQL for production deployments
- API keys: Alpha Vantage, Finnhub, Tiingo, NewsAPI, OpenAI (stored in `.env`)

### 2. Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env                  # populate with your credentials
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the stack

```bash
# Backend (terminal 1, from repo root)
python -m pipelines.realtime.api

# Frontend (terminal 2, from frontend/)
npm run dev
```

Visit http://localhost:3000 for the UI and http://localhost:8000/docs for live
API documentation.

## Helpful Scripts

All project helpers live under `scripts/`:

- `scripts/collect_training_data.py` — pulls historical prices, fundamentals,
  and directional labels into `data/training/` using real Yahoo Finance data.
- `scripts/run_tests.py` — runs formatting, linting, type checks, and the pytest
  suite with coverage.

Invoke them with `python scripts/<name>.py`.  They require a configured virtual
environment because they use the same dependencies as the main pipeline.

## Testing & Quality

```bash
# Quick quality gate (includes formatting, linting, coverage)
python scripts/run_tests.py

# Direct pytest invocation
pytest

# Frontend tests
cd frontend && npm test
```

Coverage thresholds are defined in `pytest.ini`; adjust them deliberately if the
scope of testing changes.

## Project Structure

```
capstone/
├── data/                 # Batch/realtime data lake staging areas
├── docs/                 # Architecture, design notes, research artefacts
├── frontend/             # Next.js client application
├── model/                # Forecasting models and training utilities
├── pipelines/            # FastAPI service, LangGraph workflow, data adapters
├── scripts/              # Operational helpers (data collection, test runner)
├── tests/                # Pytest suite (unit + integration)
├── requirements.txt      # Backend dependencies
└── pytest.ini            # Pytest configuration (paths, markers, coverage)
```

## Data Collection Workflow

1. Configure API keys in `.env`.
2. Run `python scripts/collect_training_data.py` to build a training dataset.
3. Train or fine-tune forecasting models (see `model/` and `pipelines/realtime/models/`).
4. Deploy or evaluate the updated model with the FastAPI endpoints.

The training script classifies forward price moves (up/down/neutral) using
real close prices and saves the output as JSON so it can be consumed by
batch pipelines or notebooks.

## Operational Notes

- Keep virtual environments out of version control—` .venv311/` has been removed
  and `.gitignore` prevents future accidental commits.
- Environment variables should stay in `.env`; never commit secrets.
- The LangGraph workflow relies on available API keys for premium providers; the
  FastAPI health check will flag missing integrations.

## Contributing

1. Fork and clone the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Run `python scripts/run_tests.py` before pushing.
4. Open a pull request describing the change and any follow-up tasks.

## License & Support

StockSense is released under the MIT license.  For support open a GitHub issue
or check the documentation under the `docs/` directory.
