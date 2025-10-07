# StockSense Setup Guide

Follow this guide the first time you prepare a development machine for StockSense.
For day-to-day usage refer to `RUNNING_INSTRUCTIONS.md`.

## 1. Clone the Repository

```bash
git clone <your-fork-or-original-repo-url>
cd capstone
```

You now have two major workspaces:

- `pipelines/realtime/` - Python services, models, and data adapters.
- `frontend/` - Next.js dashboards and client applications.

## 2. Install Required Tooling

Make sure the following are installed and discoverable on your PATH:

- Python 3.11 or newer
- Node.js 18 or newer (npm is bundled)
- Git

Double-check versions:

```bash
python --version
node --version
npm --version
git --version
```

## 3. Bootstrap the Python Environment

```bash
python -m venv .venv
# Activate the environment
#   PowerShell: .\.venv\Scripts\Activate.ps1
#   macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file installs FastAPI, LangGraph, FinBERT dependencies, and the
libraries used by the forecasting pipeline.

## 4. Configure Backend Environment Variables

Copy the template file and populate secrets:

```bash
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
```

Required keys:

| Key                       | Purpose                                   |
| ------------------------- | ----------------------------------------- |
| `ALPHA_VANTAGE_API_KEY`  | Historical price data                     |
| `TIINGO_API_KEY`         | Alternate price and fundamentals feed     |
| `FINNHUB_API_KEY`        | Real-time quotes and sentiment endpoints  |
| `NEWSAPI_API_KEY`        | News ingestion for FinBERT                |
| `OPENAI_API_KEY`         | Explanation agent (LLM narratives)        |

Optional keys can remain blank; the adapters gracefully skip providers that are
not configured.

## 5. Configure the Frontend Environment

Inside `frontend/` place environment variables used during development:

```bash
cd frontend
cp .env.local.example .env.local   # PowerShell: Copy-Item .env.local.example .env.local
```

`NEXT_PUBLIC_BACKEND_API_BASE` defaults to `http://localhost:8000`. Adjust it if
you proxy the API elsewhere. Additional frontend secrets (feature flags, analytics,
etc.) can be added here as needed.

Return to the project root when finished:

```bash
cd ..
```

## 6. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

The install step pulls down Next.js, Radix UI, Recharts, TypeScript tooling, and
testing libraries (Jest/React Testing Library).

## 7. Verify the Installation

Run the automated quality gate to ensure dependencies are wired correctly:

```bash
python scripts/run_tests.py
```

This command formats code, lints, runs mypy checks, and executes the pytest suite.
Address any errors before continuing.

## 8. (Optional) Prepare Training Data

If you plan to retrain or evaluate the probabilistic forecaster, populate the data
lake with fresh market data:

```bash
python scripts/collect_training_data.py
```

Data lands under `data/training/` and is ignored by Git.

## 9. Launch the Stack

Follow `RUNNING_INSTRUCTIONS.md` to start the backend (`python -m pipelines.realtime.api`)
and the frontend (`npm run dev` inside `frontend/`). Once both servers are live, navigate
to `http://localhost:3000`.

## Troubleshooting

- **ImportError for PyTorch/transformers** - ensure `pip install -r requirements.txt`
  completed within the activated virtual environment.
- **API calls failing with 401/403** - verify the relevant key exists in `.env`.
- **Frontend cannot reach backend** - confirm `NEXT_PUBLIC_BACKEND_API_BASE` matches the
  backend address and that CORS is enabled (it is by default in development).
- **Model downloads slow or stalled** - FinBERT and torch models can be large; give the
  first run time to download. Subsequent runs use the local cache under your home directory.

## Next Steps

- Review `docs/resources/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md` for architecture context.
- Explore `tests/` to understand unit and integration coverage.
- Configure Git hooks or CI jobs to run `scripts/run_tests.py` before every push.
