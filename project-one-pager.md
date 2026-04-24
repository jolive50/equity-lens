# Equity Lens MVP

Local-first stock analysis workflow that combines market data, transformer sentiment, and agent orchestration behind a FastAPI and Next.js interface.

## Project overview
Equity Lens is a capstone MVP that analyzes a stock ticker through a multi-step backend workflow and returns prediction, sentiment, confidence, and explanation outputs through an API consumed by a Next.js frontend. The backend composes prediction, sentiment, reflection, and explanation agents in a LangGraph state machine, then exposes results through `/analyze` with Pydantic-validated request and response models. The project is technically interesting because it combines classic data fetching, ML model wrappers, workflow orchestration, caching, and retrieval surfaces in one local development stack. It also documents and implements both happy-path analysis and fallback behavior when models, APIs, or optional services are unavailable.

## What I built
- **Orchestrated** a LangGraph workflow that validates inputs, fetches market and news data, runs prediction and sentiment analysis, applies reflection checks, and generates final explanations before formatting a structured response payload.
- **Integrated** persistence layers that cache price and news data in SQLite and index normalized news content in ChromaDB, then reuse those stores for cache-aware fetches and explanation context retrieval.
- **Connected** a Next.js client flow to the backend through a local API route and interactive UI components for ticker input, loading/error states, and result rendering for prediction, sentiment, explanation, and quality notes.

## Technical highlights
- FastAPI backend defines a typed API contract (`AnalysisRequest`, `AnalysisResponse`, nested result models), validation rules for ticker and user tier, CORS for local frontend origin, and centralized exception handling.
- Workflow configuration is runtime-driven via YAML and supports model selection, ensemble toggles, weighting strategies, confidence thresholds, and optional Alpha Vantage news blending.
- Prediction layer includes LSTM, GRU, and gradient-boost wrappers behind a shared interface, with ensemble strategies (`weighted_average`, `simple_average`, `voting`) and deterministic fallbacks when trained weights are unavailable.
- Sentiment layer supports FinBERT and RoBERTa via TensorFlow transformers, DeBERTa via PyTorch transformers, and an ensemble vote path that can blend model outputs with Alpha Vantage sentiment labels.
- Data ingestion includes Yahoo Finance price history and fundamentals, Yahoo Finance news normalization and body extraction, and optional Alpha Vantage news enrichment with normalized schema fields.
- Operational tooling includes a single launcher script to run backend and frontend together, plus structured logging setup with rotating files and optional LangChain tracing configuration.

## Quality and engineering discipline
- SQLite schema enforces uniqueness and retrieval indexes for cached prices, cached news, and historical analysis results, and the database module provides explicit cache freshness checks and history retrieval methods.
- Reflection logic validates prediction bounds, prediction-sentiment alignment, and market data freshness, and can downgrade confidence and emit issue lists into API responses.
- The repository includes unit, integration, and e2e test suites covering key modules such as database operations, API response structure, vector store behavior, and workflow assembly, with many scenarios mocked for determinism.
- Important limitation: many tests are intentionally skipped placeholder cases, there is no `.github/workflows` CI pipeline in this repository snapshot, and the frontend API route returns mock analysis data when the local FastAPI service is unavailable.

## Tech stack
Backend/ML: Python, FastAPI, Pydantic, LangGraph, LangChain, TensorFlow, PyTorch, Transformers, XGBoost, scikit-learn; Data/Storage: yfinance, requests, SQLite, ChromaDB, pandas, NumPy; Frontend: Next.js, React, TypeScript; Testing: pytest, FastAPI TestClient.

This project best demonstrates early-career strengths in backend workflow composition, applied ML integration, data persistence, and practical full-stack system assembly with explicit fallback paths.
