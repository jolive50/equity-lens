- Claim: Equity Lens is a local-first capstone MVP for stock analysis with prediction and sentiment outputs.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/AGENT.md`, `/home/runner/work/equity-lens/equity-lens/docs/ROLE_DIVISION.md`, `/home/runner/work/equity-lens/equity-lens/api/main.py`
  - Confidence: docs-supported

- Claim: Backend uses a LangGraph workflow that sequences validation, data fetch, prediction, sentiment, reflection, and explanation.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/coordinator/workflow.py`
  - Confidence: direct implementation

- Claim: API exposes `/analyze` with typed request/response schemas, validation, and error handling.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/api/main.py`, `/home/runner/work/equity-lens/equity-lens/api/models.py`
  - Confidence: direct implementation

- Claim: SQLite caching is used for prices, news, and analysis history with schema-level uniqueness and indexes.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/storage/database.py`, `/home/runner/work/equity-lens/equity-lens/db/schema.sql`, `/home/runner/work/equity-lens/equity-lens/coordinator/workflow.py`
  - Confidence: direct implementation

- Claim: ChromaDB is integrated for article indexing, similarity search, and ticker-level statistics.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/storage/vector_store.py`, `/home/runner/work/equity-lens/equity-lens/coordinator/workflow.py`
  - Confidence: direct implementation

- Claim: Frontend includes ticker form, loading/error states, and result rendering, and communicates through a local Next.js API route.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/frontend/src/app/page.tsx`, `/home/runner/work/equity-lens/equity-lens/frontend/src/components/AnalysisForm.tsx`, `/home/runner/work/equity-lens/equity-lens/frontend/src/components/ResultsDisplay.tsx`, `/home/runner/work/equity-lens/equity-lens/frontend/src/app/api/analyze/route.ts`
  - Confidence: direct implementation

- Claim: Workflow behavior is configuration-driven through YAML, including ensemble toggles, model lists, and confidence settings.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/coordinator/config.py`, `/home/runner/work/equity-lens/equity-lens/config.yaml`, `/home/runner/work/equity-lens/equity-lens/coordinator/workflow.py`
  - Confidence: direct implementation

- Claim: Prediction stack has shared interface plus LSTM, GRU, GradientBoost, and configurable ensemble with fallback logic.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/models/prediction/base_predictor.py`, `/home/runner/work/equity-lens/equity-lens/models/prediction/lstm_model.py`, `/home/runner/work/equity-lens/equity-lens/models/prediction/gru_model.py`, `/home/runner/work/equity-lens/equity-lens/models/prediction/gradient_boost_model.py`, `/home/runner/work/equity-lens/equity-lens/models/prediction/ensemble.py`
  - Confidence: direct implementation

- Claim: Sentiment stack combines TensorFlow transformer wrappers, a PyTorch DeBERTa wrapper, and ensemble blending with optional Alpha Vantage feed.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/models/sentiment/finbert_model.py`, `/home/runner/work/equity-lens/equity-lens/models/sentiment/roberta_model.py`, `/home/runner/work/equity-lens/equity-lens/models/sentiment/deberta_model.py`, `/home/runner/work/equity-lens/equity-lens/models/sentiment/ensemble.py`, `/home/runner/work/equity-lens/equity-lens/models/sentiment/alpha_vantage_sentiment.py`, `/home/runner/work/equity-lens/equity-lens/requirements.txt`
  - Confidence: direct implementation

- Claim: Data ingestion uses Yahoo Finance for prices/fundamentals and normalized Yahoo Finance news, with optional Alpha Vantage enrichment.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/data/fetchers/price_data.py`, `/home/runner/work/equity-lens/equity-lens/data/fetchers/news_data.py`, `/home/runner/work/equity-lens/equity-lens/coordinator/workflow.py`, `/home/runner/work/equity-lens/equity-lens/models/sentiment/alpha_vantage_sentiment.py`
  - Confidence: direct implementation

- Claim: Logging is centralized with rotating files and optional LangChain tracing toggles.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/utils/logging_config.py`, `/home/runner/work/equity-lens/equity-lens/api/main.py`
  - Confidence: direct implementation

- Claim: Test structure spans unit, integration, and e2e suites, but many tests are skipped placeholders.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/tests/README.md`, `/home/runner/work/equity-lens/equity-lens/tests/unit/test_agents.py`, `/home/runner/work/equity-lens/equity-lens/tests/unit/test_prediction_model.py`, `/home/runner/work/equity-lens/equity-lens/tests/integration/test_workflow.py`, `/home/runner/work/equity-lens/equity-lens/tests/e2e/test_full_analysis.py`
  - Confidence: direct implementation

- Claim: Frontend route returns mock response when backend is unavailable.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/frontend/src/app/api/analyze/route.ts`
  - Confidence: direct implementation

- Claim: No CI workflow files are present under `.github/workflows`.
  - Supporting file paths: `/home/runner/work/equity-lens/equity-lens/.github`
  - Confidence: config-supported
