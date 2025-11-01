# StockSense Remediation & Implementation Plan (No LLM Fallbacks)

This document is addressed to the implementation agent. Execute the tasks below
in order. Treat every reference to large-language-model fallbacks as a defect.

## 1. Enforce “no LLM fallback” policy across the codebase

1. **Audit runtime code**  
   - Files to inspect:  
     - `pipelines/realtime/agents.py` (all agent classes)  
     - `pipelines/realtime/langgraph_workflow.py`  
     - `pipelines/realtime/models/forecaster.py`  
     - `pipelines/realtime/sentiment/finbert.py`  
     - `pipelines/realtime/__init__.py` and `pipelines/realtime/smart_money/__init__.py`  
   - Remove every branch that catches ML/heuristic errors and falls back to an
     LLM (e.g., `build_openai_llm`, `llm.invoke`, prompt templates, etc.).
   - Replace those branches with explicit error raising (`RuntimeError`) so the
     API fails fast if the configured ML asset is unavailable.

2. **Purge documentation and comments**  
   - Update `docs/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md`, `README.md`,
     and agent docstrings to state “machine-learning only; execution aborts if
     the model is not available”.  
   - Remove any reference to “fallback”, “LLM prompt”, “GPT analysis”, or
     similar phrasing.  
   - Ensure tests (`tests/unit/test_agents.py`, `tests/test_ai_workflow*.py`)
     are rewritten to expect hard failures when the required model is missing.

3. **Delete the LLM helper code**  
   - Remove `build_openai_llm`, prompt templates, and runnable chains from
     `pipelines/realtime/agents.py`.  
   - Remove LangChain/LangGraph dependencies that only supported LLM usage once
     fallbacks are gone. Keep LangGraph if still needed for orchestration.

## 2. Machine-learning model training workflow

1. **Historical CSV sourcing**  
   - Document and automate fetching from: Alpha Vantage, Tiingo, Finnhub,
     or Yahoo Finance (already supported adapters) to produce CSV files under
     `data/training/raw/<ticker>.csv`.  
   - Create a script `scripts/build_training_dataset.py` that:  
     - Accepts tickers + date range.  
     - Retrieves prices, fundamentals, news sentiment (if needed).  
     - Writes feature-ready CSVs (`data/training/features/`) and label CSVs
       (`data/training/labels/`).

2. **Training pipeline**  
   - Add `scripts/train_forecaster.py` to:  
     - Load feature/label CSVs.  
     - Train the gradient boosting forecaster.  
     - Perform cross-validation and persist metrics to `model/metrics/`.  
     - Serialize the trained model (e.g., `model/artifacts/forecaster.pkl`).
   - Modify `pipelines/realtime/models/forecaster.py` to load the persisted model
     from disk at startup (configurable path via environment variable) and fail
     if missing.

## 3. FinBERT acquisition and packaging

1. **Download instructions**  
   - Update documentation (`docs/TECHNICAL_DOCUMENTATION_COLLEGE_LEVEL.md`,
     new `docs/ML_MODELS.md`) with exact commands to download ProsusAI/finbert
     from Hugging Face using `transformers` caching (`from_pretrained`).

2. **Offline packaging**  
   - Implement a script `scripts/download_finbert.py` that downloads the
     tokenizer and model, stores them under `model/finbert/`, and verifies the
     checksum.

3. **Runtime loading**  
   - Adjust `pipelines/realtime/sentiment/finbert.py` to load exclusively from
     the local cache and raise a descriptive error if the model is missing.
   - Remove the LLM fallback logic in sentiment agents (only FinBERT results are
     acceptable).

## 4. Data storage strategy

1. **Confirm JSON persistence**  
   - Review `pipelines/realtime/repository.py` to ensure JSON file IO supports
     all workloads (watchlists, history, alerts).  
   - Document the limitations (concurrency, growth) and outline mitigation
     (file locking, archiving).  
   - Remove references to PostgreSQL from README, docs, and config.

2. **Add JSON durability safeguards**  
   - Implement file locking (e.g., `portalocker` or cross-platform alternative)
     to prevent concurrent writes from corrupting data.  
   - Add automated backups or rotation under `data/backups/`.

## 5. SEC EDGAR data integration

1. **API client**  
   - Create `pipelines/realtime/smart_money/sec_edgar.py` to fetch 13F filings,
     insider trades, etc., via the SEC EDGAR API (use the
     [bulk data endpoints](https://www.sec.gov/edgar/sec-api-documentation)).
   - Respect SEC rate limits (add `User-Agent`, throttle requests).

2. **SmartMoneyAgent updates**  
   - Switch `SmartMoneyAgent` to consume SEC EDGAR data instead of LLM
     placeholders.  
   - Cache responses in JSON under `data/smart_money/`.

3. **Testing**  
   - Add integration tests with recorded fixtures to validate parsing logic.

## 6. Additional blockers before deployment

- **Configuration validation**: build a startup checklist that ensures every
  required model file or dataset exists before the API launches.
- **Logging/monitoring**: add structured logging to highlight missing assets,
  API failures, or stale datasets.
- **Documentation**: create `docs/OPERATIONS_RUNBOOK.md` covering model refresh,
  FinBERT updates, SEC EDGAR quotas, and JSON backup procedures.
- **CI updates**: configure automation to run the new training script on sample
  data, ensuring the pipeline does not regress.

### Summary

Eliminate all LLM fallbacks, ground the system entirely on deterministic ML
assets, and deliver the accompanying training, deployment, and data-ingestion
scripts. Treat any missing asset as a hard failure with actionable messaging.
