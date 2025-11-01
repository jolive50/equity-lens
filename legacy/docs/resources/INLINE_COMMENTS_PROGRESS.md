# Inline Comment Coverage

This document tracks the status of human-friendly inline comments across the core
Python modules. The goal is to keep every major component understandable to a
college-level reader by explaining what the code does, why it exists, and how data
moves through it.

## Coverage Snapshot

| Module / Area                                      | Status     | Notes                                                                                 |
| -------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| `pipelines/realtime/api.py`                        | Complete   | Every endpoint and helper has descriptive comments outlining request flow and errors. |
| `pipelines/realtime/agents.py`                     | Complete   | Agents document ML vs LLM fallbacks, data contracts, and orchestration responsibilities. |
| `pipelines/realtime/langgraph_workflow.py`         | Complete   | Workflow builder and coordination states include step-by-step guides.                |
| `pipelines/realtime/models/forecaster.py`          | Complete   | Feature engineering and probabilistic outputs fully annotated.                       |
| `pipelines/realtime/sentiment/finbert.py`          | Complete   | FinBERT pipeline comments cover batching, filtering, and fallback logic.             |
| `pipelines/realtime/data_adapters.py`              | In review  | Class docstrings exist; expand per-adapter examples and failure modes.               |
| `pipelines/realtime/data_sources.py`               | In review  | Requires clearer explanation of caching and rate-limiting helpers.                   |
| `pipelines/realtime/sp500_data_service.py`         | In review  | Surface data quality heuristics and ticker selection rationale.                      |
| `pipelines/realtime/repository.py`                 | Pending    | Add context around JSON persistence, schema evolution, and locking assumptions.      |
| `tests/`                                           | Pending    | Minimal inline commentary-focus on describing fixtures and deterministic stubs.      |

Status legend: *Complete* (no immediate action), *In review* (needs one more
pass for clarity), *Pending* (significant additions still required).

## Comment Style Guidelines

When adding new comments:

1. State the **what** in plain language.
2. Provide the **why** - the business or technical justification.
3. Outline the **how** with the key steps or algorithms.
4. Clarify the **data** moving in and out, including edge cases.
5. Keep comments concise; prioritize clarity over verbosity.

Example:

```python
# What: Apply the LangGraph workflow to orchestrate the agents.
# Why: Keeps the API handler lightweight and reusable across CLI/tests.
# How: Build the state machine once, then invoke it with the request payload.
result = workflow.invoke({"ticker": ticker, "user_tier": user_tier})
```

## Next Steps

- [ ] Finish documenting repositories and data source helpers.
- [ ] Add commentary to test doubles to help new contributors understand fixtures.
- [ ] Re-run this audit after major refactors to keep the table accurate.
