# Dependency Profiles

Phase 8 does not add third-party dependencies. This document records proposed future
dependency profiles so packaging can be tightened later without changing behavior now.

## Proposed Profiles

- `core`: deterministic research contracts, branch config, posterior primitives, and
  local artifact schemas.
- `cn-data`: Tushare and China-market data maintenance dependencies.
- `us-data`: US market and SEC/fundamental data dependencies.
- `llm`: LLM gateway, routing, usage tracking, and provider clients.
- `backtest`: backtest engines and performance-analysis tools.
- `model-training` (proposal only, not a current package extra): feature mining and
  heavier statistical learning dependencies.
- `web`: FastAPI, frontend build/runtime, and workspace UI dependencies.
- `dev`: pytest, mypy, flake8, formatting, and local CI tools.
- `audit`: stdlib-only staged-upgrade audit helpers plus existing project modules.

The current Phase 8 audit layer intentionally remains in the `core`/`audit` boundary:
it reads local JSON, JSONL, Markdown, and script files only, and it does not require
pandas, numpy, scipy, sklearn, web, LLM, provider, broker, or database dependencies.
