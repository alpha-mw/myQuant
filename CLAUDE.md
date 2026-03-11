# CLAUDE.md — AI Assistant Guide for myQuant

## Project Overview

**myQuant** is a professional-grade, seven-layer quantitative investment platform (V8.0) for China A-share and US equity markets. It combines classical factor investing with modern AI/ML: multi-LLM consensus (Claude, GPT-4o, DeepSeek, Gemini), genetic alpha mining, and Walk-Forward backtesting with no future data leakage.

---

## Repository Structure

```
myQuant/
├── scripts/unified/          # ★ V8.0 — ALL active development goes here
│   ├── quant_investor_v8.py  # Main pipeline orchestrator (entry point)
│   ├── multi_llm_ensemble.py # Layer 7: 4-LLM judge ensemble
│   ├── multi_model_debate.py # Layer 6: 5-expert debate
│   ├── risk_management_layer.py # Layer 5
│   ├── macro_terminal_tushare.py # Layer 4: CN/US macro signals
│   ├── enhanced_model_layer.py  # Layer 3: XGBoost/RF/SVM
│   ├── factor_analyzer.py   # Layer 2: IC/layering/turnover
│   ├── enhanced_data_layer.py   # Layer 1: OHLCV + fundamentals
│   ├── alpha_mining.py      # 3-layer alpha discovery
│   ├── portfolio_backtest.py # Walk-Forward backtest
│   ├── investment_report.py # Structured report generation
│   ├── logging_config.py    # Centralized loguru setup
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── logger.py            # Logger factory
│   ├── config.py            # Env var loading
│   ├── di_container.py      # Dependency injection
│   ├── event_bus.py         # Event-driven comms
│   ├── cache_manager.py     # API call caching
│   └── archive/             # V7 and earlier (do not modify)
├── tests/unit/              # 59 passing unit tests
├── skill/                   # Manus skill definitions (SKILL.md)
├── .github/workflows/ci-cd.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

**Important**: All legacy code lives in `scripts/v2.x/` through `scripts/v6.0/` and `scripts/unified/archive/`. Do **not** modify these directories.

---

## Seven-Layer Architecture

| Layer | File | Purpose |
|-------|------|---------|
| 1 | `enhanced_data_layer.py` | Data fetching, cleaning, standardization |
| 2 | `factor_analyzer.py` | Factor IC analysis, layered backtest, turnover |
| 3 | `enhanced_model_layer.py` | ML ensemble (XGBoost, RF, SVM, LSTM) |
| 4 | `macro_terminal_tushare.py` | Macro risk signals (4 CN modules, 5 US modules) |
| 5 | `risk_management_layer.py` | VaR/CVaR, position sizing, stress testing |
| 6 | `multi_model_debate.py` | 5 financial expert personas debate long/short |
| 7 | `multi_llm_ensemble.py` | 4-LLM judge consensus with disagreement risk signal |

Each layer is independently callable. The orchestrator in `quant_investor_v8.py` wires them together into `V8PipelineResult`.

---

## Development Setup

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
# Optional ML extras
pip install -e ".[ml]"

# 3. Configure environment
cp .env.example .env
# Edit .env: set TUSHARE_TOKEN, API keys for LLMs

# 4. Verify setup
pytest tests/unit/ -v
```

### Required Environment Variables (`.env`)

| Variable | Purpose | Required |
|----------|---------|---------|
| `TUSHARE_TOKEN` | Tushare financial data API | Yes (CN market) |
| `ANTHROPIC_API_KEY` | Claude LLM judge | Yes (Layer 7) |
| `OPENAI_API_KEY` | GPT-4o LLM judge | Yes (Layer 7) |
| `DEEPSEEK_API_KEY` | DeepSeek LLM judge | Yes (Layer 7) |
| `GEMINI_API_KEY` | Gemini LLM judge | Yes (Layer 7) |
| `DB_PATH` | SQLite database path | No (default: `data/stock_database.db`) |
| `LOG_LEVEL` | Logging verbosity | No (default: `INFO`) |
| `COMMISSION_RATE` | Backtest slippage | No (default: `0.0003`) |

---

## Running the System

### CLI (Recommended)

```bash
python scripts/unified/quant_investor_v8.py \
    --stocks 000001.SZ 600519.SH \
    --market CN \
    --capital 1000000 \
    --risk-level 中等
```

### Python API

```python
from scripts.unified.quant_investor_v8 import QuantInvestorV8

analyzer = QuantInvestorV8(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    enable_macro=True,
    enable_backtest=True,
)
result = analyzer.run()
print(result.final_report)
```

### Standalone Modules

```bash
# Macro risk terminal only
python scripts/unified/macro_terminal_tushare.py

# Alpha mining only
python scripts/unified/alpha_mining.py

# Walk-Forward backtest only
python scripts/unified/portfolio_backtest.py
```

---

## Code Conventions

### Naming

| Category | Convention | Example |
|----------|-----------|---------|
| Classes | PascalCase | `QuantInvestorV8`, `MultiLLMEnsemble` |
| Functions/methods | snake_case | `calculate_ic()`, `_call_api()` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_WEIGHT`, `MODEL_NAME` |
| Private | Leading `_` | `_logger`, `_compute_date_ic()` |
| Result dataclasses | `*Result` / `*Profile` suffix | `V8PipelineResult`, `FactorProfile` |

### Imports

```python
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Any

# Internal: use relative-style imports within scripts/unified/
from logger import get_logger
from exceptions import DataError, FactorError
```

### Logging — Always Use Loguru via Factory

```python
# ✅ Correct
from logger import get_logger
logger = get_logger("ModuleName")
logger.info("Processing {count} stocks", count=len(stocks))

# ❌ Wrong — do not use
import logging
print("debug message")
```

### Exception Handling — Use Custom Hierarchy

```python
# exceptions.py defines:
# QuantBaseError → DataError, FactorError, ModelError,
#                  RiskError, BacktestError, LLMError, ConfigError

# ✅ Correct
from exceptions import DataError
raise DataError(f"Failed to fetch data for {symbol}: {e}") from e

# ❌ Wrong
raise Exception("something went wrong")
```

### Type Hints — Full Annotations Required

```python
# ✅ All functions must have type annotations
def calculate_ic(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    ...
```

### Dataclasses for Results

Pass results between layers using typed `@dataclass` objects, never bare dicts:

```python
@dataclass
class LayerResult:
    signals: pd.DataFrame
    metadata: dict[str, Any]
    timestamp: str
```

---

## Testing

```bash
# Run all unit tests
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ -v --cov=scripts/unified --cov-report=html

# Run specific test file
pytest tests/unit/test_factor_layer.py -v
```

**Current status**: 59 tests passing, ~60% coverage.

When adding new features, add corresponding tests in `tests/unit/test_<module_name>.py`.

### Test Patterns

Tests use `unittest.mock.patch` to mock external APIs (Tushare, LLM calls):

```python
@patch("enhanced_data_layer.ts.pro_api")
def test_fetch_ohlcv(self, mock_api):
    mock_api.return_value.daily.return_value = sample_df
    ...
```

---

## CI/CD Pipeline

**File**: `.github/workflows/ci-cd.yml`
**Triggers**: Push to `main`/`develop`, PR to `main`

| Job | What it does |
|-----|-------------|
| `test` | flake8 lint → mypy type check → pytest on Python 3.10/3.11/3.12 matrix |
| `build` | Build package with hatchling, validate with twine |
| `model-training` | Model training placeholder (main only) |
| `deploy-docs` | GitHub Pages deployment (main only) |

**Pre-push checklist** (must pass locally before pushing):

```bash
flake8 scripts/unified/ --max-line-length=120
mypy scripts/unified/ --ignore-missing-imports
pytest tests/unit/ -v
```

---

## Key Design Patterns

| Pattern | Where Used | Notes |
|---------|-----------|-------|
| Factory | `logger.py` → `get_logger(name)` | Module-bound loggers |
| Singleton | `config.Config` | Single env config instance |
| Strategy | `multi_llm_ensemble.py` | Swap LLM judges independently |
| Observer | `event_bus.py` | Model training notifications |
| Builder | `V8PipelineResult` | Accumulate results across layers |
| DI Container | `di_container.py` | Loose coupling between layers |

---

## Common Pitfalls to Avoid

1. **Do not edit archive directories** (`scripts/unified/archive/`, `scripts/v*/`). These are historical snapshots.
2. **Do not use `print()` for logging** — use `get_logger()` from `logger.py`.
3. **Do not raise bare `Exception`** — use the custom hierarchy in `exceptions.py`.
4. **Do not write O(n²) loops** for time-series operations — use vectorized pandas/numpy. See the IC calculation fix in `factor_analyzer.py` as reference.
5. **Do not hardcode API tokens** — always read from `.env` via `config.py`.
6. **Do not skip type hints** — mypy runs in CI and will fail the build.
7. **Layer isolation**: Each layer should accept inputs and return a typed result dataclass. Do not import from higher layers.

---

## Adding a New Feature

1. **Identify the layer** (1–7) where the feature belongs.
2. **Create/extend the module** in `scripts/unified/`.
3. **Use existing infrastructure**: `get_logger()`, custom exceptions, `config.py`.
4. **Return typed dataclasses**, not raw dicts.
5. **Write unit tests** in `tests/unit/test_<module>.py`.
6. **Update `quant_investor_v8.py`** if the layer orchestration changes.
7. Run the full test/lint/type-check suite before committing.

---

## Documentation References

| Document | Location | Contents |
|----------|---------|---------|
| Main README | `README.md` | V8 overview, architecture diagram, quick start |
| Macro Guide | `scripts/unified/MACRO_RISK_GUIDE.md` | Market signal definitions, risk thresholds |
| Skill Definition | `skill/SKILL.md` | Manus skill spec (V2.5) |
| Skill Changelog | `skill/CHANGELOG.md` | Skill version history |
