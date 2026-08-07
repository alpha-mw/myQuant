# myQuant Agent Notes

This repository centers on the `QuantInvestor` single mainline plus the market
maintenance, analysis, backtest, and workspace surfaces. Keep repairs small,
offline by default, and compatible with the existing public CLI/API contracts.

## Boundaries

- Do not call live Tushare, yfinance, LLM, broker, or execution APIs during local
  verification unless a task explicitly requests a live run.
- Treat staged upgrade modules as offline/pure-helper by default. They should be
  importable and testable without external credentials.
- Preserve the current `research run`, `market maintain`, `market analyze`,
  `market run`, and `market backtest` commands. `market download` remains a
  compatibility alias for older callers.
- Keep review-layer LLM behavior advisory-only; deterministic control-chain
  gates and risk vetoes remain authoritative.

## Recommended Local Checks

- `pytest tests/unit/test_v17_mainline_runtime.py -v`
- `pytest tests/unit/test_v17_public_python.py -v`
- `pytest tests/unit/test_v17_public_cli.py -v`
- `pytest tests/unit/test_tushare_url_defaults.py -v`
- `pytest tests/unit/test_fundamental_provider_contract.py -v`
- `pytest tests/unit/test_fundamental_live_fetch_resilience.py -v`
- `pytest tests/unit/test_fundamental_generation_promotion.py -v`
- `pytest tests/unit/test_v17_v4_forward_evidence.py -v`

For staged upgrade work, run `PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh`.
