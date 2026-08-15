# myQuant Agent Notes

This repository centers on one unified stable runtime through `QuantInvestor`,
the `quant-investor` CLI, and responsibility-named contracts, system, factor,
intelligence, and mainline packages. Keep repairs small, offline by default,
and compatible with the stable public CLI/API contracts.

## Boundaries

- Do not call live Tushare, yfinance, LLM, broker, or execution APIs during local
  verification unless a task explicitly requests a live run.
- Stable helpers must be importable and testable without external credentials.
- Preserve `research run`, `system verify`, `system status`, `system activate`,
  `factor status`, `research compile-evidence`, `research readiness`, `research
  inspect`, `research forward`, `research evaluate`, `market maintain`, `market
  analyze`, `market run`, and `market backtest`. `market download` remains the
  compatibility alias for older maintenance callers.
- `system activate` is the only normal `_active.json` writer. It requires an
  exact validated immutable generation and filesystem write permission. Read,
  verify, status, factor, and research commands cannot activate it.
- Keep review-layer LLM behavior advisory-only; deterministic control-chain
  gates and risk vetoes remain authoritative.
- Removed secondary commands and imports must fail explicitly. Do not restore a
  compatibility executable, dynamic-import fallback, latest-result scan, stale
  substitution, or any retired factor-state surface.

## Recommended Local Checks

Use a shell-safe expansion and select the relevant contract, system, factor,
intelligence, mainline, migration, and CLI tests from the unified set:

```bash
shopt -s nullglob
unified_tests=(tests/unit/test_unified_*.py)
if (( ${#unified_tests[@]} == 0 )); then
  echo "No unified runtime tests were found."
  exit 1
fi
pytest "${unified_tests[@]}" -v
```

For a broad change, run the full CI equivalent: `uv run pytest tests/unit -q`,
then the stable contracts/system/factor/intelligence/mainline/CLI flake8, Black,
and mypy checks in `.github/workflows/ci-cd.yml`.
