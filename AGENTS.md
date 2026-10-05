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

For a focused change, select the relevant existing tests in
`tests/unit/test_unified_*.py` for the changed contract, system, factor,
intelligence, mainline, migration, or CLI responsibility. Do not run the entire
unified set merely to answer a status question or validate a documentation edit.

When the full unified set is warranted, run this explicitly through Bash:

```bash
bash <<'BASH'
shopt -s nullglob
unified_tests=(tests/unit/test_unified_*.py)
if (( ${#unified_tests[@]} == 0 )); then
  echo "No unified runtime tests were found."
  exit 1
fi
uv run pytest "${unified_tests[@]}" -v
BASH
```

For a broad change, run the full CI equivalent: `uv run pytest tests/unit -q`,
then the stable contracts/system/factor/intelligence/mainline/CLI flake8, Black,
and mypy checks in `.github/workflows/ci-cd.yml`.

## Cursor Cloud specific instructions

Install the locked runtime and dev tools with `uv sync --locked --extra dev`.
Python 3.13 comes from uv. Keep `uv` on the default `PATH` via `/usr/local/bin/uv`.
The offline checks also need the system packages `zsh` (daily slot launcher) and
`zstd` (strategy-record archive rehearsal). There is no boot-time service.

A fresh workspace has no `results/system/_active.json`. `quant-investor system
verify` exits 0 and reports `UNINITIALIZED` with blocker
`SYSTEM_ACTIVE_POINTER_ABSENT`. `quant-investor system status` exits 0 with
`status` `OK` and system capability `UNINITIALIZED`.

Daily launcher tests (`tests/unit/test_configured_source_launcher.py` and
`tests/unit/test_daily_launcher_dag_profile.py`) use the zsh test
`=~ '^[ -~]+$'`. Run those tests with `LC_ALL=C.UTF-8`. Under
`LC_ALL=en_US.UTF-8` that character range does not match printable ASCII, and
the launcher rejects valid relative paths as
`CN_DAILY_PRODUCTION_ARGUMENTS_INVALID`.

Environment checks stay offline. `system verify`, unit tests, the dashboard
`node --check` contract tests, and `uv build` do not need `TUSHARE_TOKEN` or
other API keys.
