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

## Fundamental Research v13.2

- `v13-frozen-20260707` remains an immutable performance baseline. Codex-backed
  fundamental research belongs to the prospective
  `v13.2-fundamental-research` version and must not rewrite, replay into, or be
  blended with decisions made before its explicit activation cutoff.
- The repository may implement and validate the CN-only external Codex
  request/import workflow in `off` or `shadow` mode during the current
  incubation. A validated dossier remains advisory and may never bypass
  Bayesian, RiskGuard, ICCoordinator, or PortfolioConstructor.
- `limited` or `production` score application requires the registered shadow
  gates, an explicit hash-bound activation record, and a separate Maxwell
  confirmation. Implementing or merging the feature is not production
  activation.
- Activation evidence must be v2 and reproducible from the private job,
  application, and longitudinal ledgers plus a hash-bound canonical-Parquet
  holdings snapshot. Caller-supplied counts are not activation evidence; a
  recent received-job validation rate below 80% with at least 10 samples
  automatically degrades score-affecting modes to `shadow`.
- Longitudinal activation dates must come from the deterministic companion
  analysis manifest produced by the real dual control-chain replay and, for
  NAV, the next-session strict-Parquet attribution producer. Manual metric JSON
  is not authoritative evidence.
- The dual replay must rebuild variant-specific fundamental verdicts, branch
  summaries and Bayesian records, must not reuse actual IC/Portfolio Master
  hints, and must persist the independently computed RiskGuard, IC, plan and
  portfolio decision. Import must bind requests/tasks to the prepare manifest,
  and current mode readiness must be recomputed at every consumption cutoff.
- Fundamental research artifacts are private ignored data. They must be
  atomically written mode `0600`, hash-bound, PIT-safe, and contain no account
  capital, cost basis, position weights, credentials, or absolute paths.

## Recommended Local Checks

- `pytest tests/unit/test_public_package_smoke.py -v`
- `pytest tests/unit/test_data_layer.py -v`
- `pytest tests/unit/test_forecast_snapshot_cache.py -v`
- `pytest tests/unit/test_llm_env_inventory.py -v`
- `pytest tests/unit/test_tushare_url_defaults.py -v`
- `pytest tests/integration/test_review_layer_timeout_budget.py -v`

For staged upgrade work, run `PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh`.
