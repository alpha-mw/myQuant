# CN Aggressive Tracker Split Notes

Scope for a future split of
`quant_investor/monitoring/cn_aggressive_portfolio_tracker.py`.

## Current Responsibility Clusters

- Review mainline and DAG compliance: `_run_unified_review_mainline_for_holdings`,
  `_build_dag_four_branch_compliance`, DAG artifact extraction, candidate-level
  v13 DAG helpers.
- Market and execution inputs: realtime quote parsing/fetching, canonical
  completeness checks, previous-record loading, manual ledger resolution.
- Portfolio policy: trailing take-profit review, risk-reduction sell gates,
  position action/reason classification, rebalance and switch-plan builders.
- Reporting and artifacts: theme-pool report lines, data snapshot lines, notes
  payload, output writers, manual execution files.
- CLI surface: `run_tracker`, `build_parser`, `main`.

## Proposed Package Shape

- `monitoring/cn_aggressive/review_mainline.py` for DAG/review integration.
- `monitoring/cn_aggressive/market_inputs.py` for quotes, completeness, and
  previous/manual record loading.
- `monitoring/cn_aggressive/portfolio_policy.py` for trailing-profit, sell-gate,
  switch-plan, and order application logic.
- `monitoring/cn_aggressive/reporting.py` for markdown, diagnostics, and notes
  payload assembly.
- Keep `cn_aggressive_portfolio_tracker.py` as a thin CLI compatibility shim
  until downstream scripts and schedules import the package modules directly.

## Non-Goals

- Do not change formal review semantics, ledger provenance, realtime quote
  requirements, or output file schemas during the split.
- Do not mix this with dashboard export changes or theme-pool policy changes.

## Validation

- Preserve existing unit coverage for tracker, dashboard export, theme-pool
  diagnostics, and candidate review.
- Add import-level tests for the new package modules before moving call sites.
- Run the repository global contract subset after each extraction step.
