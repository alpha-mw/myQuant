# Factor Governance

## Objective

Phase 9 Pass 1 adds an offline factor governance layer for defining factor
contracts, validating existing backtest summaries against explicit thresholds,
recording admission decisions, and materializing a production factor library.

This pass is schema, lifecycle, store, and admission contract only. It does not
calculate factor matrices, parse expressions, run backtests, fetch data, call
LLMs, or connect factors to stock selection or portfolio construction.

## Lifecycle States

- `draft`: initial idea or incomplete definition.
- `research_candidate`: definition is complete enough for research review.
- `backtested`: an offline backtest result has been attached.
- `validated_research`: validation gates passed, but production is not approved.
- `paper_trading`: approved for offline or shadow monitoring.
- `production`: manually admitted to the production factor library.
- `deprecated`: retired and retained for audit history.
- `rejected`: rejected after validation or review.
- `disabled`: disabled by governance or operator decision.

## Production Factor Hard Rules

- Production requires an explicit `approve_production` admission decision.
- Production requires a validation report ID, admission decision ID, and
  `production_since` timestamp.
- Production library entries are sorted deterministically by `factor_id` and
  `factor_version`.
- Duplicate `factor_id` + `factor_version` pairs are rejected.
- Non-production entries are filtered out before building a production library.
- Production stock selection must not consume draft, research, rejected,
  deprecated, disabled, or paper-trading factors.

## Admission Gates

`evaluate_backtest_against_thresholds` evaluates an already-produced
`FactorBacktestResult` against `FactorValidationThresholds`.

Hard gates include:
- sample days
- coverage ratio
- rank IC mean
- ICIR
- IC t-stat
- after-cost Sharpe
- positive IC ratio
- positive after-cost top-bottom spread
- monotonicity when required
- point-in-time evidence when required

Warning gates include optional drawdown, turnover, and production-correlation
threshold checks. Missing hard-gate metrics fail the related gate.

By default:
- `pass` recommends `validated_research`.
- `warn` recommends `paper_trading`.
- `fail` recommends `rejected`.
- Admission proposals from a passing report approve paper trading only; production
  approval is intentionally manual in Pass 1.

## First-Pass Non-Goals

Phase 9 Pass 1 does not implement:
- factor matrix loading or persistence
- expression parsing or sandbox execution
- single-factor backtest execution
- correlation or redundancy analysis beyond recorded snapshots
- index-enhancement optimization
- live stock scoring
- `PortfolioConstructor` integration
- `RiskGuard` integration
- posterior, calibration, or overlay changes
- provider, market download, broker, LLM, or web/frontend behavior

## Matrix Data Contract And Expression Sandbox

Phase 9 Pass 2 adds an offline matrix contract and safe expression sandbox for
research-only factor primitives. Matrix data is shaped strictly as
`symbols x dates`, with rows aligned to symbols and columns aligned to ascending
ISO trade dates.

Standard fields include:
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`
- `industry`
- `benchmark_close`
- `benchmark_weight`

The helper layer can derive:
- `vwap`: `amount / volume`, returning missing values for missing or zero volume.
- `ret1`: one-period close-to-close return per symbol.
- `benchmark_ret`: one-period benchmark return broadcast across symbols when
  `benchmark_close` is present.

Expressions are parsed through a Python AST whitelist and never through arbitrary
`eval` or `exec`. Allowed names are matrix fields, standard derived fields, and
explicit extra fixture fields supplied by the caller. Calls are limited to the
safe operator whitelist, including time-series operators such as `ts_delay`,
`ts_mean`, `ts_std`, and `ts_corr`; cross-sectional operators such as `cs_rank`,
`cs_zscore`, `cs_indneut`, and `cs_booksize`; and basic elementwise arithmetic.

This pass remains local-only and deterministic:
- no live provider calls
- no Tushare, yfinance, LLM, broker, or web/frontend access
- no wiring into stock selection, `PortfolioConstructor`, or `RiskGuard`

`ts_delay` exists only as a safe expression operator in Pass 2. Execution delay
for research PnL is handled by the Pass 3 backtester described below.

## Single-Factor Long-Short Backtester

Phase 9 Pass 3 adds an offline single-factor backtester on top of the matrix
contract and expression artifacts. It consumes a `FactorMatrix`, a
`MatrixDataBundle`, and a `FactorBacktestConfig`, then writes research artifacts:
weight matrices, daily records, aggregate `FactorBacktestResult` summaries, and
single-factor run envelopes.

Delay alignment is explicit:
- `signal_date`: the factor value date used to form the research book.
- `execution_start_date`: `signal_date + delay_days`.
- `execution_end_date`: `execution_start_date + holding_period_days`.

For the default one-day holding period and `delay_days=1`, a signal on T forms
weights at T, starts execution on T+1, and records the forward return ending on
T+2. Weights are not shifted inside the weight constructor; the daily record
builder applies the alignment.

Execution prices are local matrix fields:
- `open`: uses the bundle `open` field.
- `close`: uses the bundle `close` field.
- `vwap`: uses the bundle `vwap` field, or derives it locally from
  `amount / volume` when absent.

The initial weighting method is `equal_quantile_booksize`. Scores are
`factor_value * expected_direction`, where `expected_direction` comes from
factor matrix metadata and defaults to positive. Long books use
`long_quantile`; short books use `short_quantile` only in long-short mode.
Selected long names receive `+1 / long_count`, selected short names receive
`-1 / short_count`, and net weights are the cellwise sum.

Turnover is defined as:

```text
0.5 * sum(abs(next_weight - previous_weight))
```

across the union of symbols. The first tradable record compares the current net
book to an all-zero book. Costs deduct
`turnover * decimal(transaction_cost_bps + slippage_bps + market_impact_bps)`
from the daily long-short return.

Current Pass 3 limitations:
- no slicing or regime analysis yet
- no advanced transaction cost or capacity model yet
- no factor correlation or redundancy analysis yet
- no production admission yet
- no live provider calls
- no stock selection, `PortfolioConstructor`, or `RiskGuard` wiring

## Metrics, Slicing, Cost/Capacity Validation

Phase 9 Pass 4 adds offline pre-admission validation helpers on top of
`SingleFactorBacktestRun`. These helpers summarize return/risk behavior, slice
the existing daily records, and record cost/capacity diagnostics without changing
admission defaults or production behavior.

Return metric summaries include:
- mean daily return
- annualized return
- annualized volatility
- Sharpe ratio
- maximum drawdown
- positive return ratio
- cumulative return

Slice validation supports:
- full-sample validation
- recent 1-year, 3-year, and 5-year trailing windows when enough local records
  exist
- regime slices supplied by the caller as a local `date -> regime_label` mapping

Each slice compares before-cost (`long_short_return`) and after-cost
(`after_cost_return`) series, optional excess return series, turnover metrics,
coverage/missing ratios, and average long/short book counts. Threshold breaches
produce deterministic warnings. Insufficient sample days fail the slice.

Cost and capacity diagnostics use supplied local matrix fields only. The first
capacity proxy uses the `amount` matrix as daily traded value, active weighted
symbols from the backtest weight matrix, and a configured maximum participation
rate. Daily capacity is approximated as:

```text
average active-symbol amount * max_participation_rate / max(turnover, epsilon)
```

Participation breaches compare requested daily turnover value
(`turnover * target_capital`) against allowed trade value. This is a simple
offline ADV/participation proxy, not a broker execution model.

The cost/capacity report also records:
- before-cost versus after-cost Sharpe
- average turnover
- configured total cost bps
- estimated average cost return
- cost drag ratio, computed as
  `max(0, before_cost_sharpe - after_cost_sharpe) / abs(before_cost_sharpe)`
  when both Sharpe values are positive
- average ADV from the local `amount` matrix
- participation breach ratio
- tradability ratio from the local `tradability_mask`
- coverage ratio from the aggregate backtest result

`build_enhanced_factor_validation_report` combines the existing aggregate
`FactorBacktestResult`, optional robustness report, and optional cost/capacity
report into an enhanced `FactorValidationReport`. It can recommend
`validated_research`, `paper_trading`, or `rejected`, but it never approves
production and does not replace the Pass 1 admission decision flow.

Current Pass 4 limitations:
- no correlation or redundancy analysis yet
- no portfolio contribution or index-enhancement validation yet
- no production admission
- no live provider calls
- no stock selection, `PortfolioConstructor`, or `RiskGuard` wiring
- the capacity model is a simplified offline proxy, not a broker execution model

## Correlation, Redundancy, And Portfolio Contribution

Phase 9 Pass 5 adds an offline incremental research layer for comparing a
candidate factor against existing production or research factor artifacts before
admission review. It remains a pure helper layer and does not approve factors or
alter any default runtime behavior.

The redundancy analysis can compare:
- after-cost return-series correlation against existing
  `SingleFactorBacktestRun` records
- cross-sectional matrix rank correlation when `FactorMatrix` artifacts are
  supplied for both candidate and reference factors
- IC-series correlation when daily record metadata contains `ic` or `rank_ic`
- simple residual mean return after neutralizing candidate after-cost returns
  against one reference factor return series

Correlation pair verdicts are:
- `distinct`
- `related`
- `redundant`
- `insufficient_data`

`build_factor_redundancy_report` aggregates pair-level results, records maximum
absolute return, matrix-rank, and IC correlations, and lists related or redundant
reference factor IDs for later research review. A redundant verdict is only a
pre-admission warning; it does not disable or approve anything by itself.

The contribution analysis builds local factor-return pools from existing
single-factor backtest runs. Baseline pool returns are equal-weighted by default
or use caller-supplied run weights. Missing source dates are handled by
renormalizing available source weights for that date. The candidate factor is
then combined with the baseline pool using configured baseline and candidate
weights.

Contribution reports include:
- incremental annualized return
- incremental Sharpe
- incremental maximum drawdown, where positive means drawdown got worse
- incremental turnover
- verdicts of `improves`, `neutral`, `degrades`, or `insufficient_data`

Current Pass 5 limitations:
- this is factor-return contribution analysis, not live portfolio construction
- there is no index-enhancement optimizer yet
- there is no production approval or admission replacement
- no live provider calls, Tushare/yfinance calls, LLM calls, broker calls, or web
  calls are made
- no stock selection, `PortfolioConstructor`, or `RiskGuard` wiring is added

## Production Factor Library And Audit

Phase 9 Pass 6 closes the offline governance loop with a production library
builder, audit report, guardrail helper, and dashboard payload.

Production library construction requires an explicit
`approve_production` `FactorAdmissionDecision` with `target_status=production`.
A passing validation report alone is not sufficient, and the helper does not
auto-approve production factors. By default, each production entry must have a
matching factor definition, a matching validation report with `pass` or `warn`,
and current validation evidence. Failed validation reports are excluded from the
library.

Validation currency is checked from `expires_at` when present. Otherwise the
audit uses `last_revalidation_at`, then `production_since`, plus the policy
`production_revalidation_days`. Missing dates are treated as expired.

Redundancy and contribution artifacts are pre-admission evidence. They can warn
on redundant factors, weak contribution, or missing incremental review, but they
do not approve or reject production by themselves. Non-production factors remain
blocked from formal stock selection and portfolio construction. The guardrail
helper can report `allowed`, `blocked`, or `shadow_only` for future integration,
but it is not wired into stock selection, `PortfolioConstructor`, or `RiskGuard`
by default.

Audit outputs are local artifacts:
- append-only JSONL audit reports
- a stable markdown audit report
- a JSON dashboard payload
- a JSON-serializable context patch for future runtime shadow reads

Current Pass 6 limitations:
- no runtime stock selection integration
- no `PortfolioConstructor` integration
- no `RiskGuard` integration
- no automatic production approval
- no live provider, Tushare/yfinance, LLM, broker, or web calls

## Shadow Scoring Comparison

Phase 11 Pass 1 adds a read-only comparison layer between local production
factor-library signals and already-produced official candidate/ranking outputs.
It is observability only: it does not alter official scoring, candidate
selection, posterior scores, `RiskGuard`, `PortfolioConstructor`, target
weights, orders, providers, LLMs, broker/execution, or web/frontend behavior.

The comparison reads the local production factor library when supplied or
available under `data/factor_library/production_factors.json`. It can also read
local factor matrices from `data/factor_library/matrix/factor_matrices.jsonl`
when a caller supplies those artifacts. Missing libraries, matrices, symbols,
dates, or values produce warnings in the report instead of runtime failures.

For each production factor, the scorer extracts the latest matrix value with a
matrix date less than or equal to `as_of`. Raw values are cross-sectionally
rank-normalized across the supplied official candidate symbols. The factor's
`expected_direction` comes from the matching `FactorDefinition` when available,
then matrix metadata, then defaults to positive direction. Higher adjusted
values receive better normalized scores.

The first shadow score is equal-weighted across covered production factors.
For each candidate, the report records:
- official score and rank, deriving rank from official score when needed
- shadow factor score and shadow factor rank
- `rank_delta = official_rank - shadow_factor_rank`, where a positive value
  means shadow factors rank the candidate higher than the official output
- raw `score_delta = shadow_factor_score - official_score` when both exist
- factor coverage ratio and warning codes

Report diagnostics include official Top-N symbols, shadow Top-N symbols, their
intersection, overlap ratio, largest positive and negative rank deltas, compact
candidate tables, JSON dashboard payloads, and append-only JSONL score/report
ledgers.

Current limitations:
- shadow scores are not official scores
- no stock-selection effect
- no portfolio-construction effect
- no factor weighting optimization beyond equal weighting
- local production library and factor matrices are required for meaningful
  coverage
- audit-blocked factors remain excluded unless explicitly requested by the
  caller for read-only diagnostics

## Artifact Locations

The default local store root is `data/factor_library`.

- `factor_definitions.jsonl`
- `factor_backtest_results.jsonl`
- `factor_validation_reports.jsonl`
- `factor_admission_decisions.jsonl`
- `production_factors.json`
- `deprecated_factors.json`

The JSONL ledgers are append-only and reject duplicate IDs on append.

The matrix store root is `data/factor_library/matrix`.

- `matrix_contracts.jsonl`
- `matrix_bundles.jsonl`
- `factor_matrices.jsonl`
- `expression_results.jsonl`

These JSONL ledgers are fixture/research oriented in Pass 2 and are not a
production parquet store.

The single-factor backtest store root is `data/factor_library/backtest`.

- `factor_weight_matrices.jsonl`
- `factor_backtest_runs.jsonl`
- `factor_daily_records.jsonl`

These ledgers are append-only research artifacts. They do not approve factors
for production and are not consumed by formal selection or portfolio construction.

The enhanced validation store root is `data/factor_library/validation`.

- `factor_robustness_reports.jsonl`
- `factor_cost_capacity_reports.jsonl`
- `enhanced_validation_reports.jsonl`

These ledgers are append-only research artifacts for later admission review.
They do not wire factors into live scoring, formal selection, or portfolio
construction.

The incremental correlation/contribution store root is
`data/factor_library/incremental`.

- `factor_redundancy_reports.jsonl`
- `factor_contribution_reports.jsonl`

These ledgers are append-only pre-admission research artifacts. They do not wire
factors into stock scoring, formal selection, portfolio construction,
`PortfolioConstructor`, or `RiskGuard`.

The production library audit store root is `data/factor_library/audit`.

- `factor_library_audit_reports.jsonl`
- `factor_library_audit_report.md`
- `factor_governance_dashboard.json`

These artifacts summarize production-library readiness and known blockers. They
do not alter runtime behavior.

The shadow scoring comparison store root is
`data/factor_library/shadow_scoring`.

- `shadow_factor_scores.jsonl`
- `shadow_candidate_scores.jsonl`
- `shadow_comparison_reports.jsonl`
- `shadow_comparison_report.md`
- `shadow_scoring_dashboard.json`

These artifacts are append-only or report/dashboard outputs for read-only
official-versus-shadow comparison. They do not alter runtime behavior.

## Future Roadmap

1. Index-enhancement validation on top of the offline contribution layer.
2. Manual admission review on top of enhanced validation and incremental
   research artifacts.
3. Shadow-read production library outputs in reports after explicit admission.
4. Compare factor-driven selection against current selection before any runtime
   selection integration.
