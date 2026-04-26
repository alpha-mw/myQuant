# System Upgrade Plan

## Objective

Create a staged path for improving the research, risk, and portfolio stack while keeping
the current deterministic control chain stable. Each phase should add executable
contracts before changing model behavior.

## Phase 1: Engineering Contracts and Executable Specs

Deliverables:
- Single-source canonical branch weights for quant, kline, intelligence, fundamental,
  and macro research branches.
- Offline contract tests for branch weights, README alignment, protocol dataclasses,
  and the CalibrationStore V1 outcome ledger anchor.
- Local quality gate script for Phase 1 tests, unit tests, and lightweight type checks.

Acceptance checks:
- Branch weights validate without normalization and sum to 1.0.
- README-displayed branch percentages match the code config.
- Core protocol dataclasses default-construct and expose stable `to_dict()` payloads.
- CalibrationStore records one outcome row with the expected JSONL fields.

Phase 1 non-goals:
- Do not change Bayesian posterior math or correlation discounts.
- Do not change RiskGuard veto logic or PortfolioConstructor sizing logic.
- Do not change market data download behavior, LLM behavior, or web behavior.
- Do not add dependencies or convert protocol dataclasses to Pydantic.

## Phase 2: Bayesian Outcome Ledger

Objective:
Create an append-only local ledger for Bayesian predictions and realized outcomes so
future calibration can be trained from replayable evidence rather than ad hoc logs.

Deliverables:
- Versioned `PredictionRecord` and `OutcomeRecord` dataclasses for posterior snapshots
  and realized outcomes.
- Deterministic run, prediction, and outcome identifiers with explicit horizon labels.
- JSONL-only `OutcomeLedgerStore` with duplicate protection, unresolved prediction
  lookup, and outcome resolution helpers.
- Pure bridge helpers that convert current `PosteriorResult`, `GlobalContext`, and
  branch result artifacts into prediction records without mutating runtime inputs.
- Focused offline tests and a Phase 2 quality gate script.

Acceptance checks:
- Deterministic identifiers are stable across repeated calls.
- Prediction records preserve prior, likelihood, branch score, branch confidence, and
  posterior fields.
- The JSONL store round-trips predictions and outcomes and rejects duplicates.
- Resolved outcomes compute excess return when benchmark return is supplied.
- Missing canonical branches are represented with `0.0`; non-canonical branches are
  excluded.
- Legacy `CalibrationStore.record_outcome` still writes `bayesian_outcomes.jsonl`.

Phase 2 non-goals:
- No active calibration training yet.
- No Bayesian posterior math or likelihood mapping changes.
- No RiskGuard, PortfolioConstructor, or optimizer changes.
- No point-in-time market data governance changes yet.
- No market download, LLM, frontend, or web behavior changes.

## Phase 3: Active Calibration V2

Implemented offline empirical calibration V2 on top of the Phase 2 outcome ledger.
Training reads `PredictionRecord` and `OutcomeRecord` pairs, emits posterior and
canonical branch training examples, and fits beta-binomial bucket calibration curves
with hierarchical fallback by target, market, horizon, and macro regime.

Deliverables:
- Versioned Calibration V2 schema, training examples, bucket curves, model, metrics,
  and report dataclasses.
- Conservative score normalization for mixed `[0, 1]` and `[-1, 1]` score conventions.
- JSON model/report store under the local calibration artifact directory.
- Offline CLI script for training from a ledger directory.
- Focused unit tests and local quality gate script.

Acceptance checks:
- Training examples are built deterministically from resolved ledger outcomes.
- Empty buckets are retained for stable curve shape.
- Global fallback curves are trained when any examples exist for a target.
- `calibrate()` falls back from exact context to global target curves, then raw
  normalized values when no curve exists.
- Reports include raw and calibrated Brier/log-loss metrics.

Phase 3 non-goals:
- No live posterior math changes yet.
- No RiskGuard or PortfolioConstructor changes.
- No optimizer or portfolio allocation changes.
- No point-in-time data governance changes.
- No market download, LLM, frontend, or web behavior changes.

## Phase 4: Calibrated Posterior Overlay and Edge Diagnostics

Implemented an opt-in calibrated posterior overlay for A/B diagnostics against the
original posterior output. The overlay reads a Calibration V2 model, computes a
calibrated win probability, converts it into calibrated expected alpha, and produces
edge-after-cost diagnostics without replacing live posterior fields.

Deliverables:
- Versioned posterior overlay schema and JSON-serializable diagnostics dataclasses.
- Curve-selection diagnostics that expose which Calibration V2 fallback curve was used.
- Cost model inputs for transaction cost, slippage, market impact, capacity penalty, and
  risk capital charge.
- Batch overlay and metadata-attachment helpers for shadow comparison.
- Focused unit tests and local quality gate script.

Acceptance checks:
- Default posterior math, ranking, RiskGuard, and PortfolioConstructor behavior remain
  unchanged.
- Overlay probabilities are blended and capped before expected-alpha conversion.
- Calibrated edge-after-cost reports preserve the original posterior values for audit.
- Overlay metadata can be attached without changing core PosteriorResult numeric fields.

Phase 4 non-goals:
- No replacement of live posterior math by default.
- No RiskGuard or PortfolioConstructor changes.
- No optimizer or portfolio allocation changes.
- No point-in-time data governance changes.
- No production trading decision changes.

## Phase 5: Point-in-Time Data Governance

Implemented an offline point-in-time data quality contract layer. The layer can
describe field provenance, claimed point-in-time availability, look-ahead risk,
missing/stale/outlier fields, tradability status, and quarantine decisions without
touching provider behavior or mainline runtime decisions.

Deliverables:
- Versioned data quality contract schema and JSON-serializable dataclasses.
- Deterministic IDs for snapshots, issues, and assessments.
- Provenance checks for missing source evidence, effective-date look-ahead, and
  observed-at look-ahead.
- Freshness, missing-field, outlier, tradability, and quarantine assessment helpers.
- GlobalContext quality patch helper for future integration.
- Append-only JSONL store for point-in-time snapshots and assessments.
- Focused unit tests and local quality gate script.

Acceptance checks:
- Clean snapshots produce high data quality scores and no quarantine.
- Blocker issues quarantine a symbol and mark it non-researchable.
- Untradable symbols are represented in assessment and GlobalContext patch output.
- Snapshot and assessment ledgers reject duplicate IDs and malformed JSON.

Phase 5 non-goals:
- No market data provider changes.
- No data download behavior changes.
- No live mainline integration.
- No posterior, calibration, or overlay math changes.
- No RiskGuard or PortfolioConstructor changes.
- No optimizer or portfolio allocation changes.
- No frontend or web changes.

## Phase 6: Structured Risk Tensor and Execution Feasibility

Implemented an additive, offline structured risk tensor layer that represents symbol
and portfolio risk across data quality, tradability, liquidity, capacity, exposure,
concentration, execution feasibility, and stress dimensions. The layer produces
machine-readable bridge objects for future RiskGuard and PortfolioConstructor work
without changing current runtime decisions.

Deliverables:
- Versioned symbol and portfolio risk tensor schemas with JSON-serializable
  dataclasses.
- Liquidity, capacity, and execution feasibility diagnostics for requested target
  weights and trade values.
- Stress scenario result objects with positive-loss convention for adverse shocks.
- Future RiskGuard context patch helper that exposes blocked symbols, per-symbol
  max weights, risk issues, and execution statuses without mutating RiskGuard state.
- Append-only JSONL store for symbol tensors, portfolio tensors, and execution
  feasibility reports.
- Focused offline unit tests and local quality gate script.

Acceptance checks:
- Tensor identifiers are deterministic and contain no current timestamp.
- Dataclasses round-trip through `to_dict()` / `from_dict()` with nested objects.
- Phase 5 data quality and tradability objects can feed symbol tensors via duck
  typing.
- Feasibility reports group feasible, partially feasible, and blocked symbols in a
  deterministic order.
- JSONL ledgers reject duplicate IDs and malformed JSON.

Phase 6 non-goals:
- No RiskGuard runtime integration.
- No PortfolioConstructor sizing or allocation changes.
- No live execution routing.
- No broker integration.
- No market data provider or download changes.
- No posterior or calibration math changes.
- No LLM behavior changes.
- No frontend or web changes.

## Phase 7: Portfolio Optimizer and Walk-Forward Loop

Implemented an offline deterministic greedy portfolio optimizer and walk-forward
evaluation layer in shadow mode. The layer consumes calibrated posterior overlay and
risk tensor artifacts through bridge helpers, then emits auditable target weights,
constraint diagnostics, execution-aware turnover/cost estimates, rebalance results,
and walk-forward summaries from caller-supplied forward returns.

Deliverables:
- Versioned optimizer schema and JSON-serializable optimizer, rebalance, and
  walk-forward dataclasses.
- Deterministic bridge helpers from calibrated posterior overlays and symbol risk
  tensors into optimization candidates.
- Long-only deterministic greedy allocation with max weight, gross exposure, sector
  cap, turnover cap, min edge, max risk score, blocked-symbol, and max-name
  diagnostics.
- Rebalance evaluation and walk-forward loop using supplied forward returns only.
- Future PortfolioConstructor patch helper that does not mutate or wire into the
  current constructor.
- Append-only JSONL store for optimized plans, rebalance results, and walk-forward
  results.
- Focused offline unit tests and a Phase 7 quality gate script.

Acceptance checks:
- Optimizer identifiers are deterministic and contain no current timestamp.
- Dataclasses round-trip through `to_dict()` / `from_dict()` with nested objects.
- Phase 4 overlays and Phase 6 risk tensors can feed candidates via duck typing.
- Plans include selected, rejected, and blocked symbols plus objective value,
  turnover, sector weights, exposure totals, and constraint violations.
- Rebalance and walk-forward results use only supplied forward returns and never fetch
  market data.
- JSONL ledgers reject duplicate IDs and malformed JSON.

Phase 7 non-goals:
- No replacement of existing PortfolioConstructor.
- No RiskGuard integration.
- No broker, order, or execution routing.
- No live market data fetching.
- No posterior, calibration V1/V2, or posterior overlay math changes.
- No market data provider, download, Tushare, or yfinance changes.
- No LLM behavior changes.
- No frontend or web changes.

## Phase 8: Observability, Audit Reporting, CI Hardening, Dependency Profiles

Implemented an offline observability and audit layer for the staged upgrade modules.
The layer discovers expected artifacts across the outcome ledger, Calibration V2,
data quality contract, risk tensor, and portfolio optimizer, then emits a
machine-readable audit bundle, a markdown audit report, a run manifest, and a
dashboard-ready JSON payload without touching live runtime behavior.

Deliverables:
- Versioned observability and audit-bundle schemas with JSON-serializable dataclasses.
- Artifact discovery across Phase 2-7 artifact stores plus upgrade docs and quality
  gate scripts.
- Module health summaries for outcome ledger, Calibration V2, data quality, risk
  tensor, portfolio optimizer, and docs/scripts.
- Markdown audit report and dashboard-ready JSON payload generation.
- Offline CLI script for audit bundle generation.
- Focused Phase 8 and staged-upgrade quality gate scripts.
- Optional focused CI workflow for the staged-upgrade modules.
- Productionization runbook and documentation-only dependency profile notes.

Acceptance checks:
- Artifact identifiers and bundle identifiers are deterministic when generation time
  is supplied.
- JSON and JSONL helpers reject malformed local files with clear errors.
- Missing optional runtime artifacts warn rather than fail.
- Malformed JSON/JSONL artifacts fail the relevant module summary.
- Dashboard payload and audit bundle are JSON serializable and omit raw file contents.
- Quality gates avoid the known unrelated `tests/unit/test_data_layer.py` failure by
  default.

Phase 8 non-goals:
- No live runtime integration.
- No RiskGuard changes.
- No PortfolioConstructor changes.
- No provider, fallback, or download changes.
- No posterior, CalibrationStore V1, Calibration V2, or posterior overlay math changes.
- No frontend or web integration.
- No broker or execution integration.
