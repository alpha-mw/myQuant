# Productionization Runbook

This staged upgrade path is offline by default. It adds contracts, ledgers, shadow
artifacts, and audit reports before any runtime decision path changes.

## Markov Regime Operations

Markov regime is production-first in the v14 DAG.

- Production default: `MARKOV_REGIME_ENABLED=1`.
- Emergency disable: `MARKOV_REGIME_ENABLED=0`.
- Deprecated `MARKOV_REGIME_EXECUTION_TARGET=shadow` is normalized to production with a diagnostic note; it is not a separate execution path.
- Markov applies only when market-scope data is production eligible: full-market current input or a broad local market reference (`full_a` for CN, `full_us` for US by default).
- Small explicit pools and watchlists never define the market regime. If broad reference data is unavailable or below `MARKOV_REGIME_MIN_MARKET_SAMPLE`, Markov records `production_eligible=false`, keeps MacroAgent regime and baseline risk caps, and does not forward turnover caps.
- Markov can only tighten exposure and position limits. RiskGuard hard vetoes and PortfolioConstructor deterministic constraints remain authoritative.
- Regime history is isolated by market scope/source universe and filtered by `as_of`; future records and legacy unscoped records are not used for production scoped history.

## Local Quality Gate Sequence

Recommended single command:

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

Focused Phase 8 command:

```bash
PYTHON=./.venv/bin/python scripts/phase8_quality_gate.sh
```

Focused Phase 9 factor governance command:

```bash
PYTHON=./.venv/bin/python scripts/phase9_factor_governance_quality_gate.sh
```

Focused Phase 9 matrix/expression command:

```bash
PYTHON=./.venv/bin/python scripts/phase9_factor_matrix_quality_gate.sh
```

Focused Phase 9 factor backtest command:

```bash
PYTHON=./.venv/bin/python scripts/phase9_factor_backtest_quality_gate.sh
```

Focused Phase 9 factor validation command:

```bash
PYTHON=./.venv/bin/python scripts/phase9_factor_validation_quality_gate.sh
```

Focused Phase 9 factor incremental correlation/contribution command:

```bash
PYTHON=./.venv/bin/python scripts/phase9_factor_incremental_quality_gate.sh
```

Focused Phase 9 production factor library/audit command:

```bash
PYTHON=./.venv/bin/python scripts/phase9_factor_library_quality_gate.sh
```

Focused Phase 11 production factor shadow scoring comparison command:

```bash
PYTHON=./.venv/bin/python scripts/phase11_factor_shadow_scoring_quality_gate.sh
```

Focused Phase 12 factor backtest alignment audit command:

```bash
PYTHON=./.venv/bin/python scripts/phase12_factor_alignment_audit_quality_gate.sh
```

The full unit suite is intentionally opt-in because the local repository has a known
unrelated failure in `tests/unit/test_data_layer.py`.

```bash
PHASE8_RUN_FULL_UNIT=1 PYTHON=./.venv/bin/python scripts/phase8_quality_gate.sh
STAGED_UPGRADE_RUN_FULL_UNIT=1 PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

## Calibration V2 Artifacts

Generate Calibration V2 artifacts from an offline outcome ledger:

```bash
./.venv/bin/python scripts/train_calibration_v2.py \
  --ledger-dir data/bayesian_outcome_ledger/v14 \
  --output-dir data/bayesian_calibration_v2/v14
```

Expected outputs:
- `data/bayesian_calibration_v2/v14/calibration_model_v2.json`
- `data/bayesian_calibration_v2/v14/calibration_report_v2.json`

## Audit Bundle Generation

Build an offline audit bundle:

```bash
./.venv/bin/python scripts/build_audit_bundle.py \
  --run-id staged-upgrade-local \
  --output-dir data/observability
```

Expected outputs:
- `data/observability/audit_bundle.json`
- `data/observability/audit_report.md`
- `data/observability/dashboard_payload.json`
- `data/observability/run_manifest.json`

## Factor Library Audit Generation

Build the offline production factor library audit:

```bash
./.venv/bin/python scripts/build_factor_library_audit.py \
  --root-dir data/factor_library \
  --output-dir data/factor_library/audit \
  --as-of 2026-04-27 \
  --generated-at 2026-04-27T00:00:00
```

Expected outputs:
- `data/factor_library/production_factors.json` when explicit production
  approvals produce at least one library entry
- `data/factor_library/audit/factor_library_audit_reports.jsonl`
- `data/factor_library/audit/factor_library_audit_report.md`
- `data/factor_library/audit/factor_governance_dashboard.json`

The audit builder reads only local factor governance, validation, redundancy,
and contribution artifacts. It does not fetch data, call providers, call LLMs,
or modify strategy outputs.

## Factor Shadow Scoring Comparison

Run the Phase 11 quality gate:

```bash
PYTHON=./.venv/bin/python scripts/phase11_factor_shadow_scoring_quality_gate.sh
```

Expected artifact directory:

```text
data/factor_library/shadow_scoring
```

Expected files when a caller saves reports through `FactorShadowScoringStore`:
- `shadow_factor_scores.jsonl`
- `shadow_candidate_scores.jsonl`
- `shadow_comparison_reports.jsonl`
- `shadow_comparison_report.md`
- `shadow_scoring_dashboard.json`

The comparison reads already-local production factor libraries and factor
matrices, computes read-only shadow factor scores, and compares those ranks
against existing official candidate outputs. It does not alter official
decisions, stock selection, posterior, `RiskGuard`, `PortfolioConstructor`,
target weights, orders, providers, LLMs, or execution.

## Factor Backtest Alignment Audit

Run the Phase 12 quality gate:

```bash
PYTHON=./.venv/bin/python scripts/phase12_factor_alignment_audit_quality_gate.sh
```

Expected artifact directory:

```text
data/factor_library/alignment_audit
```

Expected files when a caller saves reports through `FactorAlignmentAuditStore`:
- `alignment_audit_reports.jsonl`
- `alignment_audit_report.md`

The alignment audit checks local `FactorMatrix`, `MatrixDataBundle`, and
`SingleFactorBacktestRun` artifacts for signal/execution/return-window
alignment. It validates T+1 delay semantics, execution price availability, VWAP
derivability, run daily-record dates, and execution-return matrix alignment.

An alignment audit must pass before a factor can be considered for future
production admission. The current pass is still diagnostics and test hardening
only: it does not approve production factors and does not wire any factor into
official scoring, stock selection, posterior math, `RiskGuard`,
`PortfolioConstructor`, target weights, orders, providers, LLMs, or execution.

## A-share Tradability and Execution Feasibility Audit

Run the Phase 12 Pass 2 quality gate:

```bash
PYTHON=./.venv/bin/python scripts/phase12_factor_tradability_audit_quality_gate.sh
```

Expected artifact directory:

```text
data/factor_library/tradability_audit
```

Expected files when a caller saves reports through
`FactorTradabilityAuditStore`:
- `tradability_masks.jsonl`
- `tradability_audit_reports.jsonl`
- `execution_feasibility_reports.jsonl`
- `tradability_audit_report.md`
- `execution_feasibility_report.md`

The tradability audit should pass, or be explicitly reviewed, before a factor
is considered for future production admission. It checks local A-share
execution constraints such as suspension, limit-up buy blockage, limit-down
sell blockage, ST / risk-warning status, delisting, new listings, invalid
price/volume, and low amount/liquidity proxies. The execution feasibility
report audits buy/sell/hold weight transitions on the execution date.

This pass is still audit-only. It does not adjust factor backtest PnL, model
partial fills, call brokers, call live data providers, or wire factors into
selection, posterior scoring, `RiskGuard`, `PortfolioConstructor`, target
weights, orders, providers, LLMs, or execution.

## Offline Execution Cost and Penalty Simulation

Run the Phase 12 Pass 3 quality gate:

```bash
PYTHON=./.venv/bin/python scripts/phase12_factor_execution_cost_quality_gate.sh
```

Expected artifact directory:

```text
data/factor_library/execution_cost
```

Expected files when a caller saves reports through
`FactorExecutionCostSimulationStore`:
- `execution_cost_reports.jsonl`
- `execution_adjusted_runs.jsonl`
- `execution_adjusted_daily_records.jsonl`
- `execution_cost_report.md`
- `execution_cost_dashboard.json`

The execution cost simulation should be reviewed before a factor is considered
for future production admission. It applies explicit transaction costs,
sell-side stamp tax by default, exchange fees, commission, slippage, spread
cost, simple participation-based impact, blocked-transition penalties, missing
data diagnostics, and long-short research short-leg caveats.

This pass does not replace original factor backtest PnL and does not wire the
factor library into official scoring, stock selection, posterior scoring,
`RiskGuard`, `PortfolioConstructor`, target weights, orders, providers, LLMs,
brokers, or execution.

`portfolio_backtest.TransactionCostModel` has an optional participation-impact
adapter behind `EXECUTION_COST_MODEL_ENABLED=0` by default, or an explicit
constructor flag for offline backtest experiments. When disabled, the legacy
flat-bps backtest cost path is preserved exactly. When enabled, the adapter can
consume `FactorExecutionCostConfig` plus local symbol amount/record inputs to
charge participation-based impact inside backtest accounting only.

## Expected Artifact Directories

- `data/bayesian_outcome_ledger/v14`
- `data/bayesian_calibration_v2/v14`
- `data/data_quality_contract`
- `data/risk_tensor`
- `data/portfolio_optimizer`
- `data/observability`
- `data/factor_library`
- `data/factor_library/matrix`
- `data/factor_library/backtest`
- `data/factor_library/validation`
- `data/factor_library/execution_cost`
- `data/factor_library/incremental`
- `data/factor_library/audit`
- `data/factor_library/shadow_scoring`
- `data/factor_library/alignment_audit`
- `data/factor_library/tradability_audit`

## Offline Default

The staged modules should operate on supplied local artifacts only. Do not call market
data providers, broker APIs, LLM providers, web routes, or live execution paths from
the audit, calibration, tensor, or optimizer helpers.

## Factor Governance Offline Boundary

Factor governance remains offline by default. Factor definitions, validation reports,
admission decisions, and production libraries are local JSON/JSONL artifacts under
`data/factor_library`.

Production stock selection must not read non-production factors. Phase 9 Pass 1 does
not wire factors into live selection, `PortfolioConstructor`, `RiskGuard`, posterior
math, provider download behavior, or portfolio construction.

Phase 9 Pass 2 remains offline as well. Matrix contracts, in-memory fixture bundles,
factor matrices, and expression results live under `data/factor_library/matrix`.
Factor expressions are research artifacts until they pass through later backtest and
admission gates. Non-production factors remain blocked from formal selection,
portfolio construction, `PortfolioConstructor`, and `RiskGuard`.

Phase 9 Pass 3 remains offline. Single-factor weight matrices, daily backtest
records, and run envelopes live under `data/factor_library/backtest`. Backtest
results are research artifacts until admitted by an explicit
`FactorAdmissionDecision`; non-production factors remain blocked from formal stock
selection, portfolio construction, `PortfolioConstructor`, and `RiskGuard`.

Phase 9 Pass 4 remains offline. Robustness reports, cost/capacity reports, and
enhanced validation reports live under `data/factor_library/validation`.
Enhanced validation reports are still research artifacts until reviewed by the
admission decision layer. Non-production factors remain blocked from formal
selection, portfolio construction, `PortfolioConstructor`, and `RiskGuard`.

Phase 9 Pass 5 remains offline. Redundancy reports and contribution reports live
under `data/factor_library/incremental`. They are pre-admission research
artifacts for comparing candidate factor returns, matrices, IC series, residual
returns, and baseline-pool contribution. Production stock selection still cannot
read non-production factors, and this pass does not approve production factors or
wire candidates into formal selection, `PortfolioConstructor`, or `RiskGuard`.

Phase 9 Pass 6 remains offline. Production factor libraries are built only from
explicit `approve_production` decisions and current validation artifacts. Audit
reports live under `data/factor_library/audit` and flag missing, expired,
rejected, disabled, duplicated, redundant, or weak-contribution factors. The
guardrail helper returns JSON-serializable `allowed`, `blocked`, or
`shadow_only` decisions for future use, but it is not wired into stock
selection, `PortfolioConstructor`, or `RiskGuard`.

Phase 11 Pass 1 remains offline. Shadow scoring comparison reads local
production libraries, local factor matrices, and already-computed candidate
payloads supplied by the caller. It produces diagnostics under
`data/factor_library/shadow_scoring` only when explicitly saved. It is not
wired into official scoring, stock selection, posterior math,
`PortfolioConstructor`, `RiskGuard`, target weights, orders, market downloads,
providers, LLMs, broker/execution, or frontend/web behavior.

Phase 12 Pass 1 remains offline. Factor backtest alignment audits read local
factor matrices, matrix bundles, optional single-factor backtest runs, and
optional execution-return matrices supplied by the caller. They produce
diagnostics under `data/factor_library/alignment_audit` only when explicitly
saved. Passing this audit is a prerequisite for future production admission
consideration, but this pass does not approve factors and does not wire factor
signals into official scoring, stock selection, posterior math,
`PortfolioConstructor`, `RiskGuard`, target weights, orders, market downloads,
providers, LLMs, broker/execution, or frontend/web behavior.

Phase 12 Pass 2 remains offline. A-share tradability masks and execution
feasibility audits read local matrix fields, factor weight matrices, optional
alignment tuples, and optional single-factor backtest runs supplied by the
caller. They produce diagnostics under
`data/factor_library/tradability_audit` only when explicitly saved. Passing or
reviewing this audit is a prerequisite for future production admission
consideration, but this pass does not adjust PnL, model fills, approve factors,
or wire factor signals into official scoring, stock selection, posterior math,
`PortfolioConstructor`, `RiskGuard`, target weights, orders, market downloads,
providers, LLMs, broker/execution, or frontend/web behavior.

Phase 13 Pass 1 remains offline. Multi-date shadow evidence collection reads
local candidate snapshots, local production factor libraries, local factor
matrices, and optional local audit reports supplied in an input manifest. It
writes evidence artifacts under `data/factor_library/evidence` and must pass
before any future paper-portfolio comparison. This pass does not wire factor
signals into selection or portfolio construction.

## Tushare Daily Download Cleaning Gate

CN Tushare daily downloads now run an offline post-download cleaning hook before
Parquet canonical writes when `MYQUANT_TUSHARE_AUTO_CLEAN=1`. The hook preserves
the raw merged frame under `data/raw_backups/tushare`, writes row/cell flags,
quarantines invalid rows, de-duplicates and sorts the cleaned frame, and emits
factor-readiness plus storage-audit sidecars. A cleaning pass does not mean the
file is factor-ready; missing trade calendar, adjusted factor, limit, suspend,
tradability, or benchmark-membership evidence is reported separately.

Parquet is the canonical runtime store. CSV deletion flags only apply to legacy
human export cleanup; production market-data reads must use the Parquet store or
JSON manifests.

### CN Canonical Coverage Provenance Gate

For every CN target date, full-A coverage is the pairwise-disjoint union of
symbols with an observed bar, exact-date verified suspended symbols without a
bar, exact-date `verified_nontrading_bak_daily_zero` symbols, date-proven
inactive/pre-listing symbols without a bar, and true missing symbols. The union
must equal the component scope after PIT membership classification. A missing
bar is therefore not itself a failure; only a PIT-active symbol without an
observed bar or valid non-trading evidence is a blocking `true_missing_symbol`.

Suspension evidence must come from a successful exact-date `suspend_d` query and
the v5 cache contract, including the `trade_date` query parameters, shared query
run ID, raw-response SHA, every exact-date S/R/other event record and its SHA,
timestamp, and reproducible payload SHA. An empty exact-date event response is
not proof that a symbol traded and cannot erase stronger non-trading evidence.
The separate `verified_suspension_continuity_absent` classification is allowed
only when the immediately previous and next open-day v5 records both contain an
S event for the same symbol, the target v5 record contains no event of any type,
the exact target `bak_daily` result rejects that symbol only as
`exact_row_missing`, PIT says listed/tradable on all three dates, and canonical
bars are absent on all three dates. Its persisted evidence binds the ordered
calendar, PIT, canonical table/serving window, all source files, and the shared
query run ID. It is an explicit continuity inference, not a claim that the
target day had a direct regulatory event. If a qualifying triplet has mixed
query run IDs, refresh exactly those three dates under one explicit shared ID
and repeat full readback before classifying it. `verified_nontrading_bak_daily_zero`
requires one exact symbol/date row for a PIT-active primary-bar absence with
zero open/high/low/volume/amount, positive `close == pre_close`, and zero change
fields when present. These rows are typed evidence only: they are never written
as synthetic OHLCV bars or described as a regulatory suspension. Inactive or
pre-listing evidence requires `list_date > target_date` or a non-empty
`delist_date <= target_date`; equality on the delisting date is excluded from
the active scope. Current status alone is not sufficient historical evidence. A generic
`--allowed-stale-symbols` list is diagnostic only and produces
`unverified_allowed_stale_symbols_not_permitted`; it cannot clear a coverage
blocker.

Historical upserts require verified latest-date coverage and must preserve the
latest coverage date, scope SHA, classification sets, and canonical fingerprint.
Historical-target evidence is stored separately under
`historical_upsert_target_coverage`. The narrow exception is an exact same-date
republish that repairs a provenance-only quarantine. After any canonical write,
validation must parse the JSON result and require `status=passed`, empty
`blockers`, and empty `coverage_provenance_blockers`; process exit code alone is
not sufficient.

The 100-open-day audit must be a full recomputation from canonical Parquet and
current hash-bound evidence. It must report `prior_trade_dates_reused=0`; a
delta report that reuses prior date classifications cannot close the gate.
Refresh the PIT stock-basic snapshot before publishing the current canonical
date so target-day delist boundaries are evaluated from current source facts.
The refresh must have a nonempty listed frame, cover every current full-A
component, and retain every symbol already present in canonical PIT; otherwise
it fails before replacing the PIT files.

```bash
PIT_UNIVERSE_BACKFILL_ENABLED=1 \
uv run python scripts/refresh_pit_universe.py \
  --market CN \
  --allow-online \
  --execute \
  --output data/cn_market_full/pit_universe_refresh_YYYYMMDD.json
```

```bash
UV_CACHE_DIR=/tmp/uv-cache \
MYQUANT_MARKET_DATA_BACKEND=parquet \
MYQUANT_MARKET_DATA_MODE_POLICY=strict \
PATH="$PWD/.venv/bin:$PATH" \
uv run python scripts/run_cn_history_audit.py \
  --market CN \
  --days 100 \
  --end-date auto \
  --allow-online \
  --output-root data/cn_market_full
```

If the full audit identifies genuine primary-bar gaps on at most five target
dates, repair only those exact dates with the bounded atomic helper, then rerun
the complete 100-open-day audit. The helper never advances or weakens the latest
coverage contract and never synthesizes bars. It rejects an audit whose payload
SHA, schema, full-window status, pointer, manifest, PIT, or table/serving window
binding is stale, and refuses to overwrite any target symbol-date that is
already present in current canonical storage.

```bash
uv run python scripts/repair_cn_history_gaps.py \
  --audit-path data/cn_market_full/history_audit_<timestamp>_full_recompute.json \
  --trade-dates YYYYMMDD [YYYYMMDD ...] \
  --allow-online \
  --output-root data/cn_market_full
```

Focused gate:

```bash
PYTHON=./.venv/bin/python scripts/tushare_data_cleaning_quality_gate.sh
```

Manual offline Parquet cleanup:

```bash
./.venv/bin/python scripts/clean_tushare_downloads.py \
  --root-dir data/cn_market_full \
  --table daily
```

Run Phase 13 evidence collection with:

```bash
PYTHON=./.venv/bin/python scripts/collect_factor_shadow_evidence.py \
  --input-manifest data/factor_library/evidence/sample_manifest.json \
  --output-dir data/factor_library/evidence \
  --generated-at 2026-04-27T00:00:00Z \
  --top-n 30 \
  --min-observation-days 20 \
  --min-average-factor-coverage 0.80 \
  --min-top-n-overlap-ratio 0.50
```

Input manifest example:

```json
{
  "as_of_dates": ["2026-04-01", "2026-04-02"],
  "date_inputs": [
    {
      "as_of": "2026-04-01",
      "candidates": [
        {"symbol": "000001.SZ", "official_score": 0.83, "official_rank": 1}
      ],
      "production_library_path": "data/factor_library/production_factors.json",
      "factor_matrix_paths": ["data/factor_library/matrix/factor_matrices.jsonl"],
      "library_audit_path": "data/factor_library/audit/factor_library_audit_report.json",
      "alignment_audit_paths": ["data/factor_library/alignment_audit/factor_alignment_audit_reports.jsonl"],
      "tradability_audit_paths": ["data/factor_library/tradability_audit/tradability_audit_reports.jsonl"],
      "execution_cost_report_paths": ["data/factor_library/execution_cost/execution_cost_reports.jsonl"]
    }
  ]
}
```

Artifacts:

- `data/factor_library/evidence/evidence_date_results.jsonl`
- `data/factor_library/evidence/multi_date_evidence_reports.jsonl`
- `data/factor_library/evidence/evidence_report.md`
- `data/factor_library/evidence/evidence_dashboard.json`

## Future Integration Checklist

1. Generate factor matrices from real point-in-time data.
2. Run single-factor backtests.
3. Run robustness, cost, and capacity validation.
4. Run redundancy and contribution reports.
5. Manually approve production factors.
6. Shadow-read the production library in reports.
7. Only then wire the production-only factor list into stock selection.
8. Compare factor-driven selection against current selection.
9. Only then consider `PortfolioConstructor` integration.
