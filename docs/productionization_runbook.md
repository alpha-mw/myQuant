# Productionization Runbook

This staged upgrade path is offline by default. It adds contracts, ledgers, shadow
artifacts, and audit reports before any runtime decision path changes.

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
  --ledger-dir data/bayesian_outcome_ledger \
  --output-dir data/bayesian_calibration_v2
```

Expected outputs:
- `data/bayesian_calibration_v2/calibration_model_v2.json`
- `data/bayesian_calibration_v2/calibration_report_v2.json`

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

## Expected Artifact Directories

- `data/bayesian_outcome_ledger`
- `data/bayesian_calibration_v2`
- `data/data_quality_contract`
- `data/risk_tensor`
- `data/portfolio_optimizer`
- `data/observability`
- `data/factor_library`
- `data/factor_library/matrix`
- `data/factor_library/backtest`
- `data/factor_library/validation`
- `data/factor_library/incremental`
- `data/factor_library/audit`

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
