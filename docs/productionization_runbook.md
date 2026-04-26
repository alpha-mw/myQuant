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

## Expected Artifact Directories

- `data/bayesian_outcome_ledger`
- `data/bayesian_calibration_v2`
- `data/data_quality_contract`
- `data/risk_tensor`
- `data/portfolio_optimizer`
- `data/observability`

## Offline Default

The staged modules should operate on supplied local artifacts only. Do not call market
data providers, broker APIs, LLM providers, web routes, or live execution paths from
the audit, calibration, tensor, or optimizer helpers.

## Future Integration Checklist

1. Wire outcome ledger to runtime behind an environment flag.
2. Train Calibration V2 from real resolved outcomes.
3. Shadow posterior overlay in production.
4. Feed data quality patch into `GlobalContext`.
5. Feed risk tensor patch into `RiskGuard`.
6. Compare offline optimizer against existing `PortfolioConstructor`.
7. Only then consider controlled runtime integration.
