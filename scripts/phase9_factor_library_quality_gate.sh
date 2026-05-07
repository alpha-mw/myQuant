#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

echo "Running Phase 9 factor library focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_governance_schema.py \
  tests/unit/test_factor_governance_admission.py \
  tests/unit/test_factor_governance_store.py \
  tests/unit/test_factor_matrix_contract.py \
  tests/unit/test_factor_operators.py \
  tests/unit/test_factor_expression_sandbox.py \
  tests/unit/test_factor_matrix_store.py \
  tests/unit/test_factor_backtest.py \
  tests/unit/test_factor_backtest_store.py \
  tests/unit/test_factor_metrics.py \
  tests/unit/test_factor_robustness.py \
  tests/unit/test_factor_capacity.py \
  tests/unit/test_factor_validation_artifact_store.py \
  tests/unit/test_factor_correlation.py \
  tests/unit/test_factor_contribution.py \
  tests/unit/test_factor_incremental_store.py \
  tests/unit/test_factor_library.py \
  tests/unit/test_factor_library_report.py \
  tests/unit/test_factor_library_store.py \
  tests/unit/test_observability_factor_governance.py \
  -q

echo "Running Phase 9 factor library focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/factors/schema.py \
  quant_investor/factors/admission.py \
  quant_investor/factors/store.py \
  quant_investor/factors/matrix.py \
  quant_investor/factors/operators.py \
  quant_investor/factors/expression.py \
  quant_investor/factors/backtest.py \
  quant_investor/factors/metrics.py \
  quant_investor/factors/robustness.py \
  quant_investor/factors/capacity.py \
  quant_investor/factors/correlation.py \
  quant_investor/factors/contribution.py \
  quant_investor/factors/library.py \
  quant_investor/factors/report.py \
  quant_investor/versioning.py \
  quant_investor/observability.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE9_RUN_STAGED_GATE:-1}" = "1" ]; then
  echo "Running staged upgrade quality gate..."
  PYTHON="$PYTHON_BIN" bash scripts/staged_upgrade_quality_gate.sh
fi

if [ "${PHASE9_RUN_FULL_UNIT:-0}" = "1" ]; then
  echo "Running full unit suite..."
  "$PYTHON_BIN" -m pytest tests/unit -q --maxfail=1
fi
