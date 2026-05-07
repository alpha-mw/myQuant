#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

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
  -q

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
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE9_RUN_STAGED_GATE:-1}" = "1" ]; then
  PYTHON="$PYTHON_BIN" scripts/staged_upgrade_quality_gate.sh
fi

if [ "${PHASE9_RUN_FULL_UNIT:-0}" = "1" ]; then
  "$PYTHON_BIN" -m pytest tests/unit -q --maxfail=1
fi
