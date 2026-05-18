#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

echo "Running Phase 12 tradability and execution feasibility quality gate..."
PHASE12_RUN_STAGED_GATE=0 PYTHON="$PYTHON_BIN" bash scripts/phase12_factor_tradability_audit_quality_gate.sh

echo "Running Phase 12 execution cost focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_execution_cost.py \
  tests/unit/test_factor_execution_cost_store.py \
  tests/unit/test_factor_execution_cost_no_runtime_effect.py \
  -q

echo "Running Phase 12 execution cost focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/factors/execution_cost.py \
  quant_investor/factors/tradability.py \
  quant_investor/factors/backtest.py \
  quant_investor/factors/metrics.py \
  quant_investor/factors/store.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE12_RUN_STAGED_GATE:-1}" = "1" ]; then
  echo "Running staged upgrade quality gate..."
  PYTHON="$PYTHON_BIN" bash scripts/staged_upgrade_quality_gate.sh
fi

if [ "${PHASE12_RUN_FULL_UNIT:-0}" = "1" ]; then
  echo "Running full unit suite..."
  "$PYTHON_BIN" -m pytest tests/unit -q --maxfail=1
fi
