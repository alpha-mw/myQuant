#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

echo "Running Phase 9 factor governance focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_governance_schema.py \
  tests/unit/test_factor_governance_admission.py \
  tests/unit/test_factor_governance_store.py \
  -q

echo "Running Phase 9 factor governance focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/factors/schema.py \
  quant_investor/factors/admission.py \
  quant_investor/factors/store.py \
  quant_investor/versioning.py \
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
