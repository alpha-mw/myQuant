#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

echo "Running Phase 11 factor shadow scoring quality gate..."
PYTHON="$PYTHON_BIN" bash scripts/phase11_factor_shadow_scoring_quality_gate.sh

echo "Running Phase 12 factor alignment audit focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_alignment_audit.py \
  tests/unit/test_factor_alignment_audit_store.py \
  -q

echo "Running Phase 12 factor alignment audit focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/factors/alignment_audit.py \
  quant_investor/factors/backtest.py \
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
