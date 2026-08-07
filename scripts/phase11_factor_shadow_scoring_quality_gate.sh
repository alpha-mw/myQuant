#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if [ "${PHASE11_RUN_PHASE10_GATE:-1}" = "1" ]; then
  echo "Running Phase 10 factor shadow report quality gate..."
  PYTHON="$PYTHON_BIN" bash scripts/phase10_factor_shadow_report_quality_gate.sh
fi

echo "Running Phase 11 factor shadow scoring focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_shadow_scoring.py \
  tests/unit/test_factor_shadow_scoring_store.py \
  -q

echo "Running Phase 11 factor shadow scoring focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/factors/shadow_scoring.py \
  quant_investor/factors/report.py \
  quant_investor/factors/store.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE11_RUN_STAGED_GATE:-1}" = "1" ]; then
  echo "Running staged upgrade quality gate..."
  PYTHON="$PYTHON_BIN" bash scripts/staged_upgrade_quality_gate.sh
fi

if [ "${PHASE11_RUN_FULL_UNIT:-0}" = "1" ]; then
  echo "Running full unit suite..."
  "$PYTHON_BIN" -m pytest tests/unit -q --maxfail=1
fi
