#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

PYTHON="$PYTHON_BIN" PHASE12_RUN_STAGED_GATE=0 scripts/phase12_factor_execution_cost_quality_gate.sh

"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_evidence.py \
  tests/unit/test_factor_evidence_store.py \
  tests/unit/test_factor_evidence_cli.py \
  tests/unit/test_factor_evidence_no_runtime_effect.py \
  -q

"$PYTHON_BIN" -m mypy \
  quant_investor/factors/evidence.py \
  quant_investor/factors/shadow_scoring.py \
  quant_investor/factors/store.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE13_RUN_STAGED_GATE:-1}" = "1" ]; then
  PYTHON="$PYTHON_BIN" scripts/staged_upgrade_quality_gate.sh
fi

if [ "${PHASE13_RUN_FULL_UNIT:-0}" = "1" ]; then
  "$PYTHON_BIN" -m pytest tests/unit -q --maxfail=1
fi
