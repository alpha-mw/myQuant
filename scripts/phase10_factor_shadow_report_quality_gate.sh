#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if [ "${PHASE10_RUN_PHASE9_GATE:-1}" = "1" ]; then
  echo "Running Phase 9 factor library quality gate..."
  PHASE9_RUN_STAGED_GATE="${PHASE10_RUN_PHASE9_STAGED_GATE:-0}" \
    PHASE9_RUN_FULL_UNIT="${PHASE10_RUN_PHASE9_FULL_UNIT:-0}" \
    PYTHON="$PYTHON_BIN" \
    bash scripts/phase9_factor_library_quality_gate.sh
fi

echo "Running Phase 10 factor shadow report focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_factor_library_shadow_status.py \
  tests/unit/test_observability_factor_governance.py \
  tests/unit/test_factor_shadow_no_runtime_wiring.py \
  tests/unit/test_cn_aggressive_portfolio_tracker.py::test_run_tracker_renders_formal_diagnostics_without_changing_action \
  -q

echo "Running Phase 10 factor shadow report focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/factors/report.py \
  quant_investor/observability.py \
  quant_investor/monitoring/cn_aggressive_portfolio_tracker.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE10_RUN_STAGED_GATE:-1}" = "1" ]; then
  echo "Running staged upgrade quality gate..."
  PYTHON="$PYTHON_BIN" bash scripts/staged_upgrade_quality_gate.sh
fi

if [ "${PHASE10_RUN_FULL_UNIT:-0}" = "1" ]; then
  echo "Running full unit suite..."
  "$PYTHON_BIN" -m pytest tests/unit -q --maxfail=1
fi
