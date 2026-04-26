#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

run() {
  echo "+ $*"
  "$@"
}

run "$PYTHON_BIN" -m pytest tests/unit/test_phase1_contracts.py -q

if [[ "${PHASE1_SKIP_FULL_UNIT:-0}" == "1" ]]; then
  echo "Skipping full unit suite because PHASE1_SKIP_FULL_UNIT=1"
else
  run "$PYTHON_BIN" -m pytest tests/unit/ -q
fi

if "$PYTHON_BIN" -m mypy --version >/dev/null 2>&1; then
  run "$PYTHON_BIN" -m mypy \
    quant_investor/agent_protocol.py \
    quant_investor/bayesian \
    quant_investor/branch_config.py \
    --ignore-missing-imports \
    --no-strict-optional
else
  echo "Skipping mypy because it is not installed for $PYTHON_BIN"
fi
