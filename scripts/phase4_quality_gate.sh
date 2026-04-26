#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

run() {
  echo "+ $*"
  "$@"
}

run "$PYTHON_BIN" -m pytest \
  tests/unit/test_phase1_contracts.py \
  tests/unit/test_bayesian_outcome_ledger.py \
  tests/unit/test_bayesian_calibration_v2.py \
  tests/unit/test_bayesian_posterior_overlay.py \
  tests/unit/test_bayesian_posterior.py \
  tests/unit/test_version_naming_consistency.py \
  -q

run "$PYTHON_BIN" -m mypy \
  quant_investor/bayesian/outcome_ledger.py \
  quant_investor/bayesian/calibration_v2.py \
  quant_investor/bayesian/posterior_overlay.py \
  quant_investor/bayesian/posterior.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [[ "${PHASE4_RUN_FULL_UNIT:-0}" == "1" ]]; then
  run "$PYTHON_BIN" -m pytest tests/unit/ -q --maxfail=1
fi
