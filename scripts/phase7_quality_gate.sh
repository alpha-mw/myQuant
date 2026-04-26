#!/usr/bin/env bash
set -euo pipefail

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

"${PYTHON_BIN}" -m pytest \
  tests/unit/test_phase1_contracts.py \
  tests/unit/test_bayesian_outcome_ledger.py \
  tests/unit/test_bayesian_calibration_v2.py \
  tests/unit/test_bayesian_posterior_overlay.py \
  tests/unit/test_data_quality_contract.py \
  tests/unit/test_risk_tensor.py \
  tests/unit/test_portfolio_optimizer.py \
  tests/unit/test_bayesian_posterior.py \
  tests/unit/test_version_naming_consistency.py \
  -q

"${PYTHON_BIN}" -m mypy \
  quant_investor/branch_config.py \
  quant_investor/bayesian/outcome_ledger.py \
  quant_investor/bayesian/calibration_v2.py \
  quant_investor/bayesian/posterior_overlay.py \
  quant_investor/data_quality_contract.py \
  quant_investor/risk_tensor.py \
  quant_investor/portfolio_optimizer.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${PHASE7_RUN_FULL_UNIT:-0}" = "1" ]; then
  "${PYTHON_BIN}" -m pytest tests/unit/ -q --maxfail=1
fi
