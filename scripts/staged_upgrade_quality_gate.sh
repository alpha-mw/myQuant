#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

echo "Running staged upgrade focused tests..."
"$PYTHON_BIN" -m pytest \
  tests/unit/test_v17_mainline_runtime.py \
  tests/unit/test_v17_public_python.py \
  tests/unit/test_v17_public_cli.py \
  tests/unit/test_v17_v4_forward_evidence.py \
  tests/unit/test_market_data_parquet_direct_maintenance.py \
  tests/unit/test_fundamental_generation_promotion.py \
  -q

echo "Running staged upgrade focused mypy..."
"$PYTHON_BIN" -m mypy \
  quant_investor/v17_mainline/constants.py \
  quant_investor/v17_mainline/contracts.py \
  quant_investor/v17_mainline/storage.py \
  quant_investor/v17_mainline/runtime.py \
  quant_investor/pipeline/mainline.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional

if [ "${STAGED_UPGRADE_RUN_FULL_UNIT:-0}" = "1" ]; then
  echo "Running full unit suite..."
  "$PYTHON_BIN" -m pytest tests/unit/ -q --maxfail=1
fi
