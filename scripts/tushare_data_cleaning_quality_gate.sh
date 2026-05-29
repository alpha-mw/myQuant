#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m pytest \
  tests/unit/test_tushare_data_cleaning.py \
  tests/unit/test_tushare_factor_readiness.py \
  tests/unit/test_tushare_storage_optimization.py \
  tests/unit/test_tushare_download_auto_clean.py \
  tests/unit/test_tushare_data_cleaning_cli.py \
  -q

"$PYTHON_BIN" -m mypy \
  quant_investor/market/tushare_data_cleaning.py \
  quant_investor/market/download_cn.py \
  quant_investor/config.py \
  quant_investor/versioning.py \
  --ignore-missing-imports \
  --no-strict-optional
