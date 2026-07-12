#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export MYQUANT_DISABLE_LOCAL_LLM=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "[v13.1] focused Dashboard, Theme, Factor and joint-gate tests"
"$PYTHON_BIN" -m pytest \
  tests/unit/test_dashboard_benchmark_backfill.py \
  tests/unit/test_dashboard_benchmark_fill_merge.py \
  tests/unit/test_dashboard_contract_v2.py \
  tests/unit/test_dashboard_export_check.py \
  tests/unit/test_dashboard_tushare_benchmark_export.py \
  tests/unit/test_theme_protocol_v2.py \
  tests/unit/test_theme_protocol_v2_cli.py \
  tests/unit/test_pevc_knowledge.py \
  tests/unit/test_theme_membership_migration.py \
  tests/unit/test_theme_dag_tactical_integration.py \
  tests/unit/test_theme_post_control_dag_integration.py \
  tests/unit/test_theme_candidate_pool.py \
  tests/unit/test_theme_metadata_in_context.py \
  tests/unit/test_theme_portfolio_caps.py \
  tests/unit/test_portfolio_constructor_theme_caps.py \
  tests/unit/test_factor_governance_protocol_v2.py \
  tests/unit/test_factor_governance.py \
  tests/unit/test_factor_health_automation.py \
  tests/unit/test_quant_branch_factor_mining.py \
  tests/unit/test_quant_governance_blocked.py \
  tests/unit/test_candidate_review_after_funnel.py \
  tests/unit/test_v13_1_joint_replay_gate.py \
  -q

echo "[v13.1] static Dashboard JavaScript contract and syntax"
node --test portfolio_dashboard/tests/dashboard_contract_v2.test.js
node --check portfolio_dashboard/app.js
node --check portfolio_dashboard/js/data.js
node --check portfolio_dashboard/js/metrics.js
node --check portfolio_dashboard/js/charts.js
node --check portfolio_dashboard/js/ui.js
node --check portfolio_dashboard/js/generated_records.js

if [ "${V13_1_RUN_FULL_SWEEPS:-1}" = "1" ]; then
  echo "[v13.1] full Theme sweep"
  "$PYTHON_BIN" -m pytest \
    tests/unit/test_theme_*.py \
    tests/unit/test_bayesian_theme_metadata.py \
    tests/unit/test_deterministic_funnel_theme_boost.py \
    tests/unit/test_risk_guard_theme.py \
    tests/unit/test_portfolio_constructor_theme_caps.py \
    -q --no-cov

  echo "[v13.1] full Factor and Quant-governance sweep"
  "$PYTHON_BIN" -m pytest \
    tests/unit/test_factor_*.py \
    tests/unit/test_quant_branch_factor_mining.py \
    tests/unit/test_quant_factor_selection_shadow.py \
    tests/unit/test_quant_governance_blocked.py \
    -q --no-cov
fi

echo "[v13.1] public CLI/package smoke"
"$PYTHON_BIN" -m pytest \
  tests/unit/test_public_package_smoke.py \
  tests/unit/test_data_layer.py \
  tests/unit/test_forecast_snapshot_cache.py \
  tests/unit/test_llm_env_inventory.py \
  tests/unit/test_tushare_url_defaults.py \
  tests/integration/test_review_layer_timeout_budget.py \
  -q

echo "[v13.1] staged-upgrade compatibility gate"
PYTHON="$PYTHON_BIN" bash scripts/staged_upgrade_quality_gate.sh

echo "[v13.1] diff hygiene"
git diff --check
