#!/bin/zsh
# myQuant 每日 factor loop 自动触发（2020 slot）
# 由 launchd（com.myquant.daily-factor-loop）在每天 20:25 (北京时间) 触发。
# 周末直接跳过；法定节假日由 daily-maintain 内部日历判断返回 NO_ACTION（幂等无害）。
#
# release 更新时，只需修改下面的 RELEASE_INSTALL_DIR / FACTOR_LOOP_CONTEXT* 变量。

set -euo pipefail
umask 077

# ---- 当前激活的 release（release 更新时改这里）----
RELEASE_INSTALL_DIR="/Users/maxwell/mySpace/myQuant-release-authority/5dd585ba1d81bb4834cb30be18a75c4b5b36da6a-unified-runtime/installs/5dd585ba1d81bb4834cb30be18a75c4b5b36da6a-f7656834455f9eb9879987972d8463ea2026de11ab1ffeea59aa3a37c8c7cc4d"
INSTALLED_PYTHON="$RELEASE_INSTALL_DIR/bin/python"
EXPECTED_IMPORT_ROOT="$RELEASE_INSTALL_DIR/lib/python3.13/site-packages"

WORKSPACE_ROOT="/Users/maxwell/mySpace/myQuant"
RUN_ROOT="$WORKSPACE_ROOT/data/private/cn_daily_maintenance"

# ---- 当前激活的 factor-loop context（context 更新时改这里）----
FACTOR_LOOP_CONTEXT="$RUN_ROOT/factor_loop_contexts/5dd585ba1d81bb4834cb30be18a75c4b5b36da6a.json"
FACTOR_LOOP_CONTEXT_SHA="15fab00b8ef865c49b64bf1df0c8cf11b5f39d962626cf94c8059bbcbddb7745"

# ---- 周末判断（Mon=1 ... Sun=7），周末跳过 ----
DOW="$(date +%u)"
if (( DOW >= 6 )); then
  print -u2 -- "[$(date -Iseconds)] 周末（weekday=$DOW），跳过 factor loop"
  exit 0
fi

# ---- 触发 2020 slot（自包含 core + factor loop）----
exec "$WORKSPACE_ROOT/scripts/operations/run_cn_daily_slot.sh" \
  --python "$INSTALLED_PYTHON" \
  --workspace-root "$WORKSPACE_ROOT" \
  --run-root "$RUN_ROOT" \
  --attempt-slot 2020 \
  --expected-import-root "$EXPECTED_IMPORT_ROOT" \
  --factor-loop-context "$FACTOR_LOOP_CONTEXT" \
  --expected-factor-loop-context-sha256 "$FACTOR_LOOP_CONTEXT_SHA"
