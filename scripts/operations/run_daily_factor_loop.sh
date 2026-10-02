#!/bin/zsh
# myQuant 每日 factor loop 自动触发（2020 slot）
# 由 launchd（com.myquant.daily-factor-loop）在每天 20:25 (北京时间) 触发。
# 周末直接跳过；法定节假日由 daily-maintain 内部日历判断返回 NO_ACTION（幂等无害）。
#
# release 更新时，只需修改下面的 RELEASE_INSTALL_DIR / FACTOR_LOOP_CONTEXT* 变量。

set -euo pipefail
umask 077

# ---- 当前激活的 release（release 更新时改这里）----
RELEASE_INSTALL_DIR="/Users/maxwell/mySpace/myQuant-release-authority/660f066fb11e3bc100c89992a989f844c0c3a975-unified-runtime/installs/660f066fb11e3bc100c89992a989f844c0c3a975-f29e64e2eb6adcb364e9a54b9b8f2e72fd6408e68d88097f134c2b9eb03e4cc6"
INSTALLED_PYTHON="$RELEASE_INSTALL_DIR/bin/python"
EXPECTED_IMPORT_ROOT="$RELEASE_INSTALL_DIR/lib/python3.13/site-packages"

WORKSPACE_ROOT="/Users/maxwell/mySpace/myQuant"
RUN_ROOT="$WORKSPACE_ROOT/data/private/cn_daily_maintenance"

# ---- 当前激活的 factor-loop context（context 更新时改这里）----
FACTOR_LOOP_CONTEXT="$RUN_ROOT/factor_loop_contexts/660f066fb11e3bc100c89992a989f844c0c3a975.json"
FACTOR_LOOP_CONTEXT_SHA="864e4e7801549b74e499e563cc3f8d2b666027edcc3760609afb020ba4945053"

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
