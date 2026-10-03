#!/bin/zsh
# myQuant 每日 factor loop 自动触发（2020 slot）
# 由 launchd（com.myquant.daily-factor-loop）在每天 20:25 (北京时间) 触发。
# 周末直接跳过；法定节假日由 daily-maintain 内部日历判断返回 NO_ACTION（幂等无害）。
#
# 当前 release 和 factor-loop context 只在 operations/releases/active.env 里定义；
# 换 release 按 docs/runbooks/release_repoint.md 操作，不要改本文件。

set -euo pipefail
umask 077

WORKSPACE_ROOT="/Users/maxwell/mySpace/myQuant"
RUN_ROOT="$WORKSPACE_ROOT/data/private/cn_daily_maintenance"
RELEASE_POINTER="$WORKSPACE_ROOT/operations/releases/active.env"

# ---- 读取并校验 release 指针；失败即告警退出，不猜任何 release ----
source "$WORKSPACE_ROOT/scripts/operations/release_pointer.sh"
if ! read_release_pointer "$RELEASE_POINTER" "$WORKSPACE_ROOT"; then
  /usr/bin/python3 "$WORKSPACE_ROOT/scripts/operations/notify_failure.py" \
    --job daily-factor-loop --exit-code 2 --detail "release pointer invalid: $RELEASE_POINTER" || true
  exit 2
fi

# ---- 周末判断（Mon=1 ... Sun=7），周末跳过 ----
DOW="$(date +%u)"
if (( DOW >= 6 )); then
  print -u2 -- "[$(date +%Y-%m-%dT%H:%M:%S%z)] 周末（weekday=$DOW），跳过 factor loop"
  exit 0
fi

# ---- 意外异常的 traceback 落到私有目录（CLI 的 stdout/stderr 仍不披露）----
export QUANT_INVESTOR_DIAGNOSTICS_DIR="$RUN_ROOT/diagnostics/internal_errors"

# ---- 触发 2020 slot（自包含 core + factor loop）----
if "$WORKSPACE_ROOT/scripts/operations/run_cn_daily_slot.sh" \
  --python "$INSTALLED_PYTHON" \
  --workspace-root "$WORKSPACE_ROOT" \
  --run-root "$RUN_ROOT" \
  --attempt-slot 2020 \
  --expected-import-root "$EXPECTED_IMPORT_ROOT" \
  --factor-loop-context "$FACTOR_LOOP_CONTEXT" \
  --expected-factor-loop-context-sha256 "$FACTOR_LOOP_CONTEXT_SHA256"; then
  exit_code=0
else
  exit_code=$?
fi

# ---- 非零退出：记入 logs/alerts.jsonl 并弹本地通知（尽力而为，不改变退出码）----
if (( exit_code != 0 )); then
  notify_args=("$WORKSPACE_ROOT/scripts/operations/notify_failure.py"
    --job daily-factor-loop --exit-code "$exit_code"
    --launcher-attempts "$RUN_ROOT/launcher_attempts")
  "$INSTALLED_PYTHON" -I "${notify_args[@]}" || /usr/bin/python3 "${notify_args[@]}" || true
fi
exit "$exit_code"
