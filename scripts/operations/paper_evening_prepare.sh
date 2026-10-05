#!/bin/zsh
# Evening Paper preparation for one CN session: capture the session's price
# limits, emit the intent + eligibility files for the plans that target it, and
# write the next session's plans. It never fills anything — the owner reviews the
# prepared files and runs `paper risk-exit-run` (docs/runbooks/paper_session_fill.md).
#
# Runs on the workspace scripts with the workspace venv, like the dashboard job:
# these are workspace plans, not part of the frozen production DAG.
set -euo pipefail

WORKSPACE_ROOT=/Users/maxwell/mySpace/myQuant
TODAY=$(TZ=Asia/Shanghai date +%Y%m%d)
LOG="$WORKSPACE_ROOT/logs/paper-evening-prepare-$TODAY.log"

cd "$WORKSPACE_ROOT"

# Only run when today is the session the pipeline is expecting next, so holidays
# and weekends never spend a provider call on a date that cannot have limits.
NEXT_SESSION=$(/usr/bin/env python3 - <<'PY'
import glob, json
proofs = sorted(glob.glob('/Users/maxwell/mySpace/myQuant/results/operations/daily_production/CN/*/calendar-future/proofs/*.json'))
print(json.load(open(proofs[-1]))['next_open_session'] if proofs else '')
PY
)

if [[ "$TODAY" != "$NEXT_SESSION" ]]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) skip: today=$TODAY next_session=$NEXT_SESSION" >> "$LOG"
  exit 0
fi

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) paper evening prepare $TODAY"
  "$WORKSPACE_ROOT/.venv/bin/python" \
    "$WORKSPACE_ROOT/scripts/operations/paper_evening_prepare.py" \
    --trade-date "$TODAY"
} >> "$LOG" 2>&1
