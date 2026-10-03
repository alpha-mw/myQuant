#!/bin/zsh
# Bound the transitional CN Market serving projection: keep it only on the
# newest 3 published snapshots (plus the active one). Runtimes still pinned to
# older releases publish ~1.7 GB of serving per snapshot and never prune it.
#
# Runs the pruner from the frozen install named by operations/releases/active.env
# (PRUNE_RELEASE_INSTALL_DIR while the active release lacks the command, else the
# active install) under the same writer lock as snapshot publication. table/ is
# never touched.
set -euo pipefail

WORKSPACE_ROOT=/Users/maxwell/mySpace/myQuant
source "$WORKSPACE_ROOT/scripts/operations/release_pointer.sh"
read_release_pointer "$WORKSPACE_ROOT/operations/releases/active.env" "$WORKSPACE_ROOT"
INSTALL="${PRUNE_RELEASE_INSTALL_DIR:-$RELEASE_INSTALL_DIR}"
cd "$WORKSPACE_ROOT"

"$INSTALL/bin/python" -P - <<'PY'
import json
from datetime import datetime, timezone

from quant_investor.market.market_data_store import MarketDataStore

store = MarketDataStore(market="CN", data_root="data")
with store._market_writer_lock():
    result = store.prune_snapshot_serving_layers(keep_recent=3, dry_run=False)
print(json.dumps({
    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "active": result.get("active_snapshot_id"),
    "pruned": result.get("pruned"),
    "released_gb": round(result.get("released_bytes", 0) / 1e9, 2),
}))
PY
