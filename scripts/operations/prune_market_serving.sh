#!/bin/zsh
# Bound the transitional CN Market serving projection: keep it only on the
# newest 3 published snapshots (plus the active one). Runtimes still pinned to
# older releases publish ~1.7 GB of serving per snapshot and never prune it.
#
# Runs the pruner from the frozen 15b3736 install (manifest-time ordering fix)
# under the same writer lock as snapshot publication. table/ is never touched.
set -euo pipefail

INSTALL=/Users/maxwell/mySpace/myQuant-release-authority/15b37361c0a180282a18731c252c90dc23e38d3c-unified-runtime/installs/15b37361c0a180282a18731c252c90dc23e38d3c-12d351e32d2841a365df5c307ec9b01c0f543386463d75930c12a676042d94e8
cd /Users/maxwell/mySpace/myQuant

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
