#!/usr/bin/env python3
"""One live forward macro observation roll, run from the repo.

This is the daily macro stage's own call (`daily_components.macro`), assembled
outside the frozen release so the store can be rolled forward while the active
release still computes the catch-up window from `local_target_trade_date`
alone.  It is a bridge, not a new capability: same prepare/commit transaction,
same market-snapshot-as-coverage binding, same live decision clock.  Remove it
from service once the active release carries `_parent_local_target`
(commit 69d1aee).

    uv run python scripts/operations/cn_macro_forward_roll.py --target 20261008 --execute

Without --execute nothing is prepared; the assembly is validated read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
MARKET_POINTER = WORKSPACE / "data/parquet/cn/_latest.json"
PIT_POINTER = WORKSPACE / "data/parquet/cn/reference/stock_basic_membership_latest.json"
SCOPE_ARTIFACT = WORKSPACE / "data/cn_universe/cn_index_components.json"
RELEASE_ROOT = WORKSPACE / "data/parquet/cn/macro_release_calendar"
OBSERVATIONS_ROOT = WORKSPACE / "data/parquet/cn/macro_observations"
TRANSACTION_ROOT = WORKSPACE / "data/private/macro_recovery_transactions"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _loaded_store(path: Path) -> tuple[dict, dict]:
    pointer = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(pointer.get("metadata") or {})
    if not metadata:
        metadata = dict((pointer.get("generation_manifest") or {}).get("metadata") or {})
    return pointer, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="open session, e.g. 20261008")
    parser.add_argument(
        "--execute", action="store_true", help="commit; default is a read-only check"
    )
    args = parser.parse_args()
    target = str(args.target).replace("-", "")

    from quant_investor.macro.maintenance import prepare_cn_macro_maintenance_transaction
    from quant_investor.macro.maintenance_transaction import (
        _preflight_prepared_commit,
        commit_prepared_macro_transaction,
    )

    market_pointer_raw = MARKET_POINTER.read_bytes()
    market = json.loads(market_pointer_raw)
    snapshot_manifest = Path(market["manifest_path"]).resolve(strict=True)
    snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    frontier = str(snapshot.get("latest_complete_trade_date") or "")
    if target > frontier:
        raise SystemExit(
            f"target {target} is beyond the market frontier {frontier}: "
            "the session's market capture must complete first"
        )
    snapshot_sha = _sha(snapshot_manifest)
    pointer, metadata = _loaded_store(OBSERVATIONS_ROOT / "_latest.json")
    if (
        str(metadata.get("local_target_trade_date") or "") == target
        and metadata.get("local_snapshot_manifest_sha256") == snapshot_sha
        and metadata.get("local_coverage_manifest_sha256") == snapshot_sha
        and metadata.get("local_scope_artifact_sha256") == _sha(SCOPE_ARTIFACT)
    ):
        print(
            json.dumps(
                {
                    "status": "NO_ACTION",
                    "target_date": target,
                    "generation_id": pointer.get("generation_id"),
                }
            )
        )
        return 0

    transaction_id = f"macro-forward-{target}"
    run_root = TRANSACTION_ROOT / transaction_id
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_root.chmod(0o700)
    journal_root = run_root / "journals"
    journal_root.mkdir(mode=0o700, exist_ok=True)
    journal_root.chmod(0o700)

    result = prepare_cn_macro_maintenance_transaction(
        market="CN",
        target_date=target,
        snapshot_manifest_path=str(snapshot_manifest),
        expected_snapshot_manifest_sha256=snapshot_sha,
        coverage_manifest_path=str(snapshot_manifest),
        expected_coverage_manifest_sha256=snapshot_sha,
        scope_artifact_path=str(SCOPE_ARTIFACT),
        expected_scope_artifact_sha256=_sha(SCOPE_ARTIFACT),
        release_root=str(RELEASE_ROOT),
        expected_release_pointer_sha256=_sha(RELEASE_ROOT / "_latest.json"),
        observations_root=str(OBSERVATIONS_ROOT),
        expected_observations_pointer_sha256=_sha(OBSERVATIONS_ROOT / "_latest.json"),
        market_pointer_path=str(MARKET_POINTER),
        expected_market_pointer_sha256=hashlib.sha256(market_pointer_raw).hexdigest(),
        pit_pointer_path=str(PIT_POINTER),
        expected_pit_pointer_sha256=_sha(PIT_POINTER),
        authority_mode="canonical",
        release_run_id=f"r{transaction_id}",
        observations_run_id=f"o{transaction_id}",
        private_run_root=str(run_root),
        transaction_run_id=transaction_id,
        allow_live=True,
    )
    prepared = {
        "status": result.get("status"),
        "target_date": result.get("target_date"),
        "prepared_path": result.get("prepared_path"),
        "prepared_sha256": result.get("prepared_sha256"),
    }
    print("PREPARED:", json.dumps(prepared, ensure_ascii=False))
    if prepared["status"] != "PREPARED":
        return 1
    preflight = _preflight_prepared_commit(
        prepared_path=prepared["prepared_path"],
        expected_prepared_sha256=prepared["prepared_sha256"],
        expected_target_date=target,
        market_pointer_path=str(MARKET_POINTER),
        expected_market_pointer_sha256=_sha(MARKET_POINTER),
        pit_pointer_path=str(PIT_POINTER),
        expected_pit_pointer_sha256=_sha(PIT_POINTER),
    )
    print("preflight:", json.dumps(preflight, ensure_ascii=False)[:300])
    if not args.execute:
        print("dry run: commit not executed (pass --execute to commit)")
        return 0

    commit = commit_prepared_macro_transaction(
        prepared_path=prepared["prepared_path"],
        expected_prepared_sha256=prepared["prepared_sha256"],
        journal_root=str(journal_root),
        journal_run_id=transaction_id,
        market_pointer_path=str(MARKET_POINTER),
        expected_market_pointer_sha256=_sha(MARKET_POINTER),
        pit_pointer_path=str(PIT_POINTER),
        expected_pit_pointer_sha256=_sha(PIT_POINTER),
    )
    print(
        "COMMIT:",
        json.dumps({key: str(value) for key, value in commit.items()}, ensure_ascii=False)[:600],
    )
    if commit.get("status") != "SUCCESS" or commit.get("terminal") is not True:
        return 1
    terminal_file = journal_root / transaction_id / "0007-terminal.json"
    receipt = {
        "schema_version": "cn-macro-forward-roll-receipt.v1",
        "target_date": target,
        "transaction_id": transaction_id,
        "committed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": {key: str(value) for key, value in commit.items()},
        "terminal_journal_path": str(terminal_file),
        "terminal_journal_sha256": _sha(terminal_file) if terminal_file.is_file() else "",
        "observations_pointer_sha256": _sha(OBSERVATIONS_ROOT / "_latest.json"),
        "release_pointer_sha256": _sha(RELEASE_ROOT / "_latest.json"),
    }
    receipt_path = run_root / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path.chmod(0o600)
    print("receipt:", json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
