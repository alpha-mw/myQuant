#!/usr/bin/env python3
"""One live forward macro observation roll, run from a clean checkout.

This is the daily macro stage's own call (`daily_components.macro`), assembled
outside the frozen release so the store can be rolled forward while the active
release still computes the catch-up window from `local_target_trade_date`
alone.  It is a bridge, not a new capability: same prepare/commit transaction,
same market-snapshot-as-coverage binding, same live decision clock (so it does
fetch the two official coverage index pages, like the daily stage does).

It is also *not* the whole 2026-10-08 evening: a blocked daily macro stage
writes a new `MACRO_WRITE_VETO.json`, so after this roll the veto must be
cleared again (with this roll's terminal journal sha) and the readiness
closure rebuilt — see `docs/runbooks/macro_recovery_20261006.md`.

Run it from a git-clean checkout of the committed code (the pinned worktree
`~/mySpace/myQuant-worktrees/macro-bridge`), pointing `--workspace` at the
live workspace; `PYTHONPATH` makes the worktree's code win over the main
tree's venv:

    PYTHONPATH=~/mySpace/myQuant-worktrees/macro-bridge \\
      /Users/maxwell/mySpace/myQuant/.venv/bin/python \\
      ~/mySpace/myQuant-worktrees/macro-bridge/scripts/operations/cn_macro_forward_roll.py \\
      --workspace /Users/maxwell/mySpace/myQuant --target 20261008 --execute

The script refuses to run when the importing repository has uncommitted
changes under `quant_investor/` or `scripts/`, when the daily maintenance lock
is held, or when the latest maintenance attempt for the target has not ended
cleanly (`core_blockers` non-empty).  It is idempotent: a repeat reports
`NO_ACTION`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
LOCK_NAME = ".daily-maintenance.lock"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _code_root() -> Path:
    import quant_investor

    return Path(quant_investor.__file__).resolve().parents[1]


def _workspace(value: str | None) -> Path:
    candidate = Path(value).expanduser() if value else DEFAULT_WORKSPACE
    resolved = candidate.resolve()
    if not (resolved / "data/parquet/cn").is_dir():
        raise SystemExit(
            f"{resolved} is not the live workspace (no data/parquet/cn); pass --workspace"
        )
    return resolved


def _assert_clean_code() -> tuple[Path, str]:
    root = _code_root()
    try:
        top = Path(_git(["rev-parse", "--show-toplevel"], root))
        status = _git(["status", "--porcelain", "--", "quant_investor", "scripts"], top)
        commit = _git(["rev-parse", "HEAD"], top)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot verify the code checkout: {exc}") from exc
    if status:
        raise SystemExit(
            "refusing to write production state from uncommitted code:\n"
            + status
            + "\nRun from a clean worktree of the committed code, e.g. "
            "~/mySpace/myQuant-worktrees/macro-bridge (see the runbook)."
        )
    return top, commit


def _assert_run_ended(workspace: Path, target: str) -> dict:
    attempts_root = workspace / "data/private/cn_daily_maintenance/attempts"
    candidates = []
    for attempt_json in sorted(attempts_root.glob("*/attempt.json")):
        try:
            payload = json.loads(attempt_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(payload.get("target_date") or "") == target and payload.get("mode") == "execute":
            candidates.append((attempt_json.parent, payload))
    if not candidates:
        raise SystemExit(f"no execute attempt for {target} yet: wait for the daily launcher")
    attempt_root, payload = candidates[-1]
    ended = attempt_root / "ended.json"
    if not ended.is_file():
        raise SystemExit(f"attempt {attempt_root.name} has not ended yet: wait for ended.json")
    ended_payload = json.loads(ended.read_text(encoding="utf-8"))
    core_blockers = [str(value) for value in payload.get("core_blockers") or []]
    if core_blockers:
        raise SystemExit(f"core_blockers non-empty on {attempt_root.name}: {core_blockers}")
    return {
        "attempt": attempt_root.name,
        "ended_state": ended_payload.get("state"),
        "ended_at": ended_payload.get("ended_at"),
        "core_blockers": core_blockers,
        "macro_status": payload.get("macro_status"),
    }


def _veto_state(workspace: Path) -> dict:
    veto = workspace / "data/private/cn_daily_maintenance/MACRO_WRITE_VETO.json"
    if not veto.is_file():
        return {"present": False, "path": str(veto), "sha256": None}
    return {"present": True, "path": str(veto), "sha256": _sha(veto)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="open session, e.g. 20261008")
    parser.add_argument("--workspace", default=None, help="live workspace (data lives here)")
    parser.add_argument(
        "--execute", action="store_true", help="commit; default prepares but does not commit"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="guards only: no network, no prepare, no writes beyond the receipt",
    )
    args = parser.parse_args()
    target = str(args.target).replace("-", "")

    workspace = _workspace(args.workspace)
    code_root, code_commit = _assert_clean_code()
    ended = _assert_run_ended(workspace, target)

    from quant_investor.market.daily_maintenance import DailyMaintenanceError, _RunLock
    from quant_investor.macro.maintenance import prepare_cn_macro_maintenance_transaction
    from quant_investor.macro.maintenance_transaction import (
        _preflight_prepared_commit,
        commit_prepared_macro_transaction,
    )

    market_pointer = workspace / "data/parquet/cn/_latest.json"
    pit_pointer = workspace / "data/parquet/cn/reference/stock_basic_membership_latest.json"
    scope_artifact = workspace / "data/cn_universe/cn_index_components.json"
    release_root = workspace / "data/parquet/cn/macro_release_calendar"
    observations_root = workspace / "data/parquet/cn/macro_observations"
    transaction_root = workspace / "data/private/macro_recovery_transactions"

    market_pointer_raw = market_pointer.read_bytes()
    market = json.loads(market_pointer_raw)
    snapshot_manifest = Path(market["manifest_path"]).resolve(strict=True)
    snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    frontier = str(snapshot.get("latest_complete_trade_date") or "")
    if frontier != target:
        raise SystemExit(
            f"market frontier is {frontier}, not {target}: the target session's capture must land first"
        )
    snapshot_sha = _sha(snapshot_manifest)
    pointer = json.loads((observations_root / "_latest.json").read_text(encoding="utf-8"))
    metadata = dict(pointer.get("metadata") or {})
    if not metadata:
        metadata = dict((pointer.get("generation_manifest") or {}).get("metadata") or {})

    run_root = transaction_root / f"macro-forward-{target}"
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_root.chmod(0o700)
    journal_root = run_root / "journals"
    journal_root.mkdir(mode=0o700, exist_ok=True)
    journal_root.chmod(0o700)
    transaction_id = f"macro-forward-{target}"

    def _receipt(extra: dict) -> dict:
        return {
            "schema_version": "cn-macro-forward-roll-receipt.v1",
            "workspace": str(workspace),
            "target_date": target,
            "code_root": str(code_root),
            "code_git_commit": code_commit,
            "run_ended": ended,
            "macro_write_veto": _veto_state(workspace),
            "market_snapshot_manifest_sha256": snapshot_sha,
            **extra,
        }

    def _finish(value: dict) -> int:
        receipt_path = run_root / "receipt.json"
        receipt_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_path.chmod(0o600)
        print("receipt:", json.dumps(value, ensure_ascii=False))
        if value["macro_write_veto"]["present"]:
            print(
                "NOTE: a macro write veto is present; the next step is to clear it with "
                f"sha256 {value['macro_write_veto']['sha256']} (reason bound to this roll's "
                "terminal journal sha), then rebuild the readiness closure."
            )
        return 0

    lock_path = workspace / "data/private/cn_daily_maintenance" / LOCK_NAME
    try:
        lock = _RunLock(lock_path)
        lock.__enter__()
    except DailyMaintenanceError as exc:
        raise SystemExit(f"daily maintenance lock refused: {exc}") from exc
    try:
        if args.check:
            return _finish(
                _receipt(
                    {
                        "status": "CHECK_OK",
                        "generation_id": pointer.get("generation_id"),
                        "observations_pointer_sha256": _sha(observations_root / "_latest.json"),
                        "note": (
                            "guards passed (lock free, run landed, frontier on target); "
                            "no network, no prepare"
                        ),
                    }
                )
            )
        if (
            str(metadata.get("local_target_trade_date") or "") == target
            and metadata.get("local_snapshot_manifest_sha256") == snapshot_sha
            and metadata.get("local_coverage_manifest_sha256") == snapshot_sha
            and metadata.get("local_scope_artifact_sha256") == _sha(scope_artifact)
        ):
            return _finish(
                _receipt(
                    {
                        "status": "NO_ACTION",
                        "generation_id": pointer.get("generation_id"),
                        "observations_pointer_sha256": _sha(observations_root / "_latest.json"),
                    }
                )
            )

        result = prepare_cn_macro_maintenance_transaction(
            market="CN",
            target_date=target,
            snapshot_manifest_path=str(snapshot_manifest),
            expected_snapshot_manifest_sha256=snapshot_sha,
            coverage_manifest_path=str(snapshot_manifest),
            expected_coverage_manifest_sha256=snapshot_sha,
            scope_artifact_path=str(scope_artifact),
            expected_scope_artifact_sha256=_sha(scope_artifact),
            release_root=str(release_root),
            expected_release_pointer_sha256=_sha(release_root / "_latest.json"),
            observations_root=str(observations_root),
            expected_observations_pointer_sha256=_sha(observations_root / "_latest.json"),
            market_pointer_path=str(market_pointer),
            expected_market_pointer_sha256=hashlib.sha256(market_pointer_raw).hexdigest(),
            pit_pointer_path=str(pit_pointer),
            expected_pit_pointer_sha256=_sha(pit_pointer),
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
            market_pointer_path=str(market_pointer),
            expected_market_pointer_sha256=_sha(market_pointer),
            pit_pointer_path=str(pit_pointer),
            expected_pit_pointer_sha256=_sha(pit_pointer),
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
            market_pointer_path=str(market_pointer),
            expected_market_pointer_sha256=_sha(market_pointer),
            pit_pointer_path=str(pit_pointer),
            expected_pit_pointer_sha256=_sha(pit_pointer),
        )
        print(
            "COMMIT:",
            json.dumps({key: str(value) for key, value in commit.items()}, ensure_ascii=False)[
                :600
            ],
        )
        if commit.get("status") != "SUCCESS" or commit.get("terminal") is not True:
            return 1
        terminal_file = journal_root / transaction_id / "0007-terminal.json"
        return _finish(
            _receipt(
                {
                    "status": "SUCCESS",
                    "transaction_id": transaction_id,
                    "committed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "commit": {key: str(value) for key, value in commit.items()},
                    "terminal_journal_path": str(terminal_file),
                    "terminal_journal_sha256": (
                        _sha(terminal_file) if terminal_file.is_file() else ""
                    ),
                    "observations_pointer_sha256": _sha(observations_root / "_latest.json"),
                    "release_pointer_sha256": _sha(release_root / "_latest.json"),
                }
            )
        )
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
