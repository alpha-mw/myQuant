#!/usr/bin/env python3
"""Guarded historical CN bar backfill.

The command is offline and dry-run by default.  A live run requires both
``--execute`` and ``--allow-live-provider`` plus explicit absolute storage,
pointer, downloader, checkpoint, and backup paths.  It never discovers a
``latest`` pointer.  Batch receipts are exact-once files, so completed batches
are idempotently skipped and checkpoint evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ENDPOINT_FIELDS: dict[str, str] = {
    "daily": ("ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"),
    "adj_factor": "ts_code,trade_date,adj_factor",
    "daily_basic": ("ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20150101", help="inclusive YYYYMMDD")
    parser.add_argument("--end", default="20221231", help="inclusive YYYYMMDD")
    parser.add_argument("--target-data-root", required=True)
    parser.add_argument("--pointer-path", required=True)
    parser.add_argument("--download-data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backup-dir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-live-provider", action="store_true")
    parser.add_argument("--sleep-ms", type=int, default=350)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--batches", type=int, default=0)
    parser.add_argument(
        "--sessions",
        default="",
        help="comma-separated exact YYYYMMDD sessions; avoids calendar lookup",
    )
    return parser.parse_args()


def _path(raw: str, *, label: str, must_exist: bool, directory: bool) -> Path:
    if not raw or any(token in raw for token in ("*", "?", "[", "]")):
        raise SystemExit(f"{label} must be an explicit path without glob syntax")
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute")
    if path.is_symlink():
        raise SystemExit(f"{label} must not be a symlink: {path}")
    if must_exist and not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    if must_exist and directory != path.is_dir():
        expected = "directory" if directory else "file"
        raise SystemExit(f"{label} must be an existing {expected}: {path}")
    return path


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_once(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != raw:
            raise SystemExit(f"refusing to overwrite checkpoint bytes: {path}")
        return "ALREADY_PRESENT"
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise SystemExit(f"checkpoint readback mismatch: {path}")
    return "WRITTEN"


def _completed_batches(checkpoint_dir: Path) -> set[str]:
    if not checkpoint_dir.exists():
        return set()
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise SystemExit("checkpoint path is not a safe directory")
    completed: set[str] = set()
    for path in sorted(checkpoint_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise SystemExit(f"unexpected checkpoint entry: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid checkpoint receipt: {path}: {exc}") from exc
        batch = path.stem
        if (
            payload.get("schema_version") != "guarded-cn-history-backfill-batch.v1"
            or payload.get("batch") != batch
        ):
            raise SystemExit(f"checkpoint receipt binding mismatch: {path}")
        completed.add(batch)
    return completed


def _fetch_with_retry(
    maintainer: Any,
    endpoint: str,
    trade_date: str,
    *,
    max_retries: int,
    sleep_ms: int,
) -> tuple[pd.DataFrame, str]:
    last_error = ""
    for attempt in range(max_retries):
        frame, error = maintainer._fetch_endpoint(endpoint, trade_date, ENDPOINT_FIELDS[endpoint])
        if not error:
            return frame, ""
        last_error = error
        if error == "empty":
            return pd.DataFrame(), "empty"
        time.sleep((sleep_ms / 1000.0) * (2**attempt))
    return pd.DataFrame(), last_error


def _validate_dates(args: argparse.Namespace) -> list[str]:
    if any(len(value) != 8 or not value.isdigit() for value in (args.start, args.end)):
        raise SystemExit("--start and --end must be YYYYMMDD")
    if args.start > args.end:
        raise SystemExit("--start must not be later than --end")
    if args.sleep_ms < 0 or args.max_retries < 1 or args.batches < 0:
        raise SystemExit("sleep/retry/batch limits are invalid")
    sessions = [item.strip() for item in args.sessions.split(",") if item.strip()]
    if any(len(item) != 8 or not item.isdigit() for item in sessions):
        raise SystemExit("--sessions entries must be YYYYMMDD")
    if len(sessions) != len(set(sessions)):
        raise SystemExit("--sessions contains duplicates")
    return sorted(sessions)


def _backup_before_mutation(backup: Path, pointer: Path, checkpoint_dir: Path) -> bytes:
    if backup.exists():
        raise SystemExit(f"backup directory already exists: {backup}")
    backup.mkdir(parents=True, mode=0o700)
    pointer_raw = pointer.read_bytes()
    shutil.copy2(pointer, backup / "market_pointer.json")
    if (backup / "market_pointer.json").read_bytes() != pointer_raw:
        raise SystemExit("market pointer backup readback mismatch")
    checkpoint_rows: dict[str, str] = {}
    if checkpoint_dir.exists():
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise SystemExit("checkpoint path is not a safe directory")
        for path in sorted(checkpoint_dir.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise SystemExit(f"unexpected checkpoint entry: {path}")
            target = backup / "checkpoint" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if target.read_bytes() != path.read_bytes():
                raise SystemExit(f"checkpoint backup readback mismatch: {path}")
            checkpoint_rows[path.name] = _sha256(path.read_bytes())
    manifest = {
        "pointer_path": str(pointer),
        "pointer_sha256": _sha256(pointer_raw),
        "checkpoint_files": checkpoint_rows,
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (backup / "backup_manifest.json").write_bytes(raw)
    if (backup / "backup_manifest.json").read_bytes() != raw:
        raise SystemExit("backup manifest readback mismatch")
    return pointer_raw


def main() -> int:
    args = _parse_args()
    named_sessions = _validate_dates(args)
    data_root = _path(
        args.target_data_root,
        label="--target-data-root",
        must_exist=True,
        directory=True,
    )
    pointer = _path(args.pointer_path, label="--pointer-path", must_exist=True, directory=False)
    download_dir = _path(
        args.download_data_dir,
        label="--download-data-dir",
        must_exist=True,
        directory=True,
    )
    checkpoint_dir = _path(
        args.checkpoint_dir,
        label="--checkpoint-dir",
        must_exist=False,
        directory=True,
    )
    expected_pointer = data_root / "parquet" / "cn" / "_latest.json"
    if pointer != expected_pointer:
        raise SystemExit(
            "--pointer-path must exactly equal <target-data-root>/parquet/cn/_latest.json"
        )

    print(f"range={args.start}->{args.end} exact_sessions={len(named_sessions)}")
    print(f"target_data_root={data_root} pointer={pointer}")
    if not args.execute:
        print("DRY RUN - offline; no provider imported and nothing written")
        return 0
    if not args.allow_live_provider:
        raise SystemExit("--execute additionally requires --allow-live-provider")
    if not args.backup_dir:
        raise SystemExit("--execute requires --backup-dir")
    backup = _path(args.backup_dir, label="--backup-dir", must_exist=False, directory=True)
    pointer_before = _backup_before_mutation(backup, pointer, checkpoint_dir)

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    import tushare as ts

    from quant_investor.market.download import CNParquetBatchMaintainer

    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is not set")
    pro = ts.pro_api(token)
    sessions = named_sessions
    if not sessions:
        calendar = pro.trade_cal(exchange="SSE", start_date=args.start, end_date=args.end)
        sessions = sorted(
            str(value) for value in calendar.loc[calendar["is_open"] == 1, "cal_date"]
        )
    by_batch: dict[str, list[str]] = {}
    for session in sessions:
        by_batch.setdefault(session[:4], []).append(session)
    completed_batches = _completed_batches(checkpoint_dir)
    pending = [batch for batch in sorted(by_batch) if batch not in completed_batches]
    if args.batches:
        pending = pending[: args.batches]

    maintainer = CNParquetBatchMaintainer(data_dir=str(download_dir), data_root=data_root)
    current_pointer_raw = pointer_before
    for batch in pending:
        frames: list[pd.DataFrame] = []
        covered: list[str] = []
        failures: list[str] = []
        for session in by_batch[batch]:
            parts: dict[str, pd.DataFrame] = {}
            for endpoint in ENDPOINT_FIELDS:
                frame, error = _fetch_with_retry(
                    maintainer,
                    endpoint,
                    session,
                    max_retries=args.max_retries,
                    sleep_ms=args.sleep_ms,
                )
                if error and error != "empty":
                    failures.append(f"{session}:{endpoint}:{error}")
                parts[endpoint] = frame
                time.sleep(args.sleep_ms / 1000.0)
            if parts["daily"].empty or parts["adj_factor"].empty:
                failures.append(f"{session}:incomplete_core_endpoints")
                continue
            bars = maintainer._build_bars_frame(
                parts["daily"], parts["adj_factor"], parts["daily_basic"]
            )
            if bars.empty:
                failures.append(f"{session}:empty_bars_frame")
                continue
            frames.append(bars)
            covered.append(session)
        if not frames:
            raise SystemExit(f"{batch}: no usable sessions; refusing mutation")
        batch_frame = pd.concat(frames, ignore_index=True)
        pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
        manifest = maintainer.store.upsert_bars(
            batch_frame,
            target_trade_date=max(covered),
            target_trade_dates=covered,
            source=f"guarded_backfill_cn_history:{batch}",
            snapshot_id=(
                "guarded-backfill-"
                + batch
                + "-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            ),
            metadata={
                "backfill_batch": batch,
                "session_count": len(covered),
                "failure_count": len(failures),
                "latest_available_trade_date": pointer_payload["latest_available_trade_date"],
                "latest_complete_trade_date": pointer_payload["latest_complete_trade_date"],
            },
            expected_latest_pointer_sha256=_sha256(current_pointer_raw),
        )
        pointer_after = pointer.read_bytes()
        try:
            readback = json.loads(pointer_after.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit("post-upsert pointer readback is invalid") from exc
        if readback.get("snapshot_id") != manifest.get("snapshot_id"):
            raise SystemExit("post-upsert pointer snapshot binding mismatch")
        for field in ("latest_available_trade_date", "latest_complete_trade_date"):
            if readback.get(field) != pointer_payload.get(field):
                raise SystemExit(f"historical backfill changed {field}")
        receipt = {
            "schema_version": "guarded-cn-history-backfill-batch.v1",
            "batch": batch,
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "sessions": covered,
            "rows": len(batch_frame),
            "failures": failures,
            "snapshot_id": manifest.get("snapshot_id"),
            "previous_pointer_sha256": _sha256(current_pointer_raw),
            "published_pointer_sha256": _sha256(pointer_after),
        }
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        status = _write_once(checkpoint_dir / f"{batch}.json", raw)
        print(f"{batch}: {status} rows={len(batch_frame)}")
        current_pointer_raw = pointer_after
    print("backfill completed; exact pointer CAS and readback validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
