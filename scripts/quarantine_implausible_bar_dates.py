#!/usr/bin/env python3
"""Quarantine implausible CN bar dates from explicitly named parquet files.

The command is offline and dry-run by default.  It never scans a snapshot,
follows ``latest``, or guesses serving/table paths.  Mutation requires
``--execute`` plus exact table/serving files, a new backup directory, a new
quarantine file, and a new evidence file.  Every input is backed up before the
first replacement and every output is read back byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

IMPLAUSIBLE_BEFORE = "19900101"
EVIDENCE_TYPE = "cn_bars_implausible_trade_date_quarantine.v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_file(raw: str, *, label: str) -> Path:
    if not raw or any(token in raw for token in ("*", "?", "[", "]")):
        raise SystemExit(f"{label} must be an explicit path without glob syntax")
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute")
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} must name an existing regular file: {path}")
    return path


def _new_path(raw: str, *, label: str) -> Path:
    if not raw or any(token in raw for token in ("*", "?", "[", "]")):
        raise SystemExit(f"{label} must be an explicit path without glob syntax")
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute")
    if path.exists() or path.is_symlink():
        raise SystemExit(f"{label} must not already exist: {path}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-file", action="append", required=True)
    parser.add_argument("--serving-file", action="append", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--backup-dir")
    parser.add_argument("--quarantine-file")
    parser.add_argument("--evidence-file")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _victims(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path)
    if "trade_date" not in frame or "ts_code" not in frame:
        raise SystemExit(f"bar schema missing trade_date/ts_code: {path}")
    mask = frame["trade_date"].astype("string") < IMPLAUSIBLE_BEFORE
    return frame, frame.loc[mask].copy()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.quarantine-tmp-{os.getpid()}")
    if tmp.exists():
        raise SystemExit(f"temporary path already exists: {tmp}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _backup_inputs(backup: Path, paths: list[Path]) -> dict[str, str]:
    backup.mkdir(parents=True, mode=0o700)
    hashes: dict[str, str] = {}
    for index, path in enumerate(paths):
        target = backup / f"{index:04d}-{path.name}"
        shutil.copy2(path, target)
        original_hash = _sha256(path)
        if _sha256(target) != original_hash:
            raise SystemExit(f"backup readback mismatch: {path}")
        hashes[str(path)] = original_hash
    manifest = (
        json.dumps({"inputs": hashes}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    manifest_path = backup / "backup_manifest.json"
    manifest_path.write_bytes(manifest)
    if manifest_path.read_bytes() != manifest:
        raise SystemExit("backup manifest readback mismatch")
    return hashes


def main() -> int:
    args = _parse_args()
    table_paths = [_exact_file(raw, label="--table-file") for raw in args.table_file]
    serving_paths = [_exact_file(raw, label="--serving-file") for raw in args.serving_file]
    all_paths = table_paths + serving_paths
    if len(set(all_paths)) != len(all_paths):
        raise SystemExit("input parquet paths must be unique")

    frames: dict[Path, pd.DataFrame] = {}
    victims: list[pd.DataFrame] = []
    for path in all_paths:
        frame, bad = _victims(path)
        frames[path] = frame
        if not bad.empty:
            tagged = bad.copy()
            tagged["_source_path"] = str(path)
            victims.append(tagged)
    if not victims:
        print("no implausible-date rows found; ALREADY_CLEAN")
        return 0
    quarantined = pd.concat(victims, ignore_index=True)
    symbols = sorted({str(value) for value in quarantined["ts_code"]})
    print(f"snapshot={args.snapshot_id} victims={len(quarantined)} symbols={symbols}")
    if not args.execute:
        print("DRY RUN - nothing written")
        return 0
    if not args.backup_dir or not args.quarantine_file or not args.evidence_file:
        raise SystemExit("--execute requires --backup-dir, --quarantine-file, and --evidence-file")
    backup = _new_path(args.backup_dir, label="--backup-dir")
    quarantine = _new_path(args.quarantine_file, label="--quarantine-file")
    evidence_path = _new_path(args.evidence_file, label="--evidence-file")
    mutation_paths = {backup, quarantine, evidence_path}
    if len(mutation_paths) != 3:
        raise SystemExit("backup, quarantine, and evidence paths must be distinct")

    before = _backup_inputs(backup, all_paths)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    quarantined.to_parquet(quarantine, index=False)
    quarantine_hash = _sha256(quarantine)
    if len(pd.read_parquet(quarantine)) != len(quarantined):
        raise SystemExit("quarantine parquet readback mismatch")

    post_hashes: dict[str, str] = {}
    post_rows: dict[str, int] = {}
    for path in all_paths:
        frame = frames[path]
        keep = frame.loc[frame["trade_date"].astype("string") >= IMPLAUSIBLE_BEFORE]
        _atomic_parquet(keep, path)
        readback = pd.read_parquet(path)
        if bool((readback["trade_date"].astype("string") < IMPLAUSIBLE_BEFORE).any()):
            raise SystemExit(f"implausible row survived exact readback: {path}")
        if len(readback) != len(keep):
            raise SystemExit(f"row-count readback mismatch: {path}")
        post_hashes[str(path)] = _sha256(path)
        post_rows[str(path)] = len(readback)

    evidence: dict[str, Any] = {
        "evidence_type": EVIDENCE_TYPE,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "snapshot_id": args.snapshot_id,
        "status": "QUARANTINED",
        "threshold": IMPLAUSIBLE_BEFORE,
        "symbols": symbols,
        "row_count": len(quarantined),
        "input_sha256": before,
        "post_sha256": post_hashes,
        "post_rows": post_rows,
        "quarantine_file": str(quarantine),
        "quarantine_sha256": quarantine_hash,
        "backup_manifest": str(backup / "backup_manifest.json"),
    }
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(raw)
    if evidence_path.read_bytes() != raw:
        raise SystemExit("evidence readback mismatch")
    print(f"QUARANTINED evidence={evidence_path} backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
