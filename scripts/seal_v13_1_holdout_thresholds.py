#!/usr/bin/env python3
"""Create the one canonical threshold seal before opening v13.1 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SEAL_ROOT = PROJECT_ROOT / "private" / "replay" / "threshold_seals"
CANONICAL_SEAL_LEDGER = CANONICAL_SEAL_ROOT / "seal_ledger.jsonl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.governance.replay_v13_1 import (  # noqa: E402
    FREEZE_EXCEPTION_CYCLE_ID,
    build_threshold_seal,
    write_manifest_atomic,
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_ledger() -> list[dict[str, Any]]:
    if not CANONICAL_SEAL_LEDGER.exists():
        return []
    rows: list[dict[str, Any]] = []
    previous_hash = ""
    for line_number, raw_line in enumerate(
        CANONICAL_SEAL_LEDGER.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"seal ledger line {line_number} is not an object")
        if payload.get("schema_version") != "myquant.holdout_threshold_seal_ledger.v2":
            raise ValueError(f"seal ledger schema mismatch at line {line_number}")
        if payload.get("freeze_exception_cycle_id") != FREEZE_EXCEPTION_CYCLE_ID:
            raise ValueError(f"seal ledger cycle mismatch at line {line_number}")
        entry_hash = str(payload.get("entry_hash") or "")
        unsigned = dict(payload)
        unsigned.pop("entry_hash", None)
        if str(payload.get("previous_entry_hash") or "") != previous_hash:
            raise ValueError(f"seal ledger chain mismatch at line {line_number}")
        if entry_hash != _canonical_sha256(unsigned):
            raise ValueError(f"seal ledger entry hash mismatch at line {line_number}")
        rows.append(payload)
        previous_hash = entry_hash
    return rows


def _append_ledger_atomic(entry: dict[str, Any]) -> str:
    CANONICAL_SEAL_ROOT.mkdir(parents=True, exist_ok=True)
    existing = (
        CANONICAL_SEAL_LEDGER.read_bytes()
        if CANONICAL_SEAL_LEDGER.exists()
        else b""
    )
    line = (
        json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = CANONICAL_SEAL_LEDGER.with_name(
        f".{CANONICAL_SEAL_LEDGER.name}.{os.getpid()}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(existing)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, CANONICAL_SEAL_LEDGER)
        os.chmod(CANONICAL_SEAL_LEDGER, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(CANONICAL_SEAL_LEDGER.read_bytes()).hexdigest()


def _acquire_cycle_seal_lock(*, dataset_sha256: str) -> Path:
    """Create the permanent one-seal lock before inspecting mutable state."""

    CANONICAL_SEAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = CANONICAL_SEAL_ROOT / ".freeze_exception_cycle.sealed.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            "freeze-exception cycle already sealed or awaiting recovery; "
            "a new dataset or threshold set is forbidden"
        ) from exc
    try:
        payload = {
            "schema_version": "myquant.holdout_cycle_lock.v1",
            "freeze_exception_cycle_id": FREEZE_EXCEPTION_CYCLE_ID,
            "dataset_sha256": dataset_sha256,
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return lock_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--validation-end-date", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_sha256 = str(args.dataset_sha256 or "").strip().lower()
    seal = build_threshold_seal(
        thresholds=json.loads(
            Path(args.thresholds_json).expanduser().resolve().read_text(
                encoding="utf-8"
            )
        ),
        dataset_sha256=dataset_sha256,
        validation_end_date=args.validation_end_date,
    )
    cycle_lock = _acquire_cycle_seal_lock(dataset_sha256=dataset_sha256)
    target = CANONICAL_SEAL_ROOT / f"{dataset_sha256}.json"
    try:
        ledger_rows = _read_ledger()
        if target.exists() or ledger_rows:
            raise FileExistsError(
                "this freeze-exception cycle already has a canonical threshold "
                "seal; a new dataset or threshold set cannot create a second "
                f"seal: {target}"
            )

        write_manifest_atomic(target, seal.to_dict())
        os.chmod(target, 0o600)
        artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        unsigned_entry = {
            "schema_version": "myquant.holdout_threshold_seal_ledger.v2",
            "freeze_exception_cycle_id": FREEZE_EXCEPTION_CYCLE_ID,
            "dataset_sha256": dataset_sha256,
            "threshold_hash": seal.threshold_hash,
            "validation_end_date": seal.validation_end_date,
            "seal_artifact_sha256": artifact_sha256,
            "seal_path": f"threshold_seals/{target.name}",
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "previous_entry_hash": "",
        }
        entry = {
            **unsigned_entry,
            "entry_hash": _canonical_sha256(unsigned_entry),
        }
        ledger_sha256 = _append_ledger_atomic(entry)
    except Exception:
        target.unlink(missing_ok=True)
        cycle_lock.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "sealed",
                "path": str(target),
                "artifact_sha256": artifact_sha256,
                "threshold_hash": seal.threshold_hash,
                "freeze_exception_cycle_id": FREEZE_EXCEPTION_CYCLE_ID,
                "seal_ledger_path": str(CANONICAL_SEAL_LEDGER),
                "seal_ledger_sha256": ledger_sha256,
                "cycle_lock_path": str(cycle_lock),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
