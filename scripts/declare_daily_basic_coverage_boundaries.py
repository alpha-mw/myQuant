#!/usr/bin/env python3
"""Validate and publish owner-supplied daily-basic coverage boundaries.

A fetch checkpoint is only diagnostic evidence.  Missing rows, a common date
floor, or a successful partial provider response can never create provider
coverage authority.  Exact ``--checkpoint-outcome`` files therefore emit an
``UNCONFIRMED``/``DIAGNOSTIC_ONLY`` report and cannot be combined with
``--execute``.

Publication requires an exact owner-supplied receipt whose content hash and
source reference validate.  Writes are exact-once: an identical destination is
an idempotent success and different existing bytes are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "daily-basic-provider-coverage-boundaries.v2"
AUTHORITY_VERSION = "daily-basic-provider-coverage-boundary-authority.v1"
SOURCE_REF_FIELDS = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "semantic_sha256",
    "cutoff",
}
AUTHORITY_FIELDS = {
    "schema_version",
    "authority",
    "status",
    "reason_code",
    "coverage_starts",
    "source_ref",
    "record_sha256",
}
MAX_DECLARED_SYMBOLS = 400


def _json_no_duplicates(raw: bytes, *, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact_path(raw: str, *, label: str, must_exist: bool) -> Path:
    if not raw or any(token in raw for token in ("*", "?", "[", "]")):
        raise SystemExit(f"{label} must be an explicit path without glob syntax")
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute")
    if path.is_symlink():
        raise SystemExit(f"{label} must not be a symlink: {path}")
    if must_exist and not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def _load_outcomes(outcome_paths: list[Path]) -> list[dict[str, Any]]:
    """Read explicitly named checkpoint outcome files for diagnostics only."""

    rows: list[dict[str, Any]] = []
    for path in outcome_paths:
        payload = _json_no_duplicates(path.read_bytes(), label=str(path))
        outcomes = payload.get("outcomes")
        if not isinstance(outcomes, list):
            raise SystemExit(f"checkpoint outcomes malformed: {path}")
        rows.extend(row for row in outcomes if isinstance(row, dict))
    return [row for row in rows if str(row.get("table")) == "daily_basic"]


def _diagnose(outcome_paths: list[Path]) -> dict[str, Any]:
    outcomes = _load_outcomes(outcome_paths)
    latest: dict[str, dict[str, Any]] = {}
    for row in outcomes:
        symbol = str(row.get("symbol") or "").strip()
        if symbol:
            latest[symbol] = row
    incomplete = sorted(
        symbol for symbol, row in latest.items() if row.get("history_complete") is not True
    )
    return {
        "lane": "DIAGNOSTIC_ONLY",
        "status": "UNCONFIRMED",
        "reason_code": "UNCONFIRMED",
        "daily_basic_outcome_count": len(outcomes),
        "symbol_count": len(latest),
        "incomplete_symbols": incomplete,
        "authority_created": False,
        "note": (
            "A checkpoint or partial response cannot prove a provider coverage "
            "boundary; supply an owner-sealed exact authority receipt."
        ),
    }


def _validate_authority(path: Path) -> dict[str, Any]:
    payload = _json_no_duplicates(path.read_bytes(), label="authority receipt")
    if set(payload) != AUTHORITY_FIELDS:
        raise SystemExit("authority receipt fields do not match the exact schema")
    if payload["schema_version"] != AUTHORITY_VERSION:
        raise SystemExit("authority receipt schema_version mismatch")
    if payload["authority"] != "OWNER_SUPPLIED_EXACT_PROVIDER_METADATA":
        raise SystemExit("authority receipt has no accepted owner authority")
    if payload["status"] != "CONFIRMED":
        raise SystemExit("authority receipt must be CONFIRMED")
    if payload["reason_code"] != "PROVIDER_COVERAGE_BOUNDARY":
        raise SystemExit("authority receipt reason_code mismatch")
    source_ref = payload["source_ref"]
    if not isinstance(source_ref, dict) or set(source_ref) != SOURCE_REF_FIELDS:
        raise SystemExit("authority source_ref fields do not match the exact schema")
    for field in ("byte_sha256", "semantic_sha256"):
        value = str(source_ref[field])
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise SystemExit(f"authority source_ref {field} is invalid")
    for field in ("artifact_id", "artifact_version"):
        if not isinstance(source_ref[field], str) or not source_ref[field].strip():
            raise SystemExit(f"authority source_ref {field} is invalid")
    try:
        datetime.strptime(str(source_ref["cutoff"]), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise SystemExit("authority source_ref cutoff must be UTC seconds") from exc
    starts = payload["coverage_starts"]
    if not isinstance(starts, dict) or not 1 <= len(starts) <= MAX_DECLARED_SYMBOLS:
        raise SystemExit("coverage_starts must contain 1-400 exact symbol rows")
    normalized: dict[str, str] = {}
    for symbol, start in starts.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise SystemExit("coverage_starts contains an invalid symbol")
        if not isinstance(start, str) or len(start) != 8 or not start.isdigit():
            raise SystemExit(f"coverage start for {symbol!r} must be YYYYMMDD")
        normalized[symbol] = start
    if list(starts) != sorted(starts):
        raise SystemExit("coverage_starts must be ASCII sorted")
    unsigned = {key: value for key, value in payload.items() if key != "record_sha256"}
    if payload["record_sha256"] != _sha256(_canonical(unsigned)):
        raise SystemExit("authority receipt record_sha256 mismatch")
    return {**payload, "coverage_starts": normalized}


def _build_publication(authority: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "lane": "AUTHORITATIVE_OWNER_SUPPLIED",
        "status": "CONFIRMED",
        "reason_code": "PROVIDER_COVERAGE_BOUNDARY",
        "coverage_starts": authority["coverage_starts"],
        "source_ref": authority["source_ref"],
        "authority_receipt_sha256": authority["record_sha256"],
    }
    document["record_sha256"] = _sha256(_canonical(document))
    return document


def _backup_before_write(backup_dir: Path, source: Path, target: Path) -> None:
    if backup_dir.exists():
        raise SystemExit(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, mode=0o700)
    shutil.copy2(source, backup_dir / "authority_receipt.json")
    manifest = {
        "target": str(target),
        "target_existed_before": target.exists(),
        "authority_receipt_sha256": _sha256(source.read_bytes()),
    }
    if target.exists():
        shutil.copy2(target, backup_dir / "existing_target.json")
        manifest["existing_target_sha256"] = _sha256(target.read_bytes())
    raw = _canonical(manifest)
    (backup_dir / "backup_manifest.json").write_bytes(raw)
    if (backup_dir / "backup_manifest.json").read_bytes() != raw:
        raise SystemExit("backup manifest readback mismatch")


def _write_once(path: Path, raw: bytes) -> str:
    if path.exists():
        if path.is_symlink():
            raise SystemExit(f"refusing symlink destination: {path}")
        if path.read_bytes() == raw:
            return "ALREADY_PRESENT"
        raise SystemExit(f"refusing to overwrite different bytes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if path.read_bytes() != raw:
        raise SystemExit("publication readback mismatch")
    return "WRITTEN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint-outcome", action="append")
    inputs.add_argument("--authority-receipt")
    parser.add_argument("--out")
    parser.add_argument("--backup-dir")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_outcome:
        if args.execute:
            raise SystemExit("diagnostic checkpoint input can never be executed")
        paths = [
            _exact_path(raw, label="--checkpoint-outcome", must_exist=True)
            for raw in args.checkpoint_outcome
        ]
        if len(paths) != len(set(paths)):
            raise SystemExit("duplicate --checkpoint-outcome path")
        print(json.dumps(_diagnose(paths), indent=2, sort_keys=True))
        return 0

    source = _exact_path(args.authority_receipt, label="--authority-receipt", must_exist=True)
    authority = _validate_authority(source)
    document = _build_publication(authority)
    raw = _canonical(document) + b"\n"
    print(json.dumps(document, indent=2, sort_keys=True))
    if not args.execute:
        print("DRY RUN - validated only; nothing written")
        return 0
    if not args.out or not args.backup_dir:
        raise SystemExit("--execute requires --out and --backup-dir")
    out = _exact_path(args.out, label="--out", must_exist=False)
    backup = _exact_path(args.backup_dir, label="--backup-dir", must_exist=False)
    if out == source or out == backup or backup in out.parents or out in backup.parents:
        raise SystemExit("source, output, and backup paths must be separate")
    _backup_before_write(backup, source, out)
    status = _write_once(out, raw)
    print(f"{status}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
