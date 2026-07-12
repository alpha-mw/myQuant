from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.themes.membership import SCHEMA_VERSION, ThemeMembership
from quant_investor.themes.taxonomy import DEFAULT_TAXONOMY_PATH, ThemeTaxonomy


DEFAULT_DRAFT_DIR = Path("private/theme_knowledge/membership_drafts")
DEFAULT_CANONICAL_PATH = Path("private/theme_knowledge/theme_membership.v2.jsonl")


def build_membership_v2_draft(
    source_path: str | Path,
    *,
    symbol_master_path: str | Path | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    draft_dir: str | Path = DEFAULT_DRAFT_DIR,
) -> dict[str, Any]:
    source = Path(source_path)
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    taxonomy = ThemeTaxonomy.load(taxonomy_path)
    taxonomy_ids = {node.theme_id for node in taxonomy.nodes}
    raw_rows = _load_local_rows(source, source_bytes)
    records: list[ThemeMembership] = []
    validation_errors: list[str] = []
    for index, raw in enumerate(raw_rows, start=1):
        payload = dict(raw or {})
        payload.setdefault("schema_version", SCHEMA_VERSION)
        try:
            membership = ThemeMembership.from_mapping(payload)
            if membership.theme_id not in taxonomy_ids:
                raise ValueError(f"taxonomy node missing: {membership.theme_id}")
            if not membership.source_type or not membership.source_ref:
                raise ValueError("source_type and source_ref are required")
            if not _valid_updated_at(membership.updated_at):
                raise ValueError(
                    "updated_at is required and must be valid for canonical v2 approval"
                )
        except (TypeError, ValueError) as exc:
            validation_errors.append(f"row {index}: {exc}")
            continue
        records.append(membership)

    master_symbols: set[str] = set()
    symbol_master_hash = ""
    coverage_blockers: list[str] = []
    if symbol_master_path is None:
        coverage_blockers.append("trusted_symbol_master_missing")
    else:
        master_path = Path(symbol_master_path)
        master_bytes = master_path.read_bytes()
        symbol_master_hash = hashlib.sha256(master_bytes).hexdigest()
        master_symbols = _load_symbol_master(master_path, master_bytes)
        if not master_symbols:
            coverage_blockers.append("trusted_symbol_master_empty")
        unknown = sorted(
            {membership.symbol for membership in records} - master_symbols
        )
        if unknown:
            coverage_blockers.append(
                "symbols_missing_from_trusted_master=" + ",".join(unknown)
            )

    if validation_errors:
        status = "validation_blocked"
    elif coverage_blockers:
        status = "coverage_blocked"
    elif not records:
        status = "coverage_blocked"
        coverage_blockers.append("no_membership_records")
    else:
        status = "ready_for_approval"
    draft_payload = {
        "draft_schema_version": "theme_membership_migration_draft.v1",
        "draft_status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_hash": source_hash,
        "symbol_master_path": str(symbol_master_path or ""),
        "symbol_master_hash": symbol_master_hash,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_version": taxonomy.version,
        "record_count": len(records),
        "source_row_count": len(raw_rows),
        "validation_errors": validation_errors,
        "coverage_blockers": coverage_blockers,
        "formal_activation_ready": status == "ready_for_approval",
        "records": [membership.to_dict() for membership in records],
        "mapping_inferred": False,
        "network_called": False,
    }
    draft_id = hashlib.sha256(
        json.dumps(
            {
                "source_hash": source_hash,
                "symbol_master_hash": symbol_master_hash,
                "records": draft_payload["records"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    draft_payload["draft_id"] = draft_id
    target = Path(draft_dir) / f"{draft_id}.json"
    _atomic_write(
        target,
        json.dumps(draft_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "status": status,
        "draft_path": str(target),
        "draft_hash": hashlib.sha256(target.read_bytes()).hexdigest(),
        "record_count": len(records),
        "validation_errors": validation_errors,
        "coverage_blockers": coverage_blockers,
        "formal_activation_ready": status == "ready_for_approval",
        "mapping_inferred": False,
        "network_called": False,
    }


def approve_membership_v2_draft(
    draft_path: str | Path,
    *,
    expected_draft_hash: str,
    canonical_path: str | Path = DEFAULT_CANONICAL_PATH,
    expected_canonical_hash: str = "",
) -> dict[str, Any]:
    source = Path(draft_path)
    source_bytes = source.read_bytes()
    actual_draft_hash = hashlib.sha256(source_bytes).hexdigest()
    if not expected_draft_hash or actual_draft_hash != expected_draft_hash:
        raise ValueError("membership draft hash changed since review")
    payload = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("membership draft must contain an object")
    if str(payload.get("draft_status") or "") != "ready_for_approval":
        raise ValueError("membership draft coverage is blocked")
    if payload.get("formal_activation_ready") is not True:
        raise ValueError("membership draft is not activation ready")
    records: list[ThemeMembership] = []
    for record in list(payload.get("records") or []):
        if not isinstance(record, Mapping):
            continue
        membership = ThemeMembership.from_mapping(record)
        if not _valid_updated_at(membership.updated_at):
            raise ValueError(
                "canonical membership updated_at is missing or invalid"
            )
        records.append(membership)
    if len(records) != int(payload.get("record_count") or 0) or not records:
        raise ValueError("membership draft record count mismatch")
    target = Path(canonical_path)
    if target.exists():
        current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if not expected_canonical_hash or current_hash != expected_canonical_hash:
            raise ValueError("canonical membership hash mismatch")
    rendered = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in sorted(records, key=lambda item: (item.symbol, item.theme_id))
    )
    _atomic_write(target, rendered)
    canonical_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "status": "approved",
        "canonical_path": str(target),
        "canonical_hash": canonical_hash,
        "record_count": len(records),
        "formal_activation_ready": True,
        "mapping_inferred": False,
        "network_called": False,
    }


def validate_membership_v2_store(
    path: str | Path,
    *,
    symbol_master_path: str | Path | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    as_of: str = "",
) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {
            "status": "coverage_blocked",
            "record_count": 0,
            "blockers": ["membership_v2_store_missing"],
            "formal_activation_ready": False,
        }
    taxonomy = ThemeTaxonomy.load(taxonomy_path)
    taxonomy_ids = {node.theme_id for node in taxonomy.nodes}
    master_symbols = (
        _load_symbol_master(Path(symbol_master_path), Path(symbol_master_path).read_bytes())
        if symbol_master_path is not None
        else set()
    )
    blockers: list[str] = []
    if symbol_master_path is None:
        blockers.append("trusted_symbol_master_missing")
    records: list[ThemeMembership] = []
    for index, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            membership = ThemeMembership.from_mapping(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            blockers.append(f"line_{index}_invalid={exc}")
            continue
        if membership.theme_id not in taxonomy_ids:
            blockers.append(f"taxonomy_node_missing={membership.theme_id}")
        if not _valid_updated_at(membership.updated_at):
            blockers.append(f"updated_at_missing_or_invalid=line_{index}")
        if master_symbols and membership.symbol not in master_symbols:
            blockers.append(f"symbol_missing_from_master={membership.symbol}")
        if as_of and not membership.is_active(as_of):
            continue
        records.append(membership)
    status = "success" if records and not blockers else "coverage_blocked"
    return {
        "status": status,
        "record_count": len(records),
        "blockers": sorted(set(blockers)),
        "formal_activation_ready": status == "success",
        "canonical_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _valid_updated_at(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _load_local_rows(path: Path, raw: bytes) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    text = raw.decode("utf-8")
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            payload = payload.get("memberships")
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise ValueError("membership JSON must contain a list or memberships list")
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"membership line {line_number} must be an object")
            rows.append(dict(payload))
        return rows
    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        return [dict(row) for row in csv.DictReader(text.splitlines(), dialect=dialect)]
    raise ValueError(f"unsupported membership source format={suffix}")


def _load_symbol_master(path: Path, raw: bytes) -> set[str]:
    suffix = path.suffix.lower()
    text = raw.decode("utf-8")
    values: list[Any]
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            payload = payload.get("symbols")
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise ValueError("symbol master JSON must contain a symbols list")
        values = list(payload)
    elif suffix in {".jsonl", ".ndjson"}:
        values = []
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            values.append(payload.get("symbol") if isinstance(payload, Mapping) else payload)
    elif suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        values = [row.get("symbol") for row in csv.DictReader(text.splitlines(), dialect=dialect)]
    else:
        raise ValueError(f"unsupported symbol master format={suffix}")
    return {str(value or "").strip().upper() for value in values if str(value or "").strip()}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
