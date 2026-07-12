"""Hash-bound historical factor baselines for report-only shadow replays.

The manifest names every historical factor explicitly and binds each entry to
the canonical JSON bytes of its current registry record.  Loading a baseline
never mutates the formal registry or makes a deprecated record selectable in
production; it only builds an in-memory report-only scoring registry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from quant_investor.factors.governance import FactorRecord
from quant_investor.factors.runtime import MinedFactorRegistry


HISTORICAL_BASELINE_MANIFEST_SCHEMA_VERSION = (
    "factor-historical-shadow-baseline-manifest.v1"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def registry_record_sha256(record_payload: Mapping[str, Any]) -> str:
    """Hash one raw registry record without FactorRecord normalization."""

    return _sha256(dict(record_payload))


def historical_baseline_manifest_sha256(payload: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(payload))
    body.pop("manifest_sha256", None)
    return _sha256(body)


def build_historical_baseline_manifest(
    *,
    registry_path: str | Path,
    baseline_id: str,
    factor_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Build (but do not write) a self-hashed explicit baseline manifest."""

    resolved = Path(registry_path).expanduser()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("factor registry must be a JSON object")
    raw_records = payload.get("factors")
    if not isinstance(raw_records, list):
        raise ValueError("factor registry factors must be a list")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "").strip()
        if name:
            by_name.setdefault(name, []).append(dict(raw))

    entries: list[dict[str, Any]] = []
    for raw_name, raw_weight in factor_weights.items():
        name = str(raw_name or "").strip()
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"historical factor registry record count:{name}:{len(matches)}")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"historical factor shadow_weight invalid:{name}") from exc
        if not math.isfinite(weight) or abs(weight) <= 1e-12:
            raise ValueError(f"historical factor shadow_weight invalid:{name}")
        record = matches[0]
        entries.append(
            {
                "name": name,
                "registry_record_sha256": registry_record_sha256(record),
                "shadow_weight": weight,
                "source_state": str(record.get("state") or ""),
                "source_version": str(record.get("version") or ""),
            }
        )
    manifest = {
        "schema_version": HISTORICAL_BASELINE_MANIFEST_SCHEMA_VERSION,
        "baseline_id": str(baseline_id or "").strip(),
        "expected_factor_count": len(entries),
        "factors": entries,
    }
    if not manifest["baseline_id"]:
        raise ValueError("historical baseline_id must be non-empty")
    manifest["manifest_sha256"] = historical_baseline_manifest_sha256(manifest)
    return manifest


def load_historical_shadow_baseline(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    expected_factor_count: int,
) -> tuple[MinedFactorRegistry, dict[str, Any]]:
    """Load and verify one shadow-only baseline from formal registry bytes."""

    manifest_resolved = Path(manifest_path).expanduser()
    registry_resolved = Path(registry_path).expanduser()
    try:
        manifest = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"historical baseline manifest unreadable:{exc}") from exc
    try:
        registry_payload = json.loads(registry_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"factor registry unreadable:{exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("historical baseline manifest must be an object")
    if manifest.get("schema_version") != HISTORICAL_BASELINE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("historical baseline manifest schema mismatch")
    supplied_manifest_hash = str(manifest.get("manifest_sha256") or "")
    actual_manifest_hash = historical_baseline_manifest_sha256(manifest)
    if supplied_manifest_hash != actual_manifest_hash:
        raise ValueError("historical baseline manifest hash mismatch")
    baseline_id = str(manifest.get("baseline_id") or "").strip()
    if not baseline_id:
        raise ValueError("historical baseline_id missing")
    entries = manifest.get("factors")
    if not isinstance(entries, list):
        raise ValueError("historical baseline factors must be a list")
    if int(manifest.get("expected_factor_count") or 0) != int(expected_factor_count):
        raise ValueError("historical baseline expected_factor_count mismatch")
    if len(entries) != int(expected_factor_count):
        raise ValueError(
            "historical baseline factor count mismatch:"
            f"expected={expected_factor_count}:actual={len(entries)}"
        )
    if not isinstance(registry_payload, Mapping):
        raise ValueError("factor registry must be a JSON object")
    raw_records = registry_payload.get("factors")
    if not isinstance(raw_records, list):
        raise ValueError("factor registry factors must be a list")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "").strip()
        if name:
            by_name.setdefault(name, []).append(dict(raw))

    records: list[FactorRecord] = []
    names: list[str] = []
    source_record_hashes: dict[str, str] = {}
    current_selectable_names: list[str] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"historical baseline factor[{index}] must be an object")
        name = str(raw_entry.get("name") or "").strip()
        if not name or name in names:
            raise ValueError(f"historical baseline factor name duplicate_or_missing:{name}")
        names.append(name)
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"historical factor registry record count:{name}:{len(matches)}")
        raw_record = matches[0]
        actual_record_hash = registry_record_sha256(raw_record)
        expected_record_hash = str(
            raw_entry.get("registry_record_sha256") or ""
        ).strip()
        if actual_record_hash != expected_record_hash:
            raise ValueError(f"historical factor registry record hash mismatch:{name}")
        if str(raw_entry.get("source_state") or "") != str(
            raw_record.get("state") or ""
        ):
            raise ValueError(f"historical factor source_state mismatch:{name}")
        if str(raw_entry.get("source_version") or "") != str(
            raw_record.get("version") or ""
        ):
            raise ValueError(f"historical factor source_version mismatch:{name}")
        try:
            shadow_weight = float(raw_entry.get("shadow_weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"historical factor shadow_weight invalid:{name}") from exc
        if not math.isfinite(shadow_weight) or abs(shadow_weight) <= 1e-12:
            raise ValueError(f"historical factor shadow_weight invalid:{name}")
        source_record = FactorRecord.from_dict(raw_record)
        if source_record.selectable_in_quant_branch():
            current_selectable_names.append(name)
        shadow_record = FactorRecord.from_dict(raw_record)
        shadow_record.weight = shadow_weight
        shadow_record.metadata = {
            **dict(shadow_record.metadata or {}),
            "historical_shadow_only": True,
            "historical_shadow_baseline_id": baseline_id,
            "historical_shadow_manifest_sha256": actual_manifest_hash,
            "historical_shadow_source_record_sha256": actual_record_hash,
        }
        records.append(shadow_record)
        source_record_hashes[name] = actual_record_hash

    audit = {
        "schema_version": HISTORICAL_BASELINE_MANIFEST_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "manifest_path": str(manifest_resolved),
        "manifest_sha256": actual_manifest_hash,
        "expected_factor_count": int(expected_factor_count),
        "factor_count": len(records),
        "factor_names": names,
        "source_record_sha256_by_name": source_record_hashes,
        "current_selectable_factor_count": len(current_selectable_names),
        "current_selectable_factor_names": current_selectable_names,
        "historical_non_selectable_factor_count": (
            len(records) - len(current_selectable_names)
        ),
        "formal_registry_mutated": False,
        "production_eligible": False,
        "runtime_mode": "report_only_shadow",
    }
    return (
        MinedFactorRegistry(
            schema_version=str(
                registry_payload.get("schema_version")
                or "mined-factor-registry.v1"
            ),
            factors=records,
            metadata={
                "historical_shadow_only": True,
                "historical_shadow_baseline": audit,
            },
        ),
        audit,
    )


__all__ = [
    "HISTORICAL_BASELINE_MANIFEST_SCHEMA_VERSION",
    "build_historical_baseline_manifest",
    "historical_baseline_manifest_sha256",
    "load_historical_shadow_baseline",
    "registry_record_sha256",
]
