"""Internal, closed-set immutable bytes for native Dashboard historical replay.

No filesystem fallback is permitted while a replay is active. Normal builders do
not create this context and retain their existing physical-source readers.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import hashlib
import json


@dataclass(frozen=True)
class RetainedDashboardSources:
    project_root: Path
    sources: Mapping[str, bytes]
    store_snapshot_ref: Mapping[str, str] | None = None

    def __post_init__(self):
        root = self.project_root.resolve(strict=True)
        copied = {}
        for relative, raw in self.sources.items():
            parts = Path(relative)
            if (
                type(relative) is not str
                or not relative
                or relative == "."
                or parts.is_absolute()
                or parts.as_posix() != relative
                or any(p in {".", ".."} for p in parts.parts)
                or "\\" in relative
                or type(raw) is not bytes
            ):
                raise ValueError("DASHBOARD_REPLAY_SOURCE_INVALID")
            copied[relative] = raw
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "sources", MappingProxyType(copied))
        if self.store_snapshot_ref is not None:
            from quant_investor.operations.daily_contract import validate_ref

            ref = validate_ref(self.store_snapshot_ref)
            object.__setattr__(self, "store_snapshot_ref", MappingProxyType(dict(ref)))

    def read(self, path: Path, project_root: Path) -> bytes:
        if project_root.resolve(strict=True) != self.project_root:
            raise ValueError("DASHBOARD_REPLAY_WORKSPACE_MISMATCH")
        try:
            relative = path.relative_to(self.project_root).as_posix()
        except ValueError as exc:
            raise ValueError("DASHBOARD_REPLAY_PATH_OUTSIDE_WORKSPACE") from exc
        if any(p in {".", ".."} for p in path.parts) or relative not in self.sources:
            raise ValueError("DASHBOARD_REPLAY_SOURCE_NOT_RETAINED:" + relative)
        return self.sources[relative]


_current: ContextVar[RetainedDashboardSources | None] = ContextVar(
    "dashboard_retained_sources", default=None
)


@contextmanager
def retained_dashboard_sources(sources: RetainedDashboardSources):
    if type(sources) is not RetainedDashboardSources or _current.get() is not None:
        raise ValueError("DASHBOARD_REPLAY_CONTEXT_INVALID")
    token = _current.set(sources)
    try:
        yield
    finally:
        _current.reset(token)


def retained_bytes(path: Path, project_root: Path) -> bytes | None:
    selected = _current.get()
    return None if selected is None else selected.read(path, project_root)


def require_live_dashboard_publication() -> None:
    if _current.get() is not None:
        raise ValueError("DASHBOARD_REPLAY_CANNOT_PUBLISH")


def registered_dashboard_store(record_root: Path, *, current_loader):
    from quant_investor.strategy_records.store import load_catalog_snapshot

    selected = _current.get()
    if selected is None:
        return current_loader(record_root)
    ref = selected.store_snapshot_ref
    if ref is None:
        raise ValueError("DASHBOARD_REPLAY_STORE_SNAPSHOT_MISSING")
    root = selected.project_root
    raw_pointer = selected.read(record_root / "_record_store/current.v1.json", root)
    if hashlib.sha256(raw_pointer).hexdigest() != ref["sha256"]:
        raise ValueError("DASHBOARD_REPLAY_STORE_POINTER_MISMATCH")
    snapshot = load_catalog_snapshot(
        record_root,
        pointer_relative_path=str((root / ref["path"]).relative_to(record_root)),
        expected_pointer_sha256=ref["sha256"],
    )
    pointer, catalog = snapshot
    if (
        json.loads(raw_pointer) != pointer
        or json.loads(selected.read(record_root / pointer["catalog_path"], root)) != catalog
    ):
        raise ValueError("DASHBOARD_REPLAY_STORE_CATALOG_MISMATCH")
    return snapshot


def retained_store_inventory(
    *, project_root: Path, record_root: Path, pointer_ref: Mapping[str, str]
) -> dict[str, bytes]:
    """Resolve the native retained catalog's finite exact file inventory, no scan."""
    import os
    import stat
    from quant_investor.strategy_records.store import load_catalog_snapshot, regular_file_sha256

    _, catalog = load_catalog_snapshot(
        record_root,
        pointer_relative_path=str((project_root / pointer_ref["path"]).relative_to(record_root)),
        expected_pointer_sha256=pointer_ref["sha256"],
    )
    sources = {}
    for record in catalog["records"]:
        if record.get("storage_state") != "ONLINE":
            continue
        for item in record.get("inventory", []):
            if item["type"] != "file":
                continue
            path = record_root / record["relative_path"] / item["path"]
            if path.resolve(strict=True) != path.absolute() or not path.is_relative_to(record_root):
                raise ValueError("DASHBOARD_REPLAY_STORE_INVENTORY_PATH_INVALID")
            metadata = path.stat()
            if (
                metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o600, 0o644}
            ):
                raise ValueError("DASHBOARD_REPLAY_STORE_INVENTORY_MODE_INVALID")
            digest, size = regular_file_sha256(path, label="retained Dashboard Store inventory")
            raw = path.read_bytes()
            if (
                digest != item["sha256"]
                or size != item["size"]
                or hashlib.sha256(raw).hexdigest() != digest
            ):
                raise ValueError("DASHBOARD_REPLAY_STORE_INVENTORY_SHA_MISMATCH")
            sources[path.relative_to(project_root).as_posix()] = raw
    # Native record validation also reads the frozen snapshot manifest and the
    # canonical table partition named by its immutable strict-close evidence.
    # Follow only those finite refs.
    from quant_investor.operations.daily_contract import validate_ref
    from quant_investor.operations.strict_close_table_source import (
        STRICT_CLOSE_EVIDENCE_SCHEMAS,
        StrictCloseTableSourceError,
        closes_from_table_partition,
        declared_table_partition,
        snapshot_manifest_relative,
        verify_evidence_closes,
    )

    for evidence_path, evidence_raw in list(sources.items()):
        if not evidence_path.endswith("/strict_market_close_evidence.json"):
            continue
        evidence = json.loads(evidence_raw)
        if evidence.get("schema_version") not in STRICT_CLOSE_EVIDENCE_SCHEMAS:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_EVIDENCE_SCHEMA_INVALID")
        try:
            manifest_relative = snapshot_manifest_relative(evidence)
        except StrictCloseTableSourceError as exc:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SNAPSHOT_INVALID") from exc
        snapshot_id = evidence["snapshot_id"]
        manifest_sha = evidence["snapshot_manifest_sha256"]
        validate_ref({"path": manifest_relative, "sha256": manifest_sha})
        manifest_path = project_root / manifest_relative
        if manifest_path.resolve(strict=True) != manifest_path.absolute():
            raise ValueError("DASHBOARD_REPLAY_CLOSE_MANIFEST_NOT_FROZEN")
        metadata = manifest_path.stat()
        if (
            metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) not in {0o600, 0o644}
        ):
            raise ValueError("DASHBOARD_REPLAY_CLOSE_MANIFEST_MODE_INVALID")
        digest, _ = regular_file_sha256(manifest_path, label="retained Dashboard close manifest")
        manifest_raw = manifest_path.read_bytes()
        if (
            digest != manifest_sha
            or hashlib.sha256(manifest_raw).hexdigest() != manifest_sha
            or json.loads(manifest_raw).get("snapshot_id") != snapshot_id
        ):
            raise ValueError("DASHBOARD_REPLAY_CLOSE_MANIFEST_SHA_MISMATCH")
        sources[manifest_relative] = manifest_raw
        try:
            candidates, pinned_sha = declared_table_partition(project_root, evidence, manifest_raw)
        except StrictCloseTableSourceError as exc:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_MANIFEST_TABLE_INVALID") from exc
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_MISSING:" + snapshot_id)
        if path.resolve(strict=True) != path.absolute():
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_PATH_INVALID")
        # Canonical table partitions are hardlinked across snapshots, so the
        # single-link rule applied to record files does not hold here.
        metadata = path.stat()
        if (
            metadata.st_uid != os.geteuid()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) not in {0o600, 0o644}
            or metadata.st_size > 256 * 1024 * 1024
        ):
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_MODE_OR_SIZE_INVALID")
        raw = path.read_bytes()
        if raw != path.read_bytes():
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_UNSTABLE")
        if pinned_sha is not None and hashlib.sha256(raw).hexdigest() != pinned_sha:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_SHA_MISMATCH")
        try:
            closes = closes_from_table_partition(
                raw,
                symbols=[
                    str(row.get("symbol") or row.get("ts_code")) for row in evidence["stocks"]
                ],
                trade_date=evidence.get("trade_date"),
            )
            verify_evidence_closes(evidence, closes)
        except StrictCloseTableSourceError as exc:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_VALUE_MISMATCH:" + str(exc)) from exc
        relative = path.relative_to(project_root).as_posix()
        if relative in sources and sources[relative] != raw:
            raise ValueError("DASHBOARD_REPLAY_CLOSE_SOURCE_CONFLICT")
        sources[relative] = raw
    return sources


class RetainedDashboardMarket:
    """Frozen native Market reads with the original captured logical pointer ref."""

    def __init__(self, *, project_root: Path, manifest_ref: Mapping[str, str]):
        from quant_investor.market.market_data_reader import MarketDataReader

        self.root = project_root
        self.reader = MarketDataReader(
            market="CN",
            data_root=project_root / "data",
            mode_policy="strict",
            frozen_snapshot_ref={
                "path": str(Path(manifest_ref["path"]).relative_to("data")),
                "sha256": manifest_ref["sha256"],
            },
        )

    def snapshot(self):
        snapshot = self.reader.snapshot()
        path = self.root / "data/parquet/cn/_latest.json"
        if retained_bytes(path, self.root) is None:
            raise ValueError("DASHBOARD_REPLAY_MARKET_REQUIRES_RETAINED_CONTEXT")
        return {**snapshot, "latest_pointer_path": str(path)}

    def table_partition_paths(self, *args, **kwargs):
        return self.reader.table_partition_paths(*args, **kwargs)

    def read_symbol_frame(self, *args, **kwargs):
        return self.reader.read_symbol_frame(*args, **kwargs)
