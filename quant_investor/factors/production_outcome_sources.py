"""Source-bound offline inputs for production observation diagnostics."""

from __future__ import annotations

import hashlib
import base64
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.close_session_authority import replay_close_session_authority
from quant_investor.market.market_data_reader import MarketDataReader
from .governance.errors import FactorGovernanceError
from .production_authority import FactorProductionStore
from .production_rollover import _read_owner_file


def exact_json(workspace: Path, ref: Mapping[str, str]) -> dict[str, Any]:
    raw, sha = _read_owner_file(Path(ref["path"]), root=workspace, label="outcome source")
    if sha != ref["sha256"]:
        raise FactorGovernanceError("OUTCOME_SOURCE_SHA_MISMATCH")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise FactorGovernanceError("OUTCOME_SOURCE_OBJECT_INVALID")
    return value


def close_calendar(workspace: Path, ref: Mapping[str, str]) -> dict[str, Any]:
    receipt = exact_json(workspace, ref)
    raw, sha = _read_owner_file(
        Path(receipt["raw_response_path"]), root=workspace, label="outcome Calendar raw"
    )
    if sha != receipt["raw_response_sha256"]:
        raise FactorGovernanceError("OUTCOME_CALENDAR_SHA_MISMATCH")
    replay_close_session_authority(receipt, raw)
    return receipt


def signal_calendar(store: FactorProductionStore, inputs: Mapping[str, Any]) -> list[str]:
    """Read the original runtime Calendar via its sealed source bundle, not a current pointer."""
    compilation = store._read_artifact_ref(
        inputs["calendar_compilation_ref"], label="signal Calendar"
    )
    bundle = store._read_artifact_ref(inputs["source_generation_ref"], label="signal source bundle")
    calendar_bundle_ref = next(
        row["source_ref"]
        for row in bundle["payload"]["sources"]
        if row["role"] == "exchange_calendar"
    )
    bundle = store._read_artifact_ref(calendar_bundle_ref, label="original Calendar sources")
    wanted = compilation["payload"]["calendar_json_file_ref"]["byte_sha256"]
    refs = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("kind") == "system.source_object" and "byte_sha256" in value:
                refs.append(value)
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(bundle["payload"])
    for ref in refs:
        metadata_path, _ = store._source_mirror_paths(ref)
        metadata = json.loads(store.read(metadata_path).data)
        if metadata.get("source_raw_sha256") != wanted:
            continue
        source, raw = store._mirrored_source_resolver(ref, 64 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() == wanted:
            rows = json.loads(raw)
            if isinstance(rows, dict):
                rows = rows.get("sessions", rows.get("rows"))
            if not isinstance(rows, list):
                raise FactorGovernanceError("SIGNAL_CALENDAR_ROWS_INVALID")
            return [str(row["date"]).replace("-", "") for row in rows if row["status"] == "OPEN"]
    raise FactorGovernanceError("SIGNAL_CALENDAR_SOURCE_MISSING")


def merged_sessions(original: list[str], current: Mapping[str, Any]) -> list[str]:
    newer = current["ordered_open_dates"]
    start, end = current["calendar_start_date"], current["calendar_end_date"]
    if not original or original[-1] < start:
        raise FactorGovernanceError("OUTCOME_CALENDAR_HISTORY_GAP")
    stop = min(original[-1], end)
    if [d for d in original if start <= d <= stop] != [d for d in newer if start <= d <= stop]:
        raise FactorGovernanceError("OUTCOME_CALENDAR_PREFIX_CONFLICT")
    return sorted(set(original) | set(newer))


def _file_hash(path: Path, root: Path) -> str:
    MarketDataReader(data_root=root / "data")._assert_path_has_no_symlink(
        path, boundary=root, label="outcome Parquet", require_exists=True
    )
    before = path.stat()
    if before.st_uid != os.geteuid() or not path.is_file():
        raise FactorGovernanceError("OUTCOME_PARQUET_UNSAFE")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise FactorGovernanceError("OUTCOME_PARQUET_CHANGED_DURING_READ")
    return digest.hexdigest()


def load_market_slice(
    workspace: Path, *, symbols: list[str], start: str, end: str
) -> dict[str, Any]:
    """Read only the observation universe and dates from the pointer-selected canonical snapshot.

    This diagnostic reader does not require unrelated FULL_MARKET completeness.
    It retains exact snapshot bytes and hashes every participating month partition.
    """
    pointer_path = workspace / "data/parquet/cn/_latest.json"
    raw, pointer_sha = _read_owner_file(
        pointer_path, root=workspace, label="outcome Market pointer"
    )
    pointer = json.loads(raw)
    reader = MarketDataReader(market="CN", data_root=workspace / "data")
    snapshot = reader._snapshot_from_payload(pointer)
    manifest_raw, manifest_sha = _read_owner_file(
        snapshot.manifest_path, root=workspace, label="outcome snapshot manifest"
    )
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("snapshot_id") != pointer.get("snapshot_id")
        or manifest.get("market") != "CN"
        or manifest.get("table_root") != str(snapshot.table_root)
    ):
        raise FactorGovernanceError("OUTCOME_SNAPSHOT_BINDING_INVALID")
    # Snapshot inventory is limited to an explicitly selected immutable table root;
    # partition names restrict I/O, never select an authoritative generation.
    files = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for month in range(1, 13):
            period = f"{year:04d}{month:02d}"
            if start[:6] <= period <= end[:6]:
                partition = snapshot.table_root / f"year={year:04d}" / f"month={month:02d}"
                if partition.exists():
                    files.extend(reader._v4_parquet_inventory(partition, label="outcome month"))
    if len(files) > 2048:
        raise FactorGovernanceError("OUTCOME_SOURCE_INVENTORY_BOUND")
    refs = [{"path": str(p), "sha256": _file_hash(p, workspace)} for p in sorted(files)]
    frames = [
        reader._read_dataset(
            Path(ref["path"]),
            date_range=(start, end),
            columns=["ts_code", "trade_date", "close"],
            symbol_filter=symbols,
            derive_symbol_column=False,
        )
        for ref in refs
    ]
    prices: dict[str, dict[str, str | None]] = {symbol: {} for symbol in symbols}
    import math

    for frame in frames:
        for row in frame.to_dict(orient="records"):
            symbol = str(row["ts_code"])
            day = str(row["trade_date"]).replace("-", "")[:8]
            if symbol not in prices or not start <= day <= end:
                raise FactorGovernanceError("OUTCOME_SLICE_SCOPE_MISMATCH")
            if day in prices[symbol]:
                raise FactorGovernanceError("OUTCOME_DUPLICATE_SYMBOL_SESSION")
            value = row.get("close")
            prices[symbol][day] = (
                format(float(value), ".17g")
                if value is not None and math.isfinite(float(value))
                else None
            )
    for ref in refs:
        if _file_hash(Path(ref["path"]), workspace) != ref["sha256"]:
            raise FactorGovernanceError("OUTCOME_SOURCE_CHANGED_DURING_QUERY")
    if (
        _read_owner_file(pointer_path, root=workspace, label="outcome Market pointer")[1]
        != pointer_sha
    ):
        raise FactorGovernanceError("OUTCOME_POINTER_CHANGED_DURING_QUERY")
    return {
        "prices": prices,
        "market_date": pointer.get("latest_complete_trade_date"),
        "pointer_bytes_base64": base64.b64encode(raw).decode("ascii"),
        "pointer_sha256": pointer_sha,
        "manifest_ref": {"path": str(snapshot.manifest_path), "sha256": manifest_sha},
        "file_refs": refs,
        "source_family": "CN_CANONICAL_RAW_CLOSE",
    }


def persist_source(store: FactorProductionStore, source: Mapping[str, Any]) -> dict[str, str]:
    raw = canonical_json_bytes(dict(source))
    sha = hashlib.sha256(raw).hexdigest()
    path = PurePosixPath("results/factors/outcome_sources") / f"{sha}.json"
    stored = store.write_exact_once(path, raw)
    return {"path": str(path), "sha256": stored.byte_sha256}
