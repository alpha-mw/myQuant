"""Strict offline readback and private publication for v4.1 cycle sources.

This module is intentionally narrower than the factor-governance runtime.  It
binds only caller-supplied absolute paths, inventories one immutable snapshot
table, and publishes owner-only research evidence.  It has no discovery,
registry, replay, proposal, provider, portfolio, broker, order, or trade path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from quant_investor.codex_review.storage import (
    CONTROL_MAX_BYTES,
    REQUEST_MAX_BYTES,
    ProtocolError,
    assert_cas,
    canonical_json_bytes,
    parse_strict_json_bytes,
    run_lock,
    sha256_bytes,
)
from quant_investor.market.pit_universe import (
    PITUniverseRecord,
    PITUniverseStore,
)
from quant_investor.factors.governance_cycle_state_v4_1 import (
    validate_genesis_cycle_state_v4_1,
)
from quant_investor.factors.governance_source_v4_1 import (
    validate_design_source_node_v4_1,
)


INPUT_BINDING_SCHEMA_VERSION = "factor-governance-cutoff-input-binding.v4.1"
READBACK_REPORT_SCHEMA_VERSION = "factor-governance-source-readback.v4.1"
BLOCKER_REPORT_SCHEMA_VERSION = "factor-governance-source-blocker.v4.1"
CUTOFF_SOURCE_NODE_SCHEMA_VERSION = "factor-governance-cutoff-source-node.v4.1"
CYCLE_ROOT_SCHEMA_VERSION = "factor-governance-cycle-root.v4.1"
INPUT_BINDING_FILENAME = "cutoff_input_binding.v4_1.json"
DESIGN_SOURCE_FILENAME = "design_source.v4_1.json"
SOURCE_CHAIN_NODE_FILENAME = "source_chain_node.v4_1.json"
PRECOMMITTED_STATE_FILENAME = "cycle_state.precommitted.v4_1.json"
READBACK_REPORT_FILENAME = "source_readback_report.v4_1.json"
BLOCKER_REPORT_FILENAME = "source_readback_blocker.v4_1.json"

EXPECTED_UNIVERSE = "full_a"
EXPECTED_FULL_A_COUNT = 5502
EXPECTED_SERVING_INVENTORY_COUNT = 5728
SOURCE_USE_PROHIBITED = "DIAGNOSTIC_ONLY_PROHIBITED_AS_ELIGIBILITY"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CN_SYMBOL = re.compile(r"[0-9]{6}\.(?:SH|SZ|BJ)")
_JSON_MAX_BYTES = 16 * 1024 * 1024


class FactorGovernanceSourceReadbackV4_1Error(ValueError):
    """Raised when an explicit v4.1 source boundary cannot be proven."""


@dataclass(frozen=True)
class BoundCutoffInputsV4_1:
    """Serializable binding plus normalized inputs for the pure source contract."""

    binding: dict[str, Any]
    calendar_sessions: tuple[str, ...]
    component_symbols: tuple[str, ...]
    pit_records: tuple[dict[str, Any], ...]
    bound_table_symbol_row_counts: tuple[tuple[str, int], ...]


def _absolute_path(value: str | Path, *, label: str) -> Path:
    raw = os.fspath(value)
    if not raw or raw != raw.strip():
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be an explicit absolute path"
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be an explicit absolute path without traversal"
        )
    return Path(os.path.abspath(os.fspath(candidate)))


def _sha256(value: str, *, label: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "empty":
        return value
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _safe_id(value: str, *, label: str) -> str:
    if (
        type(value) is not str
        or _SAFE_ID.fullmatch(value) is None
        or value in {".", ".."}
        or ".." in value
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be one safe path segment"
        )
    return value


def _iso_date(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be YYYY-MM-DD"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be canonical YYYY-MM-DD"
        )
    return value


def _file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_signature(path: Path, *, label: str) -> tuple[int, ...]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} missing or unreadable: {path}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be a real directory: {path}"
        )
    if metadata.st_uid != os.getuid():
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} owner mismatch: {path}"
        )
    return _file_signature(metadata)


def _assert_owned_no_symlink_directory_chain(
    path: Path,
    *,
    boundary: Path,
    label: str,
) -> None:
    target = _absolute_path(path, label=label)
    root = _absolute_path(boundary, label=f"{label} boundary")
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} escapes the explicit project boundary"
        ) from exc
    current = root
    _directory_signature(current, label=f"{label} boundary")
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} contains an unsafe path segment"
            )
        current = current / part
        _directory_signature(current, label=label)


def _stable_file_bytes(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    max_bytes: int | None = None,
) -> bytes:
    target = _absolute_path(path, label=label)
    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
    descriptor: int | None = None
    try:
        before = os.lstat(target)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} must be a regular non-symlink file: {target}"
            )
        if before.st_uid != os.getuid():
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} owner mismatch: {target}"
            )
        if int(before.st_nlink) != 1:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} hard-link count must be one: {target}"
            )
        if max_bytes is not None and int(before.st_size) > max_bytes:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} exceeds {max_bytes} bytes"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if _file_signature(before) != _file_signature(opened):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} changed while opening: {target}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(after):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} changed while reading: {target}"
            )
    except FactorGovernanceSourceReadbackV4_1Error:
        raise
    except OSError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} read failed: {target}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    actual = sha256_bytes(raw)
    if actual != expected:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} SHA-256 mismatch"
        )
    return raw


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = parse_strict_json_bytes(raw, max_bytes=_JSON_MAX_BYTES)
    except ProtocolError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} must be a JSON object"
        )
    return dict(value)


def _declared_path_matches(
    raw_path: Any,
    *,
    expected: Path,
    anchors: Sequence[Path],
    label: str,
) -> None:
    if type(raw_path) is not str or not raw_path or raw_path != raw_path.strip():
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} path missing or unsafe"
        )
    declared = Path(raw_path)
    if ".." in declared.parts:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} path contains traversal"
        )
    expected_absolute = _absolute_path(expected, label=label)
    if declared.is_absolute():
        candidates = {_absolute_path(declared, label=label)}
    else:
        candidates = {
            Path(os.path.abspath(os.fspath(anchor / declared)))
            for anchor in anchors
        }
    if expected_absolute not in candidates:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"{label} path does not match the explicit input"
        )


def _binding_record(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "absolute_path": str(path),
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _project_root_from_pointer(pointer_path: Path) -> Path:
    expected_tail = ("data", "parquet", "cn", "_latest.json")
    if tuple(pointer_path.parts[-4:]) != expected_tail:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "latest pointer must be the explicit project data/parquet/cn/_latest.json"
        )
    return pointer_path.parents[3]


def _tree_state(
    root: Path,
) -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[Path, str, tuple[int, ...], bool], ...],
]:
    governed_root = _absolute_path(root, label="table root")
    directories: list[tuple[str, tuple[int, ...]]] = []
    files: list[tuple[str, tuple[int, ...]]] = []
    regular_files: list[tuple[Path, str, tuple[int, ...], bool]] = []
    pending = [governed_root]
    while pending:
        directory = pending.pop()
        relative_directory = directory.relative_to(governed_root)
        relative_text = (
            "." if not relative_directory.parts else relative_directory.as_posix()
        )
        directories.append(
            (
                relative_text,
                _directory_signature(directory, label="table inventory directory"),
            )
        )
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"table inventory directory unreadable: {directory}"
            ) from exc
        for entry in entries:
            target = directory / entry.name
            relative = target.relative_to(governed_root)
            if any(part in {"", ".", ".."} for part in relative.parts):
                raise FactorGovernanceSourceReadbackV4_1Error(
                    f"unsafe table inventory path: {target}"
                )
            metadata = os.lstat(target)
            if stat.S_ISLNK(metadata.st_mode):
                raise FactorGovernanceSourceReadbackV4_1Error(
                    f"table inventory symlink rejected: {target}"
                )
            if metadata.st_uid != os.getuid():
                raise FactorGovernanceSourceReadbackV4_1Error(
                    f"table inventory owner mismatch: {target}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(target)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise FactorGovernanceSourceReadbackV4_1Error(
                    f"table inventory special file rejected: {target}"
                )
            if int(metadata.st_nlink) < 1:
                raise FactorGovernanceSourceReadbackV4_1Error(
                    f"table inventory invalid hard-link count: {target}"
                )
            relative_file = relative.as_posix()
            signature = _file_signature(metadata)
            files.append((relative_file, signature))
            dataset_member = bool(
                target.suffix == ".parquet"
                and all(not part.startswith((".", "_")) for part in relative.parts)
            )
            regular_files.append((target, relative_file, signature, dataset_member))
    if not any(item[3] for item in regular_files):
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"table Parquet inventory is empty: {governed_root}"
        )
    return (
        tuple(sorted(directories)),
        tuple(sorted(files)),
        tuple(sorted(regular_files, key=lambda item: item[1])),
    )


def _stable_file_digest(
    path: Path,
    *,
    expected_signature: tuple[int, ...],
) -> tuple[str, int, int]:
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if _file_signature(before) != expected_signature:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"table file changed before hashing: {path}"
            )
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"table file is unsafe: {path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _file_signature(before) != _file_signature(opened):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"table file changed while opening: {path}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(after):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"table file changed while hashing: {path}"
            )
        return digest.hexdigest(), int(after.st_size), int(after.st_nlink)
    except FactorGovernanceSourceReadbackV4_1Error:
        raise
    except OSError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"table file hash failed: {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _inventory_table(
    root: Path,
) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    before_directories, before_files, regular_files = _tree_state(root)
    inventory: list[dict[str, Any]] = []
    for path, relative, signature, dataset_member in regular_files:
        digest, size_bytes, hard_link_count = _stable_file_digest(
            path, expected_signature=signature
        )
        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": size_bytes,
                "sha256": digest,
                "hard_link_count": hard_link_count,
                "dataset_member": dataset_member,
            }
        )
    after = _tree_state(root)
    before = (before_directories, before_files, regular_files)
    if before != after:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "table directory or file set changed during inventory"
        )
    return inventory, after


def _extract_calendar_and_symbol_counts(
    table_root: Path,
    *,
    analysis_start: str,
    cutoff_date: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - runtime dependency gate
        raise FactorGovernanceSourceReadbackV4_1Error(
            "strict Parquet calendar extraction requires pyarrow"
        ) from exc
    compact_start = analysis_start.replace("-", "")
    compact_cutoff = cutoff_date.replace("-", "")
    try:
        dataset = ds.dataset(str(table_root), format="parquet", partitioning="hive")
        field = dataset.schema.field("trade_date")
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise FactorGovernanceSourceReadbackV4_1Error(
                "table trade_date must be a string column"
            )
        symbol_field = dataset.schema.field("ts_code")
        if not (
            pa.types.is_string(symbol_field.type)
            or pa.types.is_large_string(symbol_field.type)
        ):
            raise FactorGovernanceSourceReadbackV4_1Error(
                "table ts_code must be a string column"
            )
        table = dataset.to_table(
            columns=["trade_date", "ts_code"],
            filter=(ds.field("trade_date") >= compact_start)
            & (ds.field("trade_date") <= compact_cutoff),
        )
    except FactorGovernanceSourceReadbackV4_1Error:
        raise
    except Exception as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"strict Parquet calendar extraction failed: {exc}"
        ) from exc
    values = table.column("trade_date").to_pylist()
    symbols = table.column("ts_code").to_pylist()
    if len(values) != len(symbols):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "table calendar and symbol columns differ in length"
        )
    sessions: set[str] = set()
    symbol_counts: dict[str, int] = {}
    for raw, symbol in zip(values, symbols, strict=True):
        if type(raw) is not str or len(raw) != 8 or not raw.isdigit():
            raise FactorGovernanceSourceReadbackV4_1Error(
                "table trade_date contains a non-canonical value"
            )
        try:
            parsed = datetime.strptime(raw, "%Y%m%d").date()
        except ValueError as exc:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "table trade_date contains an invalid date"
            ) from exc
        sessions.add(parsed.isoformat())
        if type(symbol) is not str or _CN_SYMBOL.fullmatch(symbol) is None:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "bound table contains a noncanonical CN symbol"
            )
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    ordered = tuple(sorted(sessions))
    if not ordered or ordered[0] != analysis_start or ordered[-1] != cutoff_date:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "analysis start and cutoff must both be exact open sessions"
        )
    return ordered, tuple(sorted(symbol_counts.items()))


def _pit_generation_records(
    *,
    pit_generation_manifest_path: Path,
    expected_pit_generation_manifest_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    reference_root = pit_generation_manifest_path.parent.parent.parent
    try:
        loaded = PITUniverseStore(root_dir=reference_root).load_generation_binding(
            manifest_path=pit_generation_manifest_path,
            expected_manifest_sha256=expected_pit_generation_manifest_sha256,
        )
    except RuntimeError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"authoritative PIT generation binding failed: {exc}"
        ) from exc
    records = tuple(record.to_dict() for record in loaded["records"])
    return records, dict(loaded["manifest"])


def _components(
    payload: Mapping[str, Any],
    *,
    expected_count: int,
    expected_semantic_sha256: str,
) -> tuple[str, ...]:
    raw_symbols = payload.get(EXPECTED_UNIVERSE)
    if not isinstance(raw_symbols, list) or any(
        type(symbol) is not str or _CN_SYMBOL.fullmatch(symbol) is None
        for symbol in raw_symbols
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "components full_a must contain only canonical CN symbols"
        )
    symbols = tuple(raw_symbols)
    if symbols != tuple(sorted(set(symbols))):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "components full_a must be sorted and unique"
        )
    if len(symbols) != expected_count:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "components full_a count mismatch"
        )
    actual_semantic_sha = sha256_bytes("\n".join(symbols).encode("utf-8"))
    if actual_semantic_sha != expected_semantic_sha256:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "components full_a semantic SHA-256 mismatch"
        )
    stats_payload = payload.get("stats")
    if isinstance(stats_payload, Mapping) and stats_payload.get("full_a") != len(
        symbols
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "components stats.full_a count mismatch"
        )
    return symbols


def _coverage(
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_count: int,
    expected_scope_sha256: str,
    cutoff_date: str,
) -> dict[str, Any]:
    pointer_coverage = pointer.get("coverage")
    manifest_coverage = manifest.get("coverage")
    if (
        not isinstance(pointer_coverage, Mapping)
        or not isinstance(manifest_coverage, Mapping)
        or dict(pointer_coverage) != dict(manifest_coverage)
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "pointer and manifest coverage bindings differ"
        )
    coverage = dict(manifest_coverage)
    expected_compact_date = cutoff_date.replace("-", "")
    coverage_ratio = coverage.get("coverage_ratio")
    expected_scope_count = coverage.get("expected_scope_count")
    complete_count = coverage.get("coverage_complete_count")
    blocking_count = coverage.get("blocking_incomplete_count")
    true_missing = coverage.get("true_missing_symbols")
    if (
        coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("complete") is not True
        or type(coverage_ratio) is not float
        or coverage_ratio != 1.0
        or coverage.get("categories_checked") != [EXPECTED_UNIVERSE]
        or type(expected_scope_count) is not int
        or expected_scope_count != expected_count
        or type(complete_count) is not int
        or complete_count != expected_count
        or coverage.get("expected_scope_sha256") != expected_scope_sha256
        or coverage.get("coverage_trade_date") != expected_compact_date
        or coverage.get("latest_available_trade_date") != expected_compact_date
        or coverage.get("latest_complete_trade_date") != expected_compact_date
        or type(blocking_count) is not int
        or blocking_count != 0
        or coverage.get("classification_sets_disjoint") is not True
        or not isinstance(true_missing, list)
        or true_missing
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "strict exact full_a coverage is not healthy"
        )
    return coverage


def bind_explicit_cutoff_inputs_v4_1(
    *,
    latest_pointer_path: str | Path,
    expected_latest_pointer_sha256: str,
    snapshot_manifest_path: str | Path,
    expected_snapshot_manifest_sha256: str,
    components_path: str | Path,
    expected_components_sha256: str,
    expected_full_a_semantic_sha256: str,
    pit_generation_manifest_path: str | Path,
    expected_pit_generation_manifest_sha256: str,
    pit_membership_path: str | Path,
    expected_pit_membership_sha256: str,
    table_root: str | Path,
    snapshot_id: str,
    analysis_start: str,
    cutoff_date: str,
    expected_full_a_count: int = EXPECTED_FULL_A_COUNT,
    expected_serving_inventory_count: int = EXPECTED_SERVING_INVENTORY_COUNT,
) -> BoundCutoffInputsV4_1:
    """Bind one exact cutoff snapshot without any latest or serving discovery."""

    if type(expected_full_a_count) is not int or expected_full_a_count <= 0:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "expected_full_a_count must be a positive integer"
        )
    if (
        type(expected_serving_inventory_count) is not int
        or expected_serving_inventory_count <= 0
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "expected_serving_inventory_count must be a positive integer"
        )
    normalized_snapshot_id = _safe_id(snapshot_id, label="snapshot_id")
    start = _iso_date(analysis_start, label="analysis_start")
    cutoff = _iso_date(cutoff_date, label="cutoff_date")
    if start > cutoff:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "analysis_start must not be after cutoff_date"
        )
    expected_scope_sha = _sha256(
        expected_full_a_semantic_sha256,
        label="expected full_a semantic SHA-256",
    )

    pointer_path = _absolute_path(latest_pointer_path, label="latest pointer")
    project_root = _project_root_from_pointer(pointer_path)
    manifest_path = _absolute_path(
        snapshot_manifest_path, label="snapshot manifest"
    )
    component_file = _absolute_path(components_path, label="components")
    pit_manifest_path = _absolute_path(
        pit_generation_manifest_path, label="PIT generation manifest"
    )
    pit_path = _absolute_path(pit_membership_path, label="PIT membership")
    table_path = _absolute_path(table_root, label="table root")
    for parent, label in (
        (pointer_path.parent, "latest pointer directory"),
        (manifest_path.parent, "snapshot manifest directory"),
        (component_file.parent, "components directory"),
        (pit_manifest_path.parent, "PIT generation directory"),
        (pit_path.parent, "PIT membership directory"),
        (table_path, "table root"),
    ):
        _assert_owned_no_symlink_directory_chain(
            parent, boundary=project_root, label=label
        )

    pointer_raw = _stable_file_bytes(
        pointer_path,
        label="latest pointer",
        expected_sha256=expected_latest_pointer_sha256,
        max_bytes=_JSON_MAX_BYTES,
    )
    manifest_raw = _stable_file_bytes(
        manifest_path,
        label="snapshot manifest",
        expected_sha256=expected_snapshot_manifest_sha256,
        max_bytes=_JSON_MAX_BYTES,
    )
    components_raw = _stable_file_bytes(
        component_file,
        label="components",
        expected_sha256=expected_components_sha256,
        max_bytes=_JSON_MAX_BYTES,
    )
    pit_manifest_raw = _stable_file_bytes(
        pit_manifest_path,
        label="PIT generation manifest",
        expected_sha256=expected_pit_generation_manifest_sha256,
        max_bytes=_JSON_MAX_BYTES,
    )
    pit_raw = _stable_file_bytes(
        pit_path,
        label="PIT membership",
        expected_sha256=expected_pit_membership_sha256,
        max_bytes=REQUEST_MAX_BYTES,
    )
    initial_signatures = {
        path: _file_signature(os.lstat(path))
        for path in (
            pointer_path,
            manifest_path,
            component_file,
            pit_manifest_path,
            pit_path,
        )
    }
    pointer = _strict_json_object(pointer_raw, label="latest pointer")
    manifest = _strict_json_object(manifest_raw, label="snapshot manifest")
    components_payload = _strict_json_object(components_raw, label="components")
    pit_manifest = _strict_json_object(
        pit_manifest_raw, label="PIT generation manifest"
    )

    if (
        pointer.get("snapshot_id") != normalized_snapshot_id
        or pointer.get("status") != "OK"
        or pointer.get("blockers") != []
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "strict Parquet pointer is not healthy"
        )
    if (
        manifest.get("snapshot_id") != normalized_snapshot_id
        or manifest.get("market") != "CN"
        or manifest.get("status") != "OK"
        or manifest.get("readback_validated") is not True
        or manifest.get("blockers") != []
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "strict Parquet snapshot manifest is not healthy"
        )
    compact_cutoff = cutoff.replace("-", "")
    for payload, label in ((pointer, "pointer"), (manifest, "manifest")):
        if (
            payload.get("latest_available_trade_date") != compact_cutoff
            or payload.get("latest_complete_trade_date") != compact_cutoff
        ):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} cutoff date binding mismatch"
            )

    anchors = (project_root, pointer_path.parent)
    _declared_path_matches(
        pointer.get("manifest_path"),
        expected=manifest_path,
        anchors=anchors,
        label="snapshot manifest",
    )
    _declared_path_matches(
        manifest.get("manifest_path"),
        expected=manifest_path,
        anchors=anchors,
        label="snapshot manifest self-binding",
    )
    expected_table_path = pointer_path.parent / "_snapshots" / normalized_snapshot_id / "table" / "bars"
    if table_path != expected_table_path:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "explicit table root is not the snapshot immutable table root"
        )
    for payload, label in ((pointer, "pointer table root"), (manifest, "manifest table root")):
        _declared_path_matches(
            payload.get("table_root"),
            expected=table_path,
            anchors=anchors,
            label=label,
        )
    expected_serving_root = (
        pointer_path.parent
        / "_snapshots"
        / normalized_snapshot_id
        / "serving"
        / "bars"
    )
    for payload, label in (
        (pointer, "pointer serving root"),
        (manifest, "manifest serving root"),
    ):
        _declared_path_matches(
            payload.get("derived_serving_root"),
            expected=expected_serving_root,
            anchors=anchors,
            label=label,
        )

    coverage = _coverage(
        pointer,
        manifest,
        expected_count=expected_full_a_count,
        expected_scope_sha256=expected_scope_sha,
        cutoff_date=cutoff,
    )
    symbols = _components(
        components_payload,
        expected_count=expected_full_a_count,
        expected_semantic_sha256=expected_scope_sha,
    )

    pit_generation_id = coverage.get("pit_generation_id")
    if type(pit_generation_id) is not str:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PIT generation id must be an exact string"
        )
    normalized_generation_id = _safe_id(
        pit_generation_id, label="PIT generation id"
    )
    if pit_manifest_path.parent.name != normalized_generation_id:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PIT generation manifest directory mismatch"
        )
    _declared_path_matches(
        coverage.get("pit_generation_manifest_path"),
        expected=pit_manifest_path,
        anchors=(project_root, manifest_path.parent),
        label="coverage PIT generation manifest",
    )
    _declared_path_matches(
        coverage.get("pit_membership_path"),
        expected=pit_path,
        anchors=(project_root, manifest_path.parent),
        label="coverage PIT membership",
    )
    if (
        coverage.get("pit_generation_manifest_sha256")
        != sha256_bytes(pit_manifest_raw)
        or coverage.get("pit_membership_sha256") != sha256_bytes(pit_raw)
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "coverage PIT SHA-256 binding mismatch"
        )
    _declared_path_matches(
        pit_manifest.get("canonical_path"),
        expected=pit_path,
        anchors=(pit_manifest_path.parent,),
        label="PIT canonical membership",
    )
    pit_row_count = pit_manifest.get("row_count")
    if (
        pit_manifest.get("schema_version") != "cn_pit_universe_manifest.v1"
        or pit_manifest.get("membership_schema_version") != "cn_pit_universe.v1"
        or pit_manifest.get("generation_id") != normalized_generation_id
        or pit_manifest.get("canonical_sha256") != sha256_bytes(pit_raw)
        or pit_row_count != pit_manifest.get("raw_row_count")
        or type(pit_row_count) is not int
        or pit_row_count <= 0
        or pit_manifest.get("membership_quality_counts")
        != {"ok": pit_row_count}
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PIT generation manifest is not authoritative and healthy"
        )

    serving_count = manifest.get("symbol_count")
    if (
        type(serving_count) is not int
        or serving_count != expected_serving_inventory_count
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "serving inventory count diagnostic mismatch"
        )
    table_inventory, stable_tree_state = _inventory_table(table_path)
    calendar, bound_table_symbol_row_counts = _extract_calendar_and_symbol_counts(
        table_path, analysis_start=start, cutoff_date=cutoff
    )
    if _tree_state(table_path) != stable_tree_state:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "table directory or file set changed during calendar readback"
        )
    records, loaded_pit_manifest = _pit_generation_records(
        pit_generation_manifest_path=pit_manifest_path,
        expected_pit_generation_manifest_sha256=sha256_bytes(pit_manifest_raw),
    )
    if loaded_pit_manifest != pit_manifest:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PIT generation manifest normalized readback mismatch"
        )
    if len(records) != pit_row_count:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PIT Parquet row count does not match its manifest"
        )
    status_counts: dict[str, int] = {}
    for record in records:
        status = record.get("source_list_status")
        if type(status) is not str:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "PIT record status must be an exact string"
            )
        status_counts[status] = status_counts.get(status, 0) + 1
    if dict(sorted(status_counts.items())) != pit_manifest.get("status_counts"):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PIT status_counts do not match the authoritative records"
        )

    final_inputs = (
        (
            pointer_path,
            "latest pointer",
            sha256_bytes(pointer_raw),
            _JSON_MAX_BYTES,
        ),
        (
            manifest_path,
            "snapshot manifest",
            sha256_bytes(manifest_raw),
            _JSON_MAX_BYTES,
        ),
        (
            component_file,
            "components",
            sha256_bytes(components_raw),
            _JSON_MAX_BYTES,
        ),
        (
            pit_manifest_path,
            "PIT generation manifest",
            sha256_bytes(pit_manifest_raw),
            _JSON_MAX_BYTES,
        ),
        (
            pit_path,
            "PIT membership",
            sha256_bytes(pit_raw),
            REQUEST_MAX_BYTES,
        ),
    )
    for path, label, digest, max_bytes in final_inputs:
        if _file_signature(os.lstat(path)) != initial_signatures[path]:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} changed across cutoff readback"
            )
        final_raw = _stable_file_bytes(
            path,
            label=label,
            expected_sha256=digest,
            max_bytes=max_bytes,
        )
        if _file_signature(os.lstat(path)) != initial_signatures[path]:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} changed across cutoff readback"
            )
        if final_raw != {
            pointer_path: pointer_raw,
            manifest_path: manifest_raw,
            component_file: components_raw,
            pit_manifest_path: pit_manifest_raw,
            pit_path: pit_raw,
        }[path]:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"{label} byte drift across cutoff readback"
            )

    table_inventory_sha = sha256_bytes(canonical_json_bytes(table_inventory))
    calendar_payload = {
        "analysis_start": start,
        "cutoff_date": cutoff,
        "open_session_count": len(calendar),
        "open_sessions": list(calendar),
    }
    calendar_payload["semantic_sha256"] = sha256_bytes(
        canonical_json_bytes(calendar_payload)
    )
    bound_table_symbols = [symbol for symbol, _count in bound_table_symbol_row_counts]
    table_count_by_symbol = dict(bound_table_symbol_row_counts)
    historical_alias_table_evidence = [
        {
            "symbol": str(record.get("symbol")),
            "table_row_count": table_count_by_symbol.get(
                str(record.get("symbol")), 0
            ),
        }
        for record in records
        if type(record.get("symbol")) is str
        and _CN_SYMBOL.fullmatch(str(record.get("symbol"))) is None
    ]
    if any(row["table_row_count"] != 0 for row in historical_alias_table_evidence):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "historical PIT alias appears in the bound table"
        )
    binding = {
        "schema_version": INPUT_BINDING_SCHEMA_VERSION,
        "market": "CN",
        "snapshot_id": normalized_snapshot_id,
        "cutoff_date": cutoff,
        "latest_pointer": _binding_record(pointer_path, pointer_raw),
        "snapshot_manifest": _binding_record(manifest_path, manifest_raw),
        "components": {
            **_binding_record(component_file, components_raw),
            "universe": EXPECTED_UNIVERSE,
            "count": len(symbols),
            "newline_set_sha256": expected_scope_sha,
        },
        "pit_generation": {
            "generation_id": normalized_generation_id,
            "manifest": _binding_record(pit_manifest_path, pit_manifest_raw),
            "membership": _binding_record(pit_path, pit_raw),
            "row_count": len(records),
            "historical_alias_table_evidence": historical_alias_table_evidence,
        },
        "table": {
            "absolute_root": str(table_path),
            "regular_file_count": len(table_inventory),
            "parquet_file_count": sum(
                1 for item in table_inventory if item["dataset_member"]
            ),
            "inventory_sha256": table_inventory_sha,
            "parquet_inventory": table_inventory,
            "bound_symbol_inventory": {
                "symbol_count": len(bound_table_symbols),
                "symbols_newline_sha256": sha256_bytes(
                    "\n".join(bound_table_symbols).encode("ascii")
                ),
                "noncanonical_symbol_count": 0,
            },
        },
        "calendar": calendar_payload,
        "eligibility_boundary": {
            "component_source": str(component_file),
            "pit_source": str(pit_path),
            "bar_source": str(table_path),
            "serving_inventory": {
                "absolute_root": str(expected_serving_root),
                "symbol_count": serving_count,
                "use": SOURCE_USE_PROHIBITED,
                "was_scanned": False,
            },
        },
        "readiness": "EXPLORATORY_INPUT_BOUND",
        "side_effects": {
            "registry": False,
            "wal": False,
            "budget": False,
            "apply": False,
            "broker": False,
            "order": False,
            "trade": False,
            "network": False,
        },
    }
    return BoundCutoffInputsV4_1(
        binding=binding,
        calendar_sessions=calendar,
        component_symbols=symbols,
        pit_records=records,
        bound_table_symbol_row_counts=bound_table_symbol_row_counts,
    )


def binding_semantic_sha256_v4_1(binding: Mapping[str, Any]) -> str:
    if binding.get("schema_version") != INPUT_BINDING_SCHEMA_VERSION:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "input binding schema mismatch"
        )
    return sha256_bytes(canonical_json_bytes(dict(binding)))


def source_code_binding_sha256_v4_1(paths: Sequence[str | Path]) -> str:
    """Hash an explicit, stable list of source files without code discovery."""

    if not paths:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "source code binding paths must not be empty"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_path in enumerate(paths):
        path = _absolute_path(raw_path, label=f"source code path {index}")
        key = str(path)
        if key in seen:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "source code binding paths must be distinct"
            )
        seen.add(key)
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"source code file is unsafe: {path}"
            )
        signature = _file_signature(metadata)
        digest, size_bytes, hard_link_count = _stable_file_digest(
            path, expected_signature=signature
        )
        if hard_link_count != 1:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"source code hard-link count must be one: {path}"
            )
        if _file_signature(os.lstat(path)) != signature:
            raise FactorGovernanceSourceReadbackV4_1Error(
                f"source code changed while binding: {path}"
            )
        records.append(
            {"absolute_path": key, "size_bytes": size_bytes, "sha256": digest}
        )
    return sha256_bytes(canonical_json_bytes(records))


def build_cutoff_source_node_v4_1(
    *,
    cycle_id: str,
    input_binding: Mapping[str, Any],
    design_source: Mapping[str, Any],
    source_binding_sha256: str,
) -> dict[str, Any]:
    """Build the explicit cutoff chain node bound to the design source root."""

    cycle = _safe_id(cycle_id, label="cycle_id")
    binding = dict(input_binding)
    binding_sha = binding_semantic_sha256_v4_1(binding)
    source_sha = _sha256(
        source_binding_sha256, label="source binding SHA-256"
    )
    design = dict(design_source)
    if design.get("cycle_id") != cycle:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "design source cycle_id mismatch"
        )
    descriptors = design.get("session_scope_descriptors")
    if not isinstance(descriptors, list) or not descriptors:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "design source has no cutoff session descriptor"
        )
    cutoff_descriptor = descriptors[-1]
    if not isinstance(cutoff_descriptor, Mapping):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "design cutoff session descriptor is invalid"
        )
    cutoff_descriptor_sha = cutoff_descriptor.get("session_semantic_sha256")
    if type(cutoff_descriptor_sha) is not str:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "cutoff descriptor SHA-256 is missing"
        )
    _sha256(cutoff_descriptor_sha, label="cutoff descriptor SHA-256")
    design_source_root_sha = design.get("semantic_sha256")
    if type(design_source_root_sha) is not str:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "design source root SHA-256 is missing"
        )
    _sha256(design_source_root_sha, label="design source root SHA-256")
    alias_report = design.get("out_of_bound_calendar_nonparticipating")
    pit_binding = binding.get("pit_generation")
    if not isinstance(alias_report, Mapping) or not isinstance(
        pit_binding, Mapping
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "historical alias evidence is missing"
        )
    alias_records = alias_report.get("records")
    table_evidence = pit_binding.get("historical_alias_table_evidence")
    if not isinstance(alias_records, list) or not isinstance(table_evidence, list):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "historical alias evidence is invalid"
        )
    table_counts: dict[str, int] = {}
    for row in table_evidence:
        if not isinstance(row, Mapping):
            raise FactorGovernanceSourceReadbackV4_1Error(
                "historical alias table evidence is invalid"
            )
        symbol = row.get("symbol")
        count = row.get("table_row_count")
        if type(symbol) is not str or type(count) is not int or count != 0:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "historical alias must have zero bound table rows"
            )
        if symbol in table_counts:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "duplicate historical alias table evidence"
            )
        table_counts[symbol] = count
    detailed_aliases: list[dict[str, Any]] = []
    for row in alias_records:
        if not isinstance(row, Mapping) or type(row.get("symbol")) is not str:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "historical alias source evidence is invalid"
            )
        symbol = str(row["symbol"])
        if symbol not in table_counts or row.get("active_bound_session_count") != 0:
            raise FactorGovernanceSourceReadbackV4_1Error(
                "historical alias overlaps the bound source domain"
            )
        detailed_aliases.append({**dict(row), "table_row_count": table_counts[symbol]})
    if sorted(table_counts) != sorted(row["symbol"] for row in detailed_aliases):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "historical alias source/table evidence differs"
        )
    base = {
        "schema_version": CUTOFF_SOURCE_NODE_SCHEMA_VERSION,
        "protocol_version": "v4",
        "cycle_id": cycle,
        "snapshot_id": design.get("snapshot_id"),
        "cutoff_date": design.get("cutoff_date"),
        "input_binding_semantic_sha256": binding_sha,
        "design_source_root_sha256": design_source_root_sha,
        "cutoff_session_scope_semantic_sha256": cutoff_descriptor_sha,
        "source_binding_sha256": source_sha,
        "out_of_bound_calendar_nonparticipating": {
            "records": detailed_aliases,
            "count": len(detailed_aliases),
            "records_semantic_sha256": sha256_bytes(
                canonical_json_bytes(detailed_aliases)
            ),
        },
        "serving_inventory_eligibility_prohibited": True,
    }
    return {**base, "semantic_sha256": sha256_bytes(canonical_json_bytes(base))}


def validate_cutoff_source_node_v4_1(
    value: Mapping[str, Any],
    *,
    cycle_id: str,
    input_binding: Mapping[str, Any],
    design_source: Mapping[str, Any],
    source_binding_sha256: str,
) -> dict[str, Any]:
    expected = build_cutoff_source_node_v4_1(
        cycle_id=cycle_id,
        input_binding=input_binding,
        design_source=design_source,
        source_binding_sha256=source_binding_sha256,
    )
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(expected):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "cutoff source node does not match its bound inputs"
        )
    return expected


def cycle_root_semantic_sha256_v4_1(
    *,
    cycle_id: str,
    input_binding: Mapping[str, Any],
    design_source: Mapping[str, Any],
) -> str:
    cycle = _safe_id(cycle_id, label="cycle_id")
    design_source_root_sha = design_source.get("semantic_sha256")
    if type(design_source_root_sha) is not str:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "design source root SHA-256 is missing"
        )
    _sha256(design_source_root_sha, label="design source root SHA-256")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": CYCLE_ROOT_SCHEMA_VERSION,
                "cycle_id": cycle,
                "input_binding_semantic_sha256": (
                    binding_semantic_sha256_v4_1(input_binding)
                ),
                "design_source_root_sha256": design_source_root_sha,
            }
        )
    )


def _validate_precommitted_bundle(
    *,
    cycle_id: str,
    input_binding: Mapping[str, Any],
    design_source: Mapping[str, Any],
    source_chain_node: Mapping[str, Any],
    precommitted_cycle_state: Mapping[str, Any],
    pit_records: Sequence[Mapping[str, Any]],
    expected_component_count: int,
    expected_source_binding_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    cycle = _safe_id(cycle_id, label="cycle_id")
    binding = dict(input_binding)
    binding_sha = binding_semantic_sha256_v4_1(binding)
    source_binding_sha = _sha256(
        expected_source_binding_sha256,
        label="expected source binding SHA-256",
    )
    try:
        design = validate_design_source_node_v4_1(
            design_source,
            pit_records=list(pit_records),
            expected_component_count=expected_component_count,
        )
    except ValueError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"design source validation failed: {exc}"
        ) from exc
    component_binding = binding.get("components")
    pit_binding = binding.get("pit_generation")
    if not isinstance(component_binding, Mapping) or not isinstance(
        pit_binding, Mapping
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "input binding components or PIT binding missing"
        )
    if (
        design["cycle_id"] != cycle
        or design["snapshot_id"] != binding.get("snapshot_id")
        or design["cutoff_date"] != binding.get("cutoff_date")
        or design["historical_table_binding_sha256"] != binding_sha
        or design["historical_source_binding_sha256"] != source_binding_sha
        or design["component_count"]
        != component_binding.get("count")
        or design["component_symbols_semantic_sha256"]
        != component_binding.get("newline_set_sha256")
        or design["pit_record_count"]
        != pit_binding.get("row_count")
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "design source cross-artifact binding mismatch"
        )
    node = validate_cutoff_source_node_v4_1(
        source_chain_node,
        cycle_id=cycle,
        input_binding=binding,
        design_source=design,
        source_binding_sha256=source_binding_sha,
    )
    cycle_root_sha = cycle_root_semantic_sha256_v4_1(
        cycle_id=cycle,
        input_binding=binding,
        design_source=design,
    )
    try:
        state = validate_genesis_cycle_state_v4_1(
            precommitted_cycle_state,
            expected_cycle_id=cycle,
            expected_cycle_root_sha256=cycle_root_sha,
        )
    except ValueError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"PRECOMMITTED cycle-state validation failed: {exc}"
        ) from exc
    if state["source_chain_node_sha256"] != node["semantic_sha256"]:
        raise FactorGovernanceSourceReadbackV4_1Error(
            "PRECOMMITTED state source-chain binding mismatch"
        )
    return design, node, state, cycle_root_sha


def _private_file_readback(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or int(metadata.st_nlink) != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"private artifact owner/link/mode readback failed: {path}"
        )
    raw = _stable_file_bytes(
        path,
        label="private artifact",
        expected_sha256=expected_sha256,
        max_bytes=CONTROL_MAX_BYTES,
    )
    value = _strict_json_object(raw, label="private artifact")
    if raw != canonical_json_bytes(value):
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"private artifact is not canonical JSON: {path}"
        )
    return value


def _write_private_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(dict(value))
    descriptor: int | None = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(descriptor)
    except FileExistsError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"exact-once output already exists: {path}"
        ) from exc
    except OSError as exc:
        raise FactorGovernanceSourceReadbackV4_1Error(
            f"exact-once output write failed: {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    digest = sha256_bytes(raw)
    _private_file_readback(path, expected_sha256=digest)
    return digest


def _artifact_descriptor(path: Path, sha256: str) -> dict[str, Any]:
    return {"absolute_path": str(path), "sha256": sha256}


def publish_blocked_cutoff_readback_v4_1(
    *,
    private_root: str | Path,
    run_id: str,
    cycle_id: str,
    input_binding: Mapping[str, Any],
    blocker_code: str,
    blocker_detail: str,
    expected_state_sha256: str = "empty",
) -> dict[str, Any]:
    """Publish one immutable blocker; never create a source root or cycle state."""

    root = _absolute_path(private_root, label="private root")
    normalized_run_id = _safe_id(run_id, label="run_id")
    normalized_cycle_id = _safe_id(cycle_id, label="cycle_id")
    expected = _sha256(
        expected_state_sha256, label="expected state SHA-256", allow_empty=True
    )
    if expected != "empty":
        raise FactorGovernanceSourceReadbackV4_1Error(
            "cutoff blocker publication requires EMPTY state CAS"
        )
    if (
        type(blocker_code) is not str
        or not blocker_code
        or blocker_code != blocker_code.strip()
        or type(blocker_detail) is not str
        or not blocker_detail
        or blocker_detail != blocker_detail.strip()
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "blocker code and detail must be exact non-empty strings"
        )
    binding = dict(input_binding)
    binding_sha = binding_semantic_sha256_v4_1(binding)
    with run_lock(root, normalized_run_id) as (_root, run_dir):
        state_path = run_dir / PRECOMMITTED_STATE_FILENAME
        try:
            assert_cas(state_path, expected)
        except ProtocolError as exc:
            raise FactorGovernanceSourceReadbackV4_1Error(str(exc)) from exc
        report_path = run_dir / BLOCKER_REPORT_FILENAME
        report = {
            "schema_version": BLOCKER_REPORT_SCHEMA_VERSION,
            "protocol_version": "v4",
            "cycle_id": normalized_cycle_id,
            "run_id": normalized_run_id,
            "readiness": "BLOCKED_FAIL_CLOSED",
            "blockers": [
                {"code": blocker_code, "detail": blocker_detail}
            ],
            "input_binding": binding,
            "input_binding_complete": True,
            "input_binding_semantic_sha256": binding_sha,
            "created_artifacts": {
                "design_source": False,
                "source_chain_node": False,
                "precommitted_cycle_state": False,
            },
            "side_effects": {
                "registry": False,
                "wal": False,
                "budget": False,
                "apply": False,
                "broker": False,
                "order": False,
                "trade": False,
                "network": False,
            },
        }
        report_sha = _write_private_exclusive(report_path, report)
        return {
            "readiness": "BLOCKED_FAIL_CLOSED",
            "blocker_report": _artifact_descriptor(report_path, report_sha),
            "design_source": None,
            "source_chain_node": None,
            "precommitted_cycle_state": None,
        }


def publish_input_binding_failure_v4_1(
    *,
    private_root: str | Path,
    run_id: str,
    cycle_id: str,
    attempted_inputs: Mapping[str, Any],
    blocker_code: str,
    blocker_detail: str,
    expected_state_sha256: str = "empty",
) -> dict[str, Any]:
    """Publish bounded failure evidence when the input boundary itself rejects."""

    root = _absolute_path(private_root, label="private root")
    normalized_run_id = _safe_id(run_id, label="run_id")
    normalized_cycle_id = _safe_id(cycle_id, label="cycle_id")
    expected = _sha256(
        expected_state_sha256, label="expected state SHA-256", allow_empty=True
    )
    if expected != "empty":
        raise FactorGovernanceSourceReadbackV4_1Error(
            "input-binding failure publication requires EMPTY state CAS"
        )
    if (
        type(blocker_code) is not str
        or not blocker_code
        or blocker_code != blocker_code.strip()
        or type(blocker_detail) is not str
        or not blocker_detail
        or blocker_detail != blocker_detail.strip()
    ):
        raise FactorGovernanceSourceReadbackV4_1Error(
            "blocker code and detail must be exact non-empty strings"
        )
    attempted = dict(attempted_inputs)
    # Refuse non-JSON values before creating any output.
    canonical_json_bytes(attempted)
    with run_lock(root, normalized_run_id) as (_root, run_dir):
        state_path = run_dir / PRECOMMITTED_STATE_FILENAME
        try:
            assert_cas(state_path, expected)
        except ProtocolError as exc:
            raise FactorGovernanceSourceReadbackV4_1Error(str(exc)) from exc
        report_path = run_dir / BLOCKER_REPORT_FILENAME
        report = {
            "schema_version": BLOCKER_REPORT_SCHEMA_VERSION,
            "protocol_version": "v4",
            "cycle_id": normalized_cycle_id,
            "run_id": normalized_run_id,
            "readiness": "BLOCKED_FAIL_CLOSED",
            "blockers": [{"code": blocker_code, "detail": blocker_detail}],
            "input_binding_complete": False,
            "attempted_inputs": attempted,
            "created_artifacts": {
                "design_source": False,
                "source_chain_node": False,
                "precommitted_cycle_state": False,
            },
            "side_effects": {
                "registry": False,
                "wal": False,
                "budget": False,
                "apply": False,
                "broker": False,
                "order": False,
                "trade": False,
                "network": False,
            },
        }
        report_sha = _write_private_exclusive(report_path, report)
        return {
            "readiness": "BLOCKED_FAIL_CLOSED",
            "blocker_report": _artifact_descriptor(report_path, report_sha),
            "design_source": None,
            "source_chain_node": None,
            "precommitted_cycle_state": None,
        }


def publish_precommitted_cutoff_source_v4_1(
    *,
    private_root: str | Path,
    run_id: str,
    cycle_id: str,
    input_binding: Mapping[str, Any],
    design_source: Mapping[str, Any],
    source_chain_node: Mapping[str, Any],
    precommitted_cycle_state: Mapping[str, Any],
    pit_records: Sequence[Mapping[str, Any]],
    expected_component_count: int,
    expected_source_binding_sha256: str,
    expected_state_sha256: str = "empty",
) -> dict[str, Any]:
    """Publish a proven exploratory genesis bundle under EMPTY state CAS."""

    root = _absolute_path(private_root, label="private root")
    normalized_run_id = _safe_id(run_id, label="run_id")
    normalized_cycle_id = _safe_id(cycle_id, label="cycle_id")
    expected = _sha256(
        expected_state_sha256, label="expected state SHA-256", allow_empty=True
    )
    if expected != "empty":
        raise FactorGovernanceSourceReadbackV4_1Error(
            "genesis publication requires EMPTY state CAS"
        )
    binding = dict(input_binding)
    design, node, state, cycle_root_sha = _validate_precommitted_bundle(
        cycle_id=normalized_cycle_id,
        input_binding=binding,
        design_source=design_source,
        source_chain_node=source_chain_node,
        precommitted_cycle_state=precommitted_cycle_state,
        pit_records=pit_records,
        expected_component_count=expected_component_count,
        expected_source_binding_sha256=expected_source_binding_sha256,
    )
    values = {
        INPUT_BINDING_FILENAME: binding,
        DESIGN_SOURCE_FILENAME: design,
        SOURCE_CHAIN_NODE_FILENAME: node,
        PRECOMMITTED_STATE_FILENAME: state,
    }
    with run_lock(root, normalized_run_id) as (_root, run_dir):
        state_path = run_dir / PRECOMMITTED_STATE_FILENAME
        try:
            assert_cas(state_path, expected)
        except ProtocolError as exc:
            raise FactorGovernanceSourceReadbackV4_1Error(str(exc)) from exc
        descriptors: dict[str, dict[str, Any]] = {}
        for filename, value in values.items():
            target = run_dir / filename
            digest = _write_private_exclusive(target, value)
            descriptors[filename] = _artifact_descriptor(target, digest)
        state_sha = descriptors[PRECOMMITTED_STATE_FILENAME]["sha256"]
        report = {
            "schema_version": READBACK_REPORT_SCHEMA_VERSION,
            "protocol_version": "v4",
            "cycle_id": normalized_cycle_id,
            "run_id": normalized_run_id,
            "readiness": "EXPLORATORY_PRECOMMITTED",
            "qualification": False,
            "artifacts": descriptors,
            "cycle_root_semantic_sha256": cycle_root_sha,
            "state_cas": {"before": "EMPTY", "after": state_sha},
            "side_effects": {
                "registry": False,
                "wal": False,
                "budget": False,
                "apply": False,
                "broker": False,
                "order": False,
                "trade": False,
                "network": False,
            },
        }
        report_path = run_dir / READBACK_REPORT_FILENAME
        report_sha = _write_private_exclusive(report_path, report)
        return {
            "readiness": "EXPLORATORY_PRECOMMITTED",
            "qualification": False,
            "artifacts": descriptors,
            "readback_report": _artifact_descriptor(report_path, report_sha),
        }


__all__ = [
    "BLOCKER_REPORT_FILENAME",
    "BLOCKER_REPORT_SCHEMA_VERSION",
    "BoundCutoffInputsV4_1",
    "CUTOFF_SOURCE_NODE_SCHEMA_VERSION",
    "CYCLE_ROOT_SCHEMA_VERSION",
    "DESIGN_SOURCE_FILENAME",
    "EXPECTED_FULL_A_COUNT",
    "EXPECTED_SERVING_INVENTORY_COUNT",
    "FactorGovernanceSourceReadbackV4_1Error",
    "INPUT_BINDING_FILENAME",
    "INPUT_BINDING_SCHEMA_VERSION",
    "PRECOMMITTED_STATE_FILENAME",
    "READBACK_REPORT_FILENAME",
    "READBACK_REPORT_SCHEMA_VERSION",
    "SOURCE_CHAIN_NODE_FILENAME",
    "SOURCE_USE_PROHIBITED",
    "bind_explicit_cutoff_inputs_v4_1",
    "binding_semantic_sha256_v4_1",
    "build_cutoff_source_node_v4_1",
    "cycle_root_semantic_sha256_v4_1",
    "publish_blocked_cutoff_readback_v4_1",
    "publish_input_binding_failure_v4_1",
    "publish_precommitted_cutoff_source_v4_1",
    "source_code_binding_sha256_v4_1",
    "validate_cutoff_source_node_v4_1",
]
