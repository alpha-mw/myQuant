#!/usr/bin/env python3
"""Build one explicit-path, research-only Factor v4.1 signal diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import (  # noqa: E402
    governance_aquant_no_label_eval_v4_1 as evaluator,
)
from quant_investor.factors import (  # noqa: E402
    governance_no_label_diagnostic_v4_1 as diagnostic,
)
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_source_readback_v4_1 as source_readback  # noqa: E402
from quant_investor.factors import governance_source_v4_1 as source  # noqa: E402


DISCOVERY_FILENAMES = (
    "aquant_source_receipt.v4_1.json",
    "source_idea_audit.v4_1.json",
    "local_compatibility_contract.v4_1.json",
    "discovery_catalog.v4_1.json",
    "structural_collision_audit.v4_1.json",
    "discovery_source_node.v4_1.json",
    "cycle_state.discovery.v4_1.json",
    "discovery_readback_report.v4_1.json",
)
DISCOVERY_REPORT = DISCOVERY_FILENAMES[-1]
FORMAL_FILENAMES = (
    "primitive_mapping_policy.v4_1.json",
    "primitive_mapping_proof.v4_1.json",
    "primitive_ontology.v4.json",
    "candidate_catalog.v4.json",
    "formal_catalog_materialization_manifest.v4_1.json",
    "formal_catalog_adapter_validation.v4_1.json",
    "formal_catalog_materialization_readback.v4_1.json",
)
FORMAL_REPORT = FORMAL_FILENAMES[-1]
CUTOFF_FILENAMES = (
    "cutoff_input_binding.v4_1.json",
    "design_source.v4_1.json",
    "source_chain_node.v4_1.json",
    "cycle_state.precommitted.v4_1.json",
    "source_readback_report.v4_1.json",
)
CUTOFF_REPORT = CUTOFF_FILENAMES[-1]
MARKET_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)
EXPECTED_SESSION_COUNT = 1227
EXPECTED_PIT_COUNT = 5866
EXPECTED_COMPONENT_COUNT = 5502
EXPECTED_ANALYSIS_START = "2021-06-25"
EXPECTED_CUTOFF = "2026-07-17"


class FactorV4_1SignalDiagnosticRunnerError(ValueError):
    """Raised when an explicit runner binding fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"{context} is not a SHA-256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"{context} is not a SHA-256"
        ) from exc
    return value


def _absolute_path(value: Any, context: str) -> Path:
    if type(value) is not str or not value:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"{context} must be an absolute path"
        )
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"{context} must be absolute and normalized"
        )
    return path


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stable_bytes(
    path: Path,
    *,
    expected_sha256: str,
    private: bool,
) -> bytes:
    expected = _sha(expected_sha256, f"expected SHA for {path}")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"bound path is not a regular non-symlink file: {path}"
        )
    if private and (
        before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or int(before.st_nlink) != 1
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private artifact owner/mode/link check failed: {path}"
        )
    first = path.read_bytes()
    middle = os.lstat(path)
    second = path.read_bytes()
    after = os.lstat(path)
    if not _signature(before) == _signature(middle) == _signature(after) or first != second:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"bound file changed across stable readback: {path}"
        )
    if hashlib.sha256(first).hexdigest() != expected:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"bound file SHA mismatch: {path}"
        )
    return first


def _strict_json(raw: bytes, context: str, *, canonical_file: bool) -> dict[str, Any]:
    duplicate = object()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in items:
            if key in payload:
                raise FactorV4_1SignalDiagnosticRunnerError(
                    f"duplicate JSON key in {context}: {key}"
                )
            payload[key] = value
        return payload

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"bound JSON parse failed: {context}"
        ) from exc
    if value is duplicate or not isinstance(value, dict):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"bound JSON must be an object: {context}"
        )
    if canonical_file and raw != _canonical_json_bytes(value) + b"\n":
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private JSON is not canonical: {context}"
        )
    return value


def _report_semantic(value: Mapping[str, Any], field: str | None) -> str:
    if field is None:
        return _semantic_sha(value)
    stored = _sha(value.get(field), f"report {field}")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if _semantic_sha(payload) != stored:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"report self-hash mismatch: {field}"
        )
    return stored


def _read_bundle(
    *,
    bundle_path: Path,
    filenames: Sequence[str],
    report_filename: str,
    expected_report_sha256: str,
    expected_report_semantic_sha256: str,
    report_semantic_field: str | None,
    allow_lock: bool = False,
) -> dict[str, Any]:
    directory = os.lstat(bundle_path)
    if (
        stat.S_ISLNK(directory.st_mode)
        or not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.getuid()
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private bundle directory check failed: {bundle_path}"
        )
    expected_names = set(filenames) | ({".lock"} if allow_lock else set())
    if {item.name for item in bundle_path.iterdir()} != expected_names:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private bundle exact inventory mismatch: {bundle_path}"
        )
    directory_signature = _signature(directory)
    lock_descriptor: dict[str, Any] | None = None
    if allow_lock:
        lock_path = bundle_path / ".lock"
        lock_metadata = os.lstat(lock_path)
        if (
            stat.S_ISLNK(lock_metadata.st_mode)
            or not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.getuid()
            or stat.S_IMODE(lock_metadata.st_mode) != 0o600
            or int(lock_metadata.st_nlink) != 1
            or int(lock_metadata.st_size) != 0
        ):
            raise FactorV4_1SignalDiagnosticRunnerError(
                "cutoff bundle lock identity/mode/size mismatch"
            )
        lock_raw = _stable_bytes(
            lock_path,
            expected_sha256=hashlib.sha256(b"").hexdigest(),
            private=True,
        )
        lock_descriptor = {
            "absolute_path": str(lock_path),
            "byte_sha256": hashlib.sha256(lock_raw).hexdigest(),
            "size_bytes": 0,
        }
    report_path = bundle_path / report_filename
    report_raw = _stable_bytes(
        report_path,
        expected_sha256=expected_report_sha256,
        private=True,
    )
    report = _strict_json(report_raw, str(report_path), canonical_file=True)
    semantic = _report_semantic(report, report_semantic_field)
    if semantic != _sha(
        expected_report_semantic_sha256, "expected report semantic SHA"
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private bundle report semantic SHA mismatch: {bundle_path}"
        )
    bound_shas: dict[str, str] = {}
    bindings = report.get("artifact_bindings")
    if isinstance(bindings, list):
        for row in bindings:
            filename = row.get("filename")
            if type(filename) is not str or filename in bound_shas:
                raise FactorV4_1SignalDiagnosticRunnerError(
                    f"private bundle report binding mismatch: {bundle_path}"
                )
            bound_shas[filename] = _sha(
                row.get("byte_sha256"), "bundle artifact byte SHA"
            )
    artifacts = report.get("artifacts")
    if isinstance(artifacts, Mapping):
        for filename, row in artifacts.items():
            if type(filename) is not str or filename in bound_shas:
                raise FactorV4_1SignalDiagnosticRunnerError(
                    f"cutoff report artifact binding mismatch: {filename}"
                )
            bound_shas[filename] = _sha(
                row.get("sha256"), "cutoff artifact byte SHA"
            )
    expected_bound_names = set(filenames) - {report_filename}
    if set(bound_shas) != expected_bound_names:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private bundle report does not bind the exact artifact set: {bundle_path}"
        )
    values: dict[str, dict[str, Any]] = {report_filename: report}
    descriptors: dict[str, dict[str, Any]] = {
        report_filename: {
            "absolute_path": str(report_path),
            "byte_sha256": hashlib.sha256(report_raw).hexdigest(),
            "size_bytes": len(report_raw),
        }
    }
    for filename in filenames:
        if filename == report_filename:
            continue
        path = bundle_path / filename
        raw = _stable_bytes(
            path,
            expected_sha256=bound_shas[filename],
            private=True,
        )
        values[filename] = _strict_json(raw, str(path), canonical_file=True)
        descriptors[filename] = {
            "absolute_path": str(path),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    if (
        _signature(os.lstat(bundle_path)) != directory_signature
        or {item.name for item in bundle_path.iterdir()} != expected_names
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"private bundle directory changed across stable readback: {bundle_path}"
        )
    return {
        "path": str(bundle_path),
        "values": values,
        "descriptors": descriptors,
        "report_semantic_sha256": semantic,
        "lock_descriptor": lock_descriptor,
    }


def _parse_binding(value: str, context: str) -> dict[str, Any]:
    path_text, separator, digest = value.rpartition("=")
    if not separator:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"{context} must use ABSOLUTE_PATH=SHA256"
        )
    path = _absolute_path(path_text, context)
    return {
        "binding_id": path.name,
        "absolute_path": str(path),
        "byte_sha256": _sha(digest, context),
    }


def _parse_bindings(values: Sequence[str], context: str) -> list[dict[str, Any]]:
    rows = [_parse_binding(value, context) for value in values]
    if not rows:
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"at least one {context} is required"
        )
    paths = [row["absolute_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise FactorV4_1SignalDiagnosticRunnerError(
            f"{context} paths must be distinct"
        )
    rows.sort(key=lambda row: row["absolute_path"])
    return rows


def _read_control_bindings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["absolute_path"])
        raw = _stable_bytes(
            path, expected_sha256=row["byte_sha256"], private=False
        )
        result.append({**dict(row), "size_bytes": len(raw)})
    return result


def _inventory_table(table_root: Path) -> tuple[list[dict[str, Any]], str]:
    if table_root.is_symlink() or not table_root.is_dir():
        raise FactorV4_1SignalDiagnosticRunnerError(
            "table root must be a regular directory"
        )
    inventory: list[dict[str, Any]] = []
    for path in sorted(table_root.rglob("*")):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise FactorV4_1SignalDiagnosticRunnerError(
                f"table inventory contains a symlink: {path}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            continue
        relative = path.relative_to(table_root)
        raw = path.read_bytes()
        dataset_member = bool(
            path.suffix == ".parquet"
            and all(not part.startswith((".", "_")) for part in relative.parts)
        )
        inventory.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "hard_link_count": int(metadata.st_nlink),
                "dataset_member": dataset_member,
            }
        )
    if not any(row["dataset_member"] for row in inventory):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "table inventory has no dataset members"
        )
    inventory_sha = hashlib.sha256(_canonical_json_bytes(inventory) + b"\n").hexdigest()
    return inventory, inventory_sha


def _load_components(path: Path, expected_sha256: str) -> list[str]:
    raw = _stable_bytes(path, expected_sha256=expected_sha256, private=False)
    payload = _strict_json(raw, str(path), canonical_file=False)
    symbols = payload.get("full_a")
    if (
        not isinstance(symbols, list)
        or len(symbols) != EXPECTED_COMPONENT_COUNT
        or symbols != sorted(set(symbols))
        or any(type(item) is not str for item in symbols)
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "component file must contain the exact sorted 5502 full_a symbols"
        )
    return symbols


def _load_pit_records(
    *,
    membership_path: Path,
    expected_membership_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    membership_raw = _stable_bytes(
        membership_path,
        expected_sha256=expected_membership_sha256,
        private=False,
    )
    manifest_raw = _stable_bytes(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        private=False,
    )
    manifest = _strict_json(manifest_raw, str(manifest_path), canonical_file=False)
    if (
        manifest.get("row_count") != EXPECTED_PIT_COUNT
        or manifest.get("canonical_path") != str(membership_path)
        or manifest.get("canonical_sha256")
        != hashlib.sha256(membership_raw).hexdigest()
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "PIT generation manifest does not bind the explicit membership file"
        )
    frame = pd.read_parquet(membership_path)
    if len(frame) != EXPECTED_PIT_COUNT:
        raise FactorV4_1SignalDiagnosticRunnerError("PIT row count mismatch")
    records = frame.to_dict(orient="records")
    normalized = source.validate_pit_records_v4_1(records)
    if len(normalized) != EXPECTED_PIT_COUNT:
        raise FactorV4_1SignalDiagnosticRunnerError(
            "normalized PIT row count mismatch"
        )
    return records, manifest


def _reproduce_session_scope(
    *,
    design_source: Mapping[str, Any],
    pit_records: Sequence[Mapping[str, Any]],
    component_symbols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    design = source.validate_design_source_node_v4_1(
        design_source,
        pit_records=list(pit_records),
        expected_component_count=EXPECTED_COMPONENT_COUNT,
    )
    sessions = design["calendar_sessions"]
    if (
        len(sessions) != EXPECTED_SESSION_COUNT
        or sessions[0] != EXPECTED_ANALYSIS_START
        or sessions[-1] != EXPECTED_CUTOFF
        or design["pit_record_count"] != EXPECTED_PIT_COUNT
        or design["component_symbols"] != list(component_symbols)
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "design source is not the exact 1227-session frozen scope"
        )
    normalized = source.validate_pit_records_v4_1(pit_records)
    symbols = [row["symbol"] for row in normalized]
    by_symbol = {row["symbol"]: row for row in normalized}
    symbol_position = {symbol: index for index, symbol in enumerate(symbols)}
    mask = np.zeros((len(sessions), len(symbols)), dtype=bool)
    reproduced: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        scope_kind = (
            source.CUTOFF_SCOPE
            if session == design["cutoff_date"]
            else source.DESIGN_HISTORY_SCOPE
        )
        components = list(component_symbols) if scope_kind == source.CUTOFF_SCOPE else None
        descriptor = source.build_session_scope_descriptor_v4_1(
            pit_records,
            session,
            scope_kind,
            components,
        )
        if descriptor != design["session_scope_descriptors"][session_index]:
            raise FactorV4_1SignalDiagnosticRunnerError(
                f"session-scope descriptor drift: {session}"
            )
        reproduced.append(descriptor)
        domain = component_symbols if components is not None else symbols
        for symbol in domain:
            record = by_symbol[symbol]
            if record["effective_from"] <= session and (
                record["effective_to"] is None or session < record["effective_to"]
            ):
                mask[session_index, symbol_position[symbol]] = True
    dates = pd.DatetimeIndex(pd.to_datetime(sessions), name="trade_date")
    frame = pd.DataFrame(mask, index=dates, columns=symbols, dtype=bool)
    binding = {
        "session_count": len(sessions),
        "pit_record_count": len(symbols),
        "component_count": len(component_symbols),
        "descriptor_semantic_sha256": _semantic_sha(reproduced),
        "eligibility_matrix": evaluator.matrix_hash_descriptor_v4_1(
            frame.astype(float)
        ),
    }
    return frame, binding


def _derive_vwap(amount: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    if not amount.index.equals(volume.index) or not amount.columns.equals(
        volume.columns
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "VWAP inputs must share exact axes"
        )
    amount_values = amount.to_numpy(dtype=np.float64, copy=True)
    volume_values = volume.to_numpy(dtype=np.float64, copy=True)
    result = np.full(amount_values.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(amount_values)
        & np.isfinite(volume_values)
        & (volume_values != 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result[valid] = amount_values[valid] * 10.0 / volume_values[valid]
    result[~np.isfinite(result)] = np.nan
    return pd.DataFrame(result, index=amount.index, columns=amount.columns)


def _load_market_matrices(
    *,
    table_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    eligibility_mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    paths = [
        str(table_root / row["relative_path"])
        for row in inventory
        if row["dataset_member"] is True
    ]
    dataset = ds.dataset(paths, format="parquet")
    schema_names = set(dataset.schema.names)
    if not set(MARKET_COLUMNS).issubset(schema_names):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "bound table is missing one or more exact market columns"
        )
    compact_start = EXPECTED_ANALYSIS_START.replace("-", "")
    compact_cutoff = EXPECTED_CUTOFF.replace("-", "")
    table = dataset.to_table(
        columns=list(MARKET_COLUMNS),
        filter=(ds.field("trade_date") >= compact_start)
        & (ds.field("trade_date") <= compact_cutoff),
    )
    raw = table.to_pandas()
    if raw.empty or set(raw.columns) != set(MARKET_COLUMNS):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "market projection is empty or has unexpected columns"
        )
    if any(
        type(item) is not str or not compact_start <= item <= compact_cutoff
        for item in raw["trade_date"]
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "market projection contains prestart/postcutoff dates"
        )
    if raw.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1SignalDiagnosticRunnerError(
            "market projection contains duplicate symbol/session rows"
        )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    axes_index = eligibility_mask.index
    axes_columns = eligibility_mask.columns
    matrices: dict[str, pd.DataFrame] = {}
    mapping = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "vol",
        "amount": "amount",
    }
    for field, column in mapping.items():
        matrix = raw.pivot(index="trade_date", columns="ts_code", values=column)
        matrix = matrix.reindex(index=axes_index, columns=axes_columns).astype(float)
        matrices[field] = matrix.where(eligibility_mask)
    matrices["vwap"] = _derive_vwap(
        matrices["amount"], matrices["volume"]
    ).where(eligibility_mask)
    return matrices


def _binding_snapshot(paths: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for row in paths:
        path = Path(row["absolute_path"])
        raw = _stable_bytes(
            path, expected_sha256=row["byte_sha256"], private=False
        )
        snapshot[str(path)] = hashlib.sha256(raw).hexdigest()
    return snapshot


def _protected_stability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "binding_id": row["binding_id"],
            "absolute_path": row["absolute_path"],
            "expected_sha256": row["byte_sha256"],
            "before_sha256": row["byte_sha256"],
            "after_sha256": row["byte_sha256"],
        }
        for row in rows
    ]


def _revalidate_prepublication_inputs(
    *,
    precompute_paths: Sequence[Mapping[str, Any]],
    before_state: Mapping[str, str],
    table_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    inventory_sha: str,
    source_bindings: Sequence[Mapping[str, Any]],
    bundle_specs: Sequence[Mapping[str, Any]],
) -> None:
    if len(source_bindings) != 3 or len(bundle_specs) != 3:
        raise FactorV4_1SignalDiagnosticRunnerError(
            "prepublication revalidation requires exact source and predecessor sets"
        )
    if _binding_snapshot(precompute_paths) != dict(before_state):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "code or protected binding changed before publication"
        )
    current_inventory, current_sha = _inventory_table(table_root)
    if list(current_inventory) != list(inventory) or current_sha != inventory_sha:
        raise FactorV4_1SignalDiagnosticRunnerError(
            "table inventory changed before publication"
        )
    for row in source_bindings:
        _stable_bytes(
            Path(row["absolute_path"]),
            expected_sha256=row["byte_sha256"],
            private=False,
        )
    for spec in bundle_specs:
        _read_bundle(**dict(spec))


def run(args: argparse.Namespace) -> dict[str, Any]:
    private_root = _absolute_path(args.private_root, "private root")
    discovery_path = _absolute_path(args.discovery_bundle_path, "Discovery bundle")
    formal_path = _absolute_path(args.formal_bundle_path, "formal bundle")
    cutoff_path = _absolute_path(args.cutoff_bundle_path, "cutoff bundle")
    pit_path = _absolute_path(args.pit_membership_path, "PIT membership")
    pit_manifest_path = _absolute_path(
        args.pit_generation_manifest_path, "PIT generation manifest"
    )
    components_path = _absolute_path(args.components_path, "components")
    table_root = _absolute_path(args.table_root, "table root")
    code_rows = _parse_bindings(args.code_binding, "code binding")
    protected_rows = _parse_bindings(args.protected_binding, "protected binding")
    code_rows = _read_control_bindings(code_rows)
    protected_rows = _read_control_bindings(protected_rows)
    required_code_paths = {
        str(Path(__file__).resolve()),
        str(Path(evaluator.__file__).resolve()),
        str(Path(diagnostic.__file__).resolve()),
    }
    if not required_code_paths.issubset(
        {row["absolute_path"] for row in code_rows}
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "code bindings must include runner, evaluator, and diagnostic modules"
        )

    discovery = _read_bundle(
        bundle_path=discovery_path,
        filenames=DISCOVERY_FILENAMES,
        report_filename=DISCOVERY_REPORT,
        expected_report_sha256=args.expected_discovery_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_discovery_readback_report_semantic_sha256
        ),
        report_semantic_field="report_semantic_sha256",
    )
    formal = _read_bundle(
        bundle_path=formal_path,
        filenames=FORMAL_FILENAMES,
        report_filename=FORMAL_REPORT,
        expected_report_sha256=args.expected_formal_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_formal_readback_report_semantic_sha256
        ),
        report_semantic_field="report_semantic_sha256",
    )
    cutoff = _read_bundle(
        bundle_path=cutoff_path,
        filenames=CUTOFF_FILENAMES,
        report_filename=CUTOFF_REPORT,
        expected_report_sha256=args.expected_cutoff_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_cutoff_readback_report_semantic_sha256
        ),
        report_semantic_field=None,
        allow_lock=True,
    )
    if (
        discovery["values"][DISCOVERY_REPORT].get("readiness")
        != "EXPLORATORY_DISCOVERY"
        or formal["values"][FORMAL_REPORT].get("readiness")
        != "EXPLORATORY_FORMAL_CATALOG_CLASSIFICATION_ONLY"
        or cutoff["values"][CUTOFF_REPORT].get("readiness")
        != "EXPLORATORY_PRECOMMITTED"
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "one or more upstream research bundles has an invalid readiness"
        )

    cutoff_binding = cutoff["values"]["cutoff_input_binding.v4_1.json"]
    table_binding = cutoff_binding.get("table")
    pit_binding = cutoff_binding.get("pit_generation")
    component_binding = cutoff_binding.get("components")
    if not all(
        isinstance(item, Mapping)
        for item in (table_binding, pit_binding, component_binding)
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "cutoff input binding is incomplete"
        )
    cross_checks = (
        (str(table_root), table_binding.get("absolute_root"), "table root"),
        (
            str(pit_path),
            pit_binding.get("membership", {}).get("absolute_path"),
            "PIT membership",
        ),
        (
            str(pit_manifest_path),
            pit_binding.get("manifest", {}).get("absolute_path"),
            "PIT manifest",
        ),
        (
            str(components_path),
            component_binding.get("absolute_path"),
            "components",
        ),
    )
    if any(actual != bound for actual, bound, _context in cross_checks):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "explicit source path differs from cutoff input binding"
        )
    hash_checks = (
        (
            args.expected_pit_membership_sha256,
            pit_binding.get("membership", {}).get("sha256"),
        ),
        (
            args.expected_pit_generation_manifest_sha256,
            pit_binding.get("manifest", {}).get("sha256"),
        ),
        (args.expected_components_sha256, component_binding.get("sha256")),
        (args.expected_table_inventory_sha256, table_binding.get("inventory_sha256")),
    )
    if any(_sha(actual, "explicit source SHA") != bound for actual, bound in hash_checks):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "explicit source SHA differs from cutoff input binding"
        )
    inventory, inventory_sha = _inventory_table(table_root)
    if (
        inventory_sha != args.expected_table_inventory_sha256
        or inventory != table_binding.get("parquet_inventory")
        or len(inventory) != table_binding.get("regular_file_count")
        or sum(row["dataset_member"] is True for row in inventory)
        != table_binding.get("parquet_file_count")
    ):
        raise FactorV4_1SignalDiagnosticRunnerError(
            "table full inventory differs from the cutoff binding"
        )
    components = _load_components(
        components_path, args.expected_components_sha256
    )
    pit_records, _pit_manifest = _load_pit_records(
        membership_path=pit_path,
        expected_membership_sha256=args.expected_pit_membership_sha256,
        manifest_path=pit_manifest_path,
        expected_manifest_sha256=args.expected_pit_generation_manifest_sha256,
    )
    design_source = cutoff["values"]["design_source.v4_1.json"]
    eligibility_mask, scope_binding = _reproduce_session_scope(
        design_source=design_source,
        pit_records=pit_records,
        component_symbols=components,
    )
    source_node = source_readback.validate_cutoff_source_node_v4_1(
        cutoff["values"]["source_chain_node.v4_1.json"],
        cycle_id=args.cycle_id,
        input_binding=cutoff_binding,
        design_source=design_source,
        source_binding_sha256=cutoff["values"]["source_chain_node.v4_1.json"][
            "source_binding_sha256"
        ],
    )
    if source_node["cycle_id"] != args.cycle_id:
        raise FactorV4_1SignalDiagnosticRunnerError("cutoff cycle_id mismatch")

    ideas = evaluator.bind_pinned_source_ideas_v4_1(
        source_receipt=discovery["values"]["aquant_source_receipt.v4_1.json"],
        source_idea_audit=discovery["values"]["source_idea_audit.v4_1.json"],
        primitive_mapping_proof=formal["values"]["primitive_mapping_proof.v4_1.json"],
        formal_catalog=formal["values"]["candidate_catalog.v4.json"],
    )
    structural_audit = diagnostic.build_structural_no_label_audit_v4_1(
        {
            "data_builder": (
                str(Path(__file__).resolve()),
                Path(__file__).resolve().read_bytes(),
            ),
            "evaluator": (
                str(Path(evaluator.__file__).resolve()),
                Path(evaluator.__file__).resolve().read_bytes(),
            ),
        }
    )
    source_bindings = [
        {
            "binding_id": "discovery_readback",
            **discovery["descriptors"][DISCOVERY_REPORT],
            "semantic_sha256": discovery["report_semantic_sha256"],
        },
        {
            "binding_id": "formal_readback",
            **formal["descriptors"][FORMAL_REPORT],
            "semantic_sha256": formal["report_semantic_sha256"],
        },
        {
            "binding_id": "cutoff_readback",
            **cutoff["descriptors"][CUTOFF_REPORT],
            "semantic_sha256": cutoff["report_semantic_sha256"],
        },
        {
            "binding_id": "cutoff_lock",
            **cutoff["lock_descriptor"],
            "semantic_sha256": _semantic_sha(
                {"contract": "factor-cutoff-lock-empty.v1", "size_bytes": 0}
            ),
        },
    ]
    operator_profile = diagnostic.build_operator_profile_v4_1(
        cycle_id=args.cycle_id,
        bound_ideas=ideas,
        source_bindings=source_bindings,
        code_bindings=code_rows,
        structural_audit=structural_audit,
    )

    precompute_paths = [*code_rows, *protected_rows]
    before_state = _binding_snapshot(precompute_paths)
    matrices = _load_market_matrices(
        table_root=table_root,
        inventory=inventory,
        eligibility_mask=eligibility_mask,
    )
    matrix_bindings: list[dict[str, Any]] = []
    for field in sorted(matrices):
        descriptor = evaluator.matrix_hash_descriptor_v4_1(matrices[field])
        matrix_bindings.append(
            {
                "binding_id": f"matrix:{field}",
                "absolute_path": str(table_root),
                "byte_sha256": descriptor["matrix_sha256"],
                "semantic_sha256": _semantic_sha(descriptor),
            }
        )
    mask_descriptor = evaluator.matrix_hash_descriptor_v4_1(
        eligibility_mask.astype(float)
    )
    matrix_bindings.append(
        {
            "binding_id": "matrix:eligibility",
            "absolute_path": str(table_root),
            "byte_sha256": mask_descriptor["matrix_sha256"],
            "semantic_sha256": _semantic_sha(mask_descriptor),
        }
    )
    rows: list[dict[str, Any]] = []
    for idea in ideas:
        status = diagnostic.classify_idea_status_v4_1(idea)
        if status == diagnostic.STATUS_SIGNAL_DIAGNOSTIC:
            signal = evaluator.evaluate_pinned_idea_v4_1(
                idea=idea,
                matrices={field: matrices[field] for field in idea["input_fields"]},
                eligibility_mask=eligibility_mask,
            )
            row = diagnostic.build_diagnostic_row_v4_1(
                idea=idea,
                status=status,
                signal=signal,
                eligibility_mask=eligibility_mask,
            )
        else:
            row = diagnostic.build_diagnostic_row_v4_1(idea=idea, status=status)
        rows.append(row)
    after_state = _binding_snapshot(precompute_paths)
    if before_state != after_state:
        raise FactorV4_1SignalDiagnosticRunnerError(
            "code or protected binding changed during computation"
        )
    inventory_after, inventory_sha_after = _inventory_table(table_root)
    if inventory_after != inventory or inventory_sha_after != inventory_sha:
        raise FactorV4_1SignalDiagnosticRunnerError(
            "table inventory changed during computation"
        )
    vwap_semantic_sha = _semantic_sha(
        {
            "contract": "factor-vwap-derivation.v1",
            "amount_unit": "thousand_cny",
            "volume_unit": "lot_100_shares",
            "formula": "amount_times_10_divided_by_vol",
            "zero_or_nonfinite_denominator_to_nan": True,
            "nonfinite_result_to_nan": True,
        }
    )
    input_bindings = [
        *source_bindings,
        {
            "binding_id": "pit_membership",
            "absolute_path": str(pit_path),
            "byte_sha256": args.expected_pit_membership_sha256,
        },
        {
            "binding_id": "pit_generation_manifest",
            "absolute_path": str(pit_manifest_path),
            "byte_sha256": args.expected_pit_generation_manifest_sha256,
        },
        {
            "binding_id": "components",
            "absolute_path": str(components_path),
            "byte_sha256": args.expected_components_sha256,
        },
        {
            "binding_id": "table_inventory",
            "absolute_path": str(table_root),
            "byte_sha256": inventory_sha,
        },
    ]
    signal_diagnostic = diagnostic.build_signal_diagnostic_v4_1(
        cycle_id=args.cycle_id,
        operator_profile=operator_profile,
        rows=rows,
        input_bindings=input_bindings,
        protected_stability=_protected_stability(protected_rows),
        market_matrix_bindings=matrix_bindings,
        session_scope_binding=scope_binding,
        vwap_semantic_sha256=vwap_semantic_sha,
    )
    artifacts = {
        diagnostic.OPERATOR_PROFILE_FILENAME: operator_profile,
        diagnostic.DIAGNOSTIC_FILENAME: signal_diagnostic,
    }
    contract = diagnostic.build_private_bundle_contract_v4_1(
        expected_artifacts=artifacts
    )

    source_revalidation_bindings = [
        {
            "absolute_path": str(pit_path),
            "byte_sha256": args.expected_pit_membership_sha256,
        },
        {
            "absolute_path": str(pit_manifest_path),
            "byte_sha256": args.expected_pit_generation_manifest_sha256,
        },
        {
            "absolute_path": str(components_path),
            "byte_sha256": args.expected_components_sha256,
        },
    ]
    predecessor_bundle_specs = [
        {
            "bundle_path": discovery_path,
            "filenames": DISCOVERY_FILENAMES,
            "report_filename": DISCOVERY_REPORT,
            "expected_report_sha256": (
                args.expected_discovery_readback_report_sha256
            ),
            "expected_report_semantic_sha256": (
                args.expected_discovery_readback_report_semantic_sha256
            ),
            "report_semantic_field": "report_semantic_sha256",
        },
        {
            "bundle_path": formal_path,
            "filenames": FORMAL_FILENAMES,
            "report_filename": FORMAL_REPORT,
            "expected_report_sha256": args.expected_formal_readback_report_sha256,
            "expected_report_semantic_sha256": (
                args.expected_formal_readback_report_semantic_sha256
            ),
            "report_semantic_field": "report_semantic_sha256",
        },
        {
            "bundle_path": cutoff_path,
            "filenames": CUTOFF_FILENAMES,
            "report_filename": CUTOFF_REPORT,
            "expected_report_sha256": args.expected_cutoff_readback_report_sha256,
            "expected_report_semantic_sha256": (
                args.expected_cutoff_readback_report_semantic_sha256
            ),
            "report_semantic_field": None,
            "allow_lock": True,
        },
    ]

    def revalidate_inputs() -> None:
        _revalidate_prepublication_inputs(
            precompute_paths=precompute_paths,
            before_state=before_state,
            table_root=table_root,
            inventory=inventory,
            inventory_sha=inventory_sha,
            source_bindings=source_revalidation_bindings,
            bundle_specs=predecessor_bundle_specs,
        )

    published = private_io.publish_private_bundle(
        private_root=private_root,
        run_id=args.run_id,
        artifacts=artifacts,
        contract=contract,
        revalidate_inputs=revalidate_inputs,
    )
    independent = private_io.readback_private_bundle(
        published["bundle_path"], contract=contract
    )
    if independent.get("accepted") is not True:
        raise FactorV4_1SignalDiagnosticRunnerError(
            "independent diagnostic readback failed"
        )
    return {
        "accepted": True,
        "readiness": diagnostic.READINESS,
        "bundle_path": independent["bundle_path"],
        "readback_report_semantic_sha256": independent["readback_report"][
            "report_semantic_sha256"
        ],
        "candidate_count": evaluator.EXPECTED_PINNED_IDEA_COUNT,
        "status_counts": dict(diagnostic.EXACT_STATUS_COUNTS),
        "signal_computability_proven": False,
        "new_risk_authorized": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit-path research-only Factor v4.1 signal diagnostic."
    )
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--discovery-bundle-path", required=True)
    parser.add_argument("--expected-discovery-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-discovery-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--formal-bundle-path", required=True)
    parser.add_argument("--expected-formal-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-formal-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--cutoff-bundle-path", required=True)
    parser.add_argument("--expected-cutoff-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-cutoff-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--pit-membership-path", required=True)
    parser.add_argument("--expected-pit-membership-sha256", required=True)
    parser.add_argument("--pit-generation-manifest-path", required=True)
    parser.add_argument("--expected-pit-generation-manifest-sha256", required=True)
    parser.add_argument("--components-path", required=True)
    parser.add_argument("--expected-components-sha256", required=True)
    parser.add_argument("--table-root", required=True)
    parser.add_argument("--expected-table-inventory-sha256", required=True)
    parser.add_argument("--code-binding", action="append", required=True)
    parser.add_argument("--protected-binding", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (FactorV4_1SignalDiagnosticRunnerError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {"accepted": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
