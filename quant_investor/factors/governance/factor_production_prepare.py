"""Non-pointer preparation of the isolated production Factor closure.

This module materializes the exact three strict Factor inputs from one clean
CN Market snapshot, its Market-coverage-bound PIT generation, and an already
published trusted-provider calendar capture.  It never calls a provider,
broker, order, portfolio, System activation, or Factor activation API.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Final
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import (
    artifact_byte_sha256,
    canonical_json_bytes,
    parse_canonical_json_bytes,
    seal_artifact,
)
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.pit_universe import evaluate_listing_status
from quant_investor.market.tushare_calendar_authority import (
    SOURCE_LIMITATIONS,
    _rename_no_replace,
    _capture_projection,
    build_trusted_provider_calendar_compilation,
    validate_published_trusted_provider_calendar_capture_root,
)
from quant_investor.market.exchange_calendar_closure import (
    runtime_json_bytes,
    runtime_parquet_bytes,
)
from quant_investor.factors.governance.bootstrap_selection import build_market_pit_selection
from quant_investor.system.store import SystemStore, validate_object_ref
from quant_investor.system.release_install import (
    verify_running_release_install_input,
)

from .bootstrap import BLEND_W80, LOW_DOLLAR_VOLUME
from .errors import FactorGovernanceError
from .implementations import installed_semantic_row
from .production_authority import (
    build_factor_calendar_capture_custody_attestation,
    build_factor_legacy_zero_call_certificate_for_release,
    build_factor_production_generation,
    build_factor_production_market_input,
    build_factor_production_recomputation_evidence,
    build_factor_production_source_closure,
    recompute_factor_production_signals,
    replay_factor_production_generation,
    system_store_source_resolver,
)
from .source import role_schema
from .store import FactorValidationStore

_SHA256_RE: Final = frozenset("0123456789abcdef")
_SOURCE_ROOT_LABEL: Final = "factor-production-prepare"
_PREPARED_ROOT: Final = PurePosixPath("results/factors/preparations")
_REQUIRED_OPEN_SESSIONS: Final = 91
_PREPARED_RECEIPT_NAME: Final = "prepared.json"
_PREPARED_RECEIPT_FIELDS: Final = frozenset(
    {
        "status",
        "operation_id",
        "operation_inputs_sha256",
        "operation_inputs_ref",
        "as_of",
        "source_root",
        "source_root_id",
        "release_repository_root",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "legacy_zero_call_ref",
        "market_snapshot_id",
        "market_pit_selection_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "market_scope_source_ref",
        "market_input_ref",
        "factor_production_source_closure_ref",
        "factor_production_recomputation_ref",
        "factor_production_generation_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "exact_replay_sha256",
        "factor_readiness",
        "factor_authority",
        "system_pointer_writes",
        "factor_pointer_writes",
        "broker_order_trade_fund_writes",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_ref(artifact: Mapping[str, Any]) -> dict[str, str]:
    return {
        "kind": str(artifact["kind"]),
        "contract_sha256": str(artifact["contract_sha256"]),
        "artifact_id": str(artifact["artifact_id"]),
        "semantic_sha256": str(artifact["semantic_sha256"]),
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _SHA256_RE for char in value):
        raise FactorGovernanceError(f"{label} is not lowercase SHA-256")
    return value


def _as_of(value: Any) -> str:
    if type(value) is not str:
        raise FactorGovernanceError("Factor prepare as_of is invalid")
    compact = value.replace("-", "")
    if len(compact) != 8 or not compact.isdigit():
        raise FactorGovernanceError("Factor prepare as_of is invalid")
    try:
        parsed = date.fromisoformat(f"{compact[:4]}-{compact[4:6]}-{compact[6:]}")
    except ValueError as exc:
        raise FactorGovernanceError("Factor prepare as_of is invalid") from exc
    if parsed.strftime("%Y%m%d") != compact:
        raise FactorGovernanceError("Factor prepare as_of is invalid")
    return compact


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise FactorGovernanceError(f"{label} raw bytes are invalid")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FactorGovernanceError(f"{label} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise FactorGovernanceError(f"{label} has non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
        canonical_json_bytes(value)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise FactorGovernanceError(f"{label} is not strict JSON") from exc
    if type(value) is not dict:
        raise FactorGovernanceError(f"{label} is not a JSON object")
    return dict(value)


def _git_blob(root: Path, *, commit: str, relative_path: str) -> bytes:
    """Read one tracked blob from the exact verified release commit."""

    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=str(root),
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FactorGovernanceError("frozen release source cannot be read") from exc
    if (
        result.returncode != 0
        or type(result.stdout) is not bytes
        or type(result.stderr) is not bytes
        or not result.stdout
    ):
        raise FactorGovernanceError("frozen release source differs")
    return result.stdout


def _read_regular(  # noqa: C901
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 512 * 1024 * 1024,
    allow_public_read: bool = False,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FactorGovernanceError(f"{label} is unavailable") from exc
    mode = stat.S_IMODE(metadata.st_mode)
    rejected_mode_bits = 0o133 if allow_public_read else 0o177
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or mode & rejected_mode_bits
        or not mode & 0o400
        or metadata.st_size <= 0
        or metadata.st_size > maximum_bytes
    ):
        raise FactorGovernanceError(f"{label} is not a bounded regular file")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise FactorGovernanceError(f"{label} cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise FactorGovernanceError(f"{label} changed during read") from exc

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
        )

    raw = b"".join(chunks)
    if (
        identity(metadata) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(current)
        or len(raw) != after.st_size
    ):
        raise FactorGovernanceError(f"{label} changed during read")
    return raw


def _canonical_market_scope(data_root: Path) -> tuple[list[str], bytes]:
    """Read the fixed full-A scope preimage instead of the serving-symbol subset."""

    raw = _read_regular(
        data_root / "cn_universe" / "cn_index_components.json",
        label="canonical Market full-A scope",
        maximum_bytes=16 * 1024 * 1024,
        allow_public_read=True,
    )
    document = _strict_json(raw, label="canonical Market full-A scope")
    values = document.get("full_a")
    if type(values) is not list or not values:
        raise FactorGovernanceError("canonical Market full-A scope is absent")
    symbols = list(values)
    if any(
        type(symbol) is not str
        or len(symbol) != 9
        or not symbol[:6].isdigit()
        or symbol[6:] not in {".SH", ".SZ", ".BJ"}
        for symbol in symbols
    ) or symbols != sorted(set(symbols), key=lambda value: value.encode("utf-8")):
        raise FactorGovernanceError("canonical Market full-A scope is invalid")
    stats = document.get("stats")
    if type(stats) is not dict or stats.get("full_a") != len(symbols):
        raise FactorGovernanceError("canonical Market full-A scope count differs")
    for alias in ("full_market", "all_a", "all"):
        if alias in document and document[alias] != symbols:
            raise FactorGovernanceError("canonical Market full-A scope aliases differ")
    return symbols, canonical_json_bytes({"full_a": symbols, "stats": {"full_a": len(symbols)}})


def _write_source(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FactorGovernanceError("Factor preparation source path already exists")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise FactorGovernanceError("Factor preparation source write failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    observed = _read_regular(path, label="prepared source")
    if observed != raw:
        raise FactorGovernanceError("Factor preparation source readback differs")


def _ensure_owner_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    try:
        observed = path.lstat()
    except OSError as exc:
        raise FactorGovernanceError("Factor governed directory is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
    ):
        raise FactorGovernanceError("Factor governed directory custody differs")
    path.chmod(0o700)
    if stat.S_IMODE(path.lstat().st_mode) != 0o700:
        raise FactorGovernanceError("Factor governed directory mode differs")


def _read_prepared_receipt(root: Path, *, operation_id: str) -> dict[str, Any]:
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise FactorGovernanceError("Factor preparation root is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise FactorGovernanceError("Factor preparation root custody differs")
    raw = _read_regular(root / _PREPARED_RECEIPT_NAME, label="Factor prepared receipt")
    receipt = _strict_json(raw, label="Factor prepared receipt")
    if (
        set(receipt) != _PREPARED_RECEIPT_FIELDS
        or receipt.get("status") != "PREPARED"
        or receipt.get("operation_id") != operation_id
        or receipt.get("operation_inputs_sha256")
        != operation_id.removeprefix("factor-production-prepare-")
    ):
        raise FactorGovernanceError("Factor prepared receipt identity differs")
    return receipt


def _validate_existing_preparation(
    *,
    workspace: Path,
    operation_id: str,
    source_root_id: str,
) -> dict[str, Any]:
    root = workspace / _PREPARED_ROOT / operation_id
    receipt = _read_prepared_receipt(root, operation_id=operation_id)
    required = {
        "factor_production_source_closure_ref",
        "factor_production_recomputation_ref",
        "factor_production_generation_ref",
    }
    if not required <= set(receipt):
        raise FactorGovernanceError("Factor prepared receipt closure is incomplete")
    store = SystemStore(
        workspace,
        source_root=root / "sources",
        source_root_id=source_root_id,
    )
    for field in required:
        store.get_object(receipt[field])
    generation = store.get_object(receipt["factor_production_generation_ref"])
    replay_factor_production_generation(
        generation,
        artifact_resolver=store.get_object,
        source_resolver=system_store_source_resolver(store),
        validation_mode="HISTORICAL_RECOVERY",
    )
    generation_payload = generation["payload"]
    source_closure = store.get_object(generation_payload["source_closure_ref"])
    source_payload = source_closure["payload"]
    recomputation = store.get_object(generation_payload["recomputation_evidence_ref"])
    market_input = store.get_object(source_payload["market_input_ref"])
    selection = store.get_object(source_payload["market_pit_selection_ref"])
    calendar_custody = store.get_object(source_payload["calendar_capture_custody_attestation_ref"])
    calendar_execution = store.get_object(calendar_custody["payload"]["capture_execution_ref"])
    operation_ref = receipt["operation_inputs_ref"]
    if (
        type(operation_ref) is not dict
        or set(operation_ref) != {"relative_path", "byte_sha256"}
        or operation_ref["relative_path"] != "operation-inputs.json"
    ):
        raise FactorGovernanceError("Factor prepared operation input ref differs")
    operation_raw = _read_regular(root / "operation-inputs.json", label="Factor operation inputs")
    operation_document = _strict_json(operation_raw, label="Factor operation inputs")
    if (
        canonical_json_bytes(operation_document) != operation_raw
        or _sha256(operation_raw) != operation_ref["byte_sha256"]
        or _sha256(operation_raw) != receipt["operation_inputs_sha256"]
    ):
        raise FactorGovernanceError("Factor prepared operation input bytes differ")
    expected = {
        "status": "PREPARED",
        "operation_id": operation_id,
        "operation_inputs_sha256": _sha256(operation_raw),
        "operation_inputs_ref": dict(operation_ref),
        "as_of": generation_payload["as_of"],
        "source_root": f"results/factors/preparations/{operation_id}/sources",
        "source_root_id": source_root_id,
        "release_repository_root": calendar_execution["payload"]["release_repository_root"],
        "deployed_release_ref": generation_payload["deployed_release_ref"],
        "release_install_evidence_ref": generation_payload["release_install_evidence_ref"],
        "release_install_input_source_ref": generation_payload["release_install_input_source_ref"],
        "legacy_zero_call_ref": generation_payload["legacy_zero_call_ref"],
        "market_snapshot_id": market_input["payload"]["market_snapshot_id"],
        "market_pit_selection_ref": _artifact_ref(selection),
        "calendar_authority_policy_ref": source_payload["calendar_authority_policy_ref"],
        "calendar_compilation_ref": source_payload["calendar_compilation_ref"],
        "calendar_capture_custody_attestation_ref": _artifact_ref(calendar_custody),
        "factor_source_bundle_ref": source_payload["factor_source_bundle_ref"],
        "market_scope_source_ref": source_payload["market_scope_source_ref"],
        "market_input_ref": _artifact_ref(market_input),
        "factor_production_source_closure_ref": _artifact_ref(source_closure),
        "factor_production_recomputation_ref": _artifact_ref(recomputation),
        "factor_production_generation_ref": _artifact_ref(generation),
        "low_signal_sha256": generation_payload["low_signal_sha256"],
        "w80_signal_sha256": generation_payload["w80_signal_sha256"],
        "exact_replay_sha256": generation_payload["exact_replay_sha256"],
        "factor_readiness": "READY",
        "factor_authority": "INACTIVE",
        "system_pointer_writes": 0,
        "factor_pointer_writes": 0,
        "broker_order_trade_fund_writes": 0,
    }
    if canonical_json_bytes(receipt) != canonical_json_bytes(expected):
        raise FactorGovernanceError("Factor prepared receipt generation closure differs")
    return receipt


def _capture_root(
    capture_root: Path,
    *,
    expected_success_sha256: str,
) -> tuple[
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    success_path = capture_root / "capture-success.json"
    success_raw = _read_regular(success_path, label="calendar capture success")
    if _sha256(success_raw) != _sha(expected_success_sha256, label="calendar success SHA"):
        raise FactorGovernanceError("calendar capture success SHA differs")
    success = parse_canonical_json_bytes(success_raw, label="calendar capture success")
    if type(success) is not dict or type(success.get("payload")) is not dict:
        raise FactorGovernanceError("calendar capture success payload differs")
    execution_ref = success["payload"].get("capture_execution_file_ref")
    if type(execution_ref) is not dict or type(execution_ref.get("relative_path")) is not str:
        raise FactorGovernanceError("calendar capture execution ref differs")
    execution_path = capture_root / Path(execution_ref["relative_path"]).name
    execution_raw = _read_regular(execution_path, label="calendar capture execution")
    if _sha256(execution_raw) != _sha(execution_ref.get("byte_sha256"), label="execution SHA"):
        raise FactorGovernanceError("calendar capture execution SHA differs")
    execution = parse_canonical_json_bytes(execution_raw, label="calendar capture execution")
    if type(execution) is not dict:
        raise FactorGovernanceError("calendar capture execution differs")
    success_ref = {
        "relative_path": f"{capture_root.name}/capture-success.json",
        "byte_sha256": expected_success_sha256,
    }
    leaves = validate_published_trusted_provider_calendar_capture_root(
        capture_parent=capture_root.parent,
        capture_execution=execution,
        capture_execution_file_ref=execution_ref,
        capture_success=success,
        capture_success_file_ref=success_ref,
    )
    return leaves, execution, success, dict(execution_ref), success_ref


def _contained_regular(path: Path, *, root: Path, label: str) -> tuple[Path, bytes]:
    if not path.is_absolute():
        path = root / path
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise FactorGovernanceError(f"{label} escapes PIT custody") from exc
    if lexical_root.is_symlink():
        raise FactorGovernanceError("PIT custody root is a symbolic link")
    current = lexical_root
    for part in lexical_path.relative_to(lexical_root).parts:
        current = current / part
        if current.is_symlink():
            raise FactorGovernanceError(f"{label} traverses a symbolic link")
    return lexical_path, _read_regular(lexical_path, label=label)


def _resolve_market_bound_pit_pointer(  # noqa: C901
    *,
    pit_manifest_path: Path,
    pit_generation_id: str,
    pit_manifest_sha256: str,
    pit_canonical_sha256: str,
) -> tuple[Path, bytes, dict[str, Any]]:
    """Resolve exact current-or-lineage discovery bytes for the Market-bound PIT.

    A newer global PIT pointer is allowed, but it is never rewritten into a
    synthetic historical pointer.  The exact historical discovery bytes must
    already be retained as a successor generation's ``parent_pointer.json``.
    """

    reference_root = pit_manifest_path.parent.parent.parent
    latest_path = reference_root / "stock_basic_membership_latest.json"
    pointer_path, pointer_raw = _contained_regular(
        latest_path, root=reference_root, label="current PIT discovery pointer"
    )
    visited: set[str] = set()
    while True:
        pointer = _strict_json(pointer_raw, label="PIT discovery pointer")
        generation_id = pointer.get("generation_id")
        manifest_sha = pointer.get("generation_manifest_sha256")
        canonical_sha = pointer.get("canonical_sha256")
        if pointer.get("discovery_schema_version") != "cn_pit_universe_latest.v1":
            raise FactorGovernanceError("PIT discovery pointer schema differs")
        if generation_id == pit_generation_id:
            if manifest_sha != pit_manifest_sha256 or canonical_sha != pit_canonical_sha256:
                raise FactorGovernanceError("Market-bound PIT discovery identity differs")
            return pointer_path, pointer_raw, pointer
        if type(generation_id) is not str or generation_id in visited:
            raise FactorGovernanceError("Market-bound PIT discovery lineage is unavailable")
        visited.add(generation_id)
        manifest_value = pointer.get("generation_manifest_path")
        if type(manifest_value) is not str or not manifest_value:
            raise FactorGovernanceError("PIT discovery manifest path differs")
        current_manifest_path, current_manifest_raw = _contained_regular(
            Path(manifest_value), root=reference_root, label="PIT lineage manifest"
        )
        if _sha256(current_manifest_raw) != _sha(manifest_sha, label="PIT manifest SHA"):
            raise FactorGovernanceError("PIT discovery manifest SHA differs")
        manifest = _strict_json(current_manifest_raw, label="PIT lineage manifest")
        lineage = manifest.get("lineage")
        if type(lineage) is not dict:
            raise FactorGovernanceError("Market-bound PIT discovery lineage is unavailable")
        parent_pointer_value = lineage.get("parent_discovery_pointer_artifact_path")
        parent_pointer_sha = lineage.get("parent_discovery_pointer_sha256")
        if type(parent_pointer_value) is not str or not parent_pointer_value:
            raise FactorGovernanceError("PIT parent discovery pointer path differs")
        pointer_path, pointer_raw = _contained_regular(
            Path(parent_pointer_value),
            root=reference_root,
            label="PIT parent discovery pointer",
        )
        if _sha256(pointer_raw) != _sha(parent_pointer_sha, label="PIT parent pointer SHA"):
            raise FactorGovernanceError("PIT parent discovery pointer SHA differs")


def _factor_market_rows(  # noqa: C901
    reader: MarketDataReader,
    *,
    records: Mapping[str, Any],
    sessions: list[date],
    as_of: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbols = sorted(records, key=lambda value: value.encode("utf-8"))
    if not symbols:
        raise FactorGovernanceError("Market-bound PIT cohort is empty")
    frames = reader.read_symbol_frames(
        symbols,
        start_date=sessions[0].strftime("%Y%m%d"),
        end_date=as_of,
        columns=["trade_date", "symbol", "ts_code", "adj_close", "amount", "vol", "total_mv"],
    )
    required_dates = set(sessions)
    market_rows: list[dict[str, Any]] = []
    pit_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        result = frames.get(symbol)
        frame = None if result is None else result.frame
        if frame is None or frame.empty:
            rows = pd.DataFrame()
        else:
            rows = frame.copy()
        if not rows.empty:
            symbol_column = "symbol" if "symbol" in rows else "ts_code" if "ts_code" in rows else ""
            if not symbol_column or "trade_date" not in rows:
                raise FactorGovernanceError("canonical Market frame schema differs")
            rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="coerce").dt.date
            rows = rows[rows["trade_date"].isin(required_dates)].copy()
            if rows["trade_date"].isna().any() or rows["trade_date"].duplicated().any():
                raise FactorGovernanceError("canonical Market session rows differ")
        record = records[symbol]
        listing = evaluate_listing_status(record, symbol=symbol, as_of=as_of)
        usable = listing.tradable and len(rows) == len(sessions)
        cutoff_mv: float | None = None
        if usable:
            rows = rows.sort_values("trade_date", kind="mergesort")
            for row in rows.to_dict(orient="records"):
                values = {field: row.get(field) for field in ("adj_close", "amount", "vol")}
                if any(
                    value is None or not math.isfinite(float(value)) for value in values.values()
                ):
                    usable = False
                    break
                if (
                    float(values["adj_close"]) <= 0
                    or float(values["amount"]) <= 0
                    or float(values["vol"]) < 0
                ):
                    usable = False
                    break
                market_rows.append(
                    {
                        "trade_date": row["trade_date"],
                        "symbol": symbol,
                        "adj_close": float(values["adj_close"]),
                        "amount": float(values["amount"]),
                        "vol": float(values["vol"]),
                    }
                )
            if usable and "total_mv" in rows:
                value = rows.iloc[-1]["total_mv"]
                if value is not None and math.isfinite(float(value)) and float(value) > 0:
                    cutoff_mv = float(value)
        if not usable:
            market_rows = [row for row in market_rows if row["symbol"] != symbol]
        pit_rows.append(
            {
                "signal_session": sessions[-1],
                "symbol": symbol,
                "industry": getattr(record, "industry", None) or None,
                "total_mv": cutoff_mv,
                "tradable": bool(usable),
            }
        )
    if not any(row["tradable"] for row in pit_rows):
        raise FactorGovernanceError("Market-bound PIT has no factor-tradable symbols")
    return market_rows, pit_rows


def prepare_factor_production(  # noqa: C901
    *,
    workspace_root: str | os.PathLike[str],
    market_data_root: str | os.PathLike[str],
    calendar_capture_root: str | os.PathLike[str],
    expected_calendar_success_sha256: str,
) -> dict[str, Any]:
    """Prepare sealed Factor-only sources and evidence; never activate authority."""

    workspace = Path(workspace_root).resolve(strict=True)
    market_root = Path(market_data_root).resolve(strict=True)
    capture_root = Path(calendar_capture_root).resolve(strict=True)
    reader = MarketDataReader(market="CN", data_root=market_root, mode_policy="strict")
    gate = reader.clean_snapshot_gate(refresh=True)
    if gate.get("healthy") is not True or gate.get("status") != "ok":
        raise FactorGovernanceError("strict canonical Market snapshot is not clean")
    cutoff = _as_of(gate.get("latest_complete_trade_date"))
    pit_binding = reader.coverage_bound_pit(refresh=True)
    if pit_binding.get("status") != "passed" or not pit_binding.get("records"):
        raise FactorGovernanceError("Market-bound PIT generation is unavailable")
    pointer_path = Path(str(gate["latest_pointer_path"])).resolve(strict=True)
    manifest_path = Path(str(gate["manifest_path"])).resolve(strict=True)
    pointer_raw = _read_regular(
        pointer_path,
        label="Market pointer",
        allow_public_read=True,
    )
    manifest_raw = _read_regular(
        manifest_path,
        label="Market snapshot manifest",
        allow_public_read=True,
    )
    pointer_document = _strict_json(pointer_raw, label="Market pointer")
    manifest_document = _strict_json(manifest_raw, label="Market snapshot manifest")
    canonical_pit_path = Path(str(pit_binding["canonical_path"])).resolve(strict=True)
    pit_manifest_path = Path(str(pit_binding["generation_manifest_path"])).resolve(strict=True)
    canonical_pit_raw = _read_regular(canonical_pit_path, label="Market-bound PIT membership")
    pit_manifest_raw = _read_regular(pit_manifest_path, label="Market-bound PIT manifest")
    market_scope_symbols, market_scope_raw = _canonical_market_scope(market_root)
    expected_scope_sha256 = _sha256("\n".join(market_scope_symbols).encode("utf-8"))
    coverage = pointer_document.get("coverage")
    if (
        type(coverage) is not dict
        or coverage.get("expected_scope_count") != len(market_scope_symbols)
        or coverage.get("coverage_complete_count") != len(market_scope_symbols)
        or coverage.get("expected_scope_sha256") != expected_scope_sha256
    ):
        raise FactorGovernanceError("canonical Market full-A scope SHA differs")
    if not set(market_scope_symbols) <= set(pit_binding["records"]):
        raise FactorGovernanceError("canonical Market scope is outside Market-bound PIT")
    bound_pointer_path, bound_pointer_raw, bound_pointer_document = (
        _resolve_market_bound_pit_pointer(
            pit_manifest_path=pit_manifest_path,
            pit_generation_id=str(pit_binding["generation_id"]),
            pit_manifest_sha256=str(pit_binding["generation_manifest_sha256"]),
            pit_canonical_sha256=str(pit_binding["canonical_sha256"]),
        )
    )
    observed_path = pit_manifest_path.parent.parent.parent / "stock_basic_membership_latest.json"
    observed_raw = _read_regular(observed_path, label="observed current PIT pointer")
    capture_leaves, capture_execution, capture_success, execution_file_ref, success_file_ref = (
        _capture_root(capture_root, expected_success_sha256=expected_calendar_success_sha256)
    )
    execution_payload = capture_execution["payload"]
    prepared_at = execution_payload.get("observed_completed_at")
    if type(prepared_at) is not str:
        raise FactorGovernanceError("calendar capture completion time is invalid")
    release_install_raw = capture_leaves["release-install-input.json"]
    release_install_document = _strict_json(
        release_install_raw, label="calendar release-install input"
    )
    if set(release_install_document) != {"release_install_evidence", "deployed_release"}:
        raise FactorGovernanceError("calendar release-install input fields differ")
    release_document = release_install_document["deployed_release"]
    release_install_evidence = release_install_document["release_install_evidence"]
    if type(release_document) is not dict or type(release_install_evidence) is not dict:
        raise FactorGovernanceError("calendar release-install artifacts differ")
    release_ref = _artifact_ref(release_document)
    if execution_payload.get("deployed_release_ref") != release_ref:
        raise FactorGovernanceError("calendar capture release differs from deployed release")
    release_install_evidence_ref = validate_object_ref(
        execution_payload.get("release_install_evidence_ref"),
        label="calendar release_install_evidence_ref",
    )
    if release_install_evidence_ref["kind"] != "system.release_install_evidence":
        raise FactorGovernanceError("calendar release-install evidence kind differs")
    release_install_verification = verify_running_release_install_input(
        release_install_raw,
        repository_root=execution_payload["release_repository_root"],
    )
    evidence_payload = release_install_evidence.get("payload")
    if type(evidence_payload) is not dict:
        raise FactorGovernanceError("release-install evidence payload differs")
    final_commit = evidence_payload.get("final_commit")
    final_tree = evidence_payload.get("final_tree")
    if type(final_commit) is not str or type(final_tree) is not str:
        raise FactorGovernanceError("release Git identity is absent")
    legacy = build_factor_legacy_zero_call_certificate_for_release(
        repository_root=execution_payload["release_repository_root"],
        final_commit=final_commit,
        final_tree=final_tree,
        resolver_inventory_ref=release_ref,
        verified_at=prepared_at,
    )
    calendar_custody = build_factor_calendar_capture_custody_attestation(
        capture_parent=capture_root.parent,
        capture_execution=capture_execution,
        capture_execution_file_ref=execution_file_ref,
        capture_success=capture_success,
        capture_success_file_ref=success_file_ref,
        deployed_release_ref=release_ref,
        verified_at=prepared_at,
    )
    release_repository_root = Path(str(execution_payload["release_repository_root"])).resolve(
        strict=True
    )
    decision_raw = _git_blob(
        release_repository_root,
        commit=final_commit,
        relative_path="operations/unified_cutover/bootstrap-decision.json",
    )
    _strict_json(decision_raw, label="frozen bootstrap decision")
    implementation_semantics = [
        installed_semantic_row(factor_id) for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
    ]
    operation_inputs = {
        "as_of": cutoff,
        "market_pointer_sha256": _sha256(pointer_raw),
        "market_manifest_sha256": _sha256(manifest_raw),
        "market_scope_sha256": _sha256(market_scope_raw),
        "pit_membership_sha256": _sha256(canonical_pit_raw),
        "pit_manifest_sha256": _sha256(pit_manifest_raw),
        "market_bound_pit_pointer_sha256": _sha256(bound_pointer_raw),
        "observed_current_pit_pointer_sha256": _sha256(observed_raw),
        "calendar_success_sha256": expected_calendar_success_sha256,
        "calendar_custody_attestation_sha256": artifact_byte_sha256(calendar_custody),
        "release_ref": release_ref,
        "release_install_evidence_ref": release_install_evidence_ref,
        "release_install_verification": release_install_verification,
        "bootstrap_decision_sha256": _sha256(decision_raw),
        "factor_implementation_semantics": implementation_semantics,
        "final_commit": final_commit,
        "final_tree": final_tree,
        "legacy_certificate_sha256": artifact_byte_sha256(legacy),
    }
    operation_inputs_sha256 = _sha256(canonical_json_bytes(operation_inputs))
    operation_id = "factor-production-prepare-" + operation_inputs_sha256
    results_root = workspace / "results"
    _ensure_owner_directory(results_root)
    factor_root = results_root / "factors"
    _ensure_owner_directory(factor_root)
    preparations_root = factor_root / "preparations"
    _ensure_owner_directory(preparations_root)
    final_root = preparations_root / operation_id
    source_root_id = _sha256(
        canonical_json_bytes({"domain": _SOURCE_ROOT_LABEL, "id": operation_id})
    )
    if final_root.exists():
        return _validate_existing_preparation(
            workspace=workspace,
            operation_id=operation_id,
            source_root_id=source_root_id,
        )
    staging_root = preparations_root / f".{operation_id}.{uuid.uuid4().hex}.tmp"
    source_root = staging_root / "sources"
    source_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    staging_root.chmod(0o700)
    source_root.chmod(0o700)
    operation_inputs_raw = canonical_json_bytes(operation_inputs)
    operation_inputs_ref = {
        "relative_path": "operation-inputs.json",
        "byte_sha256": _sha256(operation_inputs_raw),
    }
    _write_source(staging_root / "operation-inputs.json", operation_inputs_raw)
    store = SystemStore(workspace, source_root=source_root, source_root_id=source_root_id)
    stored_release_ref = store.put_object(release_document)
    stored_install_ref = store.put_object(release_install_evidence)
    if stored_release_ref != release_ref or stored_install_ref != release_install_evidence_ref:
        raise FactorGovernanceError("release-install object storage identity differs")

    def stage(
        relative: str,
        raw: bytes,
        *,
        source_object_id: str,
        source_format: str,
        media_type: str,
    ) -> dict[str, str]:
        _write_source(source_root / relative, raw)
        return store.put_source_file(
            relative,
            source_object_id=source_object_id,
            media_type=media_type,
            source_format=source_format,
            created_at=prepared_at,
        )

    # Canonical raw sources retained for selection/Market bridge.
    market_pointer_ref = stage(
        "market/pointer.json",
        pointer_raw,
        source_object_id="factor-market-pointer",
        source_format="JSON",
        media_type="application/json",
    )
    market_manifest_ref = stage(
        "market/manifest.json",
        manifest_raw,
        source_object_id="factor-market-manifest",
        source_format="JSON",
        media_type="application/json",
    )
    market_scope_ref = stage(
        "market/scope.json",
        market_scope_raw,
        source_object_id="factor-market-scope",
        source_format="JSON",
        media_type="application/json",
    )
    canonical_pit_ref = stage(
        "pit/canonical-membership.parquet",
        canonical_pit_raw,
        source_object_id="factor-canonical-pit",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    pit_manifest_ref = stage(
        "pit/manifest.json",
        pit_manifest_raw,
        source_object_id="factor-pit-manifest",
        source_format="JSON",
        media_type="application/json",
    )
    bound_pointer_ref = stage(
        "pit/market-bound-pointer.json",
        bound_pointer_raw,
        source_object_id="factor-market-bound-pit",
        source_format="JSON",
        media_type="application/json",
    )
    observed_ref = stage(
        "pit/observed-current.json",
        observed_raw,
        source_object_id="factor-observed-current-pit",
        source_format="JSON",
        media_type="application/json",
    )

    # Rebuild the immutable selection from exact staged raw objects.
    selection = build_market_pit_selection(
        as_of=cutoff,
        market_pointer_file_ref={
            "relative_path": "market/pointer.json",
            "byte_sha256": _sha256(pointer_raw),
        },
        market_snapshot_manifest_file_ref={
            "relative_path": "market/manifest.json",
            "byte_sha256": _sha256(manifest_raw),
        },
        market_bound_pit_pointer_file_ref={
            "relative_path": "pit/market-bound-pointer.json",
            "byte_sha256": _sha256(bound_pointer_raw),
        },
        pit_generation_manifest_file_ref={
            "relative_path": "pit/manifest.json",
            "byte_sha256": _sha256(pit_manifest_raw),
        },
        pit_membership_file_ref={
            "relative_path": "pit/canonical-membership.parquet",
            "byte_sha256": _sha256(canonical_pit_raw),
        },
        observed_current_pit_pointer_file_ref={
            "relative_path": "pit/observed-current.json",
            "byte_sha256": _sha256(observed_raw),
        },
        market_pointer=pointer_document,
        market_snapshot_manifest=manifest_document,
        market_bound_pit_pointer=bound_pointer_document,
        pit_generation_manifest=_strict_json(pit_manifest_raw, label="PIT manifest"),
        observed_current_pit_pointer=_strict_json(
            observed_raw, label="observed current PIT pointer"
        ),
        created_at=execution_payload["observed_completed_at"],
    )
    selection_ref = store.put_object(selection)

    policy = _strict_json(capture_leaves["policy.json"], label="calendar policy")
    capability = _strict_json(capture_leaves["capability.json"], label="calendar capability")
    captures = [
        _strict_json(
            capture_leaves[f"capture-{exchange.lower()}.json"], label=f"calendar {exchange} capture"
        )
        for exchange in ("SSE", "SZSE", "BSE")
    ]
    policy_ref_calendar = store.put_object(policy)
    store.put_object(capability)
    for capture in captures:
        store.put_object(capture)
    store.put_object(
        _strict_json(capture_leaves["capture-transaction.json"], label="calendar transaction")
    )
    store.put_object(capture_execution)
    store.put_object(capture_success)
    calendar_custody_ref = store.put_object(calendar_custody)
    calendar_rows: list[dict[str, Any]] = []
    release_install_input_source_ref: dict[str, str] | None = None
    raw_by_ref: dict[bytes, bytes] = {}
    for leaf, raw in sorted(capture_leaves.items()):
        relative = f"{capture_root.name}/{leaf}"
        source_format = "JSON" if leaf.endswith(".json") else "BINARY"
        media_type = "application/json" if source_format == "JSON" else "application/octet-stream"
        ref = stage(
            relative,
            raw,
            source_object_id="factor-calendar-" + leaf.replace(".", "-"),
            source_format=source_format,
            media_type=media_type,
        )
        calendar_rows.append({"role": "capture-" + leaf.replace(".", "-"), "source_ref": ref})
        if leaf == "release-install-input.json":
            release_install_input_source_ref = ref
        raw_by_ref[
            canonical_json_bytes(
                {"relative_path": f"{capture_root.name}/{leaf}", "byte_sha256": _sha256(raw)}
            )
        ] = raw

    # The capture artifacts reference root-name paths; add those exact aliases in the raw resolver.
    def capture_raw_resolver(reference: Mapping[str, Any]) -> bytes:
        key = canonical_json_bytes(dict(reference))
        try:
            return raw_by_ref[key]
        except KeyError as exc:
            raise FactorGovernanceError("calendar capture raw ref is outside sealed root") from exc

    # Reconstruct the deterministic runtime bytes from the already sealed raw
    # provider responses. This is pure local replay; no client or network path
    # is reachable from production preparation.
    actual_exchange_ids = sorted(
        {{"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}[symbol[-2:]] for symbol in market_scope_symbols}
    )
    exchange_projections = {
        exchange: _capture_projection(
            capture_leaves[
                {"SSE": "response-sse.raw", "SZSE": "response-szse.raw", "BSE": "response-bse.raw"}[
                    exchange
                ]
            ]
        )
        for exchange in actual_exchange_ids
    }
    direct_exchange_ids = [
        exchange for exchange in actual_exchange_ids if exchange in {"SSE", "SZSE"}
    ]
    if (
        not direct_exchange_ids
        or ("BSE" in actual_exchange_ids and exchange_projections["BSE"] != [])
        or execution_payload.get("source_limitations") != list(SOURCE_LIMITATIONS)
    ):
        raise FactorGovernanceError("sealed degraded Calendar projection policy differs")
    projection_values = [
        sorted(
            [{"date": row["date"], "status": row["status"]} for row in rows],
            key=lambda row: row["date"],
        )
        for exchange, rows in exchange_projections.items()
        if exchange in direct_exchange_ids
    ]
    if not projection_values or any(rows != projection_values[0] for rows in projection_values[1:]):
        summary = {
            exchange: {
                "rows": len(rows),
                "open": sum(row["status"] == "OPEN" for row in rows),
                "first": rows[0] if rows else None,
                "last": rows[-1] if rows else None,
            }
            for exchange, rows in exchange_projections.items()
        }
        raise FactorGovernanceError(
            "sealed exchange Calendar projections differ: "
            + json.dumps(summary, sort_keys=True, separators=(",", ":"))
        )
    direct_rows = projection_values[0]
    runtime = []
    for row in direct_rows:
        if row["date"] < "2024-01-01":
            continue
        opened = row["status"] == "OPEN"
        session = date.fromisoformat(row["date"])
        runtime.append(
            {
                "date": row["date"],
                "status": row["status"],
                "opens_at_utc": (f"{session.isoformat()}T01:30:00+00:00" if opened else None),
                "closes_at_utc": (f"{session.isoformat()}T07:00:00+00:00" if opened else None),
            }
        )
    open_dates = [row["date"] for row in runtime if row["status"] == "OPEN"]
    if (
        len(open_dates) < 391
        or not open_dates
        or open_dates[-1] != f"{cutoff[:4]}-{cutoff[4:6]}-{cutoff[6:]}"
    ):
        raise FactorGovernanceError("sealed calendar runtime does not close Market cutoff")
    runtime_json = runtime_json_bytes(runtime)
    runtime_parquet = runtime_parquet_bytes(runtime)
    runtime_json_ref = {
        "relative_path": "calendar-runtime/exchange-calendar.json",
        "byte_sha256": _sha256(runtime_json),
    }
    runtime_parquet_ref = {
        "relative_path": "calendar-runtime/exchange-calendar.parquet",
        "byte_sha256": _sha256(runtime_parquet),
    }
    runtime_json_source_ref = stage(
        runtime_json_ref["relative_path"],
        runtime_json,
        source_object_id="factor-calendar-runtime-json",
        source_format="JSON",
        media_type="application/json",
    )
    runtime_parquet_source_ref = stage(
        runtime_parquet_ref["relative_path"],
        runtime_parquet,
        source_object_id="factor-calendar-runtime-parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    calendar_rows.extend(
        [
            {"role": "runtime-json", "source_ref": runtime_json_source_ref},
            {"role": "runtime-parquet", "source_ref": runtime_parquet_source_ref},
        ]
    )
    raw_by_ref[canonical_json_bytes(runtime_json_ref)] = runtime_json
    raw_by_ref[canonical_json_bytes(runtime_parquet_ref)] = runtime_parquet
    calendar_rows.sort(key=lambda row: row["role"])
    if release_install_input_source_ref is None:
        raise FactorGovernanceError("Calendar release-install input source is absent")

    # The compiler's Market-date assertion is intentionally minimal here: the
    # materializer below rechecks every trailing session and deep replay binds
    # the complete resulting strict Market table to this compilation.
    compilation = build_trusted_provider_calendar_compilation(
        compilation_id=operation_id + "-calendar",
        policy=policy,
        capability=capability,
        capture_documents=captures,
        docs_raw=capture_leaves["documentation.raw"],
        raw_resolver=capture_raw_resolver,
        release_ref=release_ref,
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
        market_session_dates=open_dates[-_REQUIRED_OPEN_SESSIONS:],
        cutoff_date=open_dates[-1],
        calendar_json_file_ref=runtime_json_ref,
        calendar_parquet_file_ref=runtime_parquet_ref,
        created_at=prepared_at,
    )
    compilation_ref = store.put_object(compilation)
    calendar_bundle = store.put_object(
        {
            **compilation,
        }
    )
    del calendar_bundle
    calendar_source_bundle = store.put_object(
        __import__("quant_investor.contracts", fromlist=["seal_artifact"]).seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": operation_id + "-calendar-sources",
                "state": "IMMUTABLE",
                "sources": calendar_rows,
            },
            created_at=prepared_at,
        )
    )
    calendar_table = pq.ParquetFile(pa.BufferReader(runtime_parquet)).read().to_pandas()
    sessions = list(calendar_table["open_session"])[-_REQUIRED_OPEN_SESSIONS:]
    if len(sessions) != _REQUIRED_OPEN_SESSIONS or sessions[-1].strftime("%Y%m%d") != cutoff:
        raise FactorGovernanceError("sealed calendar lacks the required Factor window")
    market_rows, pit_rows = _factor_market_rows(
        reader,
        records={symbol: pit_binding["records"][symbol] for symbol in market_scope_symbols},
        sessions=sessions,
        as_of=cutoff,
    )
    if not market_rows:
        raise FactorGovernanceError("canonical Market has no factor rows")

    def parquet_bytes(rows: list[dict[str, Any]], role: str) -> bytes:
        sink = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pylist(rows, schema=role_schema(role)),
            sink,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            version="2.6",
        )
        return sink.getvalue().to_pybytes()

    market_rows.sort(key=lambda row: (row["trade_date"], row["symbol"].encode("utf-8")))
    pit_rows.sort(key=lambda row: (row["signal_session"], row["symbol"].encode("utf-8")))
    factor_market_raw = parquet_bytes(market_rows, "market_history")
    factor_pit_raw = parquet_bytes(pit_rows, "pit_universe")
    factor_market_ref = stage(
        "factor/market-history.parquet",
        factor_market_raw,
        source_object_id="factor-market-history",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    factor_pit_ref = stage(
        "factor/pit-universe.parquet",
        factor_pit_raw,
        source_object_id="factor-pit-universe",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_source_bundle = store.put_object(
        __import__("quant_investor.contracts", fromlist=["seal_artifact"]).seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": operation_id + "-market-sources",
                "state": "IMMUTABLE",
                "sources": [
                    {"role": "factor-market-history", "source_ref": factor_market_ref},
                    {"role": "market-scope", "source_ref": market_scope_ref},
                ],
            },
            created_at=prepared_at,
        )
    )
    pit_source_bundle = store.put_object(
        __import__("quant_investor.contracts", fromlist=["seal_artifact"]).seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": operation_id + "-pit-sources",
                "state": "IMMUTABLE",
                "sources": [
                    {"role": "bound-pointer", "source_ref": bound_pointer_ref},
                    {"role": "factor-pit-universe", "source_ref": factor_pit_ref},
                    {"role": "membership", "source_ref": canonical_pit_ref},
                    {"role": "observed-pointer", "source_ref": observed_ref},
                    {"role": "pit-manifest", "source_ref": pit_manifest_ref},
                ],
            },
            created_at=prepared_at,
        )
    )
    factor_source_bundle = store.put_object(
        __import__("quant_investor.contracts", fromlist=["seal_artifact"]).seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": operation_id + "-factor-sources",
                "state": "IMMUTABLE",
                "sources": [
                    {"role": "exchange_calendar", "source_ref": calendar_source_bundle},
                    {"role": "market_history", "source_ref": market_source_bundle},
                    {"role": "pit_universe", "source_ref": pit_source_bundle},
                ],
            },
            created_at=prepared_at,
        )
    )
    market_input = build_factor_production_market_input(
        market_pit_selection=selection,
        market_pointer_source=store.get_object(market_pointer_ref),
        market_snapshot_manifest_source=store.get_object(market_manifest_ref),
        market_scope_source=store.get_object(market_scope_ref),
        market_history_source=store.get_object(factor_market_ref),
        market_pointer_raw=pointer_raw,
        market_snapshot_manifest_raw=manifest_raw,
        created_at=prepared_at,
    )
    market_input_ref = store.put_object(market_input)
    legacy_ref = store.put_object(legacy)
    recomputation = recompute_factor_production_signals(
        exchange_calendar_path=source_root / runtime_parquet_ref["relative_path"],
        pit_universe_path=source_root / "factor/pit-universe.parquet",
        market_history_path=source_root / "factor/market-history.parquet",
        exchange_calendar_sha256=runtime_parquet_ref["byte_sha256"],
        pit_universe_sha256=_sha256(factor_pit_raw),
        market_history_sha256=_sha256(factor_market_raw),
        as_of=cutoff,
    )

    implementation_component_refs: dict[str, dict[str, str]] = {}
    implementation_rows: list[dict[str, Any]] = []
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        row = installed_semantic_row(factor_id)
        component_ref = store.build_installed_component(
            component_id=row["implementation_id"],
            component_role="SOURCE_IMPLEMENTATION",
            package_name="quant_investor.factors.governance",
            module_names=[row["module_name"]],
            entrypoint_specs=[(row["module_name"], row["qualified_name"])],
            release_manifest_ref=release_ref,
            created_at=prepared_at,
        )
        implementation_component_refs[factor_id] = component_ref
        implementation_rows.append({**row, "implementation_component_ref": component_ref})
    implementation_rows.sort(key=lambda row: row["factor_id"].encode("utf-8"))
    implementation_refs = sorted(
        implementation_component_refs.values(),
        key=lambda ref: (
            ref["kind"],
            ref["contract_sha256"],
            ref["artifact_id"],
            ref["semantic_sha256"],
            ref["byte_sha256"],
        ),
    )

    def one_source_bundle(role: str, inner_role: str, ref: Mapping[str, Any]) -> dict[str, str]:
        return store.put_object(
            seal_artifact(
                "system.source_bundle",
                {
                    "source_bundle_id": f"{operation_id}-bootstrap-{role}",
                    "state": "IMMUTABLE",
                    "sources": [{"role": inner_role, "source_ref": validate_object_ref(ref)}],
                },
                created_at=prepared_at,
            )
        )

    bootstrap_market_bundle_ref = one_source_bundle("market", "market", factor_market_ref)

    implementation_raw = canonical_json_bytes(
        {
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": implementation_rows,
        }
    )
    recomputation_raw = canonical_json_bytes(
        {
            "authority": "NON_AUTHORIZING",
            "domain": "myquant-bootstrap-recomputation",
            "result": "EXACT_MATCH",
            "recomputation": recomputation,
            "source_sha256s": {
                "exchange_calendar": store.get_object(runtime_parquet_source_ref)["payload"][
                    "byte_sha256"
                ],
                "market_history": store.get_object(factor_market_ref)["payload"]["byte_sha256"],
                "pit_universe": store.get_object(factor_pit_ref)["payload"]["byte_sha256"],
            },
        }
    )
    source_generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": {
            "reader": "MarketDataReader",
            "market": "CN",
            "mode_policy": "strict",
            "source_format": "PARQUET",
            "fallback_allowed": False,
        },
        "source_rows": sorted(
            [
                {
                    "role": role,
                    "source_ref": ref,
                    "source_byte_sha256": store.get_object(ref)["payload"]["byte_sha256"],
                }
                for role, ref in (
                    ("exchange_calendar", runtime_parquet_source_ref),
                    ("market", factor_market_ref),
                    ("pit_universe", factor_pit_ref),
                )
            ],
            key=lambda row: row["role"].encode("utf-8"),
        ),
    }
    source_generation_raw = canonical_json_bytes(
        {
            **source_generation_body,
            "generation_sha256": _sha256(canonical_json_bytes(source_generation_body)),
        }
    )
    decision_source_ref = stage(
        "operations/unified_cutover/bootstrap-decision.json",
        decision_raw,
        source_object_id="factor-bootstrap-decision",
        source_format="JSON",
        media_type="application/json",
    )
    implementation_source_ref = stage(
        "bootstrap/implementation-tree.json",
        implementation_raw,
        source_object_id="factor-bootstrap-implementation",
        source_format="JSON",
        media_type="application/json",
    )
    bootstrap_recomputation_ref = stage(
        "bootstrap/recomputation.json",
        recomputation_raw,
        source_object_id="factor-bootstrap-recomputation",
        source_format="JSON",
        media_type="application/json",
    )
    source_generation_ref = stage(
        "bootstrap/source-generation.json",
        source_generation_raw,
        source_object_id="factor-bootstrap-source-generation",
        source_format="JSON",
        media_type="application/json",
    )

    factor_store = FactorValidationStore.for_sealed_operation(
        system_store=store,
        trusted_at=prepared_at,
    )
    bootstrap = factor_store.initialize_bootstrap(
        release_ref=release_ref,
        decision_source_bundle_ref=one_source_bundle(
            "decision", "bootstrap_decision", decision_source_ref
        ),
        exchange_calendar_bundle_ref=one_source_bundle(
            "calendar", "calendar", runtime_parquet_source_ref
        ),
        implementation_bundle_ref=one_source_bundle(
            "implementation", "implementation_tree_manifest", implementation_source_ref
        ),
        market_bundle_ref=bootstrap_market_bundle_ref,
        pit_universe_bundle_ref=one_source_bundle("pit", "pit", factor_pit_ref),
        recomputation_bundle_ref=one_source_bundle(
            "recomputation", "recomputation", bootstrap_recomputation_ref
        ),
        source_generation_bundle_ref=one_source_bundle(
            "source-generation", "source_generation", source_generation_ref
        ),
    )
    policy_ref = bootstrap.policy_ref
    active_ref = bootstrap.active_set_ref
    attestation_ref = bootstrap.intrinsic_receipt_ref
    source_closure = build_factor_production_source_closure(
        deployed_release_ref=release_ref,
        release_install_evidence_ref=release_install_evidence_ref,
        release_install_input_source_ref=release_install_input_source_ref,
        release_install_verification=release_install_verification,
        market_pit_selection_ref=selection_ref,
        market_scope_source_ref=market_scope_ref,
        calendar_authority_policy_ref=policy_ref_calendar,
        calendar_compilation_ref=compilation_ref,
        calendar_capture_custody_attestation_ref=calendar_custody_ref,
        factor_source_bundle_ref=factor_source_bundle,
        factor_policy_ref=policy_ref,
        factor_active_set_ref=active_ref,
        factor_validation_attestation_ref=attestation_ref,
        factor_implementation_refs=implementation_refs,
        legacy_zero_call_ref=legacy_ref,
        market_input_ref=market_input_ref,
        created_at=prepared_at,
    )
    source_closure_ref = store.put_object(source_closure)
    evidence = build_factor_production_recomputation_evidence(
        source_closure=source_closure,
        deployed_release_ref=release_ref,
        factor_active_set_ref=active_ref,
        recomputation=recomputation,
        created_at=prepared_at,
    )
    evidence_ref = store.put_object(evidence)
    generation = build_factor_production_generation(
        source_closure=source_closure,
        recomputation_evidence=evidence,
        created_at=prepared_at,
    )
    generation_ref = store.put_object(generation)
    result = {
        "status": "PREPARED",
        "operation_id": operation_id,
        "operation_inputs_sha256": operation_inputs_sha256,
        "operation_inputs_ref": operation_inputs_ref,
        "as_of": cutoff,
        "source_root": f"results/factors/preparations/{operation_id}/sources",
        "source_root_id": source_root_id,
        "release_repository_root": execution_payload["release_repository_root"],
        "deployed_release_ref": release_ref,
        "release_install_evidence_ref": release_install_evidence_ref,
        "release_install_input_source_ref": release_install_input_source_ref,
        "legacy_zero_call_ref": legacy_ref,
        "market_snapshot_id": selection["payload"]["market_snapshot_id"],
        "market_pit_selection_ref": selection_ref,
        "calendar_authority_policy_ref": policy_ref_calendar,
        "calendar_compilation_ref": compilation_ref,
        "calendar_capture_custody_attestation_ref": calendar_custody_ref,
        "factor_source_bundle_ref": factor_source_bundle,
        "market_scope_source_ref": market_scope_ref,
        "market_input_ref": market_input_ref,
        "factor_production_source_closure_ref": source_closure_ref,
        "factor_production_recomputation_ref": evidence_ref,
        "factor_production_generation_ref": generation_ref,
        "low_signal_sha256": recomputation["low_signal_sha256"],
        "w80_signal_sha256": recomputation["w80_signal_sha256"],
        "exact_replay_sha256": recomputation["exact_replay_sha256"],
        "factor_readiness": "READY",
        "factor_authority": "INACTIVE",
        "system_pointer_writes": 0,
        "factor_pointer_writes": 0,
        "broker_order_trade_fund_writes": 0,
    }
    _write_source(staging_root / _PREPARED_RECEIPT_NAME, canonical_json_bytes(result))
    parent_fd: int | None = None
    try:
        parent_fd = os.open(
            preparations_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        _rename_no_replace(parent_fd, staging_root.name, final_root.name)
        os.fsync(parent_fd)
    except Exception as exc:
        if final_root.exists():
            observed = _validate_existing_preparation(
                workspace=workspace,
                operation_id=operation_id,
                source_root_id=source_root_id,
            )
            if observed == result:
                return observed
        raise FactorGovernanceError("Factor preparation atomic publication failed") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    return _validate_existing_preparation(
        workspace=workspace,
        operation_id=operation_id,
        source_root_id=source_root_id,
    )


__all__ = ["prepare_factor_production"]
