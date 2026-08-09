"""Bounded execution of the sealed Fundamental VIP partition plan."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any

import pandas as pd

from quant_investor.v17_v4_runtime.tushare_https import TushareHttpsError

from ...._core import canonical_bytes, content_ref
from ..contracts import response_projection_sha256
from ..models import TushareRequestClient
from .contracts import (
    _build_physical_request_receipt_validated,
    _validate_physical_request_receipt_validated,
)
from .models import FundamentalV4ContractError, SOURCE_TABLES
from .schedule import validate_fundamental_execution_closure_v4


def _private_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise FundamentalV4ContractError("VIP checkpoint directory is unsafe")


def _checkpoint_root(
    value: str | Path,
    *,
    execution_closure: Mapping[str, Any],
) -> Path:
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise FundamentalV4ContractError("VIP checkpoint root must be absolute")
    execution_bytes = canonical_bytes(dict(execution_closure))
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        _private_directory(root)
    except OSError as exc:
        raise FundamentalV4ContractError("VIP checkpoint creation failed") from exc
    records = root / "partition_records"
    try:
        os.mkdir(records, 0o700)
    except FileExistsError:
        _private_directory(records)
    closure_path = root / "execution_closure.json"
    if closure_path.exists():
        if _read_private_file(closure_path) != execution_bytes:
            raise FundamentalV4ContractError("VIP checkpoint closure changed")
    else:
        _write_exact(closure_path, execution_bytes)
    _validate_checkpoint_names(root, records)
    return root


def _validate_checkpoint_names(root: Path, records: Path) -> None:
    root_names = sorted(item.name for item in os.scandir(root))
    if root_names != ["execution_closure.json", "partition_records"]:
        raise FundamentalV4ContractError("VIP checkpoint root contains unknown files")
    names = [item.name for item in os.scandir(records)]
    committed = [name for name in names if not name.startswith(".partial.")]
    if len(committed) != len({name.casefold() for name in committed}) or any(
        len(name) != 11 or not name.endswith(".json") or not name[:6].isdigit()
        for name in committed
    ):
        raise FundamentalV4ContractError("VIP checkpoint record names are invalid")


def _read_private_file(path: Path) -> bytes:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
    ):
        raise FundamentalV4ContractError("VIP checkpoint file is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_ino, opened.st_dev, opened.st_size, opened.st_mtime_ns) != (
        after.st_ino,
        after.st_dev,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise FundamentalV4ContractError("VIP checkpoint file changed during read")
    return b"".join(chunks)


def _write_exact(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FundamentalV4ContractError("VIP checkpoint write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if _read_private_file(path) != payload:
        raise FundamentalV4ContractError("VIP checkpoint readback failed")


def _atomic_record(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".partial.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FundamentalV4ContractError("VIP checkpoint write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.exists():
        raise FundamentalV4ContractError("VIP checkpoint partition already exists")
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if _read_private_file(path) != payload:
        raise FundamentalV4ContractError("VIP checkpoint partition readback failed")


def _scalar(value: Any) -> list[Any]:
    if value is None:
        return ["NULL", None]
    if type(value) is bool:
        return ["BOOL", value]
    if type(value) is int:
        return ["INT", str(value)]
    if type(value) is Decimal and value.is_finite():
        return ["DECIMAL", format(value, "f")]
    if type(value) is str:
        return ["TEXT", value]
    raise FundamentalV4ContractError("VIP checkpoint row scalar is invalid")


def _restore_scalar(value: Any) -> Any:
    if not isinstance(value, list) or len(value) != 2:
        raise FundamentalV4ContractError("VIP checkpoint scalar shape is invalid")
    kind, scalar = value
    if kind == "NULL" and scalar is None:
        return None
    if kind == "BOOL" and type(scalar) is bool:
        return scalar
    if kind == "INT" and type(scalar) is str:
        return int(scalar)
    if kind == "DECIMAL" and type(scalar) is str:
        result = Decimal(scalar)
        if result.is_finite():
            return result
    if kind == "TEXT" and type(scalar) is str:
        return scalar
    raise FundamentalV4ContractError("VIP checkpoint scalar is invalid")


def _record_bytes(
    receipt: Mapping[str, Any],
    rows: tuple[tuple[Any, ...], ...],
) -> bytes:
    return canonical_bytes(
        {
            "receipt": dict(receipt),
            "rows": [[_scalar(value) for value in row] for row in rows],
        }
    )


def _read_record(
    path: Path,
    *,
    plan: Mapping[str, Any],
    plan_ref: Mapping[str, str],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], ...]]:
    raw = _read_private_file(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FundamentalV4ContractError("VIP checkpoint record is invalid") from exc
    if canonical_bytes(value) != raw or set(value) != {"receipt", "rows"}:
        raise FundamentalV4ContractError("VIP checkpoint record is not canonical")
    receipt = _validate_physical_request_receipt_validated(
        value["receipt"],
        validated_plan=plan,
        validated_plan_ref=plan_ref,
        endpoints=endpoint_plans,
    )
    rows = tuple(tuple(_restore_scalar(scalar) for scalar in row) for row in value["rows"])
    projection = receipt["raw_response_projection_sha256"]
    if (projection is None and rows) or (
        projection is not None and response_projection_sha256(rows) != projection
    ):
        raise FundamentalV4ContractError("VIP checkpoint row closure mismatch")
    return receipt, rows


def _params(
    endpoint_plan: Mapping[str, Any],
    *,
    partition_id: str,
) -> dict[str, Any]:
    dimension, value = partition_id.split("=", 1)
    if endpoint_plan["partition_dimensions"] != [dimension]:
        raise FundamentalV4ContractError("endpoint partition dimension mismatch")
    fixed = dict(endpoint_plan["fixed_params"])
    if dimension in fixed:
        raise FundamentalV4ContractError("partition dimension must not be fixed")
    return {**fixed, dimension: value}


def _assert_endpoint_keysets(
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> None:
    for table in SOURCE_TABLES:
        endpoint = endpoint_plans[table]
        expected = [row["partition_id"] for row in plan["partition_rows"] if row["table"] == table]
        if (
            endpoint["ordered_expected_partition_keyset"] != expected
            or endpoint["planned_terminal_request_count"] != len(expected)
            or endpoint["max_attempts"] != plan["max_attempts_per_partition"]
            or endpoint["planned_max_network_attempts"]
            != len(expected) * plan["max_attempts_per_partition"]
        ):
            raise FundamentalV4ContractError("endpoint partition keyset is incomplete")


def _response_status(
    *,
    response: Any,
    table: str,
    partition_id: str,
    plan: Mapping[str, Any],
    endpoint_plan: Mapping[str, Any],
) -> tuple[str, list[str], tuple[tuple[Any, ...], ...]]:
    if (
        response.api_name != endpoint_plan["api_name"]
        or tuple(response.fields) != tuple(endpoint_plan["expected_fields"])
        or response.reported_count != len(response.rows)
    ):
        return "SCHEMA_MISMATCH", [], ()
    rows = tuple(response.rows)
    blockers: list[str] = []
    if len(rows) != len(set(rows)):
        blockers.append("DUPLICATE_ROWS")
    if response.has_more:
        blockers.append("HAS_MORE")
    if len(rows) >= endpoint_plan["documented_row_limit"]:
        blockers.append("ROW_LIMIT_HIT")
    fields = list(response.fields)
    try:
        symbol_index = fields.index("ts_code")
        partition_index = fields.index("trade_date" if table == "daily_basic" else "end_date")
    except ValueError:
        return "SCHEMA_MISMATCH", [], ()
    partition_value = partition_id.split("=", 1)[1]
    symbols = set(plan["symbols"])
    if any(
        row[symbol_index] not in symbols or row[partition_index] != partition_value for row in rows
    ):
        blockers.append("SCOPE_MISMATCH")
    if not rows and f"{table}|{partition_id}" not in plan["baseline_empty_partition_keyset"]:
        blockers.append("SCOPE_MISMATCH")
    if blockers:
        return "INCOMPLETE", sorted(set(blockers)), rows
    return ("EMPTY" if not rows else "AVAILABLE"), [], rows


def _failure_status(error: Exception) -> str:
    if isinstance(error, TushareHttpsError):
        if error.code == "TUSHARE_API_ERROR":
            return "PROVIDER_ERROR"
        if error.code == "TUSHARE_RESPONSE_INVALID":
            return "SCHEMA_MISMATCH"
    return "TRANSPORT_ERROR"


def _fetch_partition(
    *,
    client: TushareRequestClient,
    plan: Mapping[str, Any],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
    plan_ref: Mapping[str, str],
    partition: Mapping[str, Any],
    captured_at: str,
    sleeper: Callable[[float], None],
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], ...]]:
    table = partition["table"]
    endpoint = endpoint_plans[table]
    partition_id = partition["partition_id"]
    params = _params(endpoint, partition_id=partition_id)
    params_sha = hashlib.sha256(canonical_bytes(params)).hexdigest()
    response = None
    status = "TRANSPORT_ERROR"
    attempts = 0
    for attempts, delay in enumerate(endpoint["retry_schedule"], start=1):
        if delay:
            sleeper(float(delay))
        try:
            response = client.request(
                api_name=endpoint["api_name"],
                params=params,
                expected_fields=endpoint["expected_fields"],
            )
        except Exception as error:
            status = _failure_status(error)
            continue
        status, blockers, rows = _response_status(
            response=response,
            table=table,
            partition_id=partition_id,
            plan=plan,
            endpoint_plan=endpoint,
        )
        receipt = _build_physical_request_receipt_validated(
            validated_plan=plan,
            validated_plan_ref=plan_ref,
            endpoints=endpoint_plans,
            table=table,
            partition_id=partition_id,
            sanitized_params_sha256=params_sha,
            attempts=attempts,
            provider_request_id=response.request_id,
            reported_count=response.reported_count,
            accepted_count=len(rows),
            has_more=response.has_more,
            status=status,
            blocker_codes=blockers,
            raw_response_projection_sha256=response_projection_sha256(rows),
            captured_at=captured_at,
        )
        return receipt, rows
    receipt = _build_physical_request_receipt_validated(
        validated_plan=plan,
        validated_plan_ref=plan_ref,
        endpoints=endpoint_plans,
        table=table,
        partition_id=partition_id,
        sanitized_params_sha256=params_sha,
        attempts=attempts,
        provider_request_id=None,
        reported_count=0,
        accepted_count=0,
        has_more=False,
        status=status,
        blocker_codes=[],
        raw_response_projection_sha256=None,
        captured_at=captured_at,
    )
    return receipt, ()


def _frames(
    rows_by_table: Mapping[str, list[tuple[Any, ...]]],
    endpoint_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for table in SOURCE_TABLES:
        fields = list(endpoint_plans[table]["expected_fields"])
        frame = pd.DataFrame(rows_by_table[table], columns=fields)
        for field in fields:
            if any(type(value) is Decimal for value in frame[field].dropna().tolist()):
                frame[field] = frame[field].map(
                    lambda value: value if value is None else Decimal(str(value))
                )
        result[table] = frame
    return result


def acquire_fundamental_vip_v4(
    *,
    execution_closure: Mapping[str, Any],
    client: TushareRequestClient,
    captured_at: str,
    checkpoint_root: str | Path | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute every sealed partition once to a terminal physical receipt."""

    execution = validate_fundamental_execution_closure_v4(execution_closure)
    plan = execution["request_plan"]
    endpoint_plans = execution["endpoint_plans"]
    _assert_endpoint_keysets(plan, endpoint_plans)
    checkpoint = (
        None
        if checkpoint_root is None
        else _checkpoint_root(
            checkpoint_root,
            execution_closure=execution,
        )
    )
    receipts: list[dict[str, Any]] = []
    rows_by_table: dict[str, list[tuple[Any, ...]]] = {table: [] for table in SOURCE_TABLES}
    plan_ref = content_ref(plan, identity_field="plan_id")
    attempts = 0
    for partition in plan["partition_rows"]:
        record_path = (
            None
            if checkpoint is None
            else checkpoint / "partition_records" / f"{partition['ordinal']:06d}.json"
        )
        if record_path is not None and record_path.exists():
            receipt, rows = _read_record(
                record_path,
                plan=plan,
                plan_ref=plan_ref,
                endpoint_plans=endpoint_plans,
            )
            if (
                receipt["table"] != partition["table"]
                or receipt["partition_id"] != partition["partition_id"]
            ):
                raise FundamentalV4ContractError("VIP checkpoint partition identity mismatch")
        else:
            receipt, rows = _fetch_partition(
                client=client,
                plan=plan,
                endpoint_plans=endpoint_plans,
                plan_ref=plan_ref,
                partition=partition,
                captured_at=captured_at,
                sleeper=sleeper,
            )
            if record_path is not None:
                _atomic_record(record_path, _record_bytes(receipt, rows))
        receipts.append(receipt)
        attempts += receipt["attempts"]
        rows_by_table[partition["table"]].extend(rows)
        if attempts > plan["planned_max_network_attempts"]:
            raise FundamentalV4ContractError("VIP acquisition exceeded attempt bound")
    if len(receipts) != plan["planned_terminal_request_count"]:
        raise FundamentalV4ContractError("VIP acquisition terminal keyset is incomplete")
    if checkpoint is not None:
        _validate_checkpoint_names(checkpoint, checkpoint / "partition_records")
    return {
        "network_attempts": attempts,
        "physical_receipts": tuple(receipts),
        "raw_tables": _frames(rows_by_table, endpoint_plans),
        "status": (
            "COMPLETE"
            if all(row["status"] in {"AVAILABLE", "EMPTY"} for row in receipts)
            else "BLOCKED"
        ),
    }


__all__ = ["acquire_fundamental_vip_v4"]
