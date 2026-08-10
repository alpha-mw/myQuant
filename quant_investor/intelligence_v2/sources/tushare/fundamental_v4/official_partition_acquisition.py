"""Execute the sealed official-parameter plan without changing legacy v4 receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

import pandas as pd

from quant_investor.market.fundamental_provider_contract import frame_fingerprint
from quant_investor.v17_v4_runtime.tushare_https import TushareHttpsError

from ...._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from ..contracts import response_projection_sha256
from ..models import TushareRequestClient
from .acquisition import (
    _atomic_record,
    _frames,
    _private_directory,
    _read_private_file,
    _restore_scalar,
    _scalar,
    _write_exact,
)
from .comparison import (
    compare_fundamental_raw_tables,
    validate_fundamental_comparison_policy,
)
from .models import FundamentalV4ContractError, SOURCE_TABLES, fundamental_v4_contract
from .official_partition_plan import validate_official_partition_execution_plan
from .schedule import validate_fundamental_execution_closure_v4

OFFICIAL_PARTITION_REQUEST_RECEIPT_V1: Final = (
    "myquant.v17.fundamental-official-partition-request-receipt.v1"
)

_COMMON_FIELDS = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
_RECEIPT_FIELDS = _COMMON_FIELDS | {
    "accepted_count",
    "attempts",
    "baseline_partition_key",
    "blocker_codes",
    "endpoint",
    "has_more",
    "ordinal",
    "params_sha256",
    "partition_id",
    "plan_ref",
    "provider_request_id",
    "raw_response_projection_sha256",
    "receipt_id",
    "reported_count",
    "request_key",
    "status",
    "table",
}
_STATUSES = {
    "AVAILABLE",
    "EMPTY",
    "INCOMPLETE",
    "PROVIDER_ERROR",
    "SCHEMA_MISMATCH",
    "TRANSPORT_ERROR",
}
_BLOCKERS = {
    "DUPLICATE_ROWS",
    "HAS_MORE",
    "LOCAL_RESPONSE_LIMIT_EXCEEDED",
    "ROW_LIMIT_HIT",
    "SCOPE_MISMATCH",
}


def _codes(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FundamentalV4ContractError("official request blockers must be a sequence")
    rows = list(values)
    if any(type(value) is not str or value not in _BLOCKERS for value in rows):
        raise FundamentalV4ContractError("official request blocker is invalid")
    expected = sorted(rows, key=lambda value: value.encode("ascii"))
    if rows != expected or len(rows) != len(set(rows)):
        raise FundamentalV4ContractError("official request blockers are not canonical")
    return rows


def _baseline_partition_key(request: Mapping[str, Any]) -> str:
    params = request["params"]
    if request["table"] == "daily_basic":
        dimension = "trade_date"
    elif "ann_date" in params:
        dimension = "ann_date"
    else:
        dimension = "period"
    return f"{dimension}={params[dimension]}"


def _validate_receipt_status(
    *,
    status: str,
    reported_count: int,
    accepted_count: int,
    has_more: bool,
    blockers: list[str],
    projection_sha256: str | None,
) -> None:
    if status not in _STATUSES:
        raise FundamentalV4ContractError("official request status is invalid")
    if (
        type(reported_count) is not int
        or reported_count < 0
        or type(accepted_count) is not int
        or accepted_count < 0
        or type(has_more) is not bool
    ):
        raise FundamentalV4ContractError("official request counts are invalid")
    if status == "AVAILABLE" and (
        reported_count != accepted_count
        or accepted_count < 1
        or has_more
        or blockers
        or projection_sha256 is None
    ):
        raise FundamentalV4ContractError("AVAILABLE official request is not closed")
    if status == "EMPTY" and (
        reported_count or accepted_count or has_more or blockers or projection_sha256 is None
    ):
        raise FundamentalV4ContractError("EMPTY official request is not closed")
    if status == "INCOMPLETE" and not blockers:
        raise FundamentalV4ContractError("INCOMPLETE official request lacks a blocker")
    if status in {"PROVIDER_ERROR", "SCHEMA_MISMATCH", "TRANSPORT_ERROR"} and (
        reported_count or accepted_count or has_more or blockers or projection_sha256 is not None
    ):
        raise FundamentalV4ContractError("failed official request claims response data")


def _build_receipt(
    *,
    plan: Mapping[str, Any],
    plan_ref: Mapping[str, str],
    request: Mapping[str, Any],
    provider_request_id: str | None,
    reported_count: int,
    accepted_count: int,
    has_more: bool,
    status: str,
    blocker_codes: Sequence[str],
    raw_response_projection_sha256: str | None,
    captured_at: str,
) -> dict[str, Any]:
    captured = timestamp(captured_at, label="captured_at")
    if captured < plan["created_at"]:
        raise FundamentalV4ContractError("official request receipt predates its plan")
    blockers = _codes(blocker_codes)
    projection = (
        None
        if raw_response_projection_sha256 is None
        else sha256(
            raw_response_projection_sha256,
            label="raw_response_projection_sha256",
        )
    )
    _validate_receipt_status(
        status=status,
        reported_count=reported_count,
        accepted_count=accepted_count,
        has_more=has_more,
        blockers=blockers,
        projection_sha256=projection,
    )
    provider_id = (
        None
        if provider_request_id is None
        else identifier(provider_request_id, label="provider_request_id")
    )
    body = {
        **common_fields(timestamp_value=captured),
        "accepted_count": accepted_count,
        "attempts": 1,
        "baseline_partition_key": _baseline_partition_key(request),
        "blocker_codes": blockers,
        "endpoint": request["endpoint"],
        "has_more": has_more,
        "ordinal": request["ordinal"],
        "params_sha256": hashlib.sha256(canonical_bytes(request["params"])).hexdigest(),
        "partition_id": request["partition_id"],
        "plan_ref": dict(plan_ref),
        "provider_request_id": provider_id,
        "raw_response_projection_sha256": projection,
        "reported_count": reported_count,
        "request_key": request["request_key"],
        "status": status,
        "table": request["table"],
        "version": OFFICIAL_PARTITION_REQUEST_RECEIPT_V1,
    }
    return seal(body, identity_field="receipt_id")


@fundamental_v4_contract
def validate_official_partition_request_receipt(
    document: Mapping[str, Any],
    *,
    official_plan: Mapping[str, Any],
    source_execution_closure: Mapping[str, Any],
    probe_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = validate_official_partition_execution_plan(
        official_plan,
        source_execution_closure=source_execution_closure,
        probe_observations=probe_observations,
    )
    return _validate_receipt_validated(
        document,
        plan=plan,
        plan_ref=content_ref(plan, identity_field="partition_plan_id"),
        requests_by_key={row["request_key"]: row for row in plan["request_rows"]},
    )


def _validate_receipt_validated(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_ref: Mapping[str, str],
    requests_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = validate_seal(value, identity_field="receipt_id")
    require_exact_keys(receipt, _RECEIPT_FIELDS, label="official partition request receipt")
    if receipt.get("version") != OFFICIAL_PARTITION_REQUEST_RECEIPT_V1:
        raise FundamentalV4ContractError("official request receipt version mismatch")
    request = requests_by_key.get(receipt["request_key"])
    if request is None:
        raise FundamentalV4ContractError("official request is not in the sealed plan")
    expected = _build_receipt(
        plan=plan,
        plan_ref=plan_ref,
        request=request,
        provider_request_id=receipt["provider_request_id"],
        reported_count=receipt["reported_count"],
        accepted_count=receipt["accepted_count"],
        has_more=receipt["has_more"],
        status=receipt["status"],
        blocker_codes=receipt["blocker_codes"],
        raw_response_projection_sha256=receipt["raw_response_projection_sha256"],
        captured_at=receipt["timestamp"],
    )
    if receipt != expected:
        raise FundamentalV4ContractError("official request receipt replay mismatch")
    return receipt


def _checkpoint(
    value: str | Path,
    *,
    execution_bundle: Mapping[str, Any],
) -> Path:
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise FundamentalV4ContractError("official checkpoint root must be absolute")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        _private_directory(root)
    records = root / "partition_records"
    try:
        os.mkdir(records, 0o700)
    except FileExistsError:
        _private_directory(records)
    closure = root / "execution_bundle.json"
    payload = canonical_bytes(dict(execution_bundle))
    if closure.exists():
        if _read_private_file(closure) != payload:
            raise FundamentalV4ContractError("official checkpoint execution bundle changed")
    else:
        _write_exact(closure, payload)
    _checkpoint_names(root)
    return root


def _checkpoint_names(root: Path) -> None:
    if sorted(item.name for item in os.scandir(root)) != [
        "execution_bundle.json",
        "partition_records",
    ]:
        raise FundamentalV4ContractError("official checkpoint contains unknown files")
    records = root / "partition_records"
    committed = [item.name for item in os.scandir(records) if not item.name.startswith(".partial.")]
    if len(committed) != len({name.casefold() for name in committed}) or any(
        len(name) != 11 or not name.endswith(".json") or not name[:6].isdigit()
        for name in committed
    ):
        raise FundamentalV4ContractError("official checkpoint record names are invalid")


def _record_bytes(receipt: Mapping[str, Any], rows: tuple[tuple[Any, ...], ...]) -> bytes:
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
    requests_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], ...]]:
    raw = _read_private_file(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FundamentalV4ContractError("official checkpoint record is invalid") from exc
    if canonical_bytes(value) != raw or set(value) != {"receipt", "rows"}:
        raise FundamentalV4ContractError("official checkpoint record is not canonical")
    receipt = _validate_receipt_validated(
        value["receipt"],
        plan=plan,
        plan_ref=plan_ref,
        requests_by_key=requests_by_key,
    )
    rows = tuple(tuple(_restore_scalar(item) for item in row) for row in value["rows"])
    projection = receipt["raw_response_projection_sha256"]
    if (projection is None and rows) or (
        projection is not None and response_projection_sha256(rows) != projection
    ):
        raise FundamentalV4ContractError("official checkpoint row closure mismatch")
    return receipt, rows


def _row_in_scope(
    *,
    request: Mapping[str, Any],
    indices: Mapping[str, int],
    row: tuple[Any, ...],
) -> bool:
    params = request["params"]
    symbol = row[indices["ts_code"]]
    if type(symbol) is not str or not symbol or not symbol.isascii():
        return False
    if request["table"] == "daily_basic":
        return row[indices["trade_date"]] == params["trade_date"]
    if "ann_date" in params:
        announced = row[indices["ann_date"]]
        period = row[indices["end_date"]]
        return (
            announced == params["ann_date"]
            and type(period) is str
            and params["start_date"] <= period <= params["end_date"]
        )
    if row[indices["end_date"]] != params["period"]:
        return False
    if "start_date" not in params:
        return True
    announced = row[indices["ann_date"]]
    return type(announced) is str and params["start_date"] <= announced <= params["end_date"]


def _response_scope_blockers(
    *,
    request: Mapping[str, Any],
    fields: Sequence[str],
    rows: tuple[tuple[Any, ...], ...],
) -> list[str]:
    blockers: set[str] = set()
    try:
        duplicate = len(rows) != len(set(rows))
    except TypeError:
        return ["SCOPE_MISMATCH"]
    duplicate_mode = request["exact_duplicate_mode"]
    if duplicate and duplicate_mode == "REJECT_EXACT_DUPLICATES":
        blockers.add("DUPLICATE_ROWS")
    indices = {field: index for index, field in enumerate(fields)}
    required = {"ts_code", "trade_date" if request["table"] == "daily_basic" else "end_date"}
    if (
        request["table"] in {"balancesheet", "cashflow", "income"}
        or "ann_date" in request["params"]
    ):
        required.add("ann_date")
    if not required.issubset(indices) or any(len(row) != len(fields) for row in rows):
        return ["SCOPE_MISMATCH"]
    if any(not _row_in_scope(request=request, indices=indices, row=row) for row in rows):
        blockers.add("SCOPE_MISMATCH")
    return sorted(blockers, key=lambda value: value.encode("ascii"))


def _failure_status(error: Exception) -> str:
    if isinstance(error, TushareHttpsError):
        if error.code == "TUSHARE_API_ERROR":
            return "PROVIDER_ERROR"
        if error.code == "TUSHARE_RESPONSE_INVALID":
            return "SCHEMA_MISMATCH"
    return "TRANSPORT_ERROR"


def _fetch(
    *,
    client: TushareRequestClient,
    plan: Mapping[str, Any],
    plan_ref: Mapping[str, str],
    request: Mapping[str, Any],
    expected_fields: Sequence[str],
    captured_at: str,
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], ...]]:
    try:
        response = client.request(
            api_name=request["endpoint"],
            params=request["params"],
            expected_fields=expected_fields,
        )
    except Exception as error:
        receipt = _build_receipt(
            plan=plan,
            plan_ref=plan_ref,
            request=request,
            provider_request_id=None,
            reported_count=0,
            accepted_count=0,
            has_more=False,
            status=_failure_status(error),
            blocker_codes=[],
            raw_response_projection_sha256=None,
            captured_at=captured_at,
        )
        return receipt, ()
    if (
        response.api_name != request["endpoint"]
        or tuple(response.fields) != tuple(expected_fields)
        or response.reported_count != len(response.rows)
    ):
        receipt = _build_receipt(
            plan=plan,
            plan_ref=plan_ref,
            request=request,
            provider_request_id=None,
            reported_count=0,
            accepted_count=0,
            has_more=False,
            status="SCHEMA_MISMATCH",
            blocker_codes=[],
            raw_response_projection_sha256=None,
            captured_at=captured_at,
        )
        return receipt, ()
    rows = tuple(response.rows)
    blockers = _response_scope_blockers(
        request=request,
        fields=response.fields,
        rows=rows,
    )
    if response.has_more:
        blockers.append("HAS_MORE")
    official_limit = request["official_row_limit"]
    if official_limit is not None and len(rows) >= official_limit:
        blockers.append("ROW_LIMIT_HIT")
    if len(rows) > request["local_max_response_items"]:
        blockers.append("LOCAL_RESPONSE_LIMIT_EXCEEDED")
        stored_rows: tuple[tuple[Any, ...], ...] = ()
        projection = None
        accepted = 0
    else:
        stored_rows = rows
        projection = response_projection_sha256(rows)
        accepted = len(rows)
    blockers = sorted(set(blockers), key=lambda value: value.encode("ascii"))
    status = "INCOMPLETE" if blockers else ("EMPTY" if not rows else "AVAILABLE")
    receipt = _build_receipt(
        plan=plan,
        plan_ref=plan_ref,
        request=request,
        provider_request_id=response.request_id,
        reported_count=response.reported_count,
        accepted_count=accepted,
        has_more=response.has_more,
        status=status,
        blocker_codes=blockers,
        raw_response_projection_sha256=projection,
        captured_at=captured_at,
    )
    return receipt, stored_rows


def _baseline_fingerprints(
    values: Mapping[str, str],
    *,
    baseline_tables: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    if type(values) is not dict or set(values) != set(SOURCE_TABLES):
        raise FundamentalV4ContractError("baseline fingerprint set is invalid")
    if type(baseline_tables) is not dict or set(baseline_tables) != set(SOURCE_TABLES):
        raise FundamentalV4ContractError("baseline table set is invalid")
    result: dict[str, str] = {}
    for table in SOURCE_TABLES:
        expected = sha256(values[table], label=f"{table} baseline fingerprint")
        frame = baseline_tables[table]
        if not isinstance(frame, pd.DataFrame) or frame_fingerprint(frame) != expected:
            raise FundamentalV4ContractError("baseline frame fingerprint mismatch")
        result[table] = expected
    return result


@fundamental_v4_contract
def acquire_official_partition_fundamental_vip_v4(
    *,
    official_plan: Mapping[str, Any],
    source_execution_closure: Mapping[str, Any],
    probe_observations: Sequence[Mapping[str, Any]],
    baseline_tables: Mapping[str, pd.DataFrame],
    baseline_table_fingerprints: Mapping[str, str],
    comparison_policy: Mapping[str, Any],
    client: TushareRequestClient,
    captured_at: str,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute each new request once and require exact baseline reconciliation."""

    source = validate_fundamental_execution_closure_v4(source_execution_closure)
    plan = validate_official_partition_execution_plan(
        official_plan,
        source_execution_closure=source,
        probe_observations=probe_observations,
    )
    policy = validate_fundamental_comparison_policy(comparison_policy)
    fingerprints = _baseline_fingerprints(
        baseline_table_fingerprints,
        baseline_tables=baseline_tables,
    )
    captured = timestamp(captured_at, label="captured_at")
    if captured < plan["created_at"]:
        raise FundamentalV4ContractError("official acquisition predates its plan")
    endpoint_plans = source["endpoint_plans"]
    bundle = {
        "baseline_table_fingerprints": fingerprints,
        "captured_at": captured,
        "comparison_policy": policy,
        "official_plan": plan,
        "source_execution_closure": source,
    }
    checkpoint = (
        None if checkpoint_root is None else _checkpoint(checkpoint_root, execution_bundle=bundle)
    )
    receipts: list[dict[str, Any]] = []
    rows_by_table: dict[str, list[tuple[Any, ...]]] = {table: [] for table in SOURCE_TABLES}
    transport_calls = 0
    plan_ref = content_ref(plan, identity_field="partition_plan_id")
    requests_by_key = {row["request_key"]: row for row in plan["request_rows"]}
    for request in plan["request_rows"]:
        record_path = (
            None
            if checkpoint is None
            else checkpoint / "partition_records" / f"{request['ordinal']:06d}.json"
        )
        if record_path is not None and record_path.exists():
            receipt, rows = _read_record(
                record_path,
                plan=plan,
                plan_ref=plan_ref,
                requests_by_key=requests_by_key,
            )
            if receipt["request_key"] != request["request_key"]:
                raise FundamentalV4ContractError("official checkpoint request identity mismatch")
        else:
            receipt, rows = _fetch(
                client=client,
                plan=plan,
                plan_ref=plan_ref,
                request=request,
                expected_fields=endpoint_plans[request["table"]]["expected_fields"],
                captured_at=captured,
            )
            transport_calls += 1
            if record_path is not None:
                _atomic_record(record_path, _record_bytes(receipt, rows))
        receipts.append(receipt)
        rows_by_table[request["table"]].extend(rows)
        if receipt["status"] not in {"AVAILABLE", "EMPTY"}:
            break
    terminal_complete = len(receipts) == plan["planned_terminal_request_count"]
    if checkpoint is not None:
        _checkpoint_names(checkpoint)
    raw_tables = _frames(rows_by_table, endpoint_plans)
    physical_closed = terminal_complete and all(
        receipt["status"] in {"AVAILABLE", "EMPTY"} for receipt in receipts
    )
    comparison = None
    status = "ACQUISITION_BLOCKED"
    if physical_closed:
        comparison = compare_fundamental_raw_tables(
            baseline_tables=dict(baseline_tables),
            vip_tables=raw_tables,
            policy=policy,
        )
        status = "COMPLETE" if comparison["passed"] else "RECONCILIATION_BLOCKED"
    return {
        "comparison": comparison,
        "physical_receipts": tuple(receipts),
        "raw_tables": raw_tables,
        "receipt_network_attempts": sum(receipt["attempts"] for receipt in receipts),
        "status": status,
        "transport_calls": transport_calls,
    }


__all__ = [
    "OFFICIAL_PARTITION_REQUEST_RECEIPT_V1",
    "acquire_official_partition_fundamental_vip_v4",
    "validate_official_partition_request_receipt",
]
