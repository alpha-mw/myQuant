"""Strict Parquet source replay for prospective Factor governance.

Only exact ``system.source_object`` references cross this boundary.  The
System store owns descriptor-relative I/O; this module owns the closed Arrow
schemas, value policy, deterministic normalization, and non-authorizing decode
attestations.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import math
import re
import textwrap
from types import MappingProxyType
from typing import Any, BinaryIO, Final

import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
    seal_artifact,
)
from quant_investor.system import SystemError, SystemStore

from .common import (
    artifact_ref,
    business_identity,
    canonical_a_share_symbol,
    canonical_timestamp,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .errors import FactorGovernanceError
from .implementations import installed_semantic_row
from .manifest import validate_validator_manifest

SOURCE_DECODE_ATTESTATION_KIND: Final = "factor.source_decode_attestation"
STRICT_DECODER_ID: Final = "factor-strict-parquet-source-decoder"
PARQUET_MEDIA_TYPE: Final = "application/vnd.apache.parquet"
PARQUET_FORMAT: Final = "PARQUET"

MAXIMUM_SOURCE_BYTES: Final = 512 * 1024 * 1024
MAXIMUM_SOURCE_ROWS: Final = 10_000_000
MAXIMUM_SOURCE_CELLS: Final = 100_000_000
MAXIMUM_PARQUET_METADATA_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_PARQUET_ROW_GROUPS: Final = 4_096
MAXIMUM_DECODED_ROW_GROUP_BYTES: Final = 256 * 1024 * 1024
DECODED_RESERVATION_BYTES: Final = 256 * 1024 * 1024
DECODED_RESERVATION_OVERHEAD_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_DECODED_TABLE_BYTES: Final = (
    DECODED_RESERVATION_BYTES - DECODED_RESERVATION_OVERHEAD_BYTES
) // 2

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_PURPOSE_ROLES: Final = {
    "PREREGISTRATION": frozenset({"exchange_calendar", "implementation_manifest"}),
    "SIGNAL_CAPTURE": frozenset({"pit_universe", "market_history", "sparse_weights"}),
    "LABEL_OBSERVATION": frozenset({"matured_label_prices"}),
}
_ROLE_KEY_FIELDS: Final = {
    "exchange_calendar": ("ordinal",),
    "implementation_manifest": ("factor_id",),
    "pit_universe": ("signal_session", "symbol"),
    "market_history": ("trade_date", "symbol"),
    "sparse_weights": ("signal_session", "configuration_id", "symbol"),
    "matured_label_prices": ("price_date", "symbol"),
}
_SESSION_FIELD: Final = {
    "exchange_calendar": "open_session",
    "implementation_manifest": None,
    "pit_universe": "signal_session",
    "market_history": "trade_date",
    "sparse_weights": "signal_session",
    "matured_label_prices": "price_date",
}
_SESSION_CARDINALITY_LIMITS: Final = {
    "pit_universe": 1,
    "sparse_weights": 1,
    "matured_label_prices": 2,
}

_SCHEMAS: Final = {
    "exchange_calendar": pa.schema(
        [
            pa.field("ordinal", pa.int32(), nullable=False),
            pa.field("open_session", pa.date32(), nullable=False),
            pa.field("opens_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("closes_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
    "implementation_manifest": pa.schema(
        [
            pa.field("factor_id", pa.string(), nullable=False),
            pa.field("implementation_id", pa.string(), nullable=False),
            pa.field("implementation_component_kind", pa.string(), nullable=False),
            pa.field("implementation_component_contract_sha256", pa.string(), nullable=False),
            pa.field("implementation_component_artifact_id", pa.string(), nullable=False),
            pa.field("implementation_component_semantic_sha256", pa.string(), nullable=False),
            pa.field("implementation_component_byte_sha256", pa.string(), nullable=False),
            pa.field("module_name", pa.string(), nullable=False),
            pa.field("qualified_name", pa.string(), nullable=False),
            pa.field("code_sha256", pa.string(), nullable=False),
            pa.field("family", pa.string(), nullable=False),
            pa.field("primitive", pa.string(), nullable=False),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("formula", pa.string(), nullable=False),
            pa.field("normalized_expression", pa.string(), nullable=False),
            pa.field("parameters_json", pa.string(), nullable=False),
            pa.field(
                "input_fields",
                pa.list_(pa.field("element", pa.string(), nullable=True)),
                nullable=False,
            ),
            pa.field(
                "required_source_roles",
                pa.list_(pa.field("element", pa.string(), nullable=True)),
                nullable=False,
            ),
        ]
    ),
    "pit_universe": pa.schema(
        [
            pa.field("signal_session", pa.date32(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("industry", pa.string(), nullable=True),
            pa.field("total_mv", pa.float64(), nullable=True),
            pa.field("tradable", pa.bool_(), nullable=True),
        ]
    ),
    "market_history": pa.schema(
        [
            pa.field("trade_date", pa.date32(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("adj_close", pa.float64(), nullable=True),
            pa.field("amount", pa.float64(), nullable=True),
            pa.field("vol", pa.float64(), nullable=True),
        ]
    ),
    "sparse_weights": pa.schema(
        [
            pa.field("signal_session", pa.date32(), nullable=False),
            pa.field("configuration_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("weight", pa.decimal128(38, 12), nullable=False),
        ]
    ),
    "matured_label_prices": pa.schema(
        [
            pa.field("price_date", pa.date32(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("adj_close", pa.float64(), nullable=True),
        ]
    ),
}

_ATTESTATION_FIELDS: Final = {
    "source_decode_attestation_id",
    "purpose",
    "preregistration_id",
    "selection_id",
    "ordinal",
    "signal_session",
    "maturity_session",
    "decoder_contract",
    "source_bindings",
    "normalized_inputs_sha256",
    "authority",
}
_DECODER_CONTRACT_FIELDS: Final = {
    "decoder_id",
    "factor_validator_manifest_ref",
    "contextual_validator_component_ref",
    "source_decoder_component_ref",
    "decoder_code_sha256",
    "implementation_component_refs",
    "allowed_source_formats",
    "fallback_allowed",
}
_BINDING_FIELDS: Final = {
    "role",
    "source_object_ref",
    "source_root_id",
    "source_object_created_at",
    "media_type",
    "source_format",
    "source_byte_sha256",
    "source_byte_count",
    "decoded_schema_sha256",
    "normalized_sha256",
    "row_count",
    "column_count",
    "decoded_cell_count",
    "minimum_session",
    "maximum_session",
}


@dataclass(frozen=True)
class DecodedSource:
    """One bounded projection and its immutable strict-source replay summary.

    The Arrow table never crosses the decoder API.  ``projection`` is a fresh
    canonical-JSON value produced while the System-owned memory/FD lease is
    held, so it cannot contain Arrow, pandas, NumPy, or another live object.
    """

    role: str
    source_object_ref: dict[str, str]
    projection: Any
    binding: dict[str, Any]


def _error(code: str, detail: str) -> FactorGovernanceError:
    return FactorGovernanceError(detail, code=f"FACTOR_{code}")


def role_schema(role: str) -> pa.Schema:
    """Return the one compiled Arrow schema for ``role``."""

    if type(role) is not str or role not in _SCHEMAS:
        raise _error("SOURCE_ROLE_INVALID", "source role is not compiled")
    return _SCHEMAS[role]


def _schema_sha256(role: str, schema: pa.Schema) -> str:
    preimage = {
        "domain": "myquant-factor-decoded-schema",
        "role": role,
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise _error("SOURCE_VALUE_INVALID", f"{label} is not canonical text")
    value.encode("utf-8", errors="strict")
    return value


def _date_text(value: Any, *, label: str) -> str:
    if type(value) is not date or isinstance(value, datetime):
        raise _error("SOURCE_VALUE_INVALID", f"{label} is not a date32 value")
    return value.isoformat()


def _timestamp_text(value: Any, *, label: str) -> str:
    if type(value) is not datetime or value.utcoffset() is None:
        raise _error("SOURCE_VALUE_INVALID", f"{label} is not a UTC timestamp")
    observed = value.astimezone(timezone.utc)
    if value.utcoffset() != observed.utcoffset():
        raise _error("SOURCE_VALUE_INVALID", f"{label} is not in UTC")
    return observed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _finite_float(value: Any, *, label: str, positive: bool = False) -> str | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be finite float64 or null")
    if positive and value <= 0:
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be positive when present")
    return value.hex()


def _canonical_json_text(value: Any, *, label: str) -> str:
    text = _canonical_text(value, label=label)
    try:
        parsed = parse_canonical_json_bytes(text.encode("utf-8"), label=label)
    except ContractError as exc:
        raise _error("SOURCE_VALUE_INVALID", f"{label} is not canonical JSON") from exc
    if not isinstance(parsed, (dict, list)):
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be a JSON container")
    return text


def _normalize_boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be boolean")
    return value


def _normalize_int32(value: Any, *, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be int32")
    if not -(2**31) <= value < 2**31:
        raise _error("SOURCE_VALUE_INVALID", f"{label} exceeds int32")
    return value


def _normalize_decimal(value: Any, *, label: str) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be exact decimal")
    if value.as_tuple().exponent != -12:
        raise _error("SOURCE_VALUE_INVALID", f"{label} must have scale 12")
    return format(value, ".12f")


def _normalize_string_list(value: Any, *, label: str) -> list[str]:
    if type(value) is not list:
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be list<string>")
    return [_canonical_text(item, label=f"{label}[]") for item in value]


def _normalize_complex_scalar(field: pa.Field, value: Any, *, label: str) -> Any:
    if pa.types.is_date32(field.type):
        return _date_text(value, label=label)
    if pa.types.is_timestamp(field.type):
        return _timestamp_text(value, label=label)
    if pa.types.is_float64(field.type):
        return _finite_float(value, label=label)
    if pa.types.is_decimal(field.type):
        return _normalize_decimal(value, label=label)
    if pa.types.is_list(field.type):
        return _normalize_string_list(value, label=label)
    raise _error("SOURCE_SCHEMA_INVALID", f"{label} has an unsupported Arrow type")


def _normalize_generic_scalar(field: pa.Field, value: Any, *, label: str) -> Any:
    if value is None:
        if not field.nullable:
            raise _error("SOURCE_VALUE_INVALID", f"{label} may not be null")
        return None
    if pa.types.is_string(field.type):
        return _canonical_text(value, label=label)
    if pa.types.is_boolean(field.type):
        return _normalize_boolean(value, label=label)
    if pa.types.is_int32(field.type):
        return _normalize_int32(value, label=label)
    return _normalize_complex_scalar(field, value, label=label)


def _normalize_implementation_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    factor_id = _canonical_text(row["factor_id"], label=f"row[{index}].factor_id")
    component_ref = validate_artifact_ref(
        {
            "kind": row["implementation_component_kind"],
            "contract_sha256": row["implementation_component_contract_sha256"],
            "artifact_id": row["implementation_component_artifact_id"],
            "semantic_sha256": row["implementation_component_semantic_sha256"],
            "byte_sha256": row["implementation_component_byte_sha256"],
        },
        label=f"row[{index}].implementation_component_ref",
        expected_kind="system.installed_component_manifest",
    )
    expected = installed_semantic_row(factor_id)
    observed = {
        "factor_id": factor_id,
        "implementation_id": _canonical_text(
            row["implementation_id"], label=f"row[{index}].implementation_id"
        ),
        "module_name": _canonical_text(row["module_name"], label=f"row[{index}].module_name"),
        "qualified_name": _canonical_text(
            row["qualified_name"], label=f"row[{index}].qualified_name"
        ),
        "code_sha256": require_sha256(row["code_sha256"], label=f"row[{index}].code_sha256"),
        "family": _canonical_text(row["family"], label=f"row[{index}].family"),
        "primitive": _canonical_text(row["primitive"], label=f"row[{index}].primitive"),
        "direction": _canonical_text(row["direction"], label=f"row[{index}].direction"),
        "formula": _canonical_text(row["formula"], label=f"row[{index}].formula"),
        "normalized_expression": _canonical_json_text(
            row["normalized_expression"],
            label=f"row[{index}].normalized_expression",
        ),
        "parameters_json": _canonical_json_text(
            row["parameters_json"], label=f"row[{index}].parameters_json"
        ),
        "input_fields": [
            _canonical_text(value, label=f"row[{index}].input_fields[]")
            for value in row["input_fields"]
        ],
        "required_source_roles": [
            _canonical_text(value, label=f"row[{index}].required_source_roles[]")
            for value in row["required_source_roles"]
        ],
    }
    if observed["input_fields"] != sorted(set(observed["input_fields"])):
        raise _error("SOURCE_VALUE_INVALID", "implementation input fields are not exact")
    if observed["required_source_roles"] != sorted(
        set(observed["required_source_roles"]), key=lambda value: value.encode("utf-8")
    ):
        raise _error(
            "SOURCE_VALUE_INVALID",
            "implementation required source roles are not exact",
        )
    if observed != expected:
        raise _error("SOURCE_VALUE_INVALID", "implementation row differs from installed semantics")
    return {
        **observed,
        "implementation_component_ref": component_ref,
    }


def _normalize_symbol(row: Mapping[str, Any], role: str, index: int) -> str:
    try:
        return canonical_a_share_symbol(row["symbol"], label=f"{role}[{index}].symbol")
    except FactorGovernanceError as exc:
        raise _error("SOURCE_VALUE_INVALID", "source symbol is invalid") from exc


def _validate_calendar_row(row: Mapping[str, Any], normalized: dict[str, Any], index: int) -> None:
    if row["ordinal"] != index or row["opens_at_utc"] >= row["closes_at_utc"]:
        raise _error("SOURCE_ORDER_INVALID", "calendar ordinal/window is invalid")


def _validate_pit_row(row: Mapping[str, Any], normalized: dict[str, Any], index: int) -> None:
    if row["industry"] is not None:
        normalized["industry"] = _canonical_text(
            row["industry"], label=f"pit_universe[{index}].industry"
        )
    normalized["total_mv"] = _finite_float(
        row["total_mv"], label=f"pit_universe[{index}].total_mv", positive=True
    )


def _validate_market_row(row: Mapping[str, Any], normalized: dict[str, Any], index: int) -> None:
    normalized["adj_close"] = _finite_float(
        row["adj_close"], label=f"market_history[{index}].adj_close", positive=True
    )
    normalized["amount"] = _finite_float(
        row["amount"], label=f"market_history[{index}].amount", positive=True
    )
    normalized["vol"] = _finite_float(row["vol"], label=f"market_history[{index}].vol")
    if row["vol"] is not None and row["vol"] < 0:
        raise _error("SOURCE_VALUE_INVALID", "market volume may not be negative")


def _validate_weight_row(row: Mapping[str, Any], normalized: dict[str, Any], index: int) -> None:
    del normalized, index
    if row["weight"] == 0:
        raise _error("SOURCE_VALUE_INVALID", "sparse weights may not contain zero")


def _validate_label_row(row: Mapping[str, Any], normalized: dict[str, Any], index: int) -> None:
    normalized["adj_close"] = _finite_float(
        row["adj_close"],
        label=f"matured_label_prices[{index}].adj_close",
        positive=True,
    )


_ROLE_ROW_VALIDATORS: Final = {
    "exchange_calendar": _validate_calendar_row,
    "pit_universe": _validate_pit_row,
    "market_history": _validate_market_row,
    "sparse_weights": _validate_weight_row,
    "matured_label_prices": _validate_label_row,
}


def _normalize_role_row(
    role: str,
    row: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    if role == "implementation_manifest":
        return _normalize_implementation_row(row, index)
    normalized = {
        field.name: _normalize_generic_scalar(
            field, row[field.name], label=f"{role}[{index}].{field.name}"
        )
        for field in role_schema(role)
    }
    if "symbol" in row:
        normalized["symbol"] = _normalize_symbol(row, role, index)
    _ROLE_ROW_VALIDATORS[role](row, normalized, index)
    return normalized


def _key_for(role: str, normalized: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(normalized[field] for field in _ROLE_KEY_FIELDS[role])


def _normalize_table(
    role: str, table: pa.Table, schema_sha256: str
) -> tuple[str, str | None, str | None]:
    header = {
        "domain": "myquant-factor-normalized-source",
        "role": role,
        "decoded_schema_sha256": schema_sha256,
        "row_count": table.num_rows,
        "column_count": table.num_columns,
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\n")
    previous_key: tuple[Any, ...] | None = None
    previous_calendar_close: datetime | None = None
    previous_calendar_session: str | None = None
    minimum_session: str | None = None
    maximum_session: str | None = None
    observed_sessions: set[str] = set()
    index = 0
    for batch in table.to_batches(max_chunksize=65_536):
        for row in batch.to_pylist():
            normalized = _normalize_role_row(role, row, index=index)
            key = _key_for(role, normalized)
            if previous_key is not None and key <= previous_key:
                code = "SOURCE_DUPLICATE" if key == previous_key else "SOURCE_ORDER_INVALID"
                raise _error(code, "source role key order is not strict")
            previous_key = key
            if role == "exchange_calendar":
                opens_at = row["opens_at_utc"]
                calendar_session = normalized["open_session"]
                if previous_calendar_close is not None and (
                    previous_calendar_close >= opens_at
                    or previous_calendar_session is None
                    or previous_calendar_session >= calendar_session
                ):
                    raise _error("SOURCE_ORDER_INVALID", "calendar windows overlap")
                previous_calendar_close = row["closes_at_utc"]
                previous_calendar_session = calendar_session
            session_field = _SESSION_FIELD[role]
            if session_field is not None:
                session = normalized[session_field]
                minimum_session = (
                    session if minimum_session is None else min(minimum_session, session)
                )
                maximum_session = (
                    session if maximum_session is None else max(maximum_session, session)
                )
                cardinality_limit = _SESSION_CARDINALITY_LIMITS.get(role)
                if cardinality_limit is not None:
                    observed_sessions.add(session)
                    if len(observed_sessions) > cardinality_limit:
                        raise _error(
                            "SOURCE_CARDINALITY_INVALID",
                            "source role session cardinality exceeds its bound",
                        )
            digest.update(canonical_json_bytes(normalized))
            digest.update(b"\n")
            index += 1
    if index != table.num_rows:
        raise _error("SOURCE_CARDINALITY_INVALID", "decoded source row count changed")
    _validate_role_cardinality(role, table.num_rows, observed_sessions)
    return digest.hexdigest(), minimum_session, maximum_session


def _validate_role_cardinality(role: str, row_count: int, sessions: set[str]) -> None:
    if row_count <= 0:
        raise _error("SOURCE_CARDINALITY_INVALID", "source role may not be empty")
    if role == "exchange_calendar" and row_count < 391:
        raise _error("SOURCE_CARDINALITY_INVALID", "calendar requires at least 391 rows")
    if role == "implementation_manifest" and not 1 <= row_count <= 20:
        raise _error("SOURCE_CARDINALITY_INVALID", "implementation count is outside its bound")
    if role in {"pit_universe", "sparse_weights"} and len(sessions) != 1:
        raise _error("SOURCE_CARDINALITY_INVALID", "source role must bind one signal session")
    if role == "matured_label_prices" and len(sessions) != 2:
        raise _error("SOURCE_CARDINALITY_INVALID", "label source must bind exactly two sessions")


def _resolve_source_ref(
    system_store: SystemStore, source_ref: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    try:
        ref = validate_artifact_ref(
            dict(source_ref), label="source_object_ref", expected_kind="system.source_object"
        )
    except (FactorGovernanceError, TypeError, ValueError) as exc:
        raise _error("SOURCE_REF_INVALID", "source object ref is invalid") from exc
    try:
        artifact = system_store.get_object(ref)
    except SystemError as exc:
        raise _error("SOURCE_REF_INVALID", "source descriptor cannot be resolved") from exc
    payload = artifact.get("payload")
    if type(payload) is not dict:
        raise _error("SOURCE_REF_INVALID", "source descriptor payload is invalid")
    if payload.get("source_root_id") != system_store.source_root_id:
        raise _error("SOURCE_ROOT_MISMATCH", "source root identity differs")
    if (
        payload.get("media_type") != PARQUET_MEDIA_TYPE
        or payload.get("source_format") != PARQUET_FORMAT
    ):
        raise _error("SOURCE_FORMAT_INVALID", "source is not strict Parquet")
    return artifact, payload, ref


def _row_group_decoded_bytes(metadata: Any, row_group_index: int) -> int:
    row_group = metadata.row_group(row_group_index)
    sizes = [
        row_group.column(column_index).total_uncompressed_size
        for column_index in range(row_group.num_columns)
    ]
    if any(type(value) is not int or value < 0 for value in sizes):
        raise _error("SOURCE_SIZE_EXCEEDED", "Parquet decoded size metadata is invalid")
    return sum(sizes)


def _validate_parquet_metadata(metadata: Any, expected: pa.Schema) -> None:
    metadata_size = getattr(metadata, "serialized_size", None)
    if (
        type(metadata_size) is not int
        or metadata_size < 0
        or metadata_size > MAXIMUM_PARQUET_METADATA_BYTES
        or metadata.num_row_groups > MAXIMUM_PARQUET_ROW_GROUPS
        or metadata.num_rows > MAXIMUM_SOURCE_ROWS
    ):
        raise _error("SOURCE_SIZE_EXCEEDED", "Parquet metadata exceeds a resource bound")
    if metadata.num_rows * len(expected) > MAXIMUM_SOURCE_CELLS:
        raise _error("SOURCE_SIZE_EXCEEDED", "decoded source cell count exceeds its bound")
    decoded_total = 0
    for row_group_index in range(metadata.num_row_groups):
        decoded_bytes = _row_group_decoded_bytes(metadata, row_group_index)
        if decoded_bytes > MAXIMUM_DECODED_ROW_GROUP_BYTES:
            raise _error(
                "SOURCE_SIZE_EXCEEDED",
                "Parquet row group exceeds the decoded-memory bound",
            )
        decoded_total += decoded_bytes
    reservation = 2 * decoded_total + DECODED_RESERVATION_OVERHEAD_BYTES
    if reservation > DECODED_RESERVATION_BYTES:
        raise _error(
            "SOURCE_SIZE_EXCEEDED",
            "Parquet aggregate decoded size exceeds the memory reservation",
        )


def _decode_parquet(stream: BinaryIO, role: str) -> pa.Table:
    try:
        parquet = pq.ParquetFile(stream)
        metadata = parquet.metadata
    except Exception as exc:
        raise _error("SOURCE_SCHEMA_INVALID", "source is not readable Parquet") from exc
    expected = role_schema(role)
    _validate_parquet_metadata(metadata, expected)
    if not parquet.schema_arrow.equals(expected, check_metadata=True):
        raise _error("SOURCE_SCHEMA_INVALID", "Parquet Arrow schema is not exact")
    try:
        table = parquet.read(use_threads=False)
    except Exception as exc:
        raise _error("SOURCE_SCHEMA_INVALID", "Parquet decode failed") from exc
    if table.nbytes > MAXIMUM_DECODED_TABLE_BYTES:
        raise _error("SOURCE_SIZE_EXCEEDED", "decoded Arrow table exceeds its memory bound")
    if (
        table.num_rows != metadata.num_rows
        or table.num_rows > MAXIMUM_SOURCE_ROWS
        or table.num_rows * table.num_columns > MAXIMUM_SOURCE_CELLS
        or not table.schema.equals(expected, check_metadata=True)
    ):
        raise _error("SOURCE_SCHEMA_INVALID", "decoded Arrow table is not exact")
    return table


def _bounded_projection(
    projector: Callable[[pa.Table, Mapping[str, Any]], Any],
    table: pa.Table,
    binding: Mapping[str, Any],
) -> Any:
    try:
        projector_binding = parse_canonical_json_bytes(canonical_json_bytes(dict(binding)))
        if type(projector_binding) is not dict:
            raise _error("SOURCE_VALUE_INVALID", "source binding copy is invalid")
        projected = projector(table, MappingProxyType(projector_binding))
        projection_bytes = canonical_json_bytes(projected)
        return parse_canonical_json_bytes(projection_bytes)
    except FactorGovernanceError:
        raise
    except ContractError as exc:
        code = (
            "SOURCE_SIZE_EXCEEDED" if "byte bound" in str(exc).lower() else "SOURCE_VALUE_INVALID"
        )
        raise _error(code, "source projection is not bounded plain data") from exc
    except Exception as exc:
        raise _error("SOURCE_VALUE_INVALID", "source projection failed") from exc


def decode_source_role(
    *,
    system_store: SystemStore,
    source_object_ref: Mapping[str, Any],
    role: str,
    projector: Callable[[pa.Table, Mapping[str, Any]], Any],
) -> DecodedSource:
    """Decode one strict source and return only a bounded plain-data projection."""

    if not callable(projector):
        raise _error("SOURCE_VALUE_INVALID", "source projector must be callable")

    schema = role_schema(role)
    artifact, payload, ref = _resolve_source_ref(system_store, source_object_ref)
    try:
        source_context = system_store.open_source_object(
            ref,
            maximum_bytes=MAXIMUM_SOURCE_BYTES,
            decoded_reservation_bytes=DECODED_RESERVATION_BYTES,
        )
        with source_context as (observed_payload, stream):
            if observed_payload != payload:
                raise _error("SOURCE_BYTES_CHANGED", "source descriptor changed during read")
            stream.seek(0, 2)
            source_byte_count = stream.tell()
            stream.seek(0)
            if not 0 < source_byte_count <= MAXIMUM_SOURCE_BYTES:
                raise _error("SOURCE_SIZE_EXCEEDED", "source byte size is outside its bound")
            table = _decode_parquet(stream, role)
            schema_sha256 = _schema_sha256(role, schema)
            normalized_sha256, minimum_session, maximum_session = _normalize_table(
                role, table, schema_sha256
            )
            binding = {
                "role": role,
                "source_object_ref": ref,
                "source_root_id": payload["source_root_id"],
                "source_object_created_at": artifact["created_at"],
                "media_type": payload["media_type"],
                "source_format": payload["source_format"],
                "source_byte_sha256": payload["byte_sha256"],
                "source_byte_count": source_byte_count,
                "decoded_schema_sha256": schema_sha256,
                "normalized_sha256": normalized_sha256,
                "row_count": table.num_rows,
                "column_count": table.num_columns,
                "decoded_cell_count": table.num_rows * table.num_columns,
                "minimum_session": minimum_session,
                "maximum_session": maximum_session,
            }
            projection = _bounded_projection(projector, table, binding)
            del table
            returned_binding = dict(binding)
            returned_binding["source_object_ref"] = dict(ref)
            return DecodedSource(
                role=role,
                source_object_ref=dict(ref),
                projection=projection,
                binding=returned_binding,
            )
    except FactorGovernanceError:
        raise
    except SystemError as exc:
        code = (
            "SOURCE_SIZE_EXCEEDED"
            if "bound" in str(exc).lower() or "size" in str(exc).lower()
            else "SOURCE_SECURITY_FAILED"
        )
        raise _error(code, "source stream failed secure replay") from exc


def _entrypoint_code_sha256() -> str:
    try:
        parsed = ast.parse(textwrap.dedent(inspect.getsource(decode_source_role)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise _error("SOURCE_SCHEMA_INVALID", "decoder source AST is unavailable") from exc
    nodes = [
        node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(nodes) != 1 or nodes[0].name != "decode_source_role":
        raise _error("SOURCE_SCHEMA_INVALID", "decoder source AST is ambiguous")
    preimage = {
        "domain": "myquant-python-ast-entrypoint",
        "module_name": decode_source_role.__module__,
        "qualified_name": decode_source_role.__qualname__,
        "node": ast.dump(nodes[0], annotate_fields=True, include_attributes=False),
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def _ref_sort_key(ref: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        ref[field]
        for field in (
            "kind",
            "contract_sha256",
            "artifact_id",
            "semantic_sha256",
            "byte_sha256",
        )
    )


def _optional_identifier(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _canonical_text(value, label=label)


def _validate_preregistration_identity(values: tuple[Any, ...]) -> None:
    if any(value is not None for value in values):
        raise _error("SOURCE_VALUE_INVALID", "preregistration attestation nulls differ")


def _validate_signal_identity(
    preregistration_id: str | None,
    selection_id: str | None,
    ordinal: int | None,
    signal_session: str | None,
    maturity_session: str | None,
) -> None:
    valid_ordinal = type(ordinal) is int and 0 <= ordinal < 360
    if preregistration_id is None or not valid_ordinal:
        raise _error("SOURCE_VALUE_INVALID", "signal attestation identity is invalid")
    if (
        signal_session is None
        or maturity_session is not None
        or (ordinal == 0) != (selection_id is None)
    ):
        raise _error("SOURCE_VALUE_INVALID", "signal attestation nulls differ")


def _validate_label_identity(
    preregistration_id: str | None,
    selection_id: str | None,
    ordinal: int | None,
    signal_session: str | None,
    maturity_session: str | None,
) -> None:
    valid_ordinal = type(ordinal) is int and 0 <= ordinal < 360
    if (
        preregistration_id is None
        or selection_id is None
        or not valid_ordinal
        or signal_session is None
        or maturity_session is None
        or maturity_session <= signal_session
    ):
        raise _error("SOURCE_VALUE_INVALID", "label attestation identity is invalid")


def _purpose_values(
    *,
    purpose: Any,
    preregistration_id: Any,
    selection_id: Any,
    ordinal: Any,
    signal_session: Any,
    maturity_session: Any,
) -> tuple[str, str | None, str | None, int | None, str | None, str | None]:
    if type(purpose) is not str or purpose not in _PURPOSE_ROLES:
        raise _error("SOURCE_ROLE_INVALID", "source attestation purpose is invalid")
    prereg = _optional_identifier(preregistration_id, "preregistration_id")
    selection = _optional_identifier(selection_id, "selection_id")
    if ordinal is not None and (type(ordinal) is not int or isinstance(ordinal, bool)):
        raise _error("SOURCE_VALUE_INVALID", "ordinal is invalid")
    signal = (
        None if signal_session is None else _canonical_session(signal_session, "signal_session")
    )
    maturity = (
        None
        if maturity_session is None
        else _canonical_session(maturity_session, "maturity_session")
    )
    if purpose == "PREREGISTRATION":
        _validate_preregistration_identity((prereg, selection, ordinal, signal, maturity))
    elif purpose == "SIGNAL_CAPTURE":
        _validate_signal_identity(prereg, selection, ordinal, signal, maturity)
    else:
        _validate_label_identity(prereg, selection, ordinal, signal, maturity)
    return purpose, prereg, selection, ordinal, signal, maturity


def _canonical_session(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error("SOURCE_VALUE_INVALID", f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise _error("SOURCE_VALUE_INVALID", f"{label} is not canonical")
    return value


def _normalized_inputs_sha256(purpose: str, bindings: Sequence[Mapping[str, Any]]) -> str:
    preimage = {
        "domain": "myquant-factor-normalized-inputs",
        "purpose": purpose,
        "bindings": [
            {
                "role": row["role"],
                "source_object_ref": row["source_object_ref"],
                "decoded_schema_sha256": row["decoded_schema_sha256"],
                "normalized_sha256": row["normalized_sha256"],
                "row_count": row["row_count"],
            }
            for row in bindings
        ],
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def _validated_component(value: Mapping[str, Any] | bytes, *, label: str) -> dict[str, Any]:
    try:
        return validate_governance_artifact(
            value, expected_kind="system.installed_component_manifest"
        )
    except FactorGovernanceError as exc:
        raise _error("SOURCE_REF_INVALID", f"{label} is invalid") from exc


def _decoder_contract(
    *,
    factor_validator_manifest: Mapping[str, Any],
    contextual_validator_component: Mapping[str, Any],
    source_decoder_component: Mapping[str, Any],
    implementation_components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_payload = factor_validator_manifest["payload"]
    contextual_ref = artifact_ref(contextual_validator_component)
    decoder_ref = artifact_ref(source_decoder_component)
    implementation_refs = sorted(
        (artifact_ref(component) for component in implementation_components),
        key=_ref_sort_key,
    )
    if len(implementation_refs) != len({_ref_sort_key(ref) for ref in implementation_refs}):
        raise _error("SOURCE_DUPLICATE", "implementation component refs are duplicated")
    expected_implementation_refs = sorted(
        (row["implementation_component_ref"] for row in manifest_payload["implementation_rows"]),
        key=_ref_sort_key,
    )
    if (
        contextual_ref != manifest_payload["contextual_validator_component_ref"]
        or decoder_ref != manifest_payload["source_decoder_component_ref"]
        or implementation_refs != expected_implementation_refs
    ):
        raise _error("SOURCE_REF_INVALID", "decoder components differ from Factor manifest")
    decoder_payload = source_decoder_component["payload"]
    code_sha256 = _entrypoint_code_sha256()
    matches = [
        row
        for row in decoder_payload.get("entrypoints", [])
        if row.get("module_name") == decode_source_role.__module__
        and row.get("qualified_name") == decode_source_role.__qualname__
    ]
    if (
        len(matches) != 1
        or matches[0].get("code_sha256") != code_sha256
        or decoder_payload.get("allowed_source_formats") != [PARQUET_FORMAT]
        or decoder_payload.get("fallback_allowed") is not False
    ):
        raise _error("SOURCE_REF_INVALID", "decoder component entrypoint differs")
    return {
        "decoder_id": STRICT_DECODER_ID,
        "factor_validator_manifest_ref": artifact_ref(factor_validator_manifest),
        "contextual_validator_component_ref": contextual_ref,
        "source_decoder_component_ref": decoder_ref,
        "decoder_code_sha256": code_sha256,
        "implementation_component_refs": implementation_refs,
        "allowed_source_formats": [PARQUET_FORMAT],
        "fallback_allowed": False,
    }


def build_source_decode_attestation(
    *,
    purpose: str,
    preregistration_id: str | None,
    selection_id: str | None,
    ordinal: int | None,
    signal_session: str | None,
    maturity_session: str | None,
    decoded_sources: Mapping[str, DecodedSource],
    factor_validator_manifest: Mapping[str, Any] | bytes,
    contextual_validator_component: Mapping[str, Any] | bytes,
    source_decoder_component: Mapping[str, Any] | bytes,
    implementation_components: Sequence[Mapping[str, Any] | bytes],
    trusted_at: str,
) -> dict[str, Any]:
    """Seal one trusted, non-authorizing strict-source replay statement."""

    values = _purpose_values(
        purpose=purpose,
        preregistration_id=preregistration_id,
        selection_id=selection_id,
        ordinal=ordinal,
        signal_session=signal_session,
        maturity_session=maturity_session,
    )
    if type(decoded_sources) is not dict or set(decoded_sources) != set(_PURPOSE_ROLES[purpose]):
        raise _error("SOURCE_ROLE_INVALID", "decoded source role set is not exact")
    bindings: list[dict[str, Any]] = []
    for role in sorted(decoded_sources, key=lambda value: value.encode("utf-8")):
        decoded = decoded_sources[role]
        if type(decoded) is not DecodedSource or decoded.role != role:
            raise _error("SOURCE_ROLE_INVALID", "decoded source role binding differs")
        bindings.append(_validate_binding(decoded.binding, expected_role=role))
    manifest = validate_validator_manifest(factor_validator_manifest)
    contextual = _validated_component(
        contextual_validator_component, label="contextual validator component"
    )
    decoder = _validated_component(source_decoder_component, label="source decoder component")
    if isinstance(implementation_components, (str, bytes)) or not isinstance(
        implementation_components, Sequence
    ):
        raise _error("SOURCE_REF_INVALID", "implementation components must be a sequence")
    implementations = [
        _validated_component(value, label="implementation component")
        for value in implementation_components
    ]
    decoder_contract = _decoder_contract(
        factor_validator_manifest=manifest,
        contextual_validator_component=contextual,
        source_decoder_component=decoder,
        implementation_components=implementations,
    )
    normalized_inputs_sha256 = _normalized_inputs_sha256(purpose, bindings)
    identity = {
        "purpose": values[0],
        "preregistration_id": values[1],
        "selection_id": values[2],
        "ordinal": values[3],
        "signal_session": values[4],
        "maturity_session": values[5],
        "decoder_contract": decoder_contract,
        "source_bindings": bindings,
        "normalized_inputs_sha256": normalized_inputs_sha256,
        "authority": "NON_AUTHORIZING",
    }
    payload = {
        "source_decode_attestation_id": business_identity(
            "factor-source-decode-attestation", identity
        ),
        **identity,
    }
    return seal_artifact(
        SOURCE_DECODE_ATTESTATION_KIND,
        payload,
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )


def _binding_role(value: Any, expected_role: str | None) -> tuple[str, dict[str, Any]]:
    if type(value) is not dict or set(value) != _BINDING_FIELDS:
        raise _error("SOURCE_SCHEMA_INVALID", "source binding fields are not exact")
    role = value.get("role")
    if type(role) is not str or role not in _SCHEMAS:
        raise _error("SOURCE_ROLE_INVALID", "source binding role is invalid")
    if expected_role is not None and role != expected_role:
        raise _error("SOURCE_ROLE_INVALID", "source binding role differs")
    return role, dict(value)


def _validate_binding_counts(row: Mapping[str, Any], role: str) -> None:
    for field in (
        "source_byte_count",
        "row_count",
        "column_count",
        "decoded_cell_count",
    ):
        if type(row[field]) is not int or isinstance(row[field], bool) or row[field] < 0:
            raise _error("SOURCE_VALUE_INVALID", f"source binding {field} is invalid")
    if not 0 < row["source_byte_count"] <= MAXIMUM_SOURCE_BYTES:
        raise _error("SOURCE_SIZE_EXCEEDED", "source binding byte count is invalid")
    if not 0 < row["row_count"] <= MAXIMUM_SOURCE_ROWS:
        raise _error("SOURCE_CARDINALITY_INVALID", "source binding row count is invalid")
    if row["column_count"] != len(role_schema(role)) or row["decoded_cell_count"] != (
        row["row_count"] * row["column_count"]
    ):
        raise _error("SOURCE_CARDINALITY_INVALID", "source binding cell count differs")


def _validate_binding_sessions(row: Mapping[str, Any], role: str) -> None:
    session_field = _SESSION_FIELD[role]
    if session_field is None:
        if row["minimum_session"] is not None or row["maximum_session"] is not None:
            raise _error("SOURCE_VALUE_INVALID", "non-session source has session bounds")
    else:
        minimum = _canonical_session(row["minimum_session"], "minimum_session")
        maximum = _canonical_session(row["maximum_session"], "maximum_session")
        if minimum > maximum:
            raise _error("SOURCE_ORDER_INVALID", "source session bounds are reversed")


def _validate_binding(value: Any, *, expected_role: str | None = None) -> dict[str, Any]:
    role, row = _binding_role(value, expected_role)
    row["source_object_ref"] = validate_artifact_ref(
        row["source_object_ref"],
        label=f"{role}.source_object_ref",
        expected_kind="system.source_object",
    )
    _canonical_text(row["source_root_id"], label=f"{role}.source_root_id")
    canonical_timestamp(row["source_object_created_at"], label="source_object_created_at")
    if row["media_type"] != PARQUET_MEDIA_TYPE or row["source_format"] != PARQUET_FORMAT:
        raise _error("SOURCE_FORMAT_INVALID", "source binding format is invalid")
    for field in ("source_byte_sha256", "decoded_schema_sha256", "normalized_sha256"):
        require_sha256(row[field], label=f"{role}.{field}")
    _validate_binding_counts(row, role)
    _validate_binding_sessions(row, role)
    return row


def _validate_decoder_contract(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _DECODER_CONTRACT_FIELDS:
        raise _error("SOURCE_SCHEMA_INVALID", "decoder contract fields are not exact")
    row = dict(value)
    if (
        row.get("decoder_id") != STRICT_DECODER_ID
        or row.get("allowed_source_formats") != [PARQUET_FORMAT]
        or row.get("fallback_allowed") is not False
    ):
        raise _error("SOURCE_FORMAT_INVALID", "decoder contract policy differs")
    for field, kind in (
        ("factor_validator_manifest_ref", "factor.validator_manifest"),
        ("contextual_validator_component_ref", "system.installed_component_manifest"),
        ("source_decoder_component_ref", "system.installed_component_manifest"),
    ):
        row[field] = validate_artifact_ref(row[field], label=field, expected_kind=kind)
    require_sha256(row["decoder_code_sha256"], label="decoder_code_sha256")
    refs = row.get("implementation_component_refs")
    if type(refs) is not list or not 1 <= len(refs) <= 20:
        raise _error("SOURCE_CARDINALITY_INVALID", "implementation refs are invalid")
    normalized = [
        validate_artifact_ref(
            ref,
            label=f"implementation_component_refs[{index}]",
            expected_kind="system.installed_component_manifest",
        )
        for index, ref in enumerate(refs)
    ]
    if normalized != sorted(normalized, key=_ref_sort_key) or len(normalized) != len(
        {_ref_sort_key(ref) for ref in normalized}
    ):
        raise _error("SOURCE_ORDER_INVALID", "implementation refs are not canonical")
    row["implementation_component_refs"] = normalized
    return row


def validate_source_decode_attestation(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate the closed structural replay statement without resolving storage."""

    envelope, payload = exact_payload(
        document,
        kind=SOURCE_DECODE_ATTESTATION_KIND,
        fields=_ATTESTATION_FIELDS,
    )
    values = _purpose_values(
        purpose=payload["purpose"],
        preregistration_id=payload["preregistration_id"],
        selection_id=payload["selection_id"],
        ordinal=payload["ordinal"],
        signal_session=payload["signal_session"],
        maturity_session=payload["maturity_session"],
    )
    decoder_contract = _validate_decoder_contract(payload["decoder_contract"])
    bindings_value = payload.get("source_bindings")
    if type(bindings_value) is not list:
        raise _error("SOURCE_SCHEMA_INVALID", "source bindings must be a list")
    bindings = [_validate_binding(row) for row in bindings_value]
    roles = [row["role"] for row in bindings]
    if (
        roles != sorted(roles, key=lambda value: value.encode("utf-8"))
        or len(roles) != len(set(roles))
        or set(roles) != set(_PURPOSE_ROLES[values[0]])
    ):
        raise _error("SOURCE_ROLE_INVALID", "source binding roles are not exact")
    normalized_inputs_sha256 = _normalized_inputs_sha256(values[0], bindings)
    identity = {
        "purpose": values[0],
        "preregistration_id": values[1],
        "selection_id": values[2],
        "ordinal": values[3],
        "signal_session": values[4],
        "maturity_session": values[5],
        "decoder_contract": decoder_contract,
        "source_bindings": bindings,
        "normalized_inputs_sha256": normalized_inputs_sha256,
        "authority": "NON_AUTHORIZING",
    }
    expected = {
        "source_decode_attestation_id": business_identity(
            "factor-source-decode-attestation", identity
        ),
        **identity,
    }
    if payload != expected:
        raise _error("SOURCE_BYTES_CHANGED", "source decode attestation does not replay")
    return envelope


__all__ = [
    "SOURCE_DECODE_ATTESTATION_KIND",
    "role_schema",
    "validate_source_decode_attestation",
]
