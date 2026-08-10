"""Canonical scalar projection and exact raw-table reconciliation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from ...._core import (
    MAX_ARTIFACT_BYTES,
    canonical_bytes,
    common_fields,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_content_ref,
    validate_seal,
)
from .models import (
    COMPARISON_POLICY_V2,
    SCALAR_KINDS,
    SOURCE_TABLES,
    FundamentalV4ContractError,
    fundamental_v4_contract,
)

_POLICY_FIELDS = {
    "array_order_semantics",
    "authority",
    "canonical_rules",
    "created_at",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "policy_id",
    "production",
    "research_only",
    "semantic_sha256",
    "table_policies",
    "timestamp",
    "version",
}
_TABLE_POLICY_FIELDS = {
    "baseline_source_only_columns",
    "baseline_source_only_reason",
    "baseline_source_schema_evidence_ref",
    "canonical_key_columns",
    "column_rows",
    "table",
    "winner_implementation_sha256",
    "winner_order_columns",
    "winner_rule",
}
_COLUMN_ROW_FIELDS = {"column", "kind"}
_DECIMAL_QUANTUM = Decimal("0.000000000001")
_SCHEMA_DIAGNOSTIC_VERSION = "myquant.v17.intelligence-v2.tushare-schema-diagnostic-receipt.v1"
_MULTISET_HASH_DOMAIN = b"myquant.v17.canonical-row-multiset-stream.v1\0"


def _sequence(value: Any, *, label: str, maximum: int) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalV4ContractError(f"{label} must be a sequence")
    rows = list(value)
    if not rows or len(rows) > maximum:
        raise FundamentalV4ContractError(f"{label} cardinality is invalid")
    return rows


def _column_names(value: Any, *, label: str, allowed: set[str]) -> list[str]:
    rows = _sequence(value, label=label, maximum=512)
    normalized: list[str] = []
    for item in rows:
        if type(item) is not str or item not in allowed:
            raise FundamentalV4ContractError(f"{label} contains an unknown column")
        normalized.append(item)
    if len(normalized) != len(set(normalized)):
        raise FundamentalV4ContractError(f"{label} contains duplicates")
    return normalized


def _baseline_source_only_contract(
    row: Mapping[str, Any],
    *,
    expected_table: str,
    common_columns: set[str],
) -> tuple[list[str], str | None, dict[str, str] | None]:
    value = row["baseline_source_only_columns"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalV4ContractError("baseline source-only columns must be a sequence")
    columns = list(value)
    reason = row["baseline_source_only_reason"]
    evidence = row["baseline_source_schema_evidence_ref"]
    if not columns:
        if reason is not None or evidence is not None:
            raise FundamentalV4ContractError("empty source-only contract has authority fields")
        return [], None, None
    if (
        expected_table != "forecast"
        or columns != ["update_flag"]
        or "update_flag" in common_columns
        or reason != "ENDPOINT_SCHEMA_NOT_EXPOSED"
    ):
        raise FundamentalV4ContractError("baseline source-only contract is not allowlisted")
    validated_ref = validate_content_ref(evidence, label="source schema evidence ref")
    if validated_ref["artifact_version"] != _SCHEMA_DIAGNOSTIC_VERSION:
        raise FundamentalV4ContractError("source schema evidence version is invalid")
    return columns, reason, validated_ref


def _table_policy(value: Any, *, expected_table: str) -> dict[str, Any]:
    row = require_exact_keys(value, _TABLE_POLICY_FIELDS, label="table policy")
    if row.get("table") != expected_table:
        raise FundamentalV4ContractError("table policy identity mismatch")
    column_rows = _sequence(row["column_rows"], label="column_rows", maximum=512)
    normalized_columns: list[dict[str, str]] = []
    for column_row in column_rows:
        item = require_exact_keys(
            column_row,
            _COLUMN_ROW_FIELDS,
            label="column row",
        )
        column = item.get("column")
        kind = item.get("kind")
        if (
            type(column) is not str
            or not column
            or not column.isascii()
            or kind not in SCALAR_KINDS
        ):
            raise FundamentalV4ContractError("column row is invalid")
        normalized_columns.append({"column": column, "kind": kind})
    column_names = [row["column"] for row in normalized_columns]
    if len(column_names) != len(set(column_names)):
        raise FundamentalV4ContractError("table policy has duplicate columns")
    allowed = set(column_names)
    source_only_columns, source_only_reason, source_schema_ref = _baseline_source_only_contract(
        row,
        expected_table=expected_table,
        common_columns=allowed,
    )
    key_columns = _column_names(
        row["canonical_key_columns"],
        label="canonical_key_columns",
        allowed=allowed,
    )
    winner_columns = _column_names(
        row["winner_order_columns"],
        label="winner_order_columns",
        allowed=allowed,
    )
    if row.get("winner_rule") != "ASCII_CANONICAL_LAST":
        raise FundamentalV4ContractError("winner rule is invalid")
    return {
        "baseline_source_only_columns": source_only_columns,
        "baseline_source_only_reason": source_only_reason,
        "baseline_source_schema_evidence_ref": source_schema_ref,
        "canonical_key_columns": key_columns,
        "column_rows": normalized_columns,
        "table": expected_table,
        "winner_implementation_sha256": sha256(
            row.get("winner_implementation_sha256"),
            label="winner_implementation_sha256",
        ),
        "winner_order_columns": winner_columns,
        "winner_rule": "ASCII_CANONICAL_LAST",
    }


@fundamental_v4_contract
def build_fundamental_comparison_policy(
    *,
    table_policies: Mapping[str, Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Seal all scalar, key, and winner semantics without inferred defaults."""

    if type(table_policies) is not dict or set(table_policies) != set(SOURCE_TABLES):
        raise FundamentalV4ContractError("comparison table policy set is invalid")
    normalized = [
        _table_policy(table_policies[table], expected_table=table) for table in SOURCE_TABLES
    ]
    created = timestamp(created_at, label="created_at")
    body = {
        **common_fields(timestamp_value=created),
        "array_order_semantics": {
            "/table_policies": "table ASCII ascending",
            "/table_policies/*/baseline_source_only_columns": "source column order",
            "/table_policies/*/canonical_key_columns": "owner semantic order",
            "/table_policies/*/column_rows": "source column order",
            "/table_policies/*/winner_order_columns": "owner semantic order",
        },
        "canonical_rules": {
            "date": "EXACT_YYYYMMDD",
            "decimal_places": 12,
            "decimal_precision": 50,
            "decimal_rounding": "ROUND_HALF_EVEN",
            "empty_string_is_null": False,
            "input_scientific_notation": True,
            "negative_zero": "0.000000000000",
            "null": "JSON_OR_PARQUET_NULL_ONLY",
            "row_order": "IGNORED_MULTISET",
            "text_normalization": "UNICODE_NFC_NO_TRIM",
        },
        "created_at": created,
        "table_policies": normalized,
        "version": COMPARISON_POLICY_V2,
    }
    return seal(body, identity_field="policy_id")


@fundamental_v4_contract
def validate_fundamental_comparison_policy(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="policy_id")
    require_exact_keys(value, _POLICY_FIELDS, label="Fundamental comparison policy")
    if value.get("version") != COMPARISON_POLICY_V2:
        raise FundamentalV4ContractError("comparison policy version mismatch")
    policies = {
        row["table"]: row
        for row in _sequence(value["table_policies"], label="table_policies", maximum=6)
    }
    expected = build_fundamental_comparison_policy(
        table_policies=policies,
        created_at=value["created_at"],
    )
    if value != expected:
        raise FundamentalV4ContractError("comparison policy replay mismatch")
    return value


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _canonical_decimal(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise FundamentalV4ContractError("boolean is not a canonical numeric value")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FundamentalV4ContractError("numeric value must be finite")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise FundamentalV4ContractError("numeric value is invalid") from exc
    if not parsed.is_finite():
        raise FundamentalV4ContractError("numeric value must be finite")
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        result = parsed.quantize(_DECIMAL_QUANTUM)
    if result == 0:
        result = Decimal(0).quantize(_DECIMAL_QUANTUM)
    return format(result, "f")


def _canonical_integer(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise FundamentalV4ContractError("boolean is not a canonical integer")
    if isinstance(value, Integral):
        return str(int(value))
    decimal = Decimal(_canonical_decimal(value))
    if decimal != decimal.to_integral_value():
        raise FundamentalV4ContractError("integer value has a fractional component")
    return str(int(decimal))


def _canonical_date(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.time() != datetime.min.time():
            raise FundamentalV4ContractError("date value is invalid")
        text = parsed.strftime("%Y%m%d")
    else:
        if type(value) is not str:
            raise FundamentalV4ContractError("date value must be exact YYYYMMDD")
        text = value
    if len(text) != 8 or not text.isdigit():
        raise FundamentalV4ContractError("date value must be exact YYYYMMDD")
    try:
        parsed_date = datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise FundamentalV4ContractError("date value is invalid") from exc
    if parsed_date.strftime("%Y%m%d") != text:
        raise FundamentalV4ContractError("date value is not canonical")
    return text


def _canonical_text(value: Any) -> str:
    if type(value) is not str:
        raise FundamentalV4ContractError("text value must be a string")
    if unicodedata.normalize("NFC", value) != value:
        raise FundamentalV4ContractError("text value must be Unicode NFC")
    value.encode("utf-8", errors="strict")
    return value


def _canonical_scalar(value: Any, *, kind: str) -> tuple[str, str | None]:
    if _is_null(value):
        return ("NULL", None)
    if kind == "DECIMAL":
        return (kind, _canonical_decimal(value))
    if kind == "INTEGER":
        return (kind, _canonical_integer(value))
    if kind == "DATE":
        return (kind, _canonical_date(value))
    if kind == "TEXT":
        return (kind, _canonical_text(value))
    raise FundamentalV4ContractError("scalar kind is unsupported")


def _row_json(value: tuple[Any, ...]) -> list[list[Any]]:
    return [list(cell) for cell in value]


def _row_bytes(value: tuple[Any, ...]) -> bytes:
    return _projection_bytes(_row_json(value))


def _projection_bytes(value: Any) -> bytes:
    """Encode validated raw scalars for internal hashing, not as an artifact."""

    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise FundamentalV4ContractError("canonical raw row exceeds 8 MiB")
    return raw


def _multiset_sha256(rows: Sequence[tuple[str, int]]) -> str:
    """Hash a canonical row-digest multiset without one oversized artifact."""

    digest = hashlib.sha256(_MULTISET_HASH_DOMAIN)
    for row_sha256, count in rows:
        payload = canonical_bytes({"count": count, "row_sha256": row_sha256})
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _projection_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA locking_mode=EXCLUSIVE;
        CREATE TABLE row_counts (
            lane INTEGER NOT NULL,
            row_hash BLOB NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (lane, row_hash)
        ) WITHOUT ROWID;
        CREATE TABLE winner_rows (
            lane INTEGER NOT NULL,
            key_hash BLOB NOT NULL,
            order_bytes BLOB NOT NULL,
            row_hash BLOB NOT NULL,
            PRIMARY KEY (lane, key_hash)
        ) WITHOUT ROWID;
        """)


def _flush_projection_batch(
    connection: sqlite3.Connection,
    *,
    lane_id: int,
    row_hashes: Counter[bytes],
    winners: Mapping[bytes, tuple[bytes, bytes]],
) -> None:
    connection.executemany(
        """
        INSERT INTO row_counts (lane, row_hash, count) VALUES (?, ?, ?)
        ON CONFLICT (lane, row_hash) DO UPDATE SET count = count + excluded.count
        """,
        ((lane_id, row_hash, count) for row_hash, count in row_hashes.items()),
    )
    connection.executemany(
        """
        INSERT INTO winner_rows (lane, key_hash, order_bytes, row_hash)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (lane, key_hash) DO UPDATE SET
            order_bytes = excluded.order_bytes,
            row_hash = excluded.row_hash
        WHERE excluded.order_bytes >= winner_rows.order_bytes
        """,
        (
            (lane_id, key_hash, order_bytes, row_hash)
            for key_hash, (order_bytes, row_hash) in winners.items()
        ),
    )


def _write_table_projection(
    frame: pd.DataFrame,
    *,
    table_policy: Mapping[str, Any],
    lane: str,
    connection: sqlite3.Connection,
) -> int:
    if not isinstance(frame, pd.DataFrame):
        raise FundamentalV4ContractError("raw table must be a DataFrame")
    column_rows = list(table_policy["column_rows"])
    columns = [row["column"] for row in column_rows]
    if lane not in {"BASELINE", "VIP"}:
        raise FundamentalV4ContractError("raw table lane is invalid")
    source_only = list(table_policy["baseline_source_only_columns"])
    expected_columns = columns + source_only if lane == "BASELINE" else columns
    if list(frame.columns) != expected_columns:
        raise FundamentalV4ContractError("raw table column order changed")
    kinds = [row["kind"] for row in column_rows]
    key_indices = [columns.index(value) for value in table_policy["canonical_key_columns"]]
    winner_indices = [columns.index(value) for value in table_policy["winner_order_columns"]]
    lane_id = 0 if lane == "BASELINE" else 1
    row_hashes: Counter[bytes] = Counter()
    winners: dict[bytes, tuple[bytes, bytes]] = {}
    row_count = 0
    for raw_row in frame.loc[:, columns].itertuples(index=False, name=None):
        row = tuple(
            _canonical_scalar(value, kind=kinds[index]) for index, value in enumerate(raw_row)
        )
        row_bytes = _row_bytes(row)
        row_hash = hashlib.sha256(row_bytes).digest()
        key_bytes = _projection_bytes([list(row[index]) for index in key_indices])
        key_hash = hashlib.sha256(key_bytes).digest()
        order_bytes = _projection_bytes([list(row[index]) for index in winner_indices])
        row_hashes[row_hash] += 1
        prior = winners.get(key_hash)
        if prior is None or order_bytes >= prior[0]:
            winners[key_hash] = (order_bytes, row_hash)
        row_count += 1
        if row_count % 10_000 == 0:
            _flush_projection_batch(
                connection,
                lane_id=lane_id,
                row_hashes=row_hashes,
                winners=winners,
            )
            row_hashes.clear()
            winners.clear()
    _flush_projection_batch(
        connection,
        lane_id=lane_id,
        row_hashes=row_hashes,
        winners=winners,
    )
    connection.commit()
    return row_count


def _multiset_projection(connection: sqlite3.Connection, *, lane_id: int) -> dict[str, Any]:
    rows = [
        (bytes(row_hash).hex(), int(count))
        for row_hash, count in connection.execute(
            "SELECT row_hash, count FROM row_counts WHERE lane = ? ORDER BY row_hash",
            (lane_id,),
        )
    ]
    return {
        "duplicate_row_count": sum(count - 1 for _row_hash, count in rows),
        "multiset_sha256": _multiset_sha256(rows),
    }


def _row_diffs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT b.row_hash, b.count, COALESCE(v.count, 0)
        FROM row_counts AS b
        LEFT JOIN row_counts AS v ON v.lane = 1 AND v.row_hash = b.row_hash
        WHERE b.lane = 0 AND (v.count IS NULL OR b.count != v.count)
        UNION ALL
        SELECT v.row_hash, 0, v.count
        FROM row_counts AS v
        LEFT JOIN row_counts AS b ON b.lane = 0 AND b.row_hash = v.row_hash
        WHERE v.lane = 1 AND b.row_hash IS NULL
        ORDER BY 1
    """
    return [
        {
            "baseline_count": int(baseline_count),
            "row_sha256": bytes(row_hash).hex(),
            "vip_count": int(vip_count),
        }
        for row_hash, baseline_count, vip_count in connection.execute(query)
    ]


def _winner_diffs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT b.key_hash, b.row_hash, v.row_hash
        FROM winner_rows AS b
        LEFT JOIN winner_rows AS v ON v.lane = 1 AND v.key_hash = b.key_hash
        WHERE b.lane = 0 AND (v.row_hash IS NULL OR b.row_hash != v.row_hash)
        UNION ALL
        SELECT v.key_hash, NULL, v.row_hash
        FROM winner_rows AS v
        LEFT JOIN winner_rows AS b ON b.lane = 0 AND b.key_hash = v.key_hash
        WHERE v.lane = 1 AND b.key_hash IS NULL
        ORDER BY 1
    """
    return [
        {
            "baseline_winner_sha256": (
                None if baseline_winner is None else bytes(baseline_winner).hex()
            ),
            "key_sha256": bytes(key_hash).hex(),
            "vip_winner_sha256": None if vip_winner is None else bytes(vip_winner).hex(),
        }
        for key_hash, baseline_winner, vip_winner in connection.execute(query)
    ]


def _compare_table(
    *,
    baseline_frame: pd.DataFrame,
    vip_frame: pd.DataFrame,
    table_policy: Mapping[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="myquant-fundamental-compare-") as root:
        database = Path(root) / "projection.sqlite3"
        with closing(sqlite3.connect(database)) as connection:
            _projection_schema(connection)
            baseline_count = _write_table_projection(
                baseline_frame,
                table_policy=table_policy,
                lane="BASELINE",
                connection=connection,
            )
            vip_count = _write_table_projection(
                vip_frame,
                table_policy=table_policy,
                lane="VIP",
                connection=connection,
            )
            baseline = _multiset_projection(connection, lane_id=0)
            vip = _multiset_projection(connection, lane_id=1)
            return {
                "baseline": {**baseline, "row_count": baseline_count},
                "row_diff": _row_diffs(connection),
                "value_diff": _winner_diffs(connection),
                "vip": {**vip, "row_count": vip_count},
            }


@fundamental_v4_contract
def compare_fundamental_raw_tables(
    *,
    baseline_tables: Mapping[str, pd.DataFrame],
    vip_tables: Mapping[str, pd.DataFrame],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare all six tables as canonical row multisets with exact winners."""

    validated_policy = validate_fundamental_comparison_policy(policy)
    if (
        type(baseline_tables) is not dict
        or type(vip_tables) is not dict
        or set(baseline_tables) != set(SOURCE_TABLES)
        or set(vip_tables) != set(SOURCE_TABLES)
    ):
        raise FundamentalV4ContractError("raw comparison table set is invalid")
    policy_by_table = {row["table"]: row for row in validated_policy["table_policies"]}
    row_diff: dict[str, list[dict[str, Any]]] = {}
    value_diff: dict[str, list[dict[str, Any]]] = {}
    duplicate_diff: dict[str, dict[str, int]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for table in SOURCE_TABLES:
        result = _compare_table(
            baseline_frame=baseline_tables[table],
            vip_frame=vip_tables[table],
            table_policy=policy_by_table[table],
        )
        baseline = result["baseline"]
        vip = result["vip"]
        row_diff[table] = result["row_diff"]
        value_diff[table] = result["value_diff"]
        duplicate_diff[table] = {
            "baseline_duplicate_row_count": baseline["duplicate_row_count"],
            "vip_duplicate_row_count": vip["duplicate_row_count"],
        }
        evidence[table] = {
            "baseline_source_only_columns": policy_by_table[table]["baseline_source_only_columns"],
            "baseline_source_only_reason": policy_by_table[table]["baseline_source_only_reason"],
            "baseline_source_schema_evidence_ref": policy_by_table[table][
                "baseline_source_schema_evidence_ref"
            ],
            "baseline_multiset_sha256": baseline["multiset_sha256"],
            "baseline_row_count": baseline["row_count"],
            "vip_multiset_sha256": vip["multiset_sha256"],
            "vip_row_count": vip["row_count"],
            "winner_implementation_sha256": policy_by_table[table]["winner_implementation_sha256"],
        }
    passed = (
        not any(row_diff.values())
        and not any(value_diff.values())
        and all(
            row["baseline_duplicate_row_count"] == row["vip_duplicate_row_count"]
            for row in duplicate_diff.values()
        )
    )
    return {
        "duplicate_diff": duplicate_diff,
        "passed": passed,
        "raw_row_diff": row_diff,
        "raw_value_diff": value_diff,
        "table_evidence": evidence,
    }


__all__ = [
    "build_fundamental_comparison_policy",
    "compare_fundamental_raw_tables",
    "validate_fundamental_comparison_policy",
]
