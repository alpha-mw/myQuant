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

from ._codec import (
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
from ._model import (
    LEGACY_COMPARISON_POLICY_SCHEMA,
    COMPARISON_POLICY_SCHEMA,
    SCALAR_KINDS,
    SOURCE_TABLES,
    FundamentalProviderEvidenceError,
    provider_evidence_contract,
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
_POLICY_FIELDS_V3 = _POLICY_FIELDS | {"comparison_windows"}
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
_TABLE_POLICY_FIELDS_V3 = _TABLE_POLICY_FIELDS | {"winner_completeness_columns"}
_COLUMN_ROW_FIELDS = {"column", "kind"}
_COMPARISON_WINDOW_FIELDS = {"date_column", "end_date", "start_date", "table"}
_WINNER_RULES_V3 = frozenset({"ASCII_CANONICAL_LAST", "UPDATE_FLAG_THEN_COMPLETENESS_THEN_ASCII"})
_DECIMAL_QUANTUM = Decimal("0.000000000001")
_SCHEMA_DIAGNOSTIC_VERSION = "myquant.v17.intelligence-v2.tushare-schema-diagnostic-receipt.v1"
_MULTISET_HASH_DOMAIN = b"myquant.v17.canonical-row-multiset-stream.v1\0"


def _sequence(value: Any, *, label: str, maximum: int) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalProviderEvidenceError(f"{label} must be a sequence")
    rows = list(value)
    if not rows or len(rows) > maximum:
        raise FundamentalProviderEvidenceError(f"{label} cardinality is invalid")
    return rows


def _column_names(value: Any, *, label: str, allowed: set[str]) -> list[str]:
    rows = _sequence(value, label=label, maximum=512)
    normalized: list[str] = []
    for item in rows:
        if type(item) is not str or item not in allowed:
            raise FundamentalProviderEvidenceError(f"{label} contains an unknown column")
        normalized.append(item)
    if len(normalized) != len(set(normalized)):
        raise FundamentalProviderEvidenceError(f"{label} contains duplicates")
    return normalized


def _optional_column_names(value: Any, *, label: str, allowed: set[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalProviderEvidenceError(f"{label} must be a sequence")
    if not value:
        return []
    return _column_names(value, label=label, allowed=allowed)


def _baseline_source_only_contract(
    row: Mapping[str, Any],
    *,
    expected_table: str,
    common_columns: set[str],
) -> tuple[list[str], str | None, dict[str, str] | None]:
    value = row["baseline_source_only_columns"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FundamentalProviderEvidenceError("baseline source-only columns must be a sequence")
    columns = list(value)
    reason = row["baseline_source_only_reason"]
    evidence = row["baseline_source_schema_evidence_ref"]
    if not columns:
        if reason is not None or evidence is not None:
            raise FundamentalProviderEvidenceError(
                "empty source-only contract has authority fields"
            )
        return [], None, None
    if (
        expected_table != "forecast"
        or columns != ["update_flag"]
        or "update_flag" in common_columns
        or reason != "ENDPOINT_SCHEMA_NOT_EXPOSED"
    ):
        raise FundamentalProviderEvidenceError("baseline source-only contract is not allowlisted")
    validated_ref = validate_content_ref(evidence, label="source schema evidence ref")
    if validated_ref["artifact_version"] != _SCHEMA_DIAGNOSTIC_VERSION:
        raise FundamentalProviderEvidenceError("source schema evidence version is invalid")
    return columns, reason, validated_ref


def _table_policy(value: Any, *, expected_table: str) -> dict[str, Any]:
    row = require_exact_keys(value, _TABLE_POLICY_FIELDS, label="table policy")
    if row.get("table") != expected_table:
        raise FundamentalProviderEvidenceError("table policy identity mismatch")
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
            raise FundamentalProviderEvidenceError("column row is invalid")
        normalized_columns.append({"column": column, "kind": kind})
    column_names = [row["column"] for row in normalized_columns]
    if len(column_names) != len(set(column_names)):
        raise FundamentalProviderEvidenceError("table policy has duplicate columns")
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
        raise FundamentalProviderEvidenceError("winner rule is invalid")
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


def _table_policy_v3(value: Any, *, expected_table: str) -> dict[str, Any]:
    row = require_exact_keys(value, _TABLE_POLICY_FIELDS_V3, label="table policy v3")
    legacy_shape = {key: row[key] for key in _TABLE_POLICY_FIELDS}
    legacy_shape["winner_rule"] = "ASCII_CANONICAL_LAST"
    base = _table_policy(
        legacy_shape,
        expected_table=expected_table,
    )
    allowed = {item["column"] for item in base["column_rows"]}
    completeness_columns = _optional_column_names(
        row["winner_completeness_columns"],
        label="winner_completeness_columns",
        allowed=allowed,
    )
    winner_rule = row.get("winner_rule")
    if winner_rule not in _WINNER_RULES_V3:
        raise FundamentalProviderEvidenceError("winner rule is invalid")
    if winner_rule == "UPDATE_FLAG_THEN_COMPLETENESS_THEN_ASCII":
        if expected_table not in {"balancesheet", "cashflow", "income"}:
            raise FundamentalProviderEvidenceError(
                "update-flag winner rule is not allowed for table"
            )
        if "update_flag" not in allowed or not completeness_columns:
            raise FundamentalProviderEvidenceError("update-flag winner rule closure is incomplete")
        if "update_flag" in completeness_columns:
            raise FundamentalProviderEvidenceError("update_flag cannot be a completeness column")
    elif completeness_columns:
        raise FundamentalProviderEvidenceError(
            "ASCII winner rule cannot declare completeness columns"
        )
    return {
        **base,
        "winner_completeness_columns": completeness_columns,
        "winner_rule": winner_rule,
    }


def _comparison_window(value: Any, *, expected_table: str) -> dict[str, str]:
    row = require_exact_keys(value, _COMPARISON_WINDOW_FIELDS, label="comparison window")
    if row.get("table") != expected_table:
        raise FundamentalProviderEvidenceError("comparison window table mismatch")
    expected_date_column = "trade_date" if expected_table == "daily_basic" else "end_date"
    if row.get("date_column") != expected_date_column:
        raise FundamentalProviderEvidenceError("comparison window date column mismatch")
    start_date = _canonical_date(row.get("start_date"))
    end_date = _canonical_date(row.get("end_date"))
    if start_date > end_date:
        raise FundamentalProviderEvidenceError("comparison window is reversed")
    return {
        "date_column": expected_date_column,
        "end_date": end_date,
        "start_date": start_date,
        "table": expected_table,
    }


@provider_evidence_contract
def _build_legacy_fundamental_comparison_policy(
    *,
    table_policies: Mapping[str, Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Seal all scalar, key, and winner semantics without inferred defaults."""

    if type(table_policies) is not dict or set(table_policies) != set(SOURCE_TABLES):
        raise FundamentalProviderEvidenceError("comparison table policy set is invalid")
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
        "version": LEGACY_COMPARISON_POLICY_SCHEMA,
    }
    return seal(body, identity_field="policy_id")


@provider_evidence_contract
def build_fundamental_comparison_policy(
    *,
    table_policies: Mapping[str, Mapping[str, Any]],
    comparison_windows: Mapping[str, Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Seal symmetric comparison windows and versioned winner semantics."""

    if type(table_policies) is not dict or set(table_policies) != set(SOURCE_TABLES):
        raise FundamentalProviderEvidenceError("comparison table policy set is invalid")
    if type(comparison_windows) is not dict or set(comparison_windows) != set(SOURCE_TABLES):
        raise FundamentalProviderEvidenceError("comparison window set is invalid")
    normalized_policies = [
        _table_policy_v3(table_policies[table], expected_table=table) for table in SOURCE_TABLES
    ]
    normalized_windows = [
        _comparison_window(comparison_windows[table], expected_table=table)
        for table in SOURCE_TABLES
    ]
    created = timestamp(created_at, label="created_at")
    body = {
        **common_fields(timestamp_value=created),
        "array_order_semantics": {
            "/comparison_windows": "table ASCII ascending",
            "/table_policies": "table ASCII ascending",
            "/table_policies/*/baseline_source_only_columns": "source column order",
            "/table_policies/*/canonical_key_columns": "owner semantic order",
            "/table_policies/*/column_rows": "source column order",
            "/table_policies/*/winner_completeness_columns": "owner semantic order",
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
        "comparison_windows": normalized_windows,
        "created_at": created,
        "table_policies": normalized_policies,
        "version": COMPARISON_POLICY_SCHEMA,
    }
    return seal(body, identity_field="policy_id")


@provider_evidence_contract
def validate_fundamental_comparison_policy(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="policy_id")
    version = value.get("version")
    if version == LEGACY_COMPARISON_POLICY_SCHEMA:
        require_exact_keys(value, _POLICY_FIELDS, label="Fundamental comparison policy")
    elif version == COMPARISON_POLICY_SCHEMA:
        require_exact_keys(value, _POLICY_FIELDS_V3, label="Fundamental comparison policy v3")
    else:
        raise FundamentalProviderEvidenceError("comparison policy version mismatch")
    policies = {
        row["table"]: row
        for row in _sequence(value["table_policies"], label="table_policies", maximum=6)
    }
    if version == LEGACY_COMPARISON_POLICY_SCHEMA:
        expected = _build_legacy_fundamental_comparison_policy(
            table_policies=policies,
            created_at=value["created_at"],
        )
    else:
        windows = {
            row["table"]: row
            for row in _sequence(
                value["comparison_windows"],
                label="comparison_windows",
                maximum=6,
            )
        }
        expected = build_fundamental_comparison_policy(
            table_policies=policies,
            comparison_windows=windows,
            created_at=value["created_at"],
        )
    if value != expected:
        raise FundamentalProviderEvidenceError("comparison policy replay mismatch")
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
        raise FundamentalProviderEvidenceError("boolean is not a canonical numeric value")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FundamentalProviderEvidenceError("numeric value must be finite")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise FundamentalProviderEvidenceError("numeric value is invalid") from exc
    if not parsed.is_finite():
        raise FundamentalProviderEvidenceError("numeric value must be finite")
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        result = parsed.quantize(_DECIMAL_QUANTUM)
    if result == 0:
        result = Decimal(0).quantize(_DECIMAL_QUANTUM)
    return format(result, "f")


def _canonical_integer(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise FundamentalProviderEvidenceError("boolean is not a canonical integer")
    if isinstance(value, Integral):
        return str(int(value))
    decimal = Decimal(_canonical_decimal(value))
    if decimal != decimal.to_integral_value():
        raise FundamentalProviderEvidenceError("integer value has a fractional component")
    return str(int(decimal))


def _canonical_date(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.time() != datetime.min.time():
            raise FundamentalProviderEvidenceError("date value is invalid")
        text = parsed.strftime("%Y%m%d")
    else:
        if type(value) is not str:
            raise FundamentalProviderEvidenceError("date value must be exact YYYYMMDD")
        text = value
    if len(text) != 8 or not text.isdigit():
        raise FundamentalProviderEvidenceError("date value must be exact YYYYMMDD")
    try:
        parsed_date = datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise FundamentalProviderEvidenceError("date value is invalid") from exc
    if parsed_date.strftime("%Y%m%d") != text:
        raise FundamentalProviderEvidenceError("date value is not canonical")
    return text


def _canonical_text(value: Any) -> str:
    if type(value) is not str:
        raise FundamentalProviderEvidenceError("text value must be a string")
    if unicodedata.normalize("NFC", value) != value:
        raise FundamentalProviderEvidenceError("text value must be Unicode NFC")
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
    raise FundamentalProviderEvidenceError("scalar kind is unsupported")


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
        raise FundamentalProviderEvidenceError("canonical raw row exceeds 8 MiB")
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
        raise FundamentalProviderEvidenceError("raw table must be a DataFrame")
    column_rows = list(table_policy["column_rows"])
    columns = [row["column"] for row in column_rows]
    if lane not in {"BASELINE", "VIP"}:
        raise FundamentalProviderEvidenceError("raw table lane is invalid")
    source_only = list(table_policy["baseline_source_only_columns"])
    expected_columns = columns + source_only if lane == "BASELINE" else columns
    if list(frame.columns) != expected_columns:
        raise FundamentalProviderEvidenceError("raw table column order changed")
    kinds = [row["kind"] for row in column_rows]
    key_indices = [columns.index(value) for value in table_policy["canonical_key_columns"]]
    winner_indices = [columns.index(value) for value in table_policy["winner_order_columns"]]
    completeness_indices = [
        columns.index(value) for value in table_policy.get("winner_completeness_columns", [])
    ]
    update_flag_index = columns.index("update_flag") if "update_flag" in columns else None
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
        ascii_order = _projection_bytes([list(row[index]) for index in winner_indices])
        if table_policy["winner_rule"] == "UPDATE_FLAG_THEN_COMPLETENESS_THEN_ASCII":
            if update_flag_index is None:
                raise FundamentalProviderEvidenceError("update-flag winner closure is missing")
            update_flag = row[update_flag_index]
            update_rank = 1 if update_flag == ("TEXT", "1") else 0
            completeness = sum(row[index][0] != "NULL" for index in completeness_indices)
            order_bytes = (
                bytes((update_rank,))
                + completeness.to_bytes(2, byteorder="big", signed=False)
                + ascii_order
            )
        else:
            order_bytes = ascii_order
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
    comparison_window: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if comparison_window is not None:
        baseline_frame = _windowed_frame(baseline_frame, comparison_window=comparison_window)
        vip_frame = _windowed_frame(vip_frame, comparison_window=comparison_window)
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


def _windowed_frame(
    frame: pd.DataFrame,
    *,
    comparison_window: Mapping[str, Any],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise FundamentalProviderEvidenceError("raw table must be a DataFrame")
    date_column = comparison_window["date_column"]
    if date_column not in frame.columns:
        raise FundamentalProviderEvidenceError("comparison window date column is missing")
    start_date = comparison_window["start_date"]
    end_date = comparison_window["end_date"]
    values = frame[date_column]
    raw_values = values.to_numpy(dtype=object, copy=False)
    if all(type(value) is str for value in raw_values):
        exact_shape = values.str.fullmatch(r"[0-9]{8}", na=False)
        parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
        if not bool(exact_shape.all()) or bool(parsed.isna().any()):
            raise FundamentalProviderEvidenceError("date value is invalid")
        mask = values.ge(start_date) & values.le(end_date)
    else:
        mask = [start_date <= _canonical_date(value) <= end_date for value in values.tolist()]
    return frame.loc[mask].reset_index(drop=True)


@provider_evidence_contract
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
        raise FundamentalProviderEvidenceError("raw comparison table set is invalid")
    policy_by_table = {row["table"]: row for row in validated_policy["table_policies"]}
    windows_by_table = {row["table"]: row for row in validated_policy.get("comparison_windows", [])}
    row_diff: dict[str, list[dict[str, Any]]] = {}
    value_diff: dict[str, list[dict[str, Any]]] = {}
    duplicate_diff: dict[str, dict[str, int]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for table in SOURCE_TABLES:
        result = _compare_table(
            baseline_frame=baseline_tables[table],
            vip_frame=vip_tables[table],
            table_policy=policy_by_table[table],
            comparison_window=windows_by_table.get(table),
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
        if table in windows_by_table:
            evidence[table]["comparison_window"] = windows_by_table[table]
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
