"""Canonical scalar projection and exact raw-table reconciliation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import math
from numbers import Integral
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from ...._core import (
    canonical_bytes,
    common_fields,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .models import (
    COMPARISON_POLICY_V1,
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
    "canonical_key_columns",
    "column_rows",
    "table",
    "winner_implementation_sha256",
    "winner_order_columns",
    "winner_rule",
}
_COLUMN_ROW_FIELDS = {"column", "kind"}
_DECIMAL_QUANTUM = Decimal("0.000000000001")


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
        "version": COMPARISON_POLICY_V1,
    }
    return seal(body, identity_field="policy_id")


@fundamental_v4_contract
def validate_fundamental_comparison_policy(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = validate_seal(document, identity_field="policy_id")
    require_exact_keys(value, _POLICY_FIELDS, label="Fundamental comparison policy")
    if value.get("version") != COMPARISON_POLICY_V1:
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
    return canonical_bytes(_row_json(value))


def _table_projection(
    frame: pd.DataFrame,
    *,
    table_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        raise FundamentalV4ContractError("raw table must be a DataFrame")
    column_rows = list(table_policy["column_rows"])
    columns = [row["column"] for row in column_rows]
    if list(frame.columns) != columns:
        raise FundamentalV4ContractError("raw table column order changed")
    kinds = [row["kind"] for row in column_rows]
    rows = [
        tuple(_canonical_scalar(value, kind=kinds[index]) for index, value in enumerate(raw_row))
        for raw_row in frame.itertuples(index=False, name=None)
    ]
    counter = Counter(rows)
    ordered_rows = sorted(counter, key=_row_bytes)
    multiset_rows = [{"count": counter[row], "row": _row_json(row)} for row in ordered_rows]
    key_indices = [columns.index(value) for value in table_policy["canonical_key_columns"]]
    winner_indices = [columns.index(value) for value in table_policy["winner_order_columns"]]
    by_key: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for row in rows:
        key = tuple(row[index] for index in key_indices)
        by_key.setdefault(key, []).append(row)
    winners: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for key, candidates in by_key.items():
        winners[key] = sorted(
            candidates,
            key=lambda row: canonical_bytes([list(row[index]) for index in winner_indices]),
        )[-1]
    return {
        "counter": counter,
        "duplicate_row_count": sum(count - 1 for count in counter.values()),
        "multiset_rows": multiset_rows,
        "multiset_sha256": hashlib.sha256(canonical_bytes(multiset_rows)).hexdigest(),
        "row_count": len(rows),
        "winners": winners,
    }


def _row_diffs(
    baseline: Mapping[str, Any],
    vip: Mapping[str, Any],
) -> list[dict[str, Any]]:
    keys = sorted(set(baseline["counter"]) | set(vip["counter"]), key=_row_bytes)
    rows: list[dict[str, Any]] = []
    for row in keys:
        baseline_count = baseline["counter"].get(row, 0)
        vip_count = vip["counter"].get(row, 0)
        if baseline_count != vip_count:
            rows.append(
                {
                    "baseline_count": baseline_count,
                    "row_sha256": hashlib.sha256(_row_bytes(row)).hexdigest(),
                    "vip_count": vip_count,
                }
            )
    return rows


def _winner_diffs(
    baseline: Mapping[str, Any],
    vip: Mapping[str, Any],
) -> list[dict[str, Any]]:
    keys = sorted(set(baseline["winners"]) | set(vip["winners"]), key=_row_bytes)
    rows: list[dict[str, Any]] = []
    for key in keys:
        baseline_winner = baseline["winners"].get(key)
        vip_winner = vip["winners"].get(key)
        if baseline_winner != vip_winner:
            rows.append(
                {
                    "baseline_winner_sha256": (
                        None
                        if baseline_winner is None
                        else hashlib.sha256(_row_bytes(baseline_winner)).hexdigest()
                    ),
                    "key_sha256": hashlib.sha256(_row_bytes(key)).hexdigest(),
                    "vip_winner_sha256": (
                        None
                        if vip_winner is None
                        else hashlib.sha256(_row_bytes(vip_winner)).hexdigest()
                    ),
                }
            )
    return rows


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
        baseline = _table_projection(
            baseline_tables[table],
            table_policy=policy_by_table[table],
        )
        vip = _table_projection(
            vip_tables[table],
            table_policy=policy_by_table[table],
        )
        row_diff[table] = _row_diffs(baseline, vip)
        value_diff[table] = _winner_diffs(baseline, vip)
        duplicate_diff[table] = {
            "baseline_duplicate_row_count": baseline["duplicate_row_count"],
            "vip_duplicate_row_count": vip["duplicate_row_count"],
        }
        evidence[table] = {
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
