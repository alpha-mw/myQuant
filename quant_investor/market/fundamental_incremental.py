"""Fail-closed, append-only derivation for CN Fundamental successors.

This module deliberately does not fetch provider data and never writes the
canonical Fundamental pointer.  It consumes an already sealed provider/support
bundle, proves the predecessor boundary, derives only the open successor
window, and can materialise an isolated generation for the promotion layer.

The ordinary Fundamental merge is intentionally not imported here.  A safe
successor treats the predecessor tables as an immutable prefix: support rows at
or before the predecessor cutoff are validation-only, while rows after that
cutoff may create an append-only suffix.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.factors.pit_fundamentals import normalize_ts_code
from quant_investor.market.fundamental_provider_contract import (
    _scalar_token,
    assert_frame_semantics_equal,
    canonical_json_sha256,
    frame_fingerprint,
    frame_logical_schema,
)


SUCCESSOR_DERIVATION_CONTRACT = "cn-fundamental-derivation.safe-successor.v1"
SUCCESSOR_READINESS_SCHEMA = "cn-fundamental-readiness.safe-successor.v1"
SUCCESSOR_PROVENANCE_SCHEMA = "cn-fundamental-primary-provenance.v3"
SUCCESSOR_PROVENANCE_STATUS = "verified_safe_successor_mixed"
SUCCESSOR_PLAN_SCHEMA = "cn-fundamental-safe-successor-plan.v1"
SUCCESSOR_CHAIN_SCHEMA = "cn-fundamental-successor-chain.v1"
SUCCESSOR_KEYSET_SCHEMA = "cn-fundamental-successor-keyset-closure.v1"
SUCCESSOR_RESOURCE_SCHEMA = "cn-fundamental-successor-resource-preflight.v1"
SUCCESSOR_PROVIDER_MANIFEST_SCHEMA = "cn-fundamental-safe-successor-provider.v1"
SUCCESSOR_SUPPORT_PREFIX_VALIDATION_MODE = "validation_only"
SUCCESSOR_APPEND_FIRST_MODE = "immutable_predecessor_append_first"
SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SCHEMA = (
    "cn-fundamental-successor-financial-dependency.v1"
)
SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT = {
    "schema_version": SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SCHEMA,
    "state_key": ["table", "ts_code", "end_date"],
    "physical_update_policy": (
        "exact-collapse; highest update_flag per physical class; "
        "material winner ties block"
    ),
    "event_order": ["availability_date", "ts_code"],
    "atomic_batch_key": ["ts_code", "availability_date"],
    "atomic_batch_update_order": ["table", "end_date"],
    "event_trigger": (
        "any fina_indicator/income/balancesheet/cashflow row derives each "
        "affected ts_code+end_date after the whole availability batch is visible"
    ),
    "period_read_set": {
        "same_period": [
            "fina_indicator",
            "income",
            "balancesheet",
            "cashflow",
        ],
        "fallback": "income at end_date minus one calendar year",
    },
    "period_winner": ["availability_date", "end_date"],
    "daily_period_carry": (
        "latest period winner available on or before trade_date"
    ),
    "forecast_winner": ["availability_date", "forecast_end_date"],
    "daily_forecast_merge": (
        "latest forecast winner available on or before trade_date"
    ),
    "daily_basic_and_size": (
        "target-session daily_basic exact bar keyset; size rank is a "
        "cross-sectional lane independent of financial state"
    ),
    "lineage_identity": (
        "source_row_bindings for every present same-period table plus "
        "availability-aware previous_year_income row binding"
    ),
}
SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256 = canonical_json_sha256(
    SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT
)

# Long aliases are useful to callers that group constants by domain.
FUNDAMENTAL_SAFE_SUCCESSOR_DERIVATION_CONTRACT = SUCCESSOR_DERIVATION_CONTRACT
FUNDAMENTAL_SAFE_SUCCESSOR_READINESS_SCHEMA = SUCCESSOR_READINESS_SCHEMA
FUNDAMENTAL_SAFE_SUCCESSOR_PROVENANCE_SCHEMA = SUCCESSOR_PROVENANCE_SCHEMA

FUNDAMENTAL_TABLES = (
    "fundamental_period",
    "fundamental_daily",
    "fundamental_quarantine",
)
RAW_TABLES = (
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
    "daily_basic",
    "forecast",
)
FINANCIAL_TABLES = RAW_TABLES[:4]
PERIOD_VALUE_FIELDS = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "free_cashflow",
)
FORECAST_VALUE_FIELDS = (
    "forecast_revision",
    "forecast_end_date",
    "forecast_ann_date",
    "forecast_type",
    "forecast_summary",
    "forecast_change_reason",
    "forecast_source",
    "forecast_fetched_at",
    "forecast_ingest_run_id",
)
PERIOD_KEY_FIELDS = ("ts_code", "end_date", "availability_date")
DAILY_KEY_FIELDS = ("ts_code", "trade_date")
NONBAR_REASONS = (
    "suspended",
    "inactive",
    "delisted",
    "prelisting",
)
PERMANENT_SUPPORT_REFERENCE_NAMES = (
    "predecessor_pointer",
    "predecessor_manifest",
    "support_manifest",
)
MAX_SUCCESSOR_CHAIN_DEPTH = 4096
SUPPORT_STREAM_BATCH_ROWS = 2_048
SUPPORT_STREAM_BATCH_BYTES = 16 * 1024 * 1024
SUPPORT_SYMBOL_MAX_ROWS = 100_000
SUPPORT_SYMBOL_MAX_BYTES = 64 * 1024 * 1024
DERIVATION_ACCUMULATOR_BUDGET_SCHEMA = (
    "cn-fundamental-successor-derivation-accumulator-budget.v1"
)


class SafeSuccessorError(RuntimeError):
    """One deterministic stop in the safe-successor derivation contract."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class SuccessorBundle:
    """Pure derivation output; no canonical state has been touched."""

    parent_tables: Mapping[str, pd.DataFrame | Path]
    predecessor_binding: Mapping[str, Any]
    parent_cutoff: str
    target_cutoff: str
    run_id: str
    period_suffix: pd.DataFrame
    daily_suffix: pd.DataFrame
    candidate_tables: Mapping[str, pd.DataFrame] | None
    plan_metadata: Mapping[str, Any]
    keyset_closure: Mapping[str, Any]
    successor_chain: Mapping[str, Any]
    lineage: Mapping[str, Any]
    derivation_evidence: Mapping[str, Any]
    readiness: Mapping[str, Any]
    raw_table_fingerprints: Mapping[str, str]
    resource_preflight: Mapping[str, Any]


@dataclass(frozen=True)
class SuccessorStagingCapture:
    """Exact identities produced by :func:`stage_successor_generation`."""

    generation_id: str
    staging_root: Path
    pointer_path: Path
    pointer_bytes: bytes
    pointer_sha256: str
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str
    table_paths: Mapping[str, Path]
    table_sha256: Mapping[str, str]
    provider_evidence_files: Mapping[str, str]
    predecessor_binding: Mapping[str, Any]
    target_bindings: Mapping[str, Any]
    provenance_binding_sha256: str


def _fail(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise SafeSuccessorError(code, message, details=details)


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail("NON_CANONICAL_EVIDENCE", "successor evidence is not canonical JSON")
        raise AssertionError from exc


def _strict_date(value: Any, *, label: str) -> str:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        parsed = pd.Timestamp(value)
    else:
        text = str(value or "").strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        digits = "".join(character for character in text if character.isdigit())
        if len(digits) != 8:
            _fail("INVALID_DATE", f"{label} must be exact YYYYMMDD", details={"value": text})
        parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        _fail("INVALID_DATE", f"{label} is not a valid date")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _optional_date(value: Any) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    try:
        return _strict_date(value, label="date")
    except SafeSuccessorError:
        return ""


def _period(value: Any, *, label: str = "end_date") -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) != 8:
        _fail("INVALID_PERIOD", f"{label} must be exact YYYYMMDD", details={"value": text})
    return _strict_date(digits, label=label)


def _timestamp(date_text: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(date_text, format="%Y%m%d"))


def _availability(row: Mapping[str, Any], *, forecast: bool = False) -> str:
    columns = ("ann_date", "f_ann_date", "availability_date") if forecast else (
        "f_ann_date",
        "ann_date",
        "availability_date",
    )
    for column in columns:
        resolved = _optional_date(row.get(column))
        if resolved:
            return resolved
    _fail("MISSING_AVAILABILITY", "support row has no exact availability date")
    return ""


def _number(value: Any) -> float:
    if value is None or value is pd.NA:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError, InvalidOperation):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _first_number(row: Mapping[str, Any] | None, names: Sequence[str]) -> float:
    if row is None:
        return float("nan")
    for name in names:
        value = _number(row.get(name))
        if math.isfinite(value):
            return value
    return float("nan")


def _ratio(value: Any) -> float:
    number = _number(value)
    if not math.isfinite(number):
        return float("nan")
    return number / 100.0 if abs(number) > 2.0 else number


def _nullable(value: Any) -> bool:
    return _scalar_token(value)[0] == "null"


def _same_scalar(left: Any, right: Any, *, date_value: bool = False) -> bool:
    if _nullable(left) and _nullable(right):
        return True
    if date_value:
        return _optional_date(left) == _optional_date(right)
    return _scalar_token(left) == _scalar_token(right)


def _safe_symbol(value: Any) -> str:
    symbol = normalize_ts_code(value)
    if not symbol:
        _fail("INVALID_SYMBOL", "support row has an invalid symbol")
    return symbol


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(stat.S_IFMT(value.st_mode)),
    )


def _read_frame(source: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source).expanduser().resolve(strict=True)
    return pd.read_parquet(path)


def _source_path(source: pd.DataFrame | Path | str) -> Path | None:
    if isinstance(source, pd.DataFrame):
        return None
    return Path(source).expanduser().resolve(strict=True)


def _key(symbol: Any, date_value: Any) -> tuple[str, str]:
    return (_safe_symbol(symbol), _strict_date(date_value, label="key date"))


def _keys(values: Any, *, label: str) -> set[tuple[str, str]]:
    if values is None:
        return set()
    output: set[tuple[str, str]] = set()
    if isinstance(values, Mapping):
        iterable: Iterable[Any] = values.keys()
    else:
        iterable = values
    for item in iterable:
        if isinstance(item, str) and "|" in item:
            symbol, date_value = item.split("|", 1)
        elif isinstance(item, Mapping):
            symbol = item.get("ts_code", item.get("symbol"))
            date_value = item.get("trade_date", item.get("date"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            symbol, date_value = item
        else:
            _fail("INVALID_KEYSET", f"{label} contains an invalid key")
        normalized = _key(symbol, date_value)
        if normalized in output:
            _fail("DUPLICATE_KEYSET_KEY", f"{label} contains duplicate keys")
        output.add(normalized)
    return output


def _serialized_keys(values: Iterable[tuple[str, str]]) -> list[str]:
    return [f"{symbol}|{date_value}" for symbol, date_value in sorted(values)]


def _mapping_hash(mapping: Mapping[str, Any], field: str) -> str:
    body = dict(mapping)
    claimed = str(body.pop(field, "") or "").strip().lower()
    if not _valid_sha256(claimed) or canonical_json_sha256(body) != claimed:
        _fail("EVIDENCE_BINDING_MISMATCH", f"{field} is invalid")
    return claimed


def _streaming_support_store(value: Any) -> bool:
    return callable(getattr(value, "iter_batches", None)) and isinstance(
        getattr(value, "table_fingerprints", None), Mapping
    )


def _support_fingerprints(raw_tables: Mapping[str, Any]) -> dict[str, str]:
    if set(raw_tables) != set(RAW_TABLES):
        _fail("INCOMPLETE_SUPPORT_TABLE_SET", "support bundle must contain all raw tables")
    if _streaming_support_store(raw_tables):
        fingerprints = getattr(raw_tables, "table_fingerprints")
        observed = {
            str(table): str(digest).strip().lower()
            for table, digest in dict(fingerprints).items()
        }
        if set(observed) != set(RAW_TABLES) or any(
            not _valid_sha256(value) for value in observed.values()
        ):
            _fail("SUPPORT_FRAME_TAMPER", "streamed table fingerprints are incomplete")
        return observed
    output: dict[str, str] = {}
    for table in RAW_TABLES:
        frame = raw_tables[table]
        if not isinstance(frame, pd.DataFrame):
            _fail("SUPPORT_TABLE_ACCESS_INVALID", f"{table} is not a DataFrame")
        output[table] = frame_fingerprint(frame)
    return output


def _iter_support_batches(
    raw_tables: Mapping[str, Any],
    table: str,
) -> Iterator[pd.DataFrame]:
    if table not in RAW_TABLES:
        _fail("SUPPORT_TABLE_ACCESS_INVALID", f"unknown support table: {table}")
    if _streaming_support_store(raw_tables):
        iterator = getattr(raw_tables, "iter_batches")
        for batch in iterator(
            table,
            batch_rows=SUPPORT_STREAM_BATCH_ROWS,
            batch_bytes=SUPPORT_STREAM_BATCH_BYTES,
        ):
            if not isinstance(batch, pd.DataFrame) or len(batch) > SUPPORT_STREAM_BATCH_ROWS:
                _fail("SUPPORT_STREAM_BATCH_INVALID", f"{table} batch is invalid")
            estimated = int(batch.memory_usage(index=True, deep=True).sum())
            if estimated > SUPPORT_STREAM_BATCH_BYTES:
                _fail("SUPPORT_STREAM_BATCH_LIMIT", f"{table} batch exceeds byte limit")
            yield batch
        return
    frame = raw_tables[table]
    if not isinstance(frame, pd.DataFrame):
        _fail("SUPPORT_TABLE_ACCESS_INVALID", f"{table} is not a DataFrame")
    for start in range(0, len(frame), SUPPORT_STREAM_BATCH_ROWS):
        batch = frame.iloc[start : start + SUPPORT_STREAM_BATCH_ROWS].copy()
        estimated = int(batch.memory_usage(index=True, deep=True).sum())
        if estimated > SUPPORT_STREAM_BATCH_BYTES:
            _fail("SUPPORT_STREAM_BATCH_LIMIT", f"{table} batch exceeds byte limit")
        yield batch


def _iter_support_symbol_groups(
    raw_tables: Mapping[str, Any],
    table: str,
    *,
    maximum_symbol_bytes: int = SUPPORT_SYMBOL_MAX_BYTES,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    if (
        type(maximum_symbol_bytes) is not int
        or maximum_symbol_bytes < SUPPORT_STREAM_BATCH_BYTES
        or maximum_symbol_bytes > SUPPORT_SYMBOL_MAX_BYTES
    ):
        _fail("SUPPORT_SYMBOL_RESOURCE_POLICY_INVALID", table)
    current_symbol = ""
    current_rows: list[dict[str, Any]] = []
    current_bytes = 0
    previous_symbol = ""
    for batch in _iter_support_batches(raw_tables, table):
        for value in batch.to_dict("records"):
            row = dict(value)
            symbol = _safe_symbol(row.get("ts_code"))
            if previous_symbol and symbol < previous_symbol:
                _fail("SUPPORT_STREAM_SORT_ORDER", f"{table} is not sorted by symbol")
            previous_symbol = symbol
            if current_symbol and symbol != current_symbol:
                yield current_symbol, current_rows
                current_rows = []
                current_bytes = 0
            current_symbol = symbol
            encoded = json.dumps(
                {
                    str(key): list(_material_token(item))
                    for key, item in sorted(row.items(), key=lambda pair: str(pair[0]))
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            current_rows.append(row)
            current_bytes += len(encoded)
            if (
                len(current_rows) > SUPPORT_SYMBOL_MAX_ROWS
                or current_bytes > maximum_symbol_bytes
            ):
                _fail("SUPPORT_SYMBOL_RESOURCE_LIMIT", f"{table} symbol group is too large")
    if current_symbol:
        yield current_symbol, current_rows


class _AccumulatorGuard:
    def __init__(self, budget: Mapping[str, Any]) -> None:
        payload = dict(budget)
        _mapping_hash(payload, "binding_sha256")
        required = {
            "binding_sha256",
            "daily_basic_row_limit",
            "effective_memory_headroom_bytes",
            "forecast_anchor_row_limit",
            "forecast_delta_row_limit",
            "period_anchor_row_limit",
            "period_delta_row_limit",
            "period_lineage_row_limit",
            "post_capture_receipt_sha256",
            "schema_version",
            "status",
            "total_accumulator_byte_limit",
        }
        if (
            set(payload) != required
            or payload.get("schema_version") != DERIVATION_ACCUMULATOR_BUDGET_SCHEMA
            or payload.get("status") != "PASS"
            or not _valid_sha256(payload.get("post_capture_receipt_sha256"))
        ):
            _fail("DERIVATION_RESOURCE_BUDGET_INVALID", "budget contract is invalid")
        lanes = {
            "daily_basic": "daily_basic_row_limit",
            "forecast_anchor": "forecast_anchor_row_limit",
            "forecast_delta": "forecast_delta_row_limit",
            "period_anchor": "period_anchor_row_limit",
            "period_delta": "period_delta_row_limit",
            "period_lineage": "period_lineage_row_limit",
        }
        limits: dict[str, int] = {}
        for lane, field in lanes.items():
            value = payload.get(field)
            if type(value) is not int or value < 0:
                _fail("DERIVATION_RESOURCE_BUDGET_INVALID", f"invalid {field}")
            limits[lane] = value
        byte_limit = payload.get("total_accumulator_byte_limit")
        headroom = payload.get("effective_memory_headroom_bytes")
        if (
            type(byte_limit) is not int
            or byte_limit < 1
            or type(headroom) is not int
            or headroom < 128 * 1024 * 1024
            or byte_limit > headroom // 2
        ):
            _fail("DERIVATION_RESOURCE_BUDGET_INVALID", "memory budget is invalid")
        self._payload = payload
        self._limits = limits
        self._byte_limit = byte_limit
        self._counts = {lane: 0 for lane in lanes}
        self._bytes = {lane: 0 for lane in lanes}
        self._total_bytes = 0
        self._maximum_row_bytes = 0

    @property
    def maximum_symbol_source_bytes(self) -> int:
        return min(
            SUPPORT_SYMBOL_MAX_BYTES,
            int(self._payload["effective_memory_headroom_bytes"]) // 8,
        )

    @staticmethod
    def _encoded_size(value: Any) -> int:
        if isinstance(value, Mapping):
            body: dict[str, Any] = {
                str(key): list(_material_token(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        else:
            body = {"value": str(value)}
        return len(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def add(self, lane: str, value: Any) -> None:
        if lane not in self._limits:
            _fail("DERIVATION_RESOURCE_LANE_INVALID", lane)
        size = self._encoded_size(value)
        next_count = self._counts[lane] + 1
        next_total = self._total_bytes + size
        if next_count > self._limits[lane] or next_total > self._byte_limit:
            _fail(
                "DERIVATION_ACCUMULATOR_LIMIT_EXCEEDED",
                f"{lane} exceeded its sealed row/byte budget",
            )
        self._counts[lane] = next_count
        self._bytes[lane] += size
        self._total_bytes = next_total
        self._maximum_row_bytes = max(self._maximum_row_bytes, size)

    def receipt(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": "cn-fundamental-successor-derivation-accumulator.v1",
            "budget_binding_sha256": self._payload["binding_sha256"],
            "counts": dict(self._counts),
            "bytes": dict(self._bytes),
            "total_bytes": self._total_bytes,
            "maximum_row_bytes": self._maximum_row_bytes,
            "status": "PASS",
        }
        body["binding_sha256"] = canonical_json_sha256(body)
        return body


def seal_support_plan(
    raw_tables: Mapping[str, Any],
    *,
    parent_cutoff: str,
    target_cutoff: str,
    permanent_support_refs: Mapping[str, Any] | None = None,
    boundary_non_reachability: Mapping[str, Sequence[str]] | None = None,
    absence_proofs: Sequence[Mapping[str, Any]] = (),
    support_prefix_mode: str = SUCCESSOR_SUPPORT_PREFIX_VALIDATION_MODE,
    historical_taint_registry_sha256: str = "",
    append_first_income_dependencies: Sequence[Mapping[str, Any]] = (),
    append_first_financial_dependencies: Sequence[Mapping[str, Any]] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical seal expected by :func:`assemble_safe_successor`.

    This helper does not make an acquisition complete; it only seals the exact
    frames and counters that an acquisition layer has already audited.
    """

    parent = _strict_date(parent_cutoff, label="parent_cutoff")
    target = _strict_date(target_cutoff, label="target_cutoff")
    if target <= parent:
        _fail("INVALID_SUCCESSOR_WINDOW", "target cutoff must follow parent cutoff")
    prefix_mode = str(support_prefix_mode or "").strip()
    if prefix_mode not in {
        SUCCESSOR_SUPPORT_PREFIX_VALIDATION_MODE,
        SUCCESSOR_APPEND_FIRST_MODE,
    }:
        _fail("INVALID_SUPPORT_PREFIX_MODE", prefix_mode)
    registry_sha256 = str(historical_taint_registry_sha256 or "").strip().lower()
    if prefix_mode == SUCCESSOR_APPEND_FIRST_MODE:
        if not _valid_sha256(registry_sha256):
            _fail(
                "HISTORICAL_TAINT_REGISTRY_REQUIRED",
                "append-first mode requires one sealed historical-taint registry",
            )
    elif registry_sha256:
        _fail(
            "HISTORICAL_TAINT_REGISTRY_WITH_PREFIX_REPLAY",
            "historical-taint isolation is exclusive to append-first mode",
        )
    dependencies: list[dict[str, str]] = []
    dependency_values = [
        {"table": "income", **dict(value)}
        for value in append_first_income_dependencies
    ] + [dict(value) for value in append_first_financial_dependencies]
    for value in dependency_values:
        if not isinstance(value, Mapping) or set(value) != {
            "end_date",
            "table",
            "ts_code",
        }:
            _fail(
                "APPEND_FIRST_FINANCIAL_DEPENDENCY_INVALID",
                "financial dependency must contain table, ts_code and end_date",
            )
        table = str(value["table"])
        if table not in {"balancesheet", "cashflow", "income"}:
            _fail(
                "APPEND_FIRST_FINANCIAL_DEPENDENCY_INVALID",
                "financial dependency table cannot seed a production fallback",
            )
        dependencies.append(
            {
                "end_date": _period(value["end_date"]),
                "table": table,
                "ts_code": _safe_symbol(value["ts_code"]),
            }
        )
    dependencies.sort(
        key=lambda row: (row["table"], row["ts_code"], row["end_date"])
    )
    if len(
        {(row["table"], row["ts_code"], row["end_date"]) for row in dependencies}
    ) != len(dependencies):
        _fail(
            "APPEND_FIRST_FINANCIAL_DEPENDENCY_DUPLICATE",
            "financial dependencies must be unique",
        )
    if prefix_mode != SUCCESSOR_APPEND_FIRST_MODE and dependencies:
        _fail(
            "APPEND_FIRST_FINANCIAL_DEPENDENCY_MODE_MISMATCH",
            "bounded financial dependencies require append-first mode",
        )
    raw_fingerprints = _support_fingerprints(raw_tables)
    body: dict[str, Any] = {
        "schema_version": SUCCESSOR_PLAN_SCHEMA,
        "status": "sealed",
        "parent_cutoff": parent,
        "target_cutoff": target,
        "tables": list(RAW_TABLES),
        "requests_failed": 0,
        "requests_malformed": 0,
        "responses_has_more": 0,
        "schema_failures": 0,
        "duplicate_conflicts": 0,
        "support_prefix_mode": prefix_mode,
        "support_prefix_complete": (
            prefix_mode == SUCCESSOR_SUPPORT_PREFIX_VALIDATION_MODE
        ),
        "predecessor_prefix_trusted": prefix_mode == SUCCESSOR_APPEND_FIRST_MODE,
        "historical_taint_registry_sha256": registry_sha256,
        "append_first_financial_dependencies": dependencies,
        "delta_window_complete": True,
        "raw_table_fingerprints": raw_fingerprints,
        "financial_dependency_contract_sha256": (
            SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
        ),
        "permanent_support_refs": dict(permanent_support_refs or {}),
        "boundary_non_reachability": {
            lane: sorted({_safe_symbol(symbol) for symbol in values})
            for lane, values in dict(boundary_non_reachability or {}).items()
        },
        "absence_proofs": [dict(item) for item in absence_proofs],
    }
    if extra:
        for key, value in extra.items():
            if key in body:
                _fail("PLAN_FIELD_OVERRIDE", f"extra plan metadata cannot replace {key}")
            body[key] = value
    body["plan_sha256"] = canonical_json_sha256(body)
    return body


def _validate_plan(
    raw_tables: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    parent_cutoff: str,
    target_cutoff: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = dict(plan)
    _mapping_hash(payload, "plan_sha256")
    prefix_mode = str(payload.get("support_prefix_mode") or "")
    prefix_state_valid = (
        prefix_mode == SUCCESSOR_SUPPORT_PREFIX_VALIDATION_MODE
        and payload.get("support_prefix_complete") is True
        and payload.get("predecessor_prefix_trusted") is False
        and payload.get("historical_taint_registry_sha256") == ""
        and payload.get("append_first_financial_dependencies") == []
    ) or (
        prefix_mode == SUCCESSOR_APPEND_FIRST_MODE
        and payload.get("support_prefix_complete") is False
        and payload.get("predecessor_prefix_trusted") is True
        and _valid_sha256(payload.get("historical_taint_registry_sha256"))
        and isinstance(payload.get("append_first_financial_dependencies"), list)
        and payload.get("support_start")
        == (datetime.strptime(parent_cutoff, "%Y%m%d").date() + timedelta(days=1)).strftime(
            "%Y%m%d"
        )
    )
    dependencies = payload.get("append_first_financial_dependencies")
    if not isinstance(dependencies, list):
        _fail("UNSEALED_SUPPORT_PLAN", "income dependency keyset is absent")
    replayed_dependencies: list[dict[str, str]] = []
    for value in dependencies:
        if not isinstance(value, Mapping) or set(value) != {
            "end_date",
            "table",
            "ts_code",
        }:
            _fail("UNSEALED_SUPPORT_PLAN", "financial dependency keyset is invalid")
        table = str(value["table"])
        if table not in {"balancesheet", "cashflow", "income"}:
            _fail("UNSEALED_SUPPORT_PLAN", "financial dependency table is invalid")
        replayed_dependencies.append(
            {
                "end_date": _period(value["end_date"]),
                "table": table,
                "ts_code": _safe_symbol(value["ts_code"]),
            }
        )
    replayed_dependencies.sort(
        key=lambda row: (row["table"], row["ts_code"], row["end_date"])
    )
    if replayed_dependencies != dependencies or len(
        {
            (row["table"], row["ts_code"], row["end_date"])
            for row in replayed_dependencies
        }
    ) != len(replayed_dependencies):
        _fail("UNSEALED_SUPPORT_PLAN", "income dependency keyset is not canonical")
    if (
        payload.get("schema_version") != SUCCESSOR_PLAN_SCHEMA
        or payload.get("status") != "sealed"
        or payload.get("parent_cutoff") != parent_cutoff
        or payload.get("target_cutoff") != target_cutoff
        or payload.get("tables") != list(RAW_TABLES)
        or not prefix_state_valid
        or payload.get("delta_window_complete") is not True
        or payload.get("financial_dependency_contract_sha256")
        != SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
    ):
        _fail("UNSEALED_SUPPORT_PLAN", "support plan contract is incomplete")
    for counter in (
        "requests_failed",
        "requests_malformed",
        "responses_has_more",
        "schema_failures",
        "duplicate_conflicts",
    ):
        if type(payload.get(counter)) is not int or payload[counter] != 0:
            _fail("PROVIDER_AUDIT_NOT_CLEAN", f"support plan {counter} must be zero")
    declared = dict(payload.get("raw_table_fingerprints", {}) or {})
    actual = _support_fingerprints(raw_tables)
    if set(declared) != set(RAW_TABLES) or declared != actual:
        _fail("SUPPORT_FRAME_TAMPER", "support frame fingerprints do not match the seal")
    refs = dict(payload.get("permanent_support_refs", {}) or {})
    if set(refs) != set(PERMANENT_SUPPORT_REFERENCE_NAMES):
        _fail(
            "INCOMPLETE_SUPPORT_REFERENCE_SET",
            "permanent support refs must bind predecessor pointer/manifest and support manifest",
        )
    for name, reference in refs.items():
        if not isinstance(reference, Mapping):
            _fail("INVALID_SUPPORT_REFERENCE", f"support reference is invalid: {name}")
        digest = str(reference.get("sha256") or "").lower()
        relative = Path(str(reference.get("path") or ""))
        if (
            not _valid_sha256(digest)
            or relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
        ):
            _fail("INVALID_SUPPORT_REFERENCE", f"support reference SHA is invalid: {name}")
    return payload, actual


def _material_token(value: Any) -> tuple[str, str]:
    try:
        return _scalar_token(value)
    except TypeError:
        return ("json", json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _row_binding(row: Mapping[str, Any]) -> str:
    body = {
        str(key): list(_material_token(value))
        for key, value in sorted(row.items(), key=lambda item: str(item[0]))
        if not str(key).startswith("__")
    }
    return canonical_json_sha256(body)


def _normalize_financial_table(
    frame: pd.DataFrame,
    *,
    table: str,
    target_cutoff: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    required = {"ts_code", "end_date"}
    if not required.issubset(frame.columns):
        _fail("SUPPORT_SCHEMA_MISMATCH", f"{table} is missing required columns")
    records: list[dict[str, Any]] = []
    for original in frame.to_dict("records"):
        symbol = _safe_symbol(original.get("ts_code"))
        end_date = _period(original.get("end_date"))
        availability = _availability(original)
        if end_date > availability:
            _fail(
                "REPORT_PERIOD_AFTER_AVAILABILITY",
                f"{table} report period follows availability",
                details={"symbol": symbol, "end_date": end_date, "availability": availability},
            )
        if availability > target_cutoff:
            _fail(
                "FUTURE_SUPPORT_ROW",
                f"{table} contains a row beyond the target cutoff",
                details={"symbol": symbol, "availability": availability},
            )
        normalized = dict(original)
        normalized.update(
            {
                "ts_code": symbol,
                "end_date": end_date,
                "availability_date": availability,
                "__table": table,
            }
        )
        normalized["__row_binding"] = _row_binding(normalized)
        records.append(normalized)
    return _deduplicate_material_events(
        records,
        key_fields=("ts_code", "end_date", "availability_date", "__table"),
        label=table,
    )


def _deduplicate_material_events(
    records: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    winners: dict[tuple[str, ...], dict[str, Any]] = {}
    bindings: dict[tuple[str, ...], str] = {}
    for value in records:
        row = dict(value)
        key = tuple(str(row.get(field) or "") for field in key_fields)
        binding = str(row.get("__row_binding") or _row_binding(row))
        if key in winners and bindings[key] != binding:
            _fail(
                "MATERIAL_EVENT_TIE",
                f"{label} contains materially different rows for one event key",
                details={"key": list(key)},
            )
        winners[key] = row
        bindings[key] = binding
    return [winners[key] for key in sorted(winners)]


def _normalize_forecast_table(
    frame: pd.DataFrame,
    *,
    target_cutoff: str,
    run_id: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if "ts_code" not in frame.columns or "end_date" not in frame.columns:
        _fail("SUPPORT_SCHEMA_MISMATCH", "forecast is missing required columns")
    records: list[dict[str, Any]] = []
    for original in frame.to_dict("records"):
        symbol = _safe_symbol(original.get("ts_code"))
        end_date = _period(original.get("end_date"), label="forecast end_date")
        availability = _availability(original, forecast=True)
        if end_date > availability:
            _fail(
                "REPORT_PERIOD_AFTER_AVAILABILITY",
                "forecast report period follows availability",
                details={"symbol": symbol, "end_date": end_date, "availability": availability},
            )
        if availability > target_cutoff:
            _fail("FUTURE_SUPPORT_ROW", "forecast contains a row beyond target")
        p_min = _number(original.get("p_change_min"))
        p_max = _number(original.get("p_change_max"))
        available_changes = [value for value in (p_min, p_max) if math.isfinite(value)]
        revision = (
            sum(available_changes) / len(available_changes) / 100.0
            if available_changes
            else float("nan")
        )
        if not math.isfinite(revision):
            profit_min = _number(original.get("net_profit_min"))
            profit_max = _number(original.get("net_profit_max"))
            available_profits = [
                value for value in (profit_min, profit_max) if math.isfinite(value)
            ]
            last_parent = _number(original.get("last_parent_net"))
            if available_profits and math.isfinite(last_parent) and abs(last_parent) > 0:
                revision = (
                    sum(available_profits) / len(available_profits) - last_parent
                ) / abs(last_parent)
        normalized = {
            **dict(original),
            "ts_code": symbol,
            "forecast_end_date": end_date,
            "availability_date": availability,
            "forecast_ann_date": availability,
            "forecast_revision": revision,
            "forecast_type": str(original.get("type") or "").strip(),
            "forecast_summary": str(original.get("summary") or "").strip(),
            "forecast_change_reason": str(original.get("change_reason") or "").strip(),
            "forecast_source": str(original.get("source") or "live_tushare_safe_successor"),
            "forecast_fetched_at": str(original.get("fetched_at") or ""),
            "forecast_ingest_run_id": run_id,
            "__table": "forecast",
        }
        normalized["__row_binding"] = _row_binding(normalized)
        records.append(normalized)
    return _deduplicate_material_events(
        records,
        key_fields=("ts_code", "forecast_end_date", "availability_date", "__table"),
        label="forecast",
    )


def _absence_is_proven(
    plan: Mapping[str, Any],
    *,
    table: str,
    symbol: str,
    end_date: str,
    availability: str,
) -> bool:
    for value in list(plan.get("absence_proofs", []) or []):
        if not isinstance(value, Mapping):
            continue
        proof = dict(value)
        if (
            proof.get("status") == "PROVEN_ABSENT"
            and str(proof.get("table") or "") == table
            and _safe_symbol(proof.get("symbol")) == symbol
            and _period(proof.get("end_date")) == end_date
            and _strict_date(proof.get("available_through"), label="available_through")
            >= availability
            and _valid_sha256(proof.get("evidence_sha256"))
        ):
            return True
    return False


def _dependency(
    state: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    table: str,
    symbol: str,
    end_date: str,
    availability: str,
    required: bool,
) -> Mapping[str, Any] | None:
    row = state.get((symbol, end_date))
    if row is None and required and not _absence_is_proven(
        plan,
        table=table,
        symbol=symbol,
        end_date=end_date,
        availability=availability,
    ):
        support_start_value = plan.get("support_start")
        support_start = (
            _strict_date(support_start_value, label="support_start")
            if support_start_value
            else ""
        )
        code = (
            "SUPPORT_START_ANCHOR_UNCLOSED"
            if support_start and end_date < support_start
            else "HIDDEN_DEPENDENCY_UNPROVEN"
        )
        _fail(
            code,
            "a fallback dependency is neither present nor proven absent",
            details={
                "table": table,
                "symbol": symbol,
                "end_date": end_date,
                "availability": availability,
            },
        )
    return row


def _derive_period_endpoint(
    *,
    symbol: str,
    end_date: str,
    availability: str,
    state: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    plan: Mapping[str, Any],
    enforce_hidden_dependencies: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fi = state["fina_indicator"].get((symbol, end_date))
    income = state["income"].get((symbol, end_date))
    balance = state["balancesheet"].get((symbol, end_date))
    cashflow = state["cashflow"].get((symbol, end_date))

    profit = _first_number(income, ("n_income_attr_p", "n_income"))
    total_assets = _first_number(balance, ("total_assets",))
    total_liab = _first_number(balance, ("total_liab",))
    ocf = _first_number(cashflow, ("n_cashflow_act",))
    capex = _first_number(cashflow, ("c_pay_acq_const_fiolta",))
    direct_fcf = _first_number(cashflow, ("free_cashflow",))
    direct_debt = _ratio(fi.get("debt_to_assets") if fi else None)
    direct_yoy = _ratio(fi.get("netprofit_yoy") if fi else None)
    direct_ocf_profit = _ratio(fi.get("ocf_to_profit") if fi else None)

    previous_period = (pd.Timestamp(end_date) - pd.DateOffset(years=1)).strftime("%Y%m%d")
    previous_income_required = (
        enforce_hidden_dependencies
        and not math.isfinite(direct_yoy)
        and math.isfinite(profit)
    )
    previous_income = _dependency(
        state["income"],
        plan=plan,
        table="income",
        symbol=symbol,
        end_date=previous_period,
        availability=availability,
        required=previous_income_required,
    )
    previous_profit = _first_number(previous_income, ("n_income_attr_p", "n_income"))
    balance_required = enforce_hidden_dependencies and not math.isfinite(direct_debt)
    if balance_required:
        _dependency(
            state["balancesheet"],
            plan=plan,
            table="balancesheet",
            symbol=symbol,
            end_date=end_date,
            availability=availability,
            required=True,
        )
    cashflow_required = enforce_hidden_dependencies and (
        not math.isfinite(direct_ocf_profit) or not math.isfinite(direct_fcf)
    )
    if cashflow_required:
        _dependency(
            state["cashflow"],
            plan=plan,
            table="cashflow",
            symbol=symbol,
            end_date=end_date,
            availability=availability,
            required=True,
        )

    free_cashflow = (
        direct_fcf
        if math.isfinite(direct_fcf)
        else (
            ocf - capex
            if math.isfinite(ocf) and math.isfinite(capex)
            else float("nan")
        )
    )
    fallback_debt = (
        total_liab / total_assets
        if math.isfinite(total_liab) and math.isfinite(total_assets) and total_assets > 0
        else float("nan")
    )
    fallback_yoy = (
        (profit - previous_profit) / previous_profit
        if math.isfinite(profit) and math.isfinite(previous_profit) and previous_profit > 0
        else float("nan")
    )
    fallback_ocf_profit = (
        ocf / profit
        if math.isfinite(ocf) and math.isfinite(profit) and profit > 0
        else float("nan")
    )
    fcf_profit = (
        free_cashflow / profit
        if math.isfinite(free_cashflow) and math.isfinite(profit) and profit > 0
        else float("nan")
    )
    source_rows = [row for row in (fi, income, balance, cashflow) if row is not None]
    sources = sorted(
        {
            str(row.get("source") or "live_tushare_safe_successor").strip()
            for row in source_rows
        }
    )
    fetched = sorted(
        {str(row.get("fetched_at") or "").strip() for row in source_rows if row.get("fetched_at")}
    )
    output = {
        "ts_code": symbol,
        "end_date": end_date,
        "availability_date": _timestamp(availability),
        "source_version": pd.Timestamp(availability).strftime("%Y-%m-%d"),
        "source": ";".join(sources) or "live_tushare_safe_successor",
        "fetched_at": max(fetched) if fetched else "",
        "fin_roe": _ratio(
            fi.get("roe_dt")
            if fi and not _nullable(fi.get("roe_dt"))
            else (fi.get("roe") if fi else None)
        ),
        "fin_roa": _ratio(fi.get("roa") if fi else None),
        "fin_debt_to_assets": direct_debt if math.isfinite(direct_debt) else fallback_debt,
        "fin_net_profit_yoy": direct_yoy if math.isfinite(direct_yoy) else fallback_yoy,
        "fin_ocf_to_profit": (
            direct_ocf_profit
            if math.isfinite(direct_ocf_profit)
            else fallback_ocf_profit
        ),
        "fin_fcf_to_profit": fcf_profit,
        "free_cashflow": free_cashflow,
    }
    lineage = {
        "symbol": symbol,
        "end_date": end_date,
        "availability_date": availability,
        "atomic_batch": f"{symbol}|{availability}",
        "source_row_bindings": {
            table: str(row.get("__row_binding") or "")
            for table, row in (
                ("fina_indicator", fi),
                ("income", income),
                ("balancesheet", balance),
                ("cashflow", cashflow),
            )
            if row is not None
        },
        "previous_year_income": {
            "end_date": previous_period,
            "row_binding": (
                str(previous_income.get("__row_binding") or "")
                if previous_income
                else ""
            ),
            "available_as_of_event": previous_income is not None,
            "required_for_fallback": previous_income_required,
            "absence_proven": (
                previous_income is None
                and previous_income_required
                and _absence_is_proven(
                    plan,
                    table="income",
                    symbol=symbol,
                    end_date=previous_period,
                    availability=availability,
                )
            ),
        },
        "dependency_requirements": [
            {
                "table": table,
                "end_date": dependency_end,
                "required": required,
                "row_binding": (
                    str(row.get("__row_binding") or "") if row is not None else ""
                ),
                "bounded_support": (
                    row is not None
                    and bool(plan.get("support_start"))
                    and _strict_date(
                        row.get("availability_date"),
                        label="dependency row availability",
                    )
                    < _strict_date(plan["support_start"], label="support_start")
                ),
                "absence_proven": (
                    row is None
                    and required
                    and _absence_is_proven(
                        plan,
                        table=table,
                        symbol=symbol,
                        end_date=dependency_end,
                        availability=availability,
                    )
                ),
            }
            for table, dependency_end, required, row in (
                ("income", previous_period, previous_income_required, previous_income),
                ("balancesheet", end_date, balance_required, balance),
                ("cashflow", end_date, cashflow_required, cashflow),
            )
            if required
        ],
        "metric_methods": {
            "fin_debt_to_assets": "direct" if math.isfinite(direct_debt) else "fallback",
            "fin_net_profit_yoy": (
                "direct"
                if math.isfinite(direct_yoy)
                else "availability_aware_previous_year_fallback"
            ),
            "fin_ocf_to_profit": "direct" if math.isfinite(direct_ocf_profit) else "fallback",
            "free_cashflow": "direct" if math.isfinite(direct_fcf) else "ocf_minus_capex_fallback",
        },
    }
    return output, lineage


def _derive_event_graph(
    financial_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    plan: Mapping[str, Any],
    parent_cutoff: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if plan.get("support_prefix_mode") == SUCCESSOR_APPEND_FIRST_MODE:
        symbols = sorted(
            {
                str(row["ts_code"])
                for table in FINANCIAL_TABLES
                for row in financial_rows[table]
            }
        )
        append_records: list[dict[str, Any]] = []
        append_lineage: list[dict[str, Any]] = []
        for symbol in symbols:
            symbol_rows = {
                table: [
                    row
                    for row in financial_rows[table]
                    if str(row["ts_code"]) == symbol
                ]
                for table in FINANCIAL_TABLES
            }
            _boundary, delta, delta_lineage, _derived = _derive_symbol_event_window(
                symbol_rows,
                symbol=symbol,
                plan=plan,
                parent_cutoff=parent_cutoff,
                target_cutoff=str(plan["target_cutoff"]),
            )
            append_records.extend(delta)
            append_lineage.extend(delta_lineage)
        return append_records, append_lineage
    events: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for table in FINANCIAL_TABLES:
        for value in financial_rows[table]:
            row = dict(value)
            key = (str(row["ts_code"]), str(row["availability_date"]))
            events.setdefault(key, []).append(row)
    state: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {
        table: {} for table in FINANCIAL_TABLES
    }
    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for (symbol, availability), batch in sorted(
        events.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        # The whole (symbol, availability) batch becomes visible atomically.
        affected: set[str] = set()
        for row in sorted(batch, key=lambda item: (str(item["__table"]), str(item["end_date"]))):
            table = str(row["__table"])
            endpoint = (symbol, str(row["end_date"]))
            state[table][endpoint] = row
            affected.add(str(row["end_date"]))
        for end_date in sorted(affected):
            record, row_lineage = _derive_period_endpoint(
                symbol=symbol,
                end_date=end_date,
                availability=availability,
                state=state,
                plan=plan,
                enforce_hidden_dependencies=availability > parent_cutoff,
            )
            records.append(record)
            lineage.append(row_lineage)
    return records, lineage


def _derive_symbol_event_window(
    financial_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    symbol: str,
    plan: Mapping[str, Any],
    parent_cutoff: str,
    target_cutoff: str,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    """Replay one symbol while retaining only its boundary winner and delta."""

    ordered = sorted(
        (
            row
            for table in FINANCIAL_TABLES
            for row in financial_rows[table]
        ),
        key=lambda row: (
            str(row["availability_date"]),
            str(row["__table"]),
            str(row["end_date"]),
        ),
    )
    state: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {
        table: {} for table in FINANCIAL_TABLES
    }
    append_first = plan.get("support_prefix_mode") == SUCCESSOR_APPEND_FIRST_MODE
    declared_dependencies = {
        (str(value["table"]), str(value["ts_code"]), str(value["end_date"]))
        for value in list(plan.get("append_first_financial_dependencies", []) or [])
        if str(value.get("ts_code") or "") == symbol
    }
    declared_absences = {
        (str(value["table"]), str(value["symbol"]), str(value["end_date"]))
        for value in list(plan.get("absence_proofs", []) or [])
        if value.get("status") == "PROVEN_ABSENT"
        and str(value.get("symbol") or "") == symbol
    }
    observed_dependencies: set[tuple[str, str, str]] = set()
    boundary_winner: dict[str, Any] | None = None
    delta_records: list[dict[str, Any]] = []
    delta_lineage: list[dict[str, Any]] = []
    derived_records = 0
    for availability, grouped_rows in groupby(
        ordered,
        key=lambda row: str(row["availability_date"]),
    ):
        batch = list(grouped_rows)
        affected: set[str] = set()
        for row in batch:
            table = str(row["__table"])
            endpoint = (symbol, str(row["end_date"]))
            if append_first and availability <= parent_cutoff:
                dependency = (table, symbol, str(row["end_date"]))
                if dependency not in declared_dependencies:
                    _fail(
                        "APPEND_FIRST_PREFIX_ROW_PRESENT",
                        "append-first source contains an undeclared predecessor row",
                    )
                observed_dependencies.add(dependency)
            state[table][endpoint] = row
            affected.add(str(row["end_date"]))
        for end_date in sorted(affected):
            if append_first and availability <= parent_cutoff:
                continue
            record, row_lineage = _derive_period_endpoint(
                symbol=symbol,
                end_date=end_date,
                availability=availability,
                state=state,
                plan=plan,
                enforce_hidden_dependencies=availability > parent_cutoff,
            )
            derived_records += 1
            if availability <= parent_cutoff:
                boundary_winner = _winner(
                    [value for value in (boundary_winner, record) if value is not None],
                    end_field="end_date",
                    lane="period",
                )
            elif availability <= target_cutoff:
                delta_records.append(record)
                delta_lineage.append(row_lineage)
    if append_first:
        required_dependencies = {
            (str(requirement["table"]), symbol, str(requirement["end_date"]))
            for row in delta_lineage
            for requirement in row["dependency_requirements"]
            if requirement.get("required") is True
            and str(requirement.get("row_binding") or "")
            and requirement.get("bounded_support") is True
        }
        required_absences = {
            (str(requirement["table"]), symbol, str(requirement["end_date"]))
            for row in delta_lineage
            for requirement in row["dependency_requirements"]
            if requirement.get("required") is True
            and requirement.get("absence_proven") is True
        }
        if observed_dependencies != declared_dependencies:
            _fail(
                "APPEND_FIRST_FINANCIAL_DEPENDENCY_EVIDENCE_MISMATCH",
                "declared financial support was not captured exactly",
            )
        if required_dependencies != declared_dependencies:
            _fail(
                "APPEND_FIRST_FINANCIAL_DEPENDENCY_NOT_EXACTLY_CONSUMED",
                "bounded financial support must equal the production fallback read-set",
            )
        if required_absences != declared_absences:
            _fail(
                "APPEND_FIRST_FINANCIAL_ABSENCE_NOT_EXACTLY_CONSUMED",
                "bounded financial absence proofs must equal the production fallback read-set",
            )
    return boundary_winner, delta_records, delta_lineage, derived_records


def replay_successor_event_trace(
    *,
    financial_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    symbol: str,
    parent_cutoff: str,
    target_cutoff: str,
    support_start: str = "",
) -> dict[str, Any]:
    """Replay the exact production financial kernel for one taint subject.

    This diagnostic surface deliberately calls the same normalization,
    atomic-event, read-set, fallback, winner and lineage code used by
    ``assemble_safe_successor``.  It does not create a SuccessorBundle and has
    no staging or publication capability.
    """

    resolved_symbol = _safe_symbol(symbol)
    parent = _strict_date(parent_cutoff, label="parent_cutoff")
    target = _strict_date(target_cutoff, label="target_cutoff")
    if target <= parent:
        _fail("INVALID_SUCCESSOR_WINDOW", "target cutoff must follow parent cutoff")
    if set(financial_rows) != set(FINANCIAL_TABLES):
        _fail("SUPPORT_TABLE_SET_MISMATCH", "financial trace requires four tables")
    normalized: dict[str, list[dict[str, Any]]] = {}
    input_fingerprints: dict[str, str] = {}
    for table in FINANCIAL_TABLES:
        frame = pd.DataFrame([dict(row) for row in financial_rows[table]])
        values = _normalize_financial_table(
            frame,
            table=table,
            target_cutoff=target,
        )
        if any(str(row["ts_code"]) != resolved_symbol for row in values):
            _fail("SUPPORT_SYMBOL_STREAM_DRIFT", "trace crossed subject identity")
        normalized[table] = values
        input_fingerprints[table] = frame_fingerprint(pd.DataFrame(values))
    plan = {
        "support_start": (
            _strict_date(support_start, label="support_start")
            if support_start
            else ""
        ),
        "financial_dependency_contract_sha256": (
            SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
        ),
    }
    boundary, delta, lineage, derived = _derive_symbol_event_window(
        normalized,
        symbol=resolved_symbol,
        plan=plan,
        parent_cutoff=parent,
        target_cutoff=target,
    )
    if plan["financial_dependency_contract_sha256"] != (
        SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
    ):
        _fail("DEPENDENCY_CONTRACT_DRIFT", "financial dependency contract changed")
    receipt = {
        "schema_version": "cn-fundamental-successor-event-trace.v1",
        "dependency_contract_sha256": (
            SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
        ),
        "symbol": resolved_symbol,
        "parent_cutoff": parent,
        "target_cutoff": target,
        "support_start": plan["support_start"],
        "input_frame_fingerprints": input_fingerprints,
        "boundary_present": boundary is not None,
        "boundary_frame_fingerprint": (
            frame_fingerprint(pd.DataFrame([boundary]))
            if boundary is not None
            else frame_fingerprint(pd.DataFrame())
        ),
        "delta_frame_fingerprint": frame_fingerprint(pd.DataFrame(delta)),
        "lineage_frame_fingerprint": canonical_json_sha256(
            {"lineage": lineage}
        ),
        "derived_record_count": derived,
        "delta_record_count": len(delta),
        "delta_lineage_count": len(lineage),
        "kernel": "fundamental_incremental._derive_symbol_event_window",
    }
    receipt["trace_sha256"] = canonical_json_sha256(receipt)
    return {
        "boundary_winner": boundary,
        "delta_records": delta,
        "delta_lineage": lineage,
        "normalized_financial_rows": normalized,
        "trace_receipt": receipt,
    }


def _derive_streamed_financial(
    raw_tables: Mapping[str, Any],
    *,
    accumulator: _AccumulatorGuard,
    plan: Mapping[str, Any],
    parent_cutoff: str,
    target_cutoff: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    streams = {
        table: iter(
            _iter_support_symbol_groups(
                raw_tables,
                table,
                maximum_symbol_bytes=accumulator.maximum_symbol_source_bytes,
            )
        )
        for table in FINANCIAL_TABLES
    }
    heads: dict[str, tuple[str, list[dict[str, Any]]] | None] = {}
    for table, stream in streams.items():
        heads[table] = next(stream, None)
    delta_records: list[dict[str, Any]] = []
    delta_lineage: list[dict[str, Any]] = []
    boundary_winners: dict[str, dict[str, Any]] = {}
    high_rows = 0
    high_bytes = 0
    high_derived_records = 0
    symbols = 0
    while any(value is not None for value in heads.values()):
        symbol = min(
            value[0]
            for value in heads.values()
            if value is not None
        )
        symbols += 1
        financial: dict[str, list[dict[str, Any]]] = {}
        resident_rows = 0
        resident_bytes = 0
        for table in FINANCIAL_TABLES:
            head = heads[table]
            if head is not None and head[0] == symbol:
                source_rows = head[1]
                heads[table] = next(streams[table], None)
            else:
                source_rows = []
            resident_rows += len(source_rows)
            resident_bytes += sum(
                len(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
                for row in source_rows
            )
            financial[table] = _normalize_financial_table(
                pd.DataFrame(source_rows),
                table=table,
                target_cutoff=target_cutoff,
            )
        if (
            resident_rows > SUPPORT_SYMBOL_MAX_ROWS
            or resident_bytes > accumulator.maximum_symbol_source_bytes
        ):
            _fail("SUPPORT_SYMBOL_RESOURCE_LIMIT", "financial symbol state is too large")
        high_rows = max(high_rows, resident_rows)
        high_bytes = max(high_bytes, resident_bytes)
        (
            boundary_winner,
            symbol_delta_records,
            symbol_delta_lineage,
            derived_records,
        ) = _derive_symbol_event_window(
            financial,
            symbol=symbol,
            plan=plan,
            parent_cutoff=parent_cutoff,
            target_cutoff=target_cutoff,
        )
        high_derived_records = max(high_derived_records, derived_records)
        if any(str(row.get("ts_code")) != symbol for row in symbol_delta_records):
            _fail("SUPPORT_SYMBOL_STREAM_DRIFT", "financial symbol state crossed groups")
        if boundary_winner is not None:
            accumulator.add("period_anchor", boundary_winner)
            boundary_winners[symbol] = boundary_winner
        for row, row_lineage in zip(
            symbol_delta_records,
            symbol_delta_lineage,
            strict=True,
        ):
            accumulator.add("period_delta", row)
            accumulator.add("period_lineage", row_lineage)
            delta_records.append(row)
            delta_lineage.append(row_lineage)
    delta_records.sort(
        key=lambda row: (
            _strict_date(row.get("availability_date"), label="period availability"),
            str(row.get("ts_code") or ""),
            _period(row.get("end_date")),
        )
    )
    delta_lineage.sort(
        key=lambda row: (
            _strict_date(row.get("availability_date"), label="lineage availability"),
            str(row.get("symbol") or ""),
            _period(row.get("end_date")),
        )
    )
    receipt = {
        "schema_version": "cn-fundamental-support-symbol-stream.v1",
        "financial_symbols": symbols,
        "maximum_resident_source_rows": high_rows,
        "maximum_resident_source_bytes": high_bytes,
        "maximum_symbol_derived_records_processed": high_derived_records,
        "maximum_symbol_rows": SUPPORT_SYMBOL_MAX_ROWS,
        "maximum_symbol_bytes": accumulator.maximum_symbol_source_bytes,
        "same_symbol_availability_atomic": True,
        "full_table_getitem_used": False,
        "retained_prefix_winners": len(boundary_winners),
        "retained_delta_records": len(delta_records),
        "retained_delta_lineage": len(delta_lineage),
        "full_support_records_retained": False,
    }
    receipt["binding_sha256"] = canonical_json_sha256(receipt)
    return delta_records, delta_lineage, boundary_winners, receipt


def _normalize_streamed_forecast(
    raw_tables: Mapping[str, Any],
    *,
    accumulator: _AccumulatorGuard,
    parent_cutoff: str,
    target_cutoff: str,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    boundary_winners: dict[str, dict[str, Any]] = {}
    high_rows = 0
    symbols = 0
    for symbol, rows in _iter_support_symbol_groups(
        raw_tables,
        "forecast",
        maximum_symbol_bytes=accumulator.maximum_symbol_source_bytes,
    ):
        symbols += 1
        high_rows = max(high_rows, len(rows))
        replay = _forecast_records(
            _normalize_forecast_table(
                pd.DataFrame(rows),
                target_cutoff=target_cutoff,
                run_id=run_id,
            )
        )
        prefix = [
            row
            for row in replay
            if _strict_date(row.get("availability_date"), label="forecast availability")
            <= parent_cutoff
        ]
        winner = _winner(prefix, end_field="forecast_end_date", lane="forecast")
        if winner is not None:
            accumulator.add("forecast_anchor", winner)
            boundary_winners[symbol] = winner
        for row in replay:
            availability = _strict_date(
                row.get("availability_date"),
                label="forecast availability",
            )
            if parent_cutoff < availability <= target_cutoff:
                accumulator.add("forecast_delta", row)
                output.append(row)
    receipt = {
        "schema_version": "cn-fundamental-support-forecast-stream.v1",
        "symbols": symbols,
        "maximum_resident_source_rows": high_rows,
        "maximum_symbol_rows": SUPPORT_SYMBOL_MAX_ROWS,
        "full_table_getitem_used": False,
        "retained_prefix_winners": len(boundary_winners),
        "retained_delta_records": len(output),
        "full_support_records_retained": False,
    }
    receipt["binding_sha256"] = canonical_json_sha256(receipt)
    return output, boundary_winners, receipt


def _normalize_streamed_daily_basic(
    raw_tables: Mapping[str, Any],
    *,
    accumulator: _AccumulatorGuard,
    parent_cutoff: str,
    target_cutoff: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    high_rows = 0
    symbols = 0
    for _symbol, rows in _iter_support_symbol_groups(
        raw_tables,
        "daily_basic",
        maximum_symbol_bytes=accumulator.maximum_symbol_source_bytes,
    ):
        symbols += 1
        high_rows = max(high_rows, len(rows))
        normalized, batch_keys = _normalize_daily_basic(
            pd.DataFrame(rows),
            parent_cutoff=parent_cutoff,
            target_cutoff=target_cutoff,
        )
        if keys.intersection(batch_keys):
            _fail("DUPLICATE_DAILY_BASIC_KEY", "daily_basic streamed keys overlap")
        for row in normalized:
            accumulator.add("daily_basic", row)
        keys.update(batch_keys)
        output.extend(normalized)
    receipt = {
        "schema_version": "cn-fundamental-support-daily-stream.v1",
        "symbols": symbols,
        "maximum_resident_source_rows": high_rows,
        "maximum_symbol_rows": SUPPORT_SYMBOL_MAX_ROWS,
        "full_table_getitem_used": False,
    }
    receipt["binding_sha256"] = canonical_json_sha256(receipt)
    return output, keys, receipt


def _latest_parent_seeds(
    parent_daily: pd.DataFrame | Path,
    *,
    parent_cutoff: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    columns = [
        "ts_code",
        "trade_date",
        "end_date",
        "availability_date",
        "source_version",
        "source",
        "fetched_at",
        "sector",
        *PERIOD_VALUE_FIELDS,
        *FORECAST_VALUE_FIELDS,
    ]
    available_columns: list[str]
    batches: Iterator[pd.DataFrame]
    if isinstance(parent_daily, pd.DataFrame):
        available_columns = list(parent_daily.columns)
        batches = iter(
            [
                parent_daily.loc[
                    :, [column for column in columns if column in parent_daily.columns]
                ]
            ]
        )
    else:
        parquet = pq.ParquetFile(Path(parent_daily))
        available_columns = list(parquet.schema_arrow.names)
        selected = [column for column in columns if column in available_columns]
        batches = (
            batch.to_pandas()
            for batch in parquet.iter_batches(columns=selected, batch_size=100_000)
        )
    if not {"ts_code", "trade_date"}.issubset(available_columns):
        _fail("PARENT_SCHEMA_MISMATCH", "parent daily table is missing keys")
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for batch in batches:
        for row in batch.to_dict("records"):
            symbol = _safe_symbol(row.get("ts_code"))
            trade_date = _strict_date(row.get("trade_date"), label="parent trade_date")
            if trade_date > parent_cutoff:
                _fail("PARENT_PREFIX_HAS_FUTURE_ROW", "parent daily extends beyond its cutoff")
            if symbol not in latest or trade_date > latest[symbol][0]:
                latest[symbol] = (trade_date, row)
    period: dict[str, dict[str, Any]] = {}
    forecast: dict[str, dict[str, Any]] = {}
    for symbol, (_date, row) in latest.items():
        if row.get("end_date") not in (None, "") and not _nullable(row.get("end_date")):
            period[symbol] = {
                key: row.get(key)
                for key in (
                    "ts_code",
                    "end_date",
                    "availability_date",
                    "source_version",
                    "source",
                    "fetched_at",
                    "sector",
                    *PERIOD_VALUE_FIELDS,
                )
                if key in row
            }
            period[symbol]["ts_code"] = symbol
        if "forecast_revision" in row and not _nullable(row.get("forecast_revision")):
            forecast[symbol] = {
                key: row.get(key)
                for key in ("ts_code", "availability_date", *FORECAST_VALUE_FIELDS)
                if key in row
            }
            forecast[symbol]["ts_code"] = symbol
            forecast[symbol]["availability_date"] = (
                row.get("forecast_ann_date") or row.get("availability_date")
            )
    return period, forecast, available_columns


def _winner(
    candidates: Sequence[Mapping[str, Any]],
    *,
    end_field: str,
    lane: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            _strict_date(item.get("availability_date"), label=f"{lane} availability"),
            _period(item.get(end_field), label=end_field),
        ),
    )
    best_key = (
        _strict_date(ordered[-1].get("availability_date"), label=f"{lane} availability"),
        _period(ordered[-1].get(end_field), label=end_field),
    )
    tied = [
        item
        for item in ordered
        if (
            _strict_date(item.get("availability_date"), label=f"{lane} availability"),
            _period(item.get(end_field), label=end_field),
        )
        == best_key
    ]
    if len(tied) > 1:
        bindings = {_row_binding(item) for item in tied}
        if len(bindings) > 1:
            _fail("MATERIAL_WINNER_TIE", f"{lane} has a materially different winner tie")
    return tied[-1]


def _period_anchor_equal(parent: Mapping[str, Any], replay: Mapping[str, Any]) -> bool:
    if _period(parent.get("end_date")) != _period(replay.get("end_date")):
        return False
    if not _same_scalar(
        parent.get("availability_date"), replay.get("availability_date"), date_value=True
    ):
        return False
    return all(_same_scalar(parent.get(field), replay.get(field)) for field in PERIOD_VALUE_FIELDS)


def successor_period_anchor_equal(
    parent: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> bool:
    """Public diagnostic wrapper around the production period comparator."""

    return _period_anchor_equal(parent, replay)


def successor_period_winner(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Use the production period winner ordering for diagnostic replay."""

    return _winner(candidates, end_field="end_date", lane="period")


def successor_financial_row_binding(
    *,
    table: str,
    row: Mapping[str, Any],
    target_cutoff: str,
) -> str:
    """Return the production lineage identity for one provider source row."""

    if table not in FINANCIAL_TABLES:
        _fail("SUPPORT_TABLE_SET_MISMATCH", "row binding requires a financial table")
    normalized = _normalize_financial_table(
        pd.DataFrame([dict(row)]),
        table=table,
        target_cutoff=_strict_date(target_cutoff, label="target_cutoff"),
    )
    if len(normalized) != 1:
        _fail("SUPPORT_ROW_BINDING_INVALID", "one input row did not produce one binding")
    return str(normalized[0]["__row_binding"])


def _forecast_anchor_equal(parent: Mapping[str, Any], replay: Mapping[str, Any]) -> bool:
    if _period(parent.get("forecast_end_date"), label="forecast_end_date") != _period(
        replay.get("forecast_end_date"), label="forecast_end_date"
    ):
        return False
    parent_ann = parent.get("forecast_ann_date") or parent.get("availability_date")
    replay_ann = replay.get("forecast_ann_date") or replay.get("availability_date")
    if not _same_scalar(parent_ann, replay_ann, date_value=True):
        return False
    return all(
        _same_scalar(parent.get(field), replay.get(field))
        for field in FORECAST_VALUE_FIELDS
        if field
        not in {
            "forecast_end_date",
            "forecast_ann_date",
            "forecast_fetched_at",
            "forecast_ingest_run_id",
        }
    )


def _prove_boundary(
    *,
    parent_period: Mapping[str, Mapping[str, Any]],
    parent_forecast: Mapping[str, Mapping[str, Any]],
    replay_period: Sequence[Mapping[str, Any]],
    replay_forecast: Sequence[Mapping[str, Any]],
    relevant_symbols: set[str],
    plan: Mapping[str, Any],
    parent_cutoff: str,
) -> dict[str, Any]:
    declared = dict(plan.get("boundary_non_reachability", {}) or {})
    nonreachable = {
        lane: {_safe_symbol(symbol) for symbol in list(declared.get(lane, []) or [])}
        for lane in ("period", "forecast")
    }
    replay_period_by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in replay_period:
        if _strict_date(row["availability_date"], label="period availability") <= parent_cutoff:
            replay_period_by_symbol.setdefault(str(row["ts_code"]), []).append(row)
    replay_forecast_by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in replay_forecast:
        if _strict_date(row["availability_date"], label="forecast availability") <= parent_cutoff:
            replay_forecast_by_symbol.setdefault(str(row["ts_code"]), []).append(row)

    results: dict[str, dict[str, str]] = {"period": {}, "forecast": {}}
    for symbol in sorted(relevant_symbols):
        for lane, parent_map, replay_map, end_field, comparator in (
            (
                "period",
                parent_period,
                replay_period_by_symbol,
                "end_date",
                _period_anchor_equal,
            ),
            (
                "forecast",
                parent_forecast,
                replay_forecast_by_symbol,
                "forecast_end_date",
                _forecast_anchor_equal,
            ),
        ):
            parent = parent_map.get(symbol)
            replay = _winner(replay_map.get(symbol, []), end_field=end_field, lane=lane)
            if parent is None and replay is None:
                results[lane][symbol] = "LANE_NON_REACHABLE_EMPTY"
                continue
            if replay is None:
                if symbol not in nonreachable[lane]:
                    _fail(
                        "BOUNDARY_LANE_UNPROVEN",
                        f"{lane} parent seed cannot be reached from support",
                        details={"symbol": symbol},
                    )
                results[lane][symbol] = "LANE_NON_REACHABLE_DECLARED"
                continue
            if parent is None or not comparator(parent, replay):
                _fail(
                    "BOUNDARY_ANCHOR_MISMATCH",
                    f"{lane} replay does not equal the immediate predecessor anchor",
                    details={"symbol": symbol},
                )
            if symbol in nonreachable[lane]:
                _fail(
                    "FALSE_NON_REACHABILITY_CLAIM",
                    f"{lane} is reachable despite a non-reachability claim",
                    details={"symbol": symbol},
                )
            results[lane][symbol] = "ANCHOR_EQUAL"
    body: dict[str, Any] = {
        "parent_cutoff": parent_cutoff,
        "period": results["period"],
        "forecast": results["forecast"],
        "period_rule": "anchor_equality_or_lane_non_reachability",
        "forecast_rule": "anchor_equality_or_lane_non_reachability",
    }
    body["binding_sha256"] = canonical_json_sha256(body)
    return body


def _prove_boundary_winners(
    *,
    parent_period: Mapping[str, Mapping[str, Any]],
    parent_forecast: Mapping[str, Mapping[str, Any]],
    replay_period: Mapping[str, Mapping[str, Any]],
    replay_forecast: Mapping[str, Mapping[str, Any]],
    relevant_symbols: set[str],
    plan: Mapping[str, Any],
    parent_cutoff: str,
) -> dict[str, Any]:
    """Prove the seam from one bounded prefix winner per symbol and lane."""

    declared = dict(plan.get("boundary_non_reachability", {}) or {})
    nonreachable = {
        lane: {_safe_symbol(symbol) for symbol in list(declared.get(lane, []) or [])}
        for lane in ("period", "forecast")
    }
    results: dict[str, dict[str, str]] = {"period": {}, "forecast": {}}
    for symbol in sorted(relevant_symbols):
        for lane, parent_map, replay_map, comparator in (
            (
                "period",
                parent_period,
                replay_period,
                _period_anchor_equal,
            ),
            (
                "forecast",
                parent_forecast,
                replay_forecast,
                _forecast_anchor_equal,
            ),
        ):
            parent = parent_map.get(symbol)
            replay = replay_map.get(symbol)
            if parent is None and replay is None:
                results[lane][symbol] = "LANE_NON_REACHABLE_EMPTY"
                continue
            if replay is None:
                if symbol not in nonreachable[lane]:
                    _fail(
                        "BOUNDARY_LANE_UNPROVEN",
                        f"{lane} parent seed cannot be reached from support",
                        details={"symbol": symbol},
                    )
                results[lane][symbol] = "LANE_NON_REACHABLE_DECLARED"
                continue
            if parent is None or not comparator(parent, replay):
                _fail(
                    "BOUNDARY_ANCHOR_MISMATCH",
                    f"{lane} replay does not equal the immediate predecessor anchor",
                    details={"symbol": symbol},
                )
            if symbol in nonreachable[lane]:
                _fail(
                    "FALSE_NON_REACHABILITY_CLAIM",
                    f"{lane} is reachable despite a non-reachability claim",
                    details={"symbol": symbol},
                )
            results[lane][symbol] = "ANCHOR_EQUAL"
    body: dict[str, Any] = {
        "parent_cutoff": parent_cutoff,
        "period": results["period"],
        "forecast": results["forecast"],
        "period_rule": "anchor_equality_or_lane_non_reachability",
        "forecast_rule": "anchor_equality_or_lane_non_reachability",
        "prefix_replay_mode": "ONE_WINNER_PER_SYMBOL_LANE",
    }
    body["binding_sha256"] = canonical_json_sha256(body)
    return body


def _normalize_daily_basic(
    frame: pd.DataFrame,
    *,
    parent_cutoff: str,
    target_cutoff: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if frame.empty:
        return [], set()
    if "ts_code" not in frame.columns or "trade_date" not in frame.columns:
        _fail("SUPPORT_SCHEMA_MISMATCH", "daily_basic is missing required columns")
    records: list[dict[str, Any]] = []
    for original in frame.to_dict("records"):
        symbol = _safe_symbol(original.get("ts_code"))
        trade_date = _strict_date(original.get("trade_date"), label="daily_basic trade_date")
        if trade_date > target_cutoff:
            _fail("FUTURE_SUPPORT_ROW", "daily_basic contains a row beyond target")
        if "total_mv_rmb" in original and not _nullable(original.get("total_mv_rmb")):
            raw_cap = original.get("total_mv_rmb")
            multiplier = Decimal(1)
        else:
            raw_cap = original.get("total_mv")
            multiplier = Decimal(10_000)
        try:
            cap = Decimal(str(raw_cap)) * multiplier
        except (InvalidOperation, TypeError, ValueError):
            cap = Decimal("NaN")
        normalized = {
            **dict(original),
            "ts_code": symbol,
            "trade_date": trade_date,
            "__total_mv_decimal": cap,
        }
        normalized["__row_binding"] = _row_binding(normalized)
        records.append(normalized)
    records = _deduplicate_material_events(
        records,
        key_fields=("ts_code", "trade_date"),
        label="daily_basic",
    )
    delta = [row for row in records if parent_cutoff < str(row["trade_date"]) <= target_cutoff]
    return delta, {(str(row["ts_code"]), str(row["trade_date"])) for row in delta}


def _validate_keyset_input(
    closure: Mapping[str, Any],
    *,
    daily_basic_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    payload = dict(closure)
    if payload.get("schema_version") not in {None, SUCCESSOR_KEYSET_SCHEMA}:
        _fail("INVALID_KEYSET", "keyset closure schema is invalid")
    observed = _keys(payload.get("observed_bar_keys", []), label="observed_bar_keys")
    declared_daily = _keys(payload.get("daily_basic_keys", []), label="daily_basic_keys")
    if observed != declared_daily or observed != daily_basic_keys:
        _fail("OBSERVED_DAILY_BASIC_KEYSET_MISMATCH", "bar and daily_basic keys differ")
    reasons = {
        reason: _keys(payload.get(f"{reason}_keys", []), label=f"{reason}_keys")
        for reason in NONBAR_REASONS
    }
    seen = set(observed)
    for reason in NONBAR_REASONS:
        overlap = seen.intersection(reasons[reason])
        if overlap:
            _fail(
                "NON_EXCLUSIVE_KEYSET_CLASSIFICATION",
                "keyset reason classes overlap",
                details={"reason": reason, "sample": _serialized_keys(sorted(overlap)[:3])},
            )
        seen.update(reasons[reason])
    nonbar = _keys(payload.get("nonbar_keys", []), label="nonbar_keys")
    reason_union = set().union(*reasons.values()) if reasons else set()
    if nonbar and nonbar != reason_union:
        _fail("NONBAR_REASON_CLOSURE_MISMATCH", "nonbar keys do not equal reason union")
    if not nonbar:
        nonbar = reason_union
    true_missing = _keys(payload.get("true_missing_keys", []), label="true_missing_keys")
    if true_missing:
        _fail(
            "TRUE_MISSING_NOT_ZERO",
            "safe successor requires zero true-missing keys",
            details={"keys": _serialized_keys(true_missing)},
        )
    expected = _keys(payload.get("expected_scope_keys", []), label="expected_scope_keys")
    if expected and expected != observed.union(nonbar):
        _fail("EXPECTED_SCOPE_KEYSET_MISMATCH", "scope keys do not close to observed plus nonbar")
    body: dict[str, Any] = {
        "schema_version": SUCCESSOR_KEYSET_SCHEMA,
        "observed_bar_keys": _serialized_keys(observed),
        "daily_basic_keys": _serialized_keys(declared_daily),
        **{f"{reason}_keys": _serialized_keys(reasons[reason]) for reason in NONBAR_REASONS},
        "nonbar_keys": _serialized_keys(nonbar),
        "true_missing_keys": [],
        "expected_scope_keys": _serialized_keys(expected or observed.union(nonbar)),
        "classification_counts": {
            "observed": len(observed),
            **{reason: len(reasons[reason]) for reason in NONBAR_REASONS},
            "nonbar": len(nonbar),
            "true_missing": 0,
        },
    }
    body["input_binding_sha256"] = canonical_json_sha256(body)
    return body


def _forecast_records(
    normalized: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in normalized:
        row = dict(value)
        if not math.isfinite(_number(row.get("forecast_revision"))):
            continue
        output.append(
            {
                "ts_code": row["ts_code"],
                "forecast_end_date": row["forecast_end_date"],
                "availability_date": _timestamp(str(row["availability_date"])),
                "forecast_ann_date": str(row["forecast_ann_date"]),
                "forecast_revision": float(row["forecast_revision"]),
                "forecast_type": row["forecast_type"],
                "forecast_summary": row["forecast_summary"],
                "forecast_change_reason": row["forecast_change_reason"],
                "forecast_source": row["forecast_source"],
                "forecast_fetched_at": row["forecast_fetched_at"],
                "forecast_ingest_run_id": row["forecast_ingest_run_id"],
                "__row_binding": row["__row_binding"],
            }
        )
    return output


def _decimal_size_buckets(
    rows: Sequence[Mapping[str, Any]],
    *,
    trade_date: str,
) -> dict[str, str]:
    if len(rows) < 3:
        _fail(
            "SIZE_BUCKET_SESSION_TOO_SMALL",
            "a session needs at least three valid PIT-investable rows",
            details={"trade_date": trade_date, "rows": len(rows)},
        )
    values: list[tuple[str, Decimal]] = []
    for row in rows:
        value = row.get("__total_mv_decimal")
        if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
            _fail(
                "INVALID_TOTAL_MV",
                "size bucketing requires a positive finite Decimal total_mv",
                details={"symbol": row.get("ts_code"), "trade_date": trade_date},
            )
        values.append((str(row["ts_code"]), value))
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    positions: dict[Decimal, list[int]] = {}
    for position, (_symbol, value) in enumerate(ordered, start=1):
        positions.setdefault(value, []).append(position)
    total = Decimal(len(ordered))
    buckets: dict[str, str] = {}
    for symbol, value in ordered:
        rank = sum(Decimal(position) for position in positions[value]) / Decimal(
            len(positions[value])
        )
        fraction = rank / total
        if fraction <= Decimal(1) / Decimal(3):
            bucket = "small"
        elif fraction >= Decimal(2) / Decimal(3):
            bucket = "large"
        else:
            bucket = "mid"
        buckets[symbol] = bucket
    return buckets


def _derive_daily_suffix(
    *,
    daily_basic: Sequence[Mapping[str, Any]],
    parent_period: Mapping[str, Mapping[str, Any]],
    parent_forecast: Mapping[str, Mapping[str, Any]],
    period_delta: Sequence[Mapping[str, Any]],
    forecast_delta: Sequence[Mapping[str, Any]],
    parent_daily_columns: Sequence[str],
) -> tuple[pd.DataFrame, set[tuple[str, str]], dict[str, Any]]:
    periods: dict[str, list[Mapping[str, Any]]] = {}
    forecasts: dict[str, list[Mapping[str, Any]]] = {}
    for symbol, parent_row in parent_period.items():
        periods.setdefault(symbol, []).append(parent_row)
    for period_row in period_delta:
        periods.setdefault(str(period_row["ts_code"]), []).append(period_row)
    for symbol, parent_row in parent_forecast.items():
        forecasts.setdefault(symbol, []).append(parent_row)
    for forecast_row in forecast_delta:
        forecasts.setdefault(str(forecast_row["ts_code"]), []).append(forecast_row)

    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for daily_row in daily_basic:
        by_date.setdefault(str(daily_row["trade_date"]), []).append(daily_row)
    output: list[dict[str, Any]] = []
    no_period: set[tuple[str, str]] = set()
    size_evidence: dict[str, Any] = {}
    for trade_date, session in sorted(by_date.items()):
        selected: list[dict[str, Any]] = []
        for raw in sorted(session, key=lambda row: str(row["ts_code"])):
            symbol = str(raw["ts_code"])
            period_candidates = [
                row
                for row in periods.get(symbol, [])
                if _strict_date(
                    row.get("availability_date"), label="period availability"
                )
                <= trade_date
            ]
            period = _winner(period_candidates, end_field="end_date", lane="period")
            if period is None:
                no_period.add((symbol, trade_date))
                continue
            forecast_candidates = [
                row
                for row in forecasts.get(symbol, [])
                if _strict_date(
                    row.get("availability_date"), label="forecast availability"
                )
                <= trade_date
            ]
            forecast = _winner(
                forecast_candidates,
                end_field="forecast_end_date",
                lane="forecast",
            )
            total_mv = raw["__total_mv_decimal"]
            if not isinstance(total_mv, Decimal) or not total_mv.is_finite() or total_mv <= 0:
                _fail(
                    "INVALID_TOTAL_MV",
                    "daily_basic contains invalid total_mv",
                    details={"symbol": symbol, "trade_date": trade_date},
                )
            derived_row: dict[str, Any] = {
                "ts_code": symbol,
                "trade_date": _timestamp(trade_date),
                **{field: period.get(field) for field in period if not str(field).startswith("__")},
                "sector": str(raw.get("sector") or period.get("sector") or "unknown"),
                "total_mv_rmb": float(total_mv),
                "__total_mv_decimal": total_mv,
            }
            free_cashflow = _number(period.get("free_cashflow"))
            derived_row["fcf_to_price"] = (
                float(Decimal(str(free_cashflow)) / total_mv)
                if math.isfinite(free_cashflow)
                else float("nan")
            )
            if forecast is not None:
                for field in FORECAST_VALUE_FIELDS:
                    derived_row[field] = forecast.get(field)
            selected.append(derived_row)
        buckets = _decimal_size_buckets(selected, trade_date=trade_date) if selected else {}
        size_evidence[trade_date] = {
            "investable_rows": len(selected),
            "rule": "decimal_average_rank_over_n_ties_equal",
            "thresholds": {"small_lte": "1/3", "large_gte": "2/3"},
            "bucket_counts": {
                name: sum(1 for value in buckets.values() if value == name)
                for name in ("small", "mid", "large")
            },
        }
        for row in selected:
            row["size_bucket"] = buckets[str(row["ts_code"])]
            row.pop("__total_mv_decimal", None)
            output.append(row)

    if output:
        frame = pd.DataFrame(output)
        missing_material = [
            field
            for field in (
                "ts_code",
                "trade_date",
                "end_date",
                "availability_date",
                "size_bucket",
                "total_mv_rmb",
            )
            if field not in frame.columns
        ]
        if missing_material:
            _fail("DERIVED_DAILY_SCHEMA_MISMATCH", "daily suffix is missing material columns")
        # Preserve the public predecessor column order.  Columns absent from a
        # suffix row are added as nulls; no new public columns are introduced.
        frame = frame.reindex(columns=list(parent_daily_columns))
        frame = frame.sort_values(
            ["ts_code", "trade_date"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=list(parent_daily_columns))
    return frame, no_period, size_evidence


def _reference_from_parent(parent_closure: Mapping[str, Any]) -> dict[str, Any]:
    required_sha_fields = ("pointer_sha256", "manifest_sha256")
    generation_id = str(parent_closure.get("generation_id") or "").strip()
    cutoff = _strict_date(parent_closure.get("cutoff"), label="predecessor cutoff")
    if not generation_id or any(
        not _valid_sha256(parent_closure.get(field))
        for field in required_sha_fields
    ):
        _fail("INVALID_PREDECESSOR_CLOSURE", "predecessor pointer/manifest identity is incomplete")
    table_sha = dict(parent_closure.get("table_sha256", {}) or {})
    if set(table_sha) != set(FUNDAMENTAL_TABLES) or any(
        not _valid_sha256(value) for value in table_sha.values()
    ):
        _fail("INVALID_PREDECESSOR_CLOSURE", "predecessor table identities are incomplete")
    primary = dict(parent_closure.get("primary_provenance", {}) or {})
    provenance_schema = str(primary.get("schema_version") or "").strip()
    if provenance_schema not in {
        "cn-fundamental-primary-provenance.v2",
        SUCCESSOR_PROVENANCE_SCHEMA,
    }:
        _fail(
            "UNSUPPORTED_PREDECESSOR_PROVENANCE",
            "predecessor provenance must be v2 or v3",
        )
    reference: dict[str, Any] = {
        "generation_id": generation_id,
        "cutoff": cutoff,
        "pointer_sha256": str(parent_closure["pointer_sha256"]).lower(),
        "manifest_sha256": str(parent_closure["manifest_sha256"]).lower(),
        "table_sha256": {name: str(table_sha[name]).lower() for name in FUNDAMENTAL_TABLES},
        "provenance_schema_version": provenance_schema,
        "immutable_refs": dict(parent_closure.get("immutable_refs", {}) or {}),
    }
    if provenance_schema == SUCCESSOR_PROVENANCE_SCHEMA:
        parent_chain = dict(primary.get("successor_chain", {}) or {})
        original_seam = dict(parent_chain.get("original_seam", {}) or {})
        reference["original_seam"] = _strict_date(
            original_seam.get("cutoff"), label="predecessor original seam"
        )
    reference["reference_sha256"] = canonical_json_sha256(reference)
    return reference


def build_successor_chain(
    parent_closure: Mapping[str, Any],
    *,
    parent_cutoff: str,
    target_cutoff: str,
    generation_id: str,
) -> dict[str, Any]:
    immediate = _reference_from_parent(parent_closure)
    if immediate["cutoff"] != parent_cutoff:
        _fail("PREDECESSOR_CUTOFF_MISMATCH", "predecessor closure cutoff changed")
    parent_provenance = dict(parent_closure.get("primary_provenance", {}) or {})
    schema = str(parent_provenance.get("schema_version") or "")
    if schema == "cn-fundamental-primary-provenance.v2":
        root = dict(immediate)
        original_seam = {
            "cutoff": parent_cutoff,
            "root_reference_sha256": immediate["reference_sha256"],
        }
        ancestors = [immediate["generation_id"]]
    elif schema == SUCCESSOR_PROVENANCE_SCHEMA:
        parent_chain = dict(parent_provenance.get("successor_chain", {}) or {})
        _validate_successor_chain(parent_chain)
        root = dict(parent_chain["root_reference"])
        original_seam = dict(parent_chain["original_seam"])
        ancestors = list(parent_chain["ancestor_generation_ids"])
        if not ancestors or ancestors[-1] != immediate["generation_id"]:
            _fail(
                "PREDECESSOR_CHAIN_TIP_MISMATCH",
                "v3 predecessor chain tip differs from its generation",
            )
    else:
        _fail(
            "UNSUPPORTED_PREDECESSOR_PROVENANCE",
            "parent must be an immediate v2 or v3 predecessor",
        )
    if len(ancestors) != len(set(ancestors)):
        _fail("SUCCESSOR_CHAIN_CYCLE", "successor ancestry contains a cycle")
    if len(ancestors) + 1 > MAX_SUCCESSOR_CHAIN_DEPTH:
        _fail("SUCCESSOR_CHAIN_RESOURCE_LIMIT", "successor ancestry is too deep")
    if generation_id in ancestors or generation_id == immediate["generation_id"]:
        _fail("SUCCESSOR_SELF_REFERENCE", "successor generation references itself")
    body: dict[str, Any] = {
        "schema_version": SUCCESSOR_CHAIN_SCHEMA,
        "root_reference": root,
        "original_seam": original_seam,
        "immediate_predecessor": immediate,
        "append_boundary": {
            "parent_cutoff": parent_cutoff,
            "target_cutoff": target_cutoff,
            "rule": "strictly_after_immediate_cutoff",
        },
        "ancestor_generation_ids": [*ancestors, generation_id],
    }
    body["chain_fingerprint"] = canonical_json_sha256(body)
    return body


def _validate_reference(reference: Mapping[str, Any], *, label: str) -> None:
    payload = dict(reference)
    _mapping_hash(payload, "reference_sha256")
    provenance_schema = str(payload.get("provenance_schema_version") or "")
    if (
        not str(payload.get("generation_id") or "").strip()
        or not _valid_sha256(payload.get("pointer_sha256"))
        or not _valid_sha256(payload.get("manifest_sha256"))
        or set(dict(payload.get("table_sha256", {}) or {})) != set(FUNDAMENTAL_TABLES)
        or any(not _valid_sha256(value) for value in dict(payload["table_sha256"]).values())
        or provenance_schema
        not in {
            "cn-fundamental-primary-provenance.v2",
            SUCCESSOR_PROVENANCE_SCHEMA,
        }
    ):
        _fail("INVALID_SUCCESSOR_CHAIN", f"{label} reference is invalid")
    immutable_refs = payload.get("immutable_refs", {})
    if not isinstance(immutable_refs, Mapping):
        _fail("INVALID_SUCCESSOR_CHAIN", f"{label} immutable refs are invalid")
    for ref_path, raw_reference in immutable_refs.items():
        reference_value = (
            dict(raw_reference)
            if isinstance(raw_reference, Mapping)
            else {"path": ref_path, "sha256": raw_reference}
        )
        path = Path(str(reference_value.get("path") or ""))
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not _valid_sha256(reference_value.get("sha256"))
        ):
            _fail("INVALID_SUCCESSOR_CHAIN", f"{label} immutable ref is invalid")
    _strict_date(payload.get("cutoff"), label=f"{label} cutoff")


def _validate_successor_chain(chain: Mapping[str, Any]) -> str:
    payload = dict(chain)
    fingerprint = _mapping_hash(payload, "chain_fingerprint")
    if payload.get("schema_version") != SUCCESSOR_CHAIN_SCHEMA:
        _fail("INVALID_SUCCESSOR_CHAIN", "successor chain schema is invalid")
    root = dict(payload.get("root_reference", {}) or {})
    immediate = dict(payload.get("immediate_predecessor", {}) or {})
    _validate_reference(root, label="root")
    _validate_reference(immediate, label="immediate")
    ancestors = list(payload.get("ancestor_generation_ids", []) or [])
    if len(ancestors) > MAX_SUCCESSOR_CHAIN_DEPTH:
        _fail("SUCCESSOR_CHAIN_RESOURCE_LIMIT", "successor ancestry is too deep")
    if not ancestors or len(ancestors) != len(set(ancestors)):
        _fail("SUCCESSOR_CHAIN_CYCLE", "successor chain ancestry is cyclic or empty")
    if (
        len(ancestors) < 2
        or ancestors[0] != root["generation_id"]
        or ancestors[-2] != immediate["generation_id"]
        or ancestors[-1] == immediate["generation_id"]
    ):
        _fail(
            "INVALID_SUCCESSOR_CHAIN",
            "chain ancestry does not bind root, predecessor, and successor",
        )
    append = dict(payload.get("append_boundary", {}) or {})
    if (
        append.get("rule") != "strictly_after_immediate_cutoff"
        or _strict_date(append.get("parent_cutoff"), label="append parent cutoff")
        != dict(payload["immediate_predecessor"])["cutoff"]
        or _strict_date(append.get("target_cutoff"), label="append target cutoff")
        <= append["parent_cutoff"]
    ):
        _fail("INVALID_SUCCESSOR_CHAIN", "append boundary is invalid")
    seam = dict(payload.get("original_seam", {}) or {})
    if (
        _strict_date(seam.get("cutoff"), label="original seam cutoff")
        != dict(payload["root_reference"])["cutoff"]
        or seam.get("root_reference_sha256")
        != dict(payload["root_reference"])["reference_sha256"]
    ):
        _fail("INVALID_SUCCESSOR_CHAIN", "original seam changed")
    if root.get("provenance_schema_version") != (
        "cn-fundamental-primary-provenance.v2"
    ):
        _fail("INVALID_SUCCESSOR_CHAIN", "chain root is not a v2 generation")
    immediate_schema = immediate.get("provenance_schema_version")
    if immediate_schema == "cn-fundamental-primary-provenance.v2":
        if len(ancestors) != 2 or immediate != root:
            _fail("INVALID_SUCCESSOR_CHAIN", "v2 predecessor is not the chain root")
    elif immediate.get("original_seam") != seam.get("cutoff"):
        _fail("INVALID_SUCCESSOR_CHAIN", "v3 predecessor original seam changed")
    return fingerprint


def _parent_columns(source: pd.DataFrame | Path | str) -> list[str]:
    if isinstance(source, pd.DataFrame):
        return list(source.columns)
    return list(pq.ParquetFile(Path(source)).schema_arrow.names)


def _align_suffix_to_parent(
    suffix: pd.DataFrame,
    parent_source: pd.DataFrame | Path | str,
    *,
    table_name: str,
) -> pd.DataFrame:
    columns = _parent_columns(parent_source)
    extra = set(suffix.columns).difference(columns)
    if extra:
        _fail(
            "SUCCESSOR_SCHEMA_EXPANSION",
            f"{table_name} suffix introduces columns",
            details={"columns": sorted(str(value) for value in extra)},
        )
    aligned = suffix.reindex(columns=columns).copy()
    if isinstance(parent_source, pd.DataFrame):
        parent = parent_source
        for column in columns:
            dtype = parent[column].dtype
            try:
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    aligned[column] = pd.to_datetime(aligned[column], errors="raise").astype(dtype)
                elif pd.api.types.is_float_dtype(dtype):
                    aligned[column] = pd.to_numeric(aligned[column], errors="coerce").astype(dtype)
                elif pd.api.types.is_integer_dtype(dtype):
                    aligned[column] = pd.to_numeric(aligned[column], errors="raise").astype(dtype)
                elif isinstance(dtype, pd.StringDtype):
                    aligned[column] = aligned[column].astype(dtype)
            except (TypeError, ValueError) as exc:
                _fail(
                    "SUCCESSOR_SCHEMA_CAST_FAILED",
                    f"{table_name}.{column} cannot match predecessor dtype",
                )
                raise AssertionError from exc
    return aligned


def _validate_parent_sources(
    parent_tables: Mapping[str, pd.DataFrame | Path | str],
    parent_closure: Mapping[str, Any],
) -> dict[str, str]:
    if set(parent_tables) != set(FUNDAMENTAL_TABLES):
        _fail("INCOMPLETE_PARENT_TABLE_SET", "all predecessor tables are required")
    declared_sha = dict(parent_closure.get("table_sha256", {}) or {})
    declared_frames = dict(parent_closure.get("table_frame_fingerprints", {}) or {})
    if set(declared_sha) != set(FUNDAMENTAL_TABLES) or set(declared_frames) != set(
        FUNDAMENTAL_TABLES
    ):
        _fail("INVALID_PREDECESSOR_CLOSURE", "predecessor table closure is incomplete")
    observed_frames: dict[str, str] = {}
    for table_name in FUNDAMENTAL_TABLES:
        source = parent_tables[table_name]
        path = _source_path(source)
        if path is not None:
            if _sha256_file(path) != str(declared_sha[table_name]).lower():
                _fail("PARENT_TABLE_TAMPER", f"predecessor bytes changed: {table_name}")
            # Period/quarantine are small.  The daily frame fingerprint is
            # normally revalidated by the registered pointer loader before this
            # API is called; do not force a 6M-row materialisation here.
            manifest_frames = dict(parent_closure.get("validated_frame_fingerprints", {}) or {})
            observed = str(manifest_frames.get(table_name) or declared_frames[table_name]).lower()
        else:
            observed = frame_fingerprint(source)
        if observed != str(declared_frames[table_name]).lower() or not _valid_sha256(observed):
            _fail("PARENT_PREFIX_TAMPER", f"predecessor logical frame changed: {table_name}")
        observed_frames[table_name] = observed
    return observed_frames


def _materialize_candidates(
    parent_tables: Mapping[str, pd.DataFrame | Path | str],
    *,
    period_suffix: pd.DataFrame,
    daily_suffix: pd.DataFrame,
) -> Mapping[str, pd.DataFrame] | None:
    if not all(isinstance(parent_tables[name], pd.DataFrame) for name in FUNDAMENTAL_TABLES):
        return None
    parent_period = parent_tables["fundamental_period"]
    parent_daily = parent_tables["fundamental_daily"]
    parent_quarantine = parent_tables["fundamental_quarantine"]
    assert isinstance(parent_period, pd.DataFrame)
    assert isinstance(parent_daily, pd.DataFrame)
    assert isinstance(parent_quarantine, pd.DataFrame)
    candidate_period = pd.concat([parent_period, period_suffix], ignore_index=True, sort=False)
    candidate_daily = pd.concat([parent_daily, daily_suffix], ignore_index=True, sort=False)
    try:
        assert_frame_semantics_equal(
            parent_period,
            candidate_period.iloc[: len(parent_period)].reset_index(drop=True),
            label="safe successor period prefix",
        )
        assert_frame_semantics_equal(
            parent_daily,
            candidate_daily.iloc[: len(parent_daily)].reset_index(drop=True),
            label="safe successor daily prefix",
        )
    except (TypeError, ValueError) as exc:
        _fail("PARENT_PREFIX_MUTATED", "candidate materialisation changed predecessor cells")
        raise AssertionError from exc
    return {
        "fundamental_period": candidate_period,
        "fundamental_daily": candidate_daily,
        # No malformed successor row is accepted.  The quarantine frame is an
        # exact logical prefix here and must be an exact byte copy at staging.
        "fundamental_quarantine": parent_quarantine,
    }


def successor_resource_preflight(
    parent_tables: Mapping[str, pd.DataFrame | Path | str],
    *,
    staging_parent: str | Path | None = None,
    suffix_rows: Mapping[str, int] | None = None,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    """Return a bounded-resource declaration without reading parent daily."""

    if type(batch_size) is not int or batch_size < 1:
        _fail("INVALID_RESOURCE_PREFLIGHT", "batch_size must be a positive integer")
    parent_bytes = 0
    path_backed = True
    parent_rows: dict[str, int] = {}
    for table_name, source in parent_tables.items():
        path = _source_path(source)
        if path is None:
            path_backed = False
            frame = source
            assert isinstance(frame, pd.DataFrame)
            parent_bytes += int(frame.memory_usage(index=True, deep=True).sum())
            parent_rows[table_name] = len(frame)
        else:
            parent_bytes += path.stat().st_size
            parent_rows[table_name] = int(pq.ParquetFile(path).metadata.num_rows)
    estimated_suffix_bytes = sum(int(value) for value in dict(suffix_rows or {}).values()) * 512
    required_disk = int(parent_bytes * 1.25 + estimated_suffix_bytes * 2 + 16 * 1024 * 1024)
    target = Path(staging_parent or tempfile.gettempdir()).expanduser()
    while not target.exists() and target != target.parent:
        target = target.parent
    free_disk = shutil.disk_usage(target).free
    blockers = [] if free_disk >= required_disk else ["INSUFFICIENT_STAGING_DISK"]
    body: dict[str, Any] = {
        "schema_version": SUCCESSOR_RESOURCE_SCHEMA,
        "status": "PASS" if not blockers else "BLOCKED",
        "parent_path_backed": path_backed,
        "parent_rows": parent_rows,
        "parent_storage_bytes": parent_bytes,
        "estimated_suffix_bytes": estimated_suffix_bytes,
        "required_free_disk_bytes": required_disk,
        "observed_free_disk_bytes": free_disk,
        "stream_batch_rows": batch_size,
        "streaming_parent_copy_supported": path_backed,
        "blockers": blockers,
    }
    body["binding_sha256"] = canonical_json_sha256(body)
    return body


def assemble_safe_successor(
    *,
    parent_tables: Mapping[str, pd.DataFrame | Path | str],
    parent_closure: Mapping[str, Any],
    support_raw_tables: Mapping[str, Any],
    plan_metadata: Mapping[str, Any],
    keyset_closure: Mapping[str, Any],
    parent_cutoff: str,
    target_cutoff: str,
    run_id: str,
    staging_parent: str | Path | None = None,
) -> SuccessorBundle:
    """Derive and attest one append-only Fundamental successor.

    Provider acquisition and canonical publication are outside this function.
    Every support-prefix row is used only to replay/validate the immediate
    boundary; only events in ``(parent_cutoff, target_cutoff]`` are appended.
    """

    parent = _strict_date(parent_cutoff, label="parent_cutoff")
    target = _strict_date(target_cutoff, label="target_cutoff")
    if target <= parent:
        _fail("INVALID_SUCCESSOR_WINDOW", "target cutoff must follow parent cutoff")
    generation_id = str(run_id or "").strip()
    safe_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    if not generation_id or any(
        character not in safe_characters for character in generation_id
    ):
        _fail("INVALID_GENERATION_ID", "run_id is not a safe generation id")
    predecessor = _reference_from_parent(parent_closure)
    if predecessor["cutoff"] != parent:
        _fail("PREDECESSOR_CUTOFF_MISMATCH", "parent closure does not bind the requested cutoff")
    parent_fingerprints = _validate_parent_sources(parent_tables, parent_closure)
    plan, raw_fingerprints = _validate_plan(
        support_raw_tables,
        plan_metadata,
        parent_cutoff=parent,
        target_cutoff=target,
    )
    permanent_refs = dict(plan["permanent_support_refs"])
    if (
        dict(permanent_refs["predecessor_pointer"])["sha256"]
        != predecessor["pointer_sha256"]
        or dict(permanent_refs["predecessor_manifest"])["sha256"]
        != predecessor["manifest_sha256"]
    ):
        _fail(
            "PREDECESSOR_PERMANENT_REF_MISMATCH",
            "sealed predecessor pointer/manifest do not match the closure",
        )

    streaming_support = _streaming_support_store(support_raw_tables)
    accumulator: _AccumulatorGuard | None = None
    if streaming_support:
        budget = plan.get("derivation_resource_budget")
        if not isinstance(budget, Mapping):
            _fail(
                "DERIVATION_RESOURCE_BUDGET_MISSING",
                "streaming production support requires a sealed accumulator budget",
            )
        accumulator = _AccumulatorGuard(budget)
        (
            delta_period,
            period_lineage,
            replay_period_anchors,
            financial_stream,
        ) = _derive_streamed_financial(
            support_raw_tables,
            accumulator=accumulator,
            plan=plan,
            parent_cutoff=parent,
            target_cutoff=target,
        )
        (
            delta_forecast,
            replay_forecast_anchors,
            forecast_stream,
        ) = _normalize_streamed_forecast(
            support_raw_tables,
            accumulator=accumulator,
            parent_cutoff=parent,
            target_cutoff=target,
            run_id=generation_id,
        )
        daily_basic, daily_basic_keys, daily_stream = _normalize_streamed_daily_basic(
            support_raw_tables,
            accumulator=accumulator,
            parent_cutoff=parent,
            target_cutoff=target,
        )
    else:
        financial = {
            table: _normalize_financial_table(
                support_raw_tables[table], table=table, target_cutoff=target
            )
            for table in FINANCIAL_TABLES
        }
        replay_period, period_lineage = _derive_event_graph(
            financial,
            plan=plan,
            parent_cutoff=parent,
        )
        normalized_forecast = _normalize_forecast_table(
            support_raw_tables["forecast"],
            target_cutoff=target,
            run_id=generation_id,
        )
        replay_forecast = _forecast_records(normalized_forecast)
        daily_basic, daily_basic_keys = _normalize_daily_basic(
            support_raw_tables["daily_basic"],
            parent_cutoff=parent,
            target_cutoff=target,
        )
        financial_stream = {
            "schema_version": "cn-fundamental-support-symbol-stream.v1",
            "mode": "bounded_reference_frames",
            "full_table_getitem_used": True,
        }
        financial_stream["binding_sha256"] = canonical_json_sha256(financial_stream)
        forecast_stream = {
            "schema_version": "cn-fundamental-support-forecast-stream.v1",
            "mode": "bounded_reference_frame",
            "full_table_getitem_used": True,
        }
        forecast_stream["binding_sha256"] = canonical_json_sha256(forecast_stream)
        daily_stream = {
            "schema_version": "cn-fundamental-support-daily-stream.v1",
            "mode": "bounded_reference_frame",
            "full_table_getitem_used": True,
        }
        daily_stream["binding_sha256"] = canonical_json_sha256(daily_stream)
    keyset = _validate_keyset_input(keyset_closure, daily_basic_keys=daily_basic_keys)
    parent_period, parent_forecast, parent_daily_columns = _latest_parent_seeds(
        parent_tables["fundamental_daily"],
        parent_cutoff=parent,
    )
    if not streaming_support:
        period_pairs = [
            (dict(row), dict(row_lineage))
            for row, row_lineage in zip(replay_period, period_lineage, strict=True)
            if parent
            < _strict_date(row["availability_date"], label="period availability")
            <= target
        ]
        delta_period = [row for row, _row_lineage in period_pairs]
        period_lineage = [row_lineage for _row, row_lineage in period_pairs]
        delta_forecast = [
            dict(row)
            for row in replay_forecast
            if parent
            < _strict_date(row["availability_date"], label="forecast availability")
            <= target
        ]
    relevant_symbols = {
        symbol
        for symbol, _trade_date in _keys(
            keyset["expected_scope_keys"],
            label="expected_scope_keys",
        )
    }.union(str(row["ts_code"]) for row in delta_period).union(
        str(row["ts_code"]) for row in delta_forecast
    )
    prefix_mode = str(plan["support_prefix_mode"])
    if prefix_mode == SUCCESSOR_APPEND_FIRST_MODE:
        declared_dependency_keys = {
            (str(value["table"]), str(value["ts_code"]), str(value["end_date"]))
            for value in list(plan.get("append_first_financial_dependencies", []) or [])
        }
        declared_absence_keys = {
            (str(value["table"]), str(value["symbol"]), str(value["end_date"]))
            for value in list(plan.get("absence_proofs", []) or [])
            if value.get("status") == "PROVEN_ABSENT"
        }
        consumed_dependency_keys = {
            (
                str(requirement["table"]),
                str(value["symbol"]),
                str(requirement["end_date"]),
            )
            for value in period_lineage
            for requirement in value["dependency_requirements"]
            if requirement.get("required") is True
            and str(requirement.get("row_binding") or "")
            and requirement.get("bounded_support") is True
        }
        consumed_absence_keys = {
            (
                str(requirement["table"]),
                str(value["symbol"]),
                str(requirement["end_date"]),
            )
            for value in period_lineage
            for requirement in value["dependency_requirements"]
            if requirement.get("required") is True
            and requirement.get("absence_proven") is True
        }
        if consumed_dependency_keys != declared_dependency_keys:
            _fail(
                "APPEND_FIRST_FINANCIAL_DEPENDENCY_NOT_EXACTLY_CONSUMED",
                "bounded financial support must equal the complete fallback read-set",
            )
        if consumed_absence_keys != declared_absence_keys:
            _fail(
                "APPEND_FIRST_FINANCIAL_ABSENCE_NOT_EXACTLY_CONSUMED",
                "bounded financial absence proofs must equal the complete fallback read-set",
            )
        period_anchors = (
            replay_period_anchors
            if streaming_support
            else {
                str(row["ts_code"]): dict(row)
                for row in replay_period
                if _strict_date(
                    row["availability_date"],
                    label="period availability",
                )
                <= parent
            }
        )
        forecast_anchors = (
            replay_forecast_anchors
            if streaming_support
            else {
                str(row["ts_code"]): dict(row)
                for row in replay_forecast
                if _strict_date(
                    row["availability_date"],
                    label="forecast availability",
                )
                <= parent
            }
        )
        if period_anchors or forecast_anchors:
            _fail(
                "APPEND_FIRST_PREFIX_ROW_PRESENT",
                "append-first source contains a row at or before the predecessor cutoff",
            )
        boundary = {
            "schema_version": "cn-fundamental-successor-boundary.append-first.v1",
            "status": "PASS",
            "mode": SUCCESSOR_APPEND_FIRST_MODE,
            "parent_cutoff": parent,
            "predecessor_reference_sha256": predecessor["reference_sha256"],
            "historical_taint_registry_sha256": plan[
                "historical_taint_registry_sha256"
            ],
            "support_prefix_row_count": 0,
            "bounded_financial_support_key_count": len(
                list(plan.get("append_first_financial_dependencies", []) or [])
            ),
            "bounded_financial_absence_proof_count": len(
                [
                    value
                    for value in list(plan.get("absence_proofs", []) or [])
                    if value.get("status") == "PROVEN_ABSENT"
                ]
            ),
            "current_window_material_conflict_count": 0,
        }
        boundary["binding_sha256"] = canonical_json_sha256(boundary)
    elif streaming_support:
        boundary = _prove_boundary_winners(
            parent_period=parent_period,
            parent_forecast=parent_forecast,
            replay_period=replay_period_anchors,
            replay_forecast=replay_forecast_anchors,
            relevant_symbols=relevant_symbols,
            plan=plan,
            parent_cutoff=parent,
        )
    else:
        boundary = _prove_boundary(
            parent_period=parent_period,
            parent_forecast=parent_forecast,
            replay_period=replay_period,
            replay_forecast=replay_forecast,
            relevant_symbols=relevant_symbols,
            plan=plan,
            parent_cutoff=parent,
        )
    period_suffix = pd.DataFrame(delta_period)
    if period_suffix.empty:
        period_suffix = pd.DataFrame(columns=_parent_columns(parent_tables["fundamental_period"]))
    else:
        period_suffix = period_suffix.sort_values(
            ["ts_code", "end_date", "availability_date"], kind="mergesort"
        ).reset_index(drop=True)
        if period_suffix.duplicated(subset=list(PERIOD_KEY_FIELDS)).any():
            _fail("DUPLICATE_PERIOD_SUFFIX_KEY", "period suffix contains duplicate keys")
        period_suffix = _align_suffix_to_parent(
            period_suffix,
            parent_tables["fundamental_period"],
            table_name="fundamental_period",
        )
    daily_suffix, no_period, size_evidence = _derive_daily_suffix(
        daily_basic=daily_basic,
        parent_period=parent_period,
        parent_forecast=parent_forecast,
        period_delta=delta_period,
        forecast_delta=delta_forecast,
        parent_daily_columns=parent_daily_columns,
    )
    daily_suffix = _align_suffix_to_parent(
        daily_suffix,
        parent_tables["fundamental_daily"],
        table_name="fundamental_daily",
    )
    derived_keys = {
        _key(row.ts_code, row.trade_date)
        for row in daily_suffix.loc[:, ["ts_code", "trade_date"]].itertuples(index=False)
    }
    if derived_keys.intersection(no_period) or derived_keys.union(no_period) != daily_basic_keys:
        _fail("DAILY_DERIVATION_KEYSET_NOT_CLOSED", "derived and NO_PERIOD_STATE keys do not close")
    keyset = dict(keyset)
    keyset["derived_daily_keys"] = _serialized_keys(derived_keys)
    keyset["no_period_state_keys"] = _serialized_keys(no_period)
    keyset["classification_counts"] = {
        **dict(keyset["classification_counts"]),
        "derived_daily": len(derived_keys),
        "NO_PERIOD_STATE": len(no_period),
    }
    keyset["closure_binding_sha256"] = canonical_json_sha256(keyset)

    successor_chain = build_successor_chain(
        parent_closure,
        parent_cutoff=parent,
        target_cutoff=target,
        generation_id=generation_id,
    )
    lineage: dict[str, Any] = {
        "period_rows": period_lineage,
        "period_lineage_sha256": canonical_json_sha256({"rows": period_lineage}),
        "daily_selection_rule": "max_availability_then_max_end_date",
        "forecast_selection_rule": "max_availability_then_max_forecast_end_date",
        "same_symbol_availability_atomic": True,
        "future_map_lookahead": False,
        "size_bucket_sessions": size_evidence,
        "boundary": boundary,
        "support_streaming": {
            "financial": financial_stream,
            "forecast": forecast_stream,
            "daily_basic": daily_stream,
            "accumulator": (
                accumulator.receipt()
                if accumulator is not None
                else {
                    "schema_version": (
                        "cn-fundamental-successor-derivation-accumulator.v1"
                    ),
                    "mode": "bounded_reference_frames",
                }
            ),
        },
    }
    lineage["binding_sha256"] = canonical_json_sha256(lineage)
    candidate_tables = _materialize_candidates(
        parent_tables,
        period_suffix=period_suffix,
        daily_suffix=daily_suffix,
    )
    resource = successor_resource_preflight(
        parent_tables,
        staging_parent=staging_parent,
        suffix_rows={
            "fundamental_period": len(period_suffix),
            "fundamental_daily": len(daily_suffix),
        },
    )
    if resource["status"] != "PASS":
        _fail("RESOURCE_PREFLIGHT_BLOCKED", "successor staging resources are insufficient")
    derivation: dict[str, Any] = {
        "contract_version": SUCCESSOR_DERIVATION_CONTRACT,
        "run_id": generation_id,
        "parent_cutoff": parent,
        "target_cutoff": target,
        "support_prefix_mode": prefix_mode,
        "historical_taint_registry_sha256": str(
            plan.get("historical_taint_registry_sha256") or ""
        ),
        "bounded_financial_support_keyset_sha256": canonical_json_sha256(
            list(plan.get("append_first_financial_dependencies", []) or [])
        ),
        "ordinary_merge_used": False,
        "parent_table_frame_fingerprints": parent_fingerprints,
        "support_plan_sha256": plan["plan_sha256"],
        "support_provider_contract": str(
            plan.get("support_provider_contract") or SUCCESSOR_PLAN_SCHEMA
        ),
        "raw_table_fingerprints": raw_fingerprints,
        "period_suffix_rows": len(period_suffix),
        "daily_suffix_rows": len(daily_suffix),
        "period_suffix_fingerprint": frame_fingerprint(period_suffix),
        "daily_suffix_fingerprint": frame_fingerprint(daily_suffix),
        "quarantine_rule": "exact_predecessor_bytes_no_successor_rows",
        "keyset_closure_sha256": keyset["closure_binding_sha256"],
        "lineage_sha256": lineage["binding_sha256"],
        "successor_chain_sha256": successor_chain["chain_fingerprint"],
        "resource_preflight_sha256": resource["binding_sha256"],
    }
    derivation["binding_sha256"] = canonical_json_sha256(derivation)
    readiness: dict[str, Any] = {
        "schema_version": SUCCESSOR_READINESS_SCHEMA,
        "gate2_contract": SUCCESSOR_READINESS_SCHEMA,
        "status": "PASS",
        "gate2_passed": True,
        "prefix_gate_passed": True,
        "suffix_gate_passed": True,
        "structural_gate_passed": True,
        "blockers": [],
        "parent_cutoff": parent,
        "target_cutoff": target,
        "provider_failures": 0,
        "provider_malformed": 0,
        "provider_has_more": 0,
        "true_missing": 0,
        "NO_PERIOD_STATE": len(no_period),
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
        "boundary_binding_sha256": boundary["binding_sha256"],
        "derivation_binding_sha256": derivation["binding_sha256"],
    }
    readiness["binding_sha256"] = canonical_json_sha256(readiness)
    return SuccessorBundle(
        parent_tables={name: parent_tables[name] for name in FUNDAMENTAL_TABLES},
        predecessor_binding=predecessor,
        parent_cutoff=parent,
        target_cutoff=target,
        run_id=generation_id,
        period_suffix=period_suffix,
        daily_suffix=daily_suffix,
        candidate_tables=candidate_tables,
        plan_metadata=plan,
        keyset_closure=keyset,
        successor_chain=successor_chain,
        lineage=lineage,
        derivation_evidence=derivation,
        readiness=readiness,
        raw_table_fingerprints=raw_fingerprints,
        resource_preflight=resource,
    )


def seal_successor_provider_manifest(
    bundle: SuccessorBundle,
    *,
    provider: str,
    request_receipts_sha256: str,
    evidence_files: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal provider counters/bindings already audited by acquisition."""

    if not _valid_sha256(request_receipts_sha256):
        _fail("INVALID_PROVIDER_RECEIPTS", "provider receipts SHA is invalid")
    files = {str(name): str(digest).lower() for name, digest in dict(evidence_files or {}).items()}
    if any(not _valid_sha256(digest) for digest in files.values()):
        _fail("INVALID_PROVIDER_EVIDENCE", "provider evidence file SHA is invalid")
    body: dict[str, Any] = {
        "schema_version": SUCCESSOR_PROVIDER_MANIFEST_SCHEMA,
        "status": "complete",
        "provider": str(provider or "").strip(),
        "parent_cutoff": bundle.parent_cutoff,
        "target_cutoff": bundle.target_cutoff,
        "support_plan_sha256": bundle.plan_metadata["plan_sha256"],
        "request_receipts_sha256": str(request_receipts_sha256).lower(),
        "requests_failed": 0,
        "requests_malformed": 0,
        "responses_has_more": 0,
        "schema_failures": 0,
        "duplicate_conflicts": 0,
        "raw_table_fingerprints": dict(bundle.raw_table_fingerprints),
        "evidence_files": files,
    }
    if not body["provider"]:
        _fail("INVALID_PROVIDER_MANIFEST", "provider name is required")
    if extra:
        for key, value in extra.items():
            if key in body:
                _fail("PROVIDER_FIELD_OVERRIDE", f"provider extra cannot replace {key}")
            body[key] = value
    body["provider_binding_sha256"] = canonical_json_sha256(body)
    return body


def _validate_provider_manifest(
    manifest: Mapping[str, Any],
    *,
    bundle: SuccessorBundle | None = None,
) -> str:
    payload = dict(manifest)
    binding = _mapping_hash(payload, "provider_binding_sha256")
    if (
        payload.get("schema_version") != SUCCESSOR_PROVIDER_MANIFEST_SCHEMA
        or payload.get("status") != "complete"
        or not str(payload.get("provider") or "").strip()
        or not _valid_sha256(payload.get("request_receipts_sha256"))
    ):
        _fail("INVALID_PROVIDER_MANIFEST", "safe-successor provider manifest is invalid")
    for counter in (
        "requests_failed",
        "requests_malformed",
        "responses_has_more",
        "schema_failures",
        "duplicate_conflicts",
    ):
        if type(payload.get(counter)) is not int or payload[counter] != 0:
            _fail("PROVIDER_AUDIT_NOT_CLEAN", f"provider manifest {counter} must be zero")
    fingerprints = dict(payload.get("raw_table_fingerprints", {}) or {})
    if set(fingerprints) != set(RAW_TABLES) or any(
        not _valid_sha256(value) for value in fingerprints.values()
    ):
        _fail("INVALID_PROVIDER_MANIFEST", "provider raw fingerprints are incomplete")
    for name, digest in dict(payload.get("evidence_files", {}) or {}).items():
        if not str(name).strip() or not _valid_sha256(digest):
            _fail("INVALID_PROVIDER_EVIDENCE", "provider evidence binding is invalid")
    if bundle is not None and (
        payload.get("parent_cutoff") != bundle.parent_cutoff
        or payload.get("target_cutoff") != bundle.target_cutoff
        or payload.get("support_plan_sha256") != bundle.plan_metadata["plan_sha256"]
        or fingerprints != dict(bundle.raw_table_fingerprints)
    ):
        _fail("PROVIDER_DERIVATION_BINDING_MISMATCH", "provider manifest does not bind the bundle")
    return binding


def _normalize_immutable_refs(
    value: Any,
    *,
    label: str,
    require_nonempty: bool,
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        raw_items: list[Any] = [
            (
                {"path": path, "sha256": raw_reference}
                if not isinstance(raw_reference, Mapping)
                else {"path": path, **dict(raw_reference)}
            )
            for path, raw_reference in value.items()
        ]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = list(value)
    else:
        _fail(
            "INVALID_TARGET_IMMUTABLE_REFS",
            f"immutable refs are invalid: {label}",
        )
    if require_nonempty and not raw_items:
        _fail(
            "INVALID_TARGET_IMMUTABLE_REFS",
            f"immutable refs are empty: {label}",
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_reference in raw_items:
        if not isinstance(raw_reference, Mapping):
            _fail(
                "INVALID_TARGET_IMMUTABLE_REFS",
                f"immutable ref is invalid: {label}",
            )
        reference = dict(raw_reference)
        path_text = str(reference.get("path") or "").strip()
        path = Path(path_text)
        digest = str(reference.get("sha256") or "").strip().lower()
        if (
            not path_text
            or not path.is_absolute()
            or ".." in path.parts
            or path_text in seen
            or not _valid_sha256(digest)
        ):
            _fail(
                "INVALID_TARGET_IMMUTABLE_REFS",
                f"immutable ref is invalid: {label}",
            )
        seen.add(path_text)
        normalized_ref: dict[str, Any] = {
            "path": path_text,
            "sha256": digest,
        }
        size_value = reference.get("size", reference.get("byte_length"))
        if size_value is not None:
            try:
                size = int(size_value)
            except (TypeError, ValueError) as exc:
                _fail(
                    "INVALID_TARGET_IMMUTABLE_REFS",
                    f"immutable ref size is invalid: {label}",
                )
                raise AssertionError from exc
            if size < 0:
                _fail(
                    "INVALID_TARGET_IMMUTABLE_REFS",
                    f"immutable ref size is invalid: {label}",
                )
            normalized_ref["size"] = size
        normalized.append(normalized_ref)
    return sorted(normalized, key=lambda item: item["path"])


def _validate_target_bindings(
    target_bindings: Mapping[str, Any],
    *,
    target_cutoff: str,
    read_files: bool = False,
    require_sealed: bool = False,
) -> dict[str, Any]:
    payload = {
        str(name): dict(value)
        for name, value in dict(target_bindings).items()
        if isinstance(value, Mapping)
    }
    required = {
        "market_pointer",
        "pit_pointer",
        "pit_membership",
        "expected_scope",
    }
    if set(payload) != required:
        _fail(
            "TARGET_BINDING_SET_MISMATCH",
            "target bindings must include market/PIT pointers, membership, and scope",
        )
    for name in sorted(required):
        value = payload[name]
        path_text = str(value.get("path") or "").strip()
        digest = str(value.get("sha256") or "").strip().lower()
        as_of = _strict_date(value.get("as_of"), label=f"{name} as_of")
        if not path_text or not _valid_sha256(digest) or as_of != target_cutoff:
            _fail("INVALID_TARGET_BINDING", f"target binding is invalid: {name}")
        sealed = value.get("sealed_ref")
        if require_sealed and not isinstance(sealed, Mapping):
            _fail("TARGET_SEALED_REF_MISSING", f"sealed target ref is missing: {name}")
        if isinstance(sealed, Mapping):
            sealed_path = Path(str(sealed.get("path") or ""))
            if (
                sealed_path.is_absolute()
                or not sealed_path.parts
                or ".." in sealed_path.parts
                or not _valid_sha256(sealed.get("sha256"))
                or str(sealed.get("sha256")).lower() != digest
            ):
                _fail("INVALID_TARGET_SEALED_REF", f"sealed target ref is invalid: {name}")
        immutable_refs = value.get("immutable_refs", {})
        if name in {"market_pointer", "pit_pointer"}:
            value["immutable_refs"] = _normalize_immutable_refs(
                immutable_refs,
                label=name,
                require_nonempty=True,
            )
        if read_files:
            path = Path(path_text).expanduser().resolve(strict=True)
            if _sha256_file(path) != digest:
                _fail("TARGET_EVIDENCE_TAMPER", f"target evidence changed: {name}")
    body: dict[str, Any] = {name: payload[name] for name in sorted(required)}
    body["binding_sha256"] = canonical_json_sha256(body)
    return body


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_streamed_candidate(
    parent_path: Path,
    suffix: pd.DataFrame,
    destination: Path,
    *,
    table_name: str,
    batch_size: int = 100_000,
) -> int:
    parquet = pq.ParquetFile(parent_path)
    schema = parquet.schema_arrow
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(destination, schema, compression="snappy")
    parent_rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            writer.write_table(
                pa.Table.from_batches([batch], schema=schema),
                row_group_size=len(batch),
            )
            parent_rows += len(batch)
        if not suffix.empty:
            try:
                suffix_table = pa.Table.from_pandas(
                    suffix,
                    schema=schema,
                    preserve_index=False,
                    safe=True,
                )
            except (pa.ArrowException, TypeError, ValueError) as exc:
                _fail(
                    "SUCCESSOR_ARROW_CAST_FAILED",
                    f"{table_name} suffix cannot match parent schema",
                )
                raise AssertionError from exc
            for batch in suffix_table.to_batches(max_chunksize=batch_size):
                writer.write_table(
                    pa.Table.from_batches([batch], schema=schema),
                    row_group_size=len(batch),
                )
    finally:
        writer.close()
    _fsync_file(destination)
    return parent_rows


def _assert_streamed_prefix_equal(
    parent_path: Path,
    candidate_path: Path,
    *,
    table_name: str,
    batch_size: int = 100_000,
) -> None:
    parent_batches = pq.ParquetFile(parent_path).iter_batches(batch_size=batch_size)
    candidate_parquet = pq.ParquetFile(candidate_path)
    seen = 0
    for row_group, parent_batch in enumerate(parent_batches):
        if row_group >= candidate_parquet.num_row_groups:
            _fail("STAGED_PREFIX_TRUNCATED", f"{table_name} candidate truncated the parent")
        candidate_table = candidate_parquet.read_row_group(row_group)
        candidate_batches = candidate_table.to_batches()
        if len(candidate_batches) != 1:
            _fail("STAGED_PREFIX_BATCH_DRIFT", f"{table_name} prefix row group fragmented")
        candidate_batch = candidate_batches[0]
        if len(parent_batch) != len(candidate_batch):
            _fail("STAGED_PREFIX_BATCH_DRIFT", f"{table_name} prefix batch boundaries changed")
        try:
            assert_frame_semantics_equal(
                parent_batch.to_pandas(),
                candidate_batch.to_pandas(),
                label=f"{table_name} streamed parent prefix",
            )
        except (TypeError, ValueError) as exc:
            _fail("STAGED_PREFIX_MUTATED", f"{table_name} staged prefix changed")
            raise AssertionError from exc
        seen += len(parent_batch)
    if seen != int(pq.ParquetFile(parent_path).metadata.num_rows):
        _fail("STAGED_PREFIX_COUNT_MISMATCH", f"{table_name} prefix count changed")


def _streaming_table_evidence(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    columns = list(parquet.schema_arrow.names)
    logical_schema: list[dict[str, Any]] | None = None
    seen = 0
    for row_group in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(row_group).to_pandas()
        chunk_schema = frame_logical_schema(chunk)
        if logical_schema is None:
            logical_schema = [dict(item) for item in chunk_schema]
        else:
            if len(logical_schema) != len(chunk_schema):
                _fail(
                    "STAGED_TABLE_SCHEMA_DRIFT",
                    f"schema changed across row groups: {path.name}",
                )
            for aggregate, observed in zip(logical_schema, chunk_schema):
                if (
                    aggregate["position"] != observed["position"]
                    or aggregate["name"] != observed["name"]
                ):
                    _fail(
                        "STAGED_TABLE_SCHEMA_DRIFT",
                        f"columns changed across row groups: {path.name}",
                    )
                aggregate["logical_scalar_types"] = sorted(
                    set(aggregate["logical_scalar_types"]).union(
                        observed["logical_scalar_types"]
                    )
                )
                aggregate["nullable"] = bool(aggregate["nullable"] or observed["nullable"])
        seen += len(chunk)
    if logical_schema is None:
        empty = pd.read_parquet(path)
        logical_schema = frame_logical_schema(empty)
    if seen != rows:
        _fail("STAGED_TABLE_ROWCOUNT_DRIFT", f"row count changed: {path.name}")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"rows": rows, "schema": logical_schema},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    fingerprint_rows = 0
    for row_group in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(row_group).to_pandas()
        for row in chunk.itertuples(index=False, name=None):
            digest.update(b"\x00")
            digest.update(
                json.dumps(
                    [list(_scalar_token(value)) for value in row],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        fingerprint_rows += len(chunk)
    if fingerprint_rows != rows:
        _fail("STAGED_TABLE_FINGERPRINT_COUNT_DRIFT", f"fingerprint count changed: {path.name}")
    return {
        "rows": rows,
        "columns": columns,
        "sha256": _sha256_file(path),
        "frame_fingerprint": digest.hexdigest(),
        "logical_schema": logical_schema,
    }


def _copy_provider_evidence(
    files: Mapping[str, bytes | Path | str],
    destination: Path,
) -> dict[str, str]:
    output: dict[str, str] = {}
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    for name, source in sorted(files.items()):
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            _fail("UNSAFE_PROVIDER_EVIDENCE_PATH", "provider evidence path is unsafe")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = path.parent
        while current == destination or destination in current.parents:
            os.chmod(current, 0o700)
            if current == destination:
                break
            current = current.parent
        if isinstance(source, bytes):
            _atomic_write(path, source)
        else:
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            before = os.lstat(source_path)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                _fail(
                    "UNSAFE_PROVIDER_EVIDENCE_SOURCE",
                    "provider evidence source is not a regular file",
                )
            descriptor = os.open(
                source_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            temporary_descriptor = -1
            temporary_name = ""
            digest = hashlib.sha256()
            observed = 0
            try:
                opened = os.fstat(descriptor)
                if _stat_identity(opened) != _stat_identity(before):
                    _fail(
                        "PROVIDER_EVIDENCE_SOURCE_DRIFT",
                        "provider evidence changed while opening",
                    )
                temporary_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{path.name}.",
                    dir=path.parent,
                )
                os.fchmod(temporary_descriptor, 0o600)
                with os.fdopen(descriptor, "rb", closefd=False) as source_handle:
                    with os.fdopen(
                        temporary_descriptor,
                        "wb",
                        closefd=False,
                    ) as destination_handle:
                        while True:
                            chunk = source_handle.read(1024 * 1024)
                            if not chunk:
                                break
                            destination_handle.write(chunk)
                            digest.update(chunk)
                            observed += len(chunk)
                        destination_handle.flush()
                        os.fsync(destination_handle.fileno())
                after = os.lstat(source_path)
                if (
                    _stat_identity(after) != _stat_identity(before)
                    or observed != before.st_size
                ):
                    _fail(
                        "PROVIDER_EVIDENCE_SOURCE_DRIFT",
                        "provider evidence changed while copying",
                    )
                os.replace(temporary_name, path)
                temporary_name = ""
                _fsync_directory(path.parent)
            finally:
                os.close(descriptor)
                if temporary_descriptor >= 0:
                    os.close(temporary_descriptor)
                if temporary_name:
                    try:
                        os.unlink(temporary_name)
                    except OSError:
                        pass
            if _sha256_file(path) != digest.hexdigest():
                _fail(
                    "PROVIDER_EVIDENCE_READBACK_MISMATCH",
                    "provider evidence changed after streaming copy",
                )
        output[str(relative)] = _sha256_file(path)
    _fsync_directory(destination)
    return output


def _successor_provenance_envelope(
    *,
    bundle: SuccessorBundle,
    metadata: Mapping[str, Any],
    provider_manifest: Mapping[str, Any],
    target_bindings: Mapping[str, Any],
    table_manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": SUCCESSOR_PROVENANCE_SCHEMA,
        "status": SUCCESSOR_PROVENANCE_STATUS,
        "source": "live_tushare_safe_successor",
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit_safe_successor_mixed",
        "history_state": "mixed",
        "mixed_generation": True,
        "seam_trade_date": dict(bundle.successor_chain["original_seam"])[
            "cutoff"
        ],
        "prefix_contract": {
            "provenance_schema_version": bundle.predecessor_binding[
                "provenance_schema_version"
            ],
            "reference_sha256": bundle.predecessor_binding["reference_sha256"],
        },
        "suffix_contract": SUCCESSOR_DERIVATION_CONTRACT,
        "support_provider_contract": bundle.derivation_evidence[
            "support_provider_contract"
        ],
        "gate2_contract": SUCCESSOR_READINESS_SCHEMA,
        "gate2_receipt_sha256": bundle.readiness["binding_sha256"],
        "provider_manifest_sha256": canonical_json_sha256(provider_manifest),
        "metadata_sha256": canonical_json_sha256(metadata),
        "predecessor": dict(bundle.predecessor_binding),
        "target_bindings": dict(target_bindings),
        "successor_chain": dict(bundle.successor_chain),
        "support_plan_sha256": bundle.plan_metadata["plan_sha256"],
        "permanent_support_refs": dict(
            bundle.plan_metadata.get("permanent_support_refs", {}) or {}
        ),
        "derivation_binding_sha256": bundle.derivation_evidence["binding_sha256"],
        "readiness_binding_sha256": bundle.readiness["binding_sha256"],
        "keyset_closure_sha256": bundle.keyset_closure["closure_binding_sha256"],
        "resource_preflight_sha256": bundle.resource_preflight["binding_sha256"],
        "raw_table_fingerprints": dict(bundle.raw_table_fingerprints),
        "parent_prefix_frame_fingerprints": dict(
            bundle.derivation_evidence["parent_table_frame_fingerprints"]
        ),
        "output_frame_fingerprints": {
            table_name: str(table_manifest[table_name]["frame_fingerprint"])
            for table_name in FUNDAMENTAL_TABLES
        },
        "output_parquet_sha256": {
            table_name: str(table_manifest[table_name]["sha256"])
            for table_name in FUNDAMENTAL_TABLES
        },
        "quarantine_identity": {
            "mode": "exact_predecessor_bytes",
            "predecessor_sha256": bundle.predecessor_binding["table_sha256"][
                "fundamental_quarantine"
            ],
            "successor_sha256": table_manifest["fundamental_quarantine"]["sha256"],
            "exact": (
                bundle.predecessor_binding["table_sha256"]["fundamental_quarantine"]
                == table_manifest["fundamental_quarantine"]["sha256"]
            ),
        },
        "machine_states": {
            "mixed": True,
            "legacy_direct_reader_provenance": "limited",
            "binding_aware_research_ready": True,
            "homogeneous_history_ready": False,
        },
    }
    body["envelope_sha256"] = canonical_json_sha256(body)
    return body


def build_keyset_closure(
    *,
    observed_bar_keys: Iterable[Any],
    daily_basic_keys: Iterable[Any],
    suspended_keys: Iterable[Any] = (),
    inactive_keys: Iterable[Any] = (),
    delisted_keys: Iterable[Any] = (),
    prelisting_keys: Iterable[Any] = (),
    true_missing_keys: Iterable[Any] = (),
    expected_scope_keys: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Build, but do not validate against frames, a reason-coded key closure."""

    observed = _keys(observed_bar_keys, label="observed_bar_keys")
    daily = _keys(daily_basic_keys, label="daily_basic_keys")
    reasons = {
        "suspended": _keys(suspended_keys, label="suspended_keys"),
        "inactive": _keys(inactive_keys, label="inactive_keys"),
        "delisted": _keys(delisted_keys, label="delisted_keys"),
        "prelisting": _keys(prelisting_keys, label="prelisting_keys"),
    }
    nonbar = set().union(*reasons.values())
    expected = (
        _keys(expected_scope_keys, label="expected_scope_keys")
        if expected_scope_keys is not None
        else observed.union(nonbar)
    )
    return {
        "schema_version": SUCCESSOR_KEYSET_SCHEMA,
        "observed_bar_keys": _serialized_keys(observed),
        "daily_basic_keys": _serialized_keys(daily),
        **{f"{reason}_keys": _serialized_keys(values) for reason, values in reasons.items()},
        "nonbar_keys": _serialized_keys(nonbar),
        "true_missing_keys": _serialized_keys(
            _keys(true_missing_keys, label="true_missing_keys")
        ),
        "expected_scope_keys": _serialized_keys(expected),
    }


def capture_parent_closure(
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    *,
    cutoff: str,
    generation_root: str | Path | None = None,
    pointer_bytes: bytes | None = None,
    manifest_bytes: bytes | None = None,
    immutable_refs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the immediate predecessor identity expected by the assembler.

    ``pointer_bytes`` and ``manifest_bytes`` should be the exact registered
    readback.  If omitted, ``generation_root`` is used to read the pointer and
    referenced manifest directly; serialising an augmented loader object is
    never accepted as a substitute for exact bytes.
    """

    payload = dict(pointer)
    base = Path(generation_root).expanduser().resolve(strict=True) if generation_root else None
    if pointer_bytes is None:
        if base is None:
            _fail("EXACT_PARENT_BYTES_REQUIRED", "pointer bytes or generation_root are required")
        pointer_path = base / "_fundamental_latest.json"
        pointer_bytes = pointer_path.read_bytes()
    try:
        raw_pointer = json.loads(pointer_bytes.decode("utf-8"))
    except Exception as exc:
        _fail("INVALID_PARENT_POINTER", "parent pointer bytes are invalid JSON")
        raise AssertionError from exc
    expected_pointer = {
        key: value
        for key, value in payload.items()
        if key
        not in {"pointer_path", "manifest", "primary_provenance_verified"}
    }
    expected_metadata = dict(expected_pointer.get("metadata", {}) or {})
    expected_metadata.pop("primary_provenance_verified", None)
    expected_pointer["metadata"] = expected_metadata
    if not isinstance(raw_pointer, Mapping) or dict(raw_pointer) != expected_pointer:
        _fail("PARENT_POINTER_READBACK_MISMATCH", "pointer object differs from exact bytes")
    if (
        raw_pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or raw_pointer.get("status") != "OK"
    ):
        _fail("INVALID_PARENT_POINTER", "parent pointer status/schema is invalid")
    if manifest is None:
        manifest = payload.get("manifest") if isinstance(payload.get("manifest"), Mapping) else None
    manifest_path: Path | None = None
    if manifest_bytes is None:
        if base is None:
            _fail("EXACT_PARENT_BYTES_REQUIRED", "manifest bytes or generation_root are required")
        manifest_path = (base / str(raw_pointer.get("manifest_path") or "")).resolve(strict=True)
        if base not in manifest_path.parents:
            _fail("UNSAFE_PARENT_MANIFEST_PATH", "parent manifest escapes generation root")
        manifest_bytes = manifest_path.read_bytes()
    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        _fail("INVALID_PARENT_MANIFEST", "parent manifest bytes are invalid JSON")
        raise AssertionError from exc
    if not isinstance(raw_manifest, Mapping) or (
        manifest is not None and dict(raw_manifest) != dict(manifest)
    ):
        _fail("PARENT_MANIFEST_READBACK_MISMATCH", "manifest object differs from exact bytes")
    if (
        raw_manifest.get("schema_version") != "cn-fundamental-generation.v1"
        or raw_manifest.get("status") != "OK"
        or raw_manifest.get("generation_id") != raw_pointer.get("generation_id")
    ):
        _fail("INVALID_PARENT_MANIFEST", "parent generation manifest is invalid")
    tables = dict(raw_manifest.get("tables", {}) or {})
    pointer_tables = dict(raw_pointer.get("tables", {}) or {})
    if set(tables) != set(FUNDAMENTAL_TABLES) or set(pointer_tables) != set(FUNDAMENTAL_TABLES):
        _fail("INVALID_PARENT_TABLE_SET", "parent table set is incomplete")
    table_sha = {name: str(tables[name].get("sha256") or "").lower() for name in FUNDAMENTAL_TABLES}
    frame_fingerprints = {
        name: str(tables[name].get("frame_fingerprint") or "").lower()
        for name in FUNDAMENTAL_TABLES
    }
    if any(
        not _valid_sha256(value)
        for value in (*table_sha.values(), *frame_fingerprints.values())
    ):
        _fail("INVALID_PARENT_TABLE_IDENTITY", "parent table identity is incomplete")
    table_paths: dict[str, Path] = {}
    if base is not None:
        for name in FUNDAMENTAL_TABLES:
            table_path = (base / str(pointer_tables[name])).resolve(strict=True)
            if base not in table_path.parents or _sha256_file(table_path) != table_sha[name]:
                _fail("PARENT_TABLE_TAMPER", f"parent table bytes changed: {name}")
            table_paths[name] = table_path
    primary = dict(raw_manifest.get("primary_provenance", {}) or {})
    if primary != dict(raw_pointer.get("primary_provenance", {}) or {}):
        _fail("PARENT_PROVENANCE_MISMATCH", "parent pointer/manifest provenance differs")
    if primary.get("schema_version") == SUCCESSOR_PROVENANCE_SCHEMA:
        _validate_successor_chain(dict(primary.get("successor_chain", {}) or {}))
    elif primary.get("schema_version") != "cn-fundamental-primary-provenance.v2":
        _fail("UNSUPPORTED_PREDECESSOR_PROVENANCE", "parent provenance is not v2 or v3")
    captured_immutable_refs = dict(immutable_refs or {})
    if not captured_immutable_refs and base is not None:
        if manifest_path is None:
            manifest_path = (
                base / str(raw_pointer.get("manifest_path") or "")
            ).resolve(strict=True)
        captured_immutable_refs = {
            str(manifest_path): {
                "path": str(manifest_path),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            **{
                str(table_paths[name]): {
                    "path": str(table_paths[name]),
                    "sha256": table_sha[name],
                }
                for name in FUNDAMENTAL_TABLES
            },
        }
    normalized_immutable_refs = {
        reference["path"]: reference
        for reference in _normalize_immutable_refs(
            captured_immutable_refs,
            label="predecessor",
            require_nonempty=base is not None,
        )
    }
    return {
        "generation_id": str(raw_pointer["generation_id"]),
        "cutoff": _strict_date(cutoff, label="parent cutoff"),
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "table_sha256": table_sha,
        "table_frame_fingerprints": frame_fingerprints,
        "validated_frame_fingerprints": frame_fingerprints,
        "primary_provenance": primary,
        "immutable_refs": normalized_immutable_refs,
    }


def stage_successor_generation(
    bundle: SuccessorBundle,
    *,
    staging_root: str | Path,
    generation_id: str,
    provider_manifest: Mapping[str, Any],
    target_bindings: Mapping[str, Any],
    provider_evidence_files: Mapping[str, bytes | Path | str] | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
) -> SuccessorStagingCapture:
    """Write and exactly read back one isolated promotion-ready root.

    The destination must not already exist.  The function copies predecessor
    period/daily Parquet in bounded Arrow batches and copies quarantine bytes
    exactly; it never reads or writes a canonical root.
    """

    resolved_id = str(generation_id or "").strip()
    if resolved_id != bundle.run_id:
        _fail("GENERATION_ID_MISMATCH", "staging generation_id differs from derivation run_id")
    _validate_provider_manifest(provider_manifest, bundle=bundle)
    targets = _validate_target_bindings(
        target_bindings,
        target_cutoff=bundle.target_cutoff,
        read_files=True,
        require_sealed=False,
    )
    targets.pop("binding_sha256", None)
    root = Path(staging_root).expanduser()
    if root.exists():
        _fail("STAGING_ROOT_EXISTS", "isolated staging root must not already exist")
    parent_paths = {
        name: _source_path(bundle.parent_tables[name]) for name in FUNDAMENTAL_TABLES
    }
    if any(path is None for path in parent_paths.values()):
        _fail(
            "PATH_BACKED_PARENT_REQUIRED",
            "production staging requires path-backed predecessor tables",
        )
    if bundle.resource_preflight.get("status") != "PASS":
        _fail("RESOURCE_PREFLIGHT_BLOCKED", "resource preflight did not pass")
    generation_directory = root / "_fundamental_generations" / resolved_id
    generation_directory.mkdir(parents=True, exist_ok=False)
    table_paths = {
        name: generation_directory / f"{name}.parquet" for name in FUNDAMENTAL_TABLES
    }
    try:
        _write_streamed_candidate(
            parent_paths["fundamental_period"],  # type: ignore[arg-type]
            bundle.period_suffix,
            table_paths["fundamental_period"],
            table_name="fundamental_period",
        )
        _write_streamed_candidate(
            parent_paths["fundamental_daily"],  # type: ignore[arg-type]
            bundle.daily_suffix,
            table_paths["fundamental_daily"],
            table_name="fundamental_daily",
        )
        quarantine_parent = parent_paths["fundamental_quarantine"]
        assert quarantine_parent is not None
        _copy_provider_evidence(
            {
                table_paths["fundamental_quarantine"].name: quarantine_parent,
            },
            generation_directory,
        )
        if _sha256_file(table_paths["fundamental_quarantine"]) != _sha256_file(quarantine_parent):
            _fail("QUARANTINE_BYTE_IDENTITY_MISMATCH", "quarantine bytes changed")
        _assert_streamed_prefix_equal(
            parent_paths["fundamental_period"],  # type: ignore[arg-type]
            table_paths["fundamental_period"],
            table_name="fundamental_period",
        )
        _assert_streamed_prefix_equal(
            parent_paths["fundamental_daily"],  # type: ignore[arg-type]
            table_paths["fundamental_daily"],
            table_name="fundamental_daily",
        )
        table_manifest = {
            name: _streaming_table_evidence(path) for name, path in table_paths.items()
        }
        evidence_directory = generation_directory / "provider_evidence"
        evidence_inputs = dict(provider_evidence_files or {})
        for name, binding in targets.items():
            live_path = Path(str(binding["path"])).expanduser().resolve(strict=True)
            relative = f"sealed_targets/{name}{live_path.suffix or '.evidence'}"
            if relative in evidence_inputs:
                _fail(
                    "PROVIDER_EVIDENCE_PATH_COLLISION",
                    f"target evidence path collides: {relative}",
                )
            evidence_inputs[relative] = live_path
            binding["sealed_ref"] = {
                "path": relative,
                "sha256": str(binding["sha256"]).lower(),
            }
        targets = _validate_target_bindings(
            targets,
            target_cutoff=bundle.target_cutoff,
            read_files=True,
            require_sealed=True,
        )
        evidence_files = _copy_provider_evidence(
            evidence_inputs,
            evidence_directory,
        )
        provider_bytes = _canonical_bytes(provider_manifest)
        _atomic_write(evidence_directory / "provider_manifest.json", provider_bytes)
        evidence_files["provider_manifest.json"] = hashlib.sha256(provider_bytes).hexdigest()
        declared_evidence = dict(provider_manifest.get("evidence_files", {}) or {})
        for name, digest in declared_evidence.items():
            if evidence_files.get(name) != digest:
                _fail("PROVIDER_EVIDENCE_READBACK_MISMATCH", f"provider evidence changed: {name}")
        for name, reference in dict(
            bundle.plan_metadata.get("permanent_support_refs", {}) or {}
        ).items():
            relative = str(dict(reference).get("path") or "")
            digest = str(dict(reference).get("sha256") or "").lower()
            if evidence_files.get(relative) != digest:
                _fail(
                    "PERMANENT_SUPPORT_READBACK_MISMATCH",
                    f"permanent support evidence is absent or changed: {name}",
                )

        source_validation: dict[str, Any] | None = None
        source_manifest_sha256 = provider_manifest.get(
            "source_fileset_manifest_sha256"
        )
        if source_manifest_sha256 is not None:
            from .fundamental_successor_source import (
                validate_successor_support_fileset,
            )

            implementation_sha256 = str(
                provider_manifest.get("implementation_sha256") or ""
            ).lower()
            if (
                not _valid_sha256(source_manifest_sha256)
                or not _valid_sha256(implementation_sha256)
            ):
                _fail(
                    "SOURCE_FILESET_BINDING_INVALID",
                    "source fileset binding is incomplete",
                )
            validated_source = validate_successor_support_fileset(
                evidence_directory / "source",
                expected_implementation_sha256=implementation_sha256,
            )
            if (
                validated_source.get("manifest_sha256")
                != str(source_manifest_sha256).lower()
            ):
                _fail(
                    "SOURCE_FILESET_MANIFEST_MISMATCH",
                    "source fileset manifest changed during staging",
                )
            source_validation = {
                "implementation_sha256": implementation_sha256,
                "manifest_sha256": validated_source["manifest_sha256"],
                "resource_sha256": dict(
                    validated_source.get("resource_accounting", {}) or {}
                ).get("resource_sha256"),
                "schema_version": validated_source["schema_version"],
                "status": "PASS",
            }
        elif dict(metadata_extra or {}).get("maintenance_schema_version") == (
            "cn-fundamental-safe-successor-maintenance.v1"
        ):
            _fail(
                "SOURCE_FILESET_BINDING_MISSING",
                "production successor staging requires source replay evidence",
            )

        historical_taint_validation: dict[str, Any] | None = None
        if bundle.derivation_evidence.get("support_prefix_mode") == (
            SUCCESSOR_APPEND_FIRST_MODE
        ):
            from .fundamental_historical_taint import (
                validate_historical_taint_registry,
            )

            declared_registry_sha = str(
                provider_manifest.get("historical_taint_registry_sha256") or ""
            ).lower()
            declared_registry_file_sha = str(
                provider_manifest.get(
                    "historical_taint_registry_file_sha256"
                )
                or ""
            ).lower()
            registry_path = evidence_directory / "historical_taint" / "registry.json"
            if (
                not _valid_sha256(declared_registry_sha)
                or not _valid_sha256(declared_registry_file_sha)
                or evidence_files.get("historical_taint/registry.json")
                != declared_registry_file_sha
                or bundle.derivation_evidence.get(
                    "historical_taint_registry_sha256"
                )
                != declared_registry_sha
                or source_validation is None
            ):
                _fail(
                    "HISTORICAL_TAINT_BINDING_MISMATCH",
                    "append-first staging is missing its sealed taint/source closure",
                )
            registry = validate_historical_taint_registry(
                registry_path,
                evidence_root=evidence_directory,
                predecessor=bundle.predecessor_binding,
                delta_fileset_root=evidence_directory / "source",
            )
            if registry.get("registry_sha256") != declared_registry_sha:
                _fail(
                    "HISTORICAL_TAINT_BINDING_MISMATCH",
                    "historical-taint registry digest differs after replay",
                )
            historical_taint_validation = {
                "schema_version": registry["schema_version"],
                "status": registry["status"],
                "classification": registry["classification"],
                "historical_conflict_count": registry[
                    "historical_conflict_count"
                ],
                "current_window_material_conflict_count": 0,
                "same_period_delta_row_count": 0,
                "registry_sha256": declared_registry_sha,
            }

        metadata: dict[str, Any] = {
            "gate2_passed": True,
            "gate2_contract": SUCCESSOR_READINESS_SCHEMA,
            "gate2_receipt_sha256": bundle.readiness["binding_sha256"],
            "prefix_gate_passed": True,
            "suffix_gate_passed": True,
            "structural_gate_passed": True,
            "run_id": resolved_id,
            "source_priority": "tushare_primary",
            "provider_status": "live_tushare_safe_successor",
            "source_provenance": "live_tushare_explicit_safe_successor_mixed",
            "storage_backend": "parquet_canonical",
            "mixed": True,
            "legacy_direct_reader_provenance": "limited",
            "binding_aware_research_ready": True,
            "homogeneous_history_ready": False,
            "provider_manifest": dict(provider_manifest),
            "derivation": dict(bundle.derivation_evidence),
            "readiness": dict(bundle.readiness),
            "keyset_closure": dict(bundle.keyset_closure),
            "resource_preflight": dict(bundle.resource_preflight),
            "target_bindings": dict(targets),
            "successor_chain": dict(bundle.successor_chain),
            "source_fileset_validation": source_validation,
            "historical_taint_validation": historical_taint_validation,
            "merge": {
                "mode": "safe_successor_append_only",
                "ordinary_merge_used": False,
                "absence_is_deletion": False,
                "fundamental_period": {
                    "retained_existing_rows": (
                        table_manifest["fundamental_period"]["rows"]
                        - len(bundle.period_suffix)
                    ),
                    "accepted_incoming_rows": len(bundle.period_suffix),
                },
                "fundamental_daily": {
                    "retained_existing_rows": (
                        table_manifest["fundamental_daily"]["rows"]
                        - len(bundle.daily_suffix)
                    ),
                    "accepted_incoming_rows": len(bundle.daily_suffix),
                },
                "fundamental_quarantine": {
                    "retained_existing_rows": table_manifest["fundamental_quarantine"]["rows"],
                    "accepted_incoming_rows": 0,
                    "exact_predecessor_bytes": True,
                },
            },
        }
        if metadata_extra:
            for key, value in metadata_extra.items():
                if key in metadata:
                    _fail("METADATA_FIELD_OVERRIDE", f"metadata_extra cannot replace {key}")
                metadata[key] = value
        provenance = _successor_provenance_envelope(
            bundle=bundle,
            metadata=metadata,
            provider_manifest=provider_manifest,
            target_bindings=targets,
            table_manifest=table_manifest,
        )
        manifest: dict[str, Any] = {
            "schema_version": "cn-fundamental-generation.v1",
            "generation_id": resolved_id,
            "status": "OK",
            "tables": table_manifest,
            "metadata": metadata,
            "primary_provenance": provenance,
        }
        relative_generation = Path("_fundamental_generations") / resolved_id
        pointer: dict[str, Any] = {
            "schema_version": "cn-fundamental-pointer.v1",
            "generation_id": resolved_id,
            "status": "OK",
            "manifest_path": str(relative_generation / "manifest.json"),
            "tables": {
                name: str(relative_generation / f"{name}.parquet")
                for name in FUNDAMENTAL_TABLES
            },
            "metadata": metadata,
            "primary_provenance": provenance,
        }
        manifest_bytes = _canonical_bytes(manifest)
        pointer_bytes = _canonical_bytes(pointer)
        manifest_path = generation_directory / "manifest.json"
        pointer_path = root / "_fundamental_latest.json"
        _atomic_write(manifest_path, manifest_bytes)
        _atomic_write(pointer_path, pointer_bytes)
        _fsync_directory(generation_directory)
        _fsync_directory(root / "_fundamental_generations")
        _fsync_directory(root)
        if (
            manifest_path.read_bytes() != manifest_bytes
            or pointer_path.read_bytes() != pointer_bytes
        ):
            _fail("STAGED_JSON_READBACK_MISMATCH", "staged pointer or manifest bytes changed")
        validated = validate_successor_provenance(
            pointer,
            manifest,
            generation_root=root,
            historical_only=True,
        )
        return SuccessorStagingCapture(
            generation_id=resolved_id,
            staging_root=root.resolve(strict=True),
            pointer_path=pointer_path.resolve(strict=True),
            pointer_bytes=pointer_bytes,
            pointer_sha256=hashlib.sha256(pointer_bytes).hexdigest(),
            manifest_path=manifest_path.resolve(strict=True),
            manifest_bytes=manifest_bytes,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            table_paths={name: path.resolve(strict=True) for name, path in table_paths.items()},
            table_sha256={name: evidence["sha256"] for name, evidence in table_manifest.items()},
            provider_evidence_files=evidence_files,
            predecessor_binding=dict(bundle.predecessor_binding),
            target_bindings=dict(targets),
            provenance_binding_sha256=str(validated["envelope_sha256"]),
        )
    except Exception:
        # The root is isolated and never canonical.  Leave it in place for
        # audit/diagnosis instead of deleting evidence after a failed readback.
        raise


def validate_successor_provenance(
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    generation_root: str | Path | None = None,
    historical_only: bool = True,
) -> dict[str, Any]:
    """Independently validate a v3 mixed-history provenance envelope.

    ``historical_only=True`` authenticates the immutable generation and all
    recorded bindings without reopening mutable market/PIT evidence.  Promotion
    preflight should pass ``False`` immediately before CAS so the target market,
    PIT membership and expected-scope files are hashed again.

    Exact v3 envelope fields are intentionally strict: schema/status/source,
    history state, provider/metadata bindings, predecessor, target bindings,
    flattened chain, support/derivation/readiness/keyset/resource bindings, raw
    and output fingerprints, quarantine identity, machine states, and the
    envelope SHA.
    """

    if not isinstance(historical_only, bool):
        _fail("INVALID_VALIDATION_MODE", "historical_only must be bool")
    pointer_payload = dict(pointer)
    manifest_payload = dict(manifest)
    if (
        pointer_payload.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer_payload.get("status") != "OK"
        or manifest_payload.get("schema_version") != "cn-fundamental-generation.v1"
        or manifest_payload.get("status") != "OK"
        or pointer_payload.get("generation_id") != manifest_payload.get("generation_id")
    ):
        _fail("SUCCESSOR_POINTER_MANIFEST_MISMATCH", "pointer/manifest contract is invalid")
    generation_id = str(pointer_payload.get("generation_id") or "").strip()
    if not generation_id:
        _fail("SUCCESSOR_GENERATION_ID_MISSING", "successor generation id is missing")
    pointer_tables = dict(pointer_payload.get("tables", {}) or {})
    manifest_tables = dict(manifest_payload.get("tables", {}) or {})
    if set(pointer_tables) != set(FUNDAMENTAL_TABLES) or set(manifest_tables) != set(
        FUNDAMENTAL_TABLES
    ):
        _fail("SUCCESSOR_TABLE_SET_MISMATCH", "successor table set is incomplete")
    pointer_metadata = dict(pointer_payload.get("metadata", {}) or {})
    manifest_metadata = dict(manifest_payload.get("metadata", {}) or {})
    if pointer_metadata != manifest_metadata:
        _fail("SUCCESSOR_METADATA_MISMATCH", "pointer/manifest metadata differs")
    pointer_envelope = dict(pointer_payload.get("primary_provenance", {}) or {})
    envelope = dict(manifest_payload.get("primary_provenance", {}) or {})
    if pointer_envelope != envelope:
        _fail("SUCCESSOR_PROVENANCE_MISMATCH", "pointer/manifest provenance differs")
    envelope_sha = _mapping_hash(envelope, "envelope_sha256")
    required_fields = {
        "schema_version",
        "status",
        "source",
        "source_priority",
        "source_provenance",
        "history_state",
        "mixed_generation",
        "seam_trade_date",
        "prefix_contract",
        "suffix_contract",
        "support_provider_contract",
        "gate2_contract",
        "gate2_receipt_sha256",
        "provider_manifest_sha256",
        "metadata_sha256",
        "predecessor",
        "target_bindings",
        "successor_chain",
        "support_plan_sha256",
        "permanent_support_refs",
        "derivation_binding_sha256",
        "readiness_binding_sha256",
        "keyset_closure_sha256",
        "resource_preflight_sha256",
        "raw_table_fingerprints",
        "parent_prefix_frame_fingerprints",
        "output_frame_fingerprints",
        "output_parquet_sha256",
        "quarantine_identity",
        "machine_states",
        "envelope_sha256",
    }
    if set(pointer_envelope) != required_fields:
        _fail(
            "SUCCESSOR_PROVENANCE_FIELD_SET_MISMATCH",
            "v3 provenance fields are incomplete or expanded",
        )
    if (
        pointer_envelope.get("schema_version") != SUCCESSOR_PROVENANCE_SCHEMA
        or pointer_envelope.get("status") != SUCCESSOR_PROVENANCE_STATUS
        or pointer_envelope.get("source") != "live_tushare_safe_successor"
        or pointer_envelope.get("source_priority") != "tushare_primary"
        or pointer_envelope.get("source_provenance")
        != "live_tushare_explicit_safe_successor_mixed"
        or pointer_envelope.get("history_state") != "mixed"
        or pointer_envelope.get("mixed_generation") is not True
        or pointer_envelope.get("suffix_contract")
        != SUCCESSOR_DERIVATION_CONTRACT
        or not str(pointer_envelope.get("support_provider_contract") or "")
        or pointer_envelope.get("gate2_contract")
        != SUCCESSOR_READINESS_SCHEMA
    ):
        _fail("SUCCESSOR_PROVENANCE_STATE_MISMATCH", "v3 provenance state is invalid")
    machine_states = dict(pointer_envelope.get("machine_states", {}) or {})
    if machine_states != {
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }:
        _fail("SUCCESSOR_MACHINE_STATE_MISMATCH", "successor machine states changed")
    if canonical_json_sha256(pointer_metadata) != pointer_envelope.get("metadata_sha256"):
        _fail("SUCCESSOR_METADATA_BINDING_MISMATCH", "metadata SHA does not match")
    if (
        pointer_metadata.get("gate2_passed") is not True
        or pointer_metadata.get("gate2_contract")
        != SUCCESSOR_READINESS_SCHEMA
        or not _valid_sha256(pointer_metadata.get("gate2_receipt_sha256"))
        or pointer_metadata.get("prefix_gate_passed") is not True
        or pointer_metadata.get("suffix_gate_passed") is not True
        or pointer_metadata.get("structural_gate_passed") is not True
        or pointer_metadata.get("provider_status") != "live_tushare_safe_successor"
        or pointer_metadata.get("source_priority") != "tushare_primary"
        or pointer_metadata.get("source_provenance")
        != "live_tushare_explicit_safe_successor_mixed"
        or pointer_metadata.get("mixed") is not True
        or pointer_metadata.get("legacy_direct_reader_provenance") != "limited"
        or pointer_metadata.get("binding_aware_research_ready") is not True
        or pointer_metadata.get("homogeneous_history_ready") is not False
    ):
        _fail("SUCCESSOR_READINESS_STATE_MISMATCH", "metadata readiness state is invalid")
    provider = dict(pointer_metadata.get("provider_manifest", {}) or {})
    provider_binding = _validate_provider_manifest(provider)
    if canonical_json_sha256(provider) != pointer_envelope.get("provider_manifest_sha256"):
        _fail("SUCCESSOR_PROVIDER_BINDING_MISMATCH", "provider manifest SHA does not match")
    predecessor = dict(pointer_envelope.get("predecessor", {}) or {})
    _validate_reference(predecessor, label="predecessor")
    chain = dict(pointer_envelope.get("successor_chain", {}) or {})
    chain_fingerprint = _validate_successor_chain(chain)
    if chain["immediate_predecessor"] != predecessor:
        _fail("SUCCESSOR_IMMEDIATE_PREDECESSOR_MISMATCH", "chain predecessor differs")
    if chain["ancestor_generation_ids"][-1] != generation_id:
        _fail("SUCCESSOR_CHAIN_TIP_MISMATCH", "chain tip is not this generation")
    prefix_contract = dict(pointer_envelope.get("prefix_contract", {}) or {})
    if prefix_contract != {
        "provenance_schema_version": predecessor["provenance_schema_version"],
        "reference_sha256": predecessor["reference_sha256"],
    }:
        _fail("SUCCESSOR_PREFIX_CONTRACT_MISMATCH", "prefix contract changed")
    if (
        _strict_date(
            pointer_envelope.get("seam_trade_date"),
            label="successor seam",
        )
        != dict(chain["original_seam"])["cutoff"]
    ):
        _fail("SUCCESSOR_SEAM_MISMATCH", "original successor seam changed")
    target_cutoff = _strict_date(
        dict(chain.get("append_boundary", {}) or {}).get("target_cutoff"),
        label="successor target cutoff",
    )
    targets = _validate_target_bindings(
        dict(pointer_envelope.get("target_bindings", {}) or {}),
        target_cutoff=target_cutoff,
        read_files=not historical_only,
        require_sealed=True,
    )
    if dict(pointer_metadata.get("target_bindings", {}) or {}) != targets:
        _fail("SUCCESSOR_TARGET_METADATA_MISMATCH", "metadata target bindings differ")

    derivation = dict(pointer_metadata.get("derivation", {}) or {})
    readiness = dict(pointer_metadata.get("readiness", {}) or {})
    keyset = dict(pointer_metadata.get("keyset_closure", {}) or {})
    resource = dict(pointer_metadata.get("resource_preflight", {}) or {})
    if (
        _mapping_hash(derivation, "binding_sha256")
        != pointer_envelope.get("derivation_binding_sha256")
        or _mapping_hash(readiness, "binding_sha256")
        != pointer_envelope.get("readiness_binding_sha256")
        or _mapping_hash(keyset, "closure_binding_sha256")
        != pointer_envelope.get("keyset_closure_sha256")
        or _mapping_hash(resource, "binding_sha256")
        != pointer_envelope.get("resource_preflight_sha256")
    ):
        _fail("SUCCESSOR_EVIDENCE_BINDING_MISMATCH", "derivation/readiness evidence differs")
    prefix_mode = str(derivation.get("support_prefix_mode") or "")
    prefix_gate_valid = prefix_mode == SUCCESSOR_SUPPORT_PREFIX_VALIDATION_MODE
    if prefix_mode == SUCCESSOR_APPEND_FIRST_MODE:
        prefix_gate_valid = (
            _valid_sha256(
                derivation.get("historical_taint_registry_sha256")
            )
            and derivation.get("historical_taint_registry_sha256")
            == provider.get("historical_taint_registry_sha256")
            and readiness.get("prefix_gate_passed") is True
        )
    if (
        derivation.get("contract_version") != SUCCESSOR_DERIVATION_CONTRACT
        or derivation.get("ordinary_merge_used") is not False
        or not prefix_gate_valid
        or readiness.get("schema_version") != SUCCESSOR_READINESS_SCHEMA
        or readiness.get("gate2_contract") != SUCCESSOR_READINESS_SCHEMA
        or readiness.get("status") != "PASS"
        or readiness.get("gate2_passed") is not True
        or readiness.get("prefix_gate_passed") is not True
        or readiness.get("suffix_gate_passed") is not True
        or readiness.get("structural_gate_passed") is not True
        or readiness.get("blockers") != []
        or readiness.get("true_missing") != 0
        or readiness.get("mixed") is not True
        or readiness.get("legacy_direct_reader_provenance") != "limited"
        or readiness.get("binding_aware_research_ready") is not True
        or readiness.get("homogeneous_history_ready") is not False
        or resource.get("status") != "PASS"
    ):
        _fail("SUCCESSOR_GATE_STATE_MISMATCH", "successor gate evidence is not passed")
    if (
        readiness["binding_sha256"]
        != pointer_envelope.get("gate2_receipt_sha256")
        or readiness["binding_sha256"]
        != pointer_metadata.get("gate2_receipt_sha256")
        or derivation.get("support_provider_contract")
        != pointer_envelope.get("support_provider_contract")
    ):
        _fail("SUCCESSOR_GATE_BINDING_MISMATCH", "successor gate bindings differ")
    if pointer_envelope.get("support_plan_sha256") != derivation.get("support_plan_sha256"):
        _fail("SUCCESSOR_SUPPORT_BINDING_MISMATCH", "support plan binding differs")
    raw_fingerprints = dict(pointer_envelope.get("raw_table_fingerprints", {}) or {})
    if (
        raw_fingerprints != dict(provider.get("raw_table_fingerprints", {}) or {})
        or raw_fingerprints != dict(derivation.get("raw_table_fingerprints", {}) or {})
        or set(raw_fingerprints) != set(RAW_TABLES)
        or any(not _valid_sha256(value) for value in raw_fingerprints.values())
    ):
        _fail("SUCCESSOR_RAW_FINGERPRINT_MISMATCH", "raw table fingerprints differ")
    prefix_fingerprints = dict(
        pointer_envelope.get("parent_prefix_frame_fingerprints", {}) or {}
    )
    if (
        prefix_fingerprints
        != dict(derivation.get("parent_table_frame_fingerprints", {}) or {})
        or set(prefix_fingerprints) != set(FUNDAMENTAL_TABLES)
        or any(not _valid_sha256(value) for value in prefix_fingerprints.values())
    ):
        _fail("SUCCESSOR_PREFIX_FINGERPRINT_MISMATCH", "parent prefix fingerprints differ")
    output_frames = dict(pointer_envelope.get("output_frame_fingerprints", {}) or {})
    output_sha = dict(pointer_envelope.get("output_parquet_sha256", {}) or {})
    if set(output_frames) != set(FUNDAMENTAL_TABLES) or set(output_sha) != set(FUNDAMENTAL_TABLES):
        _fail("SUCCESSOR_OUTPUT_IDENTITY_MISMATCH", "output identities are incomplete")
    for table_name in FUNDAMENTAL_TABLES:
        table = dict(manifest_tables[table_name] or {})
        if (
            table.get("frame_fingerprint") != output_frames[table_name]
            or table.get("sha256") != output_sha[table_name]
            or not _valid_sha256(output_frames[table_name])
            or not _valid_sha256(output_sha[table_name])
        ):
            _fail("SUCCESSOR_OUTPUT_IDENTITY_MISMATCH", f"output identity differs: {table_name}")
    quarantine = dict(pointer_envelope.get("quarantine_identity", {}) or {})
    if (
        quarantine.get("mode") != "exact_predecessor_bytes"
        or quarantine.get("exact") is not True
        or quarantine.get("predecessor_sha256")
        != predecessor["table_sha256"]["fundamental_quarantine"]
        or quarantine.get("successor_sha256") != output_sha["fundamental_quarantine"]
        or quarantine.get("predecessor_sha256") != quarantine.get("successor_sha256")
    ):
        _fail(
            "SUCCESSOR_QUARANTINE_IDENTITY_MISMATCH",
            "quarantine bytes are not inherited exactly",
        )
    refs = dict(pointer_envelope.get("permanent_support_refs", {}) or {})
    if set(refs) != set(PERMANENT_SUPPORT_REFERENCE_NAMES):
        _fail(
            "INCOMPLETE_SUPPORT_REFERENCE_SET",
            "v3 provenance permanent support refs are incomplete",
        )
    resolved_generation_root = (
        Path(generation_root).expanduser().resolve(strict=True)
        if generation_root is not None
        else None
    )
    if resolved_generation_root is None:
        _fail(
            "PERMANENT_SUPPORT_ROOT_REQUIRED",
            "generation_root is required to authenticate permanent support bytes",
        )
    evidence_root = (
        resolved_generation_root
        / "_fundamental_generations"
        / generation_id
        / "provider_evidence"
    ).resolve(strict=True)
    permanent_payloads: dict[str, bytes] = {}
    for name, value in refs.items():
        if not isinstance(value, Mapping) or not _valid_sha256(value.get("sha256")):
            _fail("INVALID_SUPPORT_REFERENCE", f"permanent support reference is invalid: {name}")
        relative = Path(str(value.get("path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            _fail("INVALID_SUPPORT_REFERENCE", f"support reference path is unsafe: {name}")
        path = (evidence_root / relative).resolve(strict=True)
        if evidence_root not in path.parents or _sha256_file(path) != str(value["sha256"]).lower():
            _fail("SUPPORT_REFERENCE_TAMPER", f"permanent support bytes changed: {name}")
        permanent_payloads[name] = path.read_bytes()
    if (
        hashlib.sha256(permanent_payloads["predecessor_pointer"]).hexdigest()
        != predecessor["pointer_sha256"]
        or hashlib.sha256(permanent_payloads["predecessor_manifest"]).hexdigest()
        != predecessor["manifest_sha256"]
    ):
        _fail(
            "PREDECESSOR_PERMANENT_REF_MISMATCH",
            "permanent predecessor bytes do not match the v3 envelope",
        )
    try:
        predecessor_pointer = json.loads(
            permanent_payloads["predecessor_pointer"].decode("utf-8")
        )
        predecessor_manifest = json.loads(
            permanent_payloads["predecessor_manifest"].decode("utf-8")
        )
    except Exception as exc:
        _fail(
            "INVALID_PARENT_CLOSURE_BYTES",
            "sealed predecessor pointer/manifest is invalid JSON",
        )
        raise AssertionError from exc
    if (
        not isinstance(predecessor_pointer, Mapping)
        or str(predecessor_pointer.get("generation_id") or "")
        != predecessor["generation_id"]
        or not isinstance(predecessor_manifest, Mapping)
        or str(predecessor_manifest.get("generation_id") or "")
        != predecessor["generation_id"]
    ):
        _fail(
            "PREDECESSOR_CLOSURE_GENERATION_MISMATCH",
            "sealed predecessor pointer/manifest generation differs",
        )
    sealed_target_payloads: dict[str, bytes] = {}
    for name, binding in targets.items():
        if name == "binding_sha256":
            continue
        sealed = dict(binding.get("sealed_ref", {}) or {})
        relative = Path(str(sealed.get("path") or ""))
        path = (evidence_root / relative).resolve(strict=True)
        if (
            evidence_root not in path.parents
            or _sha256_file(path) != str(sealed.get("sha256") or "").lower()
            or str(sealed.get("sha256") or "").lower()
            != str(binding.get("sha256") or "").lower()
        ):
            _fail("TARGET_SEALED_REF_TAMPER", f"sealed target bytes changed: {name}")
        sealed_target_payloads[name] = path.read_bytes()
        for reference in binding.get("immutable_refs", []):
            ref_path = Path(str(reference.get("path") or "")).expanduser().resolve(
                strict=True
            )
            ref_stat = ref_path.stat()
            if (
                _sha256_file(ref_path)
                != str(reference.get("sha256") or "").lower()
                or (
                    reference.get("size") is not None
                    and ref_stat.st_size != int(reference["size"])
                )
            ):
                _fail(
                    "TARGET_IMMUTABLE_REF_TAMPER",
                    f"target immutable ref changed: {name}",
                )
    for name in ("market_pointer", "pit_pointer"):
        try:
            parsed = json.loads(sealed_target_payloads[name].decode("utf-8"))
        except Exception as exc:
            _fail("INVALID_TARGET_POINTER", f"sealed target pointer is invalid: {name}")
            raise AssertionError from exc
        if not isinstance(parsed, Mapping):
            _fail("INVALID_TARGET_POINTER", f"sealed target pointer is invalid: {name}")
    table_readback: dict[str, Mapping[str, Any]] = {}
    root = resolved_generation_root
    assert root is not None
    pointer_path = (root / "_fundamental_latest.json").resolve(strict=True)
    expected_manifest_path = str(
        Path("_fundamental_generations") / generation_id / "manifest.json"
    )
    if pointer_payload.get("manifest_path") != expected_manifest_path:
        _fail(
            "SUCCESSOR_MANIFEST_PATH_MISMATCH",
            "successor manifest path is noncanonical",
        )
    manifest_path = (
        root / str(pointer_payload.get("manifest_path") or "")
    ).resolve(strict=True)
    if root not in manifest_path.parents:
        _fail("UNSAFE_SUCCESSOR_MANIFEST_PATH", "successor manifest escapes generation root")
    pointer_bytes = pointer_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    try:
        pointer_readback = json.loads(pointer_bytes.decode("utf-8"))
        manifest_readback = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        _fail("SUCCESSOR_JSON_READBACK_MISMATCH", "staged pointer/manifest JSON is invalid")
        raise AssertionError from exc
    if pointer_readback != pointer_payload or manifest_readback != manifest_payload:
        _fail(
            "SUCCESSOR_JSON_READBACK_MISMATCH",
            "staged pointer/manifest bytes do not decode to the supplied closure",
        )
    provider_path = (evidence_root / "provider_manifest.json").resolve(strict=True)
    if (
        evidence_root not in provider_path.parents
        or provider_path.read_bytes() != _canonical_bytes(provider)
    ):
        _fail(
            "SUCCESSOR_PROVIDER_FILE_READBACK_MISMATCH",
            "sealed provider manifest bytes changed",
        )
    for relative_text, digest in dict(provider.get("evidence_files", {}) or {}).items():
        relative = Path(str(relative_text))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            _fail(
                "UNSAFE_PROVIDER_EVIDENCE_PATH",
                "provider evidence path is unsafe",
            )
        evidence_path = (evidence_root / relative).resolve(strict=True)
        if (
            evidence_root not in evidence_path.parents
            or _sha256_file(evidence_path) != str(digest).lower()
        ):
            _fail(
                "PROVIDER_EVIDENCE_READBACK_MISMATCH",
                f"provider evidence changed: {relative}",
            )
    source_validation: dict[str, Any] | None = None
    source_manifest_sha256 = provider.get("source_fileset_manifest_sha256")
    if source_manifest_sha256 is not None:
        from .fundamental_successor_source import (
            validate_successor_support_fileset,
        )

        implementation_sha256 = str(
            provider.get("implementation_sha256") or ""
        ).lower()
        if (
            not _valid_sha256(source_manifest_sha256)
            or not _valid_sha256(implementation_sha256)
        ):
            _fail(
                "SOURCE_FILESET_BINDING_INVALID",
                "source fileset binding is incomplete",
            )
        validated_source = validate_successor_support_fileset(
            evidence_root / "source",
            expected_implementation_sha256=implementation_sha256,
        )
        if (
            validated_source.get("manifest_sha256")
            != str(source_manifest_sha256).lower()
        ):
            _fail(
                "SOURCE_FILESET_MANIFEST_MISMATCH",
                "source fileset manifest changed after staging",
            )
        source_validation = {
            "implementation_sha256": implementation_sha256,
            "manifest_sha256": validated_source["manifest_sha256"],
            "resource_sha256": dict(
                validated_source.get("resource_accounting", {}) or {}
            ).get("resource_sha256"),
            "schema_version": validated_source["schema_version"],
            "status": "PASS",
        }
        if pointer_metadata.get("source_fileset_validation") != source_validation:
            _fail(
                "SOURCE_FILESET_VALIDATION_MISMATCH",
                "staging and promotion source replay receipts differ",
            )
    elif pointer_metadata.get("source_fileset_validation") is not None:
        _fail(
            "SOURCE_FILESET_BINDING_MISSING",
            "source validation receipt has no source fileset binding",
        )
    historical_taint_validation: dict[str, Any] | None = None
    if derivation.get("support_prefix_mode") == SUCCESSOR_APPEND_FIRST_MODE:
        from .fundamental_historical_taint import (
            validate_historical_taint_registry,
        )

        declared_registry_sha = str(
            provider.get("historical_taint_registry_sha256") or ""
        ).lower()
        declared_registry_file_sha = str(
            provider.get("historical_taint_registry_file_sha256") or ""
        ).lower()
        if (
            not _valid_sha256(declared_registry_file_sha)
            or _sha256_file(
                evidence_root / "historical_taint" / "registry.json"
            )
            != declared_registry_file_sha
        ):
            _fail(
                "HISTORICAL_TAINT_BINDING_MISMATCH",
                "historical-taint registry file digest differs",
            )
        registry = validate_historical_taint_registry(
            evidence_root / "historical_taint" / "registry.json",
            evidence_root=evidence_root,
            predecessor=predecessor,
            delta_fileset_root=evidence_root / "source",
        )
        historical_taint_validation = {
            "schema_version": registry["schema_version"],
            "status": registry["status"],
            "classification": registry["classification"],
            "historical_conflict_count": registry[
                "historical_conflict_count"
            ],
            "current_window_material_conflict_count": 0,
            "same_period_delta_row_count": 0,
            "registry_sha256": declared_registry_sha,
        }
        if (
            registry.get("registry_sha256") != declared_registry_sha
            or pointer_metadata.get("historical_taint_validation")
            != historical_taint_validation
        ):
            _fail(
                "HISTORICAL_TAINT_VALIDATION_MISMATCH",
                "staging and promotion historical-taint receipts differ",
            )
    elif pointer_metadata.get("historical_taint_validation") is not None:
        _fail(
            "HISTORICAL_TAINT_VALIDATION_UNEXPECTED",
            "prefix-replay successor unexpectedly carries taint isolation",
        )
    for table_name in FUNDAMENTAL_TABLES:
        if pointer_tables[table_name] != str(
            Path("_fundamental_generations")
            / generation_id
            / f"{table_name}.parquet"
        ):
            _fail(
                "SUCCESSOR_TABLE_PATH_MISMATCH",
                f"table path is noncanonical: {table_name}",
            )
        path = (root / str(pointer_tables[table_name])).resolve(strict=True)
        if root not in path.parents:
            _fail(
                "UNSAFE_SUCCESSOR_TABLE_PATH",
                f"table escapes generation root: {table_name}",
            )
        observed = _streaming_table_evidence(path)
        if observed != manifest_tables[table_name]:
            _fail(
                "SUCCESSOR_TABLE_READBACK_MISMATCH",
                f"table readback differs: {table_name}",
            )
        table_readback[table_name] = observed
    predecessor_receipt = {
        **predecessor,
        "exact_pointer_bytes_b64": base64.b64encode(
            permanent_payloads["predecessor_pointer"]
        ).decode("ascii"),
        "exact_manifest_bytes_b64": base64.b64encode(
            permanent_payloads["predecessor_manifest"]
        ).decode("ascii"),
    }
    market_receipt = {
        "live_pointer_path": targets["market_pointer"]["path"],
        "pointer_sha256": targets["market_pointer"]["sha256"],
        "as_of": targets["market_pointer"]["as_of"],
        "exact_pointer_bytes_b64": base64.b64encode(
            sealed_target_payloads["market_pointer"]
        ).decode("ascii"),
        "immutable_refs": targets["market_pointer"].get("immutable_refs", {}),
    }
    pit_receipt = {
        "live_pointer_path": targets["pit_pointer"]["path"],
        "pointer_sha256": targets["pit_pointer"]["sha256"],
        "as_of": targets["pit_pointer"]["as_of"],
        "exact_pointer_bytes_b64": base64.b64encode(
            sealed_target_payloads["pit_pointer"]
        ).decode("ascii"),
        "immutable_refs": targets["pit_pointer"].get("immutable_refs", {}),
    }
    provider_evidence_files = {
        path.relative_to(evidence_root.parent).as_posix(): _sha256_file(path)
        for path in sorted(evidence_root.rglob("*"))
        if path.is_file()
    }
    original_seam = dict(chain["original_seam"])["cutoff"]
    immediate_parent_cutoff = dict(chain["append_boundary"])["parent_cutoff"]
    return {
        "schema_version": SUCCESSOR_PROVENANCE_SCHEMA,
        "status": SUCCESSOR_PROVENANCE_STATUS,
        "generation_id": generation_id,
        "envelope_sha256": envelope_sha,
        "provenance_binding_sha256": envelope_sha,
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "table_sha256": output_sha,
        "provider_evidence_files": provider_evidence_files,
        "provider_binding_sha256": provider_binding,
        "source_fileset_validation": source_validation,
        "successor_chain_fingerprint": chain_fingerprint,
        "predecessor": predecessor_receipt,
        "target_bindings": targets,
        "market_binding": market_receipt,
        "pit_binding": pit_receipt,
        "successor_chain": chain,
        "original_seam": original_seam,
        "immediate_parent_cutoff": immediate_parent_cutoff,
        "target_cutoff": target_cutoff,
        "machine_states": machine_states,
        "historical_only": historical_only,
        "table_readback": table_readback,
    }


__all__ = [
    "DAILY_KEY_FIELDS",
    "FUNDAMENTAL_SAFE_SUCCESSOR_DERIVATION_CONTRACT",
    "FUNDAMENTAL_SAFE_SUCCESSOR_PROVENANCE_SCHEMA",
    "FUNDAMENTAL_SAFE_SUCCESSOR_READINESS_SCHEMA",
    "FUNDAMENTAL_TABLES",
    "PERIOD_KEY_FIELDS",
    "RAW_TABLES",
    "SUCCESSOR_CHAIN_SCHEMA",
    "SUCCESSOR_DERIVATION_CONTRACT",
    "SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT",
    "SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SCHEMA",
    "SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256",
    "SUCCESSOR_KEYSET_SCHEMA",
    "SUCCESSOR_PLAN_SCHEMA",
    "SUCCESSOR_PROVIDER_MANIFEST_SCHEMA",
    "SUCCESSOR_PROVENANCE_SCHEMA",
    "SUCCESSOR_PROVENANCE_STATUS",
    "SUCCESSOR_READINESS_SCHEMA",
    "SafeSuccessorError",
    "SuccessorBundle",
    "SuccessorStagingCapture",
    "assemble_safe_successor",
    "build_keyset_closure",
    "build_successor_chain",
    "capture_parent_closure",
    "replay_successor_event_trace",
    "seal_successor_provider_manifest",
    "seal_support_plan",
    "stage_successor_generation",
    "successor_resource_preflight",
    "successor_financial_row_binding",
    "successor_period_anchor_equal",
    "successor_period_winner",
    "validate_successor_provenance",
]
