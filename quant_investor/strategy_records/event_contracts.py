"""Shared exact v1 event contracts for publication and every native reader."""

from collections.abc import Mapping
from datetime import date, datetime
from pathlib import PurePosixPath
import re
from zoneinfo import ZoneInfo

from .event_store import (
    EVENT_CLOSURE_SCHEMA,
    EVENT_GENERATION_SCHEMA,
    EVENT_POINTER_SCHEMA,
    EVENT_DIMENSIONS,
    StrategyEventStoreError,
    _validate_seal,
    _ID,
    _SHA,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
SYMBOLIC_RECEIPT = re.compile(
    r"catalog:([A-Za-z0-9][A-Za-z0-9._-]{0,127})#receipt:([A-Za-z0-9][A-Za-z0-9._-]{0,255})\Z"
)
CLOSURE_FIELDS = frozenset(
    {
        "schema_id",
        "trade_date",
        "sealed_at",
        "cutoff_at",
        "status",
        "dimensions",
        "policy_ref",
        "owner_declaration_ref",
        "source_receipt_ref",
        "late_event_behavior",
        "actual_holdings_mutation_authority",
        "cash_mutation_authority",
        "broker_order_trade_authority",
        "content_sha256",
    }
)
GENERATION_FIELDS = frozenset(
    {
        "schema_id",
        "generation_id",
        "generated_at",
        "policy_ref",
        "trade_dates",
        "closures",
        "late_event_behavior",
        "broker_order_trade_authority",
        "content_sha256",
    }
)
POINTER_FIELDS = frozenset(
    {
        "schema_id",
        "generation_id",
        "generation",
        "trade_dates",
        "previous_pointer_sha256",
        "broker_order_trade_authority",
        "content_sha256",
    }
)


def instant(value, *, label):
    try:
        if type(value) is not str or value != value.strip():
            raise ValueError("timestamp must be text")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone missing")
        return parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyEventStoreError(f"event {label} timestamp is invalid") from exc


def event_date(value):
    try:
        if type(value) is not str or date.fromisoformat(value).isoformat() != value:
            raise ValueError("noncanonical date")
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise StrategyEventStoreError("event trade date is invalid") from exc


def source_ref(value, *, symbolic=False):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise StrategyEventStoreError("event source ref shape is invalid")
    text, digest = value["path"], value["sha256"]
    if type(digest) is not str or _SHA.fullmatch(digest) is None or type(text) is not str:
        raise StrategyEventStoreError("event source ref is invalid")
    if symbolic and SYMBOLIC_RECEIPT.fullmatch(text):
        return dict(value)
    path = PurePosixPath(text)
    if (
        not text
        or len(text) > 4096
        or path.is_absolute()
        or str(path) != text
        or any(part in {".", ".."} for part in path.parts)
        or any(ord(c) < 32 or ord(c) > 126 or c in "\\:#" for c in text)
    ):
        raise StrategyEventStoreError("event physical source path is invalid")
    return dict(value)


def _shape(value, fields, schema, label):
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema_id") != schema:
        raise StrategyEventStoreError(f"event {label} fields/schema mismatch")
    _validate_seal(value, label="event " + label)
    if value["broker_order_trade_authority"] is not False:
        raise StrategyEventStoreError(f"event {label} claims forbidden authority")


def validate_closure_contract(value):
    _shape(value, CLOSURE_FIELDS, EVENT_CLOSURE_SCHEMA, "closure")
    day = event_date(value["trade_date"])
    sealed = instant(value["sealed_at"], label="closure seal")
    cutoff = instant(value["cutoff_at"], label="closure cutoff")
    if sealed < cutoff or cutoff.astimezone(SHANGHAI).date() != day:
        raise StrategyEventStoreError("event closure cutoff/session ordering invalid")
    dimensions = value["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != set(EVENT_DIMENSIONS):
        raise StrategyEventStoreError("event closure dimensions are incomplete")
    if value["status"] != "CLOSED_EMPTY" or any(
        dimensions[name] != {"status": "CLOSED_EMPTY", "events": []} for name in EVENT_DIMENSIONS
    ):
        raise StrategyEventStoreError("event closure contains an unclosed dimension")
    if (
        value["actual_holdings_mutation_authority"] is not False
        or value["cash_mutation_authority"] is not False
        or value["late_event_behavior"] != "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED"
    ):
        raise StrategyEventStoreError("event closure claims forbidden authority or restatement")
    source_ref(value["policy_ref"])
    source_ref(value["owner_declaration_ref"])
    if value["source_receipt_ref"] is not None:
        source_ref(value["source_receipt_ref"], symbolic=True)
    return dict(value)


def _dates(values):
    if type(values) is not list or not values:
        raise StrategyEventStoreError("event date set is empty or invalid")
    for value in values:
        event_date(value)
    if values != sorted(set(values)):
        raise StrategyEventStoreError("event date set is not sorted unique")
    return values


def validate_pointer(value):
    _shape(value, POINTER_FIELDS, EVENT_POINTER_SCHEMA, "pointer")
    generation_id = value["generation_id"]
    if type(generation_id) is not str or _ID.fullmatch(generation_id) is None:
        raise StrategyEventStoreError("event generation ID is invalid")
    ref = source_ref(value["generation"])
    if ref["path"] != f"generations/{generation_id}.v1.json":
        raise StrategyEventStoreError("event generation path mismatch")
    previous = value["previous_pointer_sha256"]
    if previous is not None and (type(previous) is not str or _SHA.fullmatch(previous) is None):
        raise StrategyEventStoreError("event previous pointer SHA invalid")
    _dates(value["trade_dates"])
    return dict(value)


def validate_generation(value, *, pointer=None):
    _shape(value, GENERATION_FIELDS, EVENT_GENERATION_SCHEMA, "generation")
    if type(value["generation_id"]) is not str or _ID.fullmatch(value["generation_id"]) is None:
        raise StrategyEventStoreError("event generation ID is invalid")
    generated = instant(value["generated_at"], label="generation")
    source_ref(value["policy_ref"])
    if (
        value["late_event_behavior"] != "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED"
        or type(value["closures"]) is not list
    ):
        raise StrategyEventStoreError("event generation restatement/closure shape invalid")
    rows = [validate_closure_contract(row) for row in value["closures"]]
    dates = _dates(value["trade_dates"])
    if [row["trade_date"] for row in rows] != dates:
        raise StrategyEventStoreError("event generation closure date set mismatch")
    if any(instant(row["sealed_at"], label="closure seal") > generated for row in rows):
        raise StrategyEventStoreError("event generation precedes closure seal")
    if pointer is not None and (
        value["generation_id"] != pointer["generation_id"] or dates != pointer["trade_dates"]
    ):
        raise StrategyEventStoreError("event pointer/generation closure mismatch")
    return dict(value)
