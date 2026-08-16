"""Deterministic compilation of official exchange rules and notices.

The compiler never discovers sources, calls a provider, or infers sessions
from market bars.  Its inputs are already admitted, exact-byte official-source
projections.  Unknown notice semantics and incomplete category closure remain
outside this module and must fail before a compilation request is constructed.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final
from zoneinfo import ZoneInfo

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)
from quant_investor.system.errors import SystemContractError, SystemPreconditionError
from quant_investor.system.store import validate_object_ref

COMPILATION_KIND: Final = "system.exchange_calendar_compilation"
INDEX_CLOSURE_KIND: Final = "system.exchange_calendar_index_closure"
COMPILER_RELATIVE_PATH: Final = "quant_investor/market/exchange_calendar_compilation.py"
TIMEZONE: Final = "Asia/Shanghai"
RUNTIME_OPEN: Final = "09:30:00"
RUNTIME_CLOSE: Final = "15:00:00"
EXCHANGE_ISSUERS: Final = {
    "SSE": "SSE_OFFICIAL",
    "SZSE": "SZSE_OFFICIAL",
    "BSE": "BSE_OFFICIAL",
}
PRECEDENCE_RULES: Final = [
    "SESSION_CHANGE_NOTICE",
    "TEMPORARY_CLOSURE_NOTICE",
    "ANNUAL_HOLIDAY_NOTICE",
    "TRADING_WEEK_RULE",
]
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ROW_FIELDS: Final = frozenset(
    {
        "exchange_id",
        "issuer",
        "weekly_rule_intervals",
        "closure_dates",
        "session_intervals",
        "native_capture_refs",
        "decoder_admission_refs",
        "index_closure_refs",
    }
)
_RULE_INTERVAL_FIELDS: Final = frozenset({"start_date", "end_date", "weekdays"})
_SESSION_FIELDS: Final = frozenset({"phase", "opens_local", "closes_local"})
_INDEX_FIELDS: Final = frozenset(
    {
        "index_closure_id",
        "state",
        "exchange_id",
        "issuer",
        "category",
        "root_capture_ref",
        "page_capture_refs",
        "reported_page_count",
        "reported_item_count",
        "observed_item_count",
        "earliest_publish_date",
        "latest_publish_date",
        "entry_rows",
        "body_capture_refs",
        "pagination_complete",
        "date_window_complete",
        "unknown_relevant_count",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compiler_code_sha256() -> str:
    return _sha256(Path(__file__).read_bytes())


def compiler_ast_sha256() -> str:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=COMPILER_RELATIVE_PATH)
    return _sha256(ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8"))


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value or len(value) > 200:
        raise SystemContractError(f"{label} is not canonical text")
    return value


def _date(value: Any, *, label: str) -> date:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not an ISO date") from exc
    if parsed.isoformat() != value:
        raise SystemContractError(f"{label} is not an ISO date")
    return parsed


def _time(value: Any, *, label: str) -> time:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not a local time")
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not a local time") from exc
    if parsed.strftime("%H:%M:%S") != value:
        raise SystemContractError(f"{label} is not second-precision local time")
    return parsed


def _refs(
    value: Any,
    *,
    label: str,
    expected_kind: str,
    minimum: int = 1,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) < minimum:
        raise SystemContractError(f"{label} is incomplete")
    refs = [validate_object_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    if any(row["kind"] != expected_kind for row in refs):
        raise SystemContractError(f"{label} contains the wrong authority kind")
    keys = [canonical_json_bytes(row) for row in refs]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SystemContractError(f"{label} is not canonical sorted unique")
    return refs


def _file_ref(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"relative_path", "byte_sha256"}:
        raise SystemContractError(f"{label} fields are not exact")
    relative = value["relative_path"]
    path = PurePosixPath(relative) if type(relative) is str else None
    if (
        path is None
        or path.is_absolute()
        or str(path) != relative
        or any(part in {"", ".", ".."} for part in path.parts)
        or type(value["byte_sha256"]) is not str
        or _SHA_RE.fullmatch(value["byte_sha256"]) is None
    ):
        raise SystemContractError(f"{label} is invalid")
    return dict(value)


def validate_index_closure(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    del document
    raise SystemPreconditionError(
        "shallow calendar index closure is disabled; native page/body replay is required"
    )


def _disabled_shallow_validate_index_closure(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(document, expected_kind=INDEX_CLOSURE_KIND)
    except ContractError as exc:
        raise SystemContractError("calendar index closure contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _INDEX_FIELDS or payload["state"] != "COMPLETE":
        raise SystemContractError("calendar index closure fields differ")
    exchange = payload["exchange_id"]
    if exchange not in EXCHANGE_ISSUERS or payload["issuer"] != EXCHANGE_ISSUERS[exchange]:
        raise SystemContractError("calendar index closure authority differs")
    if (
        payload["pagination_complete"] is not True
        or payload["date_window_complete"] is not True
        or payload["unknown_relevant_count"] != 0
    ):
        raise SystemPreconditionError("calendar index/category closure is incomplete")
    page_refs = _refs(
        payload["page_capture_refs"],
        label="page_capture_refs",
        expected_kind="system.exchange_calendar_capture",
    )
    body_refs = _refs(
        payload["body_capture_refs"],
        label="body_capture_refs",
        expected_kind="system.exchange_calendar_capture",
        minimum=0,
    )
    root_ref = validate_object_ref(payload["root_capture_ref"], label="root_capture_ref")
    if root_ref["kind"] != "system.exchange_calendar_capture":
        raise SystemContractError("calendar index root capture kind differs")
    page_count = payload["reported_page_count"]
    reported_items = payload["reported_item_count"]
    observed_items = payload["observed_item_count"]
    entries = payload["entry_rows"]
    if (
        type(page_count) is not int
        or page_count <= 0
        or len(page_refs) != page_count
        or type(reported_items) is not int
        or reported_items < 0
        or observed_items != reported_items
        or type(entries) is not list
        or len(entries) != observed_items
        or len(body_refs) > observed_items
    ):
        raise SystemPreconditionError("calendar index pagination/item closure differs")
    entry_bytes = [canonical_json_bytes(row) for row in entries]
    if entry_bytes != sorted(entry_bytes) or len(entry_bytes) != len(set(entry_bytes)):
        raise SystemContractError("calendar index entries are not canonical unique")
    _date(payload["earliest_publish_date"], label="earliest_publish_date")
    _date(payload["latest_publish_date"], label="latest_publish_date")
    if payload["earliest_publish_date"] > payload["latest_publish_date"]:
        raise SystemContractError("calendar index publication window is inverted")
    _identifier(payload["index_closure_id"], label="index_closure_id")
    _identifier(payload["category"], label="category")
    return artifact


def _normalize_source_row(value: Any, *, coverage: date, cutoff: date) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _SOURCE_ROW_FIELDS:
        raise SystemContractError("calendar source exchange row fields are not exact")
    exchange = value["exchange_id"]
    if exchange not in EXCHANGE_ISSUERS or value["issuer"] != EXCHANGE_ISSUERS[exchange]:
        raise SystemContractError("calendar source exchange authority differs")
    intervals: list[dict[str, Any]] = []
    previous_end: date | None = None
    for index, raw in enumerate(value["weekly_rule_intervals"]):
        if type(raw) is not dict or set(raw) != _RULE_INTERVAL_FIELDS:
            raise SystemContractError("weekly rule interval fields are not exact")
        start = _date(raw["start_date"], label=f"weekly_rule[{index}].start_date")
        end = _date(raw["end_date"], label=f"weekly_rule[{index}].end_date")
        weekdays = raw["weekdays"]
        if (
            start > end
            or type(weekdays) is not list
            or weekdays != sorted(set(weekdays))
            or any(type(day) is not int or day < 1 or day > 7 for day in weekdays)
            or not weekdays
            or (previous_end is not None and start != previous_end + timedelta(days=1))
        ):
            raise SystemPreconditionError("weekly rule intervals overlap or contain a gap")
        intervals.append(
            {"start_date": start.isoformat(), "end_date": end.isoformat(), "weekdays": weekdays}
        )
        previous_end = end
    if (
        not intervals
        or intervals[0]["start_date"] > coverage.isoformat()
        or intervals[-1]["end_date"] < cutoff.isoformat()
    ):
        raise SystemPreconditionError("weekly rule intervals do not cover the compilation window")
    closures = value["closure_dates"]
    if type(closures) is not list:
        raise SystemContractError("closure_dates is not a list")
    closure_dates = [
        _date(item, label=f"closure_dates[{index}]").isoformat()
        for index, item in enumerate(closures)
    ]
    if closure_dates != sorted(set(closure_dates)):
        raise SystemContractError("closure_dates is not canonical sorted unique")
    sessions: list[dict[str, str]] = []
    for index, raw in enumerate(value["session_intervals"]):
        if type(raw) is not dict or set(raw) != _SESSION_FIELDS:
            raise SystemContractError("session interval fields are not exact")
        opens = _time(raw["opens_local"], label=f"session[{index}].opens_local")
        closes = _time(raw["closes_local"], label=f"session[{index}].closes_local")
        if opens >= closes:
            raise SystemContractError("session interval is inverted")
        sessions.append(
            {
                "phase": _identifier(raw["phase"], label=f"session[{index}].phase"),
                "opens_local": opens.strftime("%H:%M:%S"),
                "closes_local": closes.strftime("%H:%M:%S"),
            }
        )
    if not sessions or sessions != sorted(
        sessions, key=lambda row: (row["opens_local"], row["phase"])
    ):
        raise SystemContractError("session intervals are not canonical")
    if not any(
        row["opens_local"] == "09:30:00" and row["closes_local"] == "11:30:00" for row in sessions
    ) or not any(
        row["opens_local"] == "13:00:00" and row["closes_local"] == "15:00:00" for row in sessions
    ):
        raise SystemPreconditionError("official continuous-auction intervals are incomplete")
    return {
        "exchange_id": exchange,
        "issuer": value["issuer"],
        "weekly_rule_intervals": intervals,
        "closure_dates": closure_dates,
        "session_intervals": sessions,
        "native_capture_refs": _refs(
            value["native_capture_refs"],
            label="native_capture_refs",
            expected_kind="system.exchange_calendar_capture",
        ),
        "decoder_admission_refs": _refs(
            value["decoder_admission_refs"],
            label="decoder_admission_refs",
            expected_kind="system.exchange_calendar_decoder_admission",
        ),
        "index_closure_refs": _refs(
            value["index_closure_refs"],
            label="index_closure_refs",
            expected_kind=INDEX_CLOSURE_KIND,
        ),
    }


def _compile_row(source: Mapping[str, Any], *, coverage: date, cutoff: date) -> dict[str, Any]:
    rules = source["weekly_rule_intervals"]
    closures = set(source["closure_dates"])
    zone = ZoneInfo(TIMEZONE)
    daily: list[dict[str, Any]] = []
    open_sessions: list[str] = []
    cursor = coverage
    while cursor <= cutoff:
        matches = [
            interval
            for interval in rules
            if interval["start_date"] <= cursor.isoformat() <= interval["end_date"]
        ]
        if len(matches) != 1:
            raise SystemPreconditionError("weekly rule applicability is ambiguous")
        is_open = (
            cursor.isoweekday() in matches[0]["weekdays"] and cursor.isoformat() not in closures
        )
        row: dict[str, Any] = {
            "date": cursor.isoformat(),
            "status": "OPEN" if is_open else "CLOSED",
        }
        if is_open:
            opens_local = datetime.combine(cursor, time(9, 30), tzinfo=zone)
            closes_local = datetime.combine(cursor, time(15, 0), tzinfo=zone)
            row["opens_at_utc"] = opens_local.astimezone(ZoneInfo("UTC")).isoformat()
            row["closes_at_utc"] = closes_local.astimezone(ZoneInfo("UTC")).isoformat()
            open_sessions.append(cursor.isoformat())
        else:
            row["opens_at_utc"] = None
            row["closes_at_utc"] = None
        daily.append(row)
        cursor += timedelta(days=1)
    if len(open_sessions) < 391:
        raise SystemPreconditionError("exchange calendar has fewer than 391 open sessions")
    return {
        "exchange_id": source["exchange_id"],
        "issuer": source["issuer"],
        "daily_rows": daily,
        "open_session_count": len(open_sessions),
        "open_session_sha256": _sha256(canonical_json_bytes(open_sessions)),
        "session_intervals": source["session_intervals"],
    }


def build_exchange_calendar_compilation(  # noqa: C901
    *,
    compilation_id: str,
    coverage_start_date: str,
    cutoff_date: str,
    release_ref: Mapping[str, Any],
    source_exchange_rows: Sequence[Mapping[str, Any]],
    calendar_json_file_ref: Mapping[str, Any],
    calendar_parquet_file_ref: Mapping[str, Any],
    contradiction_rows: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    raise SystemPreconditionError(
        "shallow calendar compilation is disabled; native capture replay is required"
    )


def _disabled_shallow_build_exchange_calendar_compilation(  # noqa: C901
    *,
    compilation_id: str,
    coverage_start_date: str,
    cutoff_date: str,
    release_ref: Mapping[str, Any],
    source_exchange_rows: Sequence[Mapping[str, Any]],
    calendar_json_file_ref: Mapping[str, Any],
    calendar_parquet_file_ref: Mapping[str, Any],
    contradiction_rows: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    coverage = _date(coverage_start_date, label="coverage_start_date")
    cutoff = _date(cutoff_date, label="cutoff_date")
    if coverage > date(2024, 1, 1) or coverage > cutoff:
        raise SystemPreconditionError("calendar compilation coverage is insufficient")
    normalized_sources = [
        _normalize_source_row(row, coverage=coverage, cutoff=cutoff) for row in source_exchange_rows
    ]
    exchanges = [row["exchange_id"] for row in normalized_sources]
    if exchanges != ["BSE", "SSE", "SZSE"]:
        raise SystemPreconditionError("calendar compiler exchange set is not BSE/SSE/SZSE")
    compiled = [_compile_row(row, coverage=coverage, cutoff=cutoff) for row in normalized_sources]
    runtime = compiled[0]["daily_rows"]
    if any(row["daily_rows"] != runtime for row in compiled[1:]):
        raise SystemPreconditionError("exchange-less runtime projections differ")
    contradictions = [dict(row) for row in contradiction_rows]
    if contradictions:
        raise SystemPreconditionError("calendar compilation contains contradictions")
    source_capture_refs = sorted(
        [ref for row in normalized_sources for ref in row["native_capture_refs"]],
        key=canonical_json_bytes,
    )
    decoder_admission_refs = sorted(
        [ref for row in normalized_sources for ref in row["decoder_admission_refs"]],
        key=canonical_json_bytes,
    )
    index_closure_refs = sorted(
        [ref for row in normalized_sources for ref in row["index_closure_refs"]],
        key=canonical_json_bytes,
    )
    for label, refs in (
        ("source_capture_refs", source_capture_refs),
        ("decoder_admission_refs", decoder_admission_refs),
        ("index_closure_refs", index_closure_refs),
    ):
        encoded = [canonical_json_bytes(ref) for ref in refs]
        if len(encoded) != len(set(encoded)):
            raise SystemContractError(f"{label} contains duplicate authority")
    normalized_release = validate_object_ref(release_ref, label="release_ref")
    if normalized_release["kind"] != "system.release":
        raise SystemContractError("calendar compilation release authority kind differs")
    return seal_artifact(
        COMPILATION_KIND,
        {
            "compilation_id": _identifier(compilation_id, label="compilation_id"),
            "state": "COMPILED",
            "coverage_start_date": coverage.isoformat(),
            "cutoff_date": cutoff.isoformat(),
            "timezone": TIMEZONE,
            "compiler_relative_path": COMPILER_RELATIVE_PATH,
            "compiler_code_sha256": compiler_code_sha256(),
            "compiler_ast_sha256": compiler_ast_sha256(),
            "release_ref": normalized_release,
            "source_exchange_rows": normalized_sources,
            "source_capture_refs": source_capture_refs,
            "decoder_admission_refs": decoder_admission_refs,
            "index_closure_refs": index_closure_refs,
            "precedence_rules": PRECEDENCE_RULES,
            "exchange_rows": compiled,
            "runtime_projection": runtime,
            "calendar_json_file_ref": _file_ref(
                calendar_json_file_ref, label="calendar_json_file_ref"
            ),
            "calendar_parquet_file_ref": _file_ref(
                calendar_parquet_file_ref, label="calendar_parquet_file_ref"
            ),
            "contradiction_rows": [],
        },
        created_at=created_at,
    )


def validate_exchange_calendar_compilation(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    raise SystemPreconditionError(
        "shallow calendar compilation replay is disabled; native capture replay is required"
    )


def _disabled_shallow_validate_exchange_calendar_compilation(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(document, expected_kind=COMPILATION_KIND)
    except ContractError as exc:
        raise SystemContractError("exchange calendar compilation contract failed") from exc
    payload = artifact["payload"]
    rebuilt = build_exchange_calendar_compilation(
        compilation_id=payload["compilation_id"],
        coverage_start_date=payload["coverage_start_date"],
        cutoff_date=payload["cutoff_date"],
        release_ref=payload["release_ref"],
        source_exchange_rows=payload["source_exchange_rows"],
        calendar_json_file_ref=payload["calendar_json_file_ref"],
        calendar_parquet_file_ref=payload["calendar_parquet_file_ref"],
        contradiction_rows=payload["contradiction_rows"],
        created_at=artifact["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("exchange calendar compilation replay differs")
    return artifact


__all__ = [
    "COMPILATION_KIND",
    "COMPILER_RELATIVE_PATH",
    "INDEX_CLOSURE_KIND",
    "PRECEDENCE_RULES",
    "build_exchange_calendar_compilation",
    "compiler_ast_sha256",
    "compiler_code_sha256",
    "validate_exchange_calendar_compilation",
    "validate_index_closure",
]
