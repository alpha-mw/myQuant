"""Pure SOURCE/CALENDAR/PIT contracts for FactorGovernance v4.1.

The module deliberately performs no I/O.  Historical design sessions are
evaluated against the complete valid PIT interval domain; the frozen cutoff
session and every later holdout session use that node's exact component/PIT
intersection.  Session artifacts store canonical set descriptors instead of
materialising multi-million-symbol cumulative mappings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

PIT_RECORD_SCHEMA_VERSION = "factor-governance-pit-record.v4.1"
SOURCE_PIT_SCHEMA_VERSION = "cn_pit_universe.v1"
SESSION_SCOPE_SCHEMA_VERSION = "factor-governance-session-scope.v4.1"
DESIGN_SOURCE_SCHEMA_VERSION = "factor-governance-design-source.v4.1"
HOLDOUT_SOURCE_SCHEMA_VERSION = "factor-governance-holdout-source-node.v4.1"

DESIGN_SNAPSHOT_ID = "20260717T172132Z"
DESIGN_CUTOFF_DATE = "2026-07-17"
DESIGN_COMPONENT_COUNT = 5502
DESIGN_EXPLORATORY = True

DESIGN_HISTORY_SCOPE = "design_history"
CUTOFF_SCOPE = "cutoff"
HOLDOUT_SCOPE = "holdout"
COMPONENT_ELIGIBILITY_SOURCE = "component_pit_intersection"
SERVING_INVENTORY_ELIGIBILITY_SOURCE = "serving_inventory"

EMBARGO_SESSION_COUNT = 30
MIN_POST_EMBARGO_SESSION_COUNT = 240
MIN_CLOSED_POST_EMBARGO_MONTH_COUNT = 12
GENESIS_SHA256 = "0" * 64

_PIT_SYMBOL_RE = re.compile(r"^[A-Z0-9]+\.(?:SH|SZ|BJ)$")
_COMPONENT_SYMBOL_RE = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class FactorGovernanceSourceV41Error(ValueError):
    """Raised when a v4.1 source/calendar/PIT contract fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceSourceV41Error(
            f"value is not canonical JSON: {exc}"
        ) from exc


def canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def byte_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceSourceV41Error(f"{label} must be an object")
    payload = dict(value)
    if set(payload) != fields:
        raise FactorGovernanceSourceV41Error(f"{label} fields invalid")
    return payload


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FactorGovernanceSourceV41Error(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _nonzero_sha(value: Any, label: str) -> str:
    observed = _sha(value, label)
    if observed == GENESIS_SHA256:
        raise FactorGovernanceSourceV41Error(f"{label} must not be the genesis SHA")
    return observed


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FactorGovernanceSourceV41Error(f"{label} must be a positive integer")
    return value


def _safe_id(value: Any, label: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise FactorGovernanceSourceV41Error(f"{label} is not a safe identifier")
    return value


def _observed_at(value: Any) -> str:
    if type(value) is not str:
        raise FactorGovernanceSourceV41Error("observed_at must be an exact string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise FactorGovernanceSourceV41Error(
            "observed_at must be an exact UTC second timestamp"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise FactorGovernanceSourceV41Error(
            "observed_at must be an exact UTC second timestamp"
        )
    return value


def _pit_date(value: Any, label: str, *, allow_blank: bool = False) -> str | None:
    if value is None and allow_blank:
        return None
    if type(value) is not str:
        raise FactorGovernanceSourceV41Error(f"{label} is invalid")
    if value != value.strip():
        raise FactorGovernanceSourceV41Error(f"{label} is not exact")
    text = value
    if not text and allow_blank:
        return None
    if not text:
        raise FactorGovernanceSourceV41Error(f"{label} is missing")
    try:
        parsed = (
            datetime.strptime(text, "%Y%m%d").date()
            if len(text) == 8 and text.isdigit()
            else date.fromisoformat(text)
        )
    except ValueError as exc:
        raise FactorGovernanceSourceV41Error(f"{label} is invalid") from exc
    return parsed.isoformat()


def _session(value: Any, label: str = "session") -> str:
    if type(value) is not str:
        raise FactorGovernanceSourceV41Error(f"{label} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise FactorGovernanceSourceV41Error(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value or parsed.weekday() >= 5:
        raise FactorGovernanceSourceV41Error(
            f"{label} must be a canonical ISO weekday"
        )
    return value


def _calendar(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise FactorGovernanceSourceV41Error(f"{label} must be a non-empty list")
    sessions = [_session(item, f"{label}[]") for item in value]
    if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise FactorGovernanceSourceV41Error(
            f"{label} must be sorted and distinct"
        )
    return sessions


def _pit_symbol(value: Any, label: str) -> str:
    if type(value) is not str or _PIT_SYMBOL_RE.fullmatch(value) is None:
        raise FactorGovernanceSourceV41Error(f"{label} is invalid")
    return value


def _component_symbol(value: Any, label: str) -> str:
    if type(value) is not str or _COMPONENT_SYMBOL_RE.fullmatch(value) is None:
        raise FactorGovernanceSourceV41Error(f"{label} is invalid")
    return value


def _components(value: Any, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise FactorGovernanceSourceV41Error("component_symbols must be a list")
    symbols: list[str] = []
    for raw in value:
        symbols.append(_component_symbol(raw, "component symbol"))
    if not symbols and not allow_empty:
        raise FactorGovernanceSourceV41Error("component_symbols must not be empty")
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise FactorGovernanceSourceV41Error(
            "component_symbols must be sorted and distinct"
        )
    return symbols


def validate_pit_records_v4_1(rows: Any) -> list[dict[str, Any]]:
    """Normalize one authoritative fail-closed PIT row per A-share symbol."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, Mapping)):
        raise FactorGovernanceSourceV41Error("PIT records must be a sequence")
    if not rows:
        raise FactorGovernanceSourceV41Error("PIT records must not be empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise FactorGovernanceSourceV41Error(f"PIT row {index} must be an object")
        raw_schema = raw.get("schema_version")
        if raw_schema == PIT_RECORD_SCHEMA_VERSION:
            source_schema = raw.get("source_schema_version")
        else:
            source_schema = raw_schema
        if source_schema != SOURCE_PIT_SCHEMA_VERSION:
            raise FactorGovernanceSourceV41Error(
                f"unsupported PIT source schema: row {index}"
            )
        symbol = _pit_symbol(raw.get("symbol"), f"PIT row {index} symbol")
        if symbol in seen:
            raise FactorGovernanceSourceV41Error(f"duplicate PIT symbol: {symbol}")
        seen.add(symbol)

        source_status = raw.get("source_list_status")
        alias_status = raw.get("status")
        if source_status is None:
            source_status = alias_status
        elif alias_status is not None and alias_status != source_status:
            raise FactorGovernanceSourceV41Error(
                f"conflicting PIT status fields: {symbol}"
            )
        if type(source_status) is not str or source_status not in {"L", "D", "P"}:
            raise FactorGovernanceSourceV41Error(
                f"unsupported PIT status: {symbol}"
            )
        if raw.get("membership_quality") != "ok":
            raise FactorGovernanceSourceV41Error(
                f"PIT membership_quality is not ok: {symbol}"
            )
        effective_from = _pit_date(
            raw.get("effective_from"), f"PIT effective_from: {symbol}"
        )
        if effective_from is None:  # Defensive narrowing for the required date.
            raise FactorGovernanceSourceV41Error(
                f"PIT effective_from is missing: {symbol}"
            )
        effective_to = _pit_date(
            raw.get("effective_to"),
            f"PIT effective_to: {symbol}",
            allow_blank=True,
        )
        if source_status == "D" and effective_to is None:
            raise FactorGovernanceSourceV41Error(
                f"delisted PIT row has no effective_to: {symbol}"
            )
        if effective_to is not None and effective_from >= effective_to:
            raise FactorGovernanceSourceV41Error(
                f"PIT interval order is invalid: {symbol}"
            )
        if source_status in {"L", "P"} and effective_to is not None:
            raise FactorGovernanceSourceV41Error(
                f"active/pending PIT row has contradictory effective_to: {symbol}"
            )
        if "list_date" in raw:
            list_date = _pit_date(raw.get("list_date"), f"PIT list_date: {symbol}")
            if list_date != effective_from:
                raise FactorGovernanceSourceV41Error(
                    f"PIT list/effective_from conflict: {symbol}"
                )
        if "delist_date" in raw:
            delist_date = _pit_date(
                raw.get("delist_date"),
                f"PIT delist_date: {symbol}",
                allow_blank=True,
            )
            if delist_date != effective_to:
                raise FactorGovernanceSourceV41Error(
                    f"PIT delist/effective_to conflict: {symbol}"
                )
        normalized.append(
            {
                "schema_version": PIT_RECORD_SCHEMA_VERSION,
                "source_schema_version": SOURCE_PIT_SCHEMA_VERSION,
                "symbol": symbol,
                "source_list_status": source_status,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "membership_quality": "ok",
            }
        )
    return sorted(normalized, key=lambda item: item["symbol"])


def _record_domain(records: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized = validate_pit_records_v4_1(records)
    return normalized, {row["symbol"]: row for row in normalized}


def _active(record: Mapping[str, Any], session: str) -> bool:
    return bool(
        record["effective_from"] <= session
        and (record["effective_to"] is None or session < record["effective_to"])
    )


def _newline_set_sha256(symbols: Sequence[str]) -> str:
    raw = "\n".join(sorted(symbols)).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _historical_alias_report(
    normalized_records: Sequence[Mapping[str, Any]], sessions: Sequence[str]
) -> dict[str, Any]:
    records = [
        {
            "symbol": row["symbol"],
            "source_list_status": row["source_list_status"],
            "effective_from": row["effective_from"],
            "effective_to": row["effective_to"],
            "active_bound_session_count": sum(
                1 for session in sessions if _active(row, session)
            ),
        }
        for row in normalized_records
        if _COMPONENT_SYMBOL_RE.fullmatch(row["symbol"]) is None
    ]
    overlapping = [row["symbol"] for row in records if row["active_bound_session_count"]]
    if overlapping:
        raise FactorGovernanceSourceV41Error(
            "historical PIT alias overlaps bound calendar: " + ",".join(overlapping)
        )
    return {
        "records": records,
        "count": len(records),
        "records_semantic_sha256": semantic_sha256(records),
    }


def _descriptor_from_normalized(
    normalized: Sequence[Mapping[str, Any]],
    by_symbol: Mapping[str, Mapping[str, Any]],
    session: str,
    scope_kind: str,
    component_symbols: list[str] | None,
) -> dict[str, Any]:
    current_session = _session(session)
    _historical_alias_report(normalized, [current_session])
    if scope_kind == DESIGN_HISTORY_SCOPE:
        if component_symbols is not None:
            raise FactorGovernanceSourceV41Error(
                "design_history must use the full PIT interval domain"
            )
        domain = [row["symbol"] for row in normalized]
        component_sha: str | None = None
        eligibility_source = "pit_interval_domain"
    elif scope_kind in {CUTOFF_SCOPE, HOLDOUT_SCOPE}:
        components = _components(component_symbols)
        missing = sorted(set(components) - set(by_symbol))
        if missing:
            raise FactorGovernanceSourceV41Error(
                "component symbol missing from PIT domain: " + ",".join(missing)
            )
        domain = components
        component_sha = _newline_set_sha256(components)
        eligibility_source = COMPONENT_ELIGIBILITY_SOURCE
    else:
        raise FactorGovernanceSourceV41Error("unsupported session scope_kind")

    research: list[str] = []
    tradable: list[str] = []
    for symbol in domain:
        record = by_symbol[symbol]
        if not _active(record, current_session):
            continue
        research.append(symbol)
        if record["source_list_status"] in {"L", "D"}:
            tradable.append(symbol)
    base = {
        "schema_version": SESSION_SCOPE_SCHEMA_VERSION,
        "session": current_session,
        "scope_kind": scope_kind,
        "eligibility_source": eligibility_source,
        "component_symbols_semantic_sha256": component_sha,
        "research_eligible_count": len(research),
        "research_eligible_symbols_newline_sha256": _newline_set_sha256(research),
        "tradable_count": len(tradable),
        "tradable_symbols_newline_sha256": _newline_set_sha256(tradable),
    }
    return {**base, "session_semantic_sha256": semantic_sha256(base)}


def build_session_scope_descriptor_v4_1(
    records: Any,
    session: str,
    scope_kind: str,
    component_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Build one independently recomputable PIT eligibility descriptor."""

    normalized, by_symbol = _record_domain(records)
    return _descriptor_from_normalized(
        normalized, by_symbol, session, scope_kind, component_symbols
    )


def validate_calendar_prefix_v4_1(
    previous_calendar: list[str],
    cumulative_calendar: list[str],
    appended_sessions: list[str],
) -> list[str]:
    """Require an exact, non-empty, strictly later calendar prefix extension."""

    previous = _calendar(previous_calendar, "previous_calendar")
    cumulative = _calendar(cumulative_calendar, "cumulative_calendar")
    appended = _calendar(appended_sessions, "appended_sessions")
    if cumulative != previous + appended:
        raise FactorGovernanceSourceV41Error(
            "calendar is not an exact prefix extension"
        )
    if appended[0] <= previous[-1]:
        raise FactorGovernanceSourceV41Error(
            "appended sessions must be strictly later"
        )
    return cumulative


def _actual_month_ends(value: Any, cumulative: Sequence[str]) -> list[str]:
    if not isinstance(value, list):
        raise FactorGovernanceSourceV41Error(
            "actual_month_end_sessions must be a list"
        )
    sessions = [_session(item, "actual_month_end_sessions[]") for item in value]
    if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise FactorGovernanceSourceV41Error(
            "actual_month_end_sessions must be sorted and distinct"
        )
    if len({session[:7] for session in sessions}) != len(sessions):
        raise FactorGovernanceSourceV41Error(
            "actual_month_end_sessions must contain at most one session per month"
        )
    if not set(sessions).issubset(cumulative):
        raise FactorGovernanceSourceV41Error(
            "actual_month_end_sessions must belong to the cumulative calendar"
        )
    for session in sessions:
        same_month = [item for item in cumulative if item[:7] == session[:7]]
        if not same_month or session != max(same_month):
            raise FactorGovernanceSourceV41Error(
                "actual month-end must be the last supplied session in its month"
            )
    return sessions


def _closed_month_end_dates(
    post_embargo: Sequence[str],
    cumulative: Sequence[str],
    actual_month_ends: Sequence[str],
) -> list[str]:
    if not post_embargo:
        return []
    latest_month = cumulative[-1][:7]
    post_domain = set(post_embargo)
    return [
        session
        for session in actual_month_ends
        if session in post_domain and session[:7] < latest_month
    ]


def _validate_month_end_extension(
    previous: Sequence[str], current: Sequence[str], latest_session: str
) -> None:
    prior = list(previous)
    observed = list(current)
    if observed[: len(prior)] != prior or len(observed) - len(prior) not in {0, 1}:
        raise FactorGovernanceSourceV41Error(
            "actual month-end evidence is not an immutable one-step extension"
        )
    if len(observed) > len(prior) and observed[-1][:7] >= latest_session[:7]:
        raise FactorGovernanceSourceV41Error(
            "actual month-end is not closed by a later-month session"
        )


def assess_holdout_calendar_readiness_v4_1(
    design_calendar: list[str],
    cumulative_calendar: list[str],
    actual_month_end_sessions: list[str],
) -> dict[str, Any]:
    design = _calendar(design_calendar, "design_calendar")
    cumulative = _calendar(cumulative_calendar, "cumulative_calendar")
    if cumulative[: len(design)] != design or len(cumulative) < len(design):
        raise FactorGovernanceSourceV41Error("design calendar prefix changed")
    appended = cumulative[len(design) :]
    if appended and appended[0] <= design[-1]:
        raise FactorGovernanceSourceV41Error(
            "holdout sessions must be strictly after design cutoff"
        )
    embargo_count = min(len(appended), EMBARGO_SESSION_COUNT)
    post_embargo = appended[EMBARGO_SESSION_COUNT:]
    actual_month_ends = _actual_month_ends(actual_month_end_sessions, cumulative)
    closed_dates = _closed_month_end_dates(
        post_embargo, cumulative, actual_month_ends
    )
    ready = bool(
        len(post_embargo) >= MIN_POST_EMBARGO_SESSION_COUNT
        and len(closed_dates) >= MIN_CLOSED_POST_EMBARGO_MONTH_COUNT
    )
    return {
        "embargo_session_count": embargo_count,
        "post_embargo_session_count": len(post_embargo),
        "closed_post_embargo_month_end_dates": closed_dates,
        "closed_post_embargo_month_end_count": len(closed_dates),
        "ready": ready,
    }


def _artifact_sha(payload: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )


def _holdout_sha(payload: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"semantic_sha256", "terminal_holdout_source_root_sha256"}
        }
    )


def build_design_source_node_v4_1(
    *,
    cycle_id: str,
    pit_records: Any,
    component_symbols: list[str],
    calendar_sessions: list[str],
    market_binding_sha256: str,
    source_binding_sha256: str,
    expected_component_count: int = DESIGN_COMPONENT_COUNT,
) -> dict[str, Any]:
    """Build the immutable v4.1 design source root for the named cutoff."""

    cycle = _safe_id(cycle_id, "cycle_id")
    expected_count = _positive_int(
        expected_component_count, "expected_component_count"
    )
    components = _components(component_symbols)
    if len(components) != expected_count:
        raise FactorGovernanceSourceV41Error(
            "design component count mismatch"
        )
    sessions = _calendar(calendar_sessions, "calendar_sessions")
    if sessions[-1] != DESIGN_CUTOFF_DATE:
        raise FactorGovernanceSourceV41Error(
            "design calendar must end at the frozen cutoff"
        )
    normalized_records = validate_pit_records_v4_1(pit_records)
    alias_report = _historical_alias_report(normalized_records, sessions)
    by_symbol = {row["symbol"]: row for row in normalized_records}
    descriptors = [
        _descriptor_from_normalized(
            normalized_records,
            by_symbol,
            session,
            CUTOFF_SCOPE if session == DESIGN_CUTOFF_DATE else DESIGN_HISTORY_SCOPE,
            components if session == DESIGN_CUTOFF_DATE else None,
        )
        for session in sessions
    ]
    base = {
        "schema_version": DESIGN_SOURCE_SCHEMA_VERSION,
        "cycle_id": cycle,
        "snapshot_id": DESIGN_SNAPSHOT_ID,
        "cutoff_date": DESIGN_CUTOFF_DATE,
        "component_symbols": components,
        "component_count": len(components),
        "component_symbols_semantic_sha256": _newline_set_sha256(components),
        "pit_record_count": len(normalized_records),
        "pit_records_semantic_sha256": semantic_sha256(normalized_records),
        "out_of_bound_calendar_nonparticipating": alias_report,
        "calendar_sessions": sessions,
        "calendar_semantic_sha256": semantic_sha256(sessions),
        "session_scope_descriptors": descriptors,
        "session_scope_mapping_semantic_sha256": semantic_sha256(descriptors),
        "historical_table_binding_sha256": _nonzero_sha(
            market_binding_sha256, "market_binding_sha256"
        ),
        "historical_source_binding_sha256": _nonzero_sha(
            source_binding_sha256, "source_binding_sha256"
        ),
        "exploratory": DESIGN_EXPLORATORY,
    }
    return {**base, "semantic_sha256": semantic_sha256(base)}


_DESIGN_FIELDS = {
    "schema_version",
    "cycle_id",
    "snapshot_id",
    "cutoff_date",
    "component_symbols",
    "component_count",
    "component_symbols_semantic_sha256",
    "pit_record_count",
    "pit_records_semantic_sha256",
    "out_of_bound_calendar_nonparticipating",
    "calendar_sessions",
    "calendar_semantic_sha256",
    "session_scope_descriptors",
    "session_scope_mapping_semantic_sha256",
    "historical_table_binding_sha256",
    "historical_source_binding_sha256",
    "exploratory",
    "semantic_sha256",
}


def validate_design_source_node_v4_1(
    node: Any,
    *,
    pit_records: Any,
    expected_component_count: int = DESIGN_COMPONENT_COUNT,
) -> dict[str, Any]:
    payload = _exact(node, _DESIGN_FIELDS, "design source node")
    if payload["schema_version"] != DESIGN_SOURCE_SCHEMA_VERSION:
        raise FactorGovernanceSourceV41Error("design source schema mismatch")
    expected = build_design_source_node_v4_1(
        cycle_id=payload["cycle_id"],
        pit_records=pit_records,
        component_symbols=payload["component_symbols"],
        calendar_sessions=payload["calendar_sessions"],
        market_binding_sha256=payload["historical_table_binding_sha256"],
        source_binding_sha256=payload["historical_source_binding_sha256"],
        expected_component_count=expected_component_count,
    )
    if canonical_json_bytes(payload) != canonical_json_bytes(expected):
        raise FactorGovernanceSourceV41Error("design source node mismatch")
    return copy.deepcopy(expected)


_HOLDOUT_FIELDS = {
    "schema_version",
    "cycle_id",
    "design_source_root_sha256",
    "predecessor",
    "node_id",
    "observed_at",
    "component_symbols",
    "component_count",
    "component_symbols_semantic_sha256",
    "pit_record_count",
    "pit_records_semantic_sha256",
    "out_of_bound_calendar_nonparticipating",
    "serving_inventory_count",
    "eligibility_source",
    "serving_inventory_eligibility_prohibited",
    "appended_sessions",
    "cumulative_calendar_sessions",
    "calendar_semantic_sha256",
    "actual_month_end_sessions",
    "actual_month_end_sessions_semantic_sha256",
    "session_scope_descriptors",
    "session_scope_mapping_semantic_sha256",
    "historical_table_binding_sha256",
    "historical_source_binding_sha256",
    "appended_table_binding_sha256",
    "appended_source_binding_sha256",
    "embargo_session_count",
    "post_embargo_session_count",
    "closed_post_embargo_month_end_dates",
    "closed_post_embargo_month_end_count",
    "ready",
    "terminal_holdout_source_root_sha256",
    "semantic_sha256",
}


def _validate_design_shape(
    node: Any, *, expected_component_count: int
) -> dict[str, Any]:
    payload = _exact(node, _DESIGN_FIELDS, "design source node")
    if (
        payload["schema_version"] != DESIGN_SOURCE_SCHEMA_VERSION
        or payload["snapshot_id"] != DESIGN_SNAPSHOT_ID
        or payload["cutoff_date"] != DESIGN_CUTOFF_DATE
        or payload["exploratory"] is not True
    ):
        raise FactorGovernanceSourceV41Error("design source constants mismatch")
    _safe_id(payload["cycle_id"], "cycle_id")
    components = _components(payload["component_symbols"])
    if len(components) != _positive_int(
        expected_component_count, "expected_component_count"
    ) or payload["component_count"] != len(components):
        raise FactorGovernanceSourceV41Error("design component count mismatch")
    if payload["component_symbols_semantic_sha256"] != _newline_set_sha256(components):
        raise FactorGovernanceSourceV41Error("design component SHA mismatch")
    _positive_int(payload["pit_record_count"], "pit_record_count")
    _sha(payload["pit_records_semantic_sha256"], "pit_records_semantic_sha256")
    aliases = _exact(
        payload["out_of_bound_calendar_nonparticipating"],
        {"records", "count", "records_semantic_sha256"},
        "out_of_bound_calendar_nonparticipating",
    )
    if not isinstance(aliases["records"], list) or [
        row.get("symbol") for row in aliases["records"]
    ] != sorted(row.get("symbol") for row in aliases["records"]):
        raise FactorGovernanceSourceV41Error("historical alias report invalid")
    if aliases["count"] != len(aliases["records"]) or aliases[
        "records_semantic_sha256"
    ] != semantic_sha256(aliases["records"]):
        raise FactorGovernanceSourceV41Error("historical alias report mismatch")
    sessions = _calendar(payload["calendar_sessions"], "calendar_sessions")
    if sessions[-1] != DESIGN_CUTOFF_DATE:
        raise FactorGovernanceSourceV41Error("design cutoff mismatch")
    if payload["calendar_semantic_sha256"] != semantic_sha256(sessions):
        raise FactorGovernanceSourceV41Error("design calendar SHA mismatch")
    descriptors = payload["session_scope_descriptors"]
    if not isinstance(descriptors, list) or [row.get("session") for row in descriptors] != sessions:
        raise FactorGovernanceSourceV41Error("design session descriptor domain mismatch")
    if payload["session_scope_mapping_semantic_sha256"] != semantic_sha256(descriptors):
        raise FactorGovernanceSourceV41Error("design mapping SHA mismatch")
    _nonzero_sha(
        payload["historical_table_binding_sha256"], "historical table binding"
    )
    _nonzero_sha(
        payload["historical_source_binding_sha256"], "historical source binding"
    )
    supplied = _sha(payload["semantic_sha256"], "design semantic_sha256")
    if supplied != _artifact_sha(payload):
        raise FactorGovernanceSourceV41Error("design semantic SHA mismatch")
    return payload


def _predecessor(
    previous_node: Mapping[str, Any] | None,
    *,
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
) -> dict[str, str]:
    observed_byte = _sha(predecessor_byte_sha256, "predecessor_byte_sha256")
    expected_byte = _sha(
        expected_predecessor_byte_sha256, "expected_predecessor_byte_sha256"
    )
    expected_semantic = _sha(
        expected_predecessor_semantic_sha256,
        "expected_predecessor_semantic_sha256",
    )
    if previous_node is None:
        actual = {
            "kind": "genesis",
            "byte_sha256": GENESIS_SHA256,
            "semantic_sha256": GENESIS_SHA256,
        }
    else:
        actual = {
            "kind": "node",
            "byte_sha256": byte_sha256(previous_node),
            "semantic_sha256": _sha(
                previous_node.get("semantic_sha256"), "previous semantic_sha256"
            ),
        }
    if (
        observed_byte != actual["byte_sha256"]
        or expected_byte != actual["byte_sha256"]
        or expected_semantic != actual["semantic_sha256"]
    ):
        raise FactorGovernanceSourceV41Error("predecessor dual CAS mismatch")
    return actual


def append_holdout_source_node_v4_1(
    *,
    design_node: Mapping[str, Any],
    previous_node: Mapping[str, Any] | None,
    design_pit_records: Any,
    node_pit_records: Any,
    actual_month_end_sessions: list[str],
    component_symbols: list[str],
    appended_sessions: list[str],
    node_id: str,
    observed_at: str,
    serving_inventory_count: int,
    market_binding_sha256: str,
    source_binding_sha256: str,
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    expected_design_component_count: int = DESIGN_COMPONENT_COUNT,
) -> dict[str, Any]:
    """Append one immutable holdout node after exact predecessor CAS checks."""

    design = validate_design_source_node_v4_1(
        design_node,
        pit_records=design_pit_records,
        expected_component_count=expected_design_component_count,
    )
    if previous_node is None:
        prior_calendar = list(design["calendar_sessions"])
        prior_descriptors = list(design["session_scope_descriptors"])
        prior_month_ends: list[str] = []
    else:
        previous = _validate_holdout_shape(previous_node, design=design)
        if previous["ready"]:
            raise FactorGovernanceSourceV41Error("terminal holdout node cannot append")
        prior_calendar = list(previous["cumulative_calendar_sessions"])
        prior_descriptors = list(previous["session_scope_descriptors"])
        prior_month_ends = list(previous["actual_month_end_sessions"])
    appended = _calendar(appended_sessions, "appended_sessions")
    if len(appended) != 1:
        raise FactorGovernanceSourceV41Error(
            "each holdout source node must append exactly one session"
        )
    cumulative = validate_calendar_prefix_v4_1(
        prior_calendar, prior_calendar + appended, appended
    )
    actual_month_ends = _actual_month_ends(actual_month_end_sessions, cumulative)
    _validate_month_end_extension(
        prior_month_ends, actual_month_ends, cumulative[-1]
    )
    components = _components(component_symbols)
    normalized_node_records, node_by_symbol = _record_domain(node_pit_records)
    alias_report = _historical_alias_report(normalized_node_records, appended)
    descriptors = prior_descriptors + [
        _descriptor_from_normalized(
            normalized_node_records,
            node_by_symbol,
            session,
            HOLDOUT_SCOPE,
            components,
        )
        for session in appended
    ]
    readiness = assess_holdout_calendar_readiness_v4_1(
        design["calendar_sessions"], cumulative, actual_month_ends
    )
    base: dict[str, Any] = {
        "schema_version": HOLDOUT_SOURCE_SCHEMA_VERSION,
        "cycle_id": design["cycle_id"],
        "design_source_root_sha256": design["semantic_sha256"],
        "predecessor": _predecessor(
            previous_node,
            predecessor_byte_sha256=predecessor_byte_sha256,
            expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
            expected_predecessor_semantic_sha256=expected_predecessor_semantic_sha256,
        ),
        "node_id": _safe_id(node_id, "node_id"),
        "observed_at": _observed_at(observed_at),
        "component_symbols": components,
        "component_count": len(components),
        "component_symbols_semantic_sha256": _newline_set_sha256(components),
        "pit_record_count": len(normalized_node_records),
        "pit_records_semantic_sha256": semantic_sha256(normalized_node_records),
        "out_of_bound_calendar_nonparticipating": alias_report,
        "serving_inventory_count": _positive_int(
            serving_inventory_count, "serving_inventory_count"
        ),
        "eligibility_source": COMPONENT_ELIGIBILITY_SOURCE,
        "serving_inventory_eligibility_prohibited": True,
        "appended_sessions": appended,
        "cumulative_calendar_sessions": cumulative,
        "calendar_semantic_sha256": semantic_sha256(cumulative),
        "actual_month_end_sessions": actual_month_ends,
        "actual_month_end_sessions_semantic_sha256": semantic_sha256(
            actual_month_ends
        ),
        "session_scope_descriptors": descriptors,
        "session_scope_mapping_semantic_sha256": semantic_sha256(descriptors),
        "historical_table_binding_sha256": design[
            "historical_table_binding_sha256"
        ],
        "historical_source_binding_sha256": design[
            "historical_source_binding_sha256"
        ],
        "appended_table_binding_sha256": _nonzero_sha(
            market_binding_sha256, "market_binding_sha256"
        ),
        "appended_source_binding_sha256": _nonzero_sha(
            source_binding_sha256, "source_binding_sha256"
        ),
        **readiness,
        "terminal_holdout_source_root_sha256": None,
    }
    node_sha = _holdout_sha(base)
    base["semantic_sha256"] = node_sha
    if readiness["ready"]:
        base["terminal_holdout_source_root_sha256"] = node_sha
    return base


def _validate_holdout_shape(
    node: Any, *, design: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _exact(node, _HOLDOUT_FIELDS, "holdout source node")
    if payload["schema_version"] != HOLDOUT_SOURCE_SCHEMA_VERSION:
        raise FactorGovernanceSourceV41Error("holdout source schema mismatch")
    if payload["cycle_id"] != design["cycle_id"]:
        raise FactorGovernanceSourceV41Error("holdout cycle mismatch")
    if payload["design_source_root_sha256"] != design["semantic_sha256"]:
        raise FactorGovernanceSourceV41Error("holdout design root mismatch")
    _safe_id(payload["node_id"], "node_id")
    _observed_at(payload["observed_at"])
    components = _components(payload["component_symbols"])
    if payload["component_count"] != len(components):
        raise FactorGovernanceSourceV41Error("holdout component count mismatch")
    if payload["component_symbols_semantic_sha256"] != _newline_set_sha256(components):
        raise FactorGovernanceSourceV41Error("holdout component SHA mismatch")
    _positive_int(payload["pit_record_count"], "pit_record_count")
    _sha(payload["pit_records_semantic_sha256"], "pit_records_semantic_sha256")
    aliases = _exact(
        payload["out_of_bound_calendar_nonparticipating"],
        {"records", "count", "records_semantic_sha256"},
        "out_of_bound_calendar_nonparticipating",
    )
    if (
        not isinstance(aliases["records"], list)
        or aliases["count"] != len(aliases["records"])
        or aliases["records_semantic_sha256"]
        != semantic_sha256(aliases["records"])
    ):
        raise FactorGovernanceSourceV41Error("historical alias report mismatch")
    _positive_int(payload["serving_inventory_count"], "serving_inventory_count")
    if payload["eligibility_source"] != COMPONENT_ELIGIBILITY_SOURCE:
        raise FactorGovernanceSourceV41Error("holdout eligibility source invalid")
    if payload["serving_inventory_eligibility_prohibited"] is not True:
        raise FactorGovernanceSourceV41Error(
            "serving inventory eligibility must be prohibited"
        )
    if (
        payload["historical_table_binding_sha256"]
        != design["historical_table_binding_sha256"]
        or payload["historical_source_binding_sha256"]
        != design["historical_source_binding_sha256"]
    ):
        raise FactorGovernanceSourceV41Error("historical source binding drift")
    cumulative = _calendar(
        payload["cumulative_calendar_sessions"], "cumulative_calendar_sessions"
    )
    appended = _calendar(payload["appended_sessions"], "appended_sessions")
    if len(appended) != 1:
        raise FactorGovernanceSourceV41Error(
            "each holdout source node must append exactly one session"
        )
    if cumulative[-len(appended) :] != appended:
        raise FactorGovernanceSourceV41Error("holdout appended suffix mismatch")
    if cumulative[: len(design["calendar_sessions"])] != design["calendar_sessions"]:
        raise FactorGovernanceSourceV41Error("holdout design calendar prefix changed")
    if payload["calendar_semantic_sha256"] != semantic_sha256(cumulative):
        raise FactorGovernanceSourceV41Error("holdout calendar SHA mismatch")
    actual_month_ends = _actual_month_ends(
        payload["actual_month_end_sessions"], cumulative
    )
    if payload["actual_month_end_sessions_semantic_sha256"] != semantic_sha256(
        actual_month_ends
    ):
        raise FactorGovernanceSourceV41Error("actual month-end SHA mismatch")
    descriptors = payload["session_scope_descriptors"]
    if not isinstance(descriptors, list) or [row.get("session") for row in descriptors] != cumulative:
        raise FactorGovernanceSourceV41Error("holdout descriptor domain mismatch")
    if payload["session_scope_mapping_semantic_sha256"] != semantic_sha256(descriptors):
        raise FactorGovernanceSourceV41Error("holdout mapping SHA mismatch")
    readiness = assess_holdout_calendar_readiness_v4_1(
        design["calendar_sessions"], cumulative, actual_month_ends
    )
    for key, value in readiness.items():
        if payload[key] != value:
            raise FactorGovernanceSourceV41Error("holdout readiness mismatch")
    predecessor = _exact(
        payload["predecessor"],
        {"kind", "byte_sha256", "semantic_sha256"},
        "holdout predecessor",
    )
    if predecessor["kind"] not in {"genesis", "node"}:
        raise FactorGovernanceSourceV41Error("holdout predecessor kind invalid")
    _sha(predecessor["byte_sha256"], "predecessor byte SHA")
    _sha(predecessor["semantic_sha256"], "predecessor semantic SHA")
    _nonzero_sha(
        payload["historical_table_binding_sha256"], "historical table binding"
    )
    _nonzero_sha(
        payload["historical_source_binding_sha256"], "historical source binding"
    )
    _nonzero_sha(payload["appended_table_binding_sha256"], "appended table binding")
    _nonzero_sha(
        payload["appended_source_binding_sha256"], "appended source binding"
    )
    supplied = _sha(payload["semantic_sha256"], "holdout semantic_sha256")
    if supplied != _holdout_sha(payload):
        raise FactorGovernanceSourceV41Error("holdout semantic SHA mismatch")
    terminal = payload["terminal_holdout_source_root_sha256"]
    if payload["ready"]:
        if terminal != supplied:
            raise FactorGovernanceSourceV41Error("terminal holdout root mismatch")
    elif terminal is not None:
        raise FactorGovernanceSourceV41Error("unready node cannot be terminal")
    return payload


def validate_holdout_source_node_v4_1(
    node: Any,
    *,
    design_node: Mapping[str, Any],
    previous_node: Mapping[str, Any] | None,
    design_pit_records: Any,
    node_pit_records: Any,
    actual_month_end_sessions: list[str],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    expected_design_component_count: int = DESIGN_COMPONENT_COUNT,
) -> dict[str, Any]:
    """Validate lineage, prefix immutability, and appended PIT descriptors."""

    design = validate_design_source_node_v4_1(
        design_node,
        pit_records=design_pit_records,
        expected_component_count=expected_design_component_count,
    )
    payload = _validate_holdout_shape(node, design=design)
    expected_month_ends = _actual_month_ends(
        actual_month_end_sessions, payload["cumulative_calendar_sessions"]
    )
    if payload["actual_month_end_sessions"] != expected_month_ends:
        raise FactorGovernanceSourceV41Error("actual month-end source mismatch")
    expected_predecessor = _predecessor(
        previous_node,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=expected_predecessor_semantic_sha256,
    )
    if payload["predecessor"] != expected_predecessor:
        raise FactorGovernanceSourceV41Error("stale predecessor hashes")
    if previous_node is None:
        prior_calendar = design["calendar_sessions"]
        prior_descriptors = design["session_scope_descriptors"]
        prior_month_ends: list[str] = []
    else:
        previous = _validate_holdout_shape(previous_node, design=design)
        if previous["ready"]:
            raise FactorGovernanceSourceV41Error("terminal predecessor cannot append")
        if payload["node_id"] == previous["node_id"]:
            raise FactorGovernanceSourceV41Error("holdout node_id must advance")
        prior_calendar = previous["cumulative_calendar_sessions"]
        prior_descriptors = previous["session_scope_descriptors"]
        prior_month_ends = previous["actual_month_end_sessions"]
    _validate_month_end_extension(
        prior_month_ends,
        payload["actual_month_end_sessions"],
        payload["cumulative_calendar_sessions"][-1],
    )
    validate_calendar_prefix_v4_1(
        list(prior_calendar),
        payload["cumulative_calendar_sessions"],
        payload["appended_sessions"],
    )
    prefix_length = len(prior_descriptors)
    if payload["session_scope_descriptors"][:prefix_length] != prior_descriptors:
        raise FactorGovernanceSourceV41Error("historical PIT mapping prefix changed")
    normalized_node_records, node_by_symbol = _record_domain(node_pit_records)
    if payload["pit_record_count"] != len(normalized_node_records) or payload[
        "pit_records_semantic_sha256"
    ] != semantic_sha256(normalized_node_records):
        raise FactorGovernanceSourceV41Error("holdout PIT inventory mismatch")
    expected_alias_report = _historical_alias_report(
        normalized_node_records, payload["appended_sessions"]
    )
    if payload["out_of_bound_calendar_nonparticipating"] != expected_alias_report:
        raise FactorGovernanceSourceV41Error(
            "holdout historical alias report mismatch"
        )
    expected_suffix = [
        _descriptor_from_normalized(
            normalized_node_records,
            node_by_symbol,
            session,
            HOLDOUT_SCOPE,
            payload["component_symbols"],
        )
        for session in payload["appended_sessions"]
    ]
    if payload["session_scope_descriptors"][prefix_length:] != expected_suffix:
        raise FactorGovernanceSourceV41Error("appended PIT mapping mismatch")
    return copy.deepcopy(payload)


__all__ = [
    "COMPONENT_ELIGIBILITY_SOURCE",
    "CUTOFF_SCOPE",
    "DESIGN_COMPONENT_COUNT",
    "DESIGN_CUTOFF_DATE",
    "DESIGN_HISTORY_SCOPE",
    "DESIGN_SNAPSHOT_ID",
    "EMBARGO_SESSION_COUNT",
    "FactorGovernanceSourceV41Error",
    "HOLDOUT_SCOPE",
    "MIN_CLOSED_POST_EMBARGO_MONTH_COUNT",
    "MIN_POST_EMBARGO_SESSION_COUNT",
    "append_holdout_source_node_v4_1",
    "assess_holdout_calendar_readiness_v4_1",
    "build_design_source_node_v4_1",
    "build_session_scope_descriptor_v4_1",
    "byte_sha256",
    "canonical_file_bytes",
    "canonical_json_bytes",
    "semantic_sha256",
    "validate_calendar_prefix_v4_1",
    "validate_design_source_node_v4_1",
    "validate_holdout_source_node_v4_1",
    "validate_pit_records_v4_1",
]
