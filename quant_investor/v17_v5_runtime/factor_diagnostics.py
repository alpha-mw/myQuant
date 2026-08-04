"""Pure, descriptive Factor diagnostics for V17 v5 Sprint 1A.

This module accepts only caller-supplied in-memory values.  It has no file,
provider, clock, governance, portfolio, or execution surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from enum import Enum
import hashlib
import re
from typing import Any, Final, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_investor.v17_v5_contract.canonical import (
    canonical_bytes,
    seal_semantic,
)
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_sha256,
)
from quant_investor.v17_v5_contract.schema_validation import validate_artifact
from quant_investor.v17_v5_contract.validators import (
    FACTOR_DIAGNOSTIC_POLICY_BYTE_SHA256,
    FACTOR_DIAGNOSTIC_POLICY_ID,
    FACTOR_DIAGNOSTIC_POLICY_PATH,
    FACTOR_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
    FACTOR_DIAGNOSTIC_POLICY_VERSION,
    NO_AUTHORITY,
    V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256,
)

PROTOCOL_VERSION: Final = "myquant.v17.v5"
FACTOR_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-diagnostic.v1"
HORIZON_SESSIONS: Final = 20
MINIMUM_MATURED_ORIGINS: Final = 60
MINIMUM_COMPARABLE_SYMBOLS: Final = 100
MAX_ORIGINS: Final = 4_096
MAX_SYMBOLS_PER_ORIGIN: Final = 10_000
MAX_TOTAL_SYMBOL_ROWS: Final = 2_000_000
OUTPUT_SCALE: Final = Decimal("0.000000000001")
SHANGHAI_TZ: Final = ZoneInfo("Asia/Shanghai")

_DECIMAL_RE: Final = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$",
    re.ASCII,
)
_SESSION_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", re.ASCII)
_SYMBOL_RE: Final = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
_UTC_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)


class FactorDiagnosticError(ValueError):
    """Raised when caller-supplied diagnostic input is malformed."""

    exit_code = 2


class FactorDiagnosticStatus(str, Enum):
    """The only Sprint 1A diagnostic states."""

    UNOBSERVED = "UNOBSERVED"
    ACCUMULATING = "ACCUMULATING"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class FactorSampleStratum:
    """Stable identity shared by every origin in one diagnostic."""

    strategy_id: str
    factor_name: str
    factor_definition_sha256: str
    factor_implementation_sha256: str
    factor_set_sha256: str
    quant_policy_sha256: str
    adapter_policy_byte_sha256: str
    source_lineage_series_sha256: str
    market_calendar_sha256: str
    horizon_sessions: int = HORIZON_SESSIONS


@dataclass(frozen=True)
class FactorOriginSample:
    """One naturally matured origin and its cross-sectional inputs."""

    origin_id: str
    decision_session: str
    horizon_end_session: str
    label_available_at: str
    evidence_lineage_sha256: str
    factor_values: Mapping[str, str]
    forward_returns: Mapping[str, str]


def _fail(message: str) -> None:
    raise FactorDiagnosticError(message)


def _canonical_timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _fail(f"{label} must be a second-precision UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise FactorDiagnosticError(f"{label} is not a valid UTC timestamp") from exc
    return parsed


def _canonical_session(value: Any, *, label: str) -> str:
    if type(value) is not str or _SESSION_RE.fullmatch(value) is None:
        _fail(f"{label} must be an ISO session date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise FactorDiagnosticError(f"{label} is not a valid session date") from exc
    return value


def _canonical_decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise FactorDiagnosticError(f"{label} is not a finite decimal") from exc
    if not number.is_finite() or (number.is_zero() and value.startswith("-")):
        _fail(f"{label} is not a canonical finite decimal string")
    return number


def _render_decimal(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = value.quantize(OUTPUT_SCALE, rounding=ROUND_HALF_EVEN)
    if rendered.is_zero():
        rendered = abs(rendered)
    return format(rendered, ".12f")


def _stratum_document(stratum: FactorSampleStratum) -> dict[str, Any]:
    if not isinstance(stratum, FactorSampleStratum):
        _fail("stratum must be FactorSampleStratum")
    try:
        strategy_id = require_identifier(stratum.strategy_id, label="strategy_id")
        factor_name = require_identifier(stratum.factor_name, label="factor_name")
        factor_definition = require_sha256(
            stratum.factor_definition_sha256,
            label="factor_definition_sha256",
        )
        factor_implementation = require_sha256(
            stratum.factor_implementation_sha256,
            label="factor_implementation_sha256",
        )
        factor_set = require_sha256(
            stratum.factor_set_sha256,
            label="factor_set_sha256",
        )
        quant_policy = require_sha256(
            stratum.quant_policy_sha256,
            label="quant_policy_sha256",
        )
        adapter_policy = require_sha256(
            stratum.adapter_policy_byte_sha256,
            label="adapter_policy_byte_sha256",
        )
        source_series = require_sha256(
            stratum.source_lineage_series_sha256,
            label="source_lineage_series_sha256",
        )
        calendar = require_sha256(
            stratum.market_calendar_sha256,
            label="market_calendar_sha256",
        )
    except IdentityContractError as exc:
        raise FactorDiagnosticError(str(exc)) from exc
    if type(stratum.horizon_sessions) is not int or stratum.horizon_sessions != 20:
        _fail("horizon_sessions must equal 20")
    if adapter_policy != V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256:
        _fail("adapter_policy_byte_sha256 is not the sealed Sprint 1A adapter policy")
    return {
        "adapter_policy_byte_sha256": adapter_policy,
        "factor_definition_sha256": factor_definition,
        "factor_implementation_sha256": factor_implementation,
        "factor_name": factor_name,
        "factor_set_sha256": factor_set,
        "horizon_sessions": HORIZON_SESSIONS,
        "market_calendar_sha256": calendar,
        "quant_policy_sha256": quant_policy,
        "source_lineage_series_sha256": source_series,
        "strategy_id": strategy_id,
    }


def _calendar_index(
    open_sessions: Sequence[str],
    *,
    expected_sha256: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    if isinstance(open_sessions, (str, bytes)) or not isinstance(open_sessions, Sequence):
        _fail("open_sessions must be a sequence")
    sessions = tuple(
        _canonical_session(value, label=f"open_sessions[{index}]")
        for index, value in enumerate(open_sessions)
    )
    if not sessions or sessions != tuple(sorted(sessions)) or len(set(sessions)) != len(sessions):
        _fail("open_sessions must be nonempty, unique, and ASCII ascending")
    observed_sha = hashlib.sha256(canonical_bytes(list(sessions))).hexdigest()
    if observed_sha != expected_sha256:
        _fail("market calendar SHA-256 mismatch")
    return sessions, {session: index for index, session in enumerate(sessions)}


def _numeric_map(values: Mapping[str, str], *, label: str) -> dict[str, Decimal]:
    if not isinstance(values, Mapping):
        _fail(f"{label} must be a mapping")
    if len(values) > MAX_SYMBOLS_PER_ORIGIN:
        _fail(f"{label} exceeds the per-origin symbol limit")
    result: dict[str, Decimal] = {}
    for symbol, value in values.items():
        if type(symbol) is not str or _SYMBOL_RE.fullmatch(symbol) is None:
            _fail(f"{label} contains a noncanonical CN symbol")
        if symbol in result:
            _fail(f"{label} contains a duplicate symbol")
        result[symbol] = _canonical_decimal(value, label=f"{label}[{symbol}]")
    return result


def _average_ranks(values: Mapping[str, Decimal]) -> dict[str, Decimal]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, Decimal] = {}
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[start][1]:
            end += 1
        average = (Decimal(start + 1) + Decimal(end + 1)) / Decimal(2)
        for index in range(start, end + 1):
            result[ordered[index][0]] = average
        start = end + 1
    return result


def _rank_ic(
    factor_values: Mapping[str, Decimal],
    forward_returns: Mapping[str, Decimal],
    symbols: Sequence[str],
) -> tuple[str | None, tuple[str, ...]]:
    factor = {symbol: factor_values[symbol] for symbol in symbols}
    returns = {symbol: forward_returns[symbol] for symbol in symbols}
    if len(set(factor.values())) == 1:
        return None, ("constant_factor",)
    if len(set(returns.values())) == 1:
        return None, ("constant_return",)
    factor_ranks = _average_ranks(factor)
    return_ranks = _average_ranks(returns)
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        count = Decimal(len(symbols))
        factor_mean = sum(factor_ranks.values(), Decimal(0)) / count
        return_mean = sum(return_ranks.values(), Decimal(0)) / count
        numerator = sum(
            (
                (factor_ranks[symbol] - factor_mean) * (return_ranks[symbol] - return_mean)
                for symbol in symbols
            ),
            Decimal(0),
        )
        factor_ss = sum(
            ((factor_ranks[symbol] - factor_mean) ** 2 for symbol in symbols),
            Decimal(0),
        )
        return_ss = sum(
            ((return_ranks[symbol] - return_mean) ** 2 for symbol in symbols),
            Decimal(0),
        )
        denominator = (factor_ss * return_ss).sqrt()
        value = numerator / denominator
    return _render_decimal(value), ()


def _origin_document(
    origin: FactorOriginSample,
    *,
    evaluation_cutoff: datetime,
    session_index: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if not isinstance(origin, FactorOriginSample):
        _fail("origins must contain FactorOriginSample values")
    try:
        origin_id = require_identifier(origin.origin_id, label="origin_id")
        evidence_lineage = require_sha256(
            origin.evidence_lineage_sha256,
            label="evidence_lineage_sha256",
        )
    except IdentityContractError as exc:
        raise FactorDiagnosticError(str(exc)) from exc
    decision_session = _canonical_session(
        origin.decision_session,
        label=f"{origin_id}.decision_session",
    )
    horizon_end_session = _canonical_session(
        origin.horizon_end_session,
        label=f"{origin_id}.horizon_end_session",
    )
    label_available_at = _canonical_timestamp(
        origin.label_available_at,
        label=f"{origin_id}.label_available_at",
    )
    if decision_session not in session_index or horizon_end_session not in session_index:
        _fail(f"{origin_id} sessions are absent from the sealed market calendar")
    if session_index[horizon_end_session] - session_index[decision_session] != 20:
        _fail(f"{origin_id} is not an exact 20-session origin")
    if label_available_at.date() < date.fromisoformat(horizon_end_session):
        _fail(f"{origin_id} label is available before its horizon end")
    horizon_close = datetime.combine(
        date.fromisoformat(horizon_end_session),
        time(hour=15),
        tzinfo=SHANGHAI_TZ,
    )
    if label_available_at.astimezone(SHANGHAI_TZ) < horizon_close:
        _fail(f"{origin_id} label is available before the Shanghai session close")
    if label_available_at > evaluation_cutoff:
        _fail(f"{origin_id} label is not naturally matured at the evaluation cutoff")
    factor_values = _numeric_map(
        origin.factor_values,
        label=f"{origin_id}.factor_values",
    )
    forward_returns = _numeric_map(
        origin.forward_returns,
        label=f"{origin_id}.forward_returns",
    )
    total_rows = len(factor_values) + len(forward_returns)
    symbols = tuple(sorted(set(factor_values).intersection(forward_returns)))
    if len(symbols) > MAX_SYMBOLS_PER_ORIGIN:
        _fail(f"{origin_id} exceeds the comparable-symbol limit")
    rank_ic: str | None
    blockers: tuple[str, ...]
    if len(symbols) < 2:
        rank_ic = None
        blockers = ("insufficient_comparable_symbols",)
    else:
        rank_ic, blockers = _rank_ic(factor_values, forward_returns, symbols)
    normalized_input = {
        "decision_session": decision_session,
        "evidence_lineage_sha256": evidence_lineage,
        "factor_values": {
            symbol: str(origin.factor_values[symbol]) for symbol in sorted(factor_values)
        },
        "forward_returns": {
            symbol: str(origin.forward_returns[symbol]) for symbol in sorted(forward_returns)
        },
        "horizon_end_session": horizon_end_session,
        "label_available_at": origin.label_available_at,
        "origin_id": origin_id,
    }
    diagnostic = {
        "blockers": list(blockers),
        "comparable_symbol_count": len(symbols),
        "decision_session": decision_session,
        "evidence_lineage_sha256": evidence_lineage,
        "horizon_end_session": horizon_end_session,
        "label_available_at": origin.label_available_at,
        "origin_id": origin_id,
        "rank_ic": rank_ic,
        "rank_ic_status": "AVAILABLE" if rank_ic is not None else "UNAVAILABLE",
    }
    return normalized_input, diagnostic, total_rows


def _statistics(values: Sequence[str]) -> dict[str, str] | None:
    if not values:
        return None
    decimals = [Decimal(value) for value in values]
    ordered = sorted(decimals)
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        count = Decimal(len(decimals))
        mean = sum(decimals, Decimal(0)) / count
        variance = sum((value - mean) ** 2 for value in decimals) / count
        if len(ordered) % 2:
            median = ordered[len(ordered) // 2]
        else:
            middle = len(ordered) // 2
            median = (ordered[middle - 1] + ordered[middle]) / Decimal(2)
        stddev = variance.sqrt()
    return {
        "rank_ic_max": _render_decimal(max(ordered)),
        "rank_ic_mean": _render_decimal(mean),
        "rank_ic_median": _render_decimal(median),
        "rank_ic_min": _render_decimal(min(ordered)),
        "rank_ic_population_stddev": _render_decimal(stddev),
    }


def _policy_ref() -> dict[str, str]:
    return {
        "artifact_id": FACTOR_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_DIAGNOSTIC_POLICY_VERSION,
    }


def _seal_document(document: dict[str, Any]) -> dict[str, Any]:
    identity_material = dict(document)
    identity_material.pop("diagnostic_id", None)
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    document["diagnostic_id"] = f"factor-diagnostic-{identity[:32]}"
    sealed = seal_semantic(document)
    return validate_artifact(sealed)


def build_factor_diagnostic(
    *,
    stratum: FactorSampleStratum,
    evaluation_cutoff: str,
    open_sessions: Sequence[str],
    origins: Sequence[FactorOriginSample],
) -> dict[str, Any]:
    """Build an in-memory descriptive diagnostic for one exact stratum."""

    stratum_document = _stratum_document(stratum)
    cutoff = _canonical_timestamp(evaluation_cutoff, label="evaluation_cutoff")
    _, session_index = _calendar_index(
        open_sessions,
        expected_sha256=stratum.market_calendar_sha256,
    )
    if isinstance(origins, (str, bytes)) or not isinstance(origins, Sequence):
        _fail("origins must be a sequence")
    if len(origins) > MAX_ORIGINS:
        _fail("origins exceed the diagnostic resource limit")
    by_origin_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    by_session: dict[str, str] = {}
    total_rows = 0
    for origin in origins:
        normalized_input, diagnostic, row_count = _origin_document(
            origin,
            evaluation_cutoff=cutoff,
            session_index=session_index,
        )
        origin_id = diagnostic["origin_id"]
        decision_session = diagnostic["decision_session"]
        existing = by_origin_id.get(origin_id)
        if existing is not None:
            if existing[0] != normalized_input:
                _fail(f"conflicting duplicate origin_id: {origin_id}")
            continue
        previous_origin = by_session.get(decision_session)
        if previous_origin is not None:
            _fail(f"conflicting origin identity for decision session {decision_session}")
        by_origin_id[origin_id] = (normalized_input, diagnostic)
        by_session[decision_session] = origin_id
        total_rows += row_count
        if total_rows > MAX_TOTAL_SYMBOL_ROWS:
            _fail("origins exceed the total symbol-row limit")
    origin_rows = sorted(
        (value[1] for value in by_origin_id.values()),
        key=lambda row: (row["decision_session"], row["origin_id"]),
    )
    available_rows = [row for row in origin_rows if row["rank_ic_status"] == "AVAILABLE"]
    coverage_met = len(available_rows) >= MINIMUM_MATURED_ORIGINS and all(
        row["comparable_symbol_count"] >= MINIMUM_COMPARABLE_SYMBOLS for row in available_rows
    )
    if not origin_rows:
        status = FactorDiagnosticStatus.UNOBSERVED.value
        blockers = ["inference_not_implemented", "no_naturally_matured_origins"]
    else:
        status = FactorDiagnosticStatus.ACCUMULATING.value
        blockers = ["inference_not_implemented"]
        if not coverage_met:
            blockers.insert(0, "descriptive_coverage_minimum_not_met")
    stratum_sha = hashlib.sha256(canonical_bytes(stratum_document)).hexdigest()
    available_rank_ics = [row["rank_ic"] for row in available_rows if type(row["rank_ic"]) is str]
    return _seal_document(
        {
            "authority": dict(NO_AUTHORITY),
            "blockers": blockers,
            "descriptive_coverage_minimum_met": coverage_met,
            "descriptive_only": True,
            "diagnostic_id": "",
            "effectiveness_claimed": False,
            "effectiveness_conclusion": None,
            "evaluation_cutoff": evaluation_cutoff,
            "factor_tier_change_eligible": False,
            "factor_weight_change_eligible": False,
            "gate_scope": "DESCRIPTIVE_ONLY",
            "inference_eligible": False,
            "inference_gate_passed": False,
            "matured_origin_count": len(origin_rows),
            "minimum_comparable_symbol_count": (
                min(row["comparable_symbol_count"] for row in origin_rows) if origin_rows else None
            ),
            "origin_diagnostics": origin_rows,
            "policy_ref": _policy_ref(),
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "rank_ic_available_origin_count": len(available_rows),
            "statistics": _statistics(available_rank_ics),
            "status": status,
            "stratum": stratum_document,
            "stratum_sha256": stratum_sha,
            "subject_factor_name": stratum_document["factor_name"],
            "total_comparable_symbol_rows": sum(
                row["comparable_symbol_count"] for row in origin_rows
            ),
            "version": FACTOR_DIAGNOSTIC_VERSION,
        }
    )


def build_unavailable_factor_diagnostic(
    *,
    factor_name: str,
    evaluation_cutoff: str,
    unavailable_prerequisites: Sequence[str],
) -> dict[str, Any]:
    """Build a non-receipt diagnostic for an explicit prerequisite gap."""

    try:
        subject = require_identifier(factor_name, label="factor_name")
    except IdentityContractError as exc:
        raise FactorDiagnosticError(str(exc)) from exc
    _canonical_timestamp(evaluation_cutoff, label="evaluation_cutoff")
    if (
        isinstance(unavailable_prerequisites, (str, bytes))
        or not isinstance(unavailable_prerequisites, Sequence)
        or not unavailable_prerequisites
    ):
        _fail("unavailable_prerequisites must be a nonempty sequence")
    blockers: list[str] = []
    for value in unavailable_prerequisites:
        try:
            blockers.append(require_identifier(value, label="unavailable prerequisite"))
        except IdentityContractError as exc:
            raise FactorDiagnosticError(str(exc)) from exc
    reserved = {
        "descriptive_coverage_minimum_not_met",
        "inference_not_implemented",
        "no_naturally_matured_origins",
    }
    if reserved.intersection(blockers):
        _fail("unavailable_prerequisites contains a reserved diagnostic blocker")
    blockers = sorted(set(blockers))
    if "inference_not_implemented" not in blockers:
        blockers.append("inference_not_implemented")
        blockers.sort()
    return _seal_document(
        {
            "authority": dict(NO_AUTHORITY),
            "blockers": blockers,
            "descriptive_coverage_minimum_met": False,
            "descriptive_only": True,
            "diagnostic_id": "",
            "effectiveness_claimed": False,
            "effectiveness_conclusion": None,
            "evaluation_cutoff": evaluation_cutoff,
            "factor_tier_change_eligible": False,
            "factor_weight_change_eligible": False,
            "gate_scope": "DESCRIPTIVE_ONLY",
            "inference_eligible": False,
            "inference_gate_passed": False,
            "matured_origin_count": 0,
            "minimum_comparable_symbol_count": None,
            "origin_diagnostics": [],
            "policy_ref": _policy_ref(),
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "rank_ic_available_origin_count": 0,
            "statistics": None,
            "status": FactorDiagnosticStatus.UNAVAILABLE.value,
            "stratum": None,
            "stratum_sha256": None,
            "subject_factor_name": subject,
            "total_comparable_symbol_rows": 0,
            "version": FACTOR_DIAGNOSTIC_VERSION,
        }
    )


def validate_factor_diagnostic_replay(
    artifact: Mapping[str, Any],
    *,
    stratum: FactorSampleStratum | None = None,
    evaluation_cutoff: str,
    open_sessions: Sequence[str] = (),
    origins: Sequence[FactorOriginSample] = (),
    factor_name: str | None = None,
    unavailable_prerequisites: Sequence[str] = (),
) -> dict[str, Any]:
    """Rebuild and compare a diagnostic without reading or writing files."""

    validated = validate_artifact(artifact)
    if validated["status"] == FactorDiagnosticStatus.UNAVAILABLE.value:
        if factor_name is None or stratum is not None or origins or open_sessions:
            _fail("unavailable replay arguments are inconsistent")
        rebuilt = build_unavailable_factor_diagnostic(
            factor_name=factor_name,
            evaluation_cutoff=evaluation_cutoff,
            unavailable_prerequisites=unavailable_prerequisites,
        )
    else:
        if stratum is None or factor_name is not None or unavailable_prerequisites:
            _fail("observed replay arguments are inconsistent")
        rebuilt = build_factor_diagnostic(
            stratum=stratum,
            evaluation_cutoff=evaluation_cutoff,
            open_sessions=open_sessions,
            origins=origins,
        )
    if canonical_bytes(validated) != canonical_bytes(rebuilt):
        _fail("factor diagnostic replay mismatch")
    return validated


__all__ = [
    "FACTOR_DIAGNOSTIC_VERSION",
    "FactorDiagnosticError",
    "FactorDiagnosticStatus",
    "FactorOriginSample",
    "FactorSampleStratum",
    "build_factor_diagnostic",
    "build_unavailable_factor_diagnostic",
    "validate_factor_diagnostic_replay",
]
