"""Immutable official Macro release-calendar evidence.

The release calendar is a code-owned readiness control, not an observation
source.  A caller must provide exact plan, capture-manifest, raw-capture and
``market-open-days.v1`` paths plus their byte hashes.  Publication copies those
bytes into one private immutable generation and advances a small pointer only
after a compare-and-swap check.

Calendar artifacts cannot declare their own criticality.  The twelve official
production indicator series and their issuer/parser/measurement contracts are
defined below and hash-bound to the current Macro registry.  In particular,
PBC response evidence is ``pbc_official`` while the normalized observation
lineage is ``pboc_official``.
"""

from __future__ import annotations

import calendar as month_calendar
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_investor.macro.contracts import (
    MacroObservation,
    normalize_source_url,
    parse_timestamp,
)
from quant_investor.macro.official_web_compiler import (
    NBS_NATIONAL_ECONOMY_PARSER,
    NBS_OFFICIAL_PMI_PARSER,
    NBS_QUARTERLY_GDP_PARSER,
    NBS_QUARTERLY_GDP_PARSER_V2,
    PARSER_CONTRACT_SHA256,
    PBC_MONEY_STOCK_PARSER,
    PBC_MONEY_STOCK_PARSER_V2,
    parse_official_support_page,
)
from quant_investor.macro.registry import NATIONAL_INDICATORS, REGISTRY_VERSION


MACRO_RELEASE_CALENDAR_SCHEMA = "macro-release-calendar.v1"
MACRO_RELEASE_CALENDAR_PLAN_SCHEMA = "macro-release-calendar-plan.v1"
MACRO_RELEASE_CALENDAR_CAPTURE_SCHEMA = "macro-release-calendar-capture.v1"
MACRO_RELEASE_CALENDAR_POINTER_SCHEMA = "macro-release-calendar-pointer.v1"
MACRO_RELEASE_CALENDAR_GENERATION_SCHEMA = (
    "macro-release-calendar-generation.v1"
)
MARKET_OPEN_DAYS_SCHEMA = "market-open-days.v1"
CRITICAL_POLICY_VERSION = "macro-release-critical-policy.v1"
EMPTY_POINTER_SHA256 = "EMPTY"

POINTER_FILENAME = "_latest.json"
GENERATIONS_DIRNAME = "_generations"
LOCK_FILENAME = ".release-calendar.lock"

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_UTC = ZoneInfo("UTC")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_COMPACT_DATE_RE = re.compile(r"^20\d{6}$")
_ISO_MONTH_RE = re.compile(r"^(20\d{2})-(0[1-9]|1[0-2])$")
_ISO_QUARTER_RE = re.compile(r"^(20\d{2})-Q([1-4])$")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_RAW_BYTES = 64 * 1024 * 1024
_ARTIFACT_KINDS = frozenset(
    {
        "coverage_receipt",
        "official_bundle",
        "observation",
        "parser_contract",
        "release_notice",
    }
)
_EVENT_STATUSES = frozenset(
    {"scheduled", "released", "rescheduled", "cancelled"}
)


class MacroReleaseCalendarError(RuntimeError):
    """Base error for release-calendar validation and publication."""


class ReleaseCalendarValidationError(MacroReleaseCalendarError):
    """Raised when source or canonical calendar evidence is invalid."""


class ReleaseCalendarCASMismatch(MacroReleaseCalendarError):
    """Raised when the explicit pointer compare-and-swap expectation drifts."""


@dataclass(frozen=True)
class CriticalIndicatorRule:
    """Code-owned release and exact-observation equivalence policy."""

    indicator_id: str
    event_family: str
    evidence_issuer: str
    observation_issuer: str
    frequency: str
    unit: str
    measurement_basis: str
    allowed_parsers: tuple[tuple[str, str], ...]


_GDP_PARSERS = (
    (
        NBS_QUARTERLY_GDP_PARSER_V2,
        PARSER_CONTRACT_SHA256[NBS_QUARTERLY_GDP_PARSER_V2],
    ),
    (
        NBS_QUARTERLY_GDP_PARSER,
        PARSER_CONTRACT_SHA256[NBS_QUARTERLY_GDP_PARSER],
    ),
)
_PBC_PARSERS = (
    (
        PBC_MONEY_STOCK_PARSER_V2,
        PARSER_CONTRACT_SHA256[PBC_MONEY_STOCK_PARSER_V2],
    ),
    (
        PBC_MONEY_STOCK_PARSER,
        PARSER_CONTRACT_SHA256[PBC_MONEY_STOCK_PARSER],
    ),
)
_ECONOMY_PARSERS = (
    (
        NBS_NATIONAL_ECONOMY_PARSER,
        PARSER_CONTRACT_SHA256[NBS_NATIONAL_ECONOMY_PARSER],
    ),
)
_PMI_PARSERS = (
    (
        NBS_OFFICIAL_PMI_PARSER,
        PARSER_CONTRACT_SHA256[NBS_OFFICIAL_PMI_PARSER],
    ),
)

CRITICAL_INDICATOR_POLICY: tuple[CriticalIndicatorRule, ...] = (
    CriticalIndicatorRule(
        "cn.cpi_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "current_month_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.exports_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "current_month_cny_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.fixed_asset_investment_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "jan_to_month_cumulative_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.gdp_yoy",
        "nbs_quarterly_gdp",
        "nbs_official",
        "nbs_official",
        "quarterly",
        "%",
        "current_quarter_real_yoy",
        _GDP_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.imports_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "current_month_cny_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.industrial_value_added_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "current_month_real_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.m1_yoy",
        "pbc_money_stock",
        "pbc_official",
        "pboc_official",
        "monthly",
        "%",
        "month_end_yoy",
        _PBC_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.m2_yoy",
        "pbc_money_stock",
        "pbc_official",
        "pboc_official",
        "monthly",
        "%",
        "month_end_yoy",
        _PBC_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.pmi_manufacturing",
        "nbs_pmi",
        "nbs_official",
        "nbs_official",
        "monthly",
        "index",
        "headline_manufacturing_pmi",
        _PMI_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.ppi_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "current_month_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.property_investment_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "jan_to_month_cumulative_yoy",
        _ECONOMY_PARSERS,
    ),
    CriticalIndicatorRule(
        "cn.retail_sales_yoy",
        "nbs_national_economy",
        "nbs_official",
        "nbs_official",
        "monthly",
        "%",
        "current_month_nominal_yoy",
        _ECONOMY_PARSERS,
    ),
)


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseCalendarValidationError(
            "release_calendar_json_not_canonicalizable"
        ) from exc
    return encoded + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


MACRO_REGISTRY_SHA256 = _semantic_sha256(
    {
        "registry_version": REGISTRY_VERSION,
        "national_indicators": [asdict(item) for item in NATIONAL_INDICATORS],
    }
)
CRITICAL_POLICY_SHA256 = _semantic_sha256(
    {
        "schema_version": CRITICAL_POLICY_VERSION,
        "registry_version": REGISTRY_VERSION,
        "registry_sha256": MACRO_REGISTRY_SHA256,
        "indicators": [asdict(item) for item in CRITICAL_INDICATOR_POLICY],
    }
)
CRITICAL_INDICATOR_IDS = tuple(
    item.indicator_id for item in CRITICAL_INDICATOR_POLICY
)

_POLICY_BY_ID = {item.indicator_id: item for item in CRITICAL_INDICATOR_POLICY}
_FAMILY_INDICATORS: dict[str, tuple[str, ...]] = {}
for _policy_item in CRITICAL_INDICATOR_POLICY:
    _FAMILY_INDICATORS.setdefault(_policy_item.event_family, tuple())
    _FAMILY_INDICATORS[_policy_item.event_family] = (
        *_FAMILY_INDICATORS[_policy_item.event_family],
        _policy_item.indicator_id,
    )
_FAMILY_INDICATORS = {
    key: tuple(sorted(value)) for key, value in _FAMILY_INDICATORS.items()
}
CRITICAL_EVENT_FAMILIES = tuple(sorted(_FAMILY_INDICATORS))

_registry_by_id = {item.indicator_id: item for item in NATIONAL_INDICATORS}
if len(_POLICY_BY_ID) != 12 or set(_POLICY_BY_ID) != set(CRITICAL_INDICATOR_IDS):
    raise RuntimeError("macro_release_critical_policy_not_exactly_twelve")
for _policy_item in CRITICAL_INDICATOR_POLICY:
    _definition = _registry_by_id.get(_policy_item.indicator_id)
    if (
        _definition is None
        or _definition.frequency != _policy_item.frequency
        or _definition.unit != _policy_item.unit
    ):
        raise RuntimeError("macro_release_critical_policy_registry_drift")


@dataclass(frozen=True)
class IssuerCoverage:
    issuer: str
    through_at: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceArtifactRef:
    source_id: str
    issuer: str
    artifact_kind: str
    source_url: str
    http_status: int
    captured_at: str
    raw_path: str
    stored_path: str
    raw_sha256: str
    size_bytes: int
    content_sha256: str


@dataclass(frozen=True)
class ReleaseResolution:
    resolution_id: str
    event_id: str
    indicator_id: str
    period_end: str
    frequency: str
    unit: str
    measurement_basis: str
    value_decimal: str
    issuer: str
    parser_id: str
    parser_contract_sha256: str
    official_bundle_sha256: str
    observation_content_hash: str
    observation_available_at: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseEvent:
    event_id: str
    event_family: str
    issuer: str
    indicator_ids: tuple[str, ...]
    period: str
    schedule_kind: str
    scheduled_at: str
    status: str
    actual_at: str
    rescheduled_at: str
    reschedule_kind: str
    cancelled_at: str
    supersedes_event_id: str
    source_ids: tuple[str, ...]
    resolution_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseCalendarIdentity:
    pointer_path: str
    pointer_sha256: str
    generation_id: str
    generation_path: str
    manifest_sha256: str
    semantic_sha256: str
    parent_generation_id: str
    parent_pointer_sha256: str
    parent_manifest_sha256: str
    parent_semantic_sha256: str


@dataclass(frozen=True)
class ReleaseCalendarEvidence:
    """Frozen, pure canonical release-calendar evidence and its identity."""

    identity: ReleaseCalendarIdentity
    registry_version: str
    registry_sha256: str
    critical_policy_version: str
    critical_policy_sha256: str
    plan_sha256: str
    capture_manifest_sha256: str
    market_open_days_sha256: str
    captured_at: str
    open_dates: tuple[str, ...]
    issuer_coverage: tuple[IssuerCoverage, ...]
    source_artifacts: tuple[SourceArtifactRef, ...]
    events: tuple[ReleaseEvent, ...]
    resolutions: tuple[ReleaseResolution, ...]


@dataclass(frozen=True)
class ReleaseCalendarPublishResult:
    identity: ReleaseCalendarIdentity
    evidence: ReleaseCalendarEvidence
    idempotent: bool


@dataclass(frozen=True)
class SessionLagEvaluation:
    ready: bool
    session_lag: int | None
    macro_logical_date: str
    target_session_date: str
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class CriticalEventGapEvaluation:
    ready: bool
    window_start_exclusive: str
    window_end_inclusive: str
    relevant_event_ids: tuple[str, ...]
    resolved_event_ids: tuple[str, ...]
    blocking_event_ids: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseReadinessEvaluation:
    ready: bool
    session_lag: SessionLagEvaluation
    critical_event_gap: CriticalEventGapEvaluation
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class _Readback:
    path: Path
    raw: bytes
    signature: tuple[int, ...]
    max_bytes: int


@dataclass(frozen=True)
class _CalendarContent:
    plan_sha256: str
    capture_manifest_sha256: str
    market_open_days_sha256: str
    captured_at: str
    open_dates: tuple[str, ...]
    issuer_coverage: tuple[IssuerCoverage, ...]
    source_artifacts: tuple[SourceArtifactRef, ...]
    events: tuple[ReleaseEvent, ...]
    resolutions: tuple[ReleaseResolution, ...]


@dataclass(frozen=True)
class _CompiledInputs:
    content: _CalendarContent
    plan_raw: bytes
    capture_raw: bytes
    open_days_raw: bytes
    raw_by_path: tuple[tuple[str, bytes], ...]
    readbacks: tuple[_Readback, ...]
    raw_root: Path


def _exact_keys(value: Mapping[str, Any], keys: set[str], blocker: str) -> None:
    if set(value) != keys:
        raise ReleaseCalendarValidationError(blocker)


def _mapping(value: Any, blocker: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseCalendarValidationError(blocker)
    return value


def _list(value: Any, blocker: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseCalendarValidationError(blocker)
    return value


def _safe_id(value: Any, blocker: str) -> str:
    text = str(value or "")
    if text in {"", ".", ".."} or _SAFE_ID_RE.fullmatch(text) is None:
        raise ReleaseCalendarValidationError(blocker)
    return text


def _required_sha256(value: Any, blocker: str) -> str:
    text = str(value or "").lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise ReleaseCalendarValidationError(blocker)
    return text


def _canonical_decimal(value: Any, blocker: str) -> str:
    text = str(value or "")
    if not text or text.strip() != text:
        raise ReleaseCalendarValidationError(blocker)
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ReleaseCalendarValidationError(blocker) from exc
    if not parsed.is_finite():
        raise ReleaseCalendarValidationError(blocker)
    normalized = format(parsed.normalize(), "f")
    if normalized in {"", "-0"}:
        normalized = "0"
    if text != normalized:
        raise ReleaseCalendarValidationError(blocker)
    return normalized


def _utc_timestamp(value: Any, blocker: str, *, allow_empty: bool = False) -> str:
    text = str(value or "")
    if allow_empty and not text:
        return ""
    try:
        return parse_timestamp(text, field_name=blocker).isoformat()
    except ValueError as exc:
        raise ReleaseCalendarValidationError(blocker) from exc


def _schedule(value: Any, blocker: str) -> tuple[str, str]:
    text = str(value or "")
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", text):
        try:
            return "date", date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ReleaseCalendarValidationError(blocker) from exc
    return "timestamp", _utc_timestamp(text, blocker)


def _period_end(period: str, frequency: str) -> str:
    if frequency == "monthly":
        match = _ISO_MONTH_RE.fullmatch(period)
        if match is None:
            raise ReleaseCalendarValidationError(
                "release_calendar_event_period_invalid"
            )
        year, month = int(match.group(1)), int(match.group(2))
        return date(year, month, month_calendar.monthrange(year, month)[1]).isoformat()
    if frequency == "quarterly":
        match = _ISO_QUARTER_RE.fullmatch(period)
        if match is None:
            raise ReleaseCalendarValidationError(
                "release_calendar_event_period_invalid"
            )
        year, quarter = int(match.group(1)), int(match.group(2))
        month = quarter * 3
        return date(year, month, month_calendar.monthrange(year, month)[1]).isoformat()
    raise ReleaseCalendarValidationError("release_calendar_event_frequency_invalid")


def _parser_period(period: str, frequency: str) -> str:
    if frequency == "monthly":
        return period.replace("-", "")
    match = _ISO_QUARTER_RE.fullmatch(period)
    if match is None:
        raise ReleaseCalendarValidationError("release_calendar_event_period_invalid")
    return f"{match.group(1)}Q{match.group(2)}"


def _safe_relative(value: Any, blocker: str) -> str:
    text = str(value or "")
    if not text or "\\" in text:
        raise ReleaseCalendarValidationError(blocker)
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or str(relative) != text
    ):
        raise ReleaseCalendarValidationError(blocker)
    return text


def _absolute_path(value: str | Path, blocker: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ReleaseCalendarValidationError(blocker)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if os.path.lexists(current):
            try:
                if stat.S_ISLNK(os.lstat(current).st_mode):
                    raise ReleaseCalendarValidationError(blocker)
            except OSError as exc:
                raise ReleaseCalendarValidationError(blocker) from exc
    return path


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_mode & 0o777,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_bytes(
    path: Path,
    *,
    blocker: str,
    max_bytes: int,
    exact_mode: int | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > max_bytes
            or (exact_mode is not None and before.st_mode & 0o777 != exact_mode)
        ):
            raise ReleaseCalendarValidationError(blocker)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except ReleaseCalendarValidationError:
        raise
    except OSError as exc:
        raise ReleaseCalendarValidationError(blocker) from exc
    try:
        signature = _stat_signature(before)
        if _stat_signature(os.fstat(descriptor)) != signature:
            raise ReleaseCalendarValidationError(f"{blocker}_changed_during_read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ReleaseCalendarValidationError(blocker)
        if (
            _stat_signature(os.fstat(descriptor)) != signature
            or _stat_signature(os.lstat(path)) != signature
        ):
            raise ReleaseCalendarValidationError(f"{blocker}_changed_during_read")
        return b"".join(chunks), signature
    except ReleaseCalendarValidationError:
        raise
    except OSError as exc:
        raise ReleaseCalendarValidationError(f"{blocker}_changed_during_read") from exc
    finally:
        os.close(descriptor)


def _strict_json(raw: bytes, blocker: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ReleaseCalendarValidationError(f"{blocker}_duplicate_key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except ReleaseCalendarValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseCalendarValidationError(blocker) from exc
    return _mapping(payload, blocker)


def _read_json_artifact(
    path_value: str | Path,
    *,
    expected_sha256: str,
    blocker: str,
    exact_mode: int | None = None,
) -> tuple[Path, bytes, Mapping[str, Any], _Readback]:
    path = _absolute_path(path_value, f"{blocker}_path_unsafe")
    expected = _required_sha256(expected_sha256, f"{blocker}_expected_sha256_invalid")
    raw, signature = _stable_file_bytes(
        path,
        blocker=f"{blocker}_unsafe",
        max_bytes=_MAX_JSON_BYTES,
        exact_mode=exact_mode,
    )
    if _sha256_bytes(raw) != expected:
        raise ReleaseCalendarValidationError(f"{blocker}_sha256_mismatch")
    return (
        path,
        raw,
        _strict_json(raw, f"{blocker}_json_invalid"),
        _Readback(path, raw, signature, _MAX_JSON_BYTES),
    )


def _raw_files(root: Path) -> dict[str, Path]:
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ReleaseCalendarValidationError("release_calendar_raw_root_unsafe")
    result: dict[str, Path] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        if base.is_symlink() or not base.is_dir():
            raise ReleaseCalendarValidationError("release_calendar_raw_root_unsafe")
        for name in directory_names:
            child = base / name
            if child.is_symlink() or not child.is_dir():
                raise ReleaseCalendarValidationError("release_calendar_raw_symlink_rejected")
        for name in file_names:
            child = base / name
            relative = child.relative_to(root).as_posix()
            _safe_relative(relative, "release_calendar_raw_path_unsafe")
            if relative in result:
                raise ReleaseCalendarValidationError("release_calendar_raw_path_duplicate")
            result[relative] = child
    return result


def _validate_plan(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "market",
            "registry_version",
            "registry_sha256",
            "critical_policy_version",
            "critical_policy_sha256",
            "events",
        },
        "release_calendar_plan_shape_invalid",
    )
    if payload.get("schema_version") != MACRO_RELEASE_CALENDAR_PLAN_SCHEMA:
        raise ReleaseCalendarValidationError("release_calendar_plan_schema_invalid")
    if payload.get("market") != "CN":
        raise ReleaseCalendarValidationError("release_calendar_plan_market_invalid")
    if (
        payload.get("registry_version") != REGISTRY_VERSION
        or payload.get("registry_sha256") != MACRO_REGISTRY_SHA256
    ):
        raise ReleaseCalendarValidationError("release_calendar_registry_policy_drift")
    if (
        payload.get("critical_policy_version") != CRITICAL_POLICY_VERSION
        or payload.get("critical_policy_sha256") != CRITICAL_POLICY_SHA256
    ):
        raise ReleaseCalendarValidationError("release_calendar_critical_policy_drift")
    rows = _list(payload.get("events"), "release_calendar_plan_events_not_list")
    normalized: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    for raw_row in rows:
        row = _mapping(raw_row, "release_calendar_plan_event_not_object")
        _exact_keys(
            row,
            {
                "event_id",
                "event_family",
                "issuer",
                "indicator_ids",
                "period",
                "scheduled_at",
            },
            "release_calendar_plan_event_shape_invalid",
        )
        event_id = _safe_id(row.get("event_id"), "release_calendar_event_id_invalid")
        if event_id in ids:
            raise ReleaseCalendarValidationError("release_calendar_event_id_duplicate")
        ids.add(event_id)
        family = str(row.get("event_family") or "")
        expected_indicators = _FAMILY_INDICATORS.get(family)
        if expected_indicators is None:
            raise ReleaseCalendarValidationError("release_calendar_event_family_unknown")
        indicators = _list(
            row.get("indicator_ids"), "release_calendar_event_indicators_not_list"
        )
        if tuple(indicators) != expected_indicators:
            unknown = set(str(item) for item in indicators) - set(_POLICY_BY_ID)
            blocker = (
                "release_calendar_indicator_unknown"
                if unknown
                else "release_calendar_event_policy_scope_mismatch"
            )
            raise ReleaseCalendarValidationError(blocker)
        rules = [_POLICY_BY_ID[item] for item in expected_indicators]
        issuer = str(row.get("issuer") or "")
        if {item.evidence_issuer for item in rules} != {issuer}:
            raise ReleaseCalendarValidationError("release_calendar_event_issuer_invalid")
        frequency = rules[0].frequency
        period = str(row.get("period") or "")
        _period_end(period, frequency)
        schedule_kind, scheduled_at = _schedule(
            row.get("scheduled_at"), "release_calendar_event_schedule_invalid"
        )
        normalized.append(
            {
                "event_id": event_id,
                "event_family": family,
                "issuer": issuer,
                "indicator_ids": expected_indicators,
                "period": period,
                "frequency": frequency,
                "schedule_kind": schedule_kind,
                "scheduled_at": scheduled_at,
            }
        )
    return tuple(normalized)


def _validate_open_days(payload: Mapping[str, Any]) -> tuple[str, ...]:
    _exact_keys(
        payload,
        {"schema_version", "market", "open_dates"},
        "release_calendar_open_days_shape_invalid",
    )
    if payload.get("schema_version") != MARKET_OPEN_DAYS_SCHEMA:
        raise ReleaseCalendarValidationError("release_calendar_open_days_schema_invalid")
    if payload.get("market") != "CN":
        raise ReleaseCalendarValidationError("release_calendar_open_days_market_invalid")
    values = _list(payload.get("open_dates"), "release_calendar_open_dates_not_list")
    normalized: list[str] = []
    for value in values:
        text = str(value or "")
        if _COMPACT_DATE_RE.fullmatch(text) is None:
            raise ReleaseCalendarValidationError("release_calendar_open_date_invalid")
        try:
            parsed = datetime.strptime(text, "%Y%m%d").date()
        except ValueError as exc:
            raise ReleaseCalendarValidationError("release_calendar_open_date_invalid") from exc
        if parsed.weekday() >= 5:
            raise ReleaseCalendarValidationError("release_calendar_open_date_weekend")
        normalized.append(parsed.isoformat())
    if not normalized or normalized != sorted(normalized) or len(set(normalized)) != len(
        normalized
    ):
        raise ReleaseCalendarValidationError("release_calendar_open_dates_not_strict")
    return tuple(normalized)


def _observation_from_source(raw: bytes) -> MacroObservation:
    payload = _strict_json(raw, "release_calendar_observation_json_invalid")
    try:
        return MacroObservation.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise ReleaseCalendarValidationError(
            "release_calendar_observation_contract_invalid"
        ) from exc


def _validate_capture(
    payload: Mapping[str, Any],
    *,
    plan_sha256: str,
    planned_events: Sequence[Mapping[str, Any]],
    raw_root: Path,
) -> tuple[
    str,
    tuple[IssuerCoverage, ...],
    tuple[SourceArtifactRef, ...],
    tuple[ReleaseEvent, ...],
    tuple[ReleaseResolution, ...],
    tuple[tuple[str, bytes], ...],
    tuple[_Readback, ...],
]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "market",
            "plan_sha256",
            "captured_at",
            "issuer_coverage",
            "sources",
            "events",
            "resolutions",
        },
        "release_calendar_capture_shape_invalid",
    )
    if payload.get("schema_version") != MACRO_RELEASE_CALENDAR_CAPTURE_SCHEMA:
        raise ReleaseCalendarValidationError("release_calendar_capture_schema_invalid")
    if payload.get("market") != "CN":
        raise ReleaseCalendarValidationError("release_calendar_capture_market_invalid")
    if payload.get("plan_sha256") != plan_sha256:
        raise ReleaseCalendarValidationError("release_calendar_capture_plan_sha_mismatch")
    captured_at = _utc_timestamp(
        payload.get("captured_at"), "release_calendar_capture_clock_invalid"
    )
    captured_clock = parse_timestamp(captured_at, field_name="captured_at")

    declared_sources = _list(
        payload.get("sources"), "release_calendar_capture_sources_not_list"
    )
    source_rows: list[SourceArtifactRef] = []
    sources_by_id: dict[str, SourceArtifactRef] = {}
    source_payloads: dict[str, Any] = {}
    raw_by_path: list[tuple[str, bytes]] = []
    raw_readbacks: list[_Readback] = []
    declared_paths: set[str] = set()
    disk_files = _raw_files(raw_root)
    for raw_source in declared_sources:
        source = _mapping(raw_source, "release_calendar_capture_source_not_object")
        _exact_keys(
            source,
            {
                "source_id",
                "issuer",
                "artifact_kind",
                "source_url",
                "http_status",
                "captured_at",
                "raw_path",
                "raw_sha256",
                "size_bytes",
                "content_sha256",
            },
            "release_calendar_capture_source_shape_invalid",
        )
        source_id = _safe_id(
            source.get("source_id"), "release_calendar_source_id_invalid"
        )
        if source_id in sources_by_id:
            raise ReleaseCalendarValidationError("release_calendar_source_id_duplicate")
        issuer = str(source.get("issuer") or "")
        if issuer not in {"nbs_official", "pbc_official"}:
            raise ReleaseCalendarValidationError("release_calendar_source_issuer_unknown")
        artifact_kind = str(source.get("artifact_kind") or "")
        if artifact_kind not in _ARTIFACT_KINDS:
            raise ReleaseCalendarValidationError("release_calendar_artifact_kind_unknown")
        raw_url = str(source.get("source_url") or "")
        try:
            normalized_url = normalize_source_url(raw_url, source_system=issuer)
        except ValueError as exc:
            raise ReleaseCalendarValidationError(
                "release_calendar_source_url_invalid"
            ) from exc
        if raw_url != normalized_url:
            raise ReleaseCalendarValidationError(
                "release_calendar_source_url_not_normalized"
            )
        status_code = source.get("http_status")
        if isinstance(status_code, bool) or status_code != 200:
            raise ReleaseCalendarValidationError("release_calendar_http_status_invalid")
        source_captured_at = _utc_timestamp(
            source.get("captured_at"), "release_calendar_source_clock_invalid"
        )
        if parse_timestamp(source_captured_at, field_name="captured_at") > captured_clock:
            raise ReleaseCalendarValidationError("release_calendar_source_after_capture")
        raw_path = _safe_relative(
            source.get("raw_path"), "release_calendar_raw_path_unsafe"
        )
        if raw_path in declared_paths:
            raise ReleaseCalendarValidationError("release_calendar_raw_path_duplicate")
        declared_paths.add(raw_path)
        path = disk_files.get(raw_path)
        if path is None:
            raise ReleaseCalendarValidationError("release_calendar_raw_file_missing")
        expected_raw_sha = _required_sha256(
            source.get("raw_sha256"), "release_calendar_raw_sha_invalid"
        )
        size = source.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= _MAX_RAW_BYTES:
            raise ReleaseCalendarValidationError("release_calendar_raw_size_invalid")
        body, signature = _stable_file_bytes(
            path,
            blocker="release_calendar_raw_file_unsafe",
            max_bytes=_MAX_RAW_BYTES,
        )
        if len(body) != size or _sha256_bytes(body) != expected_raw_sha:
            raise ReleaseCalendarValidationError("release_calendar_raw_binding_mismatch")
        content_sha = _required_sha256(
            source.get("content_sha256"), "release_calendar_content_sha_invalid"
        )
        parsed_payload: Any = None
        if artifact_kind == "observation":
            parsed_payload = _observation_from_source(body)
            if content_sha != parsed_payload.content_hash:
                raise ReleaseCalendarValidationError(
                    "release_calendar_observation_content_sha_mismatch"
                )
        elif artifact_kind == "parser_contract":
            parsed_payload = _strict_json(
                body, "release_calendar_parser_contract_json_invalid"
            )
            _exact_keys(
                parsed_payload,
                {"schema_version", "parser_id", "parser_contract_sha256"},
                "release_calendar_parser_contract_shape_invalid",
            )
            if parsed_payload.get("schema_version") != "macro-parser-lineage.v1":
                raise ReleaseCalendarValidationError(
                    "release_calendar_parser_contract_schema_invalid"
                )
            parser_id = str(parsed_payload.get("parser_id") or "")
            parser_hash = _required_sha256(
                parsed_payload.get("parser_contract_sha256"),
                "release_calendar_parser_contract_sha_invalid",
            )
            if PARSER_CONTRACT_SHA256.get(parser_id) != parser_hash or content_sha != parser_hash:
                raise ReleaseCalendarValidationError(
                    "release_calendar_parser_contract_drift"
                )
        elif artifact_kind in {"coverage_receipt", "official_bundle"}:
            parsed_payload = _strict_json(
                body, f"release_calendar_{artifact_kind}_json_invalid"
            )
            if content_sha != expected_raw_sha:
                raise ReleaseCalendarValidationError(
                    "release_calendar_artifact_content_sha_mismatch"
                )
        elif content_sha != expected_raw_sha:
            raise ReleaseCalendarValidationError(
                "release_calendar_artifact_content_sha_mismatch"
            )
        ref = SourceArtifactRef(
            source_id=source_id,
            issuer=issuer,
            artifact_kind=artifact_kind,
            source_url=normalized_url,
            http_status=200,
            captured_at=source_captured_at,
            raw_path=raw_path,
            stored_path=f"raw/{raw_path}",
            raw_sha256=expected_raw_sha,
            size_bytes=size,
            content_sha256=content_sha,
        )
        source_rows.append(ref)
        sources_by_id[source_id] = ref
        source_payloads[source_id] = parsed_payload
        raw_by_path.append((raw_path, body))
        raw_readbacks.append(_Readback(path, body, signature, _MAX_RAW_BYTES))
    if set(disk_files) != declared_paths:
        raise ReleaseCalendarValidationError("release_calendar_raw_file_set_mismatch")

    coverage_rows = _list(
        payload.get("issuer_coverage"), "release_calendar_coverage_not_list"
    )
    coverage: list[IssuerCoverage] = []
    for raw_coverage in coverage_rows:
        item = _mapping(raw_coverage, "release_calendar_coverage_not_object")
        _exact_keys(
            item,
            {"issuer", "through", "source_ids"},
            "release_calendar_coverage_shape_invalid",
        )
        issuer = str(item.get("issuer") or "")
        through = _utc_timestamp(
            item.get("through"), "release_calendar_coverage_clock_invalid"
        )
        if parse_timestamp(through, field_name="through") > captured_clock:
            raise ReleaseCalendarValidationError("release_calendar_coverage_after_capture")
        source_ids = tuple(
            str(value)
            for value in _list(
                item.get("source_ids"), "release_calendar_coverage_sources_not_list"
            )
        )
        if len(source_ids) != 1 or len(set(source_ids)) != 1:
            raise ReleaseCalendarValidationError("release_calendar_coverage_sources_invalid")
        ref = sources_by_id.get(source_ids[0])
        receipt = source_payloads.get(source_ids[0])
        if (
            ref is None
            or ref.issuer != issuer
            or ref.artifact_kind != "coverage_receipt"
            or not isinstance(receipt, Mapping)
        ):
            raise ReleaseCalendarValidationError("release_calendar_coverage_source_mismatch")
        _exact_keys(
            receipt,
            {"schema_version", "issuer", "through"},
            "release_calendar_coverage_receipt_shape_invalid",
        )
        if (
            receipt.get("schema_version") != "macro-release-issuer-coverage.v1"
            or receipt.get("issuer") != issuer
            or _utc_timestamp(
                receipt.get("through"), "release_calendar_coverage_receipt_clock_invalid"
            )
            != through
            or parse_timestamp(ref.captured_at, field_name="captured_at")
            < parse_timestamp(through, field_name="through")
        ):
            raise ReleaseCalendarValidationError("release_calendar_coverage_receipt_mismatch")
        coverage.append(IssuerCoverage(issuer, through, source_ids))
    if [item.issuer for item in coverage] != ["nbs_official", "pbc_official"]:
        raise ReleaseCalendarValidationError("release_calendar_issuer_coverage_incomplete")

    raw_capture_events = _list(
        payload.get("events"), "release_calendar_capture_events_not_list"
    )
    if len(raw_capture_events) != len(planned_events):
        raise ReleaseCalendarValidationError("release_calendar_capture_event_count_mismatch")
    events: list[ReleaseEvent] = []
    event_by_id: dict[str, ReleaseEvent] = {}
    for index, (raw_event, plan) in enumerate(zip(raw_capture_events, planned_events)):
        item = _mapping(raw_event, "release_calendar_capture_event_not_object")
        _exact_keys(
            item,
            {
                "event_id",
                "status",
                "actual_at",
                "rescheduled_at",
                "cancelled_at",
                "supersedes_event_id",
                "source_ids",
                "resolution_ids",
            },
            "release_calendar_capture_event_shape_invalid",
        )
        event_id = str(item.get("event_id") or "")
        if event_id != plan["event_id"]:
            raise ReleaseCalendarValidationError("release_calendar_capture_event_order_mismatch")
        status_value = str(item.get("status") or "")
        if status_value not in _EVENT_STATUSES:
            raise ReleaseCalendarValidationError("release_calendar_event_status_unknown")
        actual_at = _utc_timestamp(
            item.get("actual_at"),
            "release_calendar_event_actual_invalid",
            allow_empty=True,
        )
        cancelled_at = _utc_timestamp(
            item.get("cancelled_at"),
            "release_calendar_event_cancelled_invalid",
            allow_empty=True,
        )
        raw_rescheduled = str(item.get("rescheduled_at") or "")
        if raw_rescheduled:
            reschedule_kind, rescheduled_at = _schedule(
                raw_rescheduled, "release_calendar_event_reschedule_invalid"
            )
        else:
            reschedule_kind, rescheduled_at = "", ""
        supersedes = str(item.get("supersedes_event_id") or "")
        if supersedes:
            if supersedes not in event_by_id:
                raise ReleaseCalendarValidationError(
                    "release_calendar_supersedes_unknown_or_future"
                )
            prior = event_by_id[supersedes]
            if (
                prior.event_family != plan["event_family"]
                or prior.issuer != plan["issuer"]
                or prior.period != plan["period"]
                or prior.indicator_ids != plan["indicator_ids"]
            ):
                raise ReleaseCalendarValidationError(
                    "release_calendar_supersedes_scope_mismatch"
                )
        source_ids = tuple(
            str(value)
            for value in _list(
                item.get("source_ids"), "release_calendar_event_sources_not_list"
            )
        )
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ReleaseCalendarValidationError("release_calendar_event_sources_invalid")
        event_sources = [sources_by_id.get(source_id) for source_id in source_ids]
        if any(
            source is None or source.issuer != plan["issuer"]
            for source in event_sources
        ) or sum(source.artifact_kind == "release_notice" for source in event_sources if source) != 1:
            raise ReleaseCalendarValidationError("release_calendar_event_source_mismatch")
        resolution_ids = tuple(
            str(value)
            for value in _list(
                item.get("resolution_ids"),
                "release_calendar_event_resolutions_not_list",
            )
        )
        if len(resolution_ids) != len(set(resolution_ids)):
            raise ReleaseCalendarValidationError("release_calendar_resolution_id_duplicate")
        if status_value == "scheduled":
            valid_shape = not any(
                (actual_at, rescheduled_at, cancelled_at, resolution_ids)
            )
        elif status_value == "released":
            valid_shape = bool(actual_at) and not any((rescheduled_at, cancelled_at))
        elif status_value == "rescheduled":
            valid_shape = bool(rescheduled_at) and not any(
                (actual_at, cancelled_at, resolution_ids)
            )
        else:
            valid_shape = bool(cancelled_at) and not any(
                (actual_at, rescheduled_at, resolution_ids)
            )
        if not valid_shape:
            raise ReleaseCalendarValidationError("release_calendar_event_status_shape_invalid")
        terminal_clock = actual_at or cancelled_at
        if terminal_clock and parse_timestamp(
            terminal_clock, field_name="event_clock"
        ) > max(
            parse_timestamp(source.captured_at, field_name="captured_at")
            for source in event_sources
            if source is not None
        ):
            raise ReleaseCalendarValidationError("release_calendar_event_clock_not_captured")
        event = ReleaseEvent(
            event_id=event_id,
            event_family=str(plan["event_family"]),
            issuer=str(plan["issuer"]),
            indicator_ids=tuple(plan["indicator_ids"]),
            period=str(plan["period"]),
            schedule_kind=str(plan["schedule_kind"]),
            scheduled_at=str(plan["scheduled_at"]),
            status=status_value,
            actual_at=actual_at,
            rescheduled_at=rescheduled_at,
            reschedule_kind=reschedule_kind,
            cancelled_at=cancelled_at,
            supersedes_event_id=supersedes,
            source_ids=source_ids,
            resolution_ids=resolution_ids,
        )
        events.append(event)
        event_by_id[event_id] = event
    superseded_counts: dict[str, int] = {}
    roots_by_scope: dict[tuple[str, str], int] = {}
    for event in events:
        if event.supersedes_event_id:
            superseded_counts[event.supersedes_event_id] = (
                superseded_counts.get(event.supersedes_event_id, 0) + 1
            )
        else:
            key = (event.event_family, event.period)
            roots_by_scope[key] = roots_by_scope.get(key, 0) + 1
    if any(value > 1 for value in superseded_counts.values()) or any(
        value > 1 for value in roots_by_scope.values()
    ):
        raise ReleaseCalendarValidationError("release_calendar_event_conflict")

    raw_resolutions = _list(
        payload.get("resolutions"), "release_calendar_resolutions_not_list"
    )
    resolutions: list[ReleaseResolution] = []
    resolution_by_id: dict[str, ReleaseResolution] = {}
    for raw_resolution in raw_resolutions:
        item = _mapping(raw_resolution, "release_calendar_resolution_not_object")
        _exact_keys(
            item,
            {
                "resolution_id",
                "event_id",
                "indicator_id",
                "period_end",
                "frequency",
                "unit",
                "measurement_basis",
                "value_decimal",
                "issuer",
                "parser_id",
                "parser_contract_sha256",
                "official_bundle_sha256",
                "observation_content_hash",
                "observation_available_at",
                "source_ids",
            },
            "release_calendar_resolution_shape_invalid",
        )
        resolution_id = _safe_id(
            item.get("resolution_id"), "release_calendar_resolution_id_invalid"
        )
        if resolution_id in resolution_by_id:
            raise ReleaseCalendarValidationError("release_calendar_resolution_id_duplicate")
        event_id = str(item.get("event_id") or "")
        event = event_by_id.get(event_id)
        if event is None or event.status != "released":
            raise ReleaseCalendarValidationError("release_calendar_resolution_event_invalid")
        indicator_id = str(item.get("indicator_id") or "")
        rule = _POLICY_BY_ID.get(indicator_id)
        if rule is None:
            raise ReleaseCalendarValidationError("release_calendar_indicator_unknown")
        if indicator_id not in event.indicator_ids:
            raise ReleaseCalendarValidationError("release_calendar_resolution_scope_mismatch")
        expected_period_end = _period_end(event.period, rule.frequency)
        if str(item.get("period_end") or "") != expected_period_end:
            raise ReleaseCalendarValidationError("release_calendar_resolution_period_mismatch")
        frequency = str(item.get("frequency") or "")
        unit = str(item.get("unit") or "")
        basis = str(item.get("measurement_basis") or "")
        if (
            frequency != rule.frequency
            or unit != rule.unit
            or basis != rule.measurement_basis
        ):
            raise ReleaseCalendarValidationError(
                "release_calendar_resolution_equivalence_mismatch"
            )
        decimal_value = _canonical_decimal(
            item.get("value_decimal"), "release_calendar_resolution_decimal_invalid"
        )
        issuer = str(item.get("issuer") or "")
        if issuer != rule.observation_issuer:
            raise ReleaseCalendarValidationError("release_calendar_observation_issuer_mismatch")
        parser_id = str(item.get("parser_id") or "")
        parser_hash = _required_sha256(
            item.get("parser_contract_sha256"),
            "release_calendar_resolution_parser_sha_invalid",
        )
        if (parser_id, parser_hash) not in rule.allowed_parsers:
            raise ReleaseCalendarValidationError("release_calendar_resolution_parser_drift")
        bundle_hash = _required_sha256(
            item.get("official_bundle_sha256"),
            "release_calendar_resolution_bundle_sha_invalid",
        )
        observation_hash = _required_sha256(
            item.get("observation_content_hash"),
            "release_calendar_resolution_observation_sha_invalid",
        )
        observation_available_at = _utc_timestamp(
            item.get("observation_available_at"),
            "release_calendar_resolution_available_at_invalid",
        )
        source_ids = tuple(
            str(value)
            for value in _list(
                item.get("source_ids"), "release_calendar_resolution_sources_not_list"
            )
        )
        if len(source_ids) != 3 or len(set(source_ids)) != 3:
            raise ReleaseCalendarValidationError("release_calendar_resolution_sources_invalid")
        resolution_sources = [sources_by_id.get(source_id) for source_id in source_ids]
        if any(source is None or source.issuer != event.issuer for source in resolution_sources):
            raise ReleaseCalendarValidationError("release_calendar_resolution_source_mismatch")
        by_kind = {
            source.artifact_kind: source
            for source in resolution_sources
            if source is not None
        }
        if set(by_kind) != {"official_bundle", "parser_contract", "observation"}:
            raise ReleaseCalendarValidationError("release_calendar_resolution_source_roles_invalid")
        if (
            by_kind["official_bundle"].content_sha256 != bundle_hash
            or by_kind["parser_contract"].content_sha256 != parser_hash
            or by_kind["observation"].content_sha256 != observation_hash
        ):
            raise ReleaseCalendarValidationError("release_calendar_resolution_hash_binding_mismatch")
        parser_payload = source_payloads[by_kind["parser_contract"].source_id]
        observation = source_payloads[by_kind["observation"].source_id]
        if (
            not isinstance(parser_payload, Mapping)
            or parser_payload.get("parser_id") != parser_id
            or parser_payload.get("parser_contract_sha256") != parser_hash
            or not isinstance(observation, MacroObservation)
        ):
            raise ReleaseCalendarValidationError("release_calendar_resolution_lineage_mismatch")
        if (
            observation.indicator_id != indicator_id
            or observation.period_end != expected_period_end
            or observation.frequency != frequency
            or observation.unit != unit
            or observation.source_system != issuer
            or observation.content_hash != observation_hash
            or Decimal(str(observation.value)) != Decimal(decimal_value)
            or _utc_timestamp(
                observation.available_at,
                "release_calendar_observation_available_at_invalid",
            )
            != observation_available_at
            or _utc_timestamp(
                observation.release_at,
                "release_calendar_observation_release_at_invalid",
            )
            != event.actual_at
            or parse_timestamp(observation_available_at, field_name="available_at")
            < parse_timestamp(event.actual_at, field_name="actual_at")
            or parse_timestamp(
                by_kind["observation"].captured_at, field_name="captured_at"
            )
            < parse_timestamp(observation_available_at, field_name="available_at")
        ):
            raise ReleaseCalendarValidationError(
                "release_calendar_resolution_equivalence_mismatch"
            )
        resolution = ReleaseResolution(
            resolution_id=resolution_id,
            event_id=event_id,
            indicator_id=indicator_id,
            period_end=expected_period_end,
            frequency=frequency,
            unit=unit,
            measurement_basis=basis,
            value_decimal=decimal_value,
            issuer=issuer,
            parser_id=parser_id,
            parser_contract_sha256=parser_hash,
            official_bundle_sha256=bundle_hash,
            observation_content_hash=observation_hash,
            observation_available_at=observation_available_at,
            source_ids=source_ids,
        )
        resolutions.append(resolution)
        resolution_by_id[resolution_id] = resolution
    for event in events:
        event_resolutions = [resolution_by_id.get(item) for item in event.resolution_ids]
        if event.status == "released":
            if any(item is None for item in event_resolutions) or tuple(
                item.indicator_id for item in event_resolutions if item is not None
            ) != event.indicator_ids:
                raise ReleaseCalendarValidationError(
                    "release_calendar_released_event_resolution_incomplete"
                )
            notice = next(
                sources_by_id[source_id]
                for source_id in event.source_ids
                if sources_by_id[source_id].artifact_kind == "release_notice"
            )
            notice_raw = dict(raw_by_path)[notice.raw_path]
            parser_id = event_resolutions[0].parser_id  # type: ignore[union-attr]
            try:
                parsed_notice = parse_official_support_page(
                    parser_id,
                    notice_raw,
                    source_url=notice.source_url,
                )
            except Exception as exc:
                raise ReleaseCalendarValidationError(
                    "release_calendar_release_notice_parse_failed"
                ) from exc
            if (
                parsed_notice.get("release_at") != event.actual_at
                or parsed_notice.get("source_system") != event.issuer
                or parsed_notice.get("parser_contract_sha256")
                != event_resolutions[0].parser_contract_sha256  # type: ignore[union-attr]
            ):
                raise ReleaseCalendarValidationError(
                    "release_calendar_release_notice_lineage_mismatch"
                )
            expected_values = {
                (
                    item.indicator_id,
                    _parser_period(event.period, item.frequency),
                    item.frequency,
                    item.unit,
                    item.measurement_basis,
                    item.value_decimal,
                )
                for item in event_resolutions
                if item is not None
            }
            parsed_values = {
                (
                    str(item.get("indicator_id") or ""),
                    str(item.get("period") or ""),
                    str(item.get("frequency") or ""),
                    str(item.get("unit") or ""),
                    str(item.get("measurement_basis") or ""),
                    str(item.get("value_decimal") or ""),
                )
                for item in _list(
                    parsed_notice.get("values"),
                    "release_calendar_release_notice_values_invalid",
                )
                if str(item.get("indicator_id") or "") in event.indicator_ids
                and str(item.get("period") or "")
                == _parser_period(event.period, event_resolutions[0].frequency)  # type: ignore[union-attr]
            }
            if parsed_values != expected_values:
                raise ReleaseCalendarValidationError(
                    "release_calendar_release_notice_value_mismatch"
                )
        elif event.resolution_ids:
            raise ReleaseCalendarValidationError(
                "release_calendar_nonreleased_event_has_resolution"
            )
    if set(resolution_by_id) != {
        resolution_id for event in events for resolution_id in event.resolution_ids
    }:
        raise ReleaseCalendarValidationError("release_calendar_resolution_unreferenced")
    referenced_sources = {
        source_id for item in coverage for source_id in item.source_ids
    }
    referenced_sources.update(
        source_id for event in events for source_id in event.source_ids
    )
    referenced_sources.update(
        source_id for resolution in resolutions for source_id in resolution.source_ids
    )
    if referenced_sources != set(sources_by_id):
        raise ReleaseCalendarValidationError("release_calendar_source_unreferenced")
    return (
        captured_at,
        tuple(coverage),
        tuple(source_rows),
        tuple(events),
        tuple(resolutions),
        tuple(raw_by_path),
        tuple(raw_readbacks),
    )


def _compile_inputs(
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    capture_manifest_path: str | Path,
    expected_capture_manifest_sha256: str,
    raw_root: str | Path,
    market_open_days_path: str | Path,
    expected_market_open_days_sha256: str,
    exact_mode: int | None = None,
) -> _CompiledInputs:
    plan_file, plan_raw, plan, plan_readback = _read_json_artifact(
        plan_path,
        expected_sha256=expected_plan_sha256,
        blocker="release_calendar_plan",
        exact_mode=exact_mode,
    )
    del plan_file
    planned_events = _validate_plan(plan)
    capture_file, capture_raw, capture, capture_readback = _read_json_artifact(
        capture_manifest_path,
        expected_sha256=expected_capture_manifest_sha256,
        blocker="release_calendar_capture_manifest",
        exact_mode=exact_mode,
    )
    del capture_file
    open_file, open_raw, open_payload, open_readback = _read_json_artifact(
        market_open_days_path,
        expected_sha256=expected_market_open_days_sha256,
        blocker="release_calendar_market_open_days",
        exact_mode=exact_mode,
    )
    del open_file
    root = _absolute_path(raw_root, "release_calendar_raw_root_path_unsafe")
    (
        captured_at,
        coverage,
        sources,
        events,
        resolutions,
        raw_by_path,
        raw_readbacks,
    ) = _validate_capture(
        capture,
        plan_sha256=_sha256_bytes(plan_raw),
        planned_events=planned_events,
        raw_root=root,
    )
    open_dates = _validate_open_days(open_payload)
    content = _CalendarContent(
        plan_sha256=_sha256_bytes(plan_raw),
        capture_manifest_sha256=_sha256_bytes(capture_raw),
        market_open_days_sha256=_sha256_bytes(open_raw),
        captured_at=captured_at,
        open_dates=open_dates,
        issuer_coverage=coverage,
        source_artifacts=sources,
        events=events,
        resolutions=resolutions,
    )
    return _CompiledInputs(
        content=content,
        plan_raw=plan_raw,
        capture_raw=capture_raw,
        open_days_raw=open_raw,
        raw_by_path=raw_by_path,
        readbacks=(
            plan_readback,
            capture_readback,
            open_readback,
            *raw_readbacks,
        ),
        raw_root=root,
    )
