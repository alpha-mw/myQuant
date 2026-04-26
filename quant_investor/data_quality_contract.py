"""Point-in-time data quality contracts and offline assessment helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from quant_investor.versioning import DATA_QUALITY_CONTRACT_SCHEMA_VERSION


DEFAULT_DATA_QUALITY_CONTRACT_DIR = Path("data/data_quality_contract")
DEFAULT_SNAPSHOTS_FILENAME = "snapshots.jsonl"
DEFAULT_ASSESSMENTS_FILENAME = "assessments.jsonl"

ISSUE_SEVERITY_INFO = "info"
ISSUE_SEVERITY_WARNING = "warning"
ISSUE_SEVERITY_BLOCKER = "blocker"

ISSUE_MISSING_REQUIRED_FIELD = "missing_required_field"
ISSUE_MISSING_PROVENANCE = "missing_provenance"
ISSUE_LOOKAHEAD_EFFECTIVE_DATE = "lookahead_effective_date"
ISSUE_LOOKAHEAD_OBSERVED_AT = "lookahead_observed_at"
ISSUE_STALE_FIELD = "stale_field"
ISSUE_OUTLIER_FIELD = "outlier_field"
ISSUE_INVALID_FIELD_VALUE = "invalid_field_value"
ISSUE_UNTRADABLE = "untradable"
ISSUE_LOW_LIQUIDITY = "low_liquidity"

TRADABILITY_SUSPENDED = "suspended"
TRADABILITY_LIMIT_UP = "limit_up"
TRADABILITY_LIMIT_DOWN = "limit_down"
TRADABILITY_ST = "st"
TRADABILITY_DELISTED = "delisted"
TRADABILITY_NO_VALID_PRICE = "no_valid_price"
TRADABILITY_NO_VALID_VOLUME = "no_valid_volume"
TRADABILITY_LOW_LIQUIDITY = "low_liquidity"

ASSESSMENT_POLICY_VERSION = "phase5.offline.v1"

_VALID_SEVERITIES = {
    ISSUE_SEVERITY_INFO,
    ISSUE_SEVERITY_WARNING,
    ISSUE_SEVERITY_BLOCKER,
}
_TRADABILITY_REASON_ORDER = [
    TRADABILITY_SUSPENDED,
    TRADABILITY_LIMIT_UP,
    TRADABILITY_LIMIT_DOWN,
    TRADABILITY_ST,
    TRADABILITY_DELISTED,
    TRADABILITY_NO_VALID_PRICE,
    TRADABILITY_NO_VALID_VOLUME,
    TRADABILITY_LOW_LIQUIDITY,
]
_TRADABILITY_BLOCKING_REASONS = {
    TRADABILITY_SUSPENDED,
    TRADABILITY_DELISTED,
    TRADABILITY_NO_VALID_PRICE,
    TRADABILITY_NO_VALID_VOLUME,
    TRADABILITY_LOW_LIQUIDITY,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_json_serializable(value: Any, label: str) -> Any:
    safe = _json_safe(value)
    try:
        json.dumps(safe, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise ValueError(f"{label} must contain only JSON-serializable values.") from exc
    return safe


def _coerce_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_ensure_json_serializable(value, "metadata"))


def _finite_float(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return number


def _non_negative_float_or_none(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}.")
    return number


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return slug.strip("-") or "unknown"


def _short_hash(parts: Sequence[str]) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _ordered_unique(values: Sequence[str], *, preferred_order: Sequence[str] | None = None) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    if preferred_order is None:
        return sorted(unique)
    order = {value: index for index, value in enumerate(preferred_order)}
    return sorted(unique, key=lambda value: (order.get(value, len(order)), value))


def make_snapshot_id(*, symbol: str, market: str, as_of: str, latest_trade_date: str) -> str:
    parts = [market, symbol, as_of, latest_trade_date]
    return (
        f"pit-snapshot-{_slug(market)}-{_slug(symbol)}-{_slug(as_of)}-"
        f"{_slug(latest_trade_date)}-{_short_hash(parts)}"
    )


def make_issue_id(
    *,
    symbol: str,
    market: str,
    as_of: str,
    issue_type: str,
    field_name: str | None = None,
) -> str:
    field = field_name or "global"
    parts = [market, symbol, as_of, issue_type, field]
    return f"dq-issue-{_slug(market)}-{_slug(symbol)}-{_slug(issue_type)}-{_slug(field)}-{_short_hash(parts)}"


def make_assessment_id(*, snapshot_id: str) -> str:
    return f"dq-assessment-{_slug(snapshot_id)}-{_short_hash([snapshot_id])}"


def parse_iso_date_or_datetime(value: str) -> datetime:
    stripped = str(value).strip()
    if not stripped:
        raise ValueError("ISO date/datetime value must be non-empty.")
    normalized = stripped.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date/datetime value: {value!r}.") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def iso_value_after(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return parse_iso_date_or_datetime(left) > parse_iso_date_or_datetime(right)


def days_between(left: str, right: str) -> int:
    """Return signed day difference as ``right.date() - left.date()``."""

    return (parse_iso_date_or_datetime(right).date() - parse_iso_date_or_datetime(left).date()).days


@dataclass
class FieldProvenance:
    schema_version: str = DATA_QUALITY_CONTRACT_SCHEMA_VERSION
    field_name: str = ""
    source: str = ""
    as_of: str = ""
    effective_date: str | None = None
    observed_at: str | None = None
    revision_id: str | None = None
    adjustment_flag: str | None = None
    is_point_in_time: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FieldProvenance":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", DATA_QUALITY_CONTRACT_SCHEMA_VERSION)),
            field_name=str(data.get("field_name", "")),
            source=str(data.get("source", "")),
            as_of=str(data.get("as_of", "")),
            effective_date=data.get("effective_date"),
            observed_at=data.get("observed_at"),
            revision_id=data.get("revision_id"),
            adjustment_flag=data.get("adjustment_flag"),
            is_point_in_time=bool(data.get("is_point_in_time", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class DataQualityIssue:
    schema_version: str = DATA_QUALITY_CONTRACT_SCHEMA_VERSION
    issue_id: str = ""
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    field_name: str | None = None
    issue_type: str = ""
    severity: str = ISSUE_SEVERITY_WARNING
    message: str = ""
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_VALID_SEVERITIES)}; got {self.severity!r}.")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataQualityIssue":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", DATA_QUALITY_CONTRACT_SCHEMA_VERSION)),
            issue_id=str(data.get("issue_id", "")),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            field_name=data.get("field_name"),
            issue_type=str(data.get("issue_type", "")),
            severity=str(data.get("severity", ISSUE_SEVERITY_WARNING)),
            message=str(data.get("message", "")),
            source=data.get("source"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class TradabilityStatus:
    schema_version: str = DATA_QUALITY_CONTRACT_SCHEMA_VERSION
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    latest_trade_date: str = ""
    is_tradable: bool = True
    is_suspended: bool = False
    is_limit_up: bool = False
    is_limit_down: bool = False
    is_st: bool = False
    is_delisted: bool = False
    has_valid_price: bool = True
    has_valid_volume: bool = True
    liquidity_score: float | None = None
    adv: float | None = None
    max_order_value: float | None = None
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.liquidity_score = _non_negative_float_or_none(self.liquidity_score, "liquidity_score")
        if self.liquidity_score is not None and self.liquidity_score > 1.0:
            raise ValueError("liquidity_score must be in [0, 1].")
        self.adv = _non_negative_float_or_none(self.adv, "adv")
        self.max_order_value = _non_negative_float_or_none(self.max_order_value, "max_order_value")
        self.reasons = _ordered_unique(list(self.reasons), preferred_order=_TRADABILITY_REASON_ORDER)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradabilityStatus":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", DATA_QUALITY_CONTRACT_SCHEMA_VERSION)),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            latest_trade_date=str(data.get("latest_trade_date", "")),
            is_tradable=bool(data.get("is_tradable", True)),
            is_suspended=bool(data.get("is_suspended", False)),
            is_limit_up=bool(data.get("is_limit_up", False)),
            is_limit_down=bool(data.get("is_limit_down", False)),
            is_st=bool(data.get("is_st", False)),
            is_delisted=bool(data.get("is_delisted", False)),
            has_valid_price=bool(data.get("has_valid_price", True)),
            has_valid_volume=bool(data.get("has_valid_volume", True)),
            liquidity_score=data.get("liquidity_score"),
            adv=data.get("adv"),
            max_order_value=data.get("max_order_value"),
            reasons=list(data.get("reasons", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class PointInTimeSnapshot:
    schema_version: str = DATA_QUALITY_CONTRACT_SCHEMA_VERSION
    snapshot_id: str = ""
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    latest_trade_date: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, FieldProvenance] = field(default_factory=dict)
    quality_issues: list[DataQualityIssue] = field(default_factory=list)
    tradability_status: TradabilityStatus | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fields = dict(_ensure_json_serializable(self.fields, "fields"))
        self.provenance = {
            str(key): value if isinstance(value, FieldProvenance) else FieldProvenance.from_dict(value)
            for key, value in self.provenance.items()
        }
        self.quality_issues = [
            value if isinstance(value, DataQualityIssue) else DataQualityIssue.from_dict(value)
            for value in self.quality_issues
        ]
        if self.tradability_status is not None and not isinstance(self.tradability_status, TradabilityStatus):
            self.tradability_status = TradabilityStatus.from_dict(self.tradability_status)
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "market": self.market,
            "as_of": self.as_of,
            "latest_trade_date": self.latest_trade_date,
            "fields": _json_safe(self.fields),
            "provenance": {key: value.to_dict() for key, value in self.provenance.items()},
            "quality_issues": [issue.to_dict() for issue in self.quality_issues],
            "tradability_status": self.tradability_status.to_dict() if self.tradability_status else None,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PointInTimeSnapshot":
        data = dict(payload)
        provenance_payload = dict(data.get("provenance", {}) or {})
        return cls(
            schema_version=str(data.get("schema_version", DATA_QUALITY_CONTRACT_SCHEMA_VERSION)),
            snapshot_id=str(data.get("snapshot_id", "")),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            latest_trade_date=str(data.get("latest_trade_date", "")),
            fields=dict(data.get("fields", {}) or {}),
            provenance={
                str(key): FieldProvenance.from_dict(value)
                for key, value in provenance_payload.items()
                if isinstance(value, Mapping)
            },
            quality_issues=[
                DataQualityIssue.from_dict(issue)
                for issue in list(data.get("quality_issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            tradability_status=(
                TradabilityStatus.from_dict(data["tradability_status"])
                if isinstance(data.get("tradability_status"), Mapping)
                else None
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class DataQualityAssessment:
    schema_version: str = DATA_QUALITY_CONTRACT_SCHEMA_VERSION
    assessment_id: str = ""
    snapshot_id: str = ""
    symbol: str = ""
    market: str = ""
    as_of: str = ""
    is_researchable: bool = True
    is_tradable: bool = True
    quarantine: bool = False
    quarantine_reasons: list[str] = field(default_factory=list)
    issue_count: int = 0
    blocker_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    data_quality_score: float = 1.0
    tradability_reasons: list[str] = field(default_factory=list)
    issues: list[DataQualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.issues = [
            value if isinstance(value, DataQualityIssue) else DataQualityIssue.from_dict(value)
            for value in self.issues
        ]
        self.data_quality_score = _clamp_unit(_finite_float(self.data_quality_score, "data_quality_score"))
        if self.blocker_count > 0:
            self.quarantine = True
        if self.quarantine:
            self.is_researchable = False
        self.quarantine_reasons = _ordered_unique(list(self.quarantine_reasons))
        self.tradability_reasons = _ordered_unique(
            list(self.tradability_reasons),
            preferred_order=_TRADABILITY_REASON_ORDER,
        )
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "market": self.market,
            "as_of": self.as_of,
            "is_researchable": self.is_researchable,
            "is_tradable": self.is_tradable,
            "quarantine": self.quarantine,
            "quarantine_reasons": list(self.quarantine_reasons),
            "issue_count": self.issue_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "data_quality_score": self.data_quality_score,
            "tradability_reasons": list(self.tradability_reasons),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataQualityAssessment":
        data = dict(payload)
        return cls(
            schema_version=str(data.get("schema_version", DATA_QUALITY_CONTRACT_SCHEMA_VERSION)),
            assessment_id=str(data.get("assessment_id", "")),
            snapshot_id=str(data.get("snapshot_id", "")),
            symbol=str(data.get("symbol", "")),
            market=str(data.get("market", "")),
            as_of=str(data.get("as_of", "")),
            is_researchable=bool(data.get("is_researchable", True)),
            is_tradable=bool(data.get("is_tradable", True)),
            quarantine=bool(data.get("quarantine", False)),
            quarantine_reasons=list(data.get("quarantine_reasons", []) or []),
            issue_count=int(data.get("issue_count", 0) or 0),
            blocker_count=int(data.get("blocker_count", 0) or 0),
            warning_count=int(data.get("warning_count", 0) or 0),
            info_count=int(data.get("info_count", 0) or 0),
            data_quality_score=float(data.get("data_quality_score", 1.0)),
            tradability_reasons=list(data.get("tradability_reasons", []) or []),
            issues=[
                DataQualityIssue.from_dict(issue)
                for issue in list(data.get("issues", []) or [])
                if isinstance(issue, Mapping)
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )


def build_tradability_status(
    *,
    symbol: str,
    market: str,
    as_of: str,
    latest_trade_date: str,
    is_suspended: bool = False,
    is_limit_up: bool = False,
    is_limit_down: bool = False,
    is_st: bool = False,
    is_delisted: bool = False,
    has_valid_price: bool = True,
    has_valid_volume: bool = True,
    liquidity_score: float | None = None,
    adv: float | None = None,
    max_order_value: float | None = None,
    min_liquidity_score: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TradabilityStatus:
    """Build tradability status; limit-up/down and ST are reasons but not blockers."""

    resolved_liquidity_score = _non_negative_float_or_none(liquidity_score, "liquidity_score")
    if resolved_liquidity_score is not None and resolved_liquidity_score > 1.0:
        raise ValueError("liquidity_score must be in [0, 1].")
    resolved_adv = _non_negative_float_or_none(adv, "adv")
    resolved_max_order_value = _non_negative_float_or_none(max_order_value, "max_order_value")
    resolved_min_liquidity = _non_negative_float_or_none(min_liquidity_score, "min_liquidity_score")

    reasons: list[str] = []
    if is_suspended:
        reasons.append(TRADABILITY_SUSPENDED)
    if is_limit_up:
        reasons.append(TRADABILITY_LIMIT_UP)
    if is_limit_down:
        reasons.append(TRADABILITY_LIMIT_DOWN)
    if is_st:
        reasons.append(TRADABILITY_ST)
    if is_delisted:
        reasons.append(TRADABILITY_DELISTED)
    if not has_valid_price:
        reasons.append(TRADABILITY_NO_VALID_PRICE)
    if not has_valid_volume:
        reasons.append(TRADABILITY_NO_VALID_VOLUME)
    if (
        resolved_min_liquidity is not None
        and resolved_liquidity_score is not None
        and resolved_liquidity_score < resolved_min_liquidity
    ):
        reasons.append(TRADABILITY_LOW_LIQUIDITY)
    ordered_reasons = _ordered_unique(reasons, preferred_order=_TRADABILITY_REASON_ORDER)
    is_tradable = not any(reason in _TRADABILITY_BLOCKING_REASONS for reason in ordered_reasons)
    status_metadata = _coerce_metadata(metadata)
    status_metadata.setdefault("blocking_reasons", sorted(_TRADABILITY_BLOCKING_REASONS))
    status_metadata.setdefault("non_blocking_reasons", [TRADABILITY_LIMIT_UP, TRADABILITY_LIMIT_DOWN, TRADABILITY_ST])
    status_metadata.setdefault("min_liquidity_score", resolved_min_liquidity)
    return TradabilityStatus(
        symbol=symbol,
        market=market,
        as_of=as_of,
        latest_trade_date=latest_trade_date,
        is_tradable=is_tradable,
        is_suspended=is_suspended,
        is_limit_up=is_limit_up,
        is_limit_down=is_limit_down,
        is_st=is_st,
        is_delisted=is_delisted,
        has_valid_price=has_valid_price,
        has_valid_volume=has_valid_volume,
        liquidity_score=resolved_liquidity_score,
        adv=resolved_adv,
        max_order_value=resolved_max_order_value,
        reasons=ordered_reasons,
        metadata=status_metadata,
    )


def _issue(
    snapshot: PointInTimeSnapshot,
    *,
    issue_type: str,
    severity: str,
    message: str,
    field_name: str | None = None,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DataQualityIssue:
    return DataQualityIssue(
        issue_id=make_issue_id(
            symbol=snapshot.symbol,
            market=snapshot.market,
            as_of=snapshot.as_of,
            issue_type=issue_type,
            field_name=field_name,
        ),
        symbol=snapshot.symbol,
        market=snapshot.market,
        as_of=snapshot.as_of,
        field_name=field_name,
        issue_type=issue_type,
        severity=severity,
        message=message,
        source=source,
        metadata=_coerce_metadata(metadata),
    )


def _required_field_missing(fields: Mapping[str, Any], field_name: str) -> bool:
    if field_name not in fields:
        return True
    value = fields[field_name]
    return value is None or value == ""


def generate_data_quality_issues(
    snapshot: PointInTimeSnapshot,
    *,
    required_fields: Sequence[str] | None = None,
    freshness_rules_days: Mapping[str, int] | None = None,
    outlier_flags: Mapping[str, bool] | None = None,
) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = list(snapshot.quality_issues)
    fields = snapshot.fields
    required = list(required_fields or [])
    freshness_rules = dict(freshness_rules_days or {})
    outliers = dict(outlier_flags or {})

    for field_name in required:
        if _required_field_missing(fields, field_name):
            issues.append(
                _issue(
                    snapshot,
                    issue_type=ISSUE_MISSING_REQUIRED_FIELD,
                    severity=ISSUE_SEVERITY_BLOCKER,
                    field_name=field_name,
                    message=f"Required field '{field_name}' is missing.",
                )
            )

    for field_name in sorted(fields):
        if field_name not in snapshot.provenance:
            issues.append(
                _issue(
                    snapshot,
                    issue_type=ISSUE_MISSING_PROVENANCE,
                    severity=ISSUE_SEVERITY_WARNING,
                    field_name=field_name,
                    message=f"Field '{field_name}' has no provenance record.",
                )
            )

    for field_name in sorted(snapshot.provenance):
        provenance = snapshot.provenance[field_name]
        if iso_value_after(provenance.effective_date, snapshot.as_of):
            issues.append(
                _issue(
                    snapshot,
                    issue_type=ISSUE_LOOKAHEAD_EFFECTIVE_DATE,
                    severity=ISSUE_SEVERITY_BLOCKER,
                    field_name=field_name,
                    source=provenance.source,
                    message=f"Field '{field_name}' effective_date is after as_of.",
                    metadata={"effective_date": provenance.effective_date},
                )
            )
        if iso_value_after(provenance.observed_at, snapshot.as_of):
            issues.append(
                _issue(
                    snapshot,
                    issue_type=ISSUE_LOOKAHEAD_OBSERVED_AT,
                    severity=ISSUE_SEVERITY_BLOCKER,
                    field_name=field_name,
                    source=provenance.source,
                    message=f"Field '{field_name}' observed_at is after as_of.",
                    metadata={"observed_at": provenance.observed_at},
                )
            )

    for field_name in sorted(freshness_rules):
        if field_name not in snapshot.provenance:
            continue
        provenance = snapshot.provenance[field_name]
        reference_date = provenance.observed_at or provenance.effective_date
        if reference_date is None:
            continue
        age_days = days_between(reference_date, snapshot.as_of)
        if age_days > int(freshness_rules[field_name]):
            issues.append(
                _issue(
                    snapshot,
                    issue_type=ISSUE_STALE_FIELD,
                    severity=ISSUE_SEVERITY_WARNING,
                    field_name=field_name,
                    source=provenance.source,
                    message=f"Field '{field_name}' is stale by freshness rule.",
                    metadata={
                        "age_days": age_days,
                        "max_age_days": int(freshness_rules[field_name]),
                        "reference_date": reference_date,
                    },
                )
            )

    for field_name in sorted(outliers):
        if outliers[field_name]:
            issues.append(
                _issue(
                    snapshot,
                    issue_type=ISSUE_OUTLIER_FIELD,
                    severity=ISSUE_SEVERITY_WARNING,
                    field_name=field_name,
                    message=f"Field '{field_name}' is flagged as an outlier.",
                )
            )

    if snapshot.tradability_status is not None and not snapshot.tradability_status.is_tradable:
        issues.append(
            _issue(
                snapshot,
                issue_type=ISSUE_UNTRADABLE,
                severity=ISSUE_SEVERITY_BLOCKER,
                message="Symbol is not tradable under point-in-time tradability status.",
                source="tradability_status",
                metadata={"reasons": list(snapshot.tradability_status.reasons)},
            )
        )

    deduped: list[DataQualityIssue] = []
    seen_issue_ids: set[str] = set()
    for issue in issues:
        if issue.issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue.issue_id)
        deduped.append(issue)
    return deduped


def assess_point_in_time_snapshot(
    snapshot: PointInTimeSnapshot,
    *,
    required_fields: Sequence[str] | None = None,
    freshness_rules_days: Mapping[str, int] | None = None,
    outlier_flags: Mapping[str, bool] | None = None,
) -> DataQualityAssessment:
    issues = generate_data_quality_issues(
        snapshot,
        required_fields=required_fields,
        freshness_rules_days=freshness_rules_days,
        outlier_flags=outlier_flags,
    )
    blocker_count = sum(1 for issue in issues if issue.severity == ISSUE_SEVERITY_BLOCKER)
    warning_count = sum(1 for issue in issues if issue.severity == ISSUE_SEVERITY_WARNING)
    info_count = sum(1 for issue in issues if issue.severity == ISSUE_SEVERITY_INFO)
    score = _clamp_unit(1.0 - blocker_count * 0.40 - warning_count * 0.15 - info_count * 0.05)
    quarantine = blocker_count > 0
    quarantine_reasons = _ordered_unique(
        [
            f"{issue.issue_type}:{issue.message}"
            for issue in issues
            if issue.severity == ISSUE_SEVERITY_BLOCKER
        ]
    )
    tradability_reasons = (
        list(snapshot.tradability_status.reasons)
        if snapshot.tradability_status is not None
        else []
    )
    is_tradable = snapshot.tradability_status.is_tradable if snapshot.tradability_status is not None else True
    metadata = {
        "data_quality_contract_schema_version": DATA_QUALITY_CONTRACT_SCHEMA_VERSION,
        "required_fields": list(required_fields or []),
        "freshness_rules_days": dict(freshness_rules_days or {}),
        "outlier_flag_fields": sorted([field for field, flagged in dict(outlier_flags or {}).items() if flagged]),
        "assessment_policy_version": ASSESSMENT_POLICY_VERSION,
    }
    return DataQualityAssessment(
        assessment_id=make_assessment_id(snapshot_id=snapshot.snapshot_id),
        snapshot_id=snapshot.snapshot_id,
        symbol=snapshot.symbol,
        market=snapshot.market,
        as_of=snapshot.as_of,
        is_researchable=not quarantine,
        is_tradable=is_tradable,
        quarantine=quarantine,
        quarantine_reasons=quarantine_reasons,
        issue_count=len(issues),
        blocker_count=blocker_count,
        warning_count=warning_count,
        info_count=info_count,
        data_quality_score=score,
        tradability_reasons=tradability_reasons,
        issues=issues,
        metadata=metadata,
    )


def build_global_context_quality_patch(
    assessments: Sequence[DataQualityAssessment],
) -> dict[str, Any]:
    ordered = sorted(assessments, key=lambda assessment: (assessment.symbol, assessment.market, assessment.as_of))
    quarantine_symbols = _ordered_unique([assessment.symbol for assessment in ordered if assessment.quarantine])
    blocked_symbols = _ordered_unique([assessment.symbol for assessment in ordered if not assessment.is_tradable])
    symbol_quality_scores = {assessment.symbol: assessment.data_quality_score for assessment in ordered}
    symbol_issue_counts = {assessment.symbol: assessment.issue_count for assessment in ordered}
    quarantine_reasons = {
        assessment.symbol: list(assessment.quarantine_reasons)
        for assessment in ordered
        if assessment.quarantine_reasons
    }
    tradability_reasons = {
        assessment.symbol: list(assessment.tradability_reasons)
        for assessment in ordered
        if assessment.tradability_reasons
    }
    payload = {
        "data_quality_quarantine": quarantine_symbols,
        "tradability_blocked_symbols": blocked_symbols,
        "symbol_quality_scores": symbol_quality_scores,
        "symbol_issue_counts": symbol_issue_counts,
        "quarantine_reasons": quarantine_reasons,
        "tradability_reasons": tradability_reasons,
        "metadata": {
            "data_quality_contract_schema_version": DATA_QUALITY_CONTRACT_SCHEMA_VERSION,
            "total_assessments": len(ordered),
            "quarantine_count": len(quarantine_symbols),
            "tradability_blocked_count": len(blocked_symbols),
        },
    }
    return dict(_ensure_json_serializable(payload, "global_context_quality_patch"))


RecordT = TypeVar("RecordT", PointInTimeSnapshot, DataQualityAssessment)


class DataQualityContractStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_DATA_QUALITY_CONTRACT_DIR
        self.snapshots_path = self.root_dir / DEFAULT_SNAPSHOTS_FILENAME
        self.assessments_path = self.root_dir / DEFAULT_ASSESSMENTS_FILENAME

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(_json_safe(payload)), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _read_jsonl(self, path: Path, record_cls: type[RecordT]) -> list[RecordT]:
        if not path.exists():
            return []
        records: list[RecordT] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} line {line_number}: {exc.msg}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Expected JSON object in {path} line {line_number}.")
                records.append(record_cls.from_dict(payload))
        return records

    def append_snapshot(self, snapshot: PointInTimeSnapshot) -> None:
        if snapshot.snapshot_id in self.get_snapshot_ids():
            raise ValueError(f"Duplicate snapshot_id in snapshots ledger: {snapshot.snapshot_id}")
        self._append_jsonl(self.snapshots_path, snapshot.to_dict())

    def append_assessment(self, assessment: DataQualityAssessment) -> None:
        if assessment.assessment_id in self.get_assessment_ids():
            raise ValueError(f"Duplicate assessment_id in assessments ledger: {assessment.assessment_id}")
        self._append_jsonl(self.assessments_path, assessment.to_dict())

    def append_assessments(self, assessments: Sequence[DataQualityAssessment]) -> int:
        existing = self.get_assessment_ids()
        seen: set[str] = set()
        for assessment in assessments:
            if assessment.assessment_id in existing or assessment.assessment_id in seen:
                raise ValueError(f"Duplicate assessment_id in assessments ledger: {assessment.assessment_id}")
            seen.add(assessment.assessment_id)
        for assessment in assessments:
            self._append_jsonl(self.assessments_path, assessment.to_dict())
        return len(assessments)

    def read_snapshots(self) -> list[PointInTimeSnapshot]:
        return self._read_jsonl(self.snapshots_path, PointInTimeSnapshot)

    def read_assessments(self) -> list[DataQualityAssessment]:
        return self._read_jsonl(self.assessments_path, DataQualityAssessment)

    def get_snapshot_ids(self) -> set[str]:
        return {snapshot.snapshot_id for snapshot in self.read_snapshots()}

    def get_assessment_ids(self) -> set[str]:
        return {assessment.assessment_id for assessment in self.read_assessments()}


__all__ = [
    "DEFAULT_DATA_QUALITY_CONTRACT_DIR",
    "DEFAULT_SNAPSHOTS_FILENAME",
    "DEFAULT_ASSESSMENTS_FILENAME",
    "ISSUE_SEVERITY_INFO",
    "ISSUE_SEVERITY_WARNING",
    "ISSUE_SEVERITY_BLOCKER",
    "ISSUE_MISSING_REQUIRED_FIELD",
    "ISSUE_MISSING_PROVENANCE",
    "ISSUE_LOOKAHEAD_EFFECTIVE_DATE",
    "ISSUE_LOOKAHEAD_OBSERVED_AT",
    "ISSUE_STALE_FIELD",
    "ISSUE_OUTLIER_FIELD",
    "ISSUE_INVALID_FIELD_VALUE",
    "ISSUE_UNTRADABLE",
    "ISSUE_LOW_LIQUIDITY",
    "TRADABILITY_SUSPENDED",
    "TRADABILITY_LIMIT_UP",
    "TRADABILITY_LIMIT_DOWN",
    "TRADABILITY_ST",
    "TRADABILITY_DELISTED",
    "TRADABILITY_NO_VALID_PRICE",
    "TRADABILITY_NO_VALID_VOLUME",
    "TRADABILITY_LOW_LIQUIDITY",
    "FieldProvenance",
    "DataQualityIssue",
    "TradabilityStatus",
    "PointInTimeSnapshot",
    "DataQualityAssessment",
    "DataQualityContractStore",
    "make_snapshot_id",
    "make_issue_id",
    "make_assessment_id",
    "parse_iso_date_or_datetime",
    "iso_value_after",
    "days_between",
    "build_tradability_status",
    "generate_data_quality_issues",
    "assess_point_in_time_snapshot",
    "build_global_context_quality_patch",
]
