from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "theme_membership.v1"
_DATE_FORMATS = ("%Y-%m-%d", "%Y%m%d")


@dataclass(frozen=True)
class ThemeMembership:
    membership_id: str = ""
    theme_id: str = ""
    theme_name: str = ""
    theme_type: str = "concept"
    symbol: str = ""
    symbol_name: str = ""
    effective_from: str = ""
    effective_to: str = ""
    membership_status: str = "active"
    confidence: float = 0.0
    source_type: str = ""
    source_ref: str = ""
    evidence_text: str = ""
    maintainer: str = ""
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ThemeMembership":
        schema_version = _text(value.get("schema_version") or SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={schema_version}")
        membership = cls(
            membership_id=_text(value.get("membership_id")),
            theme_id=_text(value.get("theme_id")),
            theme_name=_text(value.get("theme_name")),
            theme_type=_text(value.get("theme_type") or "concept"),
            symbol=_text(value.get("symbol")).upper(),
            symbol_name=_text(value.get("symbol_name")),
            effective_from=_text(value.get("effective_from")),
            effective_to=_text(value.get("effective_to")),
            membership_status=_text(value.get("membership_status") or "active"),
            confidence=_clamp(value.get("confidence"), default=0.0),
            source_type=_text(value.get("source_type")),
            source_ref=_text(value.get("source_ref")),
            evidence_text=_text(value.get("evidence_text")),
            maintainer=_text(value.get("maintainer")),
            created_at=_text(value.get("created_at")),
            updated_at=_text(value.get("updated_at")),
            tags=_text_list(value.get("tags")),
        )
        if not membership.theme_id:
            raise ValueError("theme_id is required")
        if not membership.symbol:
            raise ValueError("symbol is required")
        if not membership.effective_from:
            raise ValueError("effective_from is required")
        if _parse_date(membership.effective_from) is None:
            raise ValueError("effective_from is invalid")
        if membership.effective_to and _parse_date(membership.effective_to) is None:
            raise ValueError("effective_to is invalid")
        return membership

    def is_active(self, as_of: str | date | datetime) -> bool:
        status = str(self.membership_status or "active").strip().lower()
        if status not in {"active", "included", "valid"}:
            return False
        as_of_date = _parse_date(as_of)
        start = _parse_date(self.effective_from)
        if as_of_date is None or start is None or start > as_of_date:
            return False
        end = _parse_date(self.effective_to)
        if end is not None and end <= as_of_date:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "membership_id": self.membership_id,
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "theme_type": self.theme_type,
            "symbol": self.symbol,
            "symbol_name": self.symbol_name,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "membership_status": self.membership_status,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "evidence_text": self.evidence_text,
            "maintainer": self.maintainer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ThemeMembershipLoadResult:
    memberships: list[ThemeMembership] = field(default_factory=list)
    status: str = "missing"
    diagnostic_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memberships": [membership.to_dict() for membership in self.memberships],
            "status": self.status,
            "diagnostic_notes": list(self.diagnostic_notes),
        }


class ThemeMembershipStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ThemeMembershipLoadResult:
        if not self.path.exists():
            return ThemeMembershipLoadResult(
                status="missing",
                diagnostic_notes=["theme_membership_file_missing"],
            )
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            return ThemeMembershipLoadResult(
                status="error",
                diagnostic_notes=[f"theme_membership_file_read_error: {exc}"],
            )

        memberships: list[ThemeMembership] = []
        try:
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"line {line_number} is not an object")
                memberships.append(ThemeMembership.from_mapping(payload))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return ThemeMembershipLoadResult(
                status="error",
                diagnostic_notes=[f"theme_membership_file_format_error: {exc}"],
            )

        return ThemeMembershipLoadResult(
            memberships=memberships,
            status="success",
            diagnostic_notes=[f"theme_membership_count={len(memberships)}"],
        )


def active_memberships_by_symbol(
    memberships: Sequence[ThemeMembership | Mapping[str, Any]],
    *,
    as_of: str | date | datetime,
) -> dict[str, list[ThemeMembership]]:
    active: dict[tuple[str, str], ThemeMembership] = {}
    for item in list(memberships or []):
        try:
            membership = (
                item
                if isinstance(item, ThemeMembership)
                else ThemeMembership.from_mapping(item)
            )
        except (TypeError, ValueError):
            continue
        if not membership.is_active(as_of):
            continue
        key = (membership.symbol, membership.theme_id)
        previous = active.get(key)
        if previous is None or _dedupe_key(membership) >= _dedupe_key(previous):
            active[key] = membership

    grouped: dict[str, list[ThemeMembership]] = {}
    for membership in sorted(active.values(), key=lambda value: (value.symbol, value.theme_id)):
        grouped.setdefault(membership.symbol, []).append(membership)
    return grouped


def _dedupe_key(membership: ThemeMembership) -> tuple[str, str]:
    return (str(membership.updated_at or ""), str(membership.membership_id or ""))


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text[:8], fmt).date()
        except ValueError:
            continue
    return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value or [])
    except TypeError:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _clamp(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(0.0, min(1.0, numeric))
