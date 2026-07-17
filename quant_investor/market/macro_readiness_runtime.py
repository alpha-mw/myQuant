"""Freeze one release-calendar readiness decision for a market consumer run.

The canonical release-calendar pointer is captured before loading the immutable
generation.  Callers then reuse the returned evidence and cutoff for every
Macro readiness assessment in the same run; this module never falls back to an
unpinned calendar or an inferred freshness decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_investor.macro.contracts import parse_timestamp
from quant_investor.macro.release_calendar import (
    MacroReleaseCalendarError,
    ReleaseCalendarEvidence,
    load_release_calendar,
    release_calendar_pointer_sha256,
)
from quant_investor.market.branch_readiness import (
    MacroReadinessEvidence,
    build_macro_readiness_evidence,
)


MACRO_READINESS_RUNTIME_SCHEMA_VERSION = "macro-readiness-runtime.v1"
DEFAULT_MACRO_RELEASE_CALENDAR_ROOT = Path(
    "data/parquet/cn/macro_release_calendar"
)
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_UTC = timezone.utc


@dataclass(frozen=True)
class FrozenMacroReadinessRuntime:
    """One immutable runtime binding, or one explicit fail-closed blocker."""

    status: str
    macro_logical_date: str
    target_session_date: str
    decision_cutoff_at: str
    release_calendar_root: str
    release_calendar_pointer_sha256: str
    evidence: MacroReadinessEvidence | None
    blocker: str = ""

    @property
    def ready(self) -> bool:
        return (
            self.status == "ready"
            and type(self.evidence) is MacroReadinessEvidence
        )

    def metadata(self) -> dict[str, Any]:
        evidence_payload = (
            self.evidence.to_dict()
            if type(self.evidence) is MacroReadinessEvidence
            else {}
        )
        evidence_sha256 = (
            self.evidence.semantic_sha256
            if type(self.evidence) is MacroReadinessEvidence
            else ""
        )
        return {
            "schema_version": MACRO_READINESS_RUNTIME_SCHEMA_VERSION,
            "status": self.status,
            "macro_logical_date": self.macro_logical_date,
            "target_session_date": self.target_session_date,
            "decision_cutoff_at": self.decision_cutoff_at,
            "release_calendar_root": self.release_calendar_root,
            "release_calendar_pointer_sha256": (
                self.release_calendar_pointer_sha256
            ),
            "macro_readiness_evidence": evidence_payload,
            "macro_readiness_evidence_semantic_sha256": evidence_sha256,
            "blocker": self.blocker,
        }


def _absolute_calendar_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _session_date(value: Any, *, blocker: str) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(blocker)
        return value.astimezone(_SHANGHAI).date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(blocker) from exc


def _runtime_now(value: datetime | None) -> datetime:
    clock = value if value is not None else datetime.now(_UTC)
    if clock.tzinfo is None:
        raise ValueError("macro_readiness_runtime_now_timezone_required")
    return clock.astimezone(_UTC)


def _decision_cutoff(
    release_calendar: ReleaseCalendarEvidence,
    *,
    target_session_date: str,
    now: datetime | None,
) -> str:
    if type(release_calendar) is not ReleaseCalendarEvidence:
        raise TypeError("release_calendar_evidence_exact_type_required")
    if not release_calendar.issuer_coverage:
        raise ValueError("macro_release_calendar_issuer_coverage_missing")
    coverage_clocks = tuple(
        parse_timestamp(item.through_at, field_name="through")
        for item in release_calendar.issuer_coverage
    )
    earliest_coverage = min(coverage_clocks)
    earliest_coverage_date = earliest_coverage.astimezone(_SHANGHAI).date()
    target_date = _session_date(
        target_session_date,
        blocker="macro_target_session_date_invalid",
    )
    target_close = datetime.combine(
        target_date,
        time(15, 0),
        tzinfo=_SHANGHAI,
    ).astimezone(_UTC)
    runtime_now = _runtime_now(now)
    if target_date < earliest_coverage_date:
        if target_close > runtime_now:
            raise ValueError("macro_release_calendar_coverage_in_future")
        cutoff = target_close
    elif target_date == earliest_coverage_date:
        if earliest_coverage < target_close:
            raise ValueError(
                "macro_release_calendar_coverage_before_market_close"
            )
        if earliest_coverage > runtime_now:
            raise ValueError("macro_release_calendar_coverage_in_future")
        cutoff = earliest_coverage
    else:
        raise ValueError("macro_release_calendar_coverage_before_target")
    return cutoff.astimezone(_UTC).isoformat()


def freeze_macro_readiness_runtime(
    *,
    macro_logical_date: str,
    target_session_date: str,
    calendar_root: str | Path = DEFAULT_MACRO_RELEASE_CALENDAR_ROOT,
    now: datetime | None = None,
) -> FrozenMacroReadinessRuntime:
    """Capture/load one stable calendar and build one frozen Macro evidence.

    Expected canonical-data failures are returned as an unavailable binding so
    consumers can emit a blocked readiness report.  They are not retried and
    no alternative root, cutoff, or calendar generation is consulted.
    """

    root = _absolute_calendar_root(calendar_root)
    pointer_sha256 = ""
    decision_cutoff_at = ""
    try:
        pointer_sha256 = release_calendar_pointer_sha256(
            canonical_root=root
        )
        release_calendar = load_release_calendar(
            canonical_root=root,
            expected_pointer_sha256=pointer_sha256,
        )
        decision_cutoff_at = _decision_cutoff(
            release_calendar,
            target_session_date=target_session_date,
            now=now,
        )
        evidence = build_macro_readiness_evidence(
            release_calendar_evidence=release_calendar,
            macro_logical_date=macro_logical_date,
            target_session_date=target_session_date,
            target_decision_cutoff_at=decision_cutoff_at,
        )
    except (MacroReleaseCalendarError, OSError, TypeError, ValueError) as exc:
        return FrozenMacroReadinessRuntime(
            status="blocked",
            macro_logical_date=str(macro_logical_date or ""),
            target_session_date=str(target_session_date or ""),
            decision_cutoff_at=decision_cutoff_at,
            release_calendar_root=str(root),
            release_calendar_pointer_sha256=pointer_sha256,
            evidence=None,
            blocker=str(exc) or type(exc).__name__,
        )
    return FrozenMacroReadinessRuntime(
        status="ready",
        macro_logical_date=evidence.macro_logical_date,
        target_session_date=evidence.target_session_date,
        decision_cutoff_at=decision_cutoff_at,
        release_calendar_root=str(root),
        release_calendar_pointer_sha256=pointer_sha256,
        evidence=evidence,
    )


__all__ = [
    "DEFAULT_MACRO_RELEASE_CALENDAR_ROOT",
    "FrozenMacroReadinessRuntime",
    "MACRO_READINESS_RUNTIME_SCHEMA_VERSION",
    "freeze_macro_readiness_runtime",
]
