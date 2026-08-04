"""Pure maturity assessment for provisional research labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ._artifacts import ResearchArtifactError, session, timestamp

LABEL_HORIZONS: Final = (1, 5, 10, 20, 60)


@dataclass(frozen=True)
class LabelMaturity:
    horizon_sessions: int
    status: str
    blocker_codes: tuple[str, ...]
    future_sessions: tuple[str, ...]


def assess_label_maturity(
    *,
    decision_session: str,
    cutoff: str,
    horizon_sessions: int,
    future_sessions: Sequence[str] = (),
    calendar_ref: Mapping[str, object] | None = None,
) -> LabelMaturity:
    """Assess maturity without discovering a calendar or future data."""

    origin = session(decision_session, label="decision_session")
    timestamp(cutoff, label="cutoff")
    if horizon_sessions not in LABEL_HORIZONS:
        raise ResearchArtifactError("RESEARCH_LABEL_HORIZON_INVALID")
    if not future_sessions:
        return LabelMaturity(horizon_sessions, "PENDING", (), ())
    normalized = tuple(session(value, label="future_session") for value in future_sessions)
    blockers: list[str] = []
    if calendar_ref is None:
        blockers.append("EXPLICIT_CALENDAR_REF_UNAVAILABLE")
    if len(normalized) != horizon_sessions:
        blockers.append("LABEL_HORIZON_SESSION_COUNT_MISMATCH")
    if normalized != tuple(sorted(set(normalized))):
        blockers.append("LABEL_FUTURE_SESSION_ORDER_OR_DUPLICATE")
    if normalized and normalized[0] <= origin:
        blockers.append("LABEL_FUTURE_WINDOW_NOT_AFTER_ORIGIN")
    status = "MATURED" if not blockers else "BLOCKED"
    return LabelMaturity(
        horizon_sessions,
        status,
        tuple(sorted(blockers, key=lambda value: value.encode("ascii"))),
        normalized,
    )


__all__ = ["LABEL_HORIZONS", "LabelMaturity", "assess_label_maturity"]
