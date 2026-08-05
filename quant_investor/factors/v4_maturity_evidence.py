"""Build v4 maturity evidence and apply Benjamini-Hochberg within families.

`assess_candidate_maturity` grants maturity by one of two routes: at least
`MIN_MONTH_END_RANKIC_COUNT` month-end RankIC observations, or at least
`MIN_NONOVERLAP_30D_COHORT_COUNT` non-overlapping 30-session cohorts. It is
strict about shape -- a cohort must be exactly 30 contiguous calendar sessions,
carry the calendar's own sha, and have `start`/`end` agreeing with its dates --
and it *silently drops* anything malformed. A miswired producer would therefore
read as "immature" rather than "broken", which is the failure this module
exists to prevent: the shapes are built here, once, next to the tests that feed
them straight back through the protocol.

Both routes count what a factor actually observed, not what the calendar could
in principle support. A factor with no values over a stretch of sessions does
not inherit that stretch's cohorts.

`assess_factor_record_v4` separately requires `bh_q_value <= FDR_Q` with
`fdr_method == "benjamini_hochberg_by_family"`, so the multiplicity correction
is applied inside each family rather than across the whole production set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

NONOVERLAP_COHORT_SIZE = 30


def build_cohort_evidence(
    calendar: Any,
    *,
    observed_dates: Sequence[str],
    size: int = NONOVERLAP_COHORT_SIZE,
) -> list[dict[str, Any]]:
    """Cut non-overlapping cohorts from the sessions a factor actually observed.

    Cut from the most recent session backwards so the newest evidence is always
    a whole cohort, matching how forward evaluation accrues.
    """

    if size < 1:
        raise ValueError("size must be at least 1")
    observed = set(str(item) for item in observed_dates)
    # Restrict to calendar sessions, in calendar order, so the contiguity check
    # in `assess_candidate_maturity` is evaluated against calendar indexes.
    usable = [item for item in calendar.open_session_dates if item in observed]

    cohorts: list[dict[str, Any]] = []
    end = len(usable)
    while end - size >= 0:
        window = usable[end - size : end]
        cohorts.append(
            {
                "cohort_id": f"cohort-{window[0]}-{window[-1]}",
                "start": window[0],
                "end": window[-1],
                "horizon_days": size,
                "calendar_sha256": calendar.calendar_sha256,
                "open_session_dates": list(window),
            }
        )
        end -= size
    cohorts.reverse()
    return cohorts


def build_month_end_evidence(
    calendar: Any,
    *,
    observed_dates: Sequence[str],
) -> list[str]:
    """Return observed dates that are a month's actual last calendar session.

    `assess_candidate_maturity` only counts a date when it equals the calendar's
    own month-end, so filtering here keeps the producer honest instead of
    handing over dates the protocol will quietly discard.
    """

    actual_month_end: dict[str, str] = {}
    for session in calendar.open_session_dates:
        actual_month_end[session[:7]] = session

    observed = set(str(item) for item in observed_dates)
    return sorted(
        session
        for month, session in actual_month_end.items()
        if session in observed
    )


def _validated_p(name: str, raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"p-value is not a number for {name}: {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"p-value outside [0, 1] for {name}: {value}")
    return value


def benjamini_hochberg_by_family(
    p_values: Mapping[str, Any],
    families: Mapping[str, str],
) -> dict[str, float]:
    """Benjamini-Hochberg q-values computed independently within each family.

    Step-up with the usual monotonicity enforcement, so a smaller p can never
    receive a larger q. With one factor per family -- which the v4 weight caps
    force at five factors -- each family is a singleton and q equals p; the
    general form is implemented anyway so a later multi-factor family is
    corrected rather than silently under-corrected.
    """

    grouped: dict[str, list[tuple[float, str]]] = {}
    for name, raw in p_values.items():
        family = str(families.get(name) or "").strip()
        if not family:
            raise ValueError(f"factor has no family for BH correction: {name}")
        grouped.setdefault(family, []).append((_validated_p(name, raw), name))

    q_values: dict[str, float] = {}
    for _family, entries in grouped.items():
        ordered = sorted(entries)
        total = len(ordered)
        running = 1.0
        # Walk from the largest p downwards so the running minimum enforces
        # monotonicity in one pass.
        for rank in range(total, 0, -1):
            p_value, name = ordered[rank - 1]
            running = min(running, p_value * total / rank)
            q_values[name] = min(1.0, running)
    return q_values
