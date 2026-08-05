"""Shape v4 maturity evidence and apply BH by family.

`assess_candidate_maturity` grants maturity by one of two routes: at least
`MIN_MONTH_END_RANKIC_COUNT` (12) month-end RankIC observations, or at least
`MIN_NONOVERLAP_30D_COHORT_COUNT` (8) non-overlapping 30-session cohorts. It is
strict about shape -- a cohort must be exactly 30 contiguous calendar sessions,
carry the calendar's own sha, and have start/end agreeing with its dates -- and
it silently drops anything malformed, which would read as "immature" rather
than "miswired". These builders produce that shape from a factor's observed
RankIC dates so the two cannot drift.

`assess_factor_record_v4` separately demands `bh_q_value <= 0.10` with
`fdr_method == "benjamini_hochberg_by_family"`, so the correction is applied
within each family, not across the whole set.
"""

from __future__ import annotations

import pytest

from quant_investor.factors.governance_protocol_v4 import (
    FDR_Q,
    MIN_MONTH_END_RANKIC_COUNT,
    MIN_NONOVERLAP_30D_COHORT_COUNT,
    assess_candidate_maturity,
)
from quant_investor.factors.v4_maturity_evidence import (
    benjamini_hochberg_by_family,
    build_cohort_evidence,
    build_month_end_evidence,
)


class _Calendar:
    """Minimal stand-in with the two attributes the builders read."""

    def __init__(self, sessions, sha="a" * 64):
        self.open_session_dates = list(sessions)
        self.calendar_sha256 = sha


def _weekday_sessions(count: int, start: str = "2024-01-01") -> list[str]:
    import pandas as pd

    return [d.date().isoformat() for d in pd.bdate_range(start, periods=count)]


# --- cohort evidence --------------------------------------------------------


def test_cohorts_are_thirty_contiguous_sessions():
    calendar = _Calendar(_weekday_sessions(300))

    cohorts = build_cohort_evidence(calendar, observed_dates=calendar.open_session_dates)

    assert cohorts
    for cohort in cohorts:
        assert len(cohort["open_session_dates"]) == 30
        assert cohort["start"] == cohort["open_session_dates"][0]
        assert cohort["end"] == cohort["open_session_dates"][-1]
        assert cohort["horizon_days"] == 30
        assert cohort["calendar_sha256"] == calendar.calendar_sha256


def test_cohorts_only_count_sessions_the_factor_actually_observed():
    """A factor with no values on a stretch must not inherit its cohorts."""

    calendar = _Calendar(_weekday_sessions(300))
    observed = calendar.open_session_dates[:60]

    cohorts = build_cohort_evidence(calendar, observed_dates=observed)

    assert len(cohorts) == 2


def test_cohorts_are_accepted_by_the_protocol_and_grant_maturity():
    """The real check: the shape survives assess_candidate_maturity."""

    sessions = _weekday_sessions(300)
    calendar = _Calendar(sessions)
    from quant_investor.factors.governance_protocol_v4 import (
        OPEN_SESSION_CALENDAR_SCHEMA_VERSION,
        OPEN_SESSION_CALENDAR_SOURCE,
        validate_open_session_calendar_v4,
    )

    payload = {
        "schema_version": OPEN_SESSION_CALENDAR_SCHEMA_VERSION,
        "market": "CN",
        "source": OPEN_SESSION_CALENDAR_SOURCE,
        "latest_pointer_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "open_session_dates": sessions,
    }

    calendar.calendar_sha256 = validate_open_session_calendar_v4(payload)["calendar_sha256"]
    cohorts = build_cohort_evidence(calendar, observed_dates=sessions)

    assessed = assess_candidate_maturity(
        month_end_rankic_dates=[],
        forward_cohorts=cohorts,
        calendar=payload,
    )

    assert assessed["nonoverlap_30d_cohort_count"] >= MIN_NONOVERLAP_30D_COHORT_COUNT
    assert assessed["mature"] is True
    assert assessed["maturity_route"] == "nonoverlap_30d_forward_cohort"


def test_too_little_history_does_not_grant_maturity():
    sessions = _weekday_sessions(90)
    calendar = _Calendar(sessions)

    cohorts = build_cohort_evidence(calendar, observed_dates=sessions)

    assert len(cohorts) < MIN_NONOVERLAP_30D_COHORT_COUNT


def test_cohort_ids_are_unique():
    calendar = _Calendar(_weekday_sessions(300))

    cohorts = build_cohort_evidence(calendar, observed_dates=calendar.open_session_dates)

    ids = [cohort["cohort_id"] for cohort in cohorts]
    assert len(set(ids)) == len(ids)


# --- month-end evidence -----------------------------------------------------


def test_month_end_evidence_is_the_last_observed_session_of_each_month():
    calendar = _Calendar(_weekday_sessions(300))

    month_ends = build_month_end_evidence(
        calendar, observed_dates=calendar.open_session_dates
    )

    assert len(month_ends) >= MIN_MONTH_END_RANKIC_COUNT
    assert month_ends == sorted(month_ends)
    assert len({item[:7] for item in month_ends}) == len(month_ends)


def test_month_end_evidence_skips_months_the_factor_never_observed():
    calendar = _Calendar(_weekday_sessions(300))
    observed = [d for d in calendar.open_session_dates if not d.startswith("2024-03")]

    month_ends = build_month_end_evidence(calendar, observed_dates=observed)

    assert not any(item.startswith("2024-03") for item in month_ends)


def test_a_month_end_that_is_not_the_calendar_month_end_is_not_counted():
    """The protocol only counts a date that is the month's actual last session."""

    sessions = _weekday_sessions(300)
    calendar = _Calendar(sessions)
    observed = [d for d in sessions if d != "2024-01-31"]

    month_ends = build_month_end_evidence(calendar, observed_dates=observed)

    assert "2024-01-31" not in month_ends


# --- Benjamini-Hochberg -----------------------------------------------------


def test_bh_within_a_singleton_family_returns_the_raw_p_value():
    """One factor per family means no multiplicity to correct within it."""

    q = benjamini_hochberg_by_family(
        p_values={"a": 0.04, "b": 0.09},
        families={"a": "growth", "b": "liquidity"},
    )

    assert q == {"a": pytest.approx(0.04), "b": pytest.approx(0.09)}


def test_bh_corrects_within_a_family_not_across_families():
    q = benjamini_hochberg_by_family(
        p_values={"a1": 0.01, "a2": 0.04, "b1": 0.01},
        families={"a1": "growth", "a2": "growth", "b1": "liquidity"},
    )

    assert q["b1"] == pytest.approx(0.01)  # alone in its family
    assert q["a2"] == pytest.approx(0.04)  # 0.04 * 2/2
    assert q["a1"] == pytest.approx(0.02)  # 0.01 * 2/1


def test_bh_is_monotone_so_a_smaller_p_never_gets_a_larger_q():
    q = benjamini_hochberg_by_family(
        p_values={"a": 0.03, "b": 0.031, "c": 0.9},
        families=dict.fromkeys("abc", "growth"),
    )

    assert q["a"] <= q["b"] <= q["c"]


def test_bh_q_is_clipped_to_one():
    q = benjamini_hochberg_by_family(
        p_values={"a": 0.9, "b": 0.95},
        families=dict.fromkeys("ab", "growth"),
    )

    assert all(value <= 1.0 for value in q.values())


def test_bh_rejects_a_factor_with_no_family():
    with pytest.raises(ValueError, match="family"):
        benjamini_hochberg_by_family(p_values={"a": 0.01}, families={})


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_bh_rejects_an_impossible_p_value(bad):
    with pytest.raises(ValueError, match="p-value"):
        benjamini_hochberg_by_family(p_values={"a": bad}, families={"a": "growth"})


def test_the_v4_threshold_is_what_the_protocol_says():
    """Guard against silently drifting away from the protocol constant."""

    assert FDR_Q == pytest.approx(0.10)
