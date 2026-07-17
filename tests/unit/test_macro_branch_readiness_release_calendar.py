from __future__ import annotations

from dataclasses import replace

import pytest

from quant_investor.macro.release_calendar import (
    IssuerCoverage,
    ReleaseCalendarEvidence,
    ReleaseCalendarGenerationProof,
    ReleaseCalendarIdentity,
    ReleaseEvent,
    ReleaseResolution,
)
from quant_investor.market.branch_readiness import (
    STATUS_BLOCK,
    STATUS_PASS,
    assess_macro_readiness,
    build_macro_readiness_evidence,
)


def _calendar(
    *,
    event: ReleaseEvent | None = None,
    open_dates: tuple[str, ...] = (
        "2026-07-13",
        "2026-07-14",
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
        "2026-07-20",
    ),
    validated_ancestry: (
        tuple[ReleaseCalendarGenerationProof, ...] | None
    ) = None,
) -> ReleaseCalendarEvidence:
    current_proof = ReleaseCalendarGenerationProof(
        generation_id="calendar-current",
        pointer_sha256="1" * 64,
        manifest_sha256="2" * 64,
        semantic_sha256="3" * 64,
        plan_sha256="6" * 64,
        capture_manifest_sha256="7" * 64,
        market_open_days_sha256="8" * 64,
        registry_sha256="4" * 64,
        critical_policy_sha256="5" * 64,
    )
    return ReleaseCalendarEvidence(
        identity=ReleaseCalendarIdentity(
            pointer_path="/canonical/_latest.json",
            pointer_sha256="1" * 64,
            generation_id="calendar-current",
            generation_path="/canonical/_generations/calendar-current",
            manifest_sha256="2" * 64,
            semantic_sha256="3" * 64,
            parent_generation_id="",
            parent_pointer_sha256="",
            parent_manifest_sha256="",
            parent_semantic_sha256="",
        ),
        registry_version="fixture",
        registry_sha256="4" * 64,
        critical_policy_version="fixture",
        critical_policy_sha256="5" * 64,
        plan_sha256="6" * 64,
        capture_manifest_sha256="7" * 64,
        market_open_days_sha256="8" * 64,
        captured_at="2026-07-20T08:00:00+00:00",
        open_dates=open_dates,
        issuer_coverage=(
            IssuerCoverage(
                issuer="nbs_official",
                through_at="2026-07-20T08:00:00+00:00",
                source_ids=(),
            ),
            IssuerCoverage(
                issuer="pbc_official",
                through_at="2026-07-20T08:00:00+00:00",
                source_ids=(),
            ),
        ),
        source_artifacts=(),
        events=(() if event is None else (event,)),
        resolutions=(),
        validated_ancestry=(*(validated_ancestry or ()), current_proof),
    )


def _event(
    scheduled_at: str,
    *,
    schedule_kind: str = "timestamp",
    event_id: str = "critical-gdp",
) -> ReleaseEvent:
    return ReleaseEvent(
        event_id=event_id,
        event_family="nbs_quarterly_gdp",
        issuer="nbs_official",
        indicator_ids=("cn.gdp_yoy",),
        period="2026-Q2",
        schedule_kind=schedule_kind,
        scheduled_at=scheduled_at,
        status="scheduled",
        actual_at="",
        rescheduled_at="",
        reschedule_kind="",
        cancelled_at="",
        supersedes_event_id="",
        source_ids=(),
        resolution_ids=(),
    )


def _record(trade_date: str) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2026-07-15T07:00:00+00:00",
    }


def _manifest(evidence) -> dict[str, object]:
    binding = evidence.identity_binding()
    binding.pop("macro_readiness_evidence_semantic_sha256")
    return {
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "provider_status": "verified_provider_snapshot",
        "provider_fallback_used": False,
        "production_eligible": True,
        "generation_id": "macro-generation",
        **binding,
    }


def _proof_binding(proof: ReleaseCalendarGenerationProof) -> dict[str, str]:
    return {
        "macro_release_calendar_generation_id": proof.generation_id,
        "macro_release_calendar_pointer_sha256": proof.pointer_sha256,
        "macro_release_calendar_manifest_sha256": proof.manifest_sha256,
        "macro_release_calendar_semantic_sha256": proof.semantic_sha256,
        "macro_release_calendar_registry_sha256": proof.registry_sha256,
        "macro_release_calendar_plan_sha256": proof.plan_sha256,
        "macro_release_calendar_capture_manifest_sha256": (
            proof.capture_manifest_sha256
        ),
        "macro_release_calendar_market_open_days_sha256": (
            proof.market_open_days_sha256
        ),
        "macro_release_calendar_critical_policy_sha256": (
            proof.critical_policy_sha256
        ),
    }


def _assess(
    *,
    macro_date: str,
    target_date: str,
    cutoff: str,
    calendar: ReleaseCalendarEvidence | None = None,
):
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=calendar or _calendar(),
        macro_logical_date=macro_date,
        target_session_date=target_date,
        target_decision_cutoff_at=cutoff,
    )
    return assess_macro_readiness(
        macro_record=_record(macro_date),
        manifest=_manifest(pinned),
        as_of=target_date,
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=pinned,
    )


@pytest.mark.parametrize(
    ("macro_date", "expected_lag"),
    [
        ("2026-07-16", 0),
        ("2026-07-15", 1),
        ("2026-07-14", 2),
    ],
)
def test_zero_one_or_two_pinned_open_sessions_are_allowed_without_key_event(
    macro_date: str,
    expected_lag: int,
) -> None:
    result = _assess(
        macro_date=macro_date,
        target_date="2026-07-16",
        cutoff="2026-07-16T15:00:00+08:00",
    )

    assert result.status == STATUS_PASS
    assert result.blockers == []
    assert result.metadata["macro_session_lag"] == expected_lag
    assert (
        result.metadata["canonical_identity"][
            "macro_readiness_evidence_semantic_sha256"
        ]
        == result.metadata["macro_readiness_evidence"]["semantic_sha256"]
    )


def test_three_open_sessions_remain_blocked() -> None:
    result = _assess(
        macro_date="2026-07-13",
        target_date="2026-07-16",
        cutoff="2026-07-16T15:00:00+08:00",
    )

    assert result.status == STATUS_BLOCK
    assert "macro_trade_date_as_of_mismatch" in result.blockers
    assert "macro_release_session_lag_above_two" in result.blockers


@pytest.mark.parametrize(
    ("scheduled_at", "expected_status"),
    [
        ("2026-07-15T15:00:00+08:00", STATUS_PASS),
        ("2026-07-15T15:00:09+08:00", STATUS_BLOCK),
        ("2026-07-16T15:00:00+08:00", STATUS_BLOCK),
        ("2026-07-16T15:00:01+08:00", STATUS_PASS),
    ],
)
def test_timestamp_gap_boundaries_are_start_exclusive_and_cutoff_inclusive(
    scheduled_at: str,
    expected_status: str,
) -> None:
    event = _event(scheduled_at)
    result = _assess(
        macro_date="2026-07-15",
        target_date="2026-07-16",
        cutoff="2026-07-16T15:00:00+08:00",
        calendar=_calendar(event=event),
    )

    assert result.status == expected_status
    if expected_status == STATUS_BLOCK:
        assert "macro_trade_date_as_of_mismatch" in result.blockers
        assert any(
            item.startswith("macro_release_critical_event_in_gap:")
            for item in result.blockers
        )


def test_date_only_weekend_event_intersects_gap_and_blocks() -> None:
    result = _assess(
        macro_date="2026-07-17",
        target_date="2026-07-20",
        cutoff="2026-07-20T15:00:00+08:00",
        calendar=_calendar(
            event=_event(
                "2026-07-19",
                schedule_kind="date",
                event_id="critical-weekend",
            )
        ),
    )

    assert result.status == STATUS_BLOCK
    assert (
        "macro_release_critical_event_in_gap:critical-weekend"
        in result.blockers
    )


def test_lag_zero_resolved_post_close_event_still_blocks_branch() -> None:
    event_id = "critical-gdp-resolved"
    resolution_id = "critical-gdp-resolution"
    event = replace(
        _event(
            "2026-07-16T15:30:00+08:00",
            event_id=event_id,
        ),
        status="released",
        actual_at="2026-07-16T15:30:00+08:00",
        resolution_ids=(resolution_id,),
    )
    resolution = ReleaseResolution(
        resolution_id=resolution_id,
        event_id=event_id,
        indicator_id="cn.gdp_yoy",
        period_end="2026-06-30",
        frequency="quarterly",
        unit="%",
        measurement_basis="current_quarter_real_yoy",
        value_decimal="5",
        issuer="nbs_official",
        parser_id="fixture-parser",
        parser_contract_sha256="a" * 64,
        official_bundle_sha256="b" * 64,
        observation_content_hash="c" * 64,
        observation_available_at="2026-07-16T15:35:00+08:00",
        source_ids=(),
    )
    cutoff = "2026-07-16T16:00:00+08:00"
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=replace(
            _calendar(),
            events=(event,),
            resolutions=(resolution,),
        ),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at=cutoff,
    )

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=_manifest(pinned),
        as_of="2026-07-16",
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=pinned,
    )

    assert result.status == STATUS_BLOCK
    assert pinned.evaluation.session_lag.session_lag == 0
    assert "macro_release_readiness_blocked" in result.blockers
    assert (
        f"macro_release_critical_event_in_gap:{event_id}"
        in result.blockers
    )
    assert pinned.evaluation.critical_event_gap.blocking_event_ids == (
        event_id,
    )


def test_exact_date_still_requires_pinned_evidence() -> None:
    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest={
            "source": "tushare_primary",
            "source_priority": "tushare_primary",
            "provider_status": "verified_provider_snapshot",
            "production_eligible": True,
            "generation_id": "macro-generation",
        },
        as_of="2026-07-16",
        decision_cutoff_at="2026-07-16T15:00:00+08:00",
    )

    assert result.status == STATUS_BLOCK
    assert result.blockers == ["macro_release_readiness_evidence_missing"]


def test_manifest_release_calendar_identity_mismatch_fails_closed() -> None:
    cutoff = "2026-07-16T15:00:00+08:00"
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=_calendar(),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at=cutoff,
    )
    manifest = _manifest(pinned)
    manifest["macro_release_calendar_semantic_sha256"] = "f" * 64

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=manifest,
        as_of="2026-07-16",
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=pinned,
    )

    assert result.status == STATUS_BLOCK
    assert "macro_release_calendar_identity_mismatch" in result.blockers


def test_validated_release_calendar_ancestor_binding_is_accepted() -> None:
    cutoff = "2026-07-16T15:00:00+08:00"
    ancestor = ReleaseCalendarGenerationProof(
        generation_id="calendar-ancestor",
        pointer_sha256="a" * 64,
        manifest_sha256="b" * 64,
        semantic_sha256="c" * 64,
        plan_sha256="d" * 64,
        capture_manifest_sha256="e" * 64,
        market_open_days_sha256="9" * 64,
        registry_sha256="4" * 64,
        critical_policy_sha256="5" * 64,
    )
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=_calendar(validated_ancestry=(ancestor,)),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at=cutoff,
    )
    manifest = _manifest(pinned)
    manifest.update(_proof_binding(ancestor))

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=manifest,
        as_of="2026-07-16",
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=pinned,
    )

    assert result.status == STATUS_PASS
    assert result.blockers == []
    assert (
        result.metadata["canonical_identity"][
            "macro_release_calendar_generation_id"
        ]
        == "calendar-ancestor"
    )
    assert (
        result.metadata["macro_readiness_evidence"][
            "macro_release_calendar_generation_id"
        ]
        == "calendar-current"
    )


def test_partial_or_forged_ancestor_identity_is_rejected() -> None:
    cutoff = "2026-07-16T15:00:00+08:00"
    ancestor = ReleaseCalendarGenerationProof(
        generation_id="calendar-ancestor",
        pointer_sha256="a" * 64,
        manifest_sha256="b" * 64,
        semantic_sha256="c" * 64,
        plan_sha256="d" * 64,
        capture_manifest_sha256="e" * 64,
        market_open_days_sha256="9" * 64,
        registry_sha256="4" * 64,
        critical_policy_sha256="5" * 64,
    )
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=_calendar(validated_ancestry=(ancestor,)),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at=cutoff,
    )
    manifest = _manifest(pinned)
    manifest.update(_proof_binding(ancestor))
    manifest["macro_release_calendar_plan_sha256"] = "0" * 64

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=manifest,
        as_of="2026-07-16",
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=pinned,
    )

    assert result.status == STATUS_BLOCK
    assert "macro_release_calendar_identity_mismatch" in result.blockers


def test_mutated_pinned_evidence_semantic_sha_fails_closed() -> None:
    cutoff = "2026-07-16T15:00:00+08:00"
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=_calendar(),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at=cutoff,
    )
    tampered = replace(pinned, target_session_date="2026-07-17")

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=_manifest(pinned),
        as_of="2026-07-16",
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=tampered,
    )

    assert result.status == STATUS_BLOCK
    assert "macro_release_readiness_evidence_tampered" in result.blockers
    assert (
        "macro_release_readiness_evidence_target_mismatch"
        in result.blockers
    )


def test_stale_cutoff_binding_fails_closed() -> None:
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=_calendar(),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at="2026-07-16T15:00:00+08:00",
    )

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=_manifest(pinned),
        as_of="2026-07-16",
        decision_cutoff_at="2026-07-16T15:01:00+08:00",
        macro_readiness_evidence=pinned,
    )

    assert result.status == STATUS_BLOCK
    assert (
        "macro_release_readiness_evidence_cutoff_mismatch"
        in result.blockers
    )


def test_malformed_ancestry_is_blocked_without_crashing() -> None:
    cutoff = "2026-07-16T15:00:00+08:00"
    pinned = build_macro_readiness_evidence(
        release_calendar_evidence=_calendar(),
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        target_decision_cutoff_at=cutoff,
    )
    malformed = replace(
        pinned,
        validated_release_calendar_ancestry=("forged",),  # type: ignore[arg-type]
    )

    result = assess_macro_readiness(
        macro_record=_record("2026-07-16"),
        manifest=_manifest(pinned),
        as_of="2026-07-16",
        decision_cutoff_at=cutoff,
        macro_readiness_evidence=malformed,
    )

    assert result.status == STATUS_BLOCK
    assert "macro_release_readiness_evidence_tampered" in result.blockers
    assert "macro_release_readiness_ancestry_invalid" in result.blockers
