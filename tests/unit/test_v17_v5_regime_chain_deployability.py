from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.v17_v5_runtime.regime_chain_deployability import (
    MissedSessionRecoveryAudit,
    ObservedReplayOutcome,
    audit_regime_chain_deployability,
    estimate_predecessor_chain_depth,
)
import quant_investor.v17_v4_runtime.regime_evidence_v2 as v4_regime
from tests.unit.test_v17_v4_regime_evidence_v2_producer import (
    _clock,
    build_fixture,
    make_regime_fixture,
)

OPEN_SESSIONS = [
    "2026-07-29",
    "2026-07-30",
    "2026-07-31",
    "2026-08-03",
    "2026-08-04",
]


def test_chain_deployability_report_marks_long_chains_blocked_by_unchanged_v4_limits() -> None:
    audit = audit_regime_chain_deployability(
        session_counts=(20, 60, 260, 1000),
        observed_replay=ObservedReplayOutcome(
            max_successful_sessions=2,
            first_failing_session=3,
            first_failure_blocker="REGIME_EVIDENCE_V2_INPUT_TAMPER: closure validation resource budget exceeded",
        ),
        missed_session_recovery=MissedSessionRecoveryAudit(
            scenario="S0_BOOTSTRAP_S1_NORMAL_S2_MISSED_S3_ATTEMPT",
            replay_status="BLOCKED",
            blocker_codes=("REGIME_EVIDENCE_V2_INPUT_TAMPER",),
            liveness_blocker="V4_REGIME_CHAIN_LIVENESS_GAP",
        ),
    )

    assert [result.session_count for result in audit.length_results] == [20, 60, 260, 1000]
    assert estimate_predecessor_chain_depth(20) == 40
    assert all(result.replay_status == "BLOCKED" for result in audit.length_results)
    assert all(
        "V4_REGIME_CHAIN_SCALABILITY_GAP" in result.blocker_codes for result in audit.length_results
    )
    assert all(result.first_failing_session == 3 for result in audit.length_results)
    assert all(
        result.first_failure_blocker
        == "REGIME_EVIDENCE_V2_INPUT_TAMPER: closure validation resource budget exceeded"
        for result in audit.length_results
    )
    assert all(
        result.replay_duration_seconds == "NOT_MEASURED_AFTER_FIRST_FAILURE"
        for result in audit.length_results
    )
    assert audit.scalability_blocker == "V4_REGIME_CHAIN_SCALABILITY_GAP"
    assert audit.liveness_blocker == "V4_REGIME_CHAIN_LIVENESS_GAP"


def test_actual_v4_replay_resource_limit_blocks_third_contiguous_session(
    tmp_path: Path,
) -> None:
    first_fixture = make_regime_fixture(
        tmp_path,
        open_sessions=OPEN_SESSIONS,
    )
    first = build_fixture(first_fixture)
    second_fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-30",
        effective="2026-07-31",
        created_at="2026-07-30T15:21:00Z",
        open_sessions=OPEN_SESSIONS,
        prior=(first.document, first.evidence_path, first.evidence_sha256),
    )
    second = build_fixture(second_fixture)
    third_fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-31",
        effective="2026-08-03",
        created_at="2026-07-31T15:21:00Z",
        open_sessions=OPEN_SESSIONS,
        prior=(second.document, second.evidence_path, second.evidence_sha256),
    )

    with pytest.raises(v4_regime.RegimeEvidenceV2Error) as raised:
        v4_regime.build_regime_evidence_v2(
            **third_fixture.kwargs,
            _now_fn=_clock(third_fixture.kwargs["created_at"]),
        )

    assert raised.value.blocker_code == v4_regime.INPUT_TAMPER_BLOCKER
    assert "model_snapshot_ref readback failed" in raised.value.detail
    assert not (tmp_path / third_fixture.output_path).exists()


def test_missed_session_recovery_is_resource_blocked_before_it_can_resume(
    tmp_path: Path,
) -> None:
    first_fixture = make_regime_fixture(
        tmp_path,
        open_sessions=OPEN_SESSIONS,
    )
    first = build_fixture(first_fixture)
    second_fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-30",
        effective="2026-07-31",
        created_at="2026-07-30T15:21:00Z",
        open_sessions=OPEN_SESSIONS,
        prior=(first.document, first.evidence_path, first.evidence_sha256),
    )
    second = build_fixture(second_fixture)
    missed_third_attempt = make_regime_fixture(
        tmp_path,
        observed="2026-08-03",
        effective="2026-08-04",
        created_at="2026-08-03T15:21:00Z",
        open_sessions=OPEN_SESSIONS,
        prior=(second.document, second.evidence_path, second.evidence_sha256),
    )

    with pytest.raises(v4_regime.RegimeEvidenceV2Error) as raised:
        v4_regime.build_regime_evidence_v2(
            **missed_third_attempt.kwargs,
            _now_fn=_clock(missed_third_attempt.kwargs["created_at"]),
        )

    assert raised.value.blocker_code == v4_regime.INPUT_TAMPER_BLOCKER
    assert "model_snapshot_ref readback failed" in raised.value.detail
    assert not (tmp_path / missed_third_attempt.output_path).exists()

    restart_attempt = make_regime_fixture(
        tmp_path / "restart",
        observed="2026-08-03",
        effective="2026-08-04",
        created_at="2026-08-03T15:21:00Z",
        open_sessions=OPEN_SESSIONS,
    )
    with pytest.raises(v4_regime.RegimeEvidenceV2Error) as restart_raised:
        v4_regime.build_regime_evidence_v2(
            **restart_attempt.kwargs,
            _now_fn=_clock(restart_attempt.kwargs["created_at"]),
        )

    assert restart_raised.value.blocker_code == v4_regime.TEMPORAL_BLOCKER
    assert "NORMAL publication requires the contiguous prior v2" in restart_raised.value.detail
    assert not (tmp_path / "restart" / restart_attempt.output_path).exists()
