from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_investor.macro.release_calendar import (
    IssuerCoverage,
    ReleaseCalendarCASMismatch,
    ReleaseCalendarEvidence,
    ReleaseCalendarGenerationProof,
    ReleaseCalendarIdentity,
)
from quant_investor.market import macro_readiness_runtime as runtime_module
from quant_investor.market.macro_readiness_runtime import (
    DEFAULT_MACRO_RELEASE_CALENDAR_ROOT,
    freeze_macro_readiness_runtime,
)


def _calendar(
    *,
    coverage: tuple[IssuerCoverage, ...] | None = None,
) -> ReleaseCalendarEvidence:
    proof = ReleaseCalendarGenerationProof(
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
            pointer_sha256=proof.pointer_sha256,
            generation_id=proof.generation_id,
            generation_path="/canonical/_generations/calendar-current",
            manifest_sha256=proof.manifest_sha256,
            semantic_sha256=proof.semantic_sha256,
            parent_generation_id="",
            parent_pointer_sha256="",
            parent_manifest_sha256="",
            parent_semantic_sha256="",
        ),
        registry_version="fixture",
        registry_sha256=proof.registry_sha256,
        critical_policy_version="fixture",
        critical_policy_sha256=proof.critical_policy_sha256,
        plan_sha256=proof.plan_sha256,
        capture_manifest_sha256=proof.capture_manifest_sha256,
        market_open_days_sha256=proof.market_open_days_sha256,
        captured_at="2026-07-20T08:00:00+00:00",
        open_dates=(
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
            "2026-07-20",
        ),
        issuer_coverage=(
            coverage
            if coverage is not None
            else (
                IssuerCoverage(
                    issuer="nbs_official",
                    through_at="2026-07-17T08:00:00+00:00",
                    source_ids=(),
                ),
                IssuerCoverage(
                    issuer="pbc_official",
                    through_at="2026-07-17T08:30:00+00:00",
                    source_ids=(),
                ),
            )
        ),
        source_artifacts=(),
        events=(),
        resolutions=(),
        validated_ancestry=(proof,),
    )


def _patch_calendar(monkeypatch, calendar: ReleaseCalendarEvidence):
    calls: dict[str, object] = {"pointer": 0, "load": 0}

    def _pointer(*, canonical_root):
        calls["pointer"] = int(calls["pointer"]) + 1
        calls["pointer_root"] = canonical_root
        return "1" * 64

    def _load(*, canonical_root, expected_pointer_sha256):
        calls["load"] = int(calls["load"]) + 1
        calls["load_root"] = canonical_root
        calls["expected_pointer_sha256"] = expected_pointer_sha256
        return calendar

    monkeypatch.setattr(
        runtime_module,
        "release_calendar_pointer_sha256",
        _pointer,
    )
    monkeypatch.setattr(runtime_module, "load_release_calendar", _load)
    return calls


def test_target_before_earliest_coverage_uses_cn_close_and_stable_pointer(
    monkeypatch,
) -> None:
    calls = _patch_calendar(monkeypatch, _calendar())

    frozen = freeze_macro_readiness_runtime(
        macro_logical_date="20260715",
        target_session_date="20260716",
        now=datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc),
    )

    expected_root = Path.cwd() / DEFAULT_MACRO_RELEASE_CALENDAR_ROOT
    assert frozen.ready is True
    assert frozen.macro_logical_date == "2026-07-15"
    assert frozen.target_session_date == "2026-07-16"
    assert frozen.decision_cutoff_at == "2026-07-16T07:00:00+00:00"
    assert frozen.evidence.evaluation.ready is True
    assert frozen.evidence.evaluation.session_lag.session_lag == 1
    assert calls == {
        "pointer": 1,
        "load": 1,
        "pointer_root": expected_root,
        "load_root": expected_root,
        "expected_pointer_sha256": "1" * 64,
    }
    metadata = frozen.metadata()
    assert metadata["macro_readiness_evidence"] == frozen.evidence.to_dict()
    assert (
        metadata["macro_readiness_evidence_semantic_sha256"]
        == frozen.evidence.semantic_sha256
    )


def test_target_on_coverage_date_uses_earliest_issuer_through_at(
    monkeypatch,
) -> None:
    calendar = _calendar(
        coverage=(
            IssuerCoverage(
                issuer="nbs_official",
                through_at="2026-07-17T07:30:00+00:00",
                source_ids=(),
            ),
            IssuerCoverage(
                issuer="pbc_official",
                through_at="2026-07-17T08:00:00+00:00",
                source_ids=(),
            ),
        )
    )
    _patch_calendar(monkeypatch, calendar)

    frozen = freeze_macro_readiness_runtime(
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-17",
        now=datetime(2026, 7, 17, 8, 30, tzinfo=timezone.utc),
    )

    assert frozen.ready is True
    assert frozen.decision_cutoff_at == "2026-07-17T07:30:00+00:00"
    assert (
        frozen.evidence.target_decision_cutoff_at
        == frozen.decision_cutoff_at
    )


@pytest.mark.parametrize(
    ("calendar", "target", "now", "blocker"),
    [
        (
            _calendar(
                coverage=(
                    IssuerCoverage(
                        issuer="nbs_official",
                        through_at="2026-07-17T06:59:59+00:00",
                        source_ids=(),
                    ),
                )
            ),
            "2026-07-17",
            datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc),
            "macro_release_calendar_coverage_before_market_close",
        ),
        (
            _calendar(
                coverage=(
                    IssuerCoverage(
                        issuer="nbs_official",
                        through_at="2026-07-17T08:00:00+00:00",
                        source_ids=(),
                    ),
                )
            ),
            "2026-07-17",
            datetime(2026, 7, 17, 7, 59, tzinfo=timezone.utc),
            "macro_release_calendar_coverage_in_future",
        ),
        (
            _calendar(),
            "2026-07-20",
            datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
            "macro_release_calendar_coverage_before_target",
        ),
        (
            _calendar(),
            "2026-07-16",
            datetime(2026, 7, 16, 6, 59, tzinfo=timezone.utc),
            "macro_release_calendar_coverage_in_future",
        ),
        (
            _calendar(coverage=()),
            "2026-07-17",
            datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc),
            "macro_release_calendar_issuer_coverage_missing",
        ),
    ],
)
def test_invalid_coverage_fails_closed_without_evidence(
    monkeypatch,
    calendar,
    target,
    now,
    blocker,
) -> None:
    _patch_calendar(monkeypatch, calendar)

    frozen = freeze_macro_readiness_runtime(
        macro_logical_date="2026-07-16",
        target_session_date=target,
        now=now,
    )

    assert frozen.ready is False
    assert frozen.evidence is None
    assert frozen.blocker == blocker
    assert frozen.metadata()["macro_readiness_evidence"] == {}


def test_pointer_cas_failure_is_not_retried_or_fallback_loaded(
    monkeypatch,
) -> None:
    calls = {"pointer": 0, "load": 0}

    def _pointer(*, canonical_root):
        calls["pointer"] += 1
        return "1" * 64

    def _load(*, canonical_root, expected_pointer_sha256):
        calls["load"] += 1
        raise ReleaseCalendarCASMismatch(
            "release_calendar_pointer_cas_mismatch"
        )

    monkeypatch.setattr(
        runtime_module,
        "release_calendar_pointer_sha256",
        _pointer,
    )
    monkeypatch.setattr(runtime_module, "load_release_calendar", _load)

    frozen = freeze_macro_readiness_runtime(
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-17",
    )

    assert calls == {"pointer": 1, "load": 1}
    assert frozen.ready is False
    assert frozen.evidence is None
    assert frozen.blocker == "release_calendar_pointer_cas_mismatch"
    assert frozen.release_calendar_pointer_sha256 == "1" * 64


def test_frozen_runtime_metadata_cannot_be_rebound() -> None:
    blocked = runtime_module.FrozenMacroReadinessRuntime(
        status="blocked",
        macro_logical_date="",
        target_session_date="",
        decision_cutoff_at="",
        release_calendar_root="/canonical",
        release_calendar_pointer_sha256="",
        evidence=None,
        blocker="fixture",
    )

    with pytest.raises(FrozenInstanceError):
        blocked.status = "ready"  # type: ignore[misc]
