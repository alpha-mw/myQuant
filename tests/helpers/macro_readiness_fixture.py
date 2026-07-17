from __future__ import annotations

from quant_investor.macro.release_calendar import (
    IssuerCoverage,
    ReleaseCalendarEvidence,
    ReleaseCalendarGenerationProof,
    ReleaseCalendarIdentity,
)
from quant_investor.market.branch_readiness import (
    build_macro_readiness_evidence,
)
from quant_investor.market.macro_readiness_runtime import (
    FrozenMacroReadinessRuntime,
)


def _date_text(value: str) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def make_macro_readiness_runtime(
    *,
    macro_logical_date: str,
    target_session_date: str,
    decision_cutoff_at: str | None = None,
) -> FrozenMacroReadinessRuntime:
    logical_date = _date_text(macro_logical_date)
    target_date = _date_text(target_session_date)
    cutoff = decision_cutoff_at or f"{target_date}T15:00:00+08:00"
    proof = ReleaseCalendarGenerationProof(
        generation_id="fixture-release-calendar",
        pointer_sha256="1" * 64,
        manifest_sha256="2" * 64,
        semantic_sha256="3" * 64,
        plan_sha256="6" * 64,
        capture_manifest_sha256="7" * 64,
        market_open_days_sha256="8" * 64,
        registry_sha256="4" * 64,
        critical_policy_sha256="5" * 64,
    )
    calendar = ReleaseCalendarEvidence(
        identity=ReleaseCalendarIdentity(
            pointer_path="/fixture/release-calendar/_latest.json",
            pointer_sha256=proof.pointer_sha256,
            generation_id=proof.generation_id,
            generation_path=(
                "/fixture/release-calendar/_generations/fixture"
            ),
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
        captured_at="2030-01-01T00:00:00+00:00",
        open_dates=tuple(sorted({logical_date, target_date})),
        issuer_coverage=(
            IssuerCoverage(
                issuer="nbs_official",
                through_at="2030-01-01T00:00:00+00:00",
                source_ids=(),
            ),
            IssuerCoverage(
                issuer="pbc_official",
                through_at="2030-01-01T00:00:00+00:00",
                source_ids=(),
            ),
        ),
        source_artifacts=(),
        events=(),
        resolutions=(),
        validated_ancestry=(proof,),
    )
    evidence = build_macro_readiness_evidence(
        release_calendar_evidence=calendar,
        macro_logical_date=logical_date,
        target_session_date=target_date,
        target_decision_cutoff_at=cutoff,
    )
    return FrozenMacroReadinessRuntime(
        status="ready",
        macro_logical_date=logical_date,
        target_session_date=target_date,
        decision_cutoff_at=evidence.target_decision_cutoff_at,
        release_calendar_root="/fixture/release-calendar",
        release_calendar_pointer_sha256=proof.pointer_sha256,
        evidence=evidence,
    )


def macro_release_binding(
    runtime: FrozenMacroReadinessRuntime,
) -> dict[str, str]:
    assert runtime.evidence is not None
    binding = runtime.evidence.identity_binding()
    binding.pop("macro_readiness_evidence_semantic_sha256")
    return binding


def make_blocked_macro_readiness_runtime(
    *,
    macro_logical_date: str = "",
    target_session_date: str = "",
    blocker: str = "fixture_macro_readiness_blocked",
) -> FrozenMacroReadinessRuntime:
    return FrozenMacroReadinessRuntime(
        status="blocked",
        macro_logical_date=str(macro_logical_date),
        target_session_date=str(target_session_date),
        decision_cutoff_at="",
        release_calendar_root="/fixture/release-calendar",
        release_calendar_pointer_sha256="",
        evidence=None,
        blocker=blocker,
    )
