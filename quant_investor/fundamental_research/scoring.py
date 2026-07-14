"""Deterministic evidence qualification and bounded overlay scoring."""

from __future__ import annotations

from datetime import datetime

from .models import (
    ClaimKind,
    Dimension,
    DimensionContributionV1,
    DimensionSignal,
    FundamentalOverlayV1,
    FundamentalResearchDossierV1,
    FundamentalResearchRequestV1,
    SourceEligibilityPolicyV1,
    SourceTier,
)

DIMENSION_WEIGHTS = {
    Dimension.FINANCIAL_QUALITY: 0.25,
    Dimension.BUSINESS_ECONOMICS: 0.15,
    Dimension.INDUSTRY_VALUE_CHAIN: 0.15,
    Dimension.COMPETITIVE_ADVANTAGE: 0.20,
    Dimension.MANAGEMENT_CAPITAL_ALLOCATION: 0.10,
    Dimension.VALUATION_SCENARIOS: 0.15,
}
SIGNAL_VALUES = {
    DimensionSignal.STRONG_NEGATIVE: -1.0,
    DimensionSignal.NEGATIVE: -0.5,
    DimensionSignal.NEUTRAL: 0.0,
    DimensionSignal.POSITIVE: 0.5,
    DimensionSignal.STRONG_POSITIVE: 1.0,
    DimensionSignal.UNKNOWN: 0.0,
}
POSITIVE_REQUIRED = {
    Dimension.FINANCIAL_QUALITY,
    Dimension.COMPETITIVE_ADVANTAGE,
    Dimension.VALUATION_SCENARIOS,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_overlay(
    request: FundamentalResearchRequestV1,
    dossier: FundamentalResearchDossierV1,
    *,
    imported_at: datetime,
    source_policy: SourceEligibilityPolicyV1 | None = None,
) -> FundamentalOverlayV1:
    """Validate request binding and compute a local-only bounded score delta."""
    blockers: list[str] = []
    if imported_at.tzinfo is None or imported_at.utcoffset() is None:
        raise ValueError("imported_at must be timezone-aware")
    if imported_at > request.expires_at:
        blockers.append("request_expired")
    if imported_at < request.created_at:
        blockers.append("import_before_request_created")
    if dossier.produced_at < request.created_at:
        blockers.append("dossier_produced_before_request")
    if dossier.produced_at > imported_at:
        blockers.append("dossier_produced_after_import")
    if dossier.request_id != request.request_id:
        blockers.append("request_id_mismatch")
    if dossier.symbol != request.symbol or dossier.market != request.market:
        blockers.append("identity_mismatch")
    if dossier.company_name != request.company_name:
        blockers.append("company_name_mismatch")
    if dossier.decision_cutoff != request.decision_cutoff:
        blockers.append("decision_cutoff_mismatch")
    if dossier.prompt_version != request.prompt_version:
        blockers.append("prompt_version_mismatch")

    policy = source_policy or SourceEligibilityPolicyV1()
    sources = {item.source_id: item for item in dossier.sources}
    local_tiers = {item.source_id: policy.classify(item) for item in dossier.sources}
    for item in dossier.sources:
        if item.retrieved_at > dossier.produced_at:
            blockers.append(f"source_retrieved_after_dossier:{item.source_id}")
        if item.retrieved_at > imported_at:
            blockers.append(f"source_retrieved_after_import:{item.source_id}")
        if item.source_tier != local_tiers[item.source_id]:
            blockers.append(f"source_tier_mismatch:{item.source_id}")
    claims = {item.claim_id: item for item in dossier.claims}
    eligible_authorities = {
        policy.authority_key(item)
        for item in dossier.sources
        if local_tiers[item.source_id] != SourceTier.INELIGIBLE
        and item.first_available_at <= request.decision_cutoff
    }
    if len(eligible_authorities) < 2:
        blockers.append("fewer_than_two_independent_publishers")

    contributions: list[DimensionContributionV1] = []
    qualified_dimensions: set[Dimension] = set()
    unresolved_primary_conflict = False
    raw_signal = 0.0
    for assessment in dossier.dimensions:
        dimension_blockers: list[str] = []
        dimension_claims = [claims[item] for item in assessment.claim_ids]
        eligible_claims = []
        if assessment.signal == DimensionSignal.UNKNOWN:
            dimension_blockers.append("unknown_dimension")
        primary_support = False
        for claim in dimension_claims:
            support = [sources[item] for item in claim.supporting_source_ids]
            counters = [sources[item] for item in claim.counter_source_ids]
            if claim.kind == ClaimKind.UNKNOWN:
                dimension_blockers.append(f"unknown_scoring_claim:{claim.claim_id}")
                continue
            claim_pit_eligible = bool(
                (claim.valid_from is None or claim.valid_from <= request.decision_cutoff)
                and (claim.valid_until is None or claim.valid_until >= request.decision_cutoff)
                and support
                and all(
                    local_tiers[item.source_id] != SourceTier.INELIGIBLE
                    and item.first_available_at <= request.decision_cutoff
                    for item in support
                )
            )
            if not claim_pit_eligible:
                dimension_blockers.append(f"claim_not_pit_eligible:{claim.claim_id}")
                continue
            eligible_claims.append(claim)
            if any(
                local_tiers[item.source_id] == SourceTier.PRIMARY
                for item in support
            ):
                primary_support = True
            if any(
                local_tiers[item.source_id] == SourceTier.PRIMARY
                and item.first_available_at <= request.decision_cutoff
                for item in counters
            ):
                unresolved_primary_conflict = True
                dimension_blockers.append("unresolved_primary_conflict")
        if assessment.signal != DimensionSignal.UNKNOWN and not primary_support:
            dimension_blockers.append("no_eligible_primary_claim")
        expected_direction = None
        if SIGNAL_VALUES[assessment.signal] > 0:
            expected_direction = "positive"
        elif SIGNAL_VALUES[assessment.signal] < 0:
            expected_direction = "negative"
        if expected_direction and not any(
            claim.direction == expected_direction for claim in eligible_claims
        ):
            dimension_blockers.append("signal_claim_direction_mismatch")
        qualified = not dimension_blockers
        contribution = (
            DIMENSION_WEIGHTS[assessment.dimension] * SIGNAL_VALUES[assessment.signal]
            if qualified
            else 0.0
        )
        if qualified:
            qualified_dimensions.add(assessment.dimension)
            raw_signal += contribution
        contributions.append(
            DimensionContributionV1(
                dimension=assessment.dimension,
                signal=assessment.signal,
                qualified=qualified,
                weight=DIMENSION_WEIGHTS[assessment.dimension],
                contribution=contribution,
                blockers=dimension_blockers,
            )
        )

    if raw_signal > 0.0:
        if len(qualified_dimensions) < 4:
            blockers.append("positive_gate_requires_four_dimensions")
        if not POSITIVE_REQUIRED <= qualified_dimensions:
            blockers.append("positive_gate_required_dimensions_missing")
        if unresolved_primary_conflict:
            blockers.append("positive_gate_primary_conflict")

    eligible = not blockers
    computed_delta = _clamp(raw_signal * 0.10, -0.10, 0.10)
    if not eligible:
        computed_delta = min(0.0, computed_delta) if raw_signal <= 0.0 else 0.0
        # Binding, expiry, and source-diversity failures invalidate both directions.
        if any(
            item
            not in {
                "positive_gate_requires_four_dimensions",
                "positive_gate_required_dimensions_missing",
                "positive_gate_primary_conflict",
            }
            for item in blockers
        ):
            computed_delta = 0.0
    adjusted = _clamp(request.base_score + computed_delta, -1.0, 1.0)
    return FundamentalOverlayV1(
        request_id=request.request_id,
        dossier_id=dossier.dossier_id,
        symbol=request.symbol,
        base_score=request.base_score,
        computed_delta=computed_delta,
        adjusted_score=adjusted,
        eligible=eligible,
        contributions=contributions,
        blockers=blockers,
    )
