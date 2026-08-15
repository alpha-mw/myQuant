from __future__ import annotations

from decimal import Decimal

import pytest

from quant_investor.contracts import seal_artifact
from quant_investor.intelligence import (
    IntelligenceError,
    assess_fundamental,
    assess_graduation,
    assess_industry,
    assess_theme,
    build_decision_context,
    construct_research_portfolio,
    make_investment_decision,
    observe_paper_portfolio,
    replay_advisory,
    review_advisory,
)
from quant_investor.intelligence._common import artifact_ref

NOW = "2026-08-14T00:00:00Z"


def source() -> dict:
    return seal_artifact(
        "system.source_bundle",
        {"source_bundle_id": "source-a", "sources": [], "state": "READY"},
        created_at=NOW,
    )


def available_industry(*, assessment_id: str | None = None, metric_value: str = "0.8") -> dict:
    return assess_industry(
        company="000001.SZ",
        memberships=[
            {
                "available_at": NOW,
                "effective_from": NOW,
                "exposure": "1",
                "industry_id": "BANK",
                "provider": "official",
                "retired": False,
            }
        ],
        provider_precedence=["official"],
        as_of=NOW,
        metric_rows=[
            {
                "metric_id": "profitability",
                "status": "AVAILABLE",
                "value": metric_value,
                "weight": "1",
            }
        ],
        assessment_id=assessment_id,
    )


def available_theme() -> dict:
    return assess_theme(
        company="000001.SZ",
        memberships=[
            {
                "available_at": NOW,
                "exposure": "1",
                "exposure_basis": "REVENUE",
                "provider": "official",
                "score": "0.7",
                "status": "ACTIVE",
                "theme_id": "FINTECH",
            }
        ],
        provider_precedence=["official"],
        catalog_complete=True,
        as_of=NOW,
    )


def fundamental(industry: dict, theme: dict, *, missing_valuation: bool = False) -> dict:
    return assess_fundamental(
        company="000001.SZ",
        component_scores={
            "business_quality": "0.8",
            "earnings_quality": "0.7",
            "growth_durability": "0.6",
            "industry_cycle": "0.5",
            "valuation": None if missing_valuation else "0.9",
        },
        component_weights={
            component: "0.2"
            for component in (
                "business_quality",
                "earnings_quality",
                "growth_durability",
                "industry_cycle",
                "valuation",
            )
        },
        minimum_coverage="0.6",
        as_of=NOW,
        source_refs=[artifact_ref(source())],
        industry_assessment=industry,
        theme_assessment=theme,
    )


def paper_candidate() -> tuple[dict, dict, dict, dict]:
    industry = available_industry()
    theme = available_theme()
    profile = fundamental(industry, theme)
    evidence = source()
    context = build_decision_context(
        company="000001.SZ",
        as_of=NOW,
        hypothesis_status="VALID",
        risk_status="AVAILABLE",
        evidence_refs=[artifact_ref(evidence)],
        industry_assessment=industry,
        theme_assessment=theme,
        fundamental_assessment=profile,
        quant_ref=artifact_ref(evidence),
    )
    decision = make_investment_decision(
        context=context,
        deterministic_percentile="0.95",
        thresholds={"paper_candidate": "0.90", "research_approved": "0.70"},
        as_of=NOW,
    )
    return industry, theme, context, decision


def test_industry_identity_is_source_bound_and_business_id_is_independent() -> None:
    unmapped = assess_industry(
        company="000001.SZ",
        memberships=[],
        provider_precedence=["official"],
        as_of=NOW,
    )
    assert unmapped["payload"]["status"] == "UNMAPPED"
    assert unmapped["payload"]["primary_industry_id"] is None
    assert unmapped["payload"]["component_score"] is None

    first = available_industry(assessment_id="industry-owner-1", metric_value="0.8")
    changed_score = available_industry(assessment_id="industry-owner-1", metric_value="0.9")
    assert first["artifact_id"] == changed_score["artifact_id"]
    assert first["semantic_sha256"] != changed_score["semantic_sha256"]


def test_theme_no_membership_and_fundamental_missingness_are_not_imputed() -> None:
    no_membership = assess_theme(
        company="000001.SZ",
        memberships=[],
        provider_precedence=["official"],
        catalog_complete=True,
        as_of=NOW,
    )
    assert no_membership["payload"]["status"] == "NO_MEMBERSHIP"
    assert no_membership["payload"]["component_score"] is None
    assert no_membership["payload"]["exposures"] == []

    profile = fundamental(available_industry(), available_theme(), missing_valuation=True)
    valuation = next(
        row for row in profile["payload"]["component_rows"] if row["component"] == "valuation"
    )
    assert profile["payload"]["status"] == "PARTIAL"
    assert profile["payload"]["coverage"] == "0.800000000000"
    assert valuation["status"] == "MISSING"
    assert valuation["score"] is None


def test_five_state_decision_priority_and_advisory_boundary() -> None:
    _, _, context, decision = paper_candidate()
    assert decision["payload"]["state"] == "PAPER_CANDIDATE"

    invalidated_context = dict(context)
    invalidated_payload = dict(context["payload"])
    invalidated_payload["hypothesis_status"] = "INVALIDATED"
    invalidated_context = seal_artifact("decision_context", invalidated_payload, created_at=NOW)
    invalidated = make_investment_decision(
        context=invalidated_context,
        deterministic_percentile="0.99",
        thresholds={"paper_candidate": "0.90", "research_approved": "0.70"},
        as_of=NOW,
    )
    assert invalidated["payload"]["state"] == "THESIS_INVALIDATED"

    advisory = review_advisory(
        decision=decision,
        proposed_percentile="0.96",
        validated_facts=[
            {
                "confidence": "0.8",
                "fact": "validated contrary fact A",
                "source_id": "issuer",
                "source_url": "https://issuer.example/disclosure",
            },
            {
                "confidence": "0.7",
                "fact": "validated contrary fact B",
                "source_id": "exchange",
                "source_url": "https://exchange.example/notice",
            },
        ],
        as_of=NOW,
    )
    assert advisory["payload"]["absolute_delta"] == "0.010000000000"
    assert advisory["payload"]["deterministic_decision_state"] == "PAPER_CANDIDATE"
    assert replay_advisory(advisory, decision=decision) == advisory

    tampered_payload = dict(advisory["payload"])
    tampered_payload["absolute_delta"] = "0.020000000000"
    tampered = seal_artifact("advisory_review", tampered_payload, created_at=NOW)
    with pytest.raises(IntelligenceError, match="does not replay"):
        replay_advisory(tampered, decision=decision)

    with pytest.raises(IntelligenceError, match="ten percent"):
        review_advisory(
            decision=decision,
            proposed_percentile="0.70",
            validated_facts=advisory["payload"]["validated_facts"],
            as_of=NOW,
        )

    invalid_port_facts = [dict(row) for row in advisory["payload"]["validated_facts"]]
    invalid_port_facts[0]["source_url"] = "https://issuer.example:99999/disclosure"
    with pytest.raises(IntelligenceError, match="invalid"):
        review_advisory(
            decision=decision,
            proposed_percentile="0.96",
            validated_facts=invalid_port_facts,
            as_of=NOW,
        )


def test_research_portfolio_paper_observation_and_graduation_never_activate() -> None:
    _, _, _, decision = paper_candidate()
    portfolio = construct_research_portfolio(
        strategy_id="research-strategy",
        decisions=[decision],
        candidate_data={"000001.SZ": {"adv_cny": "100000000", "current_weight": "0"}},
        policy={
            "cash_floor": "0.50",
            "minimum_adv_cny": "1000000",
            "per_security_cap": "0.20",
            "target_gross": "0.40",
            "target_positions": 1,
            "turnover_cap": "1",
        },
        as_of=NOW,
    )
    assert portfolio["payload"]["status"] == "AVAILABLE"
    assert portfolio["payload"]["gross_weight"] == "0.200000000000"
    assert portfolio["payload"]["run_state"] == "INACTIVE"

    observation = observe_paper_portfolio(
        portfolio=portfolio,
        as_of=NOW,
        gross_return="0.10",
        benchmark_return="0.02",
        estimated_cost="0.01",
        drawdown="0.05",
    )
    graduation = assess_graduation(
        strategy_id="research-strategy",
        observations=[observation],
        minimum_observations=1,
        minimum_excess_return="0.05",
        maximum_drawdown="0.10",
        assessed_at=NOW,
    )
    assert Decimal(observation["payload"]["excess_return"]) == Decimal("0.07")
    assert graduation["payload"]["status"] == "ELIGIBLE_FOR_OWNER_REVIEW"
    assert graduation["payload"]["production"] is False
    assert graduation["payload"]["authority"]["mainline_activation"] is False
