"""Inactive Mainline candidate assembly.

Candidate construction is deliberately below activation authority.  It binds an
exact evidence/decision/portfolio closure and never references a generation,
readiness receipt, pointer, activation receipt, broker, order, or trade.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_investor.intelligence._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    canonical_value,
    identifier,
    require_no_activation_binding,
    require_no_control_authority,
    require_no_future,
    timestamp,
    validate_artifact_ref,
)
from quant_investor.intelligence.investment_decision import (
    validate_investment_decision,
)
from quant_investor.intelligence.portfolio import validate_research_portfolio


def build_mainline_candidate(
    *,
    strategy_id: str,
    as_of: str,
    evidence_bundle: Mapping[str, Any] | bytes,
    decision: Mapping[str, Any] | bytes,
    portfolio: Mapping[str, Any] | bytes,
    result: Mapping[str, Any],
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Build one generation-agnostic, inactive Mainline release candidate."""

    strategy = identifier(strategy_id, label="strategy_id")
    cutoff = timestamp(as_of, label="as_of")
    evidence_artifact, evidence_payload = artifact_payload(
        evidence_bundle, expected_kind="evidence_bundle"
    )
    decision_artifact = validate_investment_decision(decision)
    portfolio_artifact = validate_research_portfolio(portfolio)
    for label, artifact in (
        ("evidence_bundle", evidence_artifact),
        ("investment_decision", decision_artifact),
        ("research_portfolio", portfolio_artifact),
    ):
        require_no_future(artifact, as_of=cutoff, label=label)
    decision_payload = decision_artifact["payload"]
    portfolio_payload = portfolio_artifact["payload"]
    if (
        evidence_payload.get("status") != "READY"
        or evidence_payload.get("strategy_id") != strategy
        or evidence_payload.get("authority") != NO_AUTHORITY
        or evidence_payload.get("research_only") is not True
        or evidence_payload.get("production") is not False
        or evidence_payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("mainline candidate evidence is blocked")
    if decision_payload.get("state") != "PAPER_CANDIDATE":
        raise IntelligenceError("mainline candidate requires PAPER_CANDIDATE")
    if (
        portfolio_payload.get("status") != "AVAILABLE"
        or portfolio_payload.get("strategy_id") != strategy
    ):
        raise IntelligenceError("mainline candidate portfolio is unavailable or mismatched")
    if artifact_ref(decision_artifact) not in portfolio_payload.get("decision_refs", []):
        raise IntelligenceError("mainline candidate decision is outside portfolio closure")
    if type(result) is not dict or not result:
        raise IntelligenceError("mainline candidate result must be a nonempty object")
    result_payload = dict(result)
    canonical_value(result_payload)
    require_no_control_authority(result_payload, label="result")
    require_no_activation_binding(result_payload, label="result")
    return build_artifact(
        kind="mainline_candidate",
        identity_field="candidate_id",
        identity=candidate_id
        or business_identity(
            kind="mainline_candidate",
            identity_inputs={"as_of": cutoff, "strategy_id": strategy},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "decision_ref": artifact_ref(decision_artifact),
            "evidence_bundle_ref": artifact_ref(evidence_artifact),
            "investment_state": "PAPER_CANDIDATE",
            "portfolio_ref": artifact_ref(portfolio_artifact),
            "result": result_payload,
            "status": "CANDIDATE_READY",
            "strategy_id": strategy,
        },
    )


def validate_mainline_candidate(artifact: Mapping[str, Any] | bytes) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="mainline_candidate")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("mainline candidate has forbidden authority")
    if payload.get("status") != "CANDIDATE_READY":
        raise IntelligenceError("mainline candidate status is invalid")
    if payload.get("investment_state") != "PAPER_CANDIDATE":
        raise IntelligenceError("mainline candidate investment state is invalid")
    identifier(payload.get("strategy_id"), label="strategy_id")
    timestamp(payload.get("as_of"), label="mainline_candidate.as_of")
    for field, expected_kind in (
        ("decision_ref", "investment_decision"),
        ("evidence_bundle_ref", "evidence_bundle"),
        ("portfolio_ref", "research_portfolio"),
    ):
        reference = validate_artifact_ref(payload.get(field), label=field)
        if reference["kind"] != expected_kind:
            raise IntelligenceError(f"mainline candidate {field} kind is invalid")
    if type(payload.get("result")) is not dict or not payload["result"]:
        raise IntelligenceError("mainline candidate result is invalid")
    canonical_value(payload["result"])
    require_no_control_authority(payload["result"], label="result")
    require_no_activation_binding(payload["result"], label="result")
    return normalized


__all__ = ["build_mainline_candidate", "validate_mainline_candidate"]
