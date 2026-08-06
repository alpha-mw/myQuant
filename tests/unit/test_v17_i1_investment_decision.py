from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_investor.intelligence._core import (
    ZERO_SHA256,
    content_ref,
    seal_content_addressed,
)
from quant_investor.intelligence.bayesian import update_hypothesis
from quant_investor.intelligence.decision import (
    DecisionContractError,
    append_decision_discipline,
    assess_investment_risk,
    build_context_note,
    build_decision_policy,
    build_investment_memo,
    build_paper_intake_proposal,
    collect_investment_decision_context,
    make_investment_decision,
    validate_context_note,
    validate_decision_discipline_chain,
    validate_decision_policy,
    validate_investment_decision_context,
    validate_investment_decision_receipt,
    validate_investment_memo,
    validate_paper_intake_proposal,
    validate_risk_assessment_receipt,
)
from quant_investor.intelligence.evidence import build_ai_draft
from quant_investor.intelligence.evidence.forward_adapter import (
    build_observation_evidence_bundle,
)
from quant_investor.intelligence.fusion import (
    build_fundamental_branch,
    build_quant_branch,
    fuse_research_branches,
)
from quant_investor.intelligence.hypothesis import build_hypothesis
from quant_investor.intelligence.memory import append_memory, memory_tip
from quant_investor.intelligence.regime import infer_multilayer_regime
from quant_investor.intelligence.evaluator.forward_evaluator import (
    run_forward_research_evaluation,
)
from quant_investor.intelligence.decision import evidence_collector
from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from tests.unit import test_v17_i0_investment_intelligence as i0
from tests.unit import test_v17_r22_forward_research_evaluator as r22

AS_OF = i0.AS_OF
REVIEW_DUE_AT = "2026-01-09T07:00:00Z"
COMMON_FIELDS = {
    "authority",
    "broker",
    "decision_protocol",
    "execution",
    "mainline_authority",
    "operational_activation_unchanged",
    "order",
    "production",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "trade",
    "version",
}
REQUIREMENT_CLASSES = {
    "AI_DRAFT",
    "INDUSTRY_CONTEXT",
    "R22_EVALUATION",
    "THEME_CONTEXT",
    "VALUATION_CONTEXT",
    "WHY_NOW",
}
RISK_DIMENSIONS = ("BUSINESS", "FINANCIAL", "MARKET", "THESIS")


def _later(value: str, *, seconds: int = 1) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (
        (parsed + timedelta(seconds=seconds))
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _reseal(document: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    body = deepcopy(dict(document))
    body.pop(identity_field, None)
    body.pop("semantic_sha256", None)
    return seal_content_addressed(body, identity_field=identity_field)


def _policy(
    *,
    research_required_classes: tuple[str, ...] = (),
    paper_required_classes: tuple[str, ...] = (),
    research_required_risk_dimensions: tuple[str, ...] = RISK_DIMENSIONS,
    paper_required_risk_dimensions: tuple[str, ...] = RISK_DIMENSIONS,
    min_research_confidence: str = "0",
    min_paper_confidence: str = "0",
    min_research_posterior: str = "0",
    min_paper_posterior: str = "0",
    max_research_risk: str = "1",
    max_paper_risk: str = "1",
    hard_veto_severity: str = "0.8",
    require_r22_supported_for_research: bool = False,
    require_r22_supported_for_paper: bool = False,
) -> dict[str, Any]:
    return build_decision_policy(
        created_at=AS_OF,
        research_required_classes=research_required_classes,
        paper_required_classes=paper_required_classes,
        research_required_risk_dimensions=research_required_risk_dimensions,
        paper_required_risk_dimensions=paper_required_risk_dimensions,
        min_research_confidence=min_research_confidence,
        min_paper_confidence=min_paper_confidence,
        min_research_posterior=min_research_posterior,
        min_paper_posterior=min_paper_posterior,
        max_research_risk=max_research_risk,
        max_paper_risk=max_paper_risk,
        hard_veto_severity=hard_veto_severity,
        require_r22_supported_for_research=require_r22_supported_for_research,
        require_r22_supported_for_paper=require_r22_supported_for_paper,
        max_review_delay_seconds=31_536_000,
    )


def _i0_replay_inputs(root: Path) -> dict[str, Any]:
    session_path, session_sha, observations, _, _, closure_refs = i0._forward_closure(root)
    closure_refs = closure_refs[:6]
    bundle = build_observation_evidence_bundle(
        workspace_root=str(root),
        session_relative_path=session_path,
        session_byte_sha256=session_sha,
        observation_refs=observations,
        closure_refs=closure_refs,
        label_refs=[],
        evaluation_refs=[],
        as_of=AS_OF,
    )
    source_a_ref = next(ref for ref in closure_refs if ref["artifact_id"] == "source-a")
    source_b_ref = next(ref for ref in closure_refs if ref["artifact_id"] == "source-b")
    positive = i0._evidence(source_ref=source_a_ref)
    contrary = i0._evidence("CONTRARY", name="source-b", source_ref=source_b_ref)
    hypothesis = i0._hypothesis(positive, contrary)
    bayesian = update_hypothesis(
        hypothesis_id=hypothesis["hypothesis_id"],
        prior="0.5",
        evidence=[positive, contrary],
        as_of=AS_OF,
    )
    regime_input = i0._regime_input(source_ref=source_a_ref)
    regime = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=[positive, contrary],
        as_of=AS_OF,
    )
    quant, fundamental = i0._branches(positive, contrary)
    fusion = fuse_research_branches(branches=[quant, fundamental], as_of=AS_OF)
    memory = append_memory(
        (),
        event_type="HYPOTHESIS_CREATED",
        status="ACTIVE",
        subject_id=hypothesis["hypothesis_id"],
        summary="Created",
        artifact_refs=[content_ref(hypothesis, identity_field="hypothesis_id")],
        timestamp_value=AS_OF,
        expected_tip=ZERO_SHA256,
    )
    return {
        "observation_bundle": bundle,
        "workspace_root": str(root),
        "session_relative_path": session_path,
        "session_byte_sha256": session_sha,
        "observation_refs": observations,
        "closure_refs": closure_refs,
        "evidence": [positive, contrary],
        "bayesian_receipts": [bayesian],
        "regime_input": regime_input,
        "regime_receipt": regime,
        "branches": [quant, fundamental],
        "fusion_receipt": fusion,
        "hypotheses": [hypothesis],
        "memory_entries": memory,
        "expected_memory_tip": memory_tip(memory),
        "label_refs": [],
        "evaluation_refs": [],
    }


def _r22_i0_replay_inputs(
    root: Path, *, failed: bool = False
) -> tuple[dict[str, Any], str, str, str]:
    request_path, request_sha = r22._end_to_end_request(root)
    request = json.loads((root / request_path).read_text(encoding="utf-8"))
    origin = request["origins"][0]
    hypothesis_ref = request["policy"]["hypothesis_specs"][0]["hypothesis_ref"]
    hypothesis = json.loads((root / hypothesis_ref["relative_path"]).read_text(encoding="utf-8"))
    evidence = [
        json.loads((root / ref["relative_path"]).read_text(encoding="utf-8"))
        for ref in request["policy"]["hypothesis_specs"][0]["evidence_refs"]
    ]
    if failed:
        conditions = deepcopy(hypothesis["falsification_conditions"])
        conditions[0]["threshold"] = "2"
        hypothesis = build_hypothesis(
            thesis=hypothesis["thesis"],
            why_it_may_be_true=hypothesis["why_it_may_be_true"],
            what_would_make_it_fail=hypothesis["what_would_make_it_fail"],
            supporting_evidence=[
                row
                for row in evidence
                if content_ref(row, identity_field="evidence_id")
                in hypothesis["supporting_evidence_refs"]
            ],
            contrary_evidence=[
                row
                for row in evidence
                if content_ref(row, identity_field="evidence_id")
                in hypothesis["contrary_evidence_refs"]
            ],
            expected_window_start=hypothesis["expected_window"]["start"],
            expected_window_end=hypothesis["expected_window"]["end"],
            falsification_conditions=conditions,
            related_companies=hypothesis["related_companies"],
            related_industries=hypothesis["related_industries"],
            as_of=r22.AS_OF,
        )
        hypothesis_ref = r22._research_input_ref(root, hypothesis, identity_field="hypothesis_id")
        request["policy"]["hypothesis_specs"][0]["hypothesis_ref"] = hypothesis_ref
        policy_body = deepcopy(request["policy"])
        policy_body.pop("policy_id")
        policy_body.pop("semantic_sha256")
        request["policy"] = seal_content_addressed(policy_body, identity_field="policy_id")
        request_body = deepcopy(request)
        request_body.pop("request_id")
        request_body.pop("semantic_sha256")
        request_id = (
            "forward-evaluation-request-"
            + hashlib.sha256(canonical_bytes(request_body)).hexdigest()
        )
        request = seal_semantic({**request_body, "request_id": request_id})
        request_path = (
            "data/private/research_intelligence/evaluation_requests/" f"{request_id}.json"
        )
        raw_request = canonical_resource_bytes(request)
        r22._write_canonical(root, request_path, request)
        request_sha = hashlib.sha256(raw_request).hexdigest()
    observation_refs = sorted(
        {
            tuple(sorted(binding["observation_ref"].items()))
            for binding in origin["factor_observation_bindings"]
        }
        | {tuple(sorted(origin["universe_observation_ref"].items()))}
    )
    observation_refs = [dict(items) for items in observation_refs]
    closure_refs = [ref for ref in origin["closure_refs"] if ref["cutoff"] <= r22.AS_OF]
    bundle = build_observation_evidence_bundle(
        workspace_root=str(root),
        session_relative_path=origin["session_relative_path"],
        session_byte_sha256=origin["session_byte_sha256"],
        observation_refs=observation_refs,
        closure_refs=closure_refs,
        label_refs=[],
        evaluation_refs=[],
        as_of=r22.AS_OF,
    )
    source_ref = next(
        ref
        for ref in closure_refs
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    bayesian = update_hypothesis(
        hypothesis_id=hypothesis["hypothesis_id"],
        prior="0.5",
        evidence=evidence,
        as_of=r22.AS_OF,
    )
    regime_input = i0._regime_input(source_ref=source_ref)
    regime = infer_multilayer_regime(
        regime_input=regime_input,
        evidence=evidence,
        as_of=r22.AS_OF,
    )
    positive = next(row for row in evidence if row["direction"] == "POSITIVE")
    contrary = next(row for row in evidence if row["direction"] == "CONTRARY")
    quant = build_quant_branch(
        factor_score="0.8",
        rank_ic="0.1",
        icir="0.2",
        exposure="0.3",
        coverage="0.9",
        confidence="0.8",
        availability="1",
        evidence=[positive],
        as_of=r22.AS_OF,
    )
    fundamental = build_fundamental_branch(
        quality="0.6",
        earnings="0.5",
        valuation="0.4",
        industry_position="0.7",
        confidence="0.5",
        availability="0.5",
        evidence=[contrary],
        as_of=r22.AS_OF,
    )
    fusion = fuse_research_branches(branches=[quant, fundamental], as_of=r22.AS_OF)
    memory = append_memory(
        (),
        event_type="HYPOTHESIS_CREATED",
        status="ACTIVE",
        subject_id=hypothesis["hypothesis_id"],
        summary="Created",
        artifact_refs=[content_ref(hypothesis, identity_field="hypothesis_id")],
        timestamp_value=r22.AS_OF,
        expected_tip=ZERO_SHA256,
    )
    replay = {
        "observation_bundle": bundle,
        "workspace_root": str(root),
        "session_relative_path": origin["session_relative_path"],
        "session_byte_sha256": origin["session_byte_sha256"],
        "observation_refs": observation_refs,
        "closure_refs": closure_refs,
        "evidence": evidence,
        "bayesian_receipts": [bayesian],
        "regime_input": regime_input,
        "regime_receipt": regime,
        "branches": [quant, fundamental],
        "fusion_receipt": fusion,
        "hypotheses": [hypothesis],
        "memory_entries": memory,
        "expected_memory_tip": memory_tip(memory),
        "label_refs": [],
        "evaluation_refs": [],
    }
    return replay, request_path, request_sha, hypothesis_ref["relative_path"]


def _notes(replay: Mapping[str, Any], *, include_display_name: bool = True) -> tuple[dict, ...]:
    source_ref = next(
        ref
        for ref in replay["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    rows = []
    if include_display_name:
        rows.append(
            build_context_note(
                kind="COMPANY_DISPLAY_NAME",
                company_code="000001.SZ",
                text="平安银行",
                observed_at=AS_OF,
                available_at=AS_OF,
                source_ref=source_ref,
            )
        )
    for kind, text in (
        ("WHY_NOW", "当前证据窗口值得进入持续研究。"),
        ("INDUSTRY_CONTEXT", "银行业资产质量保持稳定。"),
        ("THEME_CONTEXT", "顺周期主题仍需后续验证。"),
        ("VALUATION_CONTEXT", "估值证据来自已绑定研究来源。"),
    ):
        rows.append(
            build_context_note(
                kind=kind,
                company_code="000001.SZ",
                text=text,
                observed_at=AS_OF,
                available_at=AS_OF,
                source_ref=source_ref,
            )
        )
    return tuple(rows)


def _ai_drafts(replay: Mapping[str, Any]) -> tuple[dict, ...]:
    source_ref = next(
        ref
        for ref in replay["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    return (
        build_ai_draft(
            kind="SUMMARY",
            payload={"summary": "Evidence-bound summary only."},
            source_refs=[source_ref],
            generated_at=AS_OF,
            confidence="0.7",
        ),
        build_ai_draft(
            kind="EXTRACTION",
            payload={"facts": ["Fact A", "Fact B"]},
            source_refs=[source_ref],
            generated_at=AS_OF,
            confidence="0.6",
        ),
        build_ai_draft(
            kind="CONTRARY_EVIDENCE_DRAFT",
            payload={"contrary_points": ["Contrary point A"]},
            source_refs=[source_ref],
            generated_at=AS_OF,
            confidence="0.5",
        ),
    )


def _context(
    replay: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    notes: tuple[dict, ...] = (),
    ai_drafts: tuple[dict, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = collect_investment_decision_context(
        i0_replay_inputs=replay,
        policy=policy,
        company_code="000001.SZ",
        as_of=AS_OF,
        review_due_at=REVIEW_DUE_AT,
        context_notes=notes,
        ai_drafts=ai_drafts,
    )
    closure = {
        "i0_replay_inputs": replay,
        "policy": policy,
        "context_notes": notes,
        "ai_drafts": ai_drafts,
        "r22_request_path": None,
        "r22_request_sha256": None,
    }
    return context, closure


def _assessment(
    evidence: Mapping[str, Any],
    *,
    severity: str = "0.1",
    reason: str = "Bounded research risk.",
    kind: str = "RISK_IDENTIFIED",
    hard_veto_code: str | None = None,
) -> dict[str, Any]:
    normalized_severity = format(Decimal(severity).quantize(Decimal("0.000000000001")), "f")
    body = {
        "kind": kind,
        "severity": normalized_severity,
        "reason": reason,
        "evidence_refs": [content_ref(evidence, identity_field="evidence_id")],
        "source_refs": [],
        "hard_veto_code": hard_veto_code,
    }
    return {
        "assessment_id": hashlib.sha256(canonical_bytes(body)).hexdigest(),
        **body,
    }


def _assessments(
    replay: Mapping[str, Any],
    *,
    unavailable: tuple[str, ...] = (),
    severity_by_dimension: Mapping[str, str] | None = None,
    veto_dimension: str | None = None,
) -> dict[str, Any]:
    evidence = replay["evidence"][0]
    severity_by_dimension = severity_by_dimension or {}
    result: dict[str, Any] = {}
    for dimension in RISK_DIMENSIONS:
        if dimension in unavailable:
            result[dimension] = {"status": "UNAVAILABLE", "assessments": []}
        else:
            severity = severity_by_dimension.get(dimension, "0.1")
            result[dimension] = {
                "status": "AVAILABLE",
                "assessments": [
                    _assessment(
                        evidence,
                        severity=severity,
                        reason=f"{dimension.lower()} risk.",
                        hard_veto_code=(
                            f"{dimension}_VETO" if dimension == veto_dimension else None
                        ),
                    )
                ],
            }
    return result


def _decision_stack(
    root: Path,
    *,
    policy: Mapping[str, Any] | None = None,
    notes: tuple[dict, ...] | None = None,
    ai_drafts: tuple[dict, ...] | None = None,
    assessments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    replay = _i0_replay_inputs(root)
    policy = _policy() if policy is None else policy
    notes = _notes(replay) if notes is None else notes
    ai_drafts = _ai_drafts(replay) if ai_drafts is None else ai_drafts
    context, context_closure = _context(
        replay,
        policy,
        notes=notes,
        ai_drafts=ai_drafts,
    )
    assessments = _assessments(replay) if assessments is None else assessments
    risk = assess_investment_risk(
        context=context,
        context_replay_closure=context_closure,
        policy=policy,
        assessments_by_dimension=assessments,
        as_of=AS_OF,
    )
    decision = make_investment_decision(
        context=context,
        context_replay_closure=context_closure,
        policy=policy,
        risk_receipt=risk,
        assessments_by_dimension=assessments,
        as_of=AS_OF,
    )
    decision_closure = {
        "context": context,
        "context_replay_closure": context_closure,
        "policy": policy,
        "risk_receipt": risk,
        "assessments_by_dimension": assessments,
        "as_of": AS_OF,
    }
    return {
        "replay": replay,
        "policy": policy,
        "notes": notes,
        "ai_drafts": ai_drafts,
        "context": context,
        "context_closure": context_closure,
        "assessments": assessments,
        "risk": risk,
        "decision": decision,
        "decision_closure": decision_closure,
    }


def test_policy_is_deterministic_closed_and_content_addressed() -> None:
    policy = _policy(
        research_required_classes=("WHY_NOW",),
        paper_required_classes=("VALUATION_CONTEXT", "WHY_NOW"),
        research_required_risk_dimensions=("BUSINESS",),
        paper_required_risk_dimensions=("BUSINESS", "FINANCIAL"),
        min_research_confidence="0.4",
        min_paper_confidence="0.6",
        min_research_posterior="0.3",
        min_paper_posterior="0.5",
        max_research_risk="0.7",
        max_paper_risk="0.4",
    )
    assert validate_decision_policy(policy) == policy
    assert (
        _policy(
            research_required_classes=("WHY_NOW",),
            paper_required_classes=("WHY_NOW", "VALUATION_CONTEXT"),
            research_required_risk_dimensions=("BUSINESS",),
            paper_required_risk_dimensions=("FINANCIAL", "BUSINESS"),
            min_research_confidence="0.400000000000",
            min_paper_confidence="0.600000000000",
            min_research_posterior="0.300000000000",
            min_paper_posterior="0.500000000000",
            max_research_risk="0.700000000000",
            max_paper_risk="0.400000000000",
        )
        == policy
    )
    assert COMMON_FIELDS | {"policy_id"} <= set(policy)
    assert policy["research_required_classes"] == ["WHY_NOW"]
    assert policy["paper_required_classes"] == ["VALUATION_CONTEXT", "WHY_NOW"]

    malformed = deepcopy(policy)
    malformed["unexpected"] = True
    with pytest.raises(DecisionContractError) as exc_info:
        validate_decision_policy(_reseal(malformed, identity_field="policy_id"))
    assert exc_info.value.code == "I1_SHAPE_INVALID"


@pytest.mark.parametrize(
    "override",
    [
        {"research_required_classes": ("NOT_A_CLASS",)},
        {"research_required_classes": ("WHY_NOW",), "paper_required_classes": ()},
        {"min_research_confidence": "0.8", "min_paper_confidence": "0.7"},
        {"min_research_posterior": "0.8", "min_paper_posterior": "0.7"},
        {"max_research_risk": "0.2", "max_paper_risk": "0.3"},
    ],
)
def test_policy_rejects_invalid_tiers(override: dict[str, Any]) -> None:
    with pytest.raises(DecisionContractError) as exc_info:
        _policy(**override)
    assert exc_info.value.code == "I1_POLICY_INVALID"


def test_context_note_is_source_bound_and_replay_validated(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    source_refs = [
        ref
        for ref in replay["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    ]
    note = build_context_note(
        kind="WHY_NOW",
        company_code="000001.SZ",
        text="A source-bound reason.",
        observed_at=AS_OF,
        available_at=AS_OF,
        source_ref=source_refs[0],
    )
    assert (
        validate_context_note(
            note,
            as_of=AS_OF,
            authorized_source_refs=source_refs,
        )
        == note
    )
    assert COMMON_FIELDS | {"note_id"} <= set(note)

    with pytest.raises(DecisionContractError) as exc_info:
        validate_context_note(note, as_of=AS_OF, authorized_source_refs=source_refs[1:])
    assert exc_info.value.code == "I1_REF_MISMATCH"
    with pytest.raises(DecisionContractError) as exc_info:
        validate_context_note(
            note, as_of="2026-01-01T07:00:00Z", authorized_source_refs=source_refs
        )
    assert exc_info.value.code == "I1_FUTURE_INPUT"


def test_context_replays_complete_i0_closure_and_binds_exact_components(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    policy = _policy()
    notes = _notes(replay)
    drafts = _ai_drafts(replay)
    context, closure = _context(replay, policy, notes=notes, ai_drafts=drafts)

    assert validate_investment_decision_context(context, **closure) == context
    assert COMMON_FIELDS | {"context_id"} <= set(context)
    assert context["company_code"] == "000001.SZ"
    assert context["review_due_at"] == REVIEW_DUE_AT
    assert context["r22_hypothesis_status"] is None
    assert context["availability"]["AI_DRAFT"]["status"] == "AVAILABLE"
    assert context["availability"]["R22_EVALUATION"] == {
        "status": "UNAVAILABLE",
        "refs": [],
    }
    assert context["hypothesis_ref"] == content_ref(
        replay["hypotheses"][0], identity_field="hypothesis_id"
    )
    assert context["fusion_ref"] == content_ref(
        replay["fusion_receipt"], identity_field="receipt_id"
    )
    assert context["evidence_refs"] == sorted(
        [content_ref(row, identity_field="evidence_id") for row in replay["evidence"]],
        key=lambda ref: (
            ref["artifact_id"],
            ref["artifact_version"],
            ref["byte_sha256"],
            ref["semantic_sha256"],
        ),
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "as_of"])
def test_context_rejects_open_or_malformed_i0_replay_closure(tmp_path: Path, mutation: str) -> None:
    replay = _i0_replay_inputs(tmp_path)
    if mutation == "missing":
        replay.pop("label_refs")
    elif mutation == "extra":
        replay["runtime_receipt"] = {}
    else:
        replay["as_of"] = AS_OF
    with pytest.raises(DecisionContractError) as exc_info:
        collect_investment_decision_context(
            i0_replay_inputs=replay,
            policy=_policy(),
            company_code="000001.SZ",
            as_of=AS_OF,
            review_due_at=REVIEW_DUE_AT,
        )
    assert exc_info.value.code == "I1_SHAPE_INVALID"


def test_context_validator_rejects_resealed_component_forgery(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    policy = _policy()
    context, closure = _context(replay, policy)
    forged = deepcopy(context)
    forged["fusion_ref"] = deepcopy(forged["bayesian_ref"])
    forged = _reseal(forged, identity_field="context_id")
    with pytest.raises(DecisionContractError) as exc_info:
        validate_investment_decision_context(forged, **closure)
    assert exc_info.value.code in {"I1_REF_MISMATCH", "I1_REPLAY_MISMATCH"}


def test_context_replays_optional_r22_and_rejects_referenced_input_forgery(
    tmp_path: Path,
) -> None:
    replay, request_path, request_sha, hypothesis_path = _r22_i0_replay_inputs(tmp_path)
    policy = _policy(
        research_required_classes=("R22_EVALUATION",),
        paper_required_classes=("R22_EVALUATION",),
        require_r22_supported_for_research=True,
        require_r22_supported_for_paper=True,
    )
    notes = _notes(replay)
    review_due = _later(r22.EVALUATED_AT, seconds=7 * 24 * 60 * 60)
    context = collect_investment_decision_context(
        i0_replay_inputs=replay,
        policy=policy,
        company_code="000001.SZ",
        as_of=r22.EVALUATED_AT,
        review_due_at=review_due,
        context_notes=notes,
        r22_request_path=request_path,
        r22_request_sha256=request_sha,
    )
    closure = {
        "i0_replay_inputs": replay,
        "policy": policy,
        "context_notes": notes,
        "ai_drafts": (),
        "r22_request_path": request_path,
        "r22_request_sha256": request_sha,
    }
    assert validate_investment_decision_context(context, **closure) == context
    assert context["availability"]["R22_EVALUATION"]["status"] == "AVAILABLE"
    assert context["r22_hypothesis_status"] == "SUPPORTED"
    assert context["r22_envelope_ref"] is not None
    assert context["r22_main_ref"] is not None
    assert context["r22_hypothesis_evaluation_ref"] is not None

    path = tmp_path / hypothesis_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DecisionContractError) as exc_info:
        collect_investment_decision_context(
            i0_replay_inputs=replay,
            policy=policy,
            company_code="000001.SZ",
            as_of=r22.EVALUATED_AT,
            review_due_at=review_due,
            context_notes=notes,
            r22_request_path=request_path,
            r22_request_sha256=request_sha,
        )
    assert exc_info.value.code == "I1_R22_CLOSURE_INVALID"


@pytest.mark.parametrize("mutation", ["duplicate", "orphan", "substitution"])
def test_context_rejects_r22_duplicate_orphan_and_substitution_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    replay, request_path, request_sha, _ = _r22_i0_replay_inputs(tmp_path)
    envelope = run_forward_research_evaluation(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    invalid = deepcopy(envelope)
    if mutation == "duplicate":
        invalid["hypothesis_evaluations"].append(deepcopy(invalid["hypothesis_evaluations"][0]))
    elif mutation == "orphan":
        orphan = deepcopy(invalid["hypothesis_evaluations"][0])
        orphan["hypothesis_ref"] = deepcopy(orphan["hypothesis_ref"])
        orphan["hypothesis_ref"]["artifact_id"] = "f" * 64
        orphan = _reseal(orphan, identity_field="receipt_id")
        invalid["hypothesis_evaluations"].append(orphan)
    else:
        main = deepcopy(invalid["main_receipt"])
        main["hypothesis_evaluation_refs"][0] = deepcopy(main["hypothesis_evaluation_refs"][0])
        main["hypothesis_evaluation_refs"][0]["artifact_id"] = "f" * 64
        invalid["main_receipt"] = _reseal(main, identity_field="evaluation_id")
    invalid = _reseal(invalid, identity_field="envelope_id")
    monkeypatch.setattr(
        evidence_collector,
        "run_forward_research_evaluation",
        lambda *_args, **_kwargs: invalid,
    )
    policy = _policy()
    with pytest.raises(DecisionContractError) as exc_info:
        collect_investment_decision_context(
            i0_replay_inputs=replay,
            policy=policy,
            company_code="000001.SZ",
            as_of=r22.EVALUATED_AT,
            review_due_at=_later(r22.EVALUATED_AT, seconds=24 * 60 * 60),
            r22_request_path=request_path,
            r22_request_sha256=request_sha,
        )
    assert exc_info.value.code == "I1_R22_CLOSURE_INVALID"


def test_preregistered_r22_failure_has_absolute_priority_and_retains_diagnostics(
    tmp_path: Path,
) -> None:
    replay, request_path, request_sha, _ = _r22_i0_replay_inputs(tmp_path, failed=True)
    policy = _policy(
        research_required_classes=("AI_DRAFT", "R22_EVALUATION"),
        paper_required_classes=("AI_DRAFT", "R22_EVALUATION"),
    )
    context = collect_investment_decision_context(
        i0_replay_inputs=replay,
        policy=policy,
        company_code="000001.SZ",
        as_of=r22.EVALUATED_AT,
        review_due_at=_later(r22.EVALUATED_AT, seconds=24 * 60 * 60),
        r22_request_path=request_path,
        r22_request_sha256=request_sha,
    )
    context_closure = {
        "i0_replay_inputs": replay,
        "policy": policy,
        "context_notes": (),
        "ai_drafts": (),
        "r22_request_path": request_path,
        "r22_request_sha256": request_sha,
    }
    assessments = _assessments(
        replay,
        unavailable=("BUSINESS",),
        severity_by_dimension={"THESIS": "0.8"},
        veto_dimension="THESIS",
    )
    risk = assess_investment_risk(
        context=context,
        context_replay_closure=context_closure,
        policy=policy,
        assessments_by_dimension=assessments,
        as_of=r22.EVALUATED_AT,
    )
    decision = make_investment_decision(
        context=context,
        context_replay_closure=context_closure,
        policy=policy,
        risk_receipt=risk,
        assessments_by_dimension=assessments,
        as_of=r22.EVALUATED_AT,
    )
    assert context["r22_hypothesis_status"] == "FAILED"
    assert decision["state"] == "THESIS_INVALIDATED"
    assert "PREREGISTERED_HYPOTHESIS_FAILED" in decision["reason_codes"]
    assert "HARD_RISK_VETO" in decision["reason_codes"]
    assert "PAPER_REQUIRED_INPUT_UNAVAILABLE" in decision["reason_codes"]
    assert "RESEARCH_REQUIRED_AI_DRAFT_UNAVAILABLE" in decision["blocker_codes"]
    assert "RESEARCH_REQUIRED_RISK_BUSINESS_UNAVAILABLE" in decision["blocker_codes"]


def test_risk_receipt_recomputes_ids_sorts_and_aggregates_max(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    policy = _policy()
    context, closure = _context(replay, policy)
    assessments = _assessments(
        replay,
        severity_by_dimension={
            "BUSINESS": "0.2",
            "FINANCIAL": "0.7",
            "MARKET": "0.3",
            "THESIS": "0.4",
        },
    )
    second = _assessment(
        replay["evidence"][0],
        severity="0.1",
        reason="Second business risk.",
    )
    assessments["BUSINESS"]["assessments"].insert(0, second)
    receipt = assess_investment_risk(
        context=context,
        context_replay_closure=closure,
        policy=policy,
        assessments_by_dimension=assessments,
        as_of=AS_OF,
    )
    assert (
        validate_risk_assessment_receipt(
            receipt,
            context=context,
            context_replay_closure=closure,
            policy=policy,
            assessments_by_dimension=assessments,
            as_of=AS_OF,
        )
        == receipt
    )
    assert receipt["overall_severity"] == "0.700000000000"
    business = next(row for row in receipt["dimension_rows"] if row["dimension"] == "BUSINESS")
    assert business["assessment_refs"] == sorted(business["assessment_refs"])
    assert business["dimension_severity"] == "0.200000000000"
    assert receipt["unavailable_dimensions"] == []


def test_risk_contract_handles_unavailable_no_material_and_veto_boundary(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    policy = _policy(hard_veto_severity="0.8")
    context, closure = _context(replay, policy)
    assessments = _assessments(
        replay,
        unavailable=("MARKET",),
        severity_by_dimension={"THESIS": "0.8"},
        veto_dimension="THESIS",
    )
    assessments["BUSINESS"] = {
        "status": "AVAILABLE",
        "assessments": [
            _assessment(
                replay["evidence"][0],
                severity="0",
                reason="No material business risk.",
                kind="NO_MATERIAL_RISK_IDENTIFIED",
            )
        ],
    }
    receipt = assess_investment_risk(
        context=context,
        context_replay_closure=closure,
        policy=policy,
        assessments_by_dimension=assessments,
        as_of=AS_OF,
    )
    assert receipt["hard_veto_codes"] == ["THESIS_VETO"]
    assert receipt["unavailable_dimensions"] == ["MARKET"]
    assert receipt["overall_severity"] == "0.800000000000"

    forged = deepcopy(receipt)
    forged["overall_severity"] = "0.100000000000"
    forged = _reseal(forged, identity_field="risk_receipt_id")
    with pytest.raises(DecisionContractError) as exc_info:
        validate_risk_assessment_receipt(
            forged,
            context=context,
            context_replay_closure=closure,
            policy=policy,
            assessments_by_dimension=assessments,
            as_of=AS_OF,
        )
    assert exc_info.value.code == "I1_REPLAY_MISMATCH"


def test_decision_five_state_truth_table_and_priority(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    notes = _notes(replay)
    drafts = _ai_drafts(replay)

    insufficient_policy = _policy(
        research_required_classes=("R22_EVALUATION",),
        paper_required_classes=("R22_EVALUATION",),
    )
    insufficient = _decision_stack(
        tmp_path / "insufficient",
        policy=insufficient_policy,
        notes=(),
        ai_drafts=(),
    )
    assert insufficient["decision"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert (
        "RESEARCH_REQUIRED_R22_EVALUATION_UNAVAILABLE" in insufficient["decision"]["blocker_codes"]
    )

    watch_policy = _policy(min_research_confidence="1", min_paper_confidence="1")
    watch = _decision_stack(tmp_path / "watch", policy=watch_policy, notes=notes, ai_drafts=drafts)
    assert watch["decision"]["state"] == "WATCHLIST"
    assert "RESEARCH_CONFIDENCE_BELOW_MIN" in watch["decision"]["reason_codes"]

    approved_policy = _policy(
        paper_required_classes=("R22_EVALUATION",),
    )
    approved = _decision_stack(
        tmp_path / "approved", policy=approved_policy, notes=notes, ai_drafts=drafts
    )
    assert approved["decision"]["state"] == "RESEARCH_APPROVED"
    assert "PAPER_REQUIRED_INPUT_UNAVAILABLE" in approved["decision"]["reason_codes"]

    candidate = _decision_stack(tmp_path / "candidate")
    assert candidate["decision"]["state"] == "PAPER_CANDIDATE"
    assert "PAPER_GATES_PASSED" in candidate["decision"]["reason_codes"]

    unavailable_with_veto = _assessments(
        replay,
        unavailable=("BUSINESS",),
        severity_by_dimension={"THESIS": "0.8"},
        veto_dimension="THESIS",
    )
    priority = _decision_stack(
        tmp_path / "priority",
        assessments=unavailable_with_veto,
    )
    assert priority["decision"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert "HARD_RISK_VETO" in priority["decision"]["reason_codes"]
    assert "RESEARCH_REQUIRED_RISK_BUSINESS_UNAVAILABLE" in priority["decision"]["blocker_codes"]


def test_decision_validator_replays_full_closure_and_rejects_resealed_state(
    tmp_path: Path,
) -> None:
    stack = _decision_stack(tmp_path)
    assert (
        validate_investment_decision_receipt(stack["decision"], **stack["decision_closure"])
        == stack["decision"]
    )
    forged = deepcopy(stack["decision"])
    forged["state"] = "WATCHLIST"
    forged = _reseal(forged, identity_field="decision_receipt_id")
    with pytest.raises(DecisionContractError) as exc_info:
        validate_investment_decision_receipt(forged, **stack["decision_closure"])
    assert exc_info.value.code == "I1_REPLAY_MISMATCH"


def test_confidence_posterior_and_risk_threshold_equality_passes(tmp_path: Path) -> None:
    baseline = _decision_stack(tmp_path / "baseline")
    exact_policy = _policy(
        min_research_confidence=baseline["decision"]["research_confidence"],
        min_paper_confidence=baseline["decision"]["research_confidence"],
        min_research_posterior=baseline["decision"]["bayesian_posterior"],
        min_paper_posterior=baseline["decision"]["bayesian_posterior"],
        max_research_risk=baseline["risk"]["overall_severity"],
        max_paper_risk=baseline["risk"]["overall_severity"],
        hard_veto_severity="1",
    )
    exact = _decision_stack(tmp_path / "exact", policy=exact_policy)
    assert exact["decision"]["state"] == "PAPER_CANDIDATE"
    assert "RESEARCH_CONFIDENCE_BELOW_MIN" not in exact["decision"]["reason_codes"]
    assert "RESEARCH_POSTERIOR_BELOW_MIN" not in exact["decision"]["reason_codes"]
    assert "RESEARCH_RISK_ABOVE_MAX" not in exact["decision"]["reason_codes"]
    assert "PAPER_CONFIDENCE_BELOW_MIN" not in exact["decision"]["reason_codes"]
    assert "PAPER_POSTERIOR_BELOW_MIN" not in exact["decision"]["reason_codes"]
    assert "PAPER_RISK_ABOVE_MAX" not in exact["decision"]["reason_codes"]


def test_memo_is_deterministic_projection_without_invented_control_fields(
    tmp_path: Path,
) -> None:
    stack = _decision_stack(tmp_path)
    memo = build_investment_memo(
        context=stack["context"],
        context_replay_closure=stack["context_closure"],
        policy=stack["policy"],
        risk_receipt=stack["risk"],
        assessments_by_dimension=stack["assessments"],
        decision_receipt=stack["decision"],
        as_of=AS_OF,
    )
    assert (
        validate_investment_memo(
            memo,
            context=stack["context"],
            context_replay_closure=stack["context_closure"],
            policy=stack["policy"],
            risk_receipt=stack["risk"],
            assessments_by_dimension=stack["assessments"],
            decision_receipt=stack["decision"],
            as_of=AS_OF,
        )
        == memo
    )
    hypothesis = stack["replay"]["hypotheses"][0]
    assert memo["investment_thesis"] == hypothesis["thesis"]
    assert memo["why_invest"] == hypothesis["why_it_may_be_true"]
    assert memo["company_display_name"] == "平安银行"
    assert memo["decision_state"] == stack["decision"]["state"]
    assert memo["supporting_evidence"] and memo["contrary_evidence"]
    assert {row["reason"] for row in memo["supporting_evidence"]} <= {
        row["reason"] for row in stack["replay"]["evidence"]
    }
    forbidden = {"action", "cash", "holding", "quantity", "side", "weight"}
    assert forbidden.isdisjoint(memo)

    forged = deepcopy(memo)
    forged["decision_state"] = "WATCHLIST"
    forged = _reseal(forged, identity_field="memo_id")
    with pytest.raises(DecisionContractError) as exc_info:
        validate_investment_memo(
            forged,
            context=stack["context"],
            context_replay_closure=stack["context_closure"],
            policy=stack["policy"],
            risk_receipt=stack["risk"],
            assessments_by_dimension=stack["assessments"],
            decision_receipt=stack["decision"],
            as_of=AS_OF,
        )
    assert exc_info.value.code == "I1_REPLAY_MISMATCH"


def test_memo_why_now_is_null_when_no_validated_note_exists(tmp_path: Path) -> None:
    replay = _i0_replay_inputs(tmp_path)
    notes = tuple(note for note in _notes(replay) if note["kind"] != "WHY_NOW")
    stack = _decision_stack(tmp_path, notes=notes)
    memo = build_investment_memo(
        context=stack["context"],
        context_replay_closure=stack["context_closure"],
        policy=stack["policy"],
        risk_receipt=stack["risk"],
        assessments_by_dimension=stack["assessments"],
        decision_receipt=stack["decision"],
        as_of=AS_OF,
    )
    assert memo["why_now"] is None


def test_paper_proposal_requires_replayed_paper_candidate(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    proposed_at = _later(AS_OF)
    proposal = build_paper_intake_proposal(
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        proposed_at=proposed_at,
    )
    assert (
        validate_paper_intake_proposal(
            proposal,
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            proposed_at=proposed_at,
        )
        == proposal
    )
    assert proposal["status"] == "PENDING_EXTERNAL_REVIEW"
    assert set(proposal) == COMMON_FIELDS | {"proposal_id", "decision_ref", "status"}

    forged = deepcopy(proposal)
    forged["status"] = "APPROVED"
    forged = _reseal(forged, identity_field="proposal_id")
    with pytest.raises(DecisionContractError):
        validate_paper_intake_proposal(
            forged,
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            proposed_at=proposed_at,
        )


def test_paper_proposal_rejects_research_approved(tmp_path: Path) -> None:
    stack = _decision_stack(
        tmp_path,
        policy=_policy(paper_required_classes=("R22_EVALUATION",)),
    )
    assert stack["decision"]["state"] == "RESEARCH_APPROVED"
    with pytest.raises(DecisionContractError) as exc_info:
        build_paper_intake_proposal(
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            proposed_at=_later(AS_OF),
        )
    assert exc_info.value.code == "I1_AUTHORITY_OPEN"


def test_paper_proposal_rejects_resealed_candidate_forged_from_noncandidate_closure(
    tmp_path: Path,
) -> None:
    stack = _decision_stack(
        tmp_path,
        policy=_policy(paper_required_classes=("R22_EVALUATION",)),
    )
    assert stack["decision"]["state"] == "RESEARCH_APPROVED"
    forged = deepcopy(stack["decision"])
    forged["state"] = "PAPER_CANDIDATE"
    forged = _reseal(forged, identity_field="decision_receipt_id")
    with pytest.raises(DecisionContractError) as exc_info:
        build_paper_intake_proposal(
            decision_receipt=forged,
            decision_validation_closure=stack["decision_closure"],
            proposed_at=_later(AS_OF),
        )
    assert exc_info.value.code == "I1_REPLAY_MISMATCH"


def test_discipline_chain_enforces_transitions_and_terminal_state(tmp_path: Path) -> None:
    stack = _decision_stack(tmp_path)
    source_ref = next(
        ref
        for ref in stack["replay"]["closure_refs"]
        if ref["artifact_version"] == "myquant.v17.v4.research-source.v1"
    )
    outcome_ref = {
        **source_ref,
        "artifact_id": "discipline-label",
        "artifact_version": "myquant.v17.v4.forward-label.v1",
        "relative_path": "results/v17_v4_shadow/forward_labels/discipline-label.json",
    }
    history = {
        stack["decision"]["decision_receipt_id"]: {
            "decision_receipt": stack["decision"],
            "decision_validation_closure": stack["decision_closure"],
        }
    }
    entries: tuple[dict[str, Any], ...] = ()
    entries = append_decision_discipline(
        entries,
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=None,
        previous_decision_validation_closure=None,
        stage="BEFORE_DECISION",
        event_type="DECISION_CREATED",
        status="ACTIVE",
        summary="Decision recorded.",
        event_at=_later(AS_OF, seconds=1),
        expected_tip=ZERO_SHA256,
    )
    entries = append_decision_discipline(
        entries,
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=stack["decision"],
        previous_decision_validation_closure=stack["decision_closure"],
        stage="AFTER_OUTCOME",
        event_type="DECISION_REVIEWED",
        status="OUTCOME_AVAILABLE",
        summary="Outcome available.",
        event_at=_later(AS_OF, seconds=2),
        outcome_refs=(outcome_ref,),
        expected_tip=entries[-1]["semantic_sha256"],
    )
    entries = append_decision_discipline(
        entries,
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=stack["decision"],
        previous_decision_validation_closure=stack["decision_closure"],
        stage="REVIEW",
        event_type="THESIS_CONFIRMED",
        status="CONFIRMED",
        summary="Thesis confirmed.",
        event_at=_later(AS_OF, seconds=3),
        expected_tip=entries[-1]["semantic_sha256"],
    )
    entries = append_decision_discipline(
        entries,
        decision_receipt=stack["decision"],
        decision_validation_closure=stack["decision_closure"],
        previous_decision_receipt=stack["decision"],
        previous_decision_validation_closure=stack["decision_closure"],
        stage="REVIEW",
        event_type="LESSON_LEARNED",
        status="LEARNED",
        summary="Lesson retained.",
        event_at=_later(AS_OF, seconds=4),
        expected_tip=entries[-1]["semantic_sha256"],
    )
    assert validate_decision_discipline_chain(entries, decision_history_by_id=history) == entries
    assert entries[0]["evidence_changes"]["added_refs"] == stack["context"]["evidence_refs"]
    assert entries[0]["evidence_changes"]["removed_refs"] == []
    assert all(
        entry["evidence_changes"] == {"added_refs": [], "removed_refs": []} for entry in entries[1:]
    )
    with pytest.raises(DecisionContractError) as exc_info:
        append_decision_discipline(
            entries,
            decision_receipt=stack["decision"],
            decision_validation_closure=stack["decision_closure"],
            previous_decision_receipt=stack["decision"],
            previous_decision_validation_closure=stack["decision_closure"],
            stage="REVIEW",
            event_type="LESSON_LEARNED",
            status="LEARNED",
            summary="Not allowed after terminal.",
            event_at=_later(AS_OF, seconds=5),
            expected_tip=entries[-1]["semantic_sha256"],
        )
    assert exc_info.value.code == "I1_DISCIPLINE_TRANSITION_INVALID"
