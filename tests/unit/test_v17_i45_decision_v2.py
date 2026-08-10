from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any

import pytest

from quant_investor.intelligence_v2._core import (
    common_fields,
    content_ref,
    seal,
)
from quant_investor.intelligence_v2.decision_v2 import (
    DecisionV2ContractError,
    FUSION_IMPLEMENTATION_SHA256,
    build_decision_policy_v2,
    build_evidence_graph_v2,
    build_fusion_projection_v2,
    make_decision_v2,
    validate_decision_policy_v2,
    validate_decision_receipt_v2,
    validate_evidence_graph_v2,
    validate_fusion_projection_v2,
)
from quant_investor.intelligence_v2.decision_v2 import engine as engine_module
from quant_investor.intelligence_v2.decision_v2 import fusion as fusion_module
from quant_investor.intelligence_v2.decision_v2 import graph as graph_module
from quant_investor.intelligence._core import (
    content_ref as v1_content_ref,
    seal_content_addressed,
)

AS_OF = "2026-08-07T08:00:00Z"
SHA = "a" * 64


def _artifact(identity_field: str, version: str, **payload: Any) -> dict[str, Any]:
    return seal(
        {
            **common_fields(timestamp_value=AS_OF),
            **payload,
            "version": version,
        },
        identity_field=identity_field,
    )


def _policy(**overrides: Any) -> dict[str, Any]:
    values = {
        "created_at": AS_OF,
        "fusion_threshold": "0.500000000000",
        "posterior_threshold": "0.600000000000",
        "max_risk": "0.400000000000",
        "required_r22_status": "SUPPORTED",
        "allowed_fundamental_stale_sessions": 1,
        "mandatory_industry_state": "AVAILABLE",
        "mandatory_theme_states": ["AVAILABLE", "NO_MEMBERSHIP"],
        "hard_veto_codes": ["ACCOUNTING_FRAUD"],
    }
    values.update(overrides)
    return build_decision_policy_v2(**values)


def _graph(company: str = "000001.SZ", **overrides: Any) -> dict[str, Any]:
    values = {
        "bayesian_posterior": "0.600000000000",
        "blocker_codes": [],
        "company_code": company,
        "fundamental_stale_sessions": 0,
        "fusion_ready": True,
        "industry_state": "AVAILABLE",
        "overall_risk": "0.400000000000",
        "policy_independent_hard_veto_codes": [],
        "quant_pool_ref": content_ref(
            _artifact("pool_id", "myquant.test.pool.v1"),
            identity_field="pool_id",
        ),
        "quant_score": "0.700000000000",
        "r22_hypothesis_status": "SUPPORTED",
        "r22_preregistered": True,
        "run_id": "run-1",
        "theme_state": "NO_MEMBERSHIP",
        "v2_manifest_ref": content_ref(
            _artifact("manifest_id", "myquant.test.manifest.v1"),
            identity_field="manifest_id",
        ),
    }
    values.update(overrides)
    return _artifact(
        "graph_id",
        "myquant.test.evidence-graph.v1",
        **values,
    )


def _profile(company: str = "000001.SZ", **overrides: Any) -> dict[str, Any]:
    values = {
        "company_code": company,
        "coverage": "1.000000000000",
        "effective_score": "0.700000000000",
        "peer_symbols": [company],
        "policy_ref": {"policy": "same"},
        "raw_score": "0.700000000000",
        "score_present": True,
        "scorer_implementation_sha256": SHA,
        "scorer_version": "myquant.test.fundamental.v1",
        "status": "COMPLETE",
    }
    values.update(overrides)
    return _artifact(
        "profile_id",
        "myquant.test.fundamental-profile.v1",
        **values,
    )


def _projection(graph: dict[str, Any], score: str = "0.500000000000") -> dict[str, Any]:
    return _artifact(
        "projection_id",
        "myquant.test.fusion-projection.v1",
        graph_refs=[content_ref(graph, identity_field="graph_id")],
        projected_records=[
            {
                "effective_score": score,
                "rank": 1,
                "symbol": graph["company_code"],
            }
        ],
        run_id=graph["run_id"],
    )


def _v1_artifact(identity_field: str, version: str, **payload: Any) -> dict[str, Any]:
    return seal_content_addressed(
        {"timestamp": AS_OF, "version": version, **payload},
        identity_field=identity_field,
    )


def _exact_v1(document: dict[str, Any], identity_field: str, name: str) -> dict[str, Any]:
    return {
        **v1_content_ref(document, identity_field=identity_field),
        "available_at": AS_OF,
        "cutoff": AS_OF,
        "relative_path": f"fixtures/{name}.json",
    }


def _source_bound_evidence(
    source_type: str,
    document: dict[str, Any],
    identity_field: str,
) -> dict[str, Any]:
    return _v1_artifact(
        "evidence_id",
        "myquant.test.evidence.v1",
        source_ref={
            **content_ref(document, identity_field=identity_field),
            "available_at": AS_OF,
            "cutoff": AS_OF,
            "relative_path": f"fixtures/{source_type.lower()}.json",
        },
        source_type=source_type,
    )


def _decision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    graph: dict[str, Any],
    policy: dict[str, Any] | None = None,
    score: str = "0.500000000000",
) -> dict[str, Any]:
    projection = _projection(graph, score)
    monkeypatch.setattr(
        engine_module,
        "validate_evidence_graph_v2",
        lambda document, **closure: document,
    )
    monkeypatch.setattr(
        engine_module,
        "validate_fusion_projection_v2",
        lambda document, **closure: document,
    )
    return make_decision_v2(
        evidence_graph=graph,
        graph_validation_closure={},
        fusion_projection=projection,
        fusion_projection_validation_closure={},
        policy=policy or _policy(),
        as_of=AS_OF,
    )


def test_policy_is_canonical_replayed_and_rejects_binary_float() -> None:
    policy = _policy(hard_veto_codes=["ZETA", "ALPHA"])
    assert policy["hard_veto_codes"] == ["ALPHA", "ZETA"]
    assert validate_decision_policy_v2(policy) == policy
    forged = deepcopy(policy)
    forged["fusion_threshold"] = "0.700000000000"
    with pytest.raises(DecisionV2ContractError):
        validate_decision_policy_v2(forged)
    with pytest.raises(DecisionV2ContractError, match="binary float"):
        _policy(fusion_threshold=0.5)


def test_evidence_graph_replays_all_bound_layers_and_rejects_resealed_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    company = "000001.SZ"
    quant = _artifact(
        "quant_branch_id",
        "myquant.test.quant.v5",
        company_code=company,
        percentile="0.700000000000",
        pool_ref=content_ref(
            _artifact("pool_id", "myquant.test.pool.v1"),
            identity_field="pool_id",
        ),
        score="7.000000000000",
    )
    industry = _artifact(
        "evaluation_id",
        "myquant.test.industry-evaluation.v1",
        state="AVAILABLE",
        subject_id=company,
    )
    industry_component = _artifact(
        "component_receipt_id",
        "myquant.test.industry-component.v1",
        component_score="0.700000000000",
        evaluation_ref=content_ref(industry, identity_field="evaluation_id"),
        status="AVAILABLE",
    )
    theme = _artifact(
        "exposure_receipt_id",
        "myquant.test.theme-exposure.v1",
        company_code=company,
        status="NO_MEMBERSHIP",
    )
    profile = _profile(company)
    evidence = [
        _source_bound_evidence("QUANT", quant, "quant_branch_id"),
        _source_bound_evidence("INDUSTRY", industry_component, "component_receipt_id"),
        _source_bound_evidence("THEME", theme, "exposure_receipt_id"),
        _source_bound_evidence("FUNDAMENTAL", profile, "profile_id"),
    ]
    i0_quant = _v1_artifact(
        "branch_id",
        "myquant.test.i0-branch.v1",
        availability="1.000000000000",
        score=quant["percentile"],
    )
    i0_fundamental = _v1_artifact(
        "branch_id",
        "myquant.test.i0-branch.v1",
        availability=profile["coverage"],
        score=profile["effective_score"],
    )
    hypothesis = _v1_artifact(
        "hypothesis_id",
        "myquant.test.hypothesis.v1",
        related_companies=[company],
    )
    i0 = {
        "bayesian": _v1_artifact(
            "receipt_id",
            "myquant.test.bayesian.v1",
            posterior="0.700000000000",
        ),
        "evidence": evidence,
        "fundamental_branch": i0_fundamental,
        "fusion": _v1_artifact("receipt_id", "myquant.test.i0-fusion.v1"),
        "hypothesis": hypothesis,
        "quant_branch": i0_quant,
        "runtime": _v1_artifact("runtime_receipt_id", "myquant.test.runtime.v1"),
    }
    binding = _artifact(
        "binding_id",
        "myquant.test.subject-binding.v1",
        company_code=company,
        frozen_v1_branch_ref=_exact_v1(i0_quant, "branch_id", "i0-quant"),
        quant_branch_ref=content_ref(quant, identity_field="quant_branch_id"),
        v2_manifest_ref=content_ref(
            _artifact("manifest_id", "myquant.test.manifest.v1"),
            identity_field="manifest_id",
        ),
    )
    readiness = _artifact(
        "readiness_id",
        "myquant.test.readiness.v1",
        rows=[{"name": "FUNDAMENTAL", "status": "AVAILABLE"}],
    )
    r22 = {
        "envelope": _v1_artifact("envelope_id", "myquant.test.r22-envelope.v1"),
        "evaluation": _v1_artifact("receipt_id", "myquant.test.r22-hypothesis.v1"),
        "main": _v1_artifact("evaluation_id", "myquant.test.r22-main.v1"),
        "preregistered": True,
        "status": "SUPPORTED",
    }
    monkeypatch.setattr(graph_module, "validate_quant_branch_v5", lambda value, **kw: value)
    monkeypatch.setattr(
        graph_module,
        "validate_subject_branch_binding",
        lambda value, **kw: value,
    )
    monkeypatch.setattr(
        graph_module,
        "validate_investment_data_readiness",
        lambda value, **kw: value,
    )
    monkeypatch.setattr(
        graph_module,
        "validate_industry_evaluation_receipt",
        lambda value, **kw: value,
    )
    monkeypatch.setattr(
        graph_module,
        "validate_industry_component_receipt",
        lambda value, **kw: value,
    )
    monkeypatch.setattr(
        graph_module,
        "validate_theme_exposure_receipt",
        lambda value, **kw: value,
    )
    monkeypatch.setattr(graph_module, "validate_fundamental_profile", lambda value, **kw: value)
    monkeypatch.setattr(graph_module, "_validate_i0", lambda *args, **kwargs: i0)
    monkeypatch.setattr(graph_module, "_validate_r22", lambda **kwargs: r22)
    closure = {
        "run_id": "run-1",
        "company_code": company,
        "selected_hypothesis_id": hypothesis["hypothesis_id"],
        "quant_branch": quant,
        "quant_branch_validation_closure": {},
        "subject_binding": binding,
        "subject_binding_validation_closure": {},
        "readiness_receipt": readiness,
        "readiness_validation_closure": {},
        "industry_identity": industry,
        "industry_identity_validation_closure": {},
        "industry_component": industry_component,
        "industry_component_validation_closure": {},
        "theme_exposure": theme,
        "theme_exposure_validation_closure": {},
        "theme_component": None,
        "theme_component_validation_closure": None,
        "fundamental_profile": profile,
        "fundamental_profile_validation_closure": {},
        "frozen_v1_fundamental_branch_ref": _exact_v1(
            i0_fundamental,
            "branch_id",
            "i0-fundamental",
        ),
        "i0_replay_inputs": {key: None for key in graph_module.I0_REPLAY_INPUT_FIELDS},
        "r22_request_path": "fixtures/request.json",
        "r22_request_sha256": SHA,
        "risk_rows": [
            {
                "dimension": dimension,
                "evidence_refs": [v1_content_ref(evidence[0], identity_field="evidence_id")],
                "hard_veto_codes": [],
                "severity": "0.100000000000",
                "status": "AVAILABLE",
            }
            for dimension in ("BUSINESS", "FINANCIAL", "MARKET", "THESIS")
        ],
        "as_of": AS_OF,
    }
    graph = build_evidence_graph_v2(**closure)
    assert graph["fusion_ready"] is True
    assert graph["hypothesis_ref"] == v1_content_ref(
        hypothesis,
        identity_field="hypothesis_id",
    )
    assert validate_evidence_graph_v2(graph, **closure) == graph
    body = {
        key: value for key, value in graph.items() if key not in {"graph_id", "semantic_sha256"}
    }
    body["quant_score"] = "9.000000000000"
    forged = seal(body, identity_field="graph_id")
    with pytest.raises(DecisionV2ContractError, match="replay mismatch"):
        validate_evidence_graph_v2(forged, **closure)


def test_fusion_calls_frozen_scorer_once_and_projects_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    profile = _profile()
    graph["fundamental_profile_ref"] = content_ref(profile, identity_field="profile_id")
    graph = seal(
        {key: value for key, value in graph.items() if key not in {"graph_id", "semantic_sha256"}},
        identity_field="graph_id",
    )
    calls = 0

    def frozen(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert kwargs["fundamental_scores"] == {"000001.SZ": "0.700000000000"}
        assert kwargs["fundamental_coverages"] == {"000001.SZ": "1.000000000000"}
        return {
            "base_weights": {"fundamental": 0.5, "quant": 0.5},
            "records": [
                {
                    "available_weight": 1.0,
                    "branch_evidence": [],
                    "confidence_penalty": 0.0,
                    "coverage": 1.0,
                    "effective_score": 0.5,
                    "rank": 1,
                    "raw_score": 0.5,
                    "status": "AVAILABLE",
                    "symbol": "000001.SZ",
                }
            ],
            "version": fusion_module.FUSION_SCORING_V3_VERSION,
        }

    monkeypatch.setattr(
        fusion_module,
        "validate_evidence_graph_v2",
        lambda document, **closure: document,
    )
    monkeypatch.setattr(
        fusion_module,
        "validate_fundamental_profile",
        lambda document, **closure: document,
    )
    monkeypatch.setattr(fusion_module, "fuse_forward_scores_v3", frozen)
    closure = {
        "evidence_graphs": [graph],
        "graph_validation_closures": [{}],
        "fundamental_profiles": [profile],
        "fundamental_profile_validation_closures": [{}],
        "fusion_implementation_sha256": FUSION_IMPLEMENTATION_SHA256,
        "run_id": "run-1",
        "as_of": AS_OF,
    }
    projection = build_fusion_projection_v2(**closure)
    assert calls == 1
    assert projection["base_weights"]["quant"] == "0.500000000000"
    assert projection["projected_records"][0]["effective_score"] == "0.500000000000"
    assert projection["raw_float_audit"]["records"][0]["effective_score"] == {
        "binary_float_repr": "0.5"
    }
    assert validate_fusion_projection_v2(projection, **closure) == projection
    assert calls == 2
    body = {
        key: value
        for key, value in projection.items()
        if key not in {"projection_id", "semantic_sha256"}
    }
    body["projected_records"][0]["effective_score"] = "0.900000000000"
    forged = seal(body, identity_field="projection_id")
    with pytest.raises(DecisionV2ContractError, match="replay mismatch"):
        validate_fusion_projection_v2(forged, **closure)
    assert calls == 3
    wrong_implementation = {**closure, "fusion_implementation_sha256": SHA}
    with pytest.raises(DecisionV2ContractError, match="implementation SHA mismatch"):
        build_fusion_projection_v2(**wrong_implementation)
    assert calls == 3


def test_fusion_rejects_nonidentical_i4_peer_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    profile = _profile(peer_symbols=["000001.SZ", "000002.SZ"])
    graph["fundamental_profile_ref"] = content_ref(profile, identity_field="profile_id")
    graph = seal(
        {key: value for key, value in graph.items() if key not in {"graph_id", "semantic_sha256"}},
        identity_field="graph_id",
    )
    monkeypatch.setattr(
        fusion_module,
        "validate_evidence_graph_v2",
        lambda document, **closure: document,
    )
    monkeypatch.setattr(
        fusion_module,
        "validate_fundamental_profile",
        lambda document, **closure: document,
    )
    with pytest.raises(DecisionV2ContractError, match="exact peer set"):
        build_fusion_projection_v2(
            evidence_graphs=[graph],
            graph_validation_closures=[{}],
            fundamental_profiles=[profile],
            fundamental_profile_validation_closures=[{}],
            fusion_implementation_sha256=FUSION_IMPLEMENTATION_SHA256,
            run_id="run-1",
            as_of=AS_OF,
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "PAPER_CANDIDATE"),
        ({"fundamental_stale_sessions": 1}, "RESEARCH_APPROVED"),
        ({"bayesian_posterior": "0.599999999999"}, "WATCHLIST"),
        ({"blocker_codes": ["RISK_MARKET_UNAVAILABLE"]}, "INSUFFICIENT_EVIDENCE"),
        (
            {
                "blocker_codes": ["RISK_MARKET_UNAVAILABLE"],
                "r22_hypothesis_status": "FAILED",
                "r22_preregistered": True,
            },
            "THESIS_INVALIDATED",
        ),
    ],
)
def test_five_state_priority(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    expected: str,
) -> None:
    receipt = _decision(monkeypatch, graph=_graph(**changes))
    assert receipt["state"] == expected
    if expected == "THESIS_INVALIDATED":
        assert "RISK_MARKET_UNAVAILABLE" in receipt["blocker_codes"]


def test_threshold_equality_and_posthoc_failure_do_not_invalidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    equality = _decision(monkeypatch, graph=_graph())
    assert equality["state"] == "PAPER_CANDIDATE"
    posthoc = _decision(
        monkeypatch,
        graph=_graph(r22_hypothesis_status="FAILED", r22_preregistered=False),
    )
    assert posthoc["state"] == "WATCHLIST"
    assert "PREREGISTERED_HYPOTHESIS_FAILED" not in posthoc["reason_codes"]


def test_veto_risk_and_missing_identity_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    veto = _decision(
        monkeypatch,
        graph=_graph(policy_independent_hard_veto_codes=["ACCOUNTING_FRAUD"]),
    )
    assert veto["state"] == "WATCHLIST"
    assert "HARD_RISK_VETO" in veto["reason_codes"]
    missing = _decision(
        monkeypatch,
        graph=_graph(
            blocker_codes=["INDUSTRY_IDENTITY_UNMAPPED"],
            industry_state="UNMAPPED",
        ),
    )
    assert missing["state"] == "INSUFFICIENT_EVIDENCE"


def test_decision_receipt_replays_and_rejects_resealed_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    policy = _policy()
    projection = _projection(graph)
    monkeypatch.setattr(
        engine_module,
        "validate_evidence_graph_v2",
        lambda document, **closure: document,
    )
    monkeypatch.setattr(
        engine_module,
        "validate_fusion_projection_v2",
        lambda document, **closure: document,
    )
    closure = {
        "evidence_graph": graph,
        "graph_validation_closure": {},
        "fusion_projection": projection,
        "fusion_projection_validation_closure": {},
        "policy": policy,
        "as_of": AS_OF,
    }
    receipt = make_decision_v2(**closure)
    assert validate_decision_receipt_v2(receipt, **closure) == receipt
    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"decision_id", "semantic_sha256"}
    }
    body["state"] = "WATCHLIST"
    forged = seal(body, identity_field="decision_id")
    with pytest.raises(DecisionV2ContractError, match="replay mismatch"):
        validate_decision_receipt_v2(forged, **closure)


def test_public_builders_exclude_macro_ai_and_operational_inputs() -> None:
    forbidden = {
        "ai",
        "broker",
        "execution",
        "llm",
        "macro",
        "order",
        "portfolio",
        "provider",
        "trade",
    }
    for function in (
        build_evidence_graph_v2,
        build_fusion_projection_v2,
        make_decision_v2,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
