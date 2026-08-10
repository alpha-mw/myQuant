"""Offline contract and deterministic replay tests for I5."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import inspect
from typing import Any

import pytest

from quant_investor.intelligence_v2._core import common_fields, content_ref, seal
from quant_investor.intelligence_v2.decision_v2 import (
    build_decision_policy_v2,
    make_decision_v2,
)
from quant_investor.intelligence_v2.decision_v2 import engine as decision_engine
from quant_investor.intelligence_v2.llm_research import (
    I5ContractError,
    OpenAIResponsesPrivateAdapter,
    OpenAIResponsesPublicAdapter,
    PRIVATE_OUTPUT_SCHEMA_SHA256,
    PRIVATE_REQUEST_SCHEMA_SHA256,
    PUBLIC_OUTPUT_SCHEMA_SHA256,
    PUBLIC_REQUEST_SCHEMA_SHA256,
    PUBLIC_SEARCH_PROMPT_SHA256,
    ROLE,
    ROUND_ORDER,
    SYSTEM_PROMPT_SHA256,
    build_advisory_rank,
    build_advisory_unavailable,
    build_capture_policy,
    build_capture_receipt,
    build_citation_lead,
    build_committee_request,
    build_committee_response,
    build_decision_evidence_projection,
    build_declassified_public_packet,
    build_historical_replay_receipt,
    build_private_capability,
    build_public_search_capability,
    build_search_request,
    build_search_response,
    build_search_run,
    build_search_run_status,
    build_search_source,
    build_validated_fact,
    validate_advisory_rank,
    validate_advisory_unavailable,
    validate_capability_isolation,
    validate_capture_policy,
    validate_capture_receipt,
    validate_committee_request,
    validate_committee_response,
    validate_decision_evidence_projection,
    validate_declassified_public_packet,
    validate_historical_replay_receipt,
    validate_private_capability,
    validate_public_search_capability,
    validate_search_run,
    validate_search_run_status,
    validate_validated_fact,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
T0 = "2025-01-02T00:00:00Z"
T1 = "2025-01-02T01:00:00Z"
T2 = "2025-01-02T02:00:00Z"
T3 = "2025-01-02T03:00:00Z"
T4 = "2025-01-02T04:00:00Z"
T5 = "2025-01-02T05:00:00Z"
T6 = "2025-01-02T06:00:00Z"
T7 = "2025-01-02T07:00:00Z"
T8 = "2025-01-02T08:00:00Z"
T9 = "2025-01-02T09:00:00Z"


def _exact_ref(name: str, *, sha: str = SHA_A, at: str = T0) -> dict[str, str]:
    return {
        "artifact_id": f"{name}-artifact",
        "artifact_version": f"{name}.v1",
        "available_at": at,
        "byte_sha256": sha,
        "cutoff": at,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": sha,
    }


def _packet() -> dict[str, Any]:
    return build_declassified_public_packet(
        company_code="600000.SH",
        display_name="Public Company",
        public_industry_ids=["BANKS"],
        public_theme_ids=["DIVIDEND"],
        thesis="Public thesis requiring skeptical verification.",
        search_questions=["What evidence falsifies the thesis?"],
        market_data_cutoff=T0,
        target_knowledge_not_before=T1,
        target_knowledge_not_after=T8,
        created_at=T0,
        declassification_evidence_ref=_exact_ref("declassification-evidence"),
    )


@pytest.fixture(autouse=True)
def _isolate_decision_v2_fixture_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        decision_engine,
        "validate_evidence_graph_v2",
        lambda document, **_closure: document,
    )
    monkeypatch.setattr(
        decision_engine,
        "validate_fusion_projection_v2",
        lambda document, **_closure: document,
    )


def _decision_v2() -> tuple[dict[str, Any], dict[str, Any]]:
    industry = seal(
        {
            **common_fields(timestamp_value=T0),
            "component_score": "0.700000000000",
            "status": "AVAILABLE",
            "version": "myquant.test.i5-industry-component.v1",
        },
        identity_field="component_receipt_id",
    )
    theme = seal(
        {
            **common_fields(timestamp_value=T0),
            "status": "NO_MEMBERSHIP",
            "version": "myquant.test.i5-theme-exposure.v1",
        },
        identity_field="exposure_receipt_id",
    )
    profile = seal(
        {
            **common_fields(timestamp_value=T0),
            "coverage": "1.000000000000",
            "effective_score": "0.700000000000",
            "score_present": True,
            "status": "COMPLETE",
            "version": "myquant.test.i5-fundamental-profile.v1",
        },
        identity_field="profile_id",
    )
    graph = seal(
        {
            **common_fields(timestamp_value=T0),
            "bayesian_posterior": "0.700000000000",
            "blocker_codes": [],
            "company_code": "600000.SH",
            "fundamental_stale_sessions": 0,
            "fusion_ready": True,
            "industry_state": "AVAILABLE",
            "industry_component_ref": content_ref(industry, identity_field="component_receipt_id"),
            "industry_identity_ref": None,
            "overall_risk": "0.100000000000",
            "policy_independent_hard_veto_codes": [],
            "risk_rows": [
                {
                    "dimension": dimension,
                    "evidence_refs": [],
                    "hard_veto_codes": [],
                    "severity": "0.100000000000",
                    "status": "AVAILABLE",
                }
                for dimension in ("BUSINESS", "FINANCIAL", "MARKET", "THESIS")
            ],
            "r22_hypothesis_status": "SUPPORTED",
            "r22_preregistered": True,
            "run_id": "I5_DECISION_RUN",
            "theme_state": "NO_MEMBERSHIP",
            "theme_component_ref": None,
            "theme_exposure_ref": content_ref(theme, identity_field="exposure_receipt_id"),
            "fundamental_profile_ref": content_ref(profile, identity_field="profile_id"),
            "version": "myquant.test.i5-evidence-graph.v1",
        },
        identity_field="graph_id",
    )
    graph_ref = content_ref(graph, identity_field="graph_id")
    projection = seal(
        {
            **common_fields(timestamp_value=T0),
            "graph_refs": [graph_ref],
            "projected_records": [
                {"effective_score": "0.900000000000", "rank": 1, "symbol": "000001.SZ"},
                {"effective_score": "0.800000000000", "rank": 2, "symbol": "000002.SZ"},
                {"effective_score": "0.700000000000", "rank": 3, "symbol": "600000.SH"},
                {"effective_score": "0.600000000000", "rank": 4, "symbol": "000004.SZ"},
                {"effective_score": "0.500000000000", "rank": 5, "symbol": "000005.SZ"},
            ],
            "run_id": "I5_DECISION_RUN",
            "version": "myquant.test.i5-fusion-projection.v1",
        },
        identity_field="projection_id",
    )
    policy = build_decision_policy_v2(
        created_at=T0,
        fusion_threshold="0.500000000000",
        posterior_threshold="0.600000000000",
        max_risk="0.400000000000",
        required_r22_status="SUPPORTED",
        allowed_fundamental_stale_sessions=1,
        mandatory_industry_state="AVAILABLE",
        mandatory_theme_states=["AVAILABLE", "NO_MEMBERSHIP"],
        hard_veto_codes=["ACCOUNTING_FRAUD"],
    )
    closure = {
        "evidence_graph": graph,
        "graph_validation_closure": {
            "industry_component": industry,
            "industry_identity": None,
            "theme_component": None,
            "theme_exposure": theme,
            "fundamental_profile": profile,
        },
        "fusion_projection": projection,
        "fusion_projection_validation_closure": {},
        "policy": policy,
        "as_of": T0,
    }
    return make_decision_v2(**closure), closure


def _public_capability() -> dict[str, Any]:
    return build_public_search_capability(
        organization="ORG_PUBLIC",
        project="PROJECT_PUBLIC",
        credential_ref="CREDENTIAL_PUBLIC",
        client_namespace="CLIENT_PUBLIC",
        history_namespace="HISTORY_PUBLIC",
        cache_namespace="CACHE_PUBLIC",
        receipt_store_namespace="RECEIPTS_PUBLIC",
        endpoint="https://api.openai.com/v1/responses",
        exact_model="gpt-5.4-public-search",
        control_evidence_ref=_exact_ref("declassification-evidence"),
        not_before=T1,
        expires_at=T8,
        created_at=T0,
    )


def _round_bundle(packet: dict[str, Any], capability: dict[str, Any], index: int) -> dict[str, Any]:
    requested = (T2, T3, T4)[index - 1]
    responded = (T3, T4, T5)[index - 1]
    request = build_search_request(
        packet=packet,
        capability=capability,
        round_name=ROUND_ORDER[index - 1],
        round_index=index,
        query=f"Round {index} skeptical public query",
        requested_at=requested,
    )
    response_id = f"RESPONSE_{index}"
    response = build_search_response(
        request=request,
        provider_response_id=response_id,
        response_text=f"Frozen public response {index}.",
        responded_at=responded,
    )
    source = build_search_source(
        request=request,
        provider_response_id=response_id,
        url=f"https://source{index}.example/report",
        title=f"Source {index}",
        publisher=f"Publisher {index}",
        publication_hint="2025-01-01",
        media_kind="HTML",
        discovered_at=responded,
    )
    lead = build_citation_lead(
        provider_response_id=response_id,
        source=source,
        citation_text=f"Citation lead {index}",
        citation_url=source["url"],
        cited_at=responded,
    )
    return {
        "request": request,
        "response": response,
        "sources": [source],
        "citation_leads": [lead],
    }


def _observation(url: str) -> dict[str, Any]:
    return {
        "redirect_chain": [
            {
                "url": url,
                "resolved_addresses": ["93.184.216.34"],
                "peer_ip": "93.184.216.34",
                "status_code": 200,
                "location": None,
            }
        ],
        "selected_headers": {
            "content-type": "text/html; charset=UTF-8",
            "content-encoding": "identity",
        },
        "content_encoding": "identity",
        "mime_type": "text/html",
        "charset": "UTF-8",
        "html_node_count": 5,
        "html_max_depth": 3,
        "header_bytes_total": 128,
        "connect_timeout_seconds": 5,
        "read_timeout_seconds": 15,
        "total_timeout_seconds": 30,
        "transfer_decoding_complete": True,
        "proxy_environment_used": False,
        "cookies_sent": False,
        "authorization_sent": False,
        "library_reresolution": False,
        "hostname_sni_preserved": True,
    }


def _private_capability() -> dict[str, Any]:
    return build_private_capability(
        organization="ORG_PRIVATE",
        project="PROJECT_PRIVATE",
        credential_ref="CREDENTIAL_PRIVATE",
        client_namespace="CLIENT_PRIVATE",
        history_namespace="HISTORY_PRIVATE",
        cache_namespace="CACHE_PRIVATE",
        receipt_store_namespace="RECEIPTS_PRIVATE",
        endpoint="https://api.openai.com/v1/responses",
        exact_model="gpt-5.4-private-committee",
        zdr_evidence_ref=_exact_ref("zdr-control", sha=SHA_B),
        not_before=T6,
        expires_at=T8,
        created_at=T0,
    )


def _stack(*, conflict_status: str = "NONE") -> dict[str, Any]:
    packet = _packet()
    public_capability = _public_capability()
    rounds = [_round_bundle(packet, public_capability, index) for index in range(1, 4)]
    search_run = build_search_run(
        packet=packet,
        capability=public_capability,
        round_bundles=rounds,
        evidence_collection_started_at=T1,
        search_completed_at=T5,
    )
    policy = build_capture_policy(
        parser_identity="STRICT_HTML_PARSER",
        parser_version="1.0.0",
        parser_options_sha256=SHA_A,
        decoder_manifest_sha256=SHA_B,
        transport_policy_ref=_exact_ref("transport-policy", sha=SHA_C),
        created_at=T0,
    )
    source = rounds[0]["sources"][0]
    request = rounds[0]["request"]
    provider_response_id = rounds[0]["response"]["provider_response_id"]
    body = b"<html><body>Revenue declined.</body></html>"
    observation = _observation(source["url"])
    capture = build_capture_receipt(
        source=source,
        source_request=request,
        provider_response_id=provider_response_id,
        policy=policy,
        transport_observation=observation,
        compressed_entity=body,
        decoded_entity=body,
        parser_input=body,
        publication_evidence_ref=_exact_ref("publication", sha=SHA_B),
        transport_attestation_ref=_exact_ref("transport-attestation", sha=SHA_C),
        captured_at=T5,
    )
    capture_closure = {
        "source": source,
        "source_request": request,
        "provider_response_id": provider_response_id,
        "policy": policy,
        "transport_observation": observation,
        "compressed_entity": body,
        "decoded_entity": body,
    }
    claim = "Revenue declined."
    start = body.index(claim.encode("utf-8"))
    fact = build_validated_fact(
        capture_receipt=capture,
        capture_closure=capture_closure,
        parser_input=body,
        subject_id="600000.SH",
        claim=claim,
        byte_start=start,
        byte_end=start + len(claim.encode("utf-8")),
        source_class="FIRST_PARTY",
        canonical_source_id="SOURCE_CANONICAL_1",
        original_source_id="ORIGINAL_1",
        syndication_group_id=None,
        conflict_status=conflict_status,
        prompt_injection_detected=True,
        validated_at=T6,
    )
    fact_bundle = {
        "fact": fact,
        "capture_receipt": capture,
        "capture_closure": capture_closure,
        "parser_input": body,
    }
    decision_receipt, decision_validation_closure = _decision_v2()
    projection = build_decision_evidence_projection(
        packet=packet,
        public_capability=public_capability,
        search_run=search_run,
        round_bundles=rounds,
        decision_receipt=decision_receipt,
        decision_validation_closure=decision_validation_closure,
        fact_bundles=[fact_bundle],
        knowledge_cutoff=T6,
    )
    private_capability = _private_capability()
    request_receipt = build_committee_request(
        capability=private_capability, projection=projection, requested_at=T6
    )
    response_receipt = build_committee_response(
        capability=private_capability,
        request=request_receipt,
        projection=projection,
        provider_response_id="PRIVATE_RESPONSE_1",
        relative_score="0.200000000000",
        findings=[
            {
                "classification": "FACT",
                "text": "Revenue declined according to the validated source.",
                "fact_refs": [fact["fact_id"]],
            },
            {
                "classification": "INFERENCE",
                "text": "The thesis may be weaker than expected.",
                "fact_refs": [fact["fact_id"]],
            },
        ],
        thesis_gaps=["Durability remains unproven."],
        source_conflicts=[],
        time_conflicts=[],
        unverifiable_claims=["Long-term recovery timing is unverifiable."],
        responded_at=T7,
    )
    advisory = build_advisory_rank(
        projection=projection,
        response=response_receipt,
        response_validation_closure={
            "capability": private_capability,
            "request": request_receipt,
        },
        fact_bundles=[fact_bundle],
        advisory_weight="0.100000000000",
        issued_at=T7,
    )
    replay = build_historical_replay_receipt(
        packet=packet,
        public_capability=public_capability,
        search_run=search_run,
        round_bundles=rounds,
        fact_bundles=[fact_bundle],
        projection=projection,
        decision_receipt=decision_receipt,
        decision_validation_closure=decision_validation_closure,
        private_capability=private_capability,
        committee_request=request_receipt,
        committee_response=response_receipt,
        advisory_rank=advisory,
        target_trade_execution_boundary=T8,
        replayed_at=T9,
    )
    return locals()


def _reseal(document: dict[str, Any], *, identity_field: str) -> dict[str, Any]:
    body = deepcopy(document)
    body.pop(identity_field)
    body.pop("semantic_sha256")
    return seal(body, identity_field=identity_field)


def test_i5_full_offline_replay_is_closed_and_deterministic() -> None:
    stack = _stack()
    assert ROLE == "怀疑型 AI 投委会"
    assert (
        validate_declassified_public_packet(
            stack["packet"],
            declassification_evidence_ref=stack["public_capability"]["control_evidence_ref"],
        )
        == stack["packet"]
    )
    assert (
        validate_public_search_capability(stack["public_capability"]) == stack["public_capability"]
    )
    assert validate_capture_policy(stack["policy"]) == stack["policy"]
    assert (
        validate_search_run(
            stack["search_run"],
            packet=stack["packet"],
            capability=stack["public_capability"],
            round_bundles=stack["rounds"],
        )
        == stack["search_run"]
    )
    assert (
        validate_capture_receipt(
            stack["capture"], parser_input=stack["body"], **stack["capture_closure"]
        )
        == stack["capture"]
    )
    assert (
        validate_validated_fact(
            stack["fact"],
            capture_receipt=stack["capture"],
            capture_closure=stack["capture_closure"],
            parser_input=stack["body"],
        )
        == stack["fact"]
    )
    assert validate_private_capability(stack["private_capability"]) == stack["private_capability"]
    validate_capability_isolation(
        public_capability=stack["public_capability"],
        private_capability=stack["private_capability"],
    )
    assert (
        validate_decision_evidence_projection(
            stack["projection"],
            packet=stack["packet"],
            public_capability=stack["public_capability"],
            search_run=stack["search_run"],
            round_bundles=stack["rounds"],
            fact_bundles=[stack["fact_bundle"]],
            decision_receipt=stack["decision_receipt"],
            decision_validation_closure=stack["decision_validation_closure"],
        )
        == stack["projection"]
    )
    assert (
        validate_committee_request(
            stack["request_receipt"],
            capability=stack["private_capability"],
            projection=stack["projection"],
        )
        == stack["request_receipt"]
    )
    assert (
        validate_committee_response(
            stack["response_receipt"],
            capability=stack["private_capability"],
            request=stack["request_receipt"],
            projection=stack["projection"],
        )
        == stack["response_receipt"]
    )
    assert (
        validate_advisory_rank(
            stack["advisory"],
            projection=stack["projection"],
            response=stack["response_receipt"],
            response_validation_closure={
                "capability": stack["private_capability"],
                "request": stack["request_receipt"],
            },
            fact_bundles=[stack["fact_bundle"]],
        )
        == stack["advisory"]
    )
    assert (
        validate_historical_replay_receipt(
            stack["replay"],
            packet=stack["packet"],
            public_capability=stack["public_capability"],
            search_run=stack["search_run"],
            round_bundles=stack["rounds"],
            fact_bundles=[stack["fact_bundle"]],
            projection=stack["projection"],
            decision_receipt=stack["decision_receipt"],
            decision_validation_closure=stack["decision_validation_closure"],
            private_capability=stack["private_capability"],
            committee_request=stack["request_receipt"],
            committee_response=stack["response_receipt"],
            advisory_rank=stack["advisory"],
        )
        == stack["replay"]
    )
    assert [row["round"] for row in stack["search_run"]["round_rows"]] == list(ROUND_ORDER)
    assert stack["request_receipt"]["system_prompt_sha256"] == SYSTEM_PROMPT_SHA256
    assert stack["request_receipt"]["request_configuration"]["tools"] == []
    assert stack["request_receipt"]["request_configuration"]["store"] is False
    assert stack["replay"]["external_call_counts"] == {
        "credential_reads": 0,
        "filesystem_discovery": 0,
        "model": 0,
        "network": 0,
    }


def test_public_search_is_exactly_three_rounds_without_domain_filter_or_fallback() -> None:
    stack = _stack()
    assert len(stack["rounds"]) == 3
    for bundle in stack["rounds"]:
        configuration = bundle["request"]["request_configuration"]
        assert configuration["tools"] == [{"type": "web_search"}]
        assert "allowed_domains" not in repr(configuration)
        assert bundle["request"]["capability_ref"] == stack["search_run"]["capability_ref"]
    with pytest.raises(I5ContractError, match="exactly three rounds"):
        build_search_run(
            packet=stack["packet"],
            capability=stack["public_capability"],
            round_bundles=stack["rounds"][:2],
            evidence_collection_started_at=T1,
            search_completed_at=T5,
        )


def test_incomplete_search_status_records_every_round_without_fallback() -> None:
    stack = _stack()
    statuses = {
        "DISCOVERY": {"status": "COMPLETED", "failure_code": None},
        "CONTRARY_GAPS": {"status": "FAILED", "failure_code": "MODEL_REQUEST_FAILED"},
        "VERIFICATION_CLOSURE": {"status": "MISSING", "failure_code": "ROUND_MISSING"},
    }
    completed = {"DISCOVERY": stack["rounds"][0]}
    status = build_search_run_status(
        packet=stack["packet"],
        capability=stack["public_capability"],
        round_statuses=statuses,
        completed_round_bundles=completed,
        evidence_collection_started_at=T1,
        status_recorded_at=T5,
    )
    assert status["status"] == "INCOMPLETE"
    assert [row["status"] for row in status["round_rows"]] == [
        "COMPLETED",
        "FAILED",
        "MISSING",
    ]
    assert (
        validate_search_run_status(
            status,
            packet=stack["packet"],
            capability=stack["public_capability"],
            round_statuses=statuses,
            completed_round_bundles=completed,
        )
        == status
    )
    all_complete = {name: {"status": "COMPLETED", "failure_code": None} for name in ROUND_ORDER}
    with pytest.raises(I5ContractError, match="must use build_search_run"):
        build_search_run_status(
            packet=stack["packet"],
            capability=stack["public_capability"],
            round_statuses=all_complete,
            completed_round_bundles={
                name: stack["rounds"][index] for index, name in enumerate(ROUND_ORDER)
            },
            evidence_collection_started_at=T1,
            status_recorded_at=T5,
        )


@pytest.mark.parametrize("availability_status", ["MISSING", "EXPIRED", "UNVERIFIABLE"])
def test_advisory_unavailable_preserves_d_and_blocks_readiness(
    availability_status: str,
) -> None:
    stack = _stack()
    closure = {
        "packet": stack["packet"],
        "public_capability": stack["public_capability"],
        "search_run": stack["search_run"],
        "round_bundles": stack["rounds"],
        "fact_bundles": [stack["fact_bundle"]],
        "decision_receipt": stack["decision_receipt"],
        "decision_validation_closure": stack["decision_validation_closure"],
    }
    capability = None if availability_status == "MISSING" else stack["private_capability"]
    evidence = (
        _exact_ref("private-capability-verification", at=T7)
        if availability_status == "UNVERIFIABLE"
        else None
    )
    issued_at = T9 if availability_status == "EXPIRED" else T7
    receipt = build_advisory_unavailable(
        projection=stack["projection"],
        projection_validation_closure=closure,
        private_capability=capability,
        availability_status=availability_status,
        verification_evidence_ref=evidence,
        issued_at=issued_at,
    )
    deterministic = stack["projection"]["deterministic_percentile"]
    assert receipt["status"] == "ADVISORY_UNAVAILABLE"
    assert receipt["deterministic_rank"] == deterministic
    assert receipt["advisory_rank"] == deterministic
    assert receipt["absolute_delta"] == "0.000000000000"
    assert receipt["research_mainline_ready"] is False
    assert (
        validate_advisory_unavailable(
            receipt,
            projection=stack["projection"],
            projection_validation_closure=closure,
            private_capability=capability,
            availability_status=availability_status,
            verification_evidence_ref=evidence,
        )
        == receipt
    )


def test_malformed_private_capability_is_not_modeled_as_ordinary_unavailable() -> None:
    stack = _stack()
    forged = deepcopy(stack["private_capability"])
    forged["tools"] = [{"type": "web_search"}]
    closure = {
        "packet": stack["packet"],
        "public_capability": stack["public_capability"],
        "search_run": stack["search_run"],
        "round_bundles": stack["rounds"],
        "fact_bundles": [stack["fact_bundle"]],
        "decision_receipt": stack["decision_receipt"],
        "decision_validation_closure": stack["decision_validation_closure"],
    }
    with pytest.raises(I5ContractError):
        build_advisory_unavailable(
            projection=stack["projection"],
            projection_validation_closure=closure,
            private_capability=forged,
            availability_status="EXPIRED",
            verification_evidence_ref=None,
            issued_at=T9,
        )


def test_resealed_forgery_is_rejected_by_full_replay() -> None:
    stack = _stack()
    forged = deepcopy(stack["advisory"])
    forged["advisory_rank"] = "0.900000000000"
    forged = _reseal(forged, identity_field="advisory_rank_id")
    with pytest.raises(I5ContractError, match="deterministic replay"):
        validate_advisory_rank(
            forged,
            projection=stack["projection"],
            response=stack["response_receipt"],
            response_validation_closure={
                "capability": stack["private_capability"],
                "request": stack["request_receipt"],
            },
            fact_bundles=[stack["fact_bundle"]],
        )


def test_advisory_rank_is_bounded_and_unresolved_conflict_forces_zero_delta() -> None:
    stack = _stack()
    assert Decimal(stack["advisory"]["absolute_delta"]) == Decimal("0.040000000000")
    assert Decimal(stack["advisory"]["absolute_delta"]) <= Decimal("0.10")
    with pytest.raises(I5ContractError, match="exceeds 10 percent"):
        build_advisory_rank(
            projection=stack["projection"],
            response=stack["response_receipt"],
            response_validation_closure={
                "capability": stack["private_capability"],
                "request": stack["request_receipt"],
            },
            fact_bundles=[stack["fact_bundle"]],
            advisory_weight="0.100000000001",
            issued_at=T7,
        )

    conflicted = _stack(conflict_status="UNRESOLVED")
    assert conflicted["advisory"]["advisory_rank"] == conflicted["advisory"]["deterministic_rank"]
    assert conflicted["advisory"]["absolute_delta"] == "0.000000000000"
    assert conflicted["advisory"]["reason_codes"] == ["UNRESOLVED_SOURCE_CONFLICT"]


def test_ai_artifacts_do_not_change_deterministic_authority() -> None:
    stack = _stack()
    assert stack["projection"]["deterministic_decision_state"] == "PAPER_CANDIDATE"
    assert "deterministic_decision_state" not in stack["advisory"]
    forbidden = {
        "action",
        "admission",
        "cash",
        "holding",
        "order",
        "quantity",
        "risk_threshold",
        "side",
        "target_price",
        "veto",
        "weight",
    }

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert forbidden.isdisjoint(
        {key.casefold() for key in keys(stack["response_receipt"]["subject_row"])}
    )
    authority = stack["advisory"]["authority"]
    assert authority["research_only"] is True
    assert all(value is False for key, value in authority.items() if key != "research_only")


def test_projection_replays_exact_decision_without_caller_authority() -> None:
    stack = _stack()
    projection = stack["projection"]
    decision = stack["decision_receipt"]
    assert projection["decision_ref"]["artifact_id"] == decision["decision_id"]
    assert projection["subject_identity"]["company_code"] == decision["company_code"]
    assert projection["deterministic_decision_state"] == decision["state"]
    assert projection["deterministic_percentile"] == decision["deterministic_percentile"]
    forbidden = {
        "subject_id",
        "deterministic_decision_state",
        "deterministic_percentile",
        "risk_codes",
        "blocker_codes",
        "validated_summaries",
    }
    assert forbidden.isdisjoint(inspect.signature(build_decision_evidence_projection).parameters)
    forged = deepcopy(decision)
    forged["state"] = "WATCHLIST"
    forged = _reseal(forged, identity_field="decision_id")
    with pytest.raises(I5ContractError, match="Decision v2 replay failed"):
        build_decision_evidence_projection(
            packet=stack["packet"],
            public_capability=stack["public_capability"],
            search_run=stack["search_run"],
            round_bundles=stack["rounds"],
            decision_receipt=forged,
            decision_validation_closure=stack["decision_validation_closure"],
            fact_bundles=[stack["fact_bundle"]],
            knowledge_cutoff=T6,
        )


def test_projection_summaries_are_exactly_graph_bound() -> None:
    stack = _stack()
    rows = {row["kind"]: row for row in stack["projection"]["validated_summaries"]}
    graph = stack["decision_validation_closure"]["evidence_graph"]
    assert rows["INDUSTRY"]["source_ref"] == graph["industry_component_ref"]
    assert rows["THEME"]["source_ref"] == graph["theme_exposure_ref"]
    assert rows["FUNDAMENTAL"]["source_ref"] == graph["fundamental_profile_ref"]
    assert {row["projection_status"] for row in rows.values()} == {"LOCAL_VALIDATED_PROJECTION"}
    closure = deepcopy(stack["decision_validation_closure"])
    closure["graph_validation_closure"]["industry_component"] = seal(
        {
            **common_fields(timestamp_value=T0),
            "component_score": "0.100000000000",
            "status": "AVAILABLE",
            "version": "myquant.test.i5-industry-component.v1",
        },
        identity_field="component_receipt_id",
    )
    with pytest.raises(I5ContractError, match="summary source differs"):
        build_decision_evidence_projection(
            packet=stack["packet"],
            public_capability=stack["public_capability"],
            search_run=stack["search_run"],
            round_bundles=stack["rounds"],
            decision_receipt=stack["decision_receipt"],
            decision_validation_closure=closure,
            fact_bundles=[stack["fact_bundle"]],
            knowledge_cutoff=T6,
        )


def test_capabilities_bind_versioned_prompt_and_schema_hashes() -> None:
    stack = _stack()
    public = stack["public_capability"]
    private = stack["private_capability"]
    assert public["system_prompt_sha256"] == PUBLIC_SEARCH_PROMPT_SHA256
    assert public["request_schema_sha256"] == PUBLIC_REQUEST_SCHEMA_SHA256
    assert public["output_schema_sha256"] == PUBLIC_OUTPUT_SCHEMA_SHA256
    assert private["system_prompt_sha256"] == SYSTEM_PROMPT_SHA256
    assert private["request_schema_sha256"] == PRIVATE_REQUEST_SCHEMA_SHA256
    assert private["output_schema_sha256"] == PRIVATE_OUTPUT_SCHEMA_SHA256
    forged = deepcopy(public)
    forged["request_schema_sha256"] = SHA_C
    forged = _reseal(forged, identity_field="capability_id")
    with pytest.raises(I5ContractError, match="deterministic replay"):
        validate_public_search_capability(forged)


def test_injected_responses_adapters_call_public_three_private_one_and_never_fallback() -> None:
    stack = _stack()
    public_calls: list[dict[str, Any]] = []

    def public_create(**kwargs: Any) -> dict[str, Any]:
        public_calls.append(kwargs)
        return {
            "provider_response_id": f"LIVE_{len(public_calls)}",
            "response_text": "Frozen fake response",
            "sources": [],
            "citation_leads": [],
        }

    public_adapter = OpenAIResponsesPublicAdapter(
        public_create,
        project=stack["public_capability"]["project"],
        credential_ref=stack["public_capability"]["credential_ref"],
        client_namespace=stack["public_capability"]["client_namespace"],
    )
    results = public_adapter.execute_three_rounds(
        requests=[row["request"] for row in stack["rounds"]],
        packet=stack["packet"],
        capability=stack["public_capability"],
    )
    assert len(results) == len(public_calls) == 3
    assert all(call["tools"] == [{"type": "web_search"}] for call in public_calls)
    assert all("allowed_domains" not in repr(call) for call in public_calls)

    private_calls: list[dict[str, Any]] = []

    def private_create(**kwargs: Any) -> dict[str, Any]:
        private_calls.append(kwargs)
        return {
            "provider_response_id": "PRIVATE_LIVE_1",
            "relative_score": "0.500000000000",
            "findings": [],
            "thesis_gaps": [],
            "source_conflicts": [],
            "time_conflicts": [],
            "unverifiable_claims": [],
            "tool_calls": [],
        }

    private_adapter = OpenAIResponsesPrivateAdapter(
        private_create,
        project=stack["private_capability"]["project"],
        credential_ref=stack["private_capability"]["credential_ref"],
        client_namespace=stack["private_capability"]["client_namespace"],
    )
    private_adapter.execute_once(
        request=stack["request_receipt"],
        capability=stack["private_capability"],
        projection=stack["projection"],
    )
    assert len(private_calls) == 1
    assert private_calls[0]["tools"] == []
    assert private_calls[0]["store"] is False

    failures = 0

    def failing_create(**_kwargs: Any) -> dict[str, Any]:
        nonlocal failures
        failures += 1
        raise RuntimeError("no fallback")

    failing = OpenAIResponsesPublicAdapter(
        failing_create,
        project=stack["public_capability"]["project"],
        credential_ref=stack["public_capability"]["credential_ref"],
        client_namespace=stack["public_capability"]["client_namespace"],
    )
    with pytest.raises(RuntimeError, match="no fallback"):
        failing.execute_three_rounds(
            requests=[row["request"] for row in stack["rounds"]],
            packet=stack["packet"],
            capability=stack["public_capability"],
        )
    assert failures == 1


def test_two_urls_with_same_original_source_do_not_authorize_nonzero_delta() -> None:
    stack = _stack()
    bundles = []
    for index in (0, 1):
        round_bundle = stack["rounds"][index]
        source = round_bundle["sources"][0]
        request = round_bundle["request"]
        response_id = round_bundle["response"]["provider_response_id"]
        observation = _observation(source["url"])
        capture = build_capture_receipt(
            source=source,
            source_request=request,
            provider_response_id=response_id,
            policy=stack["policy"],
            transport_observation=observation,
            compressed_entity=stack["body"],
            decoded_entity=stack["body"],
            parser_input=stack["body"],
            publication_evidence_ref=_exact_ref(f"publication-{index}", sha=SHA_B),
            transport_attestation_ref=_exact_ref(f"transport-{index}", sha=SHA_C),
            captured_at=T5,
        )
        closure = {
            "source": source,
            "source_request": request,
            "provider_response_id": response_id,
            "policy": stack["policy"],
            "transport_observation": observation,
            "compressed_entity": stack["body"],
            "decoded_entity": stack["body"],
        }
        claim = "Revenue declined."
        start = stack["body"].index(claim.encode())
        fact = build_validated_fact(
            capture_receipt=capture,
            capture_closure=closure,
            parser_input=stack["body"],
            subject_id="600000.SH",
            claim=claim,
            byte_start=start,
            byte_end=start + len(claim.encode()),
            source_class="ORIGINAL_SOURCE",
            canonical_source_id=f"CANONICAL_{index}",
            original_source_id="SAME_ORIGINAL",
            syndication_group_id=None,
            conflict_status="NONE",
            prompt_injection_detected=False,
            validated_at=T6,
        )
        bundles.append(
            {
                "fact": fact,
                "capture_receipt": capture,
                "capture_closure": closure,
                "parser_input": stack["body"],
            }
        )
    projection = build_decision_evidence_projection(
        packet=stack["packet"],
        public_capability=stack["public_capability"],
        search_run=stack["search_run"],
        round_bundles=stack["rounds"],
        decision_receipt=stack["decision_receipt"],
        decision_validation_closure=stack["decision_validation_closure"],
        fact_bundles=bundles,
        knowledge_cutoff=T6,
    )
    request = build_committee_request(
        capability=stack["private_capability"], projection=projection, requested_at=T6
    )
    response = build_committee_response(
        capability=stack["private_capability"],
        request=request,
        projection=projection,
        provider_response_id="PRIVATE_DUPLICATE_ORIGINAL",
        relative_score="0.200000000000",
        findings=[
            {
                "classification": "FACT",
                "text": "Two URLs repeat one original source.",
                "fact_refs": [bundle["fact"]["fact_id"] for bundle in bundles],
            }
        ],
        thesis_gaps=[],
        source_conflicts=[],
        time_conflicts=[],
        unverifiable_claims=[],
        responded_at=T7,
    )
    advisory = build_advisory_rank(
        projection=projection,
        response=response,
        response_validation_closure={
            "capability": stack["private_capability"],
            "request": request,
        },
        fact_bundles=bundles,
        advisory_weight="0.100000000000",
        issued_at=T7,
    )
    assert advisory["advisory_rank"] == advisory["deterministic_rank"]
    assert advisory["absolute_delta"] == "0.000000000000"
    assert advisory["reason_codes"] == ["INSUFFICIENT_VALIDATED_FACT_AUTHORITY"]
