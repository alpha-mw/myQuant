from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect

import pytest

from quant_investor.v16.evidence_v2.codex_authority_plan_v2 import (
    CODEX_AUTHORITY_PLAN_SCHEMA,
    FULL_UNION_POSTERIOR_SCHEMA,
    PLANNED_ARTIFACT_SCHEMAS,
    PRIVATE_ROOT_POLICY,
    READINESS_V3_SCHEMA,
    UNSUPPORTED_REQUIREMENTS,
    CodexAuthorityPlanEvidenceBundleV2,
    PlannedCodexArtifactV2,
    build_codex_authority_source_plan_v2,
    validate_codex_authority_source_plan_v2,
)
from quant_investor.v16.evidence_v2.contracts import (
    EVIDENCE_REF_SCHEMA,
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
)


ROOT = "/private/v16-codex-authority-v2"


def _bound(name: str, schema: str) -> BoundCanonicalArtifact:
    payload = seal_semantic(
        {
            "schema_version": schema,
            "fixture_id": name,
        }
    )
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=schema,
            absolute_path=f"{ROOT}/sources/{name}.json",
            byte_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_sha256=semantic_sha256(payload),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


def _planned() -> dict[str, PlannedCodexArtifactV2]:
    return {
        key: PlannedCodexArtifactV2(
            absolute_path=f"{ROOT}/future/{key}.json",
            artifact_schema=schema,
        )
        for key, schema in PLANNED_ARTIFACT_SCHEMAS.items()
    }


def _fixtures() -> tuple[
    dict,
    BoundCanonicalArtifact,
    BoundCanonicalArtifact,
]:
    posterior = _bound("full-union-posterior", FULL_UNION_POSTERIOR_SCHEMA)
    readiness_v3 = _bound("readiness-v3", READINESS_V3_SCHEMA)
    plan = build_codex_authority_source_plan_v2(
        protocol_attempt_id="attempt-v16-codex-001",
        run_id="run-v16-codex-001",
        private_root=ROOT,
        full_union_posterior_ref=posterior.reference,
        readiness_v3_ref=readiness_v3.reference,
        planned_artifacts=_planned(),
    )
    return plan, posterior, readiness_v3


def test_plan_v2_predeclares_exact_disconnected_nonauthorizing_shape() -> None:
    plan, _, _ = _fixtures()
    validated = validate_codex_authority_source_plan_v2(plan)

    assert validated["schema_version"] == CODEX_AUTHORITY_PLAN_SCHEMA
    assert list(validated["planned_artifacts"]) == list(PLANNED_ARTIFACT_SCHEMAS)
    assert validated["unsupported_requirement_ids"] == list(
        UNSUPPORTED_REQUIREMENTS
    )
    assert validated["activation_candidate"] is False
    assert validated["new_risk_authorized"] is False
    assert validated["production_apply_enabled"] is False
    serialized = canonical_json_bytes(validated)
    for forbidden in (
        b'"capital_map"',
        b'"human_authorization"',
        b'"human_authorized"',
        b'"shares"',
        b'"orders"',
    ):
        assert forbidden not in serialized


def test_plan_v2_rejects_escape_duplicate_path_and_schema_drift() -> None:
    plan, _, _ = _fixtures()
    escaped = _planned()
    escaped["menu"] = PlannedCodexArtifactV2(
        absolute_path="/private/outside/menu.json",
        artifact_schema=PLANNED_ARTIFACT_SCHEMAS["menu"],
    )
    with pytest.raises(EvidenceV2Error, match="private-root child"):
        build_codex_authority_source_plan_v2(
            protocol_attempt_id="attempt-v16-codex-001",
            run_id="run-v16-codex-001",
            private_root=ROOT,
            full_union_posterior_ref=EvidenceRef.from_dict(
                plan["full_union_posterior_ref"]
            ),
            readiness_v3_ref=EvidenceRef.from_dict(plan["readiness_v3_ref"]),
            planned_artifacts=escaped,
        )

    duplicate = deepcopy(plan)
    duplicate.pop("semantic_sha256")
    duplicate["planned_artifacts"]["stage2_request"]["absolute_path"] = duplicate[
        "planned_artifacts"
    ]["menu"]["absolute_path"]
    with pytest.raises(EvidenceV2Error, match="paths must be unique"):
        validate_codex_authority_source_plan_v2(seal_semantic(duplicate))

    drifted = deepcopy(plan)
    drifted.pop("semantic_sha256")
    drifted["planned_artifacts"]["menu"]["artifact_schema"] = "wrong.schema"
    with pytest.raises(EvidenceV2Error, match="schema mismatch: menu"):
        validate_codex_authority_source_plan_v2(seal_semantic(drifted))


def test_plan_evidence_bundle_rejects_rebound_source_reference() -> None:
    plan, posterior, readiness_v3 = _fixtures()
    bound_plan = _bound("plan-placeholder", CODEX_AUTHORITY_PLAN_SCHEMA)
    plan_raw = canonical_json_bytes(plan)
    bound_plan = BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=CODEX_AUTHORITY_PLAN_SCHEMA,
            absolute_path=f"{ROOT}/plan.json",
            byte_sha256=hashlib.sha256(plan_raw).hexdigest(),
            semantic_sha256=semantic_sha256(plan),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=plan_raw,
    )
    bundle = CodexAuthorityPlanEvidenceBundleV2(
        plan=bound_plan,
        full_union_posterior=posterior,
        readiness_v3=readiness_v3,
    )
    assert bundle.read() == plan

    rebound = _bound("other-readiness-v3", READINESS_V3_SCHEMA)
    with pytest.raises(EvidenceV2Error, match="evidence ref drift"):
        CodexAuthorityPlanEvidenceBundleV2(
            plan=bound_plan,
            full_union_posterior=posterior,
            readiness_v3=rebound,
        ).read()


def test_plan_v2_builder_has_no_authorization_or_execution_inputs() -> None:
    names = set(inspect.signature(build_codex_authority_source_plan_v2).parameters)
    assert not names.intersection(
        {
            "capital_map",
            "human_authorization",
            "human_authorized",
            "shares",
            "orders",
            "execution_plan",
            "authorization_receipt",
        }
    )
