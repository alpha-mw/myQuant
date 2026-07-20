from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import quant_investor.v16.evidence_v2.readiness_v4 as readiness_v4_module
from quant_investor.codex_review.models import MenuSeal, Stage2Request, Stage2Response
from quant_investor.v16.evidence_v2.codex_authority_plan_v2 import (
    PLANNED_ARTIFACT_SCHEMAS,
    READINESS_V3_SCHEMA,
    CodexAuthorityPlanEvidenceBundleV2,
    PlannedCodexArtifactV2,
    build_codex_authority_source_plan_v2,
)
from quant_investor.v16.evidence_v2.codex_ic_source_v2 import (
    CodexICSourceEvidenceBundleV2,
    FullUnionPosteriorEvidenceBundleV2,
    build_codex_ic_source_status_v2,
)
from quant_investor.v16.evidence_v2.contracts import (
    EVIDENCE_REF_SCHEMA,
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    decode_f64,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.execution_handoff_source_v2 import (
    ExecutionHandoffSourceEvidenceBundleV2,
    build_execution_source_status_v2,
    build_handoff_source_status_v2,
)
from quant_investor.v16.evidence_v2.publication_plan_v2 import (
    PLANNED_PUBLICATION_SCHEMAS,
    PRIVATE_ROOT_POLICY,
    PUBLICATION_OUTPUT_ORDER,
    PUBLICATION_PLAN_FILENAME,
    PUBLICATION_PLAN_SCHEMA,
    PlannedPublicationArtifactV2,
    PublicationPlanEvidenceBundleV2,
    build_publication_source_plan_v2,
    validate_publication_source_plan_v2,
)
from quant_investor.v16.evidence_v2.readiness_v3 import ReadinessEvidenceBundleV3
from quant_investor.v16.evidence_v2.readiness_v4 import (
    ReadinessEvidenceBundleV4,
    build_v16_run_readiness_v4,
)
from tests.unit.test_v16_codex_ic_source_v2 import (
    ROOT,
    _bound,
    _review_bound,
    _symbol_set_sha256,
)
from tests.unit.test_v16_evidence_v2_posterior import (
    ATTEMPT_ID,
    _build as _build_real_posterior,
    _inputs as _real_posterior_inputs,
)


def _bound_at(path: str, payload: dict[str, Any]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=str(payload["schema_version"]),
            absolute_path=path,
            byte_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_sha256=semantic_sha256(payload),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


def _actual_readiness_v4(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[BoundCanonicalArtifact, ReadinessEvidenceBundleV4]:
    inputs = _real_posterior_inputs()
    posterior_payload = _build_real_posterior(inputs)
    posterior = _bound("sources/publication-full-union-posterior.json", posterior_payload)
    stage1 = inputs["stage1"].read()
    rows = {item["symbol"]: item for item in posterior_payload["posteriors"]}
    symbols = list(posterior_payload["menu_symbols"])
    menu_items: list[dict[str, Any]] = []
    for symbol in symbols:
        source = rows[symbol]
        menu_items.append(
            {
                "symbol": symbol,
                "posterior_win_rate": decode_f64(source["posterior_win_rate"]),
                "posterior_expected_alpha": decode_f64(
                    source["posterior_expected_alpha"]
                ),
                "posterior_edge_after_costs": decode_f64(
                    source["posterior_edge_after_costs"]
                ),
                "branch_evidence": [
                    {
                        "branch": item["branch"],
                        "raw_score": decode_f64(item["raw_score"]),
                        "confidence": decode_f64(item["confidence"]),
                        "calibrated_probability": decode_f64(
                            item["calibrated_probability"]
                        ),
                        "evidence_ids": item["evidence_ids"],
                    }
                    for item in source["branch_evidence"]
                ],
                "retrieval_advisory": [
                    {
                        "symbol": symbol,
                        "branch": item["branch"],
                        "supporting_fact_ids": item["supporting_fact_ids"],
                        "contradicting_fact_ids": item["contradicting_fact_ids"],
                        "conflict_note": item["conflict_note"] or "",
                    }
                    for item in source["retrieval_advisory"]
                ],
                "risk_advisory": {
                    "severity": "low",
                    "flags": [],
                    "scenarios": [],
                    "suggestions": [],
                    "rationale": "advisory only",
                },
                "existing_weight": 0.0,
                "reference_price": 10.0,
                "existing_shares": 0.0,
            }
        )
    sealed_at = stage1.request.decision_cutoff_at + timedelta(hours=1)
    menu = _review_bound(
        "future/menu.json",
        MenuSeal,
        {
            "schema_version": "codex-review-menu.v1",
            "run_id": stage1.request.run_id,
            "stage1_response_sha256": inputs[
                "stage1"
            ].response.reference.semantic_sha256,
            "symbols": symbols,
            "items": menu_items,
            "existing_weights": {symbol: 0.0 for symbol in symbols},
            "sealed_at": sealed_at.isoformat().replace("+00:00", "Z"),
        },
        "menu_sha256",
    )
    parsed_menu = MenuSeal.model_validate_json(menu.payload)
    stage1_request = stage1.request.model_dump(mode="json")
    request = _review_bound(
        "future/stage2_request.json",
        Stage2Request,
        {
            "schema_version": "codex-review-stage2-request.v1",
            "run_id": stage1.request.run_id,
            "stage": 2,
            "git_sha": stage1_request["git_sha"],
            "config_path": stage1_request["config_path"],
            "config_sha256": stage1_request["config_sha256"],
            "prompt_path": stage1_request["prompt_path"],
            "prompt_sha256": stage1_request["prompt_sha256"],
            "model_id": stage1_request["model_id"],
            "model_sha256": stage1_request["model_sha256"],
            "pit_pointer_path": stage1_request["pit_pointer_path"],
            "pit_pointer_sha256": stage1_request["pit_pointer_sha256"],
            "symbol_set": symbols,
            "symbol_set_sha256": _symbol_set_sha256(symbols),
            "predecessor_sha256": inputs[
                "stage1"
            ].response.reference.semantic_sha256,
            "decision_cutoff_at": stage1_request["decision_cutoff_at"],
            "expires_at": stage1_request["expires_at"],
            "menu_sha256": parsed_menu.menu_sha256,
            "existing_weights": parsed_menu.existing_weights,
            "menu": menu_items,
        },
        "request_sha256",
    )
    parsed_request = Stage2Request.model_validate_json(request.payload)
    response_common = {
        key: value
        for key, value in parsed_request.model_dump(mode="json").items()
        if key
        in {
            "run_id",
            "stage",
            "git_sha",
            "config_path",
            "config_sha256",
            "prompt_path",
            "prompt_sha256",
            "model_id",
            "model_sha256",
            "pit_pointer_path",
            "pit_pointer_sha256",
            "symbol_set",
            "symbol_set_sha256",
            "predecessor_sha256",
            "decision_cutoff_at",
            "expires_at",
            "request_sha256",
        }
    }
    response = _review_bound(
        "future/stage2_response.json",
        Stage2Response,
        {
            "schema_version": "codex-review-stage2-response.v1",
            **response_common,
            "menu_sha256": parsed_menu.menu_sha256,
            "verdicts": [
                {
                    "symbol": symbol,
                    "action": "BUY" if index == 0 else "AVOID",
                    "selected_for_portfolio": index == 0,
                    "target_weight": 0.6 if index == 0 else 0.0,
                    "rationale": f"offline publication fixture for {symbol}",
                    "severe_risks": [],
                    "risk_acceptance_rationale": "",
                }
                for index, symbol in enumerate(symbols)
            ],
            "cash_ratio": 0.4,
        },
        "response_sha256",
    )
    readiness_v3 = _bound(
        "sources/publication-readiness-v3.json",
        seal_semantic(
            {
                "schema_version": READINESS_V3_SCHEMA,
                "run_id": stage1.request.run_id,
                "generated_at": "2026-07-20T00:30:00Z",
                "analysis_trade_date": "2026-07-17",
                "formal_branches": [
                    {"branch": branch, "weight": "0.25"}
                    for branch in ("quant", "fundamental", "macro", "llm")
                ],
                "retrieval_role": "evidence_only_no_scoring_or_weight",
                "risk_advisor_role": "advisory_only",
                "readiness_status": "no_new_risk",
                "blockers": [
                    "codex_authority_v2_disconnected_from_authorizing_consumers"
                ],
                "blocker_sources": [
                    {
                        "blocker": (
                            "codex_authority_v2_disconnected_from_authorizing_consumers"
                        ),
                        "source": "readiness_v3_foundation",
                    }
                ],
            }
        ),
    )
    planned_codex = {
        key: PlannedCodexArtifactV2(
            absolute_path=f"{ROOT}/future/{key}.json",
            artifact_schema=schema,
        )
        for key, schema in PLANNED_ARTIFACT_SCHEMAS.items()
    }
    codex_plan = _bound(
        "publication-codex-source-plan.json",
        build_codex_authority_source_plan_v2(
            protocol_attempt_id=ATTEMPT_ID,
            run_id=stage1.request.run_id,
            private_root=ROOT,
            full_union_posterior_ref=posterior.reference,
            readiness_v3_ref=readiness_v3.reference,
            planned_artifacts=planned_codex,
        ),
    )
    codex_plan_evidence = CodexAuthorityPlanEvidenceBundleV2(
        plan=codex_plan,
        full_union_posterior=posterior,
        readiness_v3=readiness_v3,
    )
    ic_evidence = CodexICSourceEvidenceBundleV2(
        plan=codex_plan_evidence,
        posterior=FullUnionPosteriorEvidenceBundleV2(
            posterior=posterior,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=tuple(inputs["formal"]),
            cost_artifacts=tuple(inputs["costs"]),
        ),
        menu=menu,
        stage2_request=request,
        stage2_response=response,
    )
    ic_status = _bound(
        "future/ic_status.json",
        build_codex_ic_source_status_v2(evidence=ic_evidence),
    )
    execution_evidence = ExecutionHandoffSourceEvidenceBundleV2(
        plan=codex_plan_evidence,
        ic_status=ic_status,
        ic_evidence=ic_evidence,
    )
    execution_status = _bound(
        "future/execution_status.json",
        build_execution_source_status_v2(evidence=execution_evidence),
    )
    handoff_status = _bound(
        "future/handoff_status.json",
        build_handoff_source_status_v2(evidence=execution_evidence),
    )
    monkeypatch.setattr(
        readiness_v4_module,
        "validate_v16_run_readiness_v3",
        lambda value, **_kwargs: value,
    )
    readiness_v3_evidence = ReadinessEvidenceBundleV3(
        factor_production_set=None,  # type: ignore[arg-type]
        schedule_lineage=None,  # type: ignore[arg-type]
        calibration_status=None,  # type: ignore[arg-type]
        calibration_evidence=None,  # type: ignore[arg-type]
    )
    readiness_evidence = ReadinessEvidenceBundleV4(
        readiness_v3=readiness_v3,
        readiness_v3_evidence=readiness_v3_evidence,
        plan=codex_plan_evidence,
        ic_status=ic_status,
        ic_evidence=ic_evidence,
        execution_status=execution_status,
        handoff_status=handoff_status,
        execution_handoff_evidence=execution_evidence,
    )
    readiness_payload = build_v16_run_readiness_v4(evidence=readiness_evidence)
    readiness = _bound(
        "future/readiness_v4.json",
        readiness_payload,
    )
    return readiness, readiness_evidence


def _publication_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], BoundCanonicalArtifact, ReadinessEvidenceBundleV4]:
    readiness, readiness_evidence = _actual_readiness_v4(monkeypatch)
    root = tmp_path / "publication-run"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    planned = {
        key: PlannedPublicationArtifactV2(
            absolute_path=str(root / f"{key}.json"),
            artifact_schema=schema,
        )
        for key, schema in PLANNED_PUBLICATION_SCHEMAS.items()
    }
    plan = build_publication_source_plan_v2(
        protocol_attempt_id=ATTEMPT_ID,
        run_id=str(readiness.read()["run_id"]),
        private_root=str(root),
        plan_absolute_path=str(root / PUBLICATION_PLAN_FILENAME),
        readiness_v4_ref=readiness.reference,
        planned_artifacts=planned,
    )
    return plan, readiness, readiness_evidence


def test_publication_plan_predeclares_exact_order_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, readiness, readiness_evidence = _publication_inputs(tmp_path, monkeypatch)
    validated = validate_publication_source_plan_v2(plan)

    assert validated["schema_version"] == PUBLICATION_PLAN_SCHEMA
    assert validated["output_order"] == list(PUBLICATION_OUTPUT_ORDER)
    assert list(validated["planned_artifacts"]) == list(
        PLANNED_PUBLICATION_SCHEMAS
    )
    assert validated["activation_candidate"] is False
    assert validated["new_risk_authorized"] is False
    bound_plan = _bound_at(validated["plan_absolute_path"], validated)
    bundle = PublicationPlanEvidenceBundleV2(
        plan=bound_plan,
        readiness_v4=readiness,
        readiness_evidence=readiness_evidence,
    )
    assert bundle.read()[0] == validated


def test_publication_plan_rejects_duplicate_or_escaped_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _ = _publication_inputs(tmp_path, monkeypatch)
    duplicate = deepcopy(plan)
    duplicate.pop("semantic_sha256")
    duplicate["planned_artifacts"]["dashboard_snapshot"]["absolute_path"] = duplicate[
        "planned_artifacts"
    ]["candidate_report"]["absolute_path"]
    with pytest.raises(EvidenceV2Error, match="paths must be unique"):
        validate_publication_source_plan_v2(seal_semantic(duplicate))

    escaped = deepcopy(plan)
    escaped.pop("semantic_sha256")
    escaped["planned_artifacts"]["candidate_report"]["absolute_path"] = str(
        tmp_path / "outside.json"
    )
    with pytest.raises(EvidenceV2Error, match="direct private-root child"):
        validate_publication_source_plan_v2(seal_semantic(escaped))


def test_publication_plan_builder_has_no_authority_or_current_state_inputs() -> None:
    names = set(inspect.signature(build_publication_source_plan_v2).parameters)
    assert not names.intersection(
        {
            "human_authorization",
            "human_authorized",
            "capital_map",
            "execution_plan",
            "market_state",
            "portfolio",
            "activation_receipt",
            "current_state",
        }
    )
