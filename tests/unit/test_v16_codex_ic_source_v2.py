from __future__ import annotations

import ast
from copy import deepcopy
from datetime import timedelta
import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import quant_investor.v16.evidence_v2.codex_ic_source_v2 as codex_ic_module
from quant_investor.codex_review.models import MenuSeal, Stage2Request, Stage2Response
from quant_investor.codex_review.storage import (
    canonical_json_bytes as review_canonical_json_bytes,
    sha256_bytes as review_sha256_bytes,
)
from quant_investor.v16.evidence_v2.codex_authority_plan_v2 import (
    CODEX_AUTHORITY_PLAN_SCHEMA,
    CODEX_IC_STATUS_SCHEMA,
    FULL_UNION_POSTERIOR_SCHEMA,
    PLANNED_ARTIFACT_SCHEMAS,
    PRIVATE_ROOT_POLICY,
    READINESS_V3_SCHEMA,
    CodexAuthorityPlanEvidenceBundleV2,
    PlannedCodexArtifactV2,
    build_codex_authority_source_plan_v2,
)
from quant_investor.v16.evidence_v2.codex_ic_source_v2 import (
    CodexICSourceEvidenceBundleV2,
    FullUnionPosteriorEvidenceBundleV2,
    build_codex_ic_source_status_v2,
    validate_codex_ic_source_status_v2,
)
from quant_investor.v16.evidence_v2.contracts import (
    EVIDENCE_REF_SCHEMA,
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    decode_f64,
    encode_f64,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.posterior import (
    BoundReviewArtifact,
    Stage1ReviewBinding,
)
from quant_investor.v16.evidence_v2.posterior_runtime import PosteriorRuntimeArtifacts
from tests.unit.test_v16_evidence_v2_posterior import (
    ATTEMPT_ID as REAL_POSTERIOR_ATTEMPT_ID,
    _build as _build_real_posterior,
    _inputs as _real_posterior_inputs,
)


ROOT = "/private/v16-codex-ic-v2"
RUN_ID = "run-v16-codex-001"
ATTEMPT_ID = "attempt-v16-codex-001"
STAGE1_SHA = hashlib.sha256(b"stage1-response").hexdigest()
HASH_A = hashlib.sha256(b"a").hexdigest()
HASH_B = hashlib.sha256(b"b").hexdigest()


def _ref(
    path: str,
    schema: str,
    raw: bytes,
    semantic: str,
) -> EvidenceRef:
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=f"{ROOT}/{path}",
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=semantic,
        root_policy=PRIVATE_ROOT_POLICY,
    )


def _bound(path: str, payload: dict[str, Any]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=_ref(
            path,
            str(payload["schema_version"]),
            raw,
            semantic_sha256(payload),
        ),
        payload=raw,
    )


def _review_bound(
    path: str,
    model_type: type[MenuSeal] | type[Stage2Request] | type[Stage2Response],
    value: dict[str, Any],
    digest_field: str,
) -> BoundReviewArtifact:
    payload = deepcopy(value)
    payload[digest_field] = review_sha256_bytes(review_canonical_json_bytes(payload))
    model = model_type.model_validate(payload)
    normalized = model.model_dump(mode="json")
    raw = review_canonical_json_bytes(normalized)
    return BoundReviewArtifact(
        reference=_ref(
            path,
            str(normalized["schema_version"]),
            raw,
            str(normalized[digest_field]),
        ),
        payload=raw,
    )


def _symbol_set_sha256(symbols: list[str]) -> str:
    raw = review_canonical_json_bytes(sorted(set(symbols)))
    return review_sha256_bytes(raw[:-1])


def _menu_item(symbol: str, *, win_rate: float) -> dict[str, Any]:
    branches = []
    for index, branch in enumerate(("quant", "fundamental", "macro", "llm")):
        branches.append(
            {
                "branch": branch,
                "raw_score": 0.1 + index * 0.01,
                "confidence": 0.8,
                "calibrated_probability": 0.55 + index * 0.01,
                "evidence_ids": [f"{symbol}-{branch}-evidence"],
            }
        )
    return {
        "symbol": symbol,
        "posterior_win_rate": win_rate,
        "posterior_expected_alpha": 0.03,
        "posterior_edge_after_costs": 0.02,
        "branch_evidence": branches,
        "retrieval_advisory": [
            {
                "symbol": symbol,
                "branch": "quant",
                "supporting_fact_ids": [f"{symbol}-fact"],
                "contradicting_fact_ids": [],
                "conflict_note": "",
            }
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


def _posterior_row(item: dict[str, Any], *, win_rate: float) -> dict[str, Any]:
    return {
        "symbol": item["symbol"],
        "posterior_win_rate": encode_f64(win_rate),
        "posterior_expected_alpha": encode_f64(item["posterior_expected_alpha"]),
        "posterior_edge_after_costs": encode_f64(item["posterior_edge_after_costs"]),
        "branch_evidence": [
            {
                "branch": branch["branch"],
                "raw_score": encode_f64(branch["raw_score"]),
                "confidence": encode_f64(branch["confidence"]),
                "calibrated_probability": encode_f64(
                    branch["calibrated_probability"]
                ),
                "evidence_ids": branch["evidence_ids"],
            }
            for branch in item["branch_evidence"]
        ],
        "retrieval_advisory": [
            {
                "branch": row["branch"],
                "supporting_fact_ids": row["supporting_fact_ids"],
                "contradicting_fact_ids": row["contradicting_fact_ids"],
                "conflict_note": row["conflict_note"] or None,
            }
            for row in item["retrieval_advisory"]
        ],
    }


def _dummy_runtime_artifacts(dummy: BoundCanonicalArtifact) -> PosteriorRuntimeArtifacts:
    return PosteriorRuntimeArtifacts(
        model_bundles=(("quant", dummy),),
        prior_training=dummy,
        likelihood_training=dummy,
        return_model_parameters=dummy,
        return_model_training=dummy,
        bootstrap_offsets=dummy,
        bootstrap_training=dummy,
        correlation_matrix=dummy,
        correlation_training=dummy,
    )


def _evidence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    symbol_count: int = 2,
    posterior_win_rate: float = 0.61,
    menu_win_rate: float = 0.61,
    omit_last_verdict: bool = False,
    cash_ratio: float | None = None,
) -> CodexICSourceEvidenceBundleV2:
    monkeypatch.setattr(
        codex_ic_module,
        "validate_full_union_posterior_evidence",
        lambda value, **_kwargs: value,
    )
    symbols = [f"S{index:02d}" for index in range(symbol_count)]
    menu_items = [
        _menu_item(symbol, win_rate=menu_win_rate) for symbol in symbols
    ]
    if symbol_count == 2:
        target_weights = [0.6, 0.0]
        actions = ["BUY", "AVOID"]
        selected = [True, False]
        default_cash = 0.4
    else:
        target_weights = [0.05] * symbol_count
        actions = ["BUY"] * symbol_count
        selected = [True] * symbol_count
        default_cash = 1.0 - sum(target_weights)
    existing_weights = {symbol: 0.0 for symbol in symbols}
    menu_payload = {
        "schema_version": "codex-review-menu.v1",
        "run_id": RUN_ID,
        "stage1_response_sha256": STAGE1_SHA,
        "symbols": symbols,
        "items": menu_items,
        "existing_weights": existing_weights,
        "sealed_at": "2026-07-20T01:00:00Z",
    }
    menu = _review_bound(
        "future/menu.json",
        MenuSeal,
        menu_payload,
        "menu_sha256",
    )
    parsed_menu = MenuSeal.model_validate_json(menu.payload)
    request_payload = {
        "schema_version": "codex-review-stage2-request.v1",
        "run_id": RUN_ID,
        "stage": 2,
        "git_sha": "abcdef0",
        "config_path": "config/v16.yaml",
        "config_sha256": HASH_A,
        "prompt_path": "prompts/v16.txt",
        "prompt_sha256": HASH_B,
        "model_id": "offline-model-v1",
        "model_sha256": hashlib.sha256(b"offline-model-v1").hexdigest(),
        "pit_pointer_path": "data/parquet/cn/_latest.json",
        "pit_pointer_sha256": HASH_A,
        "symbol_set": symbols,
        "symbol_set_sha256": _symbol_set_sha256(symbols),
        "predecessor_sha256": STAGE1_SHA,
        "decision_cutoff_at": "2026-07-20T00:00:00Z",
        "expires_at": "2026-07-20T02:00:00Z",
        "menu_sha256": parsed_menu.menu_sha256,
        "existing_weights": existing_weights,
        "menu": menu_items,
    }
    request = _review_bound(
        "future/stage2_request.json",
        Stage2Request,
        request_payload,
        "request_sha256",
    )
    parsed_request = Stage2Request.model_validate_json(request.payload)
    verdicts = [
        {
            "symbol": symbol,
            "action": action,
            "selected_for_portfolio": is_selected,
            "target_weight": weight,
            "rationale": f"IC decision for {symbol}",
            "severe_risks": [],
            "risk_acceptance_rationale": "",
        }
        for symbol, action, is_selected, weight in zip(
            symbols,
            actions,
            selected,
            target_weights,
            strict=True,
        )
    ]
    if omit_last_verdict:
        verdicts.pop()
    response_payload = {
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
    response_payload.update(
        {
            "schema_version": "codex-review-stage2-response.v1",
            "menu_sha256": parsed_menu.menu_sha256,
            "verdicts": verdicts,
            "cash_ratio": default_cash if cash_ratio is None else cash_ratio,
        }
    )
    response = _review_bound(
        "future/stage2_response.json",
        Stage2Response,
        response_payload,
        "response_sha256",
    )

    stage1_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema="codex-review-stage1-response.v1",
        absolute_path=f"{ROOT}/sources/stage1-response.json",
        byte_sha256=hashlib.sha256(b"stage1-response").hexdigest(),
        semantic_sha256=STAGE1_SHA,
        root_policy=PRIVATE_ROOT_POLICY,
    )
    posterior_payload = seal_semantic(
        {
            "schema_version": FULL_UNION_POSTERIOR_SCHEMA,
            "protocol_attempt_id": ATTEMPT_ID,
            "run_id": RUN_ID,
            "stage1_response_ref": stage1_ref.to_dict(),
            "menu_symbols": symbols,
            "posteriors": [
                _posterior_row(item, win_rate=posterior_win_rate)
                for item in menu_items
            ],
        }
    )
    posterior = _bound("sources/full-union-posterior.json", posterior_payload)
    readiness_v3 = _bound(
        "sources/readiness-v3.json",
        seal_semantic(
            {
                "schema_version": READINESS_V3_SCHEMA,
                "run_id": RUN_ID,
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
    planned = {
        key: PlannedCodexArtifactV2(
            absolute_path=f"{ROOT}/future/{key}.json",
            artifact_schema=schema,
        )
        for key, schema in PLANNED_ARTIFACT_SCHEMAS.items()
    }
    plan_payload = build_codex_authority_source_plan_v2(
        protocol_attempt_id=ATTEMPT_ID,
        run_id=RUN_ID,
        private_root=ROOT,
        full_union_posterior_ref=posterior.reference,
        readiness_v3_ref=readiness_v3.reference,
        planned_artifacts=planned,
    )
    plan = _bound("source-plan.json", plan_payload)
    dummy = _bound(
        "sources/dummy.json",
        seal_semantic({"schema_version": "v16.test-dummy.v1"}),
    )
    stage1_dummy = BoundReviewArtifact(
        reference=stage1_ref,
        payload=b"stage1-response",
    )
    stage1_binding = Stage1ReviewBinding(
        request=stage1_dummy,
        response=stage1_dummy,
    )
    return CodexICSourceEvidenceBundleV2(
        plan=CodexAuthorityPlanEvidenceBundleV2(
            plan=plan,
            full_union_posterior=posterior,
            readiness_v3=readiness_v3,
        ),
        posterior=FullUnionPosteriorEvidenceBundleV2(
            posterior=posterior,
            stage1_binding=stage1_binding,
            runtime_artifacts=_dummy_runtime_artifacts(dummy),
            cost_model=dummy,
            formal_branch_artifacts=(dummy,),
            cost_artifacts=(dummy,),
        ),
        menu=menu,
        stage2_request=request,
        stage2_response=response,
    )


def test_codex_ic_status_recomputes_menu_and_validates_exact_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(monkeypatch)
    status = build_codex_ic_source_status_v2(evidence=evidence)

    assert status["schema_version"] == CODEX_IC_STATUS_SCHEMA
    assert status["menu_symbols"] == ["S00", "S01"]
    assert [item["action"] for item in status["allocations"]] == ["BUY", "AVOID"]
    assert status["allocations"][0]["target_weight"] == encode_f64(0.6)
    assert status["cash_ratio"] == encode_f64(0.4)
    assert status["target_plus_cash"] == encode_f64(1.0)
    assert status["positive_weight_count"] == 1
    assert status["posterior_recomputed_from_bound_sources"] is True
    assert status["source_recomputation_complete"] is False
    assert status["readiness_status"] == "no_new_risk"
    assert status["new_risk_authorized"] is False
    assert "codex_requirement_unsupported:requirement=stage2_model_execution_attestation" in status[
        "blockers"
    ]
    assert validate_codex_ic_source_status_v2(status, evidence=evidence) == status


def test_codex_ic_rejects_posterior_menu_value_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(
        monkeypatch,
        posterior_win_rate=0.62,
        menu_win_rate=0.61,
    )
    with pytest.raises(EvidenceV2Error, match="posterior value drift"):
        build_codex_ic_source_status_v2(evidence=evidence)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"omit_last_verdict": True}, "symbol-set drift"),
        ({"cash_ratio": 0.3}, "target weights plus cash_ratio"),
        ({"symbol_count": 13}, "positive target weights exceed 12"),
    ],
)
def test_codex_ic_rejects_incomplete_or_illegal_stage2_allocation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    evidence = _evidence(monkeypatch, **kwargs)
    with pytest.raises(EvidenceV2Error, match=message):
        build_codex_ic_source_status_v2(evidence=evidence)


def test_codex_ic_has_no_legacy_authorization_or_writer_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(monkeypatch)
    parameters = set(inspect.signature(build_codex_ic_source_status_v2).parameters)
    assert parameters == {"evidence"}
    with pytest.raises(TypeError):
        build_codex_ic_source_status_v2(  # type: ignore[call-arg]
            evidence=evidence,
            human_authorization={"receipt_sha256": HASH_A},
        )

    source = Path(codex_ic_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "quant_investor.codex_review.workflow" not in imported
    assert "CapitalMap" not in imported
    assert "HumanAuthorization" not in imported
    assert "ExecutionGate" not in imported
    assert "atomic_write_bytes" not in imported
    assert "write_exact_once" not in imported


def test_codex_ic_status_rejects_resealed_authorization_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(monkeypatch)
    status = build_codex_ic_source_status_v2(evidence=evidence)
    tampered = deepcopy(status)
    tampered.pop("semantic_sha256")
    tampered["human_authorized"] = True

    with pytest.raises(EvidenceV2Error, match="fields mismatch"):
        validate_codex_ic_source_status_v2(seal_semantic(tampered), evidence=evidence)


def test_codex_ic_composes_with_actual_recomputed_full_union_posterior() -> None:
    inputs = _real_posterior_inputs()
    posterior_payload = _build_real_posterior(inputs)
    posterior = _bound("sources/real-full-union-posterior.json", posterior_payload)
    stage1 = inputs["stage1"].read()
    rows = {
        item["symbol"]: item for item in posterior_payload["posteriors"]
    }
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
    request_payload = {
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
    }
    request = _review_bound(
        "future/stage2_request.json",
        Stage2Request,
        request_payload,
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
                    "rationale": f"offline IC fixture for {symbol}",
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
        "sources/real-readiness-v3.json",
        seal_semantic(
            {
                "schema_version": READINESS_V3_SCHEMA,
                "run_id": stage1.request.run_id,
                "readiness_status": "no_new_risk",
            }
        ),
    )
    planned = {
        key: PlannedCodexArtifactV2(
            absolute_path=f"{ROOT}/future/{key}.json",
            artifact_schema=schema,
        )
        for key, schema in PLANNED_ARTIFACT_SCHEMAS.items()
    }
    plan = _bound(
        "real-source-plan.json",
        build_codex_authority_source_plan_v2(
            protocol_attempt_id=REAL_POSTERIOR_ATTEMPT_ID,
            run_id=stage1.request.run_id,
            private_root=ROOT,
            full_union_posterior_ref=posterior.reference,
            readiness_v3_ref=readiness_v3.reference,
            planned_artifacts=planned,
        ),
    )
    evidence = CodexICSourceEvidenceBundleV2(
        plan=CodexAuthorityPlanEvidenceBundleV2(
            plan=plan,
            full_union_posterior=posterior,
            readiness_v3=readiness_v3,
        ),
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

    status = build_codex_ic_source_status_v2(evidence=evidence)
    assert status["menu_symbols"] == symbols
    assert status["posterior_recomputed_from_bound_sources"] is True
