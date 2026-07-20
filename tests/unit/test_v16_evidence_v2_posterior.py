from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from quant_investor.bayesian.v16.types import CANONICAL_CORRELATION_KEYS
from quant_investor.codex_review.storage import (
    canonical_json_bytes as review_canonical_json_bytes,
)
from quant_investor.codex_review.workflow import (
    seal_json_payload,
    symbol_set_sha256,
)
from quant_investor.v16.candidate_pipeline import FORMAL_BRANCHES
from quant_investor.v16.evidence_v2.contracts import (
    BoundCanonicalArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    decode_f64,
    encode_f64,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.posterior import (
    FORMAL_BRANCH_PREDICTION_SCHEMA,
    POSTERIOR_COST_INPUT_SCHEMA,
    POSTERIOR_COST_MODEL_SCHEMA,
    BoundReviewArtifact,
    PosteriorRuntimeBundle,
    Stage1ReviewBinding,
    build_formal_branch_prediction,
    build_full_union_posterior_evidence,
    build_posterior_cost_input,
    build_posterior_cost_model,
    replay_stage1_formal_evidence,
    validate_full_union_posterior_evidence,
)
from quant_investor.v16.evidence_v2.posterior_runtime import (
    BaseRateObservation,
    LikelihoodTrainingObservation,
    PosteriorRuntimeArtifacts,
    build_base_rate_training_evidence,
    build_bootstrap_offsets,
    build_bootstrap_training_evidence,
    build_correlation_matrix,
    build_correlation_training_evidence,
    build_likelihood_training_evidence,
    build_return_model_parameters,
    build_return_model_training_evidence,
)
from quant_investor.v16.evidence_v2.runtime_identity import (
    MODEL_BUNDLE_SCHEMA,
    LLMProviderBuildIdentity,
    build_frozen_model_bundle,
)
from quant_investor.v16.evidence_v2.target import CostVector
from quant_investor.v16.stage1_contract import PITFactRow, build_stage1_fact_package

BASE_TIME = datetime(2026, 7, 18, 8, 0, tzinfo=timezone.utc)
ATTEMPT_ID = "attempt-v16-posterior-001"


def _ref(
    name: str,
    *,
    schema: str,
    payload: bytes | None = None,
    semantic: str | None = None,
) -> EvidenceRef:
    raw = payload if payload is not None else f"{name}:bytes".encode()
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=f"/private/evidence/{name}",
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=semantic or hashlib.sha256(f"{name}:semantic".encode()).hexdigest(),
        root_policy="v16.private-evidence-root.v2",
    )


def _bound(name: str, payload: dict[str, Any]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=_ref(
            name,
            schema=str(payload["schema_version"]),
            payload=raw,
            semantic=semantic_sha256(payload),
        ),
        payload=raw,
    )


def _review_artifact(
    name: str, payload: dict[str, Any], *, digest_field: str
) -> BoundReviewArtifact:
    raw = review_canonical_json_bytes(payload)
    return BoundReviewArtifact(
        reference=_ref(
            name,
            schema=str(payload["schema_version"]),
            payload=raw,
            semantic=str(payload[digest_field]),
        ),
        payload=raw,
    )


def _stage1_binding(*, retrieval_note: str = "") -> Stage1ReviewBinding:
    expiry = BASE_TIME + timedelta(days=1)
    macro_facts = {
        "regime": "neutral",
        "macro_score": 0.2,
        "liquidity_score": 0.1,
        "volatility_percentile": 40.0,
        "policy_signal": "neutral",
    }
    rows = [
        PITFactRow(
            symbol=symbol,
            stratum="funnel" if symbol in {"AAA", "BBB"} else "outside_funnel",
            eligibility_receipt_sha256=character * 64,
            formal_quant_score=score,
            quant_facts={
                "formal_score": score,
                "formal_confidence": 0.85,
                "factor_activation_receipt_sha256": "9" * 64,
            },
            fundamental_facts={
                "source": "tushare",
                "source_priority": "tushare_primary",
                "availability_date": "2026-07-17",
                "trade_date": "2026-07-17",
                "source_version": "fundamental-generation-001",
                "fin_roe": quality,
                "fin_roa": quality / 2.0,
                "fin_gross_margin": 0.3,
                "fin_net_margin": 0.1,
                "fin_revenue_yoy": quality,
                "fin_net_profit_yoy": quality,
                "fin_debt_to_assets": 0.4,
                "fin_current_ratio": 1.5,
                "fin_ocf_to_profit": 1.0,
                "pe": 20.0,
                "pb": 2.0,
                "ps": 3.0,
                "dividend_yield": 0.01,
            },
            macro_facts=macro_facts,
        )
        for symbol, character, score, quality in (
            ("AAA", "a", 0.8, 0.2),
            ("BBB", "b", -0.8, 0.05),
            ("CCC", "c", 0.2, 0.1),
        )
    ]
    package = build_stage1_fact_package(
        rows=rows,
        funnel_symbols=["AAA", "BBB"],
        cutoff_at=BASE_TIME.isoformat(),
        expires_at=expiry.isoformat(),
        pit_pointer_sha256="d" * 64,
    )
    model_id = "gpt-v16-review-build-001"
    request = seal_json_payload(
        {
            "schema_version": "codex-review-stage1-request.v1",
            "run_id": "run-posterior-001",
            "stage": 1,
            "git_sha": "e" * 40,
            "config_path": "/private/config/v16.json",
            "config_sha256": "f" * 64,
            "prompt_path": "/private/prompts/v16-stage1.md",
            "prompt_sha256": "1" * 64,
            "model_id": model_id,
            "model_sha256": hashlib.sha256(model_id.encode()).hexdigest(),
            "pit_pointer_path": "/private/data/cn/_latest.json",
            "pit_pointer_sha256": "d" * 64,
            "symbol_set": ["AAA", "BBB"],
            "symbol_set_sha256": symbol_set_sha256(["AAA", "BBB"]),
            "predecessor_sha256": "0" * 64,
            "decision_cutoff_at": BASE_TIME.isoformat().replace("+00:00", "Z"),
            "expires_at": expiry.isoformat().replace("+00:00", "Z"),
            "fact_package": package.to_dict(),
        },
        digest_field="request_sha256",
    )
    common = {
        key: request[key]
        for key in (
            "run_id",
            "git_sha",
            "config_path",
            "config_sha256",
            "prompt_path",
            "prompt_sha256",
            "model_id",
            "model_sha256",
            "pit_pointer_path",
            "pit_pointer_sha256",
            "predecessor_sha256",
            "decision_cutoff_at",
            "expires_at",
            "request_sha256",
        )
    }
    final_symbols = ["AAA", "BBB", "CCC"]
    scores = {"AAA": 0.8, "BBB": -0.8, "CCC": 0.2}
    response = seal_json_payload(
        {
            "schema_version": "codex-review-stage1-response.v1",
            **common,
            "stage": 1,
            "symbol_set": final_symbols,
            "symbol_set_sha256": symbol_set_sha256(final_symbols),
            "supplemental_candidates": [
                {"symbol": "CCC", "retrieval_reason": "sealed supplemental candidate"}
            ],
            "retrieval_evidence": [
                {
                    "symbol": symbol,
                    "branch": branch,
                    "supporting_fact_ids": [f"{symbol}-{branch}"],
                    "contradicting_fact_ids": [],
                    "conflict_note": retrieval_note,
                }
                for symbol in final_symbols
                for branch in FORMAL_BRANCHES[:3]
            ],
            "llm_verdicts": [
                {
                    "symbol": symbol,
                    "raw_score": scores[symbol],
                    "confidence": 0.8,
                    "supporting_fact_ids": [f"{symbol}-llm"],
                    "contradicting_fact_ids": [],
                    "rationale": "sealed fourth-branch verdict",
                }
                for symbol in final_symbols
            ],
        },
        digest_field="response_sha256",
    )
    return Stage1ReviewBinding(
        request=_review_artifact(
            "stage1-request.json",
            request,
            digest_field="request_sha256",
        ),
        response=_review_artifact(
            f"stage1-response-{hashlib.sha256(retrieval_note.encode()).hexdigest()[:8]}.json",
            response,
            digest_field="response_sha256",
        ),
    )


def _model_bundle_artifacts() -> tuple[tuple[str, BoundCanonicalArtifact], ...]:
    llm_identity = LLMProviderBuildIdentity(
        provider_id="openai",
        model_id="gpt-v16-review-build-001",
        immutable_model_build_id="gpt-v16-review-build-001-20260718",
        endpoint_contract_id="responses-v1",
        tokenizer_ref=_ref("llm-tokenizer.bin", schema="v16.tokenizer.v2"),
        inference_config_ref=_ref("llm-inference.json", schema="v16.inference-config.v2"),
        provider_attestation_ref=_ref(
            "llm-provider-attestation.json",
            schema="v16.provider-attestation.v2",
        ),
    )
    artifacts: list[tuple[str, BoundCanonicalArtifact]] = []
    for branch in FORMAL_BRANCHES:
        payload = build_frozen_model_bundle(
            protocol_attempt_id=ATTEMPT_ID,
            branch=branch,
            bundle_id=f"model-{branch}-001",
            training_schedule_ref=_ref(
                f"model-{branch}-schedule.json",
                schema="v16.training-schedule.v2",
            ),
            training_capture_ref=_ref(
                f"model-{branch}-capture.json",
                schema="v16.training-capture.v2",
            ),
            feature_contract_ref=_ref(
                f"model-{branch}-features.json",
                schema="v16.feature-contract.v2",
            ),
            hyperparameter_ref=_ref(
                f"model-{branch}-hyperparameters.json",
                schema="v16.hyperparameters.v2",
            ),
            serialized_model_ref=_ref(
                f"model-{branch}-serialized.bin",
                schema="v16.serialized-model.v2",
            ),
            deterministic_inference_entrypoint=(
                "quant_investor.v16.evidence_v2.posterior:replay_stage1_formal_evidence"
            ),
            llm_provider_build=llm_identity if branch == "llm" else None,
        )
        artifacts.append((branch, _bound(f"model-{branch}.json", payload)))
    return tuple(artifacts)


def _runtime_artifacts() -> PosteriorRuntimeArtifacts:
    sample_ids = [f"sample-{index:03d}" for index in range(300)]
    base = _bound(
        "prior-training.json",
        build_base_rate_training_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            receipt_id="prior-training",
            training_start="2021-07-18",
            training_end="2026-07-18",
            embargo_days=20,
            observations=[
                BaseRateObservation(
                    sample_id=sample_id,
                    positive_outcome=index >= 150,
                )
                for index, sample_id in enumerate(sample_ids)
            ],
            source_input_refs=[_ref("prior-source.parquet", schema="v16.training-source.v2")],
        ),
    )
    likelihood = _bound(
        "likelihood-training.json",
        build_likelihood_training_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            receipt_id="likelihood-training",
            training_start="2021-07-18",
            training_end="2026-07-18",
            embargo_days=20,
            observations=[
                LikelihoodTrainingObservation(
                    sample_id=sample_id,
                    branch=branch,
                    cohort_id=f"cohort-{index // 30:02d}",
                    score=-1.0 + 2.0 * index / 299.0,
                    positive_outcome=index >= 150,
                )
                for branch in FORMAL_BRANCHES
                for index, sample_id in enumerate(sample_ids)
            ],
            source_input_refs=[_ref("likelihood-source.parquet", schema="v16.training-source.v2")],
        ),
    )
    manifest_kwargs = {
        "protocol_attempt_id": ATTEMPT_ID,
        "training_start": "2021-07-18",
        "training_end": "2026-07-18",
        "embargo_days": 20,
        "sample_ids": sample_ids,
    }
    return_training = _bound(
        "return-training.json",
        build_return_model_training_evidence(
            **manifest_kwargs,
            receipt_id="return-training",
            source_input_refs=[_ref("return-source.parquet", schema="v16.training-source.v2")],
        ),
    )
    return_parameters = _bound(
        "return-parameters.json",
        build_return_model_parameters(
            protocol_attempt_id=ATTEMPT_ID,
            artifact_id="return-model-001",
            training_ref=return_training.reference,
            intercept=0.0,
            aggregate_coefficient=0.04,
        ),
    )
    bootstrap_training = _bound(
        "bootstrap-training.json",
        build_bootstrap_training_evidence(
            **manifest_kwargs,
            receipt_id="bootstrap-training",
            source_input_refs=[_ref("bootstrap-source.parquet", schema="v16.training-source.v2")],
        ),
    )
    bootstrap_offsets = _bound(
        "bootstrap-offsets.json",
        build_bootstrap_offsets(
            protocol_attempt_id=ATTEMPT_ID,
            artifact_id="bootstrap-001",
            training_ref=bootstrap_training.reference,
            block_length_days=20,
            block_count=60,
            win_rate_logit_offsets=(0.0,) * 1000,
            expected_alpha_offsets=(0.0,) * 1000,
        ),
    )
    correlation_training = _bound(
        "correlation-training.json",
        build_correlation_training_evidence(
            **manifest_kwargs,
            receipt_id="correlation-training",
            source_input_refs=[_ref("correlation-source.parquet", schema="v16.training-source.v2")],
        ),
    )
    correlation_matrix = _bound(
        "correlation-matrix.json",
        build_correlation_matrix(
            protocol_attempt_id=ATTEMPT_ID,
            training_ref=correlation_training.reference,
            correlations={key: 0.0 for key in CANONICAL_CORRELATION_KEYS},
        ),
    )
    return PosteriorRuntimeArtifacts(
        model_bundles=_model_bundle_artifacts(),
        prior_training=base,
        likelihood_training=likelihood,
        return_model_parameters=return_parameters,
        return_model_training=return_training,
        bootstrap_offsets=bootstrap_offsets,
        bootstrap_training=bootstrap_training,
        correlation_matrix=correlation_matrix,
        correlation_training=correlation_training,
    )


def _formal_artifacts(
    *,
    stage1: Stage1ReviewBinding,
    runtime: PosteriorRuntimeBundle,
) -> list[BoundCanonicalArtifact]:
    artifacts: list[BoundCanonicalArtifact] = []
    for record in replay_stage1_formal_evidence(stage1):
        payload = build_formal_branch_prediction(
            protocol_attempt_id=ATTEMPT_ID,
            symbol=record.symbol,
            branch=record.branch,
            raw_score=record.raw_score,
            confidence=record.confidence,
            evidence_ids=record.evidence_ids,
            stage1_request_ref=stage1.request.reference,
            stage1_response_ref=stage1.response.reference,
            model_bundle_ref=runtime.model_refs[record.branch],
            source_input_refs=(
                [stage1.request.reference, stage1.response.reference]
                if record.branch == "llm"
                else [stage1.request.reference]
            ),
        )
        artifacts.append(_bound(f"formal-{record.symbol}-{record.branch}.json", payload))
    return artifacts


def _cost_model() -> BoundCanonicalArtifact:
    return _bound(
        "posterior-cost-model.json",
        build_posterior_cost_model(
            protocol_attempt_id=ATTEMPT_ID,
            model_id="cn-round-trip-cost-001",
            costs=CostVector(
                (
                    0.0001,
                    0.0001,
                    0.0005,
                    0.00001,
                    0.00001,
                    0.0002,
                    0.0002,
                    0.0001,
                )
            ),
            source_input_refs=[_ref("cost-policy.json", schema="v16.cost-policy.v2")],
        ),
    )


def _cost_artifacts(
    *,
    stage1: Stage1ReviewBinding,
    cost_model: BoundCanonicalArtifact,
) -> list[BoundCanonicalArtifact]:
    return [
        _bound(
            f"cost-{symbol}.json",
            build_posterior_cost_input(
                protocol_attempt_id=ATTEMPT_ID,
                symbol=symbol,
                cost_model=cost_model,
                stage1_request_ref=stage1.request.reference,
            ),
        )
        for symbol in ("AAA", "BBB", "CCC")
    ]


def _inputs(*, retrieval_note: str = "") -> dict[str, Any]:
    stage1 = _stage1_binding(retrieval_note=retrieval_note)
    runtime_artifacts = _runtime_artifacts()
    runtime = PosteriorRuntimeBundle(artifacts=runtime_artifacts)
    cost_model = _cost_model()
    return {
        "stage1": stage1,
        "runtime": runtime,
        "runtime_artifacts": runtime_artifacts,
        "cost_model": cost_model,
        "formal": _formal_artifacts(stage1=stage1, runtime=runtime),
        "costs": _cost_artifacts(stage1=stage1, cost_model=cost_model),
    }


def _build(inputs: dict[str, Any]) -> dict[str, Any]:
    return build_full_union_posterior_evidence(
        protocol_attempt_id=ATTEMPT_ID,
        stage1_binding=inputs["stage1"],
        runtime_artifacts=inputs["runtime_artifacts"],
        cost_model=inputs["cost_model"],
        formal_branch_artifacts=inputs["formal"],
        cost_artifacts=inputs["costs"],
    )


def test_full_union_posterior_is_recomputed_before_top_50_menu() -> None:
    inputs = _inputs()
    evidence = _build(inputs)
    validated = validate_full_union_posterior_evidence(
        evidence,
        stage1_binding=inputs["stage1"],
        runtime_artifacts=inputs["runtime_artifacts"],
        cost_model=inputs["cost_model"],
        formal_branch_artifacts=inputs["formal"],
        cost_artifacts=inputs["costs"],
    )

    assert validated["candidate_symbols"] == ["AAA", "BBB", "CCC"]
    assert validated["ranked_symbols"] == ["AAA", "CCC", "BBB"]
    assert validated["menu_symbols"] == validated["ranked_symbols"]
    assert validated["branch_order"] == list(FORMAL_BRANCHES)
    assert validated["posteriors"] and len(validated["posteriors"]) == 3
    assert all(len(item["branch_evidence"]) == 4 for item in validated["posteriors"])
    assert validated["retrieval_used_in_scoring"] is False
    assert validated["risk_advisor_used_in_scoring"] is False
    assert validated["new_risk_authorized"] is False


def test_full_union_posterior_rejects_missing_branch_or_cost() -> None:
    inputs = _inputs()
    with pytest.raises(EvidenceV2Error, match="do not cover the exact full union"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"][:-1],
            cost_artifacts=inputs["costs"],
        )
    with pytest.raises(EvidenceV2Error, match="do not cover the exact full union"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"][:-1],
        )


def test_full_union_posterior_rejects_caller_metric_injection() -> None:
    inputs = _inputs()
    evidence = _build(inputs)
    tampered = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    tampered["posteriors"] = [dict(item) for item in tampered["posteriors"]]
    tampered["posteriors"][0]["posterior_win_rate"] = encode_f64(0.99)

    with pytest.raises(EvidenceV2Error, match="deterministic recomputation"):
        validate_full_union_posterior_evidence(
            seal_semantic(tampered),
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )


@pytest.mark.parametrize(
    "field",
    (
        "retrieval_used_in_scoring",
        "risk_advisor_used_in_scoring",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
    ),
)
def test_full_union_posterior_rejects_authorizing_projection(field: str) -> None:
    inputs = _inputs()
    evidence = _build(inputs)
    changed = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    changed[field] = True
    with pytest.raises(EvidenceV2Error, match="must be nonauthorizing"):
        validate_full_union_posterior_evidence(
            seal_semantic(changed),
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )


@pytest.mark.parametrize("field", ("candidate_symbols", "ranked_symbols", "menu_symbols"))
def test_union_rank_and_disconnected_blockers_are_recomputed(field: str) -> None:
    inputs = _inputs()
    evidence = _build(inputs)
    changed = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    changed[field] = list(reversed(changed[field]))
    with pytest.raises(EvidenceV2Error, match="deterministic recomputation"):
        validate_full_union_posterior_evidence(
            seal_semantic(changed),
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )

    changed = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    changed["blockers"] = []
    with pytest.raises(EvidenceV2Error, match="deterministic recomputation"):
        validate_full_union_posterior_evidence(
            seal_semantic(changed),
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )


def test_llm_formal_score_must_equal_stage1_response() -> None:
    inputs = _inputs()
    original = inputs["formal"][3]
    payload = original.read()
    changed = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    changed["raw_score"] = encode_f64(0.9)
    inputs["formal"][3] = _bound("formal-AAA-llm-drift.json", seal_semantic(changed))

    with pytest.raises(EvidenceV2Error, match="drifts from deterministic Stage1 replay"):
        _build(inputs)


@pytest.mark.parametrize(
    ("artifact_index", "branch"),
    ((0, "quant"), (1, "fundamental"), (2, "macro")),
)
def test_qfm_formal_score_must_equal_deterministic_replay(
    artifact_index: int,
    branch: str,
) -> None:
    inputs = _inputs()
    original = inputs["formal"][artifact_index]
    payload = original.read()
    changed = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    score = decode_f64(changed["raw_score"], label="raw_score")
    changed["raw_score"] = encode_f64(score - 0.01 if score > -0.99 else score + 0.01)
    inputs["formal"][artifact_index] = _bound(
        f"formal-AAA-{branch}-forged.json",
        seal_semantic(changed),
    )

    with pytest.raises(
        EvidenceV2Error,
        match=rf"{branch} formal evidence drifts from deterministic Stage1 replay",
    ):
        _build(inputs)


def test_formal_evidence_id_and_model_lineage_cannot_be_forged() -> None:
    inputs = _inputs()
    original = inputs["formal"][0]
    payload = original.read()
    changed = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    changed["evidence_ids"] = ["quant:forged"]
    inputs["formal"][0] = _bound("formal-AAA-quant-id-forged.json", seal_semantic(changed))
    with pytest.raises(EvidenceV2Error, match="drifts from deterministic Stage1 replay"):
        _build(inputs)

    inputs = _inputs()
    payload = inputs["formal"][0].read()
    changed = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    changed["model_bundle_ref"] = _ref(
        "forged-quant-model.json",
        schema=MODEL_BUNDLE_SCHEMA,
    ).to_dict()
    inputs["formal"][0] = _bound(
        "formal-AAA-quant-model-forged.json",
        seal_semantic(changed),
    )
    with pytest.raises(EvidenceV2Error, match="formal branch prediction lineage mismatch"):
        _build(inputs)


def test_cost_input_is_rebuilt_from_exact_eight_component_model() -> None:
    inputs = _inputs()
    cost_payload = inputs["cost_model"].read()
    assert len(cost_payload["costs"]) == 8
    posterior_cost = inputs["costs"][0].read()
    assert posterior_cost["costs"] == cost_payload["costs"]

    changed = {key: value for key, value in posterior_cost.items() if key != "semantic_sha256"}
    changed["fee"] = encode_f64(0.5)
    inputs["costs"][0] = _bound("cost-AAA-forged.json", seal_semantic(changed))
    with pytest.raises(EvidenceV2Error, match="posterior cost input is not canonical"):
        _build(inputs)


def test_runtime_attempt_and_frozen_llm_model_must_match_stage1() -> None:
    inputs = _inputs()
    with pytest.raises(EvidenceV2Error, match="runtime crosses protocol attempts"):
        build_full_union_posterior_evidence(
            protocol_attempt_id="attempt-v16-posterior-other",
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )

    artifacts = inputs["runtime_artifacts"]
    bundles = list(artifacts.model_bundles)
    llm = bundles[-1][1].read()
    changed = {key: value for key, value in llm.items() if key != "semantic_sha256"}
    changed["llm_provider_build"] = dict(changed["llm_provider_build"])
    changed["llm_provider_build"]["model_id"] = "different-stage1-model"
    bundles[-1] = ("llm", _bound("model-llm-drift.json", seal_semantic(changed)))
    inputs["runtime_artifacts"] = replace(artifacts, model_bundles=tuple(bundles))
    with pytest.raises(EvidenceV2Error, match="differs from frozen LLM bundle"):
        _build(inputs)


def test_retrieval_annotations_cannot_change_posterior_numbers() -> None:
    baseline = _build(_inputs())
    annotated = _build(_inputs(retrieval_note="conflicting retrieval note"))
    numeric_fields = (
        "prior_probability",
        "posterior_win_rate",
        "posterior_expected_alpha",
        "posterior_edge_after_costs",
        "raw_evidence_increment",
        "correlation_adjusted_evidence_increment",
        "correlation_vif",
        "correlation_vif_shrink",
        "rank",
    )
    for left, right in zip(baseline["posteriors"], annotated["posteriors"]):
        assert {field: left[field] for field in numeric_fields} == {
            field: right[field] for field in numeric_fields
        }
    assert baseline["ranked_symbols"] == annotated["ranked_symbols"]


def test_posterior_producer_has_no_risk_or_portfolio_input() -> None:
    parameters = inspect.signature(build_full_union_posterior_evidence).parameters
    assert "risk_advisor" not in parameters
    assert "existing_weights" not in parameters
    assert "target_weights" not in parameters
    assert "posterior_items" not in parameters
    assert "runtime" not in parameters
    assert "runtime_artifacts" in parameters


def test_runtime_receipts_must_bind_exact_training_bytes() -> None:
    artifacts = _runtime_artifacts()
    original = artifacts.prior_training
    changed = {key: value for key, value in original.read().items() if key != "semantic_sha256"}
    changed["observations"] = [dict(item) for item in changed["observations"]]
    changed["observations"][0]["positive_outcome"] = not changed["observations"][0][
        "positive_outcome"
    ]
    forged = BoundCanonicalArtifact(
        reference=original.reference,
        payload=canonical_json_bytes(seal_semantic(changed)),
    )
    with pytest.raises(EvidenceV2Error, match="byte SHA mismatch"):
        PosteriorRuntimeBundle(
            artifacts=replace(artifacts, prior_training=forged),
        )


def test_fake_artifact_objects_cannot_cross_posterior_boundaries() -> None:
    inputs = _inputs()
    fake_cost_model = SimpleNamespace(
        reference=inputs["cost_model"].reference,
        read=inputs["cost_model"].read,
    )
    with pytest.raises(EvidenceV2Error, match="actual BoundCanonicalArtifact"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=fake_cost_model,
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )

    fake_formal = SimpleNamespace(
        reference=inputs["formal"][0].reference,
        read=inputs["formal"][0].read,
    )
    with pytest.raises(EvidenceV2Error, match="actual BoundCanonicalArtifact"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=[fake_formal, *inputs["formal"][1:]],
            cost_artifacts=inputs["costs"],
        )

    fake_cost = SimpleNamespace(
        reference=inputs["costs"][0].reference,
        read=inputs["costs"][0].read,
    )
    with pytest.raises(EvidenceV2Error, match="actual BoundCanonicalArtifact"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=[fake_cost, *inputs["costs"][1:]],
        )

    fake_runtime = SimpleNamespace(**inputs["runtime_artifacts"].__dict__)
    with pytest.raises(EvidenceV2Error, match="actual runtime artifacts"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=inputs["stage1"],
            runtime_artifacts=fake_runtime,
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )

    fake_stage1 = SimpleNamespace(
        request=inputs["stage1"].request,
        response=inputs["stage1"].response,
        read=inputs["stage1"].read,
    )
    with pytest.raises(EvidenceV2Error, match="actual Stage1ReviewBinding"):
        build_full_union_posterior_evidence(
            protocol_attempt_id=ATTEMPT_ID,
            stage1_binding=fake_stage1,
            runtime_artifacts=inputs["runtime_artifacts"],
            cost_model=inputs["cost_model"],
            formal_branch_artifacts=inputs["formal"],
            cost_artifacts=inputs["costs"],
        )


def test_artifact_schemas_are_not_legacy_menu_inputs() -> None:
    assert FORMAL_BRANCH_PREDICTION_SCHEMA == "v16.formal-branch-prediction.v2"
    assert POSTERIOR_COST_INPUT_SCHEMA == "v16.posterior-cost-input.v2"
    assert POSTERIOR_COST_MODEL_SCHEMA == "v16.posterior-cost-model.v2"
