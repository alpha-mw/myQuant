from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import inspect

import pytest

from quant_investor.v16.evidence_v2.calibration_plan_v3 import (
    BRANCH_STATUS_SCHEMA,
    CALIBRATION_PLAN_V3_SCHEMA,
    COST_STATUS_SCHEMA,
    FORMAL_BRANCHES,
    LAMBDA_STATUS_SCHEMA,
    POSTERIOR_RUNTIME_SCHEMAS,
    RESOLVER_IMPLEMENTATION_SCHEMA,
    SAMPLE_ARTIFACT_SCHEMAS,
    STOCK_SOURCE_SET_SCHEMA,
    SUPPORTED_REQUIREMENTS,
    TARGET_STATUS_SCHEMA,
    CalibrationSamplePlanV3,
    LambdaFoldPlanV3,
    PlannedArtifactPath,
    build_calibration_universe_plan_v3,
    validate_calibration_universe_plan_v3,
)
from quant_investor.v16.evidence_v2.contracts import (
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
)


ROOT = "/private/v16-calibration-v3"


def _ref(name: str, schema: str) -> EvidenceRef:
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=f"{ROOT}/sources/{name}",
        byte_sha256=hashlib.sha256(f"{name}:bytes".encode()).hexdigest(),
        semantic_sha256=hashlib.sha256(f"{name}:semantic".encode()).hexdigest(),
        root_policy="v16.private-evidence-root.v2",
    )


def _planned(path: str, schema: str) -> PlannedArtifactPath:
    return PlannedArtifactPath(absolute_path=f"{ROOT}/future/{path}", artifact_schema=schema)


def _sample(branch: str, index: int) -> CalibrationSamplePlanV3:
    cohort = index // 38
    start = date(2026, 1, 5) + timedelta(days=cohort * 30)
    end = start + timedelta(days=19)
    sample_id = f"{branch}-sample-{index:03d}"
    artifacts = tuple(
        (
            key,
            _planned(
                f"{branch}/{sample_id}/{key}.bin",
                schema,
            ),
        )
        for key, schema in SAMPLE_ARTIFACT_SCHEMAS.items()
    )
    return CalibrationSamplePlanV3(
        sample_id=sample_id,
        branch=branch,
        symbol=f"S{index:06d}",
        cohort_id=f"cohort-{cohort:02d}",
        cohort_start_date=start.isoformat(),
        cohort_end_date=end.isoformat(),
        slot_id=f"slot-{index:03d}",
        artifacts=artifacts,
        stock_source_set_ref=_ref("stock-source-set.json", STOCK_SOURCE_SET_SCHEMA),
        benchmark_manifest_ref=_ref(
            "h00300-manifest.json",
            "csi-index-total-return-manifest.v1",
        ),
    )


def _build() -> dict:
    resolver_refs = {
        requirement: _ref(f"resolver-{requirement}.json", RESOLVER_IMPLEMENTATION_SCHEMA)
        for requirement in SUPPORTED_REQUIREMENTS
    }
    model_refs = {
        branch: _ref(f"model-{branch}.json", "v16.frozen-model-bundle.v2")
        for branch in FORMAL_BRANCHES
    }
    runtime_refs = {
        key: _ref(f"runtime-{key}.json", schema)
        for key, schema in POSTERIOR_RUNTIME_SCHEMAS.items()
    }
    samples = [
        _sample(branch, index)
        for branch in FORMAL_BRANCHES
        for index in range(304)
    ]
    folds = [
        LambdaFoldPlanV3(
            branch=branch,
            fold_id=f"fold-{index}",
            training_source_refs=(
                _ref(f"lambda-{branch}-{index}-training.parquet", "lambda-training.v1"),
            ),
            status_artifact=_planned(
                f"lambda/{branch}/fold-{index}.json",
                LAMBDA_STATUS_SCHEMA,
            ),
        )
        for branch in FORMAL_BRANCHES
        for index in range(2)
    ]
    return build_calibration_universe_plan_v3(
        protocol_attempt_id="attempt-v16-calibration-001",
        epoch="B",
        schedule_id="schedule-b",
        private_root=ROOT,
        runtime_capsule_ref=_ref("runtime-capsule.json", "v16.hermetic-runtime-capsule.v2"),
        resolver_implementation_refs=resolver_refs,
        model_bundle_refs=model_refs,
        posterior_runtime_refs=runtime_refs,
        sample_plans=samples,
        lambda_fold_plans=folds,
    )


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {
        "complete",
        "ready",
        "all_gates_passed",
        "metrics",
        "predicted_alpha",
        "interval_lower",
        "interval_upper",
        "lambda_value",
        "costs",
        "realized_alpha",
        "outcome",
    }
    if isinstance(value, dict):
        return bool(forbidden.intersection(value)) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def test_plan_v3_predeclares_exact_four_branch_shape_without_numeric_outputs() -> None:
    plan = _build()
    validated = validate_calibration_universe_plan_v3(plan)

    assert validated["schema_version"] == CALIBRATION_PLAN_V3_SCHEMA
    assert len(validated["sample_plans"]) == 304 * 4
    assert validated["unsupported_requirement_ids"] == [
        "branch_only_alpha_interval_model",
        "fold_training_algorithm",
        "eight_component_cost_model",
    ]
    assert _contains_forbidden_key(validated) is False
    assert {
        item["artifacts"]["branch_status"]["artifact_schema"]
        for item in validated["sample_plans"]
    } == {BRANCH_STATUS_SCHEMA}
    assert {
        item["artifacts"]["cost_status"]["artifact_schema"]
        for item in validated["sample_plans"]
    } == {COST_STATUS_SCHEMA}
    assert {
        item["artifacts"]["target_status"]["artifact_schema"]
        for item in validated["sample_plans"]
    } == {TARGET_STATUS_SCHEMA}


def test_plan_v3_rejects_resealed_path_substitution() -> None:
    plan = _build()
    tampered = deepcopy(plan)
    tampered.pop("semantic_sha256")
    tampered["sample_plans"][1]["artifacts"]["branch_status"] = deepcopy(
        tampered["sample_plans"][0]["artifacts"]["branch_status"]
    )

    with pytest.raises(EvidenceV2Error, match="globally unique"):
        validate_calibration_universe_plan_v3(seal_semantic(tampered))


def test_plan_v3_public_builder_has_no_prediction_or_completion_inputs() -> None:
    names = set(inspect.signature(build_calibration_universe_plan_v3).parameters)
    assert not names.intersection(
        {
            "probability",
            "prior_probability",
            "predicted_alpha",
            "interval_lower",
            "interval_upper",
            "lambda_value",
            "costs",
            "complete",
            "ready",
        }
    )


def test_planned_artifact_path_rejects_private_root_escape() -> None:
    with pytest.raises(EvidenceV2Error, match="strict private-root child"):
        PlannedArtifactPath(
            absolute_path="/private/outside/status.json",
            artifact_schema=BRANCH_STATUS_SCHEMA,
        ).validate_under(ROOT)
