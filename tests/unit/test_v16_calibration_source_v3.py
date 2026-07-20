from __future__ import annotations

import ast
import builtins
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import hashlib
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import quant_investor.v16.evidence_v2.calibration_plan_v3 as plan_module
import quant_investor.v16.evidence_v2.calibration_source_v3 as source_module
from quant_investor.v16.evidence_v2.calibration_plan_v3 import (
    FORMAL_BRANCHES,
    LAMBDA_STATUS_SCHEMA,
    POSTERIOR_RUNTIME_SCHEMAS,
    SAMPLE_ARTIFACT_SCHEMAS,
    STOCK_SOURCE_SET_SCHEMA,
    SUPPORTED_REQUIREMENTS,
    TARGET_STATUS_SCHEMA,
    CalibrationPlanEvidenceBundleV3,
    CalibrationSamplePlanV3,
    LambdaFoldPlanV3,
    PlannedArtifactPath,
    build_calibration_universe_plan_v3,
)
from quant_investor.v16.evidence_v2.calibration_source_v3 import (
    ALPHA_INTERVAL_REQUIREMENT,
    FOLD_TRAINING_REQUIREMENT,
    RESOLVER_ENTRYPOINTS,
    RESOLVER_MODULE_SOURCE_SCHEMA,
    BranchPredictionSourceEvidenceBundleV3,
    BranchPredictionStatusBindingV3,
    CalibrationSourceStatusEvidenceBundleV3,
    LambdaFoldSourceEvidenceBundleV3,
    ResolverImplementationEvidenceBundleV3,
    RuntimeTrainingSourceBundleV3,
    build_branch_prediction_source_status_v3,
    build_calibration_source_status_v3,
    build_lambda_fold_source_status_v3,
    build_resolver_implementation_v3,
    validate_branch_prediction_source_status_v3,
    validate_calibration_source_status_v3,
    validate_lambda_fold_source_status_v3,
)
from quant_investor.v16.evidence_v2.contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    decode_f64,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.posterior import (
    BoundReviewArtifact,
    Stage1ReviewBinding,
)
from quant_investor.v16.evidence_v2.runtime_identity import (
    REQUIRED_ENVIRONMENT_CONTROLS,
    build_runtime_capsule,
)
from quant_investor.v16.evidence_v2.schedule_v4 import ScheduleAnchorBindingV4
from quant_investor.v16.evidence_v2.target_v4 import (
    TargetSourceEvidenceBundleV4,
    build_cost_source_status_v3,
)
from tests.unit.test_v16_evidence_v2_metrics_runtime_timestamp import (
    _runtime_components,
)
from tests.unit.test_v16_evidence_v2_posterior import (
    ATTEMPT_ID,
    BASE_TIME,
    _bound,
    _ref,
    _runtime_artifacts,
    _stage1_binding,
)


ROOT = "/private/evidence"
SLOT_ID = "calibration-slot-0"


def _bound_at(path: str, payload: dict[str, Any]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=str(payload["schema_version"]),
            absolute_path=path,
            byte_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_sha256=semantic_sha256(payload),
            root_policy="v16.private-evidence-root.v2",
        ),
        payload=raw,
    )


def _planned(path: str, schema: str) -> PlannedArtifactPath:
    return PlannedArtifactPath(
        absolute_path=f"{ROOT}/{path}",
        artifact_schema=schema,
    )


def _clone_stage1(branch: str) -> Stage1ReviewBinding:
    original = _stage1_binding()
    request = BoundReviewArtifact(
        reference=replace(
            original.request.reference,
            absolute_path=f"{ROOT}/stage1-request-{branch}.json",
        ),
        payload=original.request.payload,
    )
    response = BoundReviewArtifact(
        reference=replace(
            original.response.reference,
            absolute_path=f"{ROOT}/stage1-response-{branch}.json",
        ),
        payload=original.response.payload,
    )
    return Stage1ReviewBinding(request=request, response=response)


def _raw_for_ref(reference: EvidenceRef) -> BoundRawArtifact:
    name = reference.absolute_path.rsplit("/", 1)[-1]
    return BoundRawArtifact(reference=reference, payload=f"{name}:bytes".encode())


def _timestamp_binding(
    *,
    status: BoundCanonicalArtifact,
    sample: dict[str, Any],
    schedule: dict[str, Any],
) -> SimpleNamespace:
    artifacts = sample["artifacts"]
    attempt_ref = _ref(
        f"timestamp-attempt-{sample['sample_id']}.json",
        schema=artifacts["prediction_timestamp_attempt"]["artifact_schema"],
    )
    attempt_ref = replace(
        attempt_ref,
        absolute_path=artifacts["prediction_timestamp_attempt"]["absolute_path"],
    )
    receipt_ref = _ref(
        f"timestamp-receipt-{sample['sample_id']}.json",
        schema=artifacts["prediction_timestamp_receipt"]["artifact_schema"],
    )
    receipt_ref = replace(
        receipt_ref,
        absolute_path=artifacts["prediction_timestamp_receipt"]["absolute_path"],
    )
    slot = schedule["slots"][0]
    attempt = {"protocol_attempt_id": ATTEMPT_ID}
    receipt = {
        "anchored_artifact_ref": status.reference.to_dict(),
        "anchor_kind": "prediction",
        "anchor_not_before": slot["s0_close_at"],
        "anchor_not_after": slot["s1_open_at"],
    }
    return SimpleNamespace(
        attempt=SimpleNamespace(reference=attempt_ref),
        validation_receipt=SimpleNamespace(reference=receipt_ref),
        read=lambda: (attempt, receipt),
    )


@pytest.fixture
def source_inputs(monkeypatch) -> dict[str, Any]:
    monkeypatch.setattr(plan_module, "MIN_BRANCH_SAMPLES", 1)
    monkeypatch.setattr(plan_module, "MIN_BRANCH_COHORTS", 1)

    runtime_artifacts = _runtime_artifacts()
    runtime_by_key = {
        key: getattr(runtime_artifacts, key) for key in POSTERIOR_RUNTIME_SCHEMAS
    }
    components = _runtime_components()
    runtime_capsule = _bound(
        "calibration-runtime-capsule.json",
        build_runtime_capsule(
            protocol_attempt_id=ATTEMPT_ID,
            capsule_id="calibration-runtime-capsule",
            components=components,
            environment_controls=REQUIRED_ENVIRONMENT_CONTROLS,
        ),
    )
    source_tree_component = next(
        item for item in components if item.component_id == "source_tree"
    )
    source_tree = BoundRawArtifact(
        reference=source_tree_component.artifact_ref,
        payload=b"runtime-source_tree.bin:bytes",
    )
    module_payload = b"calibration_source_v3.py:bytes"
    module_ref = _ref(
        "calibration_source_v3.py",
        schema=RESOLVER_MODULE_SOURCE_SCHEMA,
        payload=module_payload,
    )
    module_source = BoundRawArtifact(module_ref, module_payload)

    resolver_bundles: list[ResolverImplementationEvidenceBundleV3] = []
    for requirement in SUPPORTED_REQUIREMENTS:
        manifest = _bound(
            f"resolver-{requirement}.json",
            build_resolver_implementation_v3(
                protocol_attempt_id=ATTEMPT_ID,
                requirement_id=requirement,
                resolver_id=f"resolver-{requirement}-v3",
                entrypoint=RESOLVER_ENTRYPOINTS[requirement],
                module_source_ref=module_ref,
                runtime_capsule_ref=runtime_capsule.reference,
                source_tree_ref=source_tree.reference,
            ),
        )
        resolver_bundles.append(
            ResolverImplementationEvidenceBundleV3(
                manifest=manifest,
                module_source=module_source,
                runtime_capsule=runtime_capsule,
                source_tree=source_tree,
            )
        )

    stage1_by_branch = {
        branch: _clone_stage1(branch) for branch in FORMAL_BRANCHES
    }
    start = date(2026, 7, 19)
    end = start + timedelta(days=19)
    samples: list[CalibrationSamplePlanV3] = []
    for branch in FORMAL_BRANCHES:
        stage1 = stage1_by_branch[branch]
        sample_id = f"{branch}-sample"
        artifacts = []
        for key, schema in SAMPLE_ARTIFACT_SCHEMAS.items():
            if key == "stage1_request":
                path = stage1.request.reference.absolute_path
            elif key == "stage1_response":
                path = stage1.response.reference.absolute_path
            else:
                path = f"{ROOT}/{key}-{sample_id}.json"
            artifacts.append(
                (
                    key,
                    PlannedArtifactPath(
                        absolute_path=path,
                        artifact_schema=schema,
                    ),
                )
            )
        samples.append(
            CalibrationSamplePlanV3(
                sample_id=sample_id,
                branch=branch,
                symbol="AAA",
                cohort_id="cohort-0",
                cohort_start_date=start.isoformat(),
                cohort_end_date=end.isoformat(),
                slot_id=SLOT_ID,
                artifacts=tuple(artifacts),
                stock_source_set_ref=_ref(
                    "stock-source-set-source-v3.json",
                    schema=STOCK_SOURCE_SET_SCHEMA,
                ),
                benchmark_manifest_ref=_ref(
                    "h00300-manifest-source-v3.json",
                    schema="csi-index-total-return-manifest.v1",
                ),
            )
        )

    lambda_sources: dict[tuple[str, str], BoundRawArtifact] = {}
    folds: list[LambdaFoldPlanV3] = []
    for branch in FORMAL_BRANCHES:
        for index in range(2):
            fold_id = f"fold-{index}"
            source_ref = _ref(
                f"lambda-{branch}-{fold_id}.bin",
                schema="v16.lambda-training-source.v3",
            )
            lambda_sources[(branch, fold_id)] = _raw_for_ref(source_ref)
            folds.append(
                LambdaFoldPlanV3(
                    branch=branch,
                    fold_id=fold_id,
                    training_source_refs=(source_ref,),
                    status_artifact=_planned(
                        f"lambda-status-{branch}-{fold_id}.json",
                        LAMBDA_STATUS_SCHEMA,
                    ),
                )
            )

    model_refs = {
        branch: artifact.reference
        for branch, artifact in runtime_artifacts.model_bundles
    }
    plan = _bound(
        "calibration-plan-source-v3.json",
        build_calibration_universe_plan_v3(
            protocol_attempt_id=ATTEMPT_ID,
            epoch="B",
            schedule_id="schedule-b",
            private_root=ROOT,
            runtime_capsule_ref=runtime_capsule.reference,
            resolver_implementation_refs={
                requirement: bundle.manifest.reference
                for requirement, bundle in zip(
                    SUPPORTED_REQUIREMENTS,
                    resolver_bundles,
                )
            },
            model_bundle_refs=model_refs,
            posterior_runtime_refs={
                key: runtime_by_key[key].reference for key in POSTERIOR_RUNTIME_SCHEMAS
            },
            sample_plans=samples,
            lambda_fold_plans=folds,
        ),
    )
    plan_evidence = CalibrationPlanEvidenceBundleV3(
        plan=plan,
        resolver_implementations=tuple(
            bundle.manifest for bundle in resolver_bundles
        ),
        posterior_runtime_artifacts=tuple(
            runtime_by_key[key] for key in POSTERIOR_RUNTIME_SCHEMAS
        ),
    )

    model_bundles = tuple(
        SimpleNamespace(
            model_bundle=artifact,
            read=artifact.read,
        )
        for _branch, artifact in runtime_artifacts.model_bundles
    )
    schedule_ref = _ref(
        "schedule-source-v4.json",
        schema="v16.evidence-schedule-declaration.v4",
    )
    schedule = {
        "protocol_attempt_id": ATTEMPT_ID,
        "epoch": "B",
        "schedule_id": "schedule-b",
        "calibration_plan_ref": plan.reference.to_dict(),
        "open_session_calendar": _ref(
            "calendar-source-v3.json",
            schema="v16.open-session-calendar.v2",
        ).to_dict(),
        "slots": [
            {
                "slot_id": SLOT_ID,
                "s0_close_at": "2026-07-18T07:00:00Z",
                "decision_cutoff_at": BASE_TIME.isoformat().replace(
                    "+00:00",
                    "Z",
                ),
                "s1_open_at": "2026-07-18T09:00:00Z",
                "target_sessions": [
                    (start + timedelta(days=index)).isoformat()
                    for index in range(20)
                ],
            }
        ],
    }
    schedule_anchor = ScheduleAnchorBindingV4(
        evidence=SimpleNamespace(
            schedule=SimpleNamespace(reference=schedule_ref),
            model_bundles=model_bundles,
        ),
        timestamp=SimpleNamespace(),
    )
    monkeypatch.setattr(
        source_module,
        "validate_schedule_anchor_binding_v4",
        lambda _binding: schedule,
    )

    training_sources: list[
        tuple[str, tuple[BoundCanonicalArtifact | BoundRawArtifact, ...]]
    ] = []
    for key in POSTERIOR_RUNTIME_SCHEMAS:
        payload = runtime_by_key[key].read()
        training_sources.append(
            (
                key,
                tuple(
                    _raw_for_ref(EvidenceRef.from_dict(item))
                    for item in payload.get("source_input_refs", [])
                ),
            )
        )
    runtime_training_sources = RuntimeTrainingSourceBundleV3(
        source_artifacts_by_key=tuple(training_sources)
    )

    branch_sources = {
        branch: BranchPredictionSourceEvidenceBundleV3(
            plan=plan_evidence,
            schedule_anchor=schedule_anchor,
            stage1_binding=stage1_by_branch[branch],
            runtime_artifacts=runtime_artifacts,
            runtime_training_sources=runtime_training_sources,
            resolver_implementations=tuple(resolver_bundles),
            sample_id=f"{branch}-sample",
        )
        for branch in FORMAL_BRANCHES
    }
    return {
        "plan": plan,
        "plan_evidence": plan_evidence,
        "schedule": schedule,
        "schedule_anchor": schedule_anchor,
        "branch_sources": branch_sources,
        "lambda_sources": lambda_sources,
    }


def _branch_binding(
    *,
    source: BranchPredictionSourceEvidenceBundleV3,
    plan: dict[str, Any],
    schedule: dict[str, Any],
) -> BranchPredictionStatusBindingV3:
    payload = build_branch_prediction_source_status_v3(evidence=source)
    sample = next(
        item for item in plan["sample_plans"] if item["sample_id"] == payload["sample_id"]
    )
    status = _bound_at(
        sample["artifacts"]["branch_status"]["absolute_path"],
        payload,
    )
    return BranchPredictionStatusBindingV3(
        status=status,
        timestamp=_timestamp_binding(
            status=status,
            sample=sample,
            schedule=schedule,
        ),
        sources=source,
    )


def test_branch_source_recomputes_only_prior_and_probability(source_inputs) -> None:
    source = source_inputs["branch_sources"]["quant"]
    payload = build_branch_prediction_source_status_v3(evidence=source)
    validated = validate_branch_prediction_source_status_v3(
        payload,
        evidence=source,
    )

    assert decode_f64(validated["prior_base_rate"], label="prior") == 0.5
    assert 0.0 <= decode_f64(
        validated["calibrated_probability"],
        label="probability",
    ) <= 1.0
    assert validated["source_recomputation_complete"] is False
    assert validated["blockers"] == sorted(
        [
            (
                "calibration_prediction_requirement_unsupported:"
                f"branch=quant:requirement={ALPHA_INTERVAL_REQUIREMENT}"
            ),
            (
                "calibration_resolver_execution_binding_not_integrated:"
                "branch=quant:requirement=prior_probability"
            ),
            (
                "calibration_resolver_execution_binding_not_integrated:"
                "branch=quant:requirement=branch_probability"
            ),
        ]
    )
    assert not set(validated).intersection(
        {"predicted_alpha", "interval_lower", "interval_upper", "lambda_value"}
    )


def test_branch_status_binding_rejects_anchor_or_resealed_alpha(source_inputs) -> None:
    plan = source_inputs["plan_evidence"].read()
    binding = _branch_binding(
        source=source_inputs["branch_sources"]["quant"],
        plan=plan,
        schedule=source_inputs["schedule"],
    )
    assert binding.read()["branch"] == "quant"

    tampered = deepcopy(binding.status.read())
    tampered.pop("semantic_sha256")
    tampered["predicted_alpha"] = {"encoding": "ieee754-f64-be-hex-v1", "hex": "0000000000000000"}
    with pytest.raises(EvidenceV2Error, match="fields mismatch"):
        validate_branch_prediction_source_status_v3(
            seal_semantic(tampered),
            evidence=binding.sources,
        )

    bad_timestamp = SimpleNamespace(
        attempt=binding.timestamp.attempt,
        validation_receipt=binding.timestamp.validation_receipt,
        read=lambda: (
            {"protocol_attempt_id": ATTEMPT_ID},
            {
                "anchored_artifact_ref": binding.status.reference.to_dict(),
                "anchor_kind": "prediction",
                "anchor_not_before": "2026-07-18T07:00:01Z",
                "anchor_not_after": "2026-07-18T09:00:00Z",
            },
        ),
    )
    with pytest.raises(EvidenceV2Error, match="RFC3161 slot anchor"):
        replace(binding, timestamp=bad_timestamp).read()

    rebound_status = replace(
        binding.status,
        reference=replace(
            binding.status.reference,
            root_policy="v16.governed-data-root.v2",
        ),
    )
    with pytest.raises(EvidenceV2Error, match="status/timestamp paths drift"):
        replace(binding, status=rebound_status).read()


def test_lambda_status_is_explicitly_unsupported(source_inputs) -> None:
    source = LambdaFoldSourceEvidenceBundleV3(
        plan=source_inputs["plan"],
        branch="quant",
        fold_id="fold-0",
        training_source_artifacts=(
            source_inputs["lambda_sources"][("quant", "fold-0")],
        ),
    )
    payload = build_lambda_fold_source_status_v3(evidence=source)
    validated = validate_lambda_fold_source_status_v3(payload, evidence=source)

    assert validated["unsupported_requirement_id"] == FOLD_TRAINING_REQUIREMENT
    assert validated["blockers"] == [
        (
            "calibration_lambda_requirement_unsupported:"
            "branch=quant:fold=fold-0:requirement=fold_training_algorithm"
        )
    ]
    assert "lambda_value" not in validated


def _aggregate_bundle(
    source_inputs: dict[str, Any],
    monkeypatch,
) -> CalibrationSourceStatusEvidenceBundleV3:
    plan = source_inputs["plan_evidence"].read()
    branches = tuple(
        _branch_binding(
            source=source_inputs["branch_sources"][branch],
            plan=plan,
            schedule=source_inputs["schedule"],
        )
        for branch in FORMAL_BRANCHES
    )
    lambdas = []
    for fold in plan["lambda_fold_plans"]:
        sources = LambdaFoldSourceEvidenceBundleV3(
            plan=source_inputs["plan"],
            branch=fold["branch"],
            fold_id=fold["fold_id"],
            training_source_artifacts=(
                source_inputs["lambda_sources"][(fold["branch"], fold["fold_id"])],
            ),
        )
        status = _bound_at(
            fold["status_artifact"]["absolute_path"],
            build_lambda_fold_source_status_v3(evidence=sources),
        )
        lambdas.append((status, sources))

    costs = []
    targets = []
    for sample in plan["sample_plans"]:
        cost = _bound_at(
            sample["artifacts"]["cost_status"]["absolute_path"],
            build_cost_source_status_v3(
                plan=source_inputs["plan"],
                sample_id=sample["sample_id"],
                source_artifacts=(),
            ),
        )
        costs.append((cost, ()))
        target = _bound_at(
            sample["artifacts"]["target_status"]["absolute_path"],
            seal_semantic(
                {
                    "schema_version": TARGET_STATUS_SCHEMA,
                    "sample_id": sample["sample_id"],
                    "blockers": ["shared-target-blocker"],
                    "activation_candidate": False,
                    "new_risk_authorized": False,
                    "production_apply_enabled": False,
                }
            ),
        )
        target_evidence = TargetSourceEvidenceBundleV4(
            plan=source_inputs["plan"],
            schedule_anchor=source_inputs["schedule_anchor"],
            stock_marks=target,
            stock_source_set=target,
            stock_sources=SimpleNamespace(),
            benchmark_manifest=target,
            benchmark_parquet=SimpleNamespace(),
            cost_status=cost,
            cost_source_artifacts=(),
        )
        targets.append((target, target_evidence))

    monkeypatch.setattr(
        source_module,
        "validate_target_source_status_v3",
        lambda value, *, evidence: value,
    )
    return CalibrationSourceStatusEvidenceBundleV3(
        plan=source_inputs["plan_evidence"],
        schedule_anchor=source_inputs["schedule_anchor"],
        branch_statuses=branches,
        lambda_statuses=tuple(lambdas),
        cost_statuses=tuple(costs),
        target_statuses=tuple(targets),
    )


def test_aggregate_binds_paths_order_and_collision_safe_provenance(
    source_inputs,
    monkeypatch,
) -> None:
    evidence = _aggregate_bundle(source_inputs, monkeypatch)
    payload = build_calibration_source_status_v3(evidence=evidence)
    validated = validate_calibration_source_status_v3(payload, evidence=evidence)

    assert validated["source_recomputation_complete"] is False
    assert validated["activation_candidate"] is False
    assert validated["new_risk_authorized"] is False
    assert validated["blockers"].count("shared-target-blocker") == 1
    assert sum(
        item["blocker"] == "shared-target-blocker"
        for item in validated["blocker_sources"]
    ) == len(FORMAL_BRANCHES)

    rebound = replace(
        evidence.cost_statuses[0][0],
        reference=replace(
            evidence.cost_statuses[0][0].reference,
            absolute_path=f"{ROOT}/rebound-cost-status.json",
        ),
    )
    with pytest.raises(EvidenceV2Error, match="cost status path drifts"):
        replace(
            evidence,
            cost_statuses=((rebound, ()),) + evidence.cost_statuses[1:],
        ).read()

    rebound_lambda = replace(
        evidence.lambda_statuses[0][0],
        reference=replace(
            evidence.lambda_statuses[0][0].reference,
            absolute_path=f"{ROOT}/rebound-lambda-status.json",
        ),
    )
    with pytest.raises(EvidenceV2Error, match="lambda status path drifts"):
        replace(
            evidence,
            lambda_statuses=(
                (rebound_lambda, evidence.lambda_statuses[0][1]),
            )
            + evidence.lambda_statuses[1:],
        ).read()

    rebound_target = replace(
        evidence.target_statuses[0][0],
        reference=replace(
            evidence.target_statuses[0][0].reference,
            absolute_path=f"{ROOT}/rebound-target-status.json",
        ),
    )
    with pytest.raises(EvidenceV2Error, match="target status path drifts"):
        replace(
            evidence,
            target_statuses=(
                (rebound_target, evidence.target_statuses[0][1]),
            )
            + evidence.target_statuses[1:],
        ).read()

    with pytest.raises(EvidenceV2Error, match="branch statuses do not cover"):
        replace(
            evidence,
            branch_statuses=tuple(reversed(evidence.branch_statuses)),
        ).read()


def test_new_calibration_modules_are_import_pure_and_builders_make_no_calls(
    source_inputs,
    monkeypatch,
) -> None:
    module_paths = [
        Path(source_module.__file__).resolve(),
        Path(plan_module.__file__).resolve(),
        Path(source_module.__file__).with_name("schedule_v4.py"),
        Path(source_module.__file__).with_name("target_v4.py"),
        Path(source_module.__file__).with_name("readiness_v3.py"),
    ]
    forbidden_roots = {
        "requests",
        "httpx",
        "urllib",
        "socket",
        "subprocess",
        "tushare",
        "yfinance",
    }
    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            str(node.module).split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert imported.isdisjoint(forbidden_roots)

    source = source_inputs["branch_sources"]["quant"]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external or mutating call attempted")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)

    payload = build_branch_prediction_source_status_v3(evidence=source)
    assert payload["source_recomputation_complete"] is False
