from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import hashlib
from types import SimpleNamespace

import pytest

import quant_investor.v16.evidence_v2.calibration_plan_v3 as plan_module
import quant_investor.v16.evidence_v2.target_v4 as target_v4
from quant_investor.v16.evidence_v2.calibration_plan_v3 import (
    FORMAL_BRANCHES,
    LAMBDA_STATUS_SCHEMA,
    POSTERIOR_RUNTIME_SCHEMAS,
    RESOLVER_IMPLEMENTATION_SCHEMA,
    SAMPLE_ARTIFACT_SCHEMAS,
    STOCK_SOURCE_SET_SCHEMA,
    CalibrationSamplePlanV3,
    LambdaFoldPlanV3,
    PlannedArtifactPath,
    build_calibration_universe_plan_v3,
)
from quant_investor.v16.evidence_v2.contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.schedule_v4 import ScheduleAnchorBindingV4
from quant_investor.v16.evidence_v2.target import (
    INDEX_TABLE_SCHEMA,
    build_h00300_manifest,
    build_stock_mark_evidence_from_sources,
    validate_h00300_parquet,
)
from quant_investor.v16.evidence_v2.target_v4 import (
    TargetSourceEvidenceBundleV4,
    build_cost_source_status_v3,
    build_stock_source_set_v3,
    build_target_source_status_v3,
    validate_cost_source_status_v3,
    validate_stock_source_set_v3,
    validate_target_source_status_v3,
)
from tests.unit.test_v16_evidence_v2_metrics_runtime_timestamp import (
    _bound,
    _calibration_benchmark_parquet,
    _calibration_stock_sources,
    _ref,
)


ATTEMPT = "attempt-v16-target-v4-001"
ROOT = "/private/evidence"


def _bound_at(path: str, payload: dict) -> BoundCanonicalArtifact:
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
    return PlannedArtifactPath(f"{ROOT}/{path}", schema)


def _plan(
    *,
    stock_source_set_ref: EvidenceRef,
    benchmark_manifest_ref: EvidenceRef,
    symbol: str,
    start: date,
    end: date,
    schedule_ref: EvidenceRef,
) -> BoundCanonicalArtifact:
    del schedule_ref
    resolver_refs = {
        requirement: _ref(
            f"resolver-{requirement}.json",
            schema=RESOLVER_IMPLEMENTATION_SCHEMA,
        )
        for requirement in ("prior_probability", "branch_probability")
    }
    model_refs = {
        branch: _ref(f"model-{branch}.json", schema="v16.frozen-model-bundle.v2")
        for branch in FORMAL_BRANCHES
    }
    runtime_refs = {
        key: _ref(f"runtime-{key}.json", schema=schema)
        for key, schema in POSTERIOR_RUNTIME_SCHEMAS.items()
    }
    samples: list[CalibrationSamplePlanV3] = []
    for branch in FORMAL_BRANCHES:
        sample_id = f"{branch}-sample"
        artifacts = tuple(
            (
                key,
                _planned(f"{key}-{sample_id}.json", schema),
            )
            for key, schema in SAMPLE_ARTIFACT_SCHEMAS.items()
        )
        samples.append(
            CalibrationSamplePlanV3(
                sample_id=sample_id,
                branch=branch,
                symbol=symbol,
                cohort_id="cohort-0",
                cohort_start_date=start.isoformat(),
                cohort_end_date=end.isoformat(),
                slot_id="slot-0",
                artifacts=artifacts,
                stock_source_set_ref=stock_source_set_ref,
                benchmark_manifest_ref=benchmark_manifest_ref,
            )
        )
    folds = [
        LambdaFoldPlanV3(
            branch=branch,
            fold_id=f"fold-{index}",
            training_source_refs=(
                _ref(f"lambda-{branch}-{index}.bin", schema="lambda-training.v1"),
            ),
            status_artifact=_planned(
                f"lambda-{branch}-{index}.json",
                LAMBDA_STATUS_SCHEMA,
            ),
        )
        for branch in FORMAL_BRANCHES
        for index in range(2)
    ]
    payload = build_calibration_universe_plan_v3(
        protocol_attempt_id=ATTEMPT,
        epoch="B",
        schedule_id="schedule-b",
        private_root=ROOT,
        runtime_capsule_ref=_ref(
            "runtime-capsule.json",
            schema="v16.hermetic-runtime-capsule.v2",
        ),
        resolver_implementation_refs=resolver_refs,
        model_bundle_refs=model_refs,
        posterior_runtime_refs=runtime_refs,
        sample_plans=samples,
        lambda_fold_plans=folds,
    )
    return _bound("calibration-plan-v3.json", payload)


@pytest.fixture
def target_inputs(monkeypatch):
    monkeypatch.setattr(plan_module, "MIN_BRANCH_SAMPLES", 1)
    monkeypatch.setattr(plan_module, "MIN_BRANCH_COHORTS", 1)
    symbol = "S000001"
    start = date(2026, 1, 5)
    end = start + timedelta(days=19)
    calendar_ref = _ref("calendar.json", schema="v16.open-session-calendar.v2")
    schedule_ref = _ref("schedule-v4.json", schema="v16.evidence-schedule-declaration.v4")
    specs = [
        {
            "symbol": symbol,
            "cohort_start": start,
            "cohort_end": end,
            "realized": 0.03,
        }
    ]
    stock_bundle, prepared = _calibration_stock_sources(
        specs=specs,
        calendar_ref=calendar_ref,
    )
    source_set = _bound(
        "stock-source-set.json",
        build_stock_source_set_v3(
            protocol_attempt_id=ATTEMPT,
            source_refs={
                "market_parquet": stock_bundle.market_parquet.reference,
                "adjustment_factors": stock_bundle.adjustment_factors.reference,
                "pit_membership": stock_bundle.pit_membership.reference,
                "suspensions": stock_bundle.suspensions.reference,
            },
        ),
    )
    benchmark_payload = _calibration_benchmark_parquet(
        [start + timedelta(days=index) for index in range(20)]
    )
    benchmark_projection = validate_h00300_parquet(benchmark_payload)
    benchmark_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=INDEX_TABLE_SCHEMA,
        absolute_path=f"{ROOT}/h00300.parquet",
        byte_sha256=hashlib.sha256(benchmark_payload).hexdigest(),
        semantic_sha256=benchmark_projection["parquet_metadata_semantic_sha256"],
        root_policy="v16.private-evidence-root.v2",
    )
    benchmark_manifest = _bound(
        "h00300-manifest.json",
        build_h00300_manifest(
            generation_id="h00300-target-v4",
            created_at="2026-01-25T00:00:00Z",
            table_ref=benchmark_ref,
            parquet_payload=benchmark_payload,
            official_source_receipt=_ref("h00300-source.json"),
            calendar_ref=calendar_ref,
        ),
    )
    plan = _plan(
        stock_source_set_ref=source_set.reference,
        benchmark_manifest_ref=benchmark_manifest.reference,
        symbol=symbol,
        start=start,
        end=end,
        schedule_ref=schedule_ref,
    )
    stock_marks = _bound_at(
        f"{ROOT}/stock_marks-quant-sample.json",
        build_stock_mark_evidence_from_sources(
            sources=prepared,
            protocol_attempt_id=ATTEMPT,
            sample_id="quant-sample",
            symbol=symbol,
            slot_id="slot-0",
            schedule_ref=schedule_ref,
            entry_date=start.isoformat(),
            exit_date=end.isoformat(),
        ),
    )
    cost_payload = build_cost_source_status_v3(
        plan=plan,
        sample_id="quant-sample",
        source_artifacts=(),
    )
    cost_status = _bound_at(f"{ROOT}/cost_status-quant-sample.json", cost_payload)
    schedule = {
        "protocol_attempt_id": ATTEMPT,
        "epoch": "B",
        "schedule_id": "schedule-b",
        "calibration_plan_ref": plan.reference.to_dict(),
        "open_session_calendar": calendar_ref.to_dict(),
        "slots": [
            {
                "slot_id": "slot-0",
                "s0_close_at": "2026-01-04T07:00:00Z",
                "s1_open_at": "2026-01-05T01:15:00Z",
                "target_sessions": [
                    (start + timedelta(days=index)).isoformat() for index in range(20)
                ],
            }
        ],
    }
    fake_evidence = SimpleNamespace(schedule=SimpleNamespace(reference=schedule_ref))
    schedule_anchor = ScheduleAnchorBindingV4(
        evidence=fake_evidence,
        timestamp=SimpleNamespace(),
    )
    monkeypatch.setattr(target_v4, "validate_schedule_anchor_binding_v4", lambda _value: schedule)
    evidence = TargetSourceEvidenceBundleV4(
        plan=plan,
        schedule_anchor=schedule_anchor,
        stock_marks=stock_marks,
        stock_source_set=source_set,
        stock_sources=stock_bundle,
        benchmark_manifest=benchmark_manifest,
        benchmark_parquet=BoundRawArtifact(benchmark_ref, benchmark_payload),
        cost_status=cost_status,
        cost_source_artifacts=(),
    )
    return evidence, plan, source_set


def test_target_v4_recomputes_boundary_sources_and_stops_before_outcome(target_inputs) -> None:
    evidence, _plan_artifact, source_set = target_inputs
    assert (
        validate_stock_source_set_v3(source_set.read())["schema_version"]
        == STOCK_SOURCE_SET_SCHEMA
    )
    status = build_target_source_status_v3(evidence=evidence)
    validated = validate_target_source_status_v3(status, evidence=evidence)

    assert validated["source_recomputation_complete"] is False
    assert validated["blockers"] == [
        "calibration_target_outcome_blocked:"
        "sample=quant-sample:dependency=eight_component_cost_model"
    ]
    assert not set(validated).intersection(
        {"costs", "realized_alpha", "outcome", "predicted_alpha", "lambda_value"}
    )


def test_cost_status_rejects_v2_or_resealed_value_injection(target_inputs) -> None:
    evidence, plan, _source_set = target_inputs
    valid = evidence.cost_status.read()
    assert validate_cost_source_status_v3(
        valid,
        plan=plan,
        source_artifacts=(),
    ) == valid

    tampered = deepcopy(valid)
    tampered.pop("semantic_sha256")
    tampered["costs"] = []
    with pytest.raises(EvidenceV2Error, match="fields mismatch"):
        validate_cost_source_status_v3(
            seal_semantic(tampered),
            plan=plan,
            source_artifacts=(),
        )


def test_stock_source_set_rejects_unsafe_attempt_id(target_inputs) -> None:
    _evidence, _plan_artifact, source_set = target_inputs
    refs = {
        key: EvidenceRef.from_dict(source_set.read()["source_refs"][key])
        for key in ("market_parquet", "adjustment_factors", "pit_membership", "suspensions")
    }
    with pytest.raises(EvidenceV2Error, match="safe identifier"):
        build_stock_source_set_v3(
            protocol_attempt_id="attempt/escape",
            source_refs=refs,
        )


def test_target_v4_rejects_root_policy_substitution(target_inputs) -> None:
    evidence, _plan_artifact, _source_set = target_inputs
    rebound_cost = replace(
        evidence.cost_status,
        reference=replace(
            evidence.cost_status.reference,
            root_policy="v16.governed-data-root.v2",
        ),
    )

    with pytest.raises(EvidenceV2Error, match="future artifact paths drift"):
        replace(evidence, cost_status=rebound_cost).read()


def test_stock_source_set_rejects_missing_ref_key(target_inputs) -> None:
    _evidence, _plan_artifact, source_set = target_inputs
    tampered = deepcopy(source_set.read())
    tampered.pop("semantic_sha256")
    tampered["source_refs"].pop("suspensions")

    with pytest.raises(EvidenceV2Error, match="ref keys mismatch"):
        validate_stock_source_set_v3(seal_semantic(tampered))


@pytest.mark.parametrize(
    "key",
    ("market_parquet", "adjustment_factors", "pit_membership", "suspensions"),
)
def test_stock_source_set_rejects_wrong_root_policy(target_inputs, key: str) -> None:
    _evidence, _plan_artifact, source_set = target_inputs
    refs = {
        source_key: EvidenceRef.from_dict(source_set.read()["source_refs"][source_key])
        for source_key in (
            "market_parquet",
            "adjustment_factors",
            "pit_membership",
            "suspensions",
        )
    }
    refs[key] = replace(refs[key], root_policy="v16.mock-root.v0")

    with pytest.raises(EvidenceV2Error, match=f"ref is invalid: {key}"):
        build_stock_source_set_v3(
            protocol_attempt_id=ATTEMPT,
            source_refs=refs,
        )
