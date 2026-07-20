"""Truthful source status for prospective v16 calibration.

Only the prior base rate and branch probability have implemented source
recomputation.  Branch alpha/interval, fold lambda, and cost remain explicit
requirements, so every status in this module is permanently nonauthorizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from collections.abc import Mapping
from typing import Any

from .calibration_plan_v3 import (
    BRANCH_STATUS_SCHEMA,
    CALIBRATION_PLAN_V3_SCHEMA,
    CALIBRATION_SOURCE_STATUS_SCHEMA,
    FORMAL_BRANCHES,
    LAMBDA_STATUS_SCHEMA,
    POSTERIOR_RUNTIME_SCHEMAS,
    PRIVATE_ROOT_POLICY,
    RESOLVER_IMPLEMENTATION_SCHEMA,
    SUPPORTED_REQUIREMENTS,
    CalibrationPlanEvidenceBundleV3,
    validate_calibration_universe_plan_v3,
)
from .contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EvidenceRef,
    EvidenceV2Error,
    decode_f64,
    encode_f64,
    seal_semantic,
    validate_semantic_seal,
)
from .posterior import Stage1ReviewBinding, replay_stage1_formal_evidence
from .posterior_runtime import PosteriorRuntimeArtifacts, PosteriorRuntimeBundle
from .runtime_identity import RUNTIME_CAPSULE_SCHEMA, validate_runtime_capsule
from .schedule_v4 import ScheduleAnchorBindingV4, validate_schedule_anchor_binding_v4
from .target_v4 import (
    TargetSourceEvidenceBundleV4,
    validate_cost_source_status_v3,
    validate_target_source_status_v3,
)
from .timestamp import TimestampAnchorBinding

RESOLVER_MODULE_SOURCE_SCHEMA = "v16.python-module-source.v3"
FOLD_TRAINING_REQUIREMENT = "fold_training_algorithm"
ALPHA_INTERVAL_REQUIREMENT = "branch_only_alpha_interval_model"
RESOLVER_ENTRYPOINTS = {
    "prior_probability": (
        "quant_investor.v16.evidence_v2.calibration_source_v3:"
        "recompute_prior_base_rate_v3"
    ),
    "branch_probability": (
        "quant_investor.v16.evidence_v2.calibration_source_v3:"
        "recompute_calibrated_probability_v3"
    ),
}


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if (
        not text
        or text != text.strip()
        or len(text) > 128
        or any(character not in allowed for character in text)
    ):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _source_reference(
    artifact: BoundCanonicalArtifact | BoundRawArtifact,
) -> EvidenceRef:
    if isinstance(artifact, BoundCanonicalArtifact):
        artifact.read()
    elif not isinstance(artifact, BoundRawArtifact):
        raise EvidenceV2Error("runtime source has the wrong bound artifact type")
    return artifact.reference


def _canonical_utc(value: Any) -> str:
    parsed = value.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _matches_planned_reference(
    reference: EvidenceRef,
    planned: Mapping[str, Any],
) -> bool:
    return (
        reference.absolute_path == planned["absolute_path"]
        and reference.artifact_schema == planned["artifact_schema"]
        and reference.root_policy == planned["root_policy"]
    )


def _sample(plan: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    matches = [item for item in plan["sample_plans"] if item["sample_id"] == sample_id]
    if len(matches) != 1:
        raise EvidenceV2Error("calibration source sample is missing or ambiguous")
    return dict(matches[0])


def _fold(plan: Mapping[str, Any], branch: str, fold_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in plan["lambda_fold_plans"]
        if item["branch"] == branch and item["fold_id"] == fold_id
    ]
    if len(matches) != 1:
        raise EvidenceV2Error("lambda source fold is missing or ambiguous")
    return dict(matches[0])


def build_resolver_implementation_v3(
    *,
    protocol_attempt_id: str,
    requirement_id: str,
    resolver_id: str,
    entrypoint: str,
    module_source_ref: EvidenceRef,
    runtime_capsule_ref: EvidenceRef,
    source_tree_ref: EvidenceRef,
) -> dict[str, Any]:
    requirement = str(requirement_id)
    if requirement not in SUPPORTED_REQUIREMENTS:
        raise EvidenceV2Error("resolver implementation requirement is unsupported")
    if entrypoint != RESOLVER_ENTRYPOINTS[requirement]:
        raise EvidenceV2Error("resolver implementation entrypoint drift")
    if (
        module_source_ref.artifact_schema != RESOLVER_MODULE_SOURCE_SCHEMA
        or module_source_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("resolver module source ref is invalid")
    if (
        runtime_capsule_ref.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_capsule_ref.root_policy != PRIVATE_ROOT_POLICY
        or source_tree_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("resolver runtime/source-tree refs are invalid")
    return seal_semantic(
        {
            "schema_version": RESOLVER_IMPLEMENTATION_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "requirement_id": requirement,
            "resolver_id": _identifier(resolver_id, label="resolver_id"),
            "entrypoint": entrypoint,
            "module_source_ref": module_source_ref.to_dict(),
            "runtime_capsule_ref": runtime_capsule_ref.to_dict(),
            "source_tree_ref": source_tree_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_resolver_implementation_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "requirement_id",
        "resolver_id",
        "entrypoint",
        "module_source_ref",
        "runtime_capsule_ref",
        "source_tree_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="resolver implementation v3")
    if payload["schema_version"] != RESOLVER_IMPLEMENTATION_SCHEMA:
        raise EvidenceV2Error("resolver implementation v3 schema mismatch")
    rebuilt = build_resolver_implementation_v3(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        requirement_id=str(payload["requirement_id"]),
        resolver_id=str(payload["resolver_id"]),
        entrypoint=str(payload["entrypoint"]),
        module_source_ref=EvidenceRef.from_dict(payload["module_source_ref"]),
        runtime_capsule_ref=EvidenceRef.from_dict(payload["runtime_capsule_ref"]),
        source_tree_ref=EvidenceRef.from_dict(payload["source_tree_ref"]),
    )
    if rebuilt != payload:
        raise EvidenceV2Error("resolver implementation v3 is not canonical")
    return payload


@dataclass(frozen=True)
class ResolverImplementationEvidenceBundleV3:
    manifest: BoundCanonicalArtifact
    module_source: BoundRawArtifact
    runtime_capsule: BoundCanonicalArtifact
    source_tree: BoundRawArtifact

    def read(self) -> dict[str, Any]:
        if (
            self.manifest.reference.artifact_schema != RESOLVER_IMPLEMENTATION_SCHEMA
            or self.manifest.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("resolver implementation manifest ref is invalid")
        manifest = validate_resolver_implementation_v3(self.manifest.read())
        runtime = validate_runtime_capsule(self.runtime_capsule.read())
        if (
            manifest["module_source_ref"] != self.module_source.reference.to_dict()
            or manifest["runtime_capsule_ref"] != self.runtime_capsule.reference.to_dict()
            or manifest["source_tree_ref"] != self.source_tree.reference.to_dict()
            or runtime["protocol_attempt_id"] != manifest["protocol_attempt_id"]
        ):
            raise EvidenceV2Error("resolver implementation evidence refs drift")
        source_components = [
            item for item in runtime["components"] if item["component_id"] == "source_tree"
        ]
        if (
            len(source_components) != 1
            or source_components[0]["artifact_ref"]
            != self.source_tree.reference.to_dict()
        ):
            raise EvidenceV2Error("resolver source tree drifts from runtime capsule")
        return manifest


@dataclass(frozen=True)
class RuntimeTrainingSourceBundleV3:
    source_artifacts_by_key: tuple[
        tuple[str, tuple[BoundCanonicalArtifact | BoundRawArtifact, ...]], ...
    ]

    def read(self, runtime: PosteriorRuntimeBundle) -> None:
        provided = dict(self.source_artifacts_by_key)
        if len(provided) != len(self.source_artifacts_by_key) or list(provided) != list(
            POSTERIOR_RUNTIME_SCHEMAS
        ):
            raise EvidenceV2Error("runtime training source keys/order mismatch")
        artifacts = runtime.artifacts
        runtime_artifacts = {
            "prior_training": artifacts.prior_training,
            "likelihood_training": artifacts.likelihood_training,
            "return_model_parameters": artifacts.return_model_parameters,
            "return_model_training": artifacts.return_model_training,
            "bootstrap_offsets": artifacts.bootstrap_offsets,
            "bootstrap_training": artifacts.bootstrap_training,
            "correlation_matrix": artifacts.correlation_matrix,
            "correlation_training": artifacts.correlation_training,
        }
        for key in POSTERIOR_RUNTIME_SCHEMAS:
            payload = runtime_artifacts[key].read()
            expected = [
                EvidenceRef.from_dict(item)
                for item in payload.get("source_input_refs", [])
            ]
            actual = [_source_reference(item) for item in provided[key]]
            if [item.to_dict() for item in actual] != [item.to_dict() for item in expected]:
                raise EvidenceV2Error(f"runtime training sources drift: {key}")


def recompute_prior_base_rate_v3(*, runtime: PosteriorRuntimeBundle) -> float:
    if not isinstance(runtime, PosteriorRuntimeBundle):
        raise EvidenceV2Error("prior resolver requires PosteriorRuntimeBundle")
    return float(runtime.prior.base_rate)


def recompute_calibrated_probability_v3(
    *,
    runtime: PosteriorRuntimeBundle,
    stage1_binding: Stage1ReviewBinding,
    symbol: str,
    branch: str,
) -> float:
    if not isinstance(runtime, PosteriorRuntimeBundle):
        raise EvidenceV2Error("branch probability resolver requires PosteriorRuntimeBundle")
    records = [
        item
        for item in replay_stage1_formal_evidence(stage1_binding)
        if item.symbol == symbol and item.branch == branch
    ]
    if len(records) != 1:
        raise EvidenceV2Error("formal branch replay row is missing or ambiguous")
    stats = runtime.calibration_store.calibration_stats(branch, records[0].raw_score)
    return float(stats["probability"])


@dataclass(frozen=True)
class BranchPredictionSourceEvidenceBundleV3:
    plan: CalibrationPlanEvidenceBundleV3
    schedule_anchor: ScheduleAnchorBindingV4
    stage1_binding: Stage1ReviewBinding
    runtime_artifacts: PosteriorRuntimeArtifacts
    runtime_training_sources: RuntimeTrainingSourceBundleV3
    resolver_implementations: tuple[ResolverImplementationEvidenceBundleV3, ...]
    sample_id: str

    def read(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], PosteriorRuntimeBundle]:
        plan = self.plan.read()
        schedule = validate_schedule_anchor_binding_v4(self.schedule_anchor)
        sample = _sample(plan, self.sample_id)
        if (
            schedule["protocol_attempt_id"] != plan["protocol_attempt_id"]
            or schedule["epoch"] != plan["epoch"]
            or schedule["schedule_id"] != plan["schedule_id"]
            or schedule["calibration_plan_ref"] != self.plan.plan.reference.to_dict()
        ):
            raise EvidenceV2Error("branch source schedule/plan lineage mismatch")
        slots = [item for item in schedule["slots"] if item["slot_id"] == sample["slot_id"]]
        if len(slots) != 1:
            raise EvidenceV2Error("branch source schedule slot is missing or ambiguous")
        slot = slots[0]
        stage1 = self.stage1_binding.read()
        request_path = sample["artifacts"]["stage1_request"]
        response_path = sample["artifacts"]["stage1_response"]
        if (
            not _matches_planned_reference(
                self.stage1_binding.request.reference,
                request_path,
            )
            or not _matches_planned_reference(
                self.stage1_binding.response.reference,
                response_path,
            )
            or _canonical_utc(stage1.request.decision_cutoff_at)
            != slot["decision_cutoff_at"]
            or _canonical_utc(stage1.response.decision_cutoff_at)
            != slot["decision_cutoff_at"]
        ):
            raise EvidenceV2Error("Stage1 paths/cutoff drift from the planned slot")
        formal = [
            item
            for item in replay_stage1_formal_evidence(self.stage1_binding)
            if item.symbol == sample["symbol"] and item.branch == sample["branch"]
        ]
        if len(formal) != 1:
            raise EvidenceV2Error("Stage1 replay does not contain the planned symbol/branch")
        runtime = PosteriorRuntimeBundle(artifacts=self.runtime_artifacts)
        if runtime.protocol_attempt_id != plan["protocol_attempt_id"]:
            raise EvidenceV2Error("posterior runtime protocol attempt drift")
        expected_runtime_refs = plan["posterior_runtime_refs"]
        if runtime.refs_projection() != {
            "model_bundle_refs": plan["model_bundle_refs"],
            "prior_training_ref": expected_runtime_refs["prior_training"],
            "likelihood_training_ref": expected_runtime_refs["likelihood_training"],
            "return_model_parameters_ref": expected_runtime_refs["return_model_parameters"],
            "return_model_training_ref": expected_runtime_refs["return_model_training"],
            "bootstrap_offsets_ref": expected_runtime_refs["bootstrap_offsets"],
            "bootstrap_training_ref": expected_runtime_refs["bootstrap_training"],
            "correlation_matrix_ref": expected_runtime_refs["correlation_matrix"],
            "correlation_training_ref": expected_runtime_refs["correlation_training"],
        }:
            raise EvidenceV2Error("posterior runtime refs drift from calibration plan")
        self.runtime_training_sources.read(runtime)
        model_bundles = self.schedule_anchor.evidence.model_bundles
        model_payloads = [item.read() for item in model_bundles]
        model_matches = [
            (bundle, payload)
            for bundle, payload in zip(model_bundles, model_payloads)
            if payload["branch"] == sample["branch"]
        ]
        if len(model_matches) != 1:
            raise EvidenceV2Error("planned branch model evidence is missing or ambiguous")
        model_bundle, _model_payload = model_matches[0]
        if (
            model_bundle.model_bundle.reference.to_dict()
            != plan["model_bundle_refs"][sample["branch"]]
        ):
            raise EvidenceV2Error("planned branch model ref drift")
        if len(self.resolver_implementations) != len(SUPPORTED_REQUIREMENTS):
            raise EvidenceV2Error("branch source requires two resolver implementations")
        resolver_payloads = [item.read() for item in self.resolver_implementations]
        if [item["requirement_id"] for item in resolver_payloads] != list(
            SUPPORTED_REQUIREMENTS
        ):
            raise EvidenceV2Error("branch resolver implementation order drift")
        for requirement, bundle in zip(SUPPORTED_REQUIREMENTS, self.resolver_implementations):
            if (
                bundle.manifest.reference.to_dict()
                != plan["resolver_implementation_refs"][requirement]
                or bundle.runtime_capsule.reference.to_dict() != plan["runtime_capsule_ref"]
            ):
                raise EvidenceV2Error("branch resolver implementation ref drift")
        return plan, schedule, sample, runtime


def _branch_blockers(branch: str) -> list[str]:
    return sorted(
        [
            (
                "calibration_prediction_requirement_unsupported:"
                f"branch={branch}:requirement={ALPHA_INTERVAL_REQUIREMENT}"
            ),
            (
                "calibration_resolver_execution_binding_not_integrated:"
                f"branch={branch}:requirement=prior_probability"
            ),
            (
                "calibration_resolver_execution_binding_not_integrated:"
                f"branch={branch}:requirement=branch_probability"
            ),
        ]
    )


def build_branch_prediction_source_status_v3(
    *,
    evidence: BranchPredictionSourceEvidenceBundleV3,
) -> dict[str, Any]:
    if not isinstance(evidence, BranchPredictionSourceEvidenceBundleV3):
        raise EvidenceV2Error("branch status requires BranchPredictionSourceEvidenceBundleV3")
    plan, schedule, sample, runtime = evidence.read()
    prior = recompute_prior_base_rate_v3(runtime=runtime)
    probability = recompute_calibrated_probability_v3(
        runtime=runtime,
        stage1_binding=evidence.stage1_binding,
        symbol=sample["symbol"],
        branch=sample["branch"],
    )
    slots = [item for item in schedule["slots"] if item["slot_id"] == sample["slot_id"]]
    slot = slots[0]
    resolver_refs = {
        requirement: bundle.manifest.reference.to_dict()
        for requirement, bundle in zip(
            SUPPORTED_REQUIREMENTS,
            evidence.resolver_implementations,
        )
    }
    return seal_semantic(
        {
            "schema_version": BRANCH_STATUS_SCHEMA,
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "epoch": plan["epoch"],
            "schedule_id": plan["schedule_id"],
            "slot_id": sample["slot_id"],
            "sample_id": sample["sample_id"],
            "symbol": sample["symbol"],
            "branch": sample["branch"],
            "decision_cutoff_at": slot["decision_cutoff_at"],
            "stage1_request_ref": evidence.stage1_binding.request.reference.to_dict(),
            "stage1_response_ref": evidence.stage1_binding.response.reference.to_dict(),
            "schedule_ref": evidence.schedule_anchor.evidence.schedule.reference.to_dict(),
            "model_bundle_ref": plan["model_bundle_refs"][sample["branch"]],
            "runtime_capsule_ref": plan["runtime_capsule_ref"],
            "resolver_manifest_refs": resolver_refs,
            "prior_base_rate": encode_f64(prior),
            "calibrated_probability": encode_f64(probability),
            "source_recomputation_complete": False,
            "blockers": _branch_blockers(sample["branch"]),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_branch_prediction_source_status_v3(
    value: Mapping[str, Any],
    *,
    evidence: BranchPredictionSourceEvidenceBundleV3,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "slot_id",
        "sample_id",
        "symbol",
        "branch",
        "decision_cutoff_at",
        "stage1_request_ref",
        "stage1_response_ref",
        "schedule_ref",
        "model_bundle_ref",
        "runtime_capsule_ref",
        "resolver_manifest_refs",
        "prior_base_rate",
        "calibrated_probability",
        "source_recomputation_complete",
        "blockers",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="branch prediction source status v3")
    if payload["schema_version"] != BRANCH_STATUS_SCHEMA:
        raise EvidenceV2Error("branch prediction source status v3 schema mismatch")
    decode_f64(payload["prior_base_rate"], label="prior_base_rate")
    decode_f64(payload["calibrated_probability"], label="calibrated_probability")
    rebuilt = build_branch_prediction_source_status_v3(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("branch prediction source status drifts from source replay")
    return payload


@dataclass(frozen=True)
class BranchPredictionStatusBindingV3:
    status: BoundCanonicalArtifact
    timestamp: TimestampAnchorBinding
    sources: BranchPredictionSourceEvidenceBundleV3

    def read(self) -> dict[str, Any]:
        payload = validate_branch_prediction_source_status_v3(
            self.status.read(),
            evidence=self.sources,
        )
        plan = self.sources.plan.read()
        sample = _sample(plan, payload["sample_id"])
        planned = sample["artifacts"]
        if not _matches_planned_reference(
            self.status.reference,
            planned["branch_status"],
        ) or not _matches_planned_reference(
            self.timestamp.attempt.reference,
            planned["prediction_timestamp_attempt"],
        ) or not _matches_planned_reference(
            self.timestamp.validation_receipt.reference,
            planned["prediction_timestamp_receipt"],
        ):
            raise EvidenceV2Error("branch prediction status/timestamp paths drift from plan")
        schedule = validate_schedule_anchor_binding_v4(self.sources.schedule_anchor)
        slots = [item for item in schedule["slots"] if item["slot_id"] == payload["slot_id"]]
        if len(slots) != 1:
            raise EvidenceV2Error("branch prediction status slot is missing or ambiguous")
        attempt, receipt = self.timestamp.read()
        slot = slots[0]
        if (
            receipt["anchored_artifact_ref"] != self.status.reference.to_dict()
            or receipt["anchor_kind"] != "prediction"
            or receipt["anchor_not_before"] != slot["s0_close_at"]
            or receipt["anchor_not_after"] != slot["s1_open_at"]
            or attempt["protocol_attempt_id"] != payload["protocol_attempt_id"]
        ):
            raise EvidenceV2Error("branch prediction RFC3161 slot anchor lineage mismatch")
        return payload


@dataclass(frozen=True)
class LambdaFoldSourceEvidenceBundleV3:
    plan: BoundCanonicalArtifact
    branch: str
    fold_id: str
    training_source_artifacts: tuple[BoundCanonicalArtifact | BoundRawArtifact, ...]

    def read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            self.plan.reference.artifact_schema != CALIBRATION_PLAN_V3_SCHEMA
            or self.plan.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("lambda source requires a bound calibration plan v3")
        plan = validate_calibration_universe_plan_v3(self.plan.read())
        fold = _fold(plan, self.branch, self.fold_id)
        expected = [EvidenceRef.from_dict(item) for item in fold["training_source_refs"]]
        actual = [_source_reference(item) for item in self.training_source_artifacts]
        if [item.to_dict() for item in actual] != [item.to_dict() for item in expected]:
            raise EvidenceV2Error("lambda training sources drift from pre-s0 plan")
        return plan, fold


def _lambda_blocker(branch: str, fold_id: str) -> str:
    return (
        "calibration_lambda_requirement_unsupported:"
        f"branch={branch}:fold={fold_id}:requirement={FOLD_TRAINING_REQUIREMENT}"
    )


def build_lambda_fold_source_status_v3(
    *,
    evidence: LambdaFoldSourceEvidenceBundleV3,
) -> dict[str, Any]:
    if not isinstance(evidence, LambdaFoldSourceEvidenceBundleV3):
        raise EvidenceV2Error("lambda status requires LambdaFoldSourceEvidenceBundleV3")
    plan, fold = evidence.read()
    return seal_semantic(
        {
            "schema_version": LAMBDA_STATUS_SCHEMA,
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "epoch": plan["epoch"],
            "schedule_id": plan["schedule_id"],
            "branch": fold["branch"],
            "fold_id": fold["fold_id"],
            "training_source_refs": fold["training_source_refs"],
            "unsupported_requirement_id": FOLD_TRAINING_REQUIREMENT,
            "source_recomputation_complete": False,
            "blockers": [_lambda_blocker(fold["branch"], fold["fold_id"])],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_lambda_fold_source_status_v3(
    value: Mapping[str, Any],
    *,
    evidence: LambdaFoldSourceEvidenceBundleV3,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "branch",
        "fold_id",
        "training_source_refs",
        "unsupported_requirement_id",
        "source_recomputation_complete",
        "blockers",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="lambda fold source status v3")
    if payload["schema_version"] != LAMBDA_STATUS_SCHEMA:
        raise EvidenceV2Error("lambda fold source status v3 schema mismatch")
    rebuilt = build_lambda_fold_source_status_v3(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("lambda fold source status drifts from bound sources")
    return payload


@dataclass(frozen=True)
class CalibrationSourceStatusEvidenceBundleV3:
    plan: CalibrationPlanEvidenceBundleV3
    schedule_anchor: ScheduleAnchorBindingV4
    branch_statuses: tuple[BranchPredictionStatusBindingV3, ...]
    lambda_statuses: tuple[
        tuple[BoundCanonicalArtifact, LambdaFoldSourceEvidenceBundleV3], ...
    ]
    cost_statuses: tuple[
        tuple[
            BoundCanonicalArtifact,
            tuple[BoundCanonicalArtifact | BoundRawArtifact, ...],
        ],
        ...,
    ]
    target_statuses: tuple[
        tuple[BoundCanonicalArtifact, TargetSourceEvidenceBundleV4], ...
    ]

    def read(self) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        if not isinstance(self.plan, CalibrationPlanEvidenceBundleV3) or not isinstance(
            self.schedule_anchor,
            ScheduleAnchorBindingV4,
        ):
            raise EvidenceV2Error("calibration aggregate plan/schedule types are invalid")
        if (
            type(self.branch_statuses) is not tuple
            or any(
                not isinstance(item, BranchPredictionStatusBindingV3)
                for item in self.branch_statuses
            )
            or type(self.lambda_statuses) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], BoundCanonicalArtifact)
                or not isinstance(item[1], LambdaFoldSourceEvidenceBundleV3)
                for item in self.lambda_statuses
            )
            or type(self.cost_statuses) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], BoundCanonicalArtifact)
                or type(item[1]) is not tuple
                or any(
                    not isinstance(source, (BoundCanonicalArtifact, BoundRawArtifact))
                    for source in item[1]
                )
                for item in self.cost_statuses
            )
            or type(self.target_statuses) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], BoundCanonicalArtifact)
                or not isinstance(item[1], TargetSourceEvidenceBundleV4)
                for item in self.target_statuses
            )
        ):
            raise EvidenceV2Error("calibration aggregate evidence types are invalid")
        plan = self.plan.read()
        schedule = validate_schedule_anchor_binding_v4(self.schedule_anchor)
        if schedule["calibration_plan_ref"] != self.plan.plan.reference.to_dict():
            raise EvidenceV2Error("calibration aggregate schedule/plan ref drift")
        branches = [item.read() for item in self.branch_statuses]
        lambda_rows: list[dict[str, Any]] = []
        for status, sources in self.lambda_statuses:
            row = validate_lambda_fold_source_status_v3(
                status.read(),
                evidence=sources,
            )
            fold = _fold(plan, row["branch"], row["fold_id"])
            if not _matches_planned_reference(
                status.reference,
                fold["status_artifact"],
            ):
                raise EvidenceV2Error("calibration lambda status path drifts from plan")
            lambda_rows.append(row)
        cost_by_id = {item["sample_id"]: item for item in plan["sample_plans"]}
        costs: list[dict[str, Any]] = []
        for status, source_artifacts in self.cost_statuses:
            raw = status.read()
            sample_id = str(raw.get("sample_id") or "")
            if sample_id not in cost_by_id:
                raise EvidenceV2Error("calibration cost status sample escapes the plan")
            planned = cost_by_id[sample_id]["artifacts"]["cost_status"]
            if not _matches_planned_reference(status.reference, planned):
                raise EvidenceV2Error("calibration cost status path drifts from plan")
            costs.append(
                validate_cost_source_status_v3(
                    raw,
                    plan=self.plan.plan,
                    source_artifacts=source_artifacts,
                )
            )
        targets: list[dict[str, Any]] = []
        for status, target_evidence in self.target_statuses:
            row = validate_target_source_status_v3(
                status.read(),
                evidence=target_evidence,
            )
            if row["sample_id"] not in cost_by_id:
                raise EvidenceV2Error("calibration target status sample escapes the plan")
            planned = cost_by_id[row["sample_id"]]["artifacts"]["target_status"]
            if not _matches_planned_reference(status.reference, planned):
                raise EvidenceV2Error("calibration target status path drifts from plan")
            targets.append(row)
        planned_sample_ids = [item["sample_id"] for item in plan["sample_plans"]]
        for label, rows in (
            ("branch", branches),
            ("cost", costs),
            ("target", targets),
        ):
            ids = [item["sample_id"] for item in rows]
            if ids != planned_sample_ids or len(ids) != len(set(ids)):
                raise EvidenceV2Error(f"calibration {label} statuses do not cover the plan")
        planned_folds = [
            (item["branch"], item["fold_id"]) for item in plan["lambda_fold_plans"]
        ]
        actual_folds = [(item["branch"], item["fold_id"]) for item in lambda_rows]
        if actual_folds != planned_folds or len(actual_folds) != len(set(actual_folds)):
            raise EvidenceV2Error("calibration lambda statuses do not cover the plan")
        return plan, schedule, branches, lambda_rows, costs, targets


def build_calibration_source_status_v3(
    *,
    evidence: CalibrationSourceStatusEvidenceBundleV3,
) -> dict[str, Any]:
    if not isinstance(evidence, CalibrationSourceStatusEvidenceBundleV3):
        raise EvidenceV2Error("calibration source status requires its typed evidence bundle")
    plan, _schedule, branches, lambdas, costs, targets = evidence.read()
    blocker_sources: list[dict[str, str]] = [
        {
            "blocker": "calibration_source_recomputation_incomplete",
            "source": "calibration_source_status",
        }
    ]
    for source_name, rows in (
        ("branch_status", branches),
        ("lambda_status", lambdas),
        ("cost_status", costs),
        ("target_status", targets),
    ):
        for row in rows:
            identity = str(row.get("sample_id") or row.get("fold_id") or "unknown")
            for blocker in row["blockers"]:
                blocker_sources.append(
                    {
                        "blocker": str(blocker),
                        "source": f"{source_name}:{identity}",
                    }
                )
    blocker_sources.sort(key=lambda item: (item["blocker"], item["source"]))
    blockers = sorted({item["blocker"] for item in blocker_sources})
    branch_refs = [item.status.reference.to_dict() for item in evidence.branch_statuses]
    lambda_refs = [item.reference.to_dict() for item, _sources in evidence.lambda_statuses]
    cost_refs = [item.reference.to_dict() for item, _sources in evidence.cost_statuses]
    target_refs = [item.reference.to_dict() for item, _sources in evidence.target_statuses]
    return seal_semantic(
        {
            "schema_version": CALIBRATION_SOURCE_STATUS_SCHEMA,
            "protocol_attempt_id": plan["protocol_attempt_id"],
            "epoch": plan["epoch"],
            "schedule_id": plan["schedule_id"],
            "plan_ref": evidence.plan.plan.reference.to_dict(),
            "schedule_ref": evidence.schedule_anchor.evidence.schedule.reference.to_dict(),
            "branch_status_refs": branch_refs,
            "lambda_status_refs": lambda_refs,
            "cost_status_refs": cost_refs,
            "target_status_refs": target_refs,
            "source_recomputation_complete": False,
            "blockers": blockers,
            "blocker_sources": blocker_sources,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_calibration_source_status_v3(
    value: Mapping[str, Any],
    *,
    evidence: CalibrationSourceStatusEvidenceBundleV3,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "plan_ref",
        "schedule_ref",
        "branch_status_refs",
        "lambda_status_refs",
        "cost_status_refs",
        "target_status_refs",
        "source_recomputation_complete",
        "blockers",
        "blocker_sources",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="calibration source status v3")
    if payload["schema_version"] != CALIBRATION_SOURCE_STATUS_SCHEMA:
        raise EvidenceV2Error("calibration source status v3 schema mismatch")
    rebuilt = build_calibration_source_status_v3(evidence=evidence)
    if rebuilt != payload:
        raise EvidenceV2Error("calibration source status v3 drifts from evidence bundle")
    return payload


__all__ = [
    "ALPHA_INTERVAL_REQUIREMENT",
    "FOLD_TRAINING_REQUIREMENT",
    "RESOLVER_ENTRYPOINTS",
    "RESOLVER_MODULE_SOURCE_SCHEMA",
    "BranchPredictionSourceEvidenceBundleV3",
    "BranchPredictionStatusBindingV3",
    "CalibrationSourceStatusEvidenceBundleV3",
    "LambdaFoldSourceEvidenceBundleV3",
    "ResolverImplementationEvidenceBundleV3",
    "RuntimeTrainingSourceBundleV3",
    "build_branch_prediction_source_status_v3",
    "build_calibration_source_status_v3",
    "build_lambda_fold_source_status_v3",
    "build_resolver_implementation_v3",
    "recompute_calibrated_probability_v3",
    "recompute_prior_base_rate_v3",
    "validate_branch_prediction_source_status_v3",
    "validate_calibration_source_status_v3",
    "validate_lambda_fold_source_status_v3",
    "validate_resolver_implementation_v3",
]
