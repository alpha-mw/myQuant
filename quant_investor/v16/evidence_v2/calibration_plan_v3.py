"""Pre-s0 calibration source plan for the disconnected v16 evidence lane.

The plan fixes source and future-artifact identities before observation.  It
contains no prediction, lambda, cost, outcome, readiness, or authorization
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import posixpath
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)

CALIBRATION_PLAN_V3_SCHEMA = "v16.calibration-universe-plan.v3"
RESOLVER_IMPLEMENTATION_SCHEMA = "v16.calibration-resolver-implementation.v3"
BRANCH_STATUS_SCHEMA = "v16.branch-calibration-prediction-source-status.v3"
LAMBDA_STATUS_SCHEMA = "v16.lambda-fold-source-status.v3"
COST_STATUS_SCHEMA = "v16.mark-cost-source-status.v3"
TARGET_STATUS_SCHEMA = "v16.mark-target-source-status.v3"
CALIBRATION_SOURCE_STATUS_SCHEMA = "v16.prospective-calibration-source-status.v3"
SCHEDULE_V4_SCHEMA = "v16.evidence-schedule-declaration.v4"
READINESS_V3_SCHEMA = "v16_run_readiness.v3"
STOCK_SOURCE_SET_SCHEMA = "v16.stock-mark-source-set.v3"
PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"
FORMAL_BRANCHES = ("quant", "fundamental", "macro", "llm")
SUPPORTED_REQUIREMENTS = ("prior_probability", "branch_probability")
UNSUPPORTED_REQUIREMENTS = (
    "branch_only_alpha_interval_model",
    "fold_training_algorithm",
    "eight_component_cost_model",
)
MIN_BRANCH_SAMPLES = 300
MIN_BRANCH_COHORTS = 8
MODEL_BUNDLE_SCHEMA = "v16.frozen-model-bundle.v2"
RUNTIME_CAPSULE_SCHEMA = "v16.hermetic-runtime-capsule.v2"
INDEX_MANIFEST_SCHEMA = "csi-index-total-return-manifest.v1"
STOCK_MARK_EVIDENCE_SCHEMA = "v16.stock-mark-evidence.v2"
TIMESTAMP_ATTEMPT_SCHEMA = "v16.rfc3161-attempt-state.v2"
TIMESTAMP_RECEIPT_SCHEMA = "v16.rfc3161-validation-receipt.v2"

POSTERIOR_RUNTIME_SCHEMAS = {
    "prior_training": "v16.base-rate-training-evidence.v2",
    "likelihood_training": "v16.likelihood-training-evidence.v2",
    "return_model_parameters": "v16.return-model-parameters.v2",
    "return_model_training": "v16.return-model-training-evidence.v2",
    "bootstrap_offsets": "v16.bootstrap-offsets.v2",
    "bootstrap_training": "v16.bootstrap-training-evidence.v2",
    "correlation_matrix": "v16.correlation-matrix.v2",
    "correlation_training": "v16.correlation-training-evidence.v2",
}

SAMPLE_ARTIFACT_SCHEMAS = {
    "stage1_request": "codex-review-stage1-request.v1",
    "stage1_response": "codex-review-stage1-response.v1",
    "branch_status": BRANCH_STATUS_SCHEMA,
    "prediction_timestamp_attempt": TIMESTAMP_ATTEMPT_SCHEMA,
    "prediction_timestamp_receipt": TIMESTAMP_RECEIPT_SCHEMA,
    "stock_marks": STOCK_MARK_EVIDENCE_SCHEMA,
    "cost_status": COST_STATUS_SCHEMA,
    "target_status": TARGET_STATUS_SCHEMA,
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


def _symbol(value: Any) -> str:
    text = str(value or "")
    if (
        not text
        or text != text.strip().upper()
        or len(text) > 32
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in text
        )
    ):
        raise EvidenceV2Error("calibration plan symbol is not normalized")
    return text


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        normalized = date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be an ISO date") from exc
    if text != normalized:
        raise EvidenceV2Error(f"{label} must be a canonical ISO date")
    return text


def _private_root(value: Any) -> str:
    text = str(value or "")
    if (
        not text.startswith("/")
        or text.startswith("//")
        or text.endswith("/")
        or "\x00" in text
        or posixpath.normpath(text) != text
    ):
        raise EvidenceV2Error("calibration private root must be canonical and absolute")
    return text


@dataclass(frozen=True)
class PlannedArtifactPath:
    absolute_path: str
    artifact_schema: str
    root_policy: str = PRIVATE_ROOT_POLICY

    def validate_under(self, private_root: str) -> None:
        root = _private_root(private_root)
        path = str(self.absolute_path or "")
        if (
            not path.startswith("/")
            or path.startswith("//")
            or path.endswith("/")
            or "\x00" in path
            or posixpath.normpath(path) != path
        ):
            raise EvidenceV2Error("planned artifact path must be canonical and absolute")
        try:
            common = posixpath.commonpath((root, path))
        except ValueError as exc:
            raise EvidenceV2Error("planned artifact path is outside the private root") from exc
        if common != root or path == root:
            raise EvidenceV2Error("planned artifact path must be a strict private-root child")
        if not self.artifact_schema or self.root_policy != PRIVATE_ROOT_POLICY:
            raise EvidenceV2Error("planned artifact schema/root policy is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "absolute_path": self.absolute_path,
            "artifact_schema": self.artifact_schema,
            "root_policy": self.root_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannedArtifactPath":
        payload = _exact(
            value,
            {"absolute_path", "artifact_schema", "root_policy"},
            label="planned artifact path",
        )
        return cls(**{key: str(payload[key]) for key in payload})


@dataclass(frozen=True)
class CalibrationSamplePlanV3:
    sample_id: str
    branch: str
    symbol: str
    cohort_id: str
    cohort_start_date: str
    cohort_end_date: str
    slot_id: str
    artifacts: tuple[tuple[str, PlannedArtifactPath], ...]
    stock_source_set_ref: EvidenceRef
    benchmark_manifest_ref: EvidenceRef
    cost_source_refs: tuple[EvidenceRef, ...] = ()

    def normalized(self, *, private_root: str) -> dict[str, Any]:
        sample_id = _identifier(self.sample_id, label="sample_id")
        branch = str(self.branch)
        if branch not in FORMAL_BRANCHES:
            raise EvidenceV2Error("calibration sample branch is not formal v16")
        start = _iso_date(self.cohort_start_date, label="cohort_start_date")
        end = _iso_date(self.cohort_end_date, label="cohort_end_date")
        if end < start:
            raise EvidenceV2Error("calibration sample cohort range is reversed")
        artifact_map = dict(self.artifacts)
        if len(artifact_map) != len(self.artifacts) or set(artifact_map) != set(
            SAMPLE_ARTIFACT_SCHEMAS
        ):
            raise EvidenceV2Error("calibration sample planned artifact keys mismatch")
        for key, expected_schema in SAMPLE_ARTIFACT_SCHEMAS.items():
            artifact = artifact_map[key]
            if not isinstance(artifact, PlannedArtifactPath):
                raise EvidenceV2Error("calibration sample artifact has the wrong type")
            artifact.validate_under(private_root)
            if artifact.artifact_schema != expected_schema:
                raise EvidenceV2Error(
                    f"calibration sample artifact schema mismatch: {key}"
                )
        if (
            self.stock_source_set_ref.artifact_schema != STOCK_SOURCE_SET_SCHEMA
            or self.stock_source_set_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("calibration stock source-set ref is invalid")
        if (
            self.benchmark_manifest_ref.artifact_schema != INDEX_MANIFEST_SCHEMA
            or self.benchmark_manifest_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("calibration benchmark manifest ref is invalid")
        costs = tuple(self.cost_source_refs)
        if any(item.root_policy != PRIVATE_ROOT_POLICY for item in costs):
            raise EvidenceV2Error("calibration cost source refs must use the private root")
        if len({item.byte_sha256 for item in costs}) != len(costs):
            raise EvidenceV2Error("calibration cost source refs must be distinct")
        return {
            "sample_id": sample_id,
            "branch": branch,
            "symbol": _symbol(self.symbol),
            "cohort_id": _identifier(self.cohort_id, label="cohort_id"),
            "cohort_start_date": start,
            "cohort_end_date": end,
            "slot_id": _identifier(self.slot_id, label="slot_id"),
            "artifacts": {
                key: artifact_map[key].to_dict() for key in SAMPLE_ARTIFACT_SCHEMAS
            },
            "stock_source_set_ref": self.stock_source_set_ref.to_dict(),
            "benchmark_manifest_ref": self.benchmark_manifest_ref.to_dict(),
            "cost_source_refs": [item.to_dict() for item in costs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationSamplePlanV3":
        payload = _exact(
            value,
            {
                "sample_id",
                "branch",
                "symbol",
                "cohort_id",
                "cohort_start_date",
                "cohort_end_date",
                "slot_id",
                "artifacts",
                "stock_source_set_ref",
                "benchmark_manifest_ref",
                "cost_source_refs",
            },
            label="calibration sample plan v3",
        )
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, Mapping):
            raise EvidenceV2Error("calibration sample artifacts must be an object")
        if not isinstance(payload["cost_source_refs"], list):
            raise EvidenceV2Error("calibration sample cost_source_refs must be a list")
        return cls(
            sample_id=str(payload["sample_id"]),
            branch=str(payload["branch"]),
            symbol=str(payload["symbol"]),
            cohort_id=str(payload["cohort_id"]),
            cohort_start_date=str(payload["cohort_start_date"]),
            cohort_end_date=str(payload["cohort_end_date"]),
            slot_id=str(payload["slot_id"]),
            artifacts=tuple(
                (key, PlannedArtifactPath.from_dict(artifacts[key]))
                for key in SAMPLE_ARTIFACT_SCHEMAS
            ),
            stock_source_set_ref=EvidenceRef.from_dict(payload["stock_source_set_ref"]),
            benchmark_manifest_ref=EvidenceRef.from_dict(
                payload["benchmark_manifest_ref"]
            ),
            cost_source_refs=tuple(
                EvidenceRef.from_dict(item) for item in payload["cost_source_refs"]
            ),
        )


@dataclass(frozen=True)
class LambdaFoldPlanV3:
    branch: str
    fold_id: str
    training_source_refs: tuple[EvidenceRef, ...]
    status_artifact: PlannedArtifactPath

    def normalized(self, *, private_root: str) -> dict[str, Any]:
        if self.branch not in FORMAL_BRANCHES:
            raise EvidenceV2Error("lambda fold branch is not formal v16")
        sources = tuple(self.training_source_refs)
        if not sources or len({item.byte_sha256 for item in sources}) != len(sources):
            raise EvidenceV2Error("lambda fold requires distinct training source refs")
        if any(item.root_policy != PRIVATE_ROOT_POLICY for item in sources):
            raise EvidenceV2Error("lambda fold training refs must be private")
        self.status_artifact.validate_under(private_root)
        if self.status_artifact.artifact_schema != LAMBDA_STATUS_SCHEMA:
            raise EvidenceV2Error("lambda fold status artifact schema mismatch")
        return {
            "branch": self.branch,
            "fold_id": _identifier(self.fold_id, label="fold_id"),
            "training_source_refs": [item.to_dict() for item in sources],
            "status_artifact": self.status_artifact.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LambdaFoldPlanV3":
        payload = _exact(
            value,
            {"branch", "fold_id", "training_source_refs", "status_artifact"},
            label="lambda fold plan v3",
        )
        if not isinstance(payload["training_source_refs"], list):
            raise EvidenceV2Error("lambda fold training refs must be a list")
        return cls(
            branch=str(payload["branch"]),
            fold_id=str(payload["fold_id"]),
            training_source_refs=tuple(
                EvidenceRef.from_dict(item) for item in payload["training_source_refs"]
            ),
            status_artifact=PlannedArtifactPath.from_dict(payload["status_artifact"]),
        )


def _reference_map(
    value: Mapping[str, EvidenceRef],
    *,
    keys: Sequence[str],
    schemas: Mapping[str, str],
    label: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or list(value) != list(keys):
        raise EvidenceV2Error(f"{label} keys/order mismatch")
    result: dict[str, dict[str, str]] = {}
    for key in keys:
        reference = value[key]
        if (
            not isinstance(reference, EvidenceRef)
            or reference.artifact_schema != schemas[key]
            or reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error(f"{label} ref is invalid: {key}")
        result[key] = reference.to_dict()
    return result


def _validate_sample_sets(rows: Sequence[dict[str, Any]]) -> None:
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise EvidenceV2Error("calibration sample IDs must be globally unique")
    by_branch = {
        branch: [row for row in rows if row["branch"] == branch]
        for branch in FORMAL_BRANCHES
    }
    if any(len(items) < MIN_BRANCH_SAMPLES for items in by_branch.values()):
        raise EvidenceV2Error(
            f"every calibration branch requires at least {MIN_BRANCH_SAMPLES} samples"
        )
    reference_keys = {
        branch: {(item["slot_id"], item["symbol"]) for item in items}
        for branch, items in by_branch.items()
    }
    if any(
        reference_keys[branch] != reference_keys[FORMAL_BRANCHES[0]]
        for branch in FORMAL_BRANCHES[1:]
    ):
        raise EvidenceV2Error("calibration branches must share the slot/symbol universe")
    for branch, items in by_branch.items():
        if len(reference_keys[branch]) != len(items):
            raise EvidenceV2Error("calibration branch repeats a slot/symbol sample")
        windows: dict[str, tuple[str, str]] = {}
        for item in items:
            window = (item["cohort_start_date"], item["cohort_end_date"])
            prior = windows.setdefault(item["cohort_id"], window)
            if prior != window:
                raise EvidenceV2Error("calibration cohort window drifts within a branch")
        if len(windows) < MIN_BRANCH_COHORTS:
            raise EvidenceV2Error(
                f"branch {branch} requires at least {MIN_BRANCH_COHORTS} cohorts"
            )
        ordered = sorted((start, end, cohort) for cohort, (start, end) in windows.items())
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] <= previous[1]:
                raise EvidenceV2Error("calibration cohort windows overlap")


def _validate_future_paths(
    rows: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
) -> None:
    paths = [
        artifact["absolute_path"]
        for row in rows
        for artifact in row["artifacts"].values()
    ] + [fold["status_artifact"]["absolute_path"] for fold in folds]
    if len(paths) != len(set(paths)):
        raise EvidenceV2Error("all planned future artifact paths must be globally unique")


def build_calibration_universe_plan_v3(
    *,
    protocol_attempt_id: str,
    epoch: str,
    schedule_id: str,
    private_root: str,
    runtime_capsule_ref: EvidenceRef,
    resolver_implementation_refs: Mapping[str, EvidenceRef],
    model_bundle_refs: Mapping[str, EvidenceRef],
    posterior_runtime_refs: Mapping[str, EvidenceRef],
    sample_plans: Sequence[CalibrationSamplePlanV3],
    lambda_fold_plans: Sequence[LambdaFoldPlanV3],
) -> dict[str, Any]:
    epoch_name = str(epoch)
    if epoch_name not in {"B", "C"}:
        raise EvidenceV2Error("calibration plan v3 epoch must be B or C")
    root = _private_root(private_root)
    if (
        runtime_capsule_ref.artifact_schema != RUNTIME_CAPSULE_SCHEMA
        or runtime_capsule_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise EvidenceV2Error("calibration plan runtime capsule ref is invalid")
    resolver_refs = _reference_map(
        resolver_implementation_refs,
        keys=SUPPORTED_REQUIREMENTS,
        schemas={key: RESOLVER_IMPLEMENTATION_SCHEMA for key in SUPPORTED_REQUIREMENTS},
        label="calibration resolver implementation",
    )
    model_refs = _reference_map(
        model_bundle_refs,
        keys=FORMAL_BRANCHES,
        schemas={branch: MODEL_BUNDLE_SCHEMA for branch in FORMAL_BRANCHES},
        label="calibration model bundle",
    )
    runtime_refs = _reference_map(
        posterior_runtime_refs,
        keys=tuple(POSTERIOR_RUNTIME_SCHEMAS),
        schemas=POSTERIOR_RUNTIME_SCHEMAS,
        label="calibration posterior runtime",
    )
    if any(not isinstance(item, CalibrationSamplePlanV3) for item in sample_plans):
        raise EvidenceV2Error("sample plans must be CalibrationSamplePlanV3 values")
    rows = [item.normalized(private_root=root) for item in sample_plans]
    if any(not isinstance(item, LambdaFoldPlanV3) for item in lambda_fold_plans):
        raise EvidenceV2Error("lambda plans must be LambdaFoldPlanV3 values")
    folds = [item.normalized(private_root=root) for item in lambda_fold_plans]
    _validate_sample_sets(rows)
    fold_keys = [(item["branch"], item["fold_id"]) for item in folds]
    if len(fold_keys) != len(set(fold_keys)):
        raise EvidenceV2Error("lambda fold branch/ID pairs must be unique")
    if any(sum(item["branch"] == branch for item in folds) < 2 for branch in FORMAL_BRANCHES):
        raise EvidenceV2Error("every branch requires at least two predeclared lambda folds")
    _validate_future_paths(rows, folds)
    rank = {branch: index for index, branch in enumerate(FORMAL_BRANCHES)}
    rows.sort(key=lambda item: (rank[item["branch"]], item["sample_id"]))
    folds.sort(key=lambda item: (rank[item["branch"]], item["fold_id"]))
    return seal_semantic(
        {
            "schema_version": CALIBRATION_PLAN_V3_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "epoch": epoch_name,
            "schedule_id": _identifier(schedule_id, label="schedule_id"),
            "private_root": root,
            "runtime_capsule_ref": runtime_capsule_ref.to_dict(),
            "resolver_implementation_refs": resolver_refs,
            "unsupported_requirement_ids": list(UNSUPPORTED_REQUIREMENTS),
            "model_bundle_refs": model_refs,
            "posterior_runtime_refs": runtime_refs,
            "sample_plans": rows,
            "lambda_fold_plans": folds,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_calibration_universe_plan_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "private_root",
        "runtime_capsule_ref",
        "resolver_implementation_refs",
        "unsupported_requirement_ids",
        "model_bundle_refs",
        "posterior_runtime_refs",
        "sample_plans",
        "lambda_fold_plans",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    payload = _exact(payload, fields, label="calibration universe plan v3")
    if payload["schema_version"] != CALIBRATION_PLAN_V3_SCHEMA:
        raise EvidenceV2Error("calibration universe plan v3 schema mismatch")
    for field in (
        "resolver_implementation_refs",
        "model_bundle_refs",
        "posterior_runtime_refs",
    ):
        if not isinstance(payload[field], Mapping):
            raise EvidenceV2Error(f"calibration plan {field} must be an object")
    for field in ("sample_plans", "lambda_fold_plans"):
        if not isinstance(payload[field], list):
            raise EvidenceV2Error(f"calibration plan {field} must be a list")
    rebuilt = build_calibration_universe_plan_v3(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        epoch=str(payload["epoch"]),
        schedule_id=str(payload["schedule_id"]),
        private_root=str(payload["private_root"]),
        runtime_capsule_ref=EvidenceRef.from_dict(payload["runtime_capsule_ref"]),
        resolver_implementation_refs={
            key: EvidenceRef.from_dict(payload["resolver_implementation_refs"][key])
            for key in SUPPORTED_REQUIREMENTS
        },
        model_bundle_refs={
            branch: EvidenceRef.from_dict(payload["model_bundle_refs"][branch])
            for branch in FORMAL_BRANCHES
        },
        posterior_runtime_refs={
            key: EvidenceRef.from_dict(payload["posterior_runtime_refs"][key])
            for key in POSTERIOR_RUNTIME_SCHEMAS
        },
        sample_plans=[CalibrationSamplePlanV3.from_dict(item) for item in payload["sample_plans"]],
        lambda_fold_plans=[
            LambdaFoldPlanV3.from_dict(item) for item in payload["lambda_fold_plans"]
        ],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("calibration universe plan v3 is not canonical")
    return payload


@dataclass(frozen=True)
class CalibrationPlanEvidenceBundleV3:
    plan: BoundCanonicalArtifact
    resolver_implementations: tuple[BoundCanonicalArtifact, ...]
    posterior_runtime_artifacts: tuple[BoundCanonicalArtifact, ...]

    def read(self) -> dict[str, Any]:
        if (
            self.plan.reference.artifact_schema != CALIBRATION_PLAN_V3_SCHEMA
            or self.plan.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("calibration plan evidence ref is invalid")
        payload = validate_calibration_universe_plan_v3(self.plan.read())
        expected_resolvers = [
            payload["resolver_implementation_refs"][key]
            for key in SUPPORTED_REQUIREMENTS
        ]
        if len(expected_resolvers) != len(self.resolver_implementations):
            raise EvidenceV2Error("calibration resolver evidence count drift")
        for expected, artifact in zip(expected_resolvers, self.resolver_implementations):
            if expected != artifact.reference.to_dict():
                raise EvidenceV2Error("calibration resolver evidence ref drift")
            artifact.read()
        expected_runtime = [
            payload["posterior_runtime_refs"][key] for key in POSTERIOR_RUNTIME_SCHEMAS
        ]
        if len(expected_runtime) != len(self.posterior_runtime_artifacts):
            raise EvidenceV2Error("calibration runtime evidence count drift")
        for expected, artifact in zip(expected_runtime, self.posterior_runtime_artifacts):
            if expected != artifact.reference.to_dict():
                raise EvidenceV2Error("calibration runtime evidence ref drift")
            artifact.read()
        return payload


__all__ = [
    "BRANCH_STATUS_SCHEMA",
    "CALIBRATION_PLAN_V3_SCHEMA",
    "CALIBRATION_SOURCE_STATUS_SCHEMA",
    "COST_STATUS_SCHEMA",
    "FORMAL_BRANCHES",
    "LAMBDA_STATUS_SCHEMA",
    "MIN_BRANCH_COHORTS",
    "MIN_BRANCH_SAMPLES",
    "POSTERIOR_RUNTIME_SCHEMAS",
    "PRIVATE_ROOT_POLICY",
    "READINESS_V3_SCHEMA",
    "RESOLVER_IMPLEMENTATION_SCHEMA",
    "SAMPLE_ARTIFACT_SCHEMAS",
    "SCHEDULE_V4_SCHEMA",
    "STOCK_SOURCE_SET_SCHEMA",
    "SUPPORTED_REQUIREMENTS",
    "TARGET_STATUS_SCHEMA",
    "UNSUPPORTED_REQUIREMENTS",
    "CalibrationPlanEvidenceBundleV3",
    "CalibrationSamplePlanV3",
    "LambdaFoldPlanV3",
    "PlannedArtifactPath",
    "build_calibration_universe_plan_v3",
    "validate_calibration_universe_plan_v3",
]
