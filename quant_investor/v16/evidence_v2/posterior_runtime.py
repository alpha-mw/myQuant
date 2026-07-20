"""Byte-bound runtime reconstruction for the disconnected v16 posterior lane."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.bayesian.v16.bootstrap import BlockBootstrapArtifact
from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.bayesian.v16.calibration import (
    CalibrationObservation,
    CalibrationStore,
)
from quant_investor.bayesian.v16.return_calibration import (
    ArtifactReturnCalibration,
    ROBUST_RETURN_MODEL_TYPE,
    RobustReturnModelArtifact,
)
from quant_investor.bayesian.v16.training import (
    BENCHMARK,
    HORIZON_DAYS,
    TARGET_DEFINITION,
    TrainingReceipt,
)
from quant_investor.bayesian.v16.types import CANONICAL_CORRELATION_KEYS, PriorSet

from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    decode_f64,
    encode_f64,
    seal_semantic,
    validate_semantic_seal,
)
from .runtime_identity import MODEL_BUNDLE_SCHEMA, validate_frozen_model_bundle

BASE_RATE_TRAINING_SCHEMA = "v16.base-rate-training-evidence.v2"
LIKELIHOOD_TRAINING_SCHEMA = "v16.likelihood-training-evidence.v2"
RETURN_MODEL_PARAMETERS_SCHEMA = "v16.return-model-parameters.v2"
RETURN_MODEL_TRAINING_SCHEMA = "v16.return-model-training-evidence.v2"
BOOTSTRAP_OFFSETS_SCHEMA = "v16.bootstrap-offsets.v2"
BOOTSTRAP_TRAINING_SCHEMA = "v16.bootstrap-training-evidence.v2"
CORRELATION_MATRIX_SCHEMA = "v16.correlation-matrix.v2"
CORRELATION_TRAINING_SCHEMA = "v16.correlation-training-evidence.v2"

PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"
MIN_RUNTIME_SAMPLES = 300
MIN_RUNTIME_COHORTS = 8


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not text or text != text.strip() or len(text) > 128:
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    if any(character not in allowed for character in text):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _private_ref(reference: EvidenceRef, *, schema: str | None = None) -> EvidenceRef:
    if not isinstance(reference, EvidenceRef):
        raise EvidenceV2Error("posterior runtime requires an EvidenceRef")
    if reference.root_policy != PRIVATE_ROOT_POLICY:
        raise EvidenceV2Error("posterior runtime artifacts must use the private root")
    if schema is not None and reference.artifact_schema != schema:
        raise EvidenceV2Error(f"posterior runtime artifact schema mismatch: expected {schema}")
    return reference


def _ordered_refs(value: Sequence[EvidenceRef], *, label: str) -> tuple[EvidenceRef, ...]:
    refs = tuple(value)
    if not refs:
        raise EvidenceV2Error(f"{label} must bind source evidence")
    for reference in refs:
        _private_ref(reference)
    keys = [(reference.absolute_path, reference.byte_sha256) for reference in refs]
    if len(keys) != len(set(keys)):
        raise EvidenceV2Error(f"{label} source references must be unique")
    return tuple(sorted(refs, key=lambda item: (item.absolute_path, item.byte_sha256)))


def _exact(value: Mapping[str, Any], fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error(f"{label} fields mismatch")
    return dict(value)


def _nonauthorizing(payload: Mapping[str, Any], *, label: str) -> None:
    if any(
        payload.get(field) is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error(f"{label} must be permanently nonauthorizing")


def _training_window(
    *,
    receipt_id: str,
    training_start: str,
    training_end: str,
    sample_count: int,
    embargo_days: int,
) -> None:
    try:
        TrainingReceipt(
            receipt_id=_safe_id(receipt_id, label="receipt_id"),
            evidence_sha256="0" * 64,
            training_start=str(training_start),
            training_end=str(training_end),
            sample_count=sample_count,
            purged=True,
            embargo_complete=True,
            embargo_days=embargo_days,
        )
    except ValueError as exc:
        raise EvidenceV2Error(str(exc)) from exc


def _receipt(payload: Mapping[str, Any], reference: EvidenceRef) -> TrainingReceipt:
    try:
        return TrainingReceipt(
            receipt_id=str(payload["receipt_id"]),
            evidence_sha256=reference.byte_sha256,
            training_start=str(payload["training_start"]),
            training_end=str(payload["training_end"]),
            sample_count=int(payload["sample_count"]),
            purged=True,
            embargo_complete=True,
            embargo_days=int(payload["embargo_days"]),
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceV2Error(str(exc)) from exc


def _training_projection(
    *,
    protocol_attempt_id: str,
    receipt_id: str,
    training_start: str,
    training_end: str,
    sample_count: int,
    embargo_days: int,
    source_input_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    if isinstance(sample_count, bool) or sample_count < MIN_RUNTIME_SAMPLES:
        raise EvidenceV2Error(
            f"posterior runtime training requires at least {MIN_RUNTIME_SAMPLES} samples"
        )
    if isinstance(embargo_days, bool):
        raise EvidenceV2Error("training embargo_days must be an integer")
    _training_window(
        receipt_id=receipt_id,
        training_start=training_start,
        training_end=training_end,
        sample_count=sample_count,
        embargo_days=embargo_days,
    )
    sources = _ordered_refs(source_input_refs, label="training evidence")
    return {
        "protocol_attempt_id": _safe_id(
            protocol_attempt_id,
            label="protocol_attempt_id",
        ),
        "receipt_id": _safe_id(receipt_id, label="receipt_id"),
        "training_start": str(training_start),
        "training_end": str(training_end),
        "sample_count": sample_count,
        "target_definition": TARGET_DEFINITION,
        "benchmark": BENCHMARK,
        "horizon_days": HORIZON_DAYS,
        "lookback_years": 5,
        "purged": True,
        "embargo_complete": True,
        "embargo_days": embargo_days,
        "source_input_refs": [reference.to_dict() for reference in sources],
        "activation_candidate": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
    }


_TRAINING_FIELDS = {
    "schema_version",
    "protocol_attempt_id",
    "receipt_id",
    "training_start",
    "training_end",
    "sample_count",
    "target_definition",
    "benchmark",
    "horizon_days",
    "lookback_years",
    "purged",
    "embargo_complete",
    "embargo_days",
    "source_input_refs",
    "activation_candidate",
    "new_risk_authorized",
    "production_apply_enabled",
    "semantic_sha256",
}


@dataclass(frozen=True)
class BaseRateObservation:
    sample_id: str
    positive_outcome: bool

    def __post_init__(self) -> None:
        _safe_id(self.sample_id, label="base-rate sample_id")
        if not isinstance(self.positive_outcome, bool):
            raise EvidenceV2Error("base-rate outcome must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "positive_outcome": self.positive_outcome,
        }


@dataclass(frozen=True)
class LikelihoodTrainingObservation:
    sample_id: str
    branch: str
    cohort_id: str
    score: float
    positive_outcome: bool

    def __post_init__(self) -> None:
        _safe_id(self.sample_id, label="likelihood sample_id")
        _safe_id(self.cohort_id, label="likelihood cohort_id")
        if self.branch not in CANONICAL_BRANCH_ORDER:
            raise EvidenceV2Error("likelihood observation branch is not formal v16")
        number = float(self.score)
        if not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise EvidenceV2Error("likelihood observation score is outside [-1, 1]")
        if not isinstance(self.positive_outcome, bool):
            raise EvidenceV2Error("likelihood outcome must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "branch": self.branch,
            "cohort_id": self.cohort_id,
            "score": encode_f64(self.score),
            "positive_outcome": self.positive_outcome,
        }


def build_base_rate_training_evidence(
    *,
    protocol_attempt_id: str,
    receipt_id: str,
    training_start: str,
    training_end: str,
    embargo_days: int,
    observations: Sequence[BaseRateObservation],
    source_input_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    rows = sorted(tuple(observations), key=lambda item: item.sample_id)
    if any(not isinstance(item, BaseRateObservation) for item in rows):
        raise EvidenceV2Error("base-rate rows must be BaseRateObservation values")
    ids = [item.sample_id for item in rows]
    if len(ids) != len(set(ids)):
        raise EvidenceV2Error("base-rate sample IDs must be unique")
    positives = sum(item.positive_outcome for item in rows)
    if positives == 0 or positives == len(rows):
        raise EvidenceV2Error("base-rate training requires both outcome classes")
    projection = _training_projection(
        protocol_attempt_id=protocol_attempt_id,
        receipt_id=receipt_id,
        training_start=training_start,
        training_end=training_end,
        sample_count=len(rows),
        embargo_days=embargo_days,
        source_input_refs=source_input_refs,
    )
    return seal_semantic(
        {
            "schema_version": BASE_RATE_TRAINING_SCHEMA,
            **projection,
            "observations": [item.to_dict() for item in rows],
        }
    )


def validate_base_rate_training_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = _TRAINING_FIELDS | {"observations"}
    _exact(payload, fields, label="base-rate training evidence")
    if payload["schema_version"] != BASE_RATE_TRAINING_SCHEMA:
        raise EvidenceV2Error("base-rate training schema mismatch")
    if not isinstance(payload["observations"], list):
        raise EvidenceV2Error("base-rate observations must be a list")
    rows = [
        BaseRateObservation(
            sample_id=str(
                _exact(item, {"sample_id", "positive_outcome"}, label="base row")["sample_id"]
            ),
            positive_outcome=_exact(
                item,
                {"sample_id", "positive_outcome"},
                label="base row",
            )["positive_outcome"],
        )
        for item in payload["observations"]
    ]
    rebuilt = build_base_rate_training_evidence(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        receipt_id=str(payload["receipt_id"]),
        training_start=str(payload["training_start"]),
        training_end=str(payload["training_end"]),
        embargo_days=payload["embargo_days"],
        observations=rows,
        source_input_refs=[EvidenceRef.from_dict(item) for item in payload["source_input_refs"]],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("base-rate training evidence is not canonical")
    return payload


def _likelihood_rows(
    observations: Sequence[LikelihoodTrainingObservation],
) -> tuple[LikelihoodTrainingObservation, ...]:
    rows = tuple(observations)
    if any(not isinstance(item, LikelihoodTrainingObservation) for item in rows):
        raise EvidenceV2Error("likelihood rows must be LikelihoodTrainingObservation values")
    rank = {branch: index for index, branch in enumerate(CANONICAL_BRANCH_ORDER)}
    rows = tuple(sorted(rows, key=lambda item: (rank[item.branch], item.sample_id)))
    by_branch = {
        branch: [item for item in rows if item.branch == branch]
        for branch in CANONICAL_BRANCH_ORDER
    }
    if any(len(items) < MIN_RUNTIME_SAMPLES for items in by_branch.values()):
        raise EvidenceV2Error(
            f"every likelihood branch requires at least {MIN_RUNTIME_SAMPLES} samples"
        )
    base = {
        item.sample_id: (item.cohort_id, item.positive_outcome)
        for item in by_branch[CANONICAL_BRANCH_ORDER[0]]
    }
    if len(base) != len(by_branch[CANONICAL_BRANCH_ORDER[0]]):
        raise EvidenceV2Error("likelihood branch sample IDs must be unique")
    for branch in CANONICAL_BRANCH_ORDER[1:]:
        projected = {
            item.sample_id: (item.cohort_id, item.positive_outcome) for item in by_branch[branch]
        }
        if len(projected) != len(by_branch[branch]) or projected != base:
            raise EvidenceV2Error(
                "likelihood branches must share the exact sample/cohort/outcome set"
            )
    if len({cohort for cohort, _outcome in base.values()}) < MIN_RUNTIME_COHORTS:
        raise EvidenceV2Error(
            f"likelihood training requires at least {MIN_RUNTIME_COHORTS} cohorts"
        )
    return rows


def build_likelihood_training_evidence(
    *,
    protocol_attempt_id: str,
    receipt_id: str,
    training_start: str,
    training_end: str,
    embargo_days: int,
    observations: Sequence[LikelihoodTrainingObservation],
    source_input_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    rows = _likelihood_rows(observations)
    per_branch = len(rows) // len(CANONICAL_BRANCH_ORDER)
    projection = _training_projection(
        protocol_attempt_id=protocol_attempt_id,
        receipt_id=receipt_id,
        training_start=training_start,
        training_end=training_end,
        sample_count=per_branch,
        embargo_days=embargo_days,
        source_input_refs=source_input_refs,
    )
    return seal_semantic(
        {
            "schema_version": LIKELIHOOD_TRAINING_SCHEMA,
            **projection,
            "minimum_samples_per_branch": MIN_RUNTIME_SAMPLES,
            "minimum_cohorts": MIN_RUNTIME_COHORTS,
            "beta_prior_alpha": encode_f64(1.0),
            "beta_prior_beta": encode_f64(1.0),
            "observations": [item.to_dict() for item in rows],
        }
    )


def validate_likelihood_training_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = _TRAINING_FIELDS | {
        "minimum_samples_per_branch",
        "minimum_cohorts",
        "beta_prior_alpha",
        "beta_prior_beta",
        "observations",
    }
    _exact(payload, fields, label="likelihood training evidence")
    if payload["schema_version"] != LIKELIHOOD_TRAINING_SCHEMA:
        raise EvidenceV2Error("likelihood training schema mismatch")
    if not isinstance(payload["observations"], list):
        raise EvidenceV2Error("likelihood observations must be a list")
    rows: list[LikelihoodTrainingObservation] = []
    for item in payload["observations"]:
        row = _exact(
            item,
            {"sample_id", "branch", "cohort_id", "score", "positive_outcome"},
            label="likelihood row",
        )
        rows.append(
            LikelihoodTrainingObservation(
                sample_id=str(row["sample_id"]),
                branch=str(row["branch"]),
                cohort_id=str(row["cohort_id"]),
                score=decode_f64(row["score"], label="likelihood score"),
                positive_outcome=row["positive_outcome"],
            )
        )
    rebuilt = build_likelihood_training_evidence(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        receipt_id=str(payload["receipt_id"]),
        training_start=str(payload["training_start"]),
        training_end=str(payload["training_end"]),
        embargo_days=payload["embargo_days"],
        observations=rows,
        source_input_refs=[EvidenceRef.from_dict(item) for item in payload["source_input_refs"]],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("likelihood training evidence is not canonical")
    return payload


def _build_training_manifest(
    *,
    schema: str,
    protocol_attempt_id: str,
    receipt_id: str,
    training_start: str,
    training_end: str,
    embargo_days: int,
    sample_ids: Sequence[str],
    source_input_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    ids = sorted(_safe_id(item, label="training sample_id") for item in sample_ids)
    if len(ids) != len(set(ids)):
        raise EvidenceV2Error("training manifest sample IDs must be unique")
    projection = _training_projection(
        protocol_attempt_id=protocol_attempt_id,
        receipt_id=receipt_id,
        training_start=training_start,
        training_end=training_end,
        sample_count=len(ids),
        embargo_days=embargo_days,
        source_input_refs=source_input_refs,
    )
    return seal_semantic(
        {
            "schema_version": schema,
            **projection,
            "sample_ids": ids,
        }
    )


def _validate_training_manifest(
    value: Mapping[str, Any],
    *,
    schema: str,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    _exact(payload, _TRAINING_FIELDS | {"sample_ids"}, label="training manifest")
    if payload["schema_version"] != schema or not isinstance(payload["sample_ids"], list):
        raise EvidenceV2Error("training manifest schema or samples mismatch")
    rebuilt = _build_training_manifest(
        schema=schema,
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        receipt_id=str(payload["receipt_id"]),
        training_start=str(payload["training_start"]),
        training_end=str(payload["training_end"]),
        embargo_days=payload["embargo_days"],
        sample_ids=[str(item) for item in payload["sample_ids"]],
        source_input_refs=[EvidenceRef.from_dict(item) for item in payload["source_input_refs"]],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("training manifest is not canonical")
    return payload


def build_return_model_training_evidence(**kwargs: Any) -> dict[str, Any]:
    return _build_training_manifest(schema=RETURN_MODEL_TRAINING_SCHEMA, **kwargs)


def validate_return_model_training_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_training_manifest(value, schema=RETURN_MODEL_TRAINING_SCHEMA)


def build_bootstrap_training_evidence(**kwargs: Any) -> dict[str, Any]:
    return _build_training_manifest(schema=BOOTSTRAP_TRAINING_SCHEMA, **kwargs)


def validate_bootstrap_training_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_training_manifest(value, schema=BOOTSTRAP_TRAINING_SCHEMA)


def build_correlation_training_evidence(**kwargs: Any) -> dict[str, Any]:
    return _build_training_manifest(schema=CORRELATION_TRAINING_SCHEMA, **kwargs)


def validate_correlation_training_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_training_manifest(value, schema=CORRELATION_TRAINING_SCHEMA)


def build_return_model_parameters(
    *,
    protocol_attempt_id: str,
    artifact_id: str,
    training_ref: EvidenceRef,
    intercept: float,
    aggregate_coefficient: float,
) -> dict[str, Any]:
    _private_ref(training_ref, schema=RETURN_MODEL_TRAINING_SCHEMA)
    intercept_value = float(intercept)
    coefficient = float(aggregate_coefficient)
    if not math.isfinite(intercept_value) or not math.isfinite(coefficient):
        raise EvidenceV2Error("return model parameters must be finite")
    if coefficient <= 0.0:
        raise EvidenceV2Error("return model aggregate coefficient must be positive")
    return seal_semantic(
        {
            "schema_version": RETURN_MODEL_PARAMETERS_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "artifact_id": _safe_id(artifact_id, label="return artifact_id"),
            "model_type": ROBUST_RETURN_MODEL_TYPE,
            "training_ref": training_ref.to_dict(),
            "intercept": encode_f64(intercept_value),
            "aggregate_coefficient": encode_f64(coefficient),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_return_model_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "artifact_id",
        "model_type",
        "training_ref",
        "intercept",
        "aggregate_coefficient",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    _exact(payload, fields, label="return model parameters")
    if payload["schema_version"] != RETURN_MODEL_PARAMETERS_SCHEMA:
        raise EvidenceV2Error("return model parameter schema mismatch")
    rebuilt = build_return_model_parameters(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        artifact_id=str(payload["artifact_id"]),
        training_ref=EvidenceRef.from_dict(payload["training_ref"]),
        intercept=decode_f64(payload["intercept"], label="return intercept"),
        aggregate_coefficient=decode_f64(
            payload["aggregate_coefficient"],
            label="return aggregate coefficient",
        ),
    )
    if rebuilt != payload:
        raise EvidenceV2Error("return model parameters are not canonical")
    return payload


def build_bootstrap_offsets(
    *,
    protocol_attempt_id: str,
    artifact_id: str,
    training_ref: EvidenceRef,
    block_length_days: int,
    block_count: int,
    win_rate_logit_offsets: Sequence[float],
    expected_alpha_offsets: Sequence[float],
) -> dict[str, Any]:
    _private_ref(training_ref, schema=BOOTSTRAP_TRAINING_SCHEMA)
    win = tuple(float(item) for item in win_rate_logit_offsets)
    alpha = tuple(float(item) for item in expected_alpha_offsets)
    try:
        BlockBootstrapArtifact(
            artifact_id=_safe_id(artifact_id, label="bootstrap artifact_id"),
            artifact_sha256="0" * 64,
            receipt=TrainingReceipt(
                receipt_id="bootstrap-shape-check",
                evidence_sha256="0" * 64,
                training_start="2021-01-01",
                training_end="2026-01-01",
                sample_count=MIN_RUNTIME_SAMPLES,
                purged=True,
                embargo_complete=True,
                embargo_days=HORIZON_DAYS,
            ),
            block_length_days=block_length_days,
            block_count=block_count,
            win_rate_logit_offsets=win,
            expected_alpha_offsets=alpha,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceV2Error(str(exc)) from exc
    return seal_semantic(
        {
            "schema_version": BOOTSTRAP_OFFSETS_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "artifact_id": _safe_id(artifact_id, label="bootstrap artifact_id"),
            "training_ref": training_ref.to_dict(),
            "block_length_days": block_length_days,
            "block_count": block_count,
            "win_rate_logit_offsets": [encode_f64(item) for item in win],
            "expected_alpha_offsets": [encode_f64(item) for item in alpha],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_bootstrap_offsets(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "artifact_id",
        "training_ref",
        "block_length_days",
        "block_count",
        "win_rate_logit_offsets",
        "expected_alpha_offsets",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    _exact(payload, fields, label="bootstrap offsets")
    if (
        payload["schema_version"] != BOOTSTRAP_OFFSETS_SCHEMA
        or not isinstance(payload["win_rate_logit_offsets"], list)
        or not isinstance(payload["expected_alpha_offsets"], list)
    ):
        raise EvidenceV2Error("bootstrap offset schema or arrays mismatch")
    rebuilt = build_bootstrap_offsets(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        artifact_id=str(payload["artifact_id"]),
        training_ref=EvidenceRef.from_dict(payload["training_ref"]),
        block_length_days=payload["block_length_days"],
        block_count=payload["block_count"],
        win_rate_logit_offsets=[
            decode_f64(item, label="win-rate bootstrap offset")
            for item in payload["win_rate_logit_offsets"]
        ],
        expected_alpha_offsets=[
            decode_f64(item, label="alpha bootstrap offset")
            for item in payload["expected_alpha_offsets"]
        ],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("bootstrap offsets are not canonical")
    return payload


def build_correlation_matrix(
    *,
    protocol_attempt_id: str,
    training_ref: EvidenceRef,
    correlations: Mapping[str, float],
) -> dict[str, Any]:
    _private_ref(training_ref, schema=CORRELATION_TRAINING_SCHEMA)
    if set(correlations) != set(CANONICAL_CORRELATION_KEYS):
        raise EvidenceV2Error("correlation matrix must contain the exact six branch pairs")
    normalized = {key: float(correlations[key]) for key in sorted(correlations)}
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in normalized.values()):
        raise EvidenceV2Error("correlation matrix values must be finite in [-1, 1]")
    return seal_semantic(
        {
            "schema_version": CORRELATION_MATRIX_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "training_ref": training_ref.to_dict(),
            "correlations": [
                {"pair": key, "value": encode_f64(value)} for key, value in normalized.items()
            ],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_correlation_matrix(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "training_ref",
        "correlations",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    _exact(payload, fields, label="correlation matrix")
    if payload["schema_version"] != CORRELATION_MATRIX_SCHEMA or not isinstance(
        payload["correlations"], list
    ):
        raise EvidenceV2Error("correlation matrix schema or rows mismatch")
    correlations: dict[str, float] = {}
    for item in payload["correlations"]:
        row = _exact(item, {"pair", "value"}, label="correlation row")
        key = str(row["pair"])
        if key in correlations:
            raise EvidenceV2Error("correlation pair is duplicated")
        correlations[key] = decode_f64(row["value"], label=f"correlation {key}")
    rebuilt = build_correlation_matrix(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        training_ref=EvidenceRef.from_dict(payload["training_ref"]),
        correlations=correlations,
    )
    if rebuilt != payload:
        raise EvidenceV2Error("correlation matrix is not canonical")
    return payload


@dataclass(frozen=True)
class PosteriorRuntimeArtifacts:
    model_bundles: tuple[tuple[str, BoundCanonicalArtifact], ...]
    prior_training: BoundCanonicalArtifact
    likelihood_training: BoundCanonicalArtifact
    return_model_parameters: BoundCanonicalArtifact
    return_model_training: BoundCanonicalArtifact
    bootstrap_offsets: BoundCanonicalArtifact
    bootstrap_training: BoundCanonicalArtifact
    correlation_matrix: BoundCanonicalArtifact
    correlation_training: BoundCanonicalArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.model_bundles, tuple):
            raise EvidenceV2Error("model bundle artifacts must be a tuple")
        for item in self.model_bundles:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], BoundCanonicalArtifact)
            ):
                raise EvidenceV2Error(
                    "model bundle entries require branch and BoundCanonicalArtifact"
                )
        for field in (
            "prior_training",
            "likelihood_training",
            "return_model_parameters",
            "return_model_training",
            "bootstrap_offsets",
            "bootstrap_training",
            "correlation_matrix",
            "correlation_training",
        ):
            if not isinstance(getattr(self, field), BoundCanonicalArtifact):
                raise EvidenceV2Error(
                    f"posterior runtime {field} must be an actual BoundCanonicalArtifact"
                )


@dataclass(frozen=True, init=False)
class PosteriorRuntimeBundle:
    protocol_attempt_id: str
    prior: PriorSet
    calibration_store: CalibrationStore
    return_calibration: ArtifactReturnCalibration
    bootstrap_artifact: BlockBootstrapArtifact
    correlation_matrix: tuple[tuple[str, float], ...]
    model_bundle_refs: tuple[tuple[str, EvidenceRef], ...]
    model_bundle_payloads: tuple[tuple[str, Mapping[str, Any]], ...]
    artifacts: PosteriorRuntimeArtifacts

    def __init__(self, *, artifacts: PosteriorRuntimeArtifacts) -> None:
        if not isinstance(artifacts, PosteriorRuntimeArtifacts):
            raise EvidenceV2Error("posterior runtime requires byte-bound artifacts")
        model_artifacts = tuple(artifacts.model_bundles)
        if tuple(branch for branch, _artifact in model_artifacts) != CANONICAL_BRANCH_ORDER:
            raise EvidenceV2Error("model bundle artifacts must use exact Q/F/M/LLM order")
        if len({id(artifact) for _branch, artifact in model_artifacts}) != len(model_artifacts):
            raise EvidenceV2Error("model bundle artifacts must be distinct")
        if len({artifact.reference.byte_sha256 for _branch, artifact in model_artifacts}) != len(
            model_artifacts
        ):
            raise EvidenceV2Error("model bundle byte identities must be distinct")

        model_payloads: list[tuple[str, Mapping[str, Any]]] = []
        attempts: set[str] = set()
        for branch, artifact in model_artifacts:
            _private_ref(artifact.reference, schema=MODEL_BUNDLE_SCHEMA)
            payload = validate_frozen_model_bundle(artifact.read())
            if payload["branch"] != branch:
                raise EvidenceV2Error("model bundle branch differs from artifact order")
            for field in (
                "training_schedule_ref",
                "training_capture_ref",
                "feature_contract_ref",
                "hyperparameter_ref",
                "serialized_model_ref",
            ):
                _private_ref(EvidenceRef.from_dict(payload[field]))
            provider = payload["llm_provider_build"]
            if isinstance(provider, Mapping):
                for field in (
                    "tokenizer_ref",
                    "inference_config_ref",
                    "provider_attestation_ref",
                ):
                    _private_ref(EvidenceRef.from_dict(provider[field]))
            attempts.add(str(payload["protocol_attempt_id"]))
            model_payloads.append((branch, payload))

        prior_payload = self._read(
            artifacts.prior_training,
            schema=BASE_RATE_TRAINING_SCHEMA,
            validator=validate_base_rate_training_evidence,
        )
        likelihood_payload = self._read(
            artifacts.likelihood_training,
            schema=LIKELIHOOD_TRAINING_SCHEMA,
            validator=validate_likelihood_training_evidence,
        )
        return_training = self._read(
            artifacts.return_model_training,
            schema=RETURN_MODEL_TRAINING_SCHEMA,
            validator=validate_return_model_training_evidence,
        )
        return_parameters = self._read(
            artifacts.return_model_parameters,
            schema=RETURN_MODEL_PARAMETERS_SCHEMA,
            validator=validate_return_model_parameters,
        )
        bootstrap_training = self._read(
            artifacts.bootstrap_training,
            schema=BOOTSTRAP_TRAINING_SCHEMA,
            validator=validate_bootstrap_training_evidence,
        )
        bootstrap_offsets = self._read(
            artifacts.bootstrap_offsets,
            schema=BOOTSTRAP_OFFSETS_SCHEMA,
            validator=validate_bootstrap_offsets,
        )
        correlation_training = self._read(
            artifacts.correlation_training,
            schema=CORRELATION_TRAINING_SCHEMA,
            validator=validate_correlation_training_evidence,
        )
        correlation_matrix = self._read(
            artifacts.correlation_matrix,
            schema=CORRELATION_MATRIX_SCHEMA,
            validator=validate_correlation_matrix,
        )
        runtime_payloads = (
            prior_payload,
            likelihood_payload,
            return_training,
            return_parameters,
            bootstrap_training,
            bootstrap_offsets,
            correlation_training,
            correlation_matrix,
        )
        attempts.update(str(payload["protocol_attempt_id"]) for payload in runtime_payloads)
        if len(attempts) != 1:
            raise EvidenceV2Error("posterior runtime artifacts cross protocol attempts")
        attempt = attempts.pop()

        if return_parameters["training_ref"] != artifacts.return_model_training.reference.to_dict():
            raise EvidenceV2Error("return parameters do not bind exact training bytes")
        if bootstrap_offsets["training_ref"] != artifacts.bootstrap_training.reference.to_dict():
            raise EvidenceV2Error("bootstrap offsets do not bind exact training bytes")
        if correlation_matrix["training_ref"] != artifacts.correlation_training.reference.to_dict():
            raise EvidenceV2Error("correlations do not bind exact training bytes")

        prior_rows = prior_payload["observations"]
        prior_receipt = _receipt(prior_payload, artifacts.prior_training.reference)
        prior = PriorSet(
            base_rate=sum(bool(item["positive_outcome"]) for item in prior_rows) / len(prior_rows),
            receipt=prior_receipt,
        )

        likelihood_receipt = _receipt(
            likelihood_payload,
            artifacts.likelihood_training.reference,
        )
        likelihood_rows = [
            CalibrationObservation(
                sample_id=str(item["sample_id"]),
                branch_name=str(item["branch"]),
                score=decode_f64(item["score"], label="likelihood score"),
                positive_outcome=item["positive_outcome"],
            )
            for item in likelihood_payload["observations"]
        ]
        try:
            calibration_store = CalibrationStore.from_training_evidence(
                likelihood_rows,
                receipt=likelihood_receipt,
                min_samples_per_branch=MIN_RUNTIME_SAMPLES,
                beta_prior_alpha=1.0,
                beta_prior_beta=1.0,
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceV2Error(str(exc)) from exc

        return_receipt = _receipt(return_training, artifacts.return_model_training.reference)
        return_calibration = ArtifactReturnCalibration(
            RobustReturnModelArtifact(
                artifact_id=str(return_parameters["artifact_id"]),
                parameters_sha256=artifacts.return_model_parameters.reference.byte_sha256,
                receipt=return_receipt,
                intercept=decode_f64(return_parameters["intercept"], label="return intercept"),
                aggregate_coefficient=decode_f64(
                    return_parameters["aggregate_coefficient"],
                    label="return coefficient",
                ),
            )
        )

        bootstrap_receipt = _receipt(
            bootstrap_training,
            artifacts.bootstrap_training.reference,
        )
        try:
            bootstrap_artifact = BlockBootstrapArtifact(
                artifact_id=str(bootstrap_offsets["artifact_id"]),
                artifact_sha256=artifacts.bootstrap_offsets.reference.byte_sha256,
                receipt=bootstrap_receipt,
                block_length_days=bootstrap_offsets["block_length_days"],
                block_count=bootstrap_offsets["block_count"],
                win_rate_logit_offsets=tuple(
                    decode_f64(item, label="win-rate bootstrap offset")
                    for item in bootstrap_offsets["win_rate_logit_offsets"]
                ),
                expected_alpha_offsets=tuple(
                    decode_f64(item, label="alpha bootstrap offset")
                    for item in bootstrap_offsets["expected_alpha_offsets"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceV2Error(str(exc)) from exc

        correlations = tuple(
            (str(item["pair"]), decode_f64(item["value"], label="correlation"))
            for item in correlation_matrix["correlations"]
        )
        object.__setattr__(self, "protocol_attempt_id", attempt)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "calibration_store", calibration_store)
        object.__setattr__(self, "return_calibration", return_calibration)
        object.__setattr__(self, "bootstrap_artifact", bootstrap_artifact)
        object.__setattr__(self, "correlation_matrix", correlations)
        object.__setattr__(
            self,
            "model_bundle_refs",
            tuple((branch, artifact.reference) for branch, artifact in model_artifacts),
        )
        object.__setattr__(self, "model_bundle_payloads", tuple(model_payloads))
        object.__setattr__(self, "artifacts", artifacts)

    @staticmethod
    def _read(
        artifact: BoundCanonicalArtifact,
        *,
        schema: str,
        validator: Any,
    ) -> dict[str, Any]:
        if not isinstance(artifact, BoundCanonicalArtifact):
            raise EvidenceV2Error("posterior runtime input is not a BoundCanonicalArtifact")
        _private_ref(artifact.reference, schema=schema)
        return validator(artifact.read())

    @property
    def model_refs(self) -> dict[str, EvidenceRef]:
        return dict(self.model_bundle_refs)

    @property
    def model_payloads(self) -> dict[str, Mapping[str, Any]]:
        return dict(self.model_bundle_payloads)

    @property
    def correlations(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.correlation_matrix}

    def refs_projection(self) -> dict[str, Any]:
        return {
            "model_bundle_refs": {
                branch: reference.to_dict() for branch, reference in self.model_bundle_refs
            },
            "prior_training_ref": self.artifacts.prior_training.reference.to_dict(),
            "likelihood_training_ref": self.artifacts.likelihood_training.reference.to_dict(),
            "return_model_parameters_ref": (
                self.artifacts.return_model_parameters.reference.to_dict()
            ),
            "return_model_training_ref": self.artifacts.return_model_training.reference.to_dict(),
            "bootstrap_offsets_ref": self.artifacts.bootstrap_offsets.reference.to_dict(),
            "bootstrap_training_ref": self.artifacts.bootstrap_training.reference.to_dict(),
            "correlation_matrix_ref": self.artifacts.correlation_matrix.reference.to_dict(),
            "correlation_training_ref": self.artifacts.correlation_training.reference.to_dict(),
        }


__all__ = [
    "BASE_RATE_TRAINING_SCHEMA",
    "BOOTSTRAP_OFFSETS_SCHEMA",
    "BOOTSTRAP_TRAINING_SCHEMA",
    "CORRELATION_MATRIX_SCHEMA",
    "CORRELATION_TRAINING_SCHEMA",
    "LIKELIHOOD_TRAINING_SCHEMA",
    "MIN_RUNTIME_COHORTS",
    "MIN_RUNTIME_SAMPLES",
    "RETURN_MODEL_PARAMETERS_SCHEMA",
    "RETURN_MODEL_TRAINING_SCHEMA",
    "BaseRateObservation",
    "LikelihoodTrainingObservation",
    "PosteriorRuntimeArtifacts",
    "PosteriorRuntimeBundle",
    "build_base_rate_training_evidence",
    "build_bootstrap_offsets",
    "build_bootstrap_training_evidence",
    "build_correlation_matrix",
    "build_correlation_training_evidence",
    "build_likelihood_training_evidence",
    "build_return_model_parameters",
    "build_return_model_training_evidence",
    "validate_base_rate_training_evidence",
    "validate_bootstrap_offsets",
    "validate_bootstrap_training_evidence",
    "validate_correlation_matrix",
    "validate_correlation_training_evidence",
    "validate_likelihood_training_evidence",
    "validate_return_model_parameters",
    "validate_return_model_training_evidence",
]
