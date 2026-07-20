"""Deterministic prospective calibration statistics for evidence-v2.

The functions in this module recompute every reported metric from sealed
prediction/outcome references.  They do not authorize activation or expose a
way to inject precomputed readiness metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import hmac
import math
import posixpath
from collections.abc import Mapping, Sequence
from typing import Any

from scipy import stats

from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    decode_f64,
    encode_f64,
    seal_semantic,
    validate_semantic_seal,
)
from .target import (
    COST_EVIDENCE_SCHEMA,
    STOCK_MARK_EVIDENCE_SCHEMA,
    TARGET_OUTCOME_SCHEMA,
    MarkTargetEvidenceBundle,
    ValidatedMarkTargetCommonEvidence,
    ValidatedStockMarkSources,
    prepare_mark_target_common_evidence,
    prepare_stock_mark_sources,
    validate_mark_target_outcome_from_common_evidence,
)
from .timestamp import (
    TIMESTAMP_ATTEMPT_SCHEMA,
    TIMESTAMP_RECEIPT_SCHEMA,
    TimestampAnchorBinding,
)
from .runtime_identity import MODEL_BUNDLE_SCHEMA

CALIBRATION_EVIDENCE_SCHEMA = "v16.prospective-calibration-evidence.v2"
BRANCH_PREDICTION_SCHEMA = "v16.branch-prediction.v2"
LAMBDA_FOLD_SCHEMA = "v16.lambda-fold-evidence.v2"
CALIBRATION_UNIVERSE_SCHEMA = "v16.calibration-universe-plan.v2"
CALIBRATION_BRANCHES = ("quant", "fundamental", "macro", "llm")
MIN_BRANCH_SAMPLES = 300
MIN_BRANCH_COHORTS = 8
MIN_COHORT_SAMPLES = 5
ECE_BIN_COUNT = 5
MAX_ECE = 0.05
MIN_INTERVAL_COVERAGE = 0.85
MAX_INTERVAL_COVERAGE = 0.95
MAX_LAMBDA_FOLD_RANGE = 0.20
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_ONE_SIDED_ALPHA = 0.05
FACTOR_B_FAMILY_ALPHA = 0.05
FACTOR_B_BH_Q = 0.10
POST_ACTIVATION_HYPOTHESIS_COUNT = 15
POST_ACTIVATION_ALPHA = 0.05 / POST_ACTIVATION_HYPOTHESIS_COUNT
_PROBABILITY_CLIP = 1e-6
_PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not text or text != text.strip() or len(text) > 128:
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    if any(character not in allowed for character in text):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != text:
        raise EvidenceV2Error(f"{label} must be a canonical ISO date")
    return text


def _finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceV2Error(f"{label} must be finite")
    return number


def _probability(value: Any, *, label: str) -> float:
    number = _finite(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise EvidenceV2Error(f"{label} must be in [0, 1]")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise EvidenceV2Error("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _symbol(value: Any) -> str:
    text = str(value or "")
    if (
        not text
        or text != text.strip().upper()
        or len(text) > 32
        or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in text)
    ):
        raise EvidenceV2Error("calibration universe symbol must be normalized")
    return text


def _absolute_artifact_path(value: Any, *, label: str) -> str:
    text = str(value or "")
    if (
        not text.startswith("/")
        or "\x00" in text
        or posixpath.normpath(text) != text
        or text.startswith("//")
        or text.endswith("/")
    ):
        raise EvidenceV2Error(f"{label} must be a canonical absolute artifact path")
    return text


_SAMPLE_PLAN_PATH_SCHEMAS = {
    "prediction_path": BRANCH_PREDICTION_SCHEMA,
    "outcome_path": TARGET_OUTCOME_SCHEMA,
    "stock_marks_path": STOCK_MARK_EVIDENCE_SCHEMA,
    "costs_path": COST_EVIDENCE_SCHEMA,
    "prediction_timestamp_attempt_path": TIMESTAMP_ATTEMPT_SCHEMA,
    "prediction_timestamp_receipt_path": TIMESTAMP_RECEIPT_SCHEMA,
}
_SAMPLE_PLAN_FIELDS = {
    "sample_id",
    "branch",
    "symbol",
    "cohort_id",
    "slot_id",
    *_SAMPLE_PLAN_PATH_SCHEMAS,
}


def _normalize_sample_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SAMPLE_PLAN_FIELDS:
        raise EvidenceV2Error("calibration universe sample plan fields mismatch")
    branch = str(value["branch"])
    if branch not in CALIBRATION_BRANCHES:
        raise EvidenceV2Error("calibration universe sample branch is invalid")
    return {
        "sample_id": _safe_id(value["sample_id"], label="sample_id"),
        "branch": branch,
        "symbol": _symbol(value["symbol"]),
        "cohort_id": _safe_id(value["cohort_id"], label="cohort_id"),
        "slot_id": _safe_id(value["slot_id"], label="slot_id"),
        **{
            field: _absolute_artifact_path(value[field], label=field)
            for field in _SAMPLE_PLAN_PATH_SCHEMAS
        },
    }


def build_calibration_universe_plan(
    *,
    protocol_attempt_id: str,
    epoch: str,
    schedule_id: str,
    model_bundle_refs: Mapping[str, EvidenceRef],
    sample_plans: Sequence[Mapping[str, Any]],
    lambda_fold_refs_by_branch: Mapping[str, Sequence[EvidenceRef]],
) -> dict[str, Any]:
    if epoch not in {"B", "C"}:
        raise EvidenceV2Error("calibration universe epoch must be B or C")
    if set(model_bundle_refs) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration universe must bind exactly four model bundles")
    if set(lambda_fold_refs_by_branch) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration universe must bind exactly four lambda branches")
    if any(
        reference.artifact_schema != MODEL_BUNDLE_SCHEMA
        or reference.root_policy != _PRIVATE_ROOT_POLICY
        for reference in model_bundle_refs.values()
    ):
        raise EvidenceV2Error("calibration universe model refs are not frozen private bundles")
    normalized_samples = [_normalize_sample_plan(item) for item in sample_plans]
    if not normalized_samples:
        raise EvidenceV2Error("calibration universe sample plan is empty")
    sample_ids = [item["sample_id"] for item in normalized_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise EvidenceV2Error("calibration universe sample IDs must be globally unique")
    for field in _SAMPLE_PLAN_PATH_SCHEMAS:
        paths = [item[field] for item in normalized_samples]
        if len(paths) != len(set(paths)):
            raise EvidenceV2Error(f"calibration universe {field} values must be unique")
    planned_sets = {
        branch: {
            (item["slot_id"], item["symbol"])
            for item in normalized_samples
            if item["branch"] == branch
        }
        for branch in CALIBRATION_BRANCHES
    }
    for branch in CALIBRATION_BRANCHES:
        branch_keys = [
            (item["slot_id"], item["symbol"])
            for item in normalized_samples
            if item["branch"] == branch
        ]
        if len(branch_keys) != len(set(branch_keys)):
            raise EvidenceV2Error(
                "calibration universe permits only one sample per branch/slot/symbol"
            )
    if any(not planned_sets[branch] for branch in CALIBRATION_BRANCHES) or any(
        planned_sets[branch] != planned_sets[CALIBRATION_BRANCHES[0]]
        for branch in CALIBRATION_BRANCHES[1:]
    ):
        raise EvidenceV2Error(
            "all four branches must predeclare the same slot/symbol sample universe"
        )
    lambda_refs: dict[str, list[dict[str, str]]] = {}
    lambda_bytes: set[str] = set()
    for branch in CALIBRATION_BRANCHES:
        refs = list(lambda_fold_refs_by_branch[branch])
        if len(refs) < 2:
            raise EvidenceV2Error("calibration universe requires at least two lambda folds")
        if any(ref.artifact_schema != LAMBDA_FOLD_SCHEMA for ref in refs):
            raise EvidenceV2Error("calibration universe lambda fold schema mismatch")
        if any(ref.root_policy != _PRIVATE_ROOT_POLICY for ref in refs):
            raise EvidenceV2Error("calibration universe lambda refs must be private")
        for ref in refs:
            if ref.byte_sha256 in lambda_bytes:
                raise EvidenceV2Error("calibration universe lambda refs must be unique")
            lambda_bytes.add(ref.byte_sha256)
        lambda_refs[branch] = [ref.to_dict() for ref in refs]
    normalized_samples.sort(key=lambda item: (item["branch"], item["sample_id"]))
    return seal_semantic(
        {
            "schema_version": CALIBRATION_UNIVERSE_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "epoch": epoch,
            "schedule_id": _safe_id(schedule_id, label="schedule_id"),
            "artifact_root_policy": "v16.private-evidence-root.v2",
            "model_bundle_refs": {
                branch: model_bundle_refs[branch].to_dict() for branch in CALIBRATION_BRANCHES
            },
            "sample_plans": normalized_samples,
            "lambda_fold_refs": lambda_refs,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_calibration_universe_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "schedule_id",
        "artifact_root_policy",
        "model_bundle_refs",
        "sample_plans",
        "lambda_fold_refs",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != CALIBRATION_UNIVERSE_SCHEMA:
        raise EvidenceV2Error("calibration universe envelope mismatch")
    if payload["artifact_root_policy"] != "v16.private-evidence-root.v2":
        raise EvidenceV2Error("calibration universe root policy mismatch")
    refs = payload["model_bundle_refs"]
    lambda_refs = payload["lambda_fold_refs"]
    if not isinstance(refs, Mapping) or set(refs) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration universe model refs shape mismatch")
    if not isinstance(lambda_refs, Mapping) or set(lambda_refs) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration universe lambda refs shape mismatch")
    if not isinstance(payload["sample_plans"], list):
        raise EvidenceV2Error("calibration universe sample plans must be a list")
    rebuilt = build_calibration_universe_plan(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        epoch=str(payload["epoch"]),
        schedule_id=str(payload["schedule_id"]),
        model_bundle_refs={
            branch: EvidenceRef.from_dict(refs[branch]) for branch in CALIBRATION_BRANCHES
        },
        sample_plans=payload["sample_plans"],
        lambda_fold_refs_by_branch={
            branch: [EvidenceRef.from_dict(item) for item in lambda_refs[branch]]
            for branch in CALIBRATION_BRANCHES
        },
    )
    if rebuilt != payload:
        raise EvidenceV2Error("calibration universe is not canonical")
    return payload


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    branch: str
    cohort_id: str
    cohort_start_date: str
    cohort_end_date: str
    probability: float
    prior_probability: float
    predicted_alpha: float
    realized_alpha: float
    interval_lower: float
    interval_upper: float
    prediction_ref: EvidenceRef
    outcome_ref: EvidenceRef

    def __post_init__(self) -> None:
        _safe_id(self.sample_id, label="sample_id")
        _safe_id(self.cohort_id, label="cohort_id")
        if self.branch not in CALIBRATION_BRANCHES:
            raise EvidenceV2Error("calibration branch is not formal v16")
        start = _iso_date(self.cohort_start_date, label="cohort_start_date")
        end = _iso_date(self.cohort_end_date, label="cohort_end_date")
        if end < start:
            raise EvidenceV2Error("calibration cohort date range is reversed")
        _probability(self.probability, label="probability")
        _probability(self.prior_probability, label="prior_probability")
        lower = _finite(self.interval_lower, label="interval_lower")
        upper = _finite(self.interval_upper, label="interval_upper")
        _finite(self.predicted_alpha, label="predicted_alpha")
        _finite(self.realized_alpha, label="realized_alpha")
        if upper < lower:
            raise EvidenceV2Error("calibration interval is reversed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "branch": self.branch,
            "cohort_id": self.cohort_id,
            "cohort_start_date": self.cohort_start_date,
            "cohort_end_date": self.cohort_end_date,
            "probability": encode_f64(self.probability),
            "prior_probability": encode_f64(self.prior_probability),
            "predicted_alpha": encode_f64(self.predicted_alpha),
            "realized_alpha": encode_f64(self.realized_alpha),
            "interval_lower": encode_f64(self.interval_lower),
            "interval_upper": encode_f64(self.interval_upper),
            "prediction_ref": self.prediction_ref.to_dict(),
            "outcome_ref": self.outcome_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationSample":
        fields = {
            "sample_id",
            "branch",
            "cohort_id",
            "cohort_start_date",
            "cohort_end_date",
            "probability",
            "prior_probability",
            "predicted_alpha",
            "realized_alpha",
            "interval_lower",
            "interval_upper",
            "prediction_ref",
            "outcome_ref",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise EvidenceV2Error("calibration sample fields mismatch")
        return cls(
            sample_id=str(value["sample_id"]),
            branch=str(value["branch"]),
            cohort_id=str(value["cohort_id"]),
            cohort_start_date=str(value["cohort_start_date"]),
            cohort_end_date=str(value["cohort_end_date"]),
            probability=decode_f64(value["probability"], label="probability"),
            prior_probability=decode_f64(value["prior_probability"], label="prior_probability"),
            predicted_alpha=decode_f64(value["predicted_alpha"], label="predicted_alpha"),
            realized_alpha=decode_f64(value["realized_alpha"], label="realized_alpha"),
            interval_lower=decode_f64(value["interval_lower"], label="interval_lower"),
            interval_upper=decode_f64(value["interval_upper"], label="interval_upper"),
            prediction_ref=EvidenceRef.from_dict(value["prediction_ref"]),
            outcome_ref=EvidenceRef.from_dict(value["outcome_ref"]),
        )


@dataclass(frozen=True)
class CalibrationArtifactPair:
    prediction: BoundCanonicalArtifact
    prediction_timestamp: TimestampAnchorBinding
    outcome: BoundCanonicalArtifact
    target_sources: MarkTargetEvidenceBundle


def build_branch_prediction(
    *,
    protocol_attempt_id: str,
    epoch: str,
    sample_id: str,
    branch: str,
    cohort_id: str,
    cohort_start_date: str,
    cohort_end_date: str,
    probability: float,
    prior_probability: float,
    predicted_alpha: float,
    interval_lower: float,
    interval_upper: float,
    model_bundle_ref: EvidenceRef,
    schedule_ref: EvidenceRef,
) -> dict[str, Any]:
    if epoch not in {"B", "C"}:
        raise EvidenceV2Error("branch prediction epoch must be B or C")
    if branch not in CALIBRATION_BRANCHES:
        raise EvidenceV2Error("branch prediction branch is not formal v16")
    start = _iso_date(cohort_start_date, label="cohort_start_date")
    end = _iso_date(cohort_end_date, label="cohort_end_date")
    if end < start:
        raise EvidenceV2Error("branch prediction cohort range is reversed")
    probability_value = _probability(probability, label="probability")
    prior_value = _probability(prior_probability, label="prior_probability")
    predicted = _finite(predicted_alpha, label="predicted_alpha")
    lower = _finite(interval_lower, label="interval_lower")
    upper = _finite(interval_upper, label="interval_upper")
    if upper < lower:
        raise EvidenceV2Error("branch prediction interval is reversed")
    return seal_semantic(
        {
            "schema_version": BRANCH_PREDICTION_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "epoch": epoch,
            "sample_id": _safe_id(sample_id, label="sample_id"),
            "branch": branch,
            "cohort_id": _safe_id(cohort_id, label="cohort_id"),
            "cohort_start_date": start,
            "cohort_end_date": end,
            "probability": encode_f64(probability_value),
            "prior_probability": encode_f64(prior_value),
            "predicted_alpha": encode_f64(predicted),
            "interval_lower": encode_f64(lower),
            "interval_upper": encode_f64(upper),
            "model_bundle_ref": model_bundle_ref.to_dict(),
            "schedule_ref": schedule_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_branch_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "sample_id",
        "branch",
        "cohort_id",
        "cohort_start_date",
        "cohort_end_date",
        "probability",
        "prior_probability",
        "predicted_alpha",
        "interval_lower",
        "interval_upper",
        "model_bundle_ref",
        "schedule_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != BRANCH_PREDICTION_SCHEMA:
        raise EvidenceV2Error("branch prediction envelope mismatch")
    _safe_id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    if payload["epoch"] not in {"B", "C"}:
        raise EvidenceV2Error("branch prediction epoch is invalid")
    if payload["branch"] not in CALIBRATION_BRANCHES:
        raise EvidenceV2Error("branch prediction branch is invalid")
    _safe_id(payload["sample_id"], label="sample_id")
    _safe_id(payload["cohort_id"], label="cohort_id")
    start = _iso_date(payload["cohort_start_date"], label="cohort_start_date")
    end = _iso_date(payload["cohort_end_date"], label="cohort_end_date")
    if end < start:
        raise EvidenceV2Error("branch prediction cohort range is reversed")
    _probability(decode_f64(payload["probability"], label="probability"), label="probability")
    _probability(
        decode_f64(payload["prior_probability"], label="prior_probability"),
        label="prior_probability",
    )
    decode_f64(payload["predicted_alpha"], label="predicted_alpha")
    lower = decode_f64(payload["interval_lower"], label="interval_lower")
    upper = decode_f64(payload["interval_upper"], label="interval_upper")
    if upper < lower:
        raise EvidenceV2Error("branch prediction interval is reversed")
    for field in ("model_bundle_ref", "schedule_ref"):
        EvidenceRef.from_dict(payload[field])
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("branch prediction must be nonauthorizing")
    return payload


def build_lambda_fold_evidence(
    *,
    protocol_attempt_id: str,
    epoch: str,
    branch: str,
    fold_id: str,
    lambda_value: float,
    model_bundle_ref: EvidenceRef,
    fit_sample_ref: EvidenceRef,
    holdout_sample_ref: EvidenceRef,
) -> dict[str, Any]:
    if epoch not in {"B", "C"} or branch not in CALIBRATION_BRANCHES:
        raise EvidenceV2Error("lambda fold epoch/branch is invalid")
    value = _finite(lambda_value, label="lambda_value")
    if not 0.0 <= value <= 1.0:
        raise EvidenceV2Error("lambda_value must be in [0, 1]")
    if fit_sample_ref.byte_sha256 == holdout_sample_ref.byte_sha256:
        raise EvidenceV2Error("lambda fit and holdout samples must be disjoint")
    return seal_semantic(
        {
            "schema_version": LAMBDA_FOLD_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "epoch": epoch,
            "branch": branch,
            "fold_id": _safe_id(fold_id, label="fold_id"),
            "lambda_value": encode_f64(value),
            "model_bundle_ref": model_bundle_ref.to_dict(),
            "fit_sample_ref": fit_sample_ref.to_dict(),
            "holdout_sample_ref": holdout_sample_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_lambda_fold_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "branch",
        "fold_id",
        "lambda_value",
        "model_bundle_ref",
        "fit_sample_ref",
        "holdout_sample_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != LAMBDA_FOLD_SCHEMA:
        raise EvidenceV2Error("lambda fold evidence envelope mismatch")
    _safe_id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    if payload["epoch"] not in {"B", "C"} or payload["branch"] not in CALIBRATION_BRANCHES:
        raise EvidenceV2Error("lambda fold epoch/branch is invalid")
    _safe_id(payload["fold_id"], label="fold_id")
    lambda_value = decode_f64(payload["lambda_value"], label="lambda_value")
    if not 0.0 <= lambda_value <= 1.0:
        raise EvidenceV2Error("lambda_value must be in [0, 1]")
    refs = {
        field: EvidenceRef.from_dict(payload[field])
        for field in ("model_bundle_ref", "fit_sample_ref", "holdout_sample_ref")
    }
    if refs["fit_sample_ref"].byte_sha256 == refs["holdout_sample_ref"].byte_sha256:
        raise EvidenceV2Error("lambda fit and holdout samples must be disjoint")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("lambda fold evidence must be nonauthorizing")
    return payload


def _log_loss(probability: float, outcome: float) -> float:
    clipped = min(max(probability, _PROBABILITY_CLIP), 1.0 - _PROBABILITY_CLIP)
    return -(outcome * math.log(clipped) + (1.0 - outcome) * math.log(1.0 - clipped))


def _ece(samples: Sequence[CalibrationSample]) -> float:
    ordered = sorted(samples, key=lambda sample: (sample.probability, sample.sample_id))
    quotient, remainder = divmod(len(ordered), ECE_BIN_COUNT)
    sizes = [quotient + (1 if index < remainder else 0) for index in range(ECE_BIN_COUNT)]
    cursor = 0
    weighted_errors: list[float] = []
    for size in sizes:
        if size == 0:
            continue
        bucket = ordered[cursor : cursor + size]
        cursor += size
        probabilities = [sample.probability for sample in bucket]
        outcomes = [1.0 if sample.realized_alpha > 0.0 else 0.0 for sample in bucket]
        weighted_errors.append(size * abs(_mean(probabilities) - _mean(outcomes)))
    if cursor != len(ordered):
        raise EvidenceV2Error("ECE partition did not consume every sample")
    return math.fsum(weighted_errors) / len(ordered)


def _cohort_statistics(samples: Sequence[CalibrationSample]) -> dict[str, float]:
    if len(samples) < MIN_COHORT_SAMPLES:
        raise EvidenceV2Error(f"calibration cohort has fewer than {MIN_COHORT_SAMPLES} samples")
    outcomes = [1.0 if sample.realized_alpha > 0.0 else 0.0 for sample in samples]
    brier = [(sample.probability - outcome) ** 2 for sample, outcome in zip(samples, outcomes)]
    brier_prior = [
        (sample.prior_probability - outcome) ** 2 for sample, outcome in zip(samples, outcomes)
    ]
    logloss = [_log_loss(sample.probability, outcome) for sample, outcome in zip(samples, outcomes)]
    logloss_prior = [
        _log_loss(sample.prior_probability, outcome) for sample, outcome in zip(samples, outcomes)
    ]
    coverage = [
        1.0 if sample.interval_lower <= sample.realized_alpha <= sample.interval_upper else 0.0
        for sample in samples
    ]
    alpha_errors = [abs(sample.predicted_alpha - sample.realized_alpha) for sample in samples]
    zero_errors = [abs(sample.realized_alpha) for sample in samples]
    ordered_top = sorted(samples, key=lambda sample: (-sample.predicted_alpha, sample.sample_id))
    top_count = math.ceil(len(ordered_top) / 5)
    return {
        "brier": _mean(brier),
        "brier_prior": _mean(brier_prior),
        "brier_delta": _mean([model - prior for model, prior in zip(brier, brier_prior)]),
        "logloss": _mean(logloss),
        "logloss_prior": _mean(logloss_prior),
        "logloss_delta": _mean([model - prior for model, prior in zip(logloss, logloss_prior)]),
        "ece": _ece(samples),
        "interval_coverage": _mean(coverage),
        "alpha_mae": _mean(alpha_errors),
        "zero_alpha_mae": _mean(zero_errors),
        "top_bucket_edge": _mean([sample.realized_alpha for sample in ordered_top[:top_count]]),
    }


def bootstrap_draw_index(*, seed_hex: str, replicate: int, draw: int, count: int) -> int:
    """Map ``r:d`` through HMAC-SHA256 into one deterministic cohort index."""

    if len(seed_hex) != 64 or any(character not in "0123456789abcdef" for character in seed_hex):
        raise EvidenceV2Error("bootstrap seed must be exactly 32 lowercase-hex bytes")
    if replicate < 0 or draw < 0 or count <= 0:
        raise EvidenceV2Error("bootstrap coordinates are outside their domain")
    message = f"{replicate}:{draw}".encode("ascii")
    digest = hmac.new(bytes.fromhex(seed_hex), message, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") % count


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 < quantile <= 1.0:
        raise EvidenceV2Error("nearest-rank input is invalid")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _bootstrap_bound(
    cohort_values: Sequence[float],
    *,
    seed_hex: str,
    upper: bool,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if not cohort_values or replicates != BOOTSTRAP_REPLICATES:
        raise EvidenceV2Error(
            f"evidence-v2 requires exactly {BOOTSTRAP_REPLICATES} bootstrap replicates"
        )
    count = len(cohort_values)
    estimates: list[float] = []
    for replicate in range(replicates):
        selected = [
            cohort_values[
                bootstrap_draw_index(
                    seed_hex=seed_hex,
                    replicate=replicate,
                    draw=draw,
                    count=count,
                )
            ]
            for draw in range(count)
        ]
        estimates.append(_mean(selected))
    quantile = 1.0 - BOOTSTRAP_ONE_SIDED_ALPHA if upper else BOOTSTRAP_ONE_SIDED_ALPHA
    return _nearest_rank(estimates, quantile)


def _validate_nonoverlap(samples: Sequence[CalibrationSample], *, branch: str) -> None:
    windows: dict[str, tuple[str, str]] = {}
    for sample in samples:
        window = (sample.cohort_start_date, sample.cohort_end_date)
        prior = windows.setdefault(sample.cohort_id, window)
        if prior != window:
            raise EvidenceV2Error(f"branch {branch} cohort window drifts within its samples")
    ordered = sorted((start, end, cohort_id) for cohort_id, (start, end) in windows.items())
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] <= previous[1]:
            raise EvidenceV2Error(f"branch {branch} cohort windows overlap")


def _branch_statistics(
    samples: Sequence[CalibrationSample],
    *,
    lambda_folds: Sequence[float],
    seed_hex: str,
) -> dict[str, Any]:
    _validate_nonoverlap(samples, branch=samples[0].branch)
    cohorts: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        cohorts.setdefault(sample.cohort_id, []).append(sample)
    ordered_cohorts = [
        sorted(cohorts[cohort_id], key=lambda sample: sample.sample_id)
        for cohort_id in sorted(cohorts)
    ]
    statistics = [_cohort_statistics(cohort) for cohort in ordered_cohorts]
    point = {name: _mean([cohort[name] for cohort in statistics]) for name in statistics[0]}
    lambdas = [_finite(value, label="lambda fold") for value in lambda_folds]
    if len(lambdas) < 2 or any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise EvidenceV2Error("lambda folds require at least two values in [0, 1]")
    lambda_min = min(lambdas)
    lambda_max = max(lambdas)
    brier_upper = _bootstrap_bound(
        [cohort["brier_delta"] for cohort in statistics],
        seed_hex=seed_hex,
        upper=True,
    )
    logloss_upper = _bootstrap_bound(
        [cohort["logloss_delta"] for cohort in statistics],
        seed_hex=seed_hex,
        upper=True,
    )
    top_edge_lower = _bootstrap_bound(
        [cohort["top_bucket_edge"] for cohort in statistics],
        seed_hex=seed_hex,
        upper=False,
    )
    gates = {
        "sample_count_gte_300": len(samples) >= MIN_BRANCH_SAMPLES,
        "nonoverlap_cohort_count_gte_8": len(cohorts) >= MIN_BRANCH_COHORTS,
        "brier_delta_bootstrap_upper_lt_zero": brier_upper < 0.0,
        "logloss_delta_bootstrap_upper_lt_zero": logloss_upper < 0.0,
        "ece_lte_0_05": point["ece"] <= MAX_ECE,
        "interval_coverage_0_85_to_0_95": (
            MIN_INTERVAL_COVERAGE <= point["interval_coverage"] <= MAX_INTERVAL_COVERAGE
        ),
        "alpha_mae_lt_zero_alpha_mae": point["alpha_mae"] < point["zero_alpha_mae"],
        "top_bucket_edge_bootstrap_lower_gt_zero": top_edge_lower > 0.0,
        "lambda_fold_range_lte_0_20": lambda_max - lambda_min <= MAX_LAMBDA_FOLD_RANGE,
    }
    return {
        "samples": len(samples),
        "nonoverlap_cohorts": len(cohorts),
        "cohort_ids": sorted(cohorts),
        "metrics": {
            **{name: encode_f64(value) for name, value in sorted(point.items())},
            "brier_delta_bootstrap_upper": encode_f64(brier_upper),
            "logloss_delta_bootstrap_upper": encode_f64(logloss_upper),
            "top_bucket_edge_bootstrap_lower": encode_f64(top_edge_lower),
            "lambda_fold_min": encode_f64(lambda_min),
            "lambda_fold_max": encode_f64(lambda_max),
            "lambda_fold_range": encode_f64(lambda_max - lambda_min),
        },
        "lambda_folds": [encode_f64(value) for value in lambdas],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def _samples_from_artifacts(
    pairs: Sequence[CalibrationArtifactPair],
    *,
    universe: BoundCanonicalArtifact,
    protocol_attempt_id: str,
    epoch: str,
    schedule_ref: EvidenceRef,
    model_bundle_refs: Mapping[str, EvidenceRef],
) -> tuple[list[CalibrationSample], str]:
    universe_payload = validate_calibration_universe_plan(universe.read())
    if (
        universe_payload["protocol_attempt_id"] != protocol_attempt_id
        or universe_payload["epoch"] != epoch
        or universe_payload["model_bundle_refs"]
        != {branch: model_bundle_refs[branch].to_dict() for branch in CALIBRATION_BRANCHES}
    ):
        raise EvidenceV2Error("calibration universe protocol/model lineage mismatch")
    planned_by_id = {item["sample_id"]: item for item in universe_payload["sample_plans"]}
    samples: list[CalibrationSample] = []
    common_by_source: dict[
        tuple[str, str, str],
        ValidatedMarkTargetCommonEvidence,
    ] = {}
    parquet_hash_by_identity: dict[int, tuple[bytes, str]] = {}
    stock_sources_by_key: dict[
        tuple[str, str, str, str],
        ValidatedStockMarkSources,
    ] = {}
    seen_sample_ids: set[str] = set()
    schedule_seed: str | None = None
    for pair in pairs:
        prediction = validate_branch_prediction(pair.prediction.read())
        sample_id = str(prediction["sample_id"])
        if sample_id in seen_sample_ids or sample_id not in planned_by_id:
            raise EvidenceV2Error("calibration artifacts drift from predeclared universe")
        seen_sample_ids.add(sample_id)
        plan = planned_by_id[sample_id]
        source_bundle = pair.target_sources
        parquet_identity = id(source_bundle.benchmark_parquet)
        cached_hash = parquet_hash_by_identity.get(parquet_identity)
        if cached_hash is None or cached_hash[0] is not source_bundle.benchmark_parquet:
            cached_hash = (
                source_bundle.benchmark_parquet,
                hashlib.sha256(source_bundle.benchmark_parquet).hexdigest(),
            )
            parquet_hash_by_identity[parquet_identity] = cached_hash
        common_key = (
            source_bundle.schedule_anchor.schedule.reference.byte_sha256,
            source_bundle.benchmark_manifest.reference.byte_sha256,
            cached_hash[1],
        )
        common = common_by_source.get(common_key)
        if common is None:
            common = prepare_mark_target_common_evidence(
                schedule_anchor=source_bundle.schedule_anchor,
                benchmark_manifest=source_bundle.benchmark_manifest,
                benchmark_parquet=source_bundle.benchmark_parquet,
            )
            common_by_source[common_key] = common
        if (
            common.schedule_anchor.schedule.reference.to_dict() != schedule_ref.to_dict()
            or common.protocol_attempt_id != protocol_attempt_id
            or common.epoch != epoch
            or common.calibration_universe_ref.to_dict() != universe.reference.to_dict()
        ):
            raise EvidenceV2Error("calibration target-source schedule lineage mismatch")
        if universe_payload["schedule_id"] != common.schedule_id:
            raise EvidenceV2Error("calibration universe schedule ID mismatch")
        if schedule_seed is None:
            schedule_seed = common.seed_hex
        elif schedule_seed != common.seed_hex:
            raise EvidenceV2Error("calibration artifacts bind inconsistent schedule seeds")
        expected_model_refs = tuple(
            (branch, model_bundle_refs[branch]) for branch in CALIBRATION_BRANCHES
        )
        if common.model_bundle_refs != expected_model_refs:
            raise EvidenceV2Error("calibration target-source model-bundle lineage mismatch")
        stock_source_bundle = source_bundle.stock_sources
        stock_key = (
            stock_source_bundle.market_parquet.reference.byte_sha256,
            stock_source_bundle.adjustment_factors.reference.byte_sha256,
            stock_source_bundle.pit_membership.reference.byte_sha256,
            stock_source_bundle.suspensions.reference.byte_sha256,
        )
        stock_sources = stock_sources_by_key.get(stock_key)
        if stock_sources is None:
            stock_sources = prepare_stock_mark_sources(stock_source_bundle)
            stock_sources_by_key[stock_key] = stock_sources
        outcome = validate_mark_target_outcome_from_common_evidence(
            pair.outcome.read(),
            common=common,
            stock_marks=source_bundle.stock_marks,
            stock_sources=stock_sources,
            costs=source_bundle.costs,
        )
        timestamp_attempt, timestamp_receipt = pair.prediction_timestamp.read()
        branch = str(prediction["branch"])
        if (
            prediction["protocol_attempt_id"] != protocol_attempt_id
            or outcome["protocol_attempt_id"] != protocol_attempt_id
            or prediction["epoch"] != epoch
        ):
            raise EvidenceV2Error("calibration sample protocol/epoch lineage mismatch")
        if prediction["sample_id"] != outcome["sample_id"]:
            raise EvidenceV2Error("calibration prediction/outcome sample ID mismatch")
        matching_slots = [
            (s0_close, s1_open)
            for slot_id, s0_close, s1_open in common.prediction_anchor_windows
            if slot_id == plan["slot_id"]
        ]
        if len(matching_slots) != 1:
            raise EvidenceV2Error("calibration universe schedule slot mismatch")
        s0_close, s1_open = matching_slots[0]
        if (
            timestamp_receipt["anchored_artifact_ref"] != pair.prediction.reference.to_dict()
            or timestamp_receipt["anchor_kind"] != "prediction"
            or timestamp_receipt["anchor_not_before"] != s0_close
            or timestamp_receipt["anchor_not_after"] != s1_open
            or timestamp_attempt["protocol_attempt_id"] != protocol_attempt_id
        ):
            raise EvidenceV2Error("calibration prediction RFC3161 anchor lineage mismatch")
        if (
            prediction["cohort_start_date"] != outcome["s1_date"]
            or prediction["cohort_end_date"] != outcome["s20_date"]
        ):
            raise EvidenceV2Error("calibration prediction/outcome cohort window mismatch")
        if (
            prediction["schedule_ref"] != schedule_ref.to_dict()
            or outcome["schedule_ref"] != schedule_ref.to_dict()
        ):
            raise EvidenceV2Error("calibration sample schedule lineage mismatch")
        if prediction["model_bundle_ref"] != model_bundle_refs[branch].to_dict():
            raise EvidenceV2Error("calibration sample model-bundle lineage mismatch")
        actual_plan = {
            "sample_id": sample_id,
            "branch": branch,
            "symbol": outcome["symbol"],
            "cohort_id": prediction["cohort_id"],
            "slot_id": source_bundle.stock_marks.read()["slot_id"],
            "prediction_path": pair.prediction.reference.absolute_path,
            "outcome_path": pair.outcome.reference.absolute_path,
            "stock_marks_path": source_bundle.stock_marks.reference.absolute_path,
            "costs_path": source_bundle.costs.reference.absolute_path,
            "prediction_timestamp_attempt_path": (
                pair.prediction_timestamp.attempt.reference.absolute_path
            ),
            "prediction_timestamp_receipt_path": (
                pair.prediction_timestamp.validation_receipt.reference.absolute_path
            ),
        }
        if actual_plan != plan:
            raise EvidenceV2Error("calibration artifact paths drift from predeclared universe")
        actual_refs = {
            "prediction_path": pair.prediction.reference,
            "outcome_path": pair.outcome.reference,
            "stock_marks_path": source_bundle.stock_marks.reference,
            "costs_path": source_bundle.costs.reference,
            "prediction_timestamp_attempt_path": (pair.prediction_timestamp.attempt.reference),
            "prediction_timestamp_receipt_path": (
                pair.prediction_timestamp.validation_receipt.reference
            ),
        }
        for field, reference in actual_refs.items():
            if (
                reference.artifact_schema != _SAMPLE_PLAN_PATH_SCHEMAS[field]
                or reference.root_policy != universe_payload["artifact_root_policy"]
            ):
                raise EvidenceV2Error("calibration artifact schema/root policy mismatch")
        samples.append(
            CalibrationSample(
                sample_id=sample_id,
                branch=branch,
                cohort_id=prediction["cohort_id"],
                cohort_start_date=prediction["cohort_start_date"],
                cohort_end_date=prediction["cohort_end_date"],
                probability=decode_f64(
                    prediction["probability"],
                    label="probability",
                ),
                prior_probability=decode_f64(
                    prediction["prior_probability"],
                    label="prior_probability",
                ),
                predicted_alpha=decode_f64(
                    prediction["predicted_alpha"],
                    label="predicted_alpha",
                ),
                realized_alpha=decode_f64(
                    outcome["realized_mark_alpha"],
                    label="realized_mark_alpha",
                ),
                interval_lower=decode_f64(
                    prediction["interval_lower"],
                    label="interval_lower",
                ),
                interval_upper=decode_f64(
                    prediction["interval_upper"],
                    label="interval_upper",
                ),
                prediction_ref=pair.prediction.reference,
                outcome_ref=pair.outcome.reference,
            )
        )
    if seen_sample_ids != set(planned_by_id):
        missing = sorted(set(planned_by_id) - seen_sample_ids)
        raise EvidenceV2Error(
            "calibration artifacts omit predeclared samples: " + ",".join(missing[:10])
        )
    if schedule_seed is None:
        raise EvidenceV2Error("calibration artifacts do not bind a schedule seed")
    return samples, schedule_seed


def _lambda_values_from_artifacts(
    artifacts_by_branch: Mapping[str, Sequence[BoundCanonicalArtifact]],
    *,
    universe_payload: Mapping[str, Any],
    protocol_attempt_id: str,
    epoch: str,
    model_bundle_refs: Mapping[str, EvidenceRef],
) -> dict[str, list[float]]:
    if set(artifacts_by_branch) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("lambda fold artifacts must bind exactly four branches")
    result: dict[str, list[float]] = {}
    seen_bytes: set[str] = set()
    for branch in CALIBRATION_BRANCHES:
        if [artifact.reference.to_dict() for artifact in artifacts_by_branch[branch]] != list(
            universe_payload["lambda_fold_refs"][branch]
        ):
            raise EvidenceV2Error("lambda fold artifacts drift from predeclared universe")
        rows = [
            validate_lambda_fold_evidence(artifact.read())
            for artifact in artifacts_by_branch[branch]
        ]
        if len(rows) < 2 or len({row["fold_id"] for row in rows}) != len(rows):
            raise EvidenceV2Error("lambda fold IDs require at least two unique folds")
        for artifact, row in zip(artifacts_by_branch[branch], rows):
            if artifact.reference.byte_sha256 in seen_bytes:
                raise EvidenceV2Error("lambda fold artifact bytes must be globally unique")
            seen_bytes.add(artifact.reference.byte_sha256)
            if (
                row["protocol_attempt_id"] != protocol_attempt_id
                or row["epoch"] != epoch
                or row["branch"] != branch
                or row["model_bundle_ref"] != model_bundle_refs[branch].to_dict()
            ):
                raise EvidenceV2Error("lambda fold evidence lineage mismatch")
        ordered = sorted(rows, key=lambda row: row["fold_id"])
        result[branch] = [decode_f64(row["lambda_value"], label="lambda_value") for row in ordered]
    return result


def _compute_calibration_payload(
    *,
    protocol_attempt_id: str,
    epoch: str,
    universe_ref: EvidenceRef,
    schedule_ref: EvidenceRef,
    model_bundle_refs: Mapping[str, EvidenceRef],
    samples: Sequence[CalibrationSample],
    lambda_folds_by_branch: Mapping[str, Sequence[float]],
    seed_hex: str,
) -> dict[str, Any]:
    attempt = _safe_id(protocol_attempt_id, label="protocol_attempt_id")
    if epoch not in {"B", "C"}:
        raise EvidenceV2Error("calibration evidence epoch must be B or C")
    if set(model_bundle_refs) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration model bundles must bind exactly four branches")
    if set(lambda_folds_by_branch) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration lambda folds must bind exactly four branches")
    if len(seed_hex) != 64 or any(character not in "0123456789abcdef" for character in seed_hex):
        raise EvidenceV2Error("calibration seed must be exactly 32 lowercase-hex bytes")
    ordered_samples = sorted(samples, key=lambda sample: (sample.branch, sample.sample_id))
    sample_keys = [(sample.branch, sample.sample_id) for sample in ordered_samples]
    if len(sample_keys) != len(set(sample_keys)):
        raise EvidenceV2Error("calibration sample IDs must be unique within branch")
    prediction_shas = [sample.prediction_ref.byte_sha256 for sample in ordered_samples]
    outcome_shas = [sample.outcome_ref.byte_sha256 for sample in ordered_samples]
    if len(prediction_shas) != len(set(prediction_shas)):
        raise EvidenceV2Error("every calibration sample requires distinct prediction bytes")
    if len(outcome_shas) != len(set(outcome_shas)):
        raise EvidenceV2Error("every calibration sample requires distinct outcome bytes")
    if set(prediction_shas).intersection(outcome_shas):
        raise EvidenceV2Error("prediction and outcome byte identities must be disjoint")
    grouped: dict[str, list[CalibrationSample]] = {branch: [] for branch in CALIBRATION_BRANCHES}
    for sample in ordered_samples:
        grouped[sample.branch].append(sample)
    if any(not grouped[branch] for branch in CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration evidence requires all four formal branches")
    cohort_windows = {
        branch: {(sample.cohort_start_date, sample.cohort_end_date) for sample in grouped[branch]}
        for branch in CALIBRATION_BRANCHES
    }
    if any(
        cohort_windows[branch] != cohort_windows[CALIBRATION_BRANCHES[0]]
        for branch in CALIBRATION_BRANCHES[1:]
    ):
        raise EvidenceV2Error("formal branches must use the same prospective cohort windows")
    branch_results = {
        branch: _branch_statistics(
            grouped[branch],
            lambda_folds=lambda_folds_by_branch[branch],
            seed_hex=seed_hex,
        )
        for branch in CALIBRATION_BRANCHES
    }
    blockers = [
        f"calibration_gate_failed:{branch}:{gate}"
        for branch in CALIBRATION_BRANCHES
        for gate, passed in branch_results[branch]["gates"].items()
        if not passed
    ]
    return {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA,
        "protocol_attempt_id": attempt,
        "epoch": epoch,
        "universe_ref": universe_ref.to_dict(),
        "schedule_ref": schedule_ref.to_dict(),
        "model_bundle_refs": {
            branch: model_bundle_refs[branch].to_dict() for branch in CALIBRATION_BRANCHES
        },
        "bootstrap": {
            "algorithm": "hmac-sha256-cluster-bootstrap-nearest-rank-v1",
            "seed_hex": seed_hex,
            "message_format": "r:d",
            "replicates": BOOTSTRAP_REPLICATES,
            "one_sided_alpha": encode_f64(BOOTSTRAP_ONE_SIDED_ALPHA),
            "cohort_weighting": "equal_cohort_then_equal_sample",
        },
        "branches": branch_results,
        "samples": [sample.to_dict() for sample in ordered_samples],
        "all_metric_gates_passed": not blockers,
        "blockers": blockers,
        "activation_candidate": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
    }


def build_calibration_evidence(
    *,
    protocol_attempt_id: str,
    epoch: str,
    universe: BoundCanonicalArtifact,
    schedule_ref: EvidenceRef,
    model_bundle_refs: Mapping[str, EvidenceRef],
    sample_artifacts: Sequence[CalibrationArtifactPair],
    lambda_fold_artifacts_by_branch: Mapping[
        str,
        Sequence[BoundCanonicalArtifact],
    ],
) -> dict[str, Any]:
    if set(model_bundle_refs) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration model bundles must bind exactly four branches")
    if universe.reference.root_policy != _PRIVATE_ROOT_POLICY:
        raise EvidenceV2Error("calibration universe artifact must use the private root")
    universe_payload = validate_calibration_universe_plan(universe.read())
    samples, schedule_seed = _samples_from_artifacts(
        sample_artifacts,
        universe=universe,
        protocol_attempt_id=protocol_attempt_id,
        epoch=epoch,
        schedule_ref=schedule_ref,
        model_bundle_refs=model_bundle_refs,
    )
    lambda_folds_by_branch = _lambda_values_from_artifacts(
        lambda_fold_artifacts_by_branch,
        universe_payload=universe_payload,
        protocol_attempt_id=protocol_attempt_id,
        epoch=epoch,
        model_bundle_refs=model_bundle_refs,
    )
    return seal_semantic(
        _compute_calibration_payload(
            protocol_attempt_id=protocol_attempt_id,
            epoch=epoch,
            universe_ref=universe.reference,
            schedule_ref=schedule_ref,
            model_bundle_refs=model_bundle_refs,
            samples=samples,
            lambda_folds_by_branch=lambda_folds_by_branch,
            seed_hex=schedule_seed,
        )
    )


def validate_calibration_evidence(
    value: Mapping[str, Any],
    *,
    universe: BoundCanonicalArtifact,
    sample_artifacts: Sequence[CalibrationArtifactPair],
    lambda_fold_artifacts_by_branch: Mapping[
        str,
        Sequence[BoundCanonicalArtifact],
    ],
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "epoch",
        "universe_ref",
        "schedule_ref",
        "model_bundle_refs",
        "bootstrap",
        "branches",
        "samples",
        "all_metric_gates_passed",
        "blockers",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != CALIBRATION_EVIDENCE_SCHEMA:
        raise EvidenceV2Error("calibration evidence envelope mismatch")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("calibration evidence must be permanently nonauthorizing")
    bootstrap = payload["bootstrap"]
    expected_bootstrap_fields = {
        "algorithm",
        "seed_hex",
        "message_format",
        "replicates",
        "one_sided_alpha",
        "cohort_weighting",
    }
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != expected_bootstrap_fields:
        raise EvidenceV2Error("calibration bootstrap contract mismatch")
    if (
        bootstrap["algorithm"] != "hmac-sha256-cluster-bootstrap-nearest-rank-v1"
        or bootstrap["message_format"] != "r:d"
        or bootstrap["replicates"] != BOOTSTRAP_REPLICATES
        or bootstrap["cohort_weighting"] != "equal_cohort_then_equal_sample"
        or decode_f64(bootstrap["one_sided_alpha"], label="one_sided_alpha")
        != BOOTSTRAP_ONE_SIDED_ALPHA
    ):
        raise EvidenceV2Error("calibration bootstrap parameters drift")
    refs = payload["model_bundle_refs"]
    if not isinstance(refs, Mapping) or set(refs) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration model bundle shape mismatch")
    model_refs = {branch: EvidenceRef.from_dict(refs[branch]) for branch in CALIBRATION_BRANCHES}
    universe_payload = validate_calibration_universe_plan(universe.read())
    if payload["universe_ref"] != universe.reference.to_dict():
        raise EvidenceV2Error("calibration evidence universe ref mismatch")
    if not isinstance(payload["samples"], list):
        raise EvidenceV2Error("calibration samples must be a list")
    schedule_ref = EvidenceRef.from_dict(payload["schedule_ref"])
    samples, schedule_seed = _samples_from_artifacts(
        sample_artifacts,
        universe=universe,
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        epoch=str(payload["epoch"]),
        schedule_ref=schedule_ref,
        model_bundle_refs=model_refs,
    )
    if bootstrap["seed_hex"] != schedule_seed:
        raise EvidenceV2Error("calibration bootstrap seed drifts from pre-s0 schedule")
    branches = payload["branches"]
    if not isinstance(branches, Mapping) or set(branches) != set(CALIBRATION_BRANCHES):
        raise EvidenceV2Error("calibration branch shape mismatch")
    lambda_folds = _lambda_values_from_artifacts(
        lambda_fold_artifacts_by_branch,
        universe_payload=universe_payload,
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        epoch=str(payload["epoch"]),
        model_bundle_refs=model_refs,
    )
    for branch in CALIBRATION_BRANCHES:
        branch_payload = branches[branch]
        if not isinstance(branch_payload, Mapping) or "lambda_folds" not in branch_payload:
            raise EvidenceV2Error("calibration branch lambda folds are missing")
        raw_lambdas = branch_payload["lambda_folds"]
        if not isinstance(raw_lambdas, list):
            raise EvidenceV2Error("calibration lambda folds must be a list")
        declared_lambdas = [decode_f64(item, label=f"{branch}.lambda_fold") for item in raw_lambdas]
        if declared_lambdas != lambda_folds[branch]:
            raise EvidenceV2Error("calibration lambda folds drift from bound artifacts")
    recomputed = seal_semantic(
        _compute_calibration_payload(
            protocol_attempt_id=str(payload["protocol_attempt_id"]),
            epoch=str(payload["epoch"]),
            universe_ref=universe.reference,
            schedule_ref=schedule_ref,
            model_bundle_refs=model_refs,
            samples=samples,
            lambda_folds_by_branch=lambda_folds,
            seed_hex=schedule_seed,
        )
    )
    if recomputed != payload:
        raise EvidenceV2Error("calibration evidence does not match deterministic recomputation")
    return payload


def one_sided_student_t_pvalue(values: Sequence[float]) -> float:
    """Return the greater-than-zero one-sample Student-t p-value."""

    normalized = [_finite(value, label="Student-t observation") for value in values]
    if len(normalized) < 2:
        raise EvidenceV2Error("Student-t gate requires at least two observations")
    mean = _mean(normalized)
    variance = math.fsum((value - mean) ** 2 for value in normalized) / (len(normalized) - 1)
    if variance <= 0.0:
        raise EvidenceV2Error("Student-t gate rejects a constant series")
    statistic = mean / math.sqrt(variance / len(normalized))
    pvalue = float(stats.t.sf(statistic, df=len(normalized) - 1))
    if not math.isfinite(pvalue):
        raise EvidenceV2Error("Student-t gate produced a non-finite p-value")
    return pvalue


def benjamini_hochberg_qvalues(pvalues: Sequence[float]) -> list[float]:
    normalized = [_finite(value, label="p-value") for value in pvalues]
    if not normalized or any(not 0.0 <= value <= 1.0 for value in normalized):
        raise EvidenceV2Error("BH requires nonempty p-values in [0, 1]")
    count = len(normalized)
    ordered = sorted(enumerate(normalized), key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index, pvalue = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, pvalue * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def factor_b_multiple_testing_gates(pvalues: Sequence[float]) -> list[dict[str, Any]]:
    normalized = [_finite(value, label="Factor B p-value") for value in pvalues]
    if not normalized or any(not 0.0 <= value <= 1.0 for value in normalized):
        raise EvidenceV2Error("Factor B p-values must be nonempty and in [0, 1]")
    bonferroni = FACTOR_B_FAMILY_ALPHA / len(normalized)
    qvalues = benjamini_hochberg_qvalues(normalized)
    return [
        {
            "p_value": encode_f64(pvalue),
            "bonferroni_alpha": encode_f64(bonferroni),
            "bh_q_value": encode_f64(qvalue),
            "bonferroni_pass": pvalue <= bonferroni,
            "bh_q_lte_0_10": qvalue <= FACTOR_B_BH_Q,
            "pass": pvalue <= bonferroni and qvalue <= FACTOR_B_BH_Q,
        }
        for pvalue, qvalue in zip(normalized, qvalues)
    ]


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CALIBRATION_BRANCHES",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CALIBRATION_UNIVERSE_SCHEMA",
    "BRANCH_PREDICTION_SCHEMA",
    "CalibrationArtifactPair",
    "CalibrationSample",
    "FACTOR_B_BH_Q",
    "FACTOR_B_FAMILY_ALPHA",
    "LAMBDA_FOLD_SCHEMA",
    "POST_ACTIVATION_ALPHA",
    "POST_ACTIVATION_HYPOTHESIS_COUNT",
    "benjamini_hochberg_qvalues",
    "bootstrap_draw_index",
    "build_branch_prediction",
    "build_calibration_evidence",
    "build_calibration_universe_plan",
    "build_lambda_fold_evidence",
    "factor_b_multiple_testing_gates",
    "one_sided_student_t_pvalue",
    "validate_branch_prediction",
    "validate_calibration_evidence",
    "validate_calibration_universe_plan",
    "validate_lambda_fold_evidence",
]
