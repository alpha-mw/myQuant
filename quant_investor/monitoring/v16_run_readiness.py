"""Fail-closed v16 four-branch readiness contract.

This module is an add-only v16 boundary.  It does not upgrade or reinterpret a
v15 readiness payload and performs no provider, LLM, broker, or order I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quant_investor.codex_review.models import (
    AuthorizationDecision,
    HumanAuthorization,
)
from quant_investor.codex_review.storage import (
    canonical_json_bytes as review_canonical_json_bytes,
)
from quant_investor.codex_review.storage import sha256_bytes as review_sha256_bytes
from quant_investor.factors.governance_transaction_v4 import (
    validate_activation_receipt_v4,
)
from quant_investor.factors.governance_protocol_v4 import (
    semantic_sha256 as factor_semantic_sha256,
)
from quant_investor.factors.governance_quality_v1 import (
    validate_factor_quality_readiness_v1,
)
from quant_investor.factors.runtime import production_factor_set_sha256

SCHEMA_VERSION = "v16_run_readiness.v1"
ARCHITECTURE_VERSION = "16.0.0"
BRANCH_SCHEMA_VERSION = "v16.four-branch"
RESULTS_NAMESPACE = "results/v16"
READINESS_FILENAME = "v16_run_readiness.json"
REQUIRED_BRANCHES = ("quant", "fundamental", "macro", "llm")
MIN_ACTIVE_FACTOR_COUNT = 5
MIN_ACTIVE_FACTOR_FAMILY_COUNT = 3
MAX_FACTOR_ABS_WEIGHT = 0.20
MAX_FACTOR_FAMILY_ABS_WEIGHT = 0.35
MAX_IC_SELECTED = 12
FACTOR_READINESS_SCHEMA_VERSION = "factor-governance-readiness.v4"
FACTOR_PROTOCOL_VERSION = "v4"
CALIBRATION_READINESS_SCHEMA_VERSION = "calibration-readiness.v16.four-evidence"
MIN_CALIBRATION_BRANCH_SAMPLES = 300
MIN_CALIBRATION_BRANCH_COHORTS = 8
MAX_CALIBRATION_ECE = 0.05
MIN_INTERVAL_COVERAGE = 0.85
MAX_INTERVAL_COVERAGE = 0.95
MAX_LAMBDA_FOLD_RANGE = 0.20
EVIDENCE_V2_MIGRATION_BLOCKERS = (
    "global_attempt_registry_authority_not_integrated",
    "evidence_v2_disconnected_from_authorizing_consumers",
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_STATUSES = frozenset({"blocked", "empty", "complete"})


class V16ReadinessError(ValueError):
    """Raised when a v16 readiness artifact violates its contract."""


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(values: Sequence[Any] | None) -> list[str]:
    return sorted({str(value).strip() for value in (values or ()) if str(value).strip()})


def _symbols(values: Any) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _valid_hash(value: Any) -> bool:
    return bool(_HASH_PATTERN.fullmatch(str(value or "").strip().lower()))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _results_v16_relative(path: Path) -> str:
    parts = path.parts
    if ".." in parts:
        raise V16ReadinessError("readiness path must not contain parent traversal")
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("results", "v16"):
            return Path(*parts[index:]).as_posix()
    raise V16ReadinessError(f"readiness path must be below {RESULTS_NAMESPACE}: {path}")


def _validate_readiness_path(path: Path) -> str:
    relative = _results_v16_relative(path)
    if path.name != READINESS_FILENAME:
        raise V16ReadinessError(f"readiness path must end with {READINESS_FILENAME}")
    return relative


def _branch_contract(
    branch_readiness: Mapping[str, Any] | None,
    branch_objects: Mapping[str, Any] | None,
) -> tuple[dict[str, bool], dict[str, bool], list[str]]:
    readiness = _mapping(branch_readiness)
    objects = _mapping(branch_objects)
    materialized: dict[str, bool] = {}
    ready: dict[str, bool] = {}
    blockers: list[str] = []
    unexpected = sorted(set(readiness) - set(REQUIRED_BRANCHES))
    unexpected += sorted(set(objects) - set(REQUIRED_BRANCHES))
    if unexpected:
        blockers.extend(f"unsupported_branch:{branch}" for branch in sorted(set(unexpected)))
    for branch in REQUIRED_BRANCHES:
        branch_payload = _mapping(readiness.get(branch))
        materialized[branch] = objects.get(branch) is True
        branch_blockers = _strings(branch_payload.get("blockers"))
        ready[branch] = bool(
            materialized[branch]
            and str(branch_payload.get("status") or "").strip().lower() in {"pass", "ready"}
            and not branch_blockers
        )
        if not materialized[branch]:
            blockers.append(f"branch_object_missing:{branch}")
        elif not ready[branch]:
            if branch_blockers:
                blockers.extend(
                    f"branch_data_not_ready:{branch}:{item}" for item in branch_blockers
                )
            else:
                blockers.append(f"branch_data_not_ready:{branch}")
    return materialized, ready, blockers


def _factor_quality_summary(factor: Mapping[str, Any]) -> dict[str, Any]:
    unavailable = {
        "availability": "unavailable",
        "schema_version": "missing",
        "policy_hash": None,
        "status": "unavailable",
        "valid": False,
        "report_only": True,
        "quality_ready": False,
        "shadow_observation_eligible": False,
        "factor_count": 0,
        "family_count": 0,
        "qualified_factor_count": 0,
        "qualified_family_count": 0,
        "quality_set_sha256": None,
        "assessment_sha256": None,
        "blockers": [],
    }
    if "quality_assessment" not in factor:
        return unavailable
    quality = factor.get("quality_assessment")
    if not isinstance(quality, Mapping):
        return {
            **unavailable,
            "availability": "invalid",
            "status": "invalid",
            "blockers": ["quality_assessment_invalid"],
        }
    try:
        validated = validate_factor_quality_readiness_v1(quality)
    except (TypeError, ValueError):
        return {
            **unavailable,
            "availability": "invalid",
            "status": "invalid",
            "blockers": ["quality_assessment_invalid"],
        }
    if validated["status"] == "invalid" or validated["input_valid"] is not True:
        return {
            **unavailable,
            "availability": "invalid",
            "schema_version": validated["schema_version"],
            "policy_hash": validated["quality_policy_hash"],
            "status": "invalid",
            "blockers": ["quality_assessment_invalid"],
        }
    return {
        "availability": "available",
        "schema_version": validated["schema_version"],
        "policy_hash": validated["quality_policy_hash"],
        "status": validated["status"],
        "valid": True,
        "report_only": True,
        "quality_ready": validated["quality_ready"],
        "shadow_observation_eligible": validated["shadow_observation_eligible"],
        "factor_count": validated["quality_factor_count"],
        "family_count": validated["quality_family_count"],
        "qualified_factor_count": validated["qualified_factor_count"],
        "qualified_family_count": validated["qualified_family_count"],
        "quality_set_sha256": validated["quality_set_sha256"],
        "assessment_sha256": validated["assessment_sha256"],
        "blockers": list(validated["blockers"]),
    }


def _factor_contract(
    value: Mapping[str, Any] | None,
) -> tuple[int, int, bool, dict[str, Any], list[str]]:
    factor = _mapping(value)
    raw_count = factor.get("production_factor_count", 0)
    factor_count = (
        raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) else 0
    )
    raw_family_count = factor.get("production_family_count", 0)
    family_count = (
        raw_family_count
        if isinstance(raw_family_count, int) and not isinstance(raw_family_count, bool)
        else 0
    )
    factor_blockers = _strings(factor.get("blockers"))
    factor_rows = factor.get("factors")
    factor_rows_valid = isinstance(factor_rows, list)
    factor_names: list[str] = []
    runtime_contract_hashes: list[str] = []
    if factor_rows_valid:
        for row in factor_rows:
            if not isinstance(row, Mapping) or row.get("healthy") is not True:
                factor_rows_valid = False
                continue
            factor_names.append(str(row.get("name") or "").strip())
            runtime_contract_hashes.append(str(row.get("runtime_contract_sha256") or "").strip())
        factor_rows_valid = bool(
            factor_rows_valid
            and len(factor_names) == factor_count
            and len(factor_names) == len(set(factor_names))
            and all(factor_names)
            and all(_valid_hash(item) for item in runtime_contract_hashes)
            and production_factor_set_sha256(sorted(factor_names))
            == factor.get("production_factor_set_sha256")
            and factor_semantic_sha256(sorted(runtime_contract_hashes))
            == factor.get("runtime_contracts_sha256")
        )
    receipt = _mapping(factor.get("activation_receipt"))
    receipt_present = bool(receipt)
    embedded_receipt = _mapping(receipt.get("receipt"))
    receipt_valid = False
    if (
        receipt_present
        and receipt.get("valid") is True
        and _valid_hash(receipt.get("receipt_sha256"))
        and embedded_receipt
        and not _strings(receipt.get("blockers"))
    ):
        try:
            validated_receipt = validate_activation_receipt_v4(
                embedded_receipt,
                expected_as_of=str(factor.get("as_of") or ""),
                expected_protocol_hash=str(factor.get("protocol_hash") or ""),
                expected_registry_file_sha256=str(factor.get("registry_file_sha256") or ""),
                expected_production_factor_set_sha256=str(
                    factor.get("production_factor_set_sha256") or ""
                ),
                expected_runtime_contracts_sha256=str(factor.get("runtime_contracts_sha256") or ""),
            )
            receipt_valid = validated_receipt["receipt_sha256"] == receipt.get("receipt_sha256")
        except (TypeError, ValueError):
            receipt_valid = False
    factor_weights = _mapping(factor.get("normalized_abs_weights"))
    family_weights = _mapping(factor.get("family_normalized_abs_weights"))
    normalized_factor_weights = {
        str(name): _finite(weight) for name, weight in factor_weights.items()
    }
    normalized_family_weights = {
        str(name): _finite(weight) for name, weight in family_weights.items()
    }
    weights_valid = bool(
        len(normalized_factor_weights) == factor_count
        and set(normalized_factor_weights) == set(factor_names)
        and all(
            weight is not None and 0.0 < weight <= MAX_FACTOR_ABS_WEIGHT + 1e-12
            for weight in normalized_factor_weights.values()
        )
        and math.isclose(
            sum(weight or 0.0 for weight in normalized_factor_weights.values()),
            1.0,
            abs_tol=1e-9,
        )
    )
    family_weights_valid = bool(
        len(normalized_family_weights) == family_count
        and all(
            weight is not None and 0.0 < weight <= MAX_FACTOR_FAMILY_ABS_WEIGHT + 1e-12
            for weight in normalized_family_weights.values()
        )
        and math.isclose(
            sum(weight or 0.0 for weight in normalized_family_weights.values()),
            1.0,
            abs_tol=1e-9,
        )
    )
    ready = bool(
        factor.get("schema_version") == FACTOR_READINESS_SCHEMA_VERSION
        and factor.get("protocol_version") == FACTOR_PROTOCOL_VERSION
        and factor_count >= MIN_ACTIVE_FACTOR_COUNT
        and family_count >= MIN_ACTIVE_FACTOR_FAMILY_COUNT
        and factor.get("factor_governance_ready") is True
        and factor.get("new_risk_eligible") is True
        and factor.get("healthy_factor_count") == factor_count
        and factor_rows_valid
        and receipt_valid
        and weights_valid
        and family_weights_valid
        and not factor_blockers
    )
    blockers: list[str] = []
    if factor.get("schema_version") != FACTOR_READINESS_SCHEMA_VERSION:
        blockers.append("factor_readiness_schema_not_v4")
    if factor.get("protocol_version") != FACTOR_PROTOCOL_VERSION:
        blockers.append("factor_protocol_not_v4")
    if factor_count < MIN_ACTIVE_FACTOR_COUNT:
        blockers.append(
            "factor_count_below_minimum:"
            f"actual={factor_count}:required={MIN_ACTIVE_FACTOR_COUNT}"
        )
    if family_count < MIN_ACTIVE_FACTOR_FAMILY_COUNT:
        blockers.append(
            "factor_family_count_below_minimum:"
            f"actual={family_count}:required={MIN_ACTIVE_FACTOR_FAMILY_COUNT}"
        )
    if not factor_rows_valid:
        blockers.append("factor_rows_or_bound_hashes_invalid")
    if not receipt_present:
        blockers.append("factor_activation_receipt_missing")
    elif not receipt_valid:
        blockers.append("factor_activation_receipt_missing_or_invalid")
    if not weights_valid:
        blockers.append("factor_weight_limit_or_normalization_invalid")
    if not family_weights_valid:
        blockers.append("factor_family_weight_limit_or_normalization_invalid")
    blockers.extend(f"factor_governance_not_ready:{item}" for item in factor_blockers)
    if not ready and not blockers:
        blockers.append("factor_governance_not_ready")
    summary = {
        "schema_version": str(factor.get("schema_version") or "missing"),
        "protocol_version": str(factor.get("protocol_version") or "missing"),
        "factor_count": factor_count,
        "family_count": family_count,
        "activation_receipt": {
            "present": receipt_present,
            "valid": receipt_valid,
            "sha256": (
                str(receipt.get("receipt_sha256") or "").strip().lower()
                if _valid_hash(receipt.get("receipt_sha256"))
                else None
            ),
        },
        "max_factor_abs_weight": max(
            (weight or 0.0 for weight in normalized_factor_weights.values()),
            default=0.0,
        ),
        "max_family_abs_weight": max(
            (weight or 0.0 for weight in normalized_family_weights.values()),
            default=0.0,
        ),
        "normalized_abs_weights": {
            name: weight for name, weight in sorted(normalized_factor_weights.items())
        },
        "family_normalized_abs_weights": {
            name: weight for name, weight in sorted(normalized_family_weights.items())
        },
        "quality_assessment": _factor_quality_summary(factor),
        "blockers": factor_blockers,
    }
    return factor_count, family_count, ready, summary, blockers


def _calibration_contract(
    value: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any], list[str]]:
    calibration = _mapping(value)
    expected_fields = {
        "schema_version",
        "branches",
        "metrics",
        "artifact_sha256",
        "blockers",
    }
    calibration_blockers = _strings(calibration.get("blockers"))
    shape_valid = set(calibration) == expected_fields
    schema_valid = calibration.get("schema_version") == CALIBRATION_READINESS_SCHEMA_VERSION
    branches = _mapping(calibration.get("branches"))
    branch_shape_valid = set(branches) == set(REQUIRED_BRANCHES)
    branch_summaries: dict[str, dict[str, int]] = {}
    branches_ready = branch_shape_valid
    for branch in REQUIRED_BRANCHES:
        branch_payload = _mapping(branches.get(branch))
        samples = branch_payload.get("samples", 0)
        cohorts = branch_payload.get("nonoverlap_cohorts", 0)
        if not isinstance(samples, int) or isinstance(samples, bool):
            samples = 0
        if not isinstance(cohorts, int) or isinstance(cohorts, bool):
            cohorts = 0
        branch_summaries[branch] = {
            "samples": samples,
            "nonoverlap_cohorts": cohorts,
        }
        branches_ready = bool(
            branches_ready
            and set(branch_payload) == {"samples", "nonoverlap_cohorts"}
            and samples >= MIN_CALIBRATION_BRANCH_SAMPLES
            and cohorts >= MIN_CALIBRATION_BRANCH_COHORTS
        )
    metrics = _mapping(calibration.get("metrics"))
    metric_fields = {
        "brier_bootstrap_upper",
        "brier_baseline",
        "logloss_bootstrap_upper",
        "logloss_baseline",
        "ece",
        "interval_coverage",
        "alpha_mae",
        "zero_alpha_mae",
        "top_bucket_edge_lower",
        "lambda_fold_min",
        "lambda_fold_max",
    }
    metric_values = {field: _finite(metrics.get(field)) for field in metric_fields}
    metrics_shape_valid = set(metrics) == metric_fields and all(
        value is not None for value in metric_values.values()
    )
    brier_ready = bool(
        metrics_shape_valid
        and 0.0 <= metric_values["brier_bootstrap_upper"] < metric_values["brier_baseline"] <= 1.0
    )
    logloss_ready = bool(
        metrics_shape_valid
        and 0.0 <= metric_values["logloss_bootstrap_upper"] < metric_values["logloss_baseline"]
    )
    ece_ready = bool(metrics_shape_valid and 0.0 <= metric_values["ece"] <= MAX_CALIBRATION_ECE)
    coverage_ready = bool(
        metrics_shape_valid
        and MIN_INTERVAL_COVERAGE <= metric_values["interval_coverage"] <= MAX_INTERVAL_COVERAGE
    )
    alpha_ready = bool(
        metrics_shape_valid and 0.0 <= metric_values["alpha_mae"] < metric_values["zero_alpha_mae"]
    )
    edge_ready = bool(metrics_shape_valid and metric_values["top_bucket_edge_lower"] > 0.0)
    lambda_ready = bool(
        metrics_shape_valid
        and 0.0 <= metric_values["lambda_fold_min"] <= metric_values["lambda_fold_max"] <= 1.0
        and metric_values["lambda_fold_max"] - metric_values["lambda_fold_min"]
        <= MAX_LAMBDA_FOLD_RANGE + 1e-12
    )
    artifact_valid = _valid_hash(calibration.get("artifact_sha256"))
    ready = bool(
        shape_valid
        and schema_valid
        and branches_ready
        and metrics_shape_valid
        and brier_ready
        and logloss_ready
        and ece_ready
        and coverage_ready
        and alpha_ready
        and edge_ready
        and lambda_ready
        and artifact_valid
        and not calibration_blockers
    )
    blockers = [f"calibration_not_ready:{item}" for item in calibration_blockers]
    checks = {
        "shape": shape_valid,
        "schema": schema_valid,
        "branch_samples_and_cohorts": branches_ready,
        "brier_bootstrap_upper_better_than_baseline": brier_ready,
        "logloss_bootstrap_upper_better_than_baseline": logloss_ready,
        "ece_lte_0_05": ece_ready,
        "interval_coverage_0_85_to_0_95": coverage_ready,
        "alpha_mae_better_than_zero": alpha_ready,
        "top_bucket_edge_lower_gt_zero": edge_ready,
        "lambda_fold_range_lte_0_20": lambda_ready,
        "artifact_sha256": artifact_valid,
    }
    blockers.extend(
        f"calibration_gate_failed:{name}" for name, passed in checks.items() if not passed
    )
    if not ready:
        blockers.append("calibration_threshold_not_met")
    summary = {
        "schema_version": str(calibration.get("schema_version") or "missing"),
        "branches": branch_summaries,
        "metrics": {field: metric_values[field] for field in sorted(metric_fields)},
        "checks": checks,
        "artifact_sha256": (
            str(calibration.get("artifact_sha256") or "").strip().lower()
            if artifact_valid
            else None
        ),
        "blockers": calibration_blockers,
    }
    return ready, summary, blockers


def build_v16_run_readiness(
    *,
    run_id: str,
    generated_at: str,
    analysis_trade_date: str,
    market_data_ready: bool,
    market_data_blockers: Sequence[Any] | None,
    branch_readiness: Mapping[str, Any] | None,
    branch_objects: Mapping[str, Any] | None,
    factor_governance: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    candidate_decision: Mapping[str, Any] | None,
    eligibility: Mapping[str, Any] | None,
    handoff: Mapping[str, Any] | None,
    execution_plan: Mapping[str, Any] | None,
    activation_gates: Mapping[str, Any] | None = None,
    human_authorization: Mapping[str, Any] | None = None,
    risk_reduction_quote_gate: Mapping[str, Any] | None = None,
    material_warnings: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic v16 new-risk decision and blocker inventory."""

    run_id_text = str(run_id or "").strip()
    generated_at_text = str(generated_at or "").strip()
    trade_date_text = str(analysis_trade_date or "").strip()
    if not run_id_text or not generated_at_text or not trade_date_text:
        raise V16ReadinessError("run_id, generated_at, and analysis_trade_date are required")

    objects, branches_ready, branch_blockers = _branch_contract(branch_readiness, branch_objects)
    (
        factor_count,
        factor_family_count,
        factor_ready,
        factor_summary,
        factor_blockers,
    ) = _factor_contract(factor_governance)
    calibration_ready, calibration_summary, calibration_blockers = _calibration_contract(
        calibration
    )

    candidate = _mapping(candidate_decision)
    candidate_status = str(candidate.get("candidate_decision_status") or "blocked").strip().lower()
    if candidate_status not in _CANDIDATE_STATUSES:
        candidate_status = "blocked"
    selected_symbols = _symbols(candidate.get("selected_symbols"))
    selected_count_valid = len(selected_symbols) <= MAX_IC_SELECTED
    candidate_ready = bool(
        candidate_status == "complete" and selected_symbols and selected_count_valid
    )

    eligibility_payload = _mapping(eligibility)
    eligibility_blockers = _strings(eligibility_payload.get("blockers"))
    eligibility_ready = bool(
        eligibility_payload.get("eligible") is True and not eligibility_blockers
    )

    execution = _mapping(execution_plan)
    execution_blockers = _strings(execution.get("blockers"))
    execution_plan_sha256 = canonical_sha256(execution)
    execution_symbols = _symbols(execution.get("selected_symbols"))
    execution_ready = bool(
        execution
        and execution.get("valid") is True
        and execution.get("broker_side_effects") is False
        and execution_symbols == selected_symbols
        and not execution_blockers
    )

    handoff_payload = _mapping(handoff)
    handoff_present = bool(handoff_payload)
    handoff_blockers = _strings(handoff_payload.get("blockers"))
    handoff_status = str(handoff_payload.get("status") or "missing").strip().lower()
    handoff_ready = bool(
        handoff_present
        and handoff_status in {"complete", "ready"}
        and handoff_payload.get("execution_plan_sha256") == execution_plan_sha256
        and _valid_hash(handoff_payload.get("artifact_sha256"))
        and _valid_hash(handoff_payload.get("stage2_response_sha256"))
        and _valid_hash(handoff_payload.get("capital_map_sha256"))
        and _valid_hash(handoff_payload.get("authorization_receipt_sha256"))
        and not handoff_blockers
    )

    authorization = _mapping(human_authorization)
    authorization_valid = False
    authorization_receipt_sha256: str | None = None
    try:
        authorization_model = HumanAuthorization.model_validate(authorization)
        authorization_payload = authorization_model.model_dump(mode="json")
        supplied_receipt_sha = authorization_payload.pop("receipt_sha256")
        expected_receipt_sha = review_sha256_bytes(
            review_canonical_json_bytes(authorization_payload)
        )
        generated_at_dt = datetime.fromisoformat(generated_at_text.replace("Z", "+00:00"))
        authorization_receipt_sha256 = supplied_receipt_sha
        authorization_valid = bool(
            supplied_receipt_sha == expected_receipt_sha
            and authorization_model.decision == AuthorizationDecision.AUTHORIZED
            and authorization_model.run_id == run_id_text
            and authorization_model.stage2_response_sha256
            == handoff_payload.get("stage2_response_sha256")
            and authorization_model.capital_map_sha256 == handoff_payload.get("capital_map_sha256")
            and supplied_receipt_sha == handoff_payload.get("authorization_receipt_sha256")
            and authorization_model.authorized_at
            <= generated_at_dt
            < authorization_model.expires_at
        )
    except (TypeError, ValueError, ValidationError):
        authorization_valid = False

    activation_payload = _mapping(activation_gates)
    codex_activation_ready = activation_payload.get("codex_ready") is True
    dashboard_activation_ready = activation_payload.get("dashboard_ready") is True
    activation_blockers = _strings(activation_payload.get("blockers"))
    if not factor_ready:
        activation_blockers.append("activation_factor_gate_not_ready")
    if not calibration_ready:
        activation_blockers.append("activation_calibration_gate_not_ready")
    if not codex_activation_ready:
        activation_blockers.append("activation_codex_gate_not_ready")
    if not dashboard_activation_ready:
        activation_blockers.append("activation_dashboard_gate_not_ready")
    activation_blockers.extend(EVIDENCE_V2_MIGRATION_BLOCKERS)
    activation_blockers = sorted(set(activation_blockers))
    activation_candidate = bool(
        factor_ready
        and calibration_ready
        and codex_activation_ready
        and dashboard_activation_ready
        and not activation_blockers
    )

    blockers = _strings(market_data_blockers)
    if market_data_ready is not True:
        blockers.append("market_data_not_ready")
    blockers.extend(branch_blockers)
    blockers.extend(factor_blockers)
    blockers.extend(calibration_blockers)
    if candidate_status == "blocked":
        blockers.append(str(candidate.get("blocker") or "candidate_decision_blocked"))
    elif candidate_status == "empty" or not selected_symbols:
        blockers.append(str(candidate.get("blocker") or "no_candidate_selected_by_ic_coordinator"))
    if not selected_count_valid:
        blockers.append(
            f"ic_selected_count_exceeds_limit:actual={len(selected_symbols)}:"
            f"maximum={MAX_IC_SELECTED}"
        )
    if not eligibility_ready:
        blockers.extend(f"eligibility_not_ready:{item}" for item in eligibility_blockers)
        if not eligibility_blockers:
            blockers.append("eligibility_not_ready")
    if not execution_ready:
        blockers.extend(f"execution_not_ready:{item}" for item in execution_blockers)
        if not execution_blockers:
            blockers.append("execution_plan_not_ready")
    if not handoff_present:
        blockers.append("handoff_missing")
    elif not handoff_ready:
        blockers.extend(f"handoff_not_ready:{item}" for item in handoff_blockers)
        blockers.append("handoff_not_ready")
    if not authorization_valid:
        blockers.append("new_risk_human_authorization_missing_or_invalid")
    blockers.extend(activation_blockers)
    blockers = sorted(set(item for item in blockers if item))

    new_risk_authorized = bool(
        market_data_ready is True
        and all(branches_ready.values())
        and factor_ready
        and calibration_ready
        and candidate_ready
        and eligibility_ready
        and execution_ready
        and handoff_ready
        and authorization_valid
        and activation_candidate
    )
    execution_status = "authorized" if new_risk_authorized else "no_new_risk"
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "results_namespace": RESULTS_NAMESPACE,
        "run_id": run_id_text,
        "generated_at": generated_at_text,
        "analysis_trade_date": trade_date_text,
        "market_data_ready": market_data_ready is True,
        "branch_objects_materialized": objects,
        "branch_data_ready": branches_ready,
        "factor_governance_ready": factor_ready,
        "factor_governance": factor_summary,
        "active_factor_count": factor_count,
        "minimum_active_factor_count": MIN_ACTIVE_FACTOR_COUNT,
        "active_factor_family_count": factor_family_count,
        "minimum_active_factor_family_count": MIN_ACTIVE_FACTOR_FAMILY_COUNT,
        "maximum_factor_abs_weight": MAX_FACTOR_ABS_WEIGHT,
        "maximum_factor_family_abs_weight": MAX_FACTOR_FAMILY_ABS_WEIGHT,
        "calibration_ready": calibration_ready,
        "calibration": calibration_summary,
        "activation_candidate": activation_candidate,
        "activation_blockers": activation_blockers,
        "activation_gates": {
            "factor_ready": factor_ready,
            "calibration_ready": calibration_ready,
            "codex_ready": codex_activation_ready,
            "dashboard_ready": dashboard_activation_ready,
        },
        "candidate_decision_status": candidate_status,
        "selected_symbols": selected_symbols,
        "selected_symbol_count": len(selected_symbols),
        "maximum_selected_symbol_count": MAX_IC_SELECTED,
        "eligibility": {
            "eligible": eligibility_ready,
            "blockers": eligibility_blockers,
        },
        "handoff": {
            "present": handoff_present,
            "status": handoff_status,
            "valid": handoff_ready,
            "artifact_sha256": (
                str(handoff_payload.get("artifact_sha256") or "").strip().lower()
                if _valid_hash(handoff_payload.get("artifact_sha256"))
                else None
            ),
            "execution_plan_sha256": execution_plan_sha256,
            "stage2_response_sha256": handoff_payload.get("stage2_response_sha256"),
            "capital_map_sha256": handoff_payload.get("capital_map_sha256"),
            "authorization_receipt_sha256": handoff_payload.get("authorization_receipt_sha256"),
            "blockers": handoff_blockers,
        },
        "execution": {
            "status": execution_status,
            "plan_valid": execution_ready,
            "plan_sha256": execution_plan_sha256,
            "new_risk_authorized": new_risk_authorized,
            "broker_side_effects": False,
            "blockers": blockers,
        },
        "human_authorization": {
            "present": bool(authorization),
            "valid": authorization_valid,
            "receipt_sha256": authorization_receipt_sha256,
        },
        "new_risk_authorized": new_risk_authorized,
        "readiness_status": "ready" if new_risk_authorized else "no_new_risk",
        "risk_reduction_quote_gate": _mapping(risk_reduction_quote_gate),
        "blockers": blockers,
        "material_warnings": _strings(material_warnings),
    }


def validate_v16_run_readiness(payload: Mapping[str, Any]) -> None:
    readiness = _mapping(payload)
    if not readiness:
        raise V16ReadinessError("v16 readiness must be a mapping")
    expected_fields = {
        "schema_version",
        "architecture_version",
        "branch_schema_version",
        "results_namespace",
        "run_id",
        "generated_at",
        "analysis_trade_date",
        "market_data_ready",
        "branch_objects_materialized",
        "branch_data_ready",
        "factor_governance_ready",
        "factor_governance",
        "active_factor_count",
        "minimum_active_factor_count",
        "active_factor_family_count",
        "minimum_active_factor_family_count",
        "maximum_factor_abs_weight",
        "maximum_factor_family_abs_weight",
        "calibration_ready",
        "calibration",
        "activation_candidate",
        "activation_blockers",
        "activation_gates",
        "candidate_decision_status",
        "selected_symbols",
        "selected_symbol_count",
        "maximum_selected_symbol_count",
        "eligibility",
        "handoff",
        "execution",
        "human_authorization",
        "new_risk_authorized",
        "readiness_status",
        "risk_reduction_quote_gate",
        "blockers",
        "material_warnings",
    }
    if set(readiness) != expected_fields:
        missing = sorted(expected_fields - set(readiness))
        unexpected = sorted(set(readiness) - expected_fields)
        raise V16ReadinessError(
            "v16 readiness fields mismatch: " f"missing={missing}, unexpected={unexpected}"
        )
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "results_namespace": RESULTS_NAMESPACE,
    }
    for field, expected in envelope.items():
        if readiness.get(field) != expected:
            raise V16ReadinessError(
                f"{field} mismatch: expected {expected!r}, got {readiness.get(field)!r}"
            )
    for field in ("run_id", "generated_at", "analysis_trade_date"):
        if not isinstance(readiness.get(field), str) or not readiness[field].strip():
            raise V16ReadinessError(f"{field} must be a non-empty string")
    if not isinstance(readiness.get("market_data_ready"), bool):
        raise V16ReadinessError("market_data_ready must be boolean")
    for field in ("factor_governance_ready", "calibration_ready"):
        if not isinstance(readiness.get(field), bool):
            raise V16ReadinessError(f"{field} must be boolean")
    for field in ("branch_objects_materialized", "branch_data_ready"):
        branch_map = _mapping(readiness.get(field))
        if set(branch_map) != set(REQUIRED_BRANCHES):
            raise V16ReadinessError(f"{field} must contain exactly {','.join(REQUIRED_BRANCHES)}")
        if not all(isinstance(value, bool) for value in branch_map.values()):
            raise V16ReadinessError(f"{field} values must be boolean")
    active_count = readiness.get("active_factor_count")
    if not isinstance(active_count, int) or isinstance(active_count, bool) or active_count < 0:
        raise V16ReadinessError("active_factor_count must be a non-negative integer")
    if readiness.get("minimum_active_factor_count") != MIN_ACTIVE_FACTOR_COUNT:
        raise V16ReadinessError("minimum_active_factor_count mismatch")
    family_count = readiness.get("active_factor_family_count")
    if not isinstance(family_count, int) or isinstance(family_count, bool) or family_count < 0:
        raise V16ReadinessError("active_factor_family_count must be a non-negative integer")
    if readiness.get("minimum_active_factor_family_count") != MIN_ACTIVE_FACTOR_FAMILY_COUNT:
        raise V16ReadinessError("minimum_active_factor_family_count mismatch")
    if readiness.get("maximum_factor_abs_weight") != MAX_FACTOR_ABS_WEIGHT:
        raise V16ReadinessError("maximum_factor_abs_weight mismatch")
    if readiness.get("maximum_factor_family_abs_weight") != MAX_FACTOR_FAMILY_ABS_WEIGHT:
        raise V16ReadinessError("maximum_factor_family_abs_weight mismatch")
    factor_summary = _mapping(readiness.get("factor_governance"))
    receipt = _mapping(factor_summary.get("activation_receipt"))
    factor_weights = _mapping(factor_summary.get("normalized_abs_weights"))
    family_weights = _mapping(factor_summary.get("family_normalized_abs_weights"))
    factor_weight_values = [_finite(value) for value in factor_weights.values()]
    family_weight_values = [_finite(value) for value in family_weights.values()]
    if readiness.get("factor_governance_ready") is True:
        if factor_summary.get("schema_version") != FACTOR_READINESS_SCHEMA_VERSION:
            raise V16ReadinessError("ready factor gate must use schema v4")
        if factor_summary.get("protocol_version") != FACTOR_PROTOCOL_VERSION:
            raise V16ReadinessError("ready factor gate must use protocol v4")
        if active_count < MIN_ACTIVE_FACTOR_COUNT:
            raise V16ReadinessError("ready factor gate has too few factors")
        if family_count < MIN_ACTIVE_FACTOR_FAMILY_COUNT:
            raise V16ReadinessError("ready factor gate has too few families")
        if receipt.get("present") is not True or receipt.get("valid") is not True:
            raise V16ReadinessError("ready factor gate requires activation receipt")
        if not _valid_hash(receipt.get("sha256")):
            raise V16ReadinessError("ready factor gate receipt sha256 invalid")
        if len(factor_weight_values) != active_count or not all(
            value is not None and 0.0 < value <= MAX_FACTOR_ABS_WEIGHT + 1e-12
            for value in factor_weight_values
        ):
            raise V16ReadinessError("ready factor weights violate v4 limits")
        if not math.isclose(
            sum(value or 0.0 for value in factor_weight_values),
            1.0,
            abs_tol=1e-9,
        ):
            raise V16ReadinessError("ready factor weights are not normalized")
        if len(family_weight_values) != family_count or not all(
            value is not None and 0.0 < value <= MAX_FACTOR_FAMILY_ABS_WEIGHT + 1e-12
            for value in family_weight_values
        ):
            raise V16ReadinessError("ready family weights violate v4 limits")
        if not math.isclose(
            sum(value or 0.0 for value in family_weight_values),
            1.0,
            abs_tol=1e-9,
        ):
            raise V16ReadinessError("ready family weights are not normalized")
        if factor_summary.get("max_factor_abs_weight") != max(factor_weight_values):
            raise V16ReadinessError("factor max-weight summary mismatch")
        if factor_summary.get("max_family_abs_weight") != max(family_weight_values):
            raise V16ReadinessError("family max-weight summary mismatch")
    activation_candidate = readiness.get("activation_candidate")
    if not isinstance(activation_candidate, bool):
        raise V16ReadinessError("activation_candidate must be boolean")
    activation_blockers = readiness.get("activation_blockers")
    if not isinstance(activation_blockers, list) or activation_blockers != sorted(
        set(activation_blockers)
    ):
        raise V16ReadinessError("activation_blockers must be a sorted unique array")
    if not set(EVIDENCE_V2_MIGRATION_BLOCKERS).issubset(activation_blockers):
        raise V16ReadinessError(
            "activation_blockers must preserve disconnected evidence-v2 migration gates"
        )
    activation_gates = _mapping(readiness.get("activation_gates"))
    if set(activation_gates) != {
        "factor_ready",
        "calibration_ready",
        "codex_ready",
        "dashboard_ready",
    } or not all(isinstance(value, bool) for value in activation_gates.values()):
        raise V16ReadinessError("activation_gates contract invalid")
    if activation_gates["factor_ready"] is not readiness["factor_governance_ready"]:
        raise V16ReadinessError("activation factor gate mismatch")
    if activation_gates["calibration_ready"] is not readiness["calibration_ready"]:
        raise V16ReadinessError("activation calibration gate mismatch")
    expected_activation_candidate = bool(all(activation_gates.values()) and not activation_blockers)
    if activation_candidate != expected_activation_candidate:
        raise V16ReadinessError("activation_candidate gate mismatch")
    if not activation_candidate and not activation_blockers:
        raise V16ReadinessError("non-candidate activation must include activation blockers")
    calibration_summary = _mapping(readiness.get("calibration"))
    calibration_checks = _mapping(calibration_summary.get("checks"))
    input_shape_valid = calibration_checks.get("shape")
    if not isinstance(input_shape_valid, bool):
        raise V16ReadinessError("calibration shape check must be boolean")
    recomputed_calibration_ready, recomputed_calibration_summary, _ = _calibration_contract(
        {
            "schema_version": calibration_summary.get("schema_version"),
            "branches": calibration_summary.get("branches"),
            "metrics": calibration_summary.get("metrics"),
            "artifact_sha256": calibration_summary.get("artifact_sha256"),
            "blockers": calibration_summary.get("blockers"),
        }
    )
    # The summary is intentionally normalized even when the source mapping was
    # missing or malformed. Preserve the builder's source-shape result while
    # recomputing every semantic field from that normalized summary.
    recomputed_calibration_summary["checks"]["shape"] = input_shape_valid
    recomputed_calibration_ready = bool(
        recomputed_calibration_ready and input_shape_valid
    )
    if calibration_summary != recomputed_calibration_summary:
        raise V16ReadinessError("calibration summary is not canonical v16 data")
    if readiness.get("calibration_ready") is not recomputed_calibration_ready:
        raise V16ReadinessError("calibration_ready does not match explicit metrics")
    selected = _symbols(readiness.get("selected_symbols"))
    if len(selected) != readiness.get("selected_symbol_count"):
        raise V16ReadinessError("selected_symbol_count mismatch")
    if len(selected) > MAX_IC_SELECTED:
        raise V16ReadinessError("selected_symbol_count exceeds v16 IC limit")
    if readiness.get("maximum_selected_symbol_count") != MAX_IC_SELECTED:
        raise V16ReadinessError("maximum_selected_symbol_count mismatch")
    new_risk = readiness.get("new_risk_authorized")
    if not isinstance(new_risk, bool):
        raise V16ReadinessError("new_risk_authorized must be boolean")
    blockers = readiness.get("blockers")
    if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
        raise V16ReadinessError("blockers must be a sorted unique array")
    if not set(EVIDENCE_V2_MIGRATION_BLOCKERS).issubset(blockers):
        raise V16ReadinessError(
            "blockers must preserve disconnected evidence-v2 migration gates"
        )
    execution = _mapping(readiness.get("execution"))
    if execution.get("new_risk_authorized") is not new_risk:
        raise V16ReadinessError("execution authorization mismatch")
    if execution.get("broker_side_effects") is not False:
        raise V16ReadinessError("readiness cannot claim broker side effects")
    if execution.get("blockers") != blockers:
        raise V16ReadinessError("execution blocker inventory mismatch")
    expected_execution_status = "authorized" if new_risk else "no_new_risk"
    if execution.get("status") != expected_execution_status:
        raise V16ReadinessError("execution status mismatch")
    expected_status = "ready" if new_risk else "no_new_risk"
    if readiness.get("readiness_status") != expected_status:
        raise V16ReadinessError("readiness_status mismatch")
    if new_risk:
        required_true = (
            readiness.get("market_data_ready"),
            readiness.get("factor_governance_ready"),
            readiness.get("calibration_ready"),
            _mapping(readiness.get("eligibility")).get("eligible"),
            _mapping(readiness.get("handoff")).get("valid"),
            _mapping(readiness.get("human_authorization")).get("valid"),
            execution.get("plan_valid"),
            activation_candidate,
        )
        if not all(value is True for value in required_true):
            raise V16ReadinessError("authorized readiness has a failed required gate")
        if active_count < MIN_ACTIVE_FACTOR_COUNT or blockers:
            raise V16ReadinessError("authorized readiness has factor or blocker failure")
        if not all(_mapping(readiness.get("branch_data_ready")).values()):
            raise V16ReadinessError("authorized readiness requires all four branches")
    elif not blockers:
        raise V16ReadinessError("no-new-risk readiness must include blockers")


def readiness_reference(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_v16_run_readiness(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "path": _validate_readiness_path(path),
        "sha256": canonical_sha256(payload),
        "new_risk_authorized": payload["new_risk_authorized"],
        "blockers": list(payload["blockers"]),
        "activation_candidate": payload["activation_candidate"],
        "activation_blockers": list(payload["activation_blockers"]),
    }


def write_v16_run_readiness(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically persist owner-only v16 readiness below ``results/v16``."""

    validate_v16_run_readiness(payload)
    reference = readiness_reference(path, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise RuntimeError("v16 run readiness readback hash mismatch")
    return reference


def load_v16_run_readiness(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    _validate_readiness_path(path)
    if path.is_symlink() or not path.is_file():
        raise V16ReadinessError("v16 readiness must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V16ReadinessError("v16 readiness must be a JSON object")
    validate_v16_run_readiness(payload)
    if canonical_sha256(payload) != str(expected_sha256 or "").strip().lower():
        raise V16ReadinessError("v16 readiness sha256 mismatch")
    return payload


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_SCHEMA_VERSION",
    "MAX_IC_SELECTED",
    "MIN_ACTIVE_FACTOR_COUNT",
    "READINESS_FILENAME",
    "REQUIRED_BRANCHES",
    "RESULTS_NAMESPACE",
    "SCHEMA_VERSION",
    "V16ReadinessError",
    "build_v16_run_readiness",
    "canonical_sha256",
    "load_v16_run_readiness",
    "readiness_reference",
    "validate_v16_run_readiness",
    "write_v16_run_readiness",
]
