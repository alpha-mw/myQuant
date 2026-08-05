"""Exact, deterministic orchestration for Sprint R2.2 forward evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
import hashlib
import re
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import (
    CanonicalContractError,
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_runtime.factor_observation import (
    FACTOR_UNIVERSE_OBSERVATION_VERSION,
    FactorObservationError,
    validate_factor_forward_label,
    validate_factor_observation,
)

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    content_ref,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    sha256,
    sorted_exact_refs,
    timestamp,
    validate_content_addressed,
)
from ..evidence.forward_adapter import (
    ExactArtifactReader,
    replay_forward_evaluation_inputs,
)
from ..evidence.models import EVIDENCE_VERSION, validate_evidence
from ..hypothesis.models import HYPOTHESIS_VERSION, validate_hypothesis
from ..memory.chain import append_memory, memory_tip
from ..package import verify_package
from ..regime.engine import REGIME_RECEIPT_VERSION as I0_REGIME_RECEIPT_VERSION
from ..regime.engine import validate_regime_receipt
from ..regime.input import REGIME_INPUT_VERSION
from .receipts import (
    ENVELOPE_VERSION,
    MEMORY_INVENTORY_VERSION,
    POLICY_VERSION,
    REQUEST_VERSION,
    MAX_ENVELOPE_BYTES,
    build_calibration_receipt,
    build_envelope,
    build_hypothesis_receipt,
    build_main_receipt,
    build_memory_proposal,
    build_regime_receipt,
    build_subject_receipt,
    build_universe_inventory,
    build_variant_comparison_receipt,
    receipt_ref,
    research_input_ref,
    validate_memory_inventory,
)
from .regime_evaluator import RegimeEvaluationError

REQUEST_PREFIX: Final = "data/private/research_intelligence/evaluation_requests/"
REQUEST_NAME_RE: Final = re.compile(r"^forward-evaluation-request-[0-9a-f]{64}$")
MAX_REQUEST_BYTES: Final = 4 * 1024 * 1024
MAX_ORIGINS: Final = 64
MAX_FACTORS: Final = 16
MAX_HYPOTHESES: Final = 16
MAX_EVIDENCE: Final = 256
MAX_RULES: Final = 64
MAX_DISTINCT_HYPOTHESIS_WINDOWS: Final = 128
MAX_GLOBAL_REFS: Final = 4096
VARIANT_IDS: Final = (
    "v17-quant-core",
    "v17-quant-plus-industry",
    "v17-quant-plus-industry-theme",
)
VARIANT_METRIC_IDS: Final = {
    "cost_adjusted_return",
    "drawdown",
    "icir",
    "joint_coverage",
    "long_short_spread",
    "rank_ic",
    "turnover",
}
RULE_OPERATORS: Final = {"EQ", "GT", "GTE", "LT", "LTE", "NEQ"}
INPUT_IDENTITY_FIELDS: Final = {
    EVIDENCE_VERSION: "evidence_id",
    HYPOTHESIS_VERSION: "hypothesis_id",
    I0_REGIME_RECEIPT_VERSION: "receipt_id",
    REGIME_INPUT_VERSION: "input_id",
    MEMORY_INVENTORY_VERSION: "inventory_id",
}
REQUEST_FIELDS: Final = {
    "as_of",
    "authority",
    "broker",
    "default_protocol_state",
    "evaluated_at",
    "execution",
    "global_activation_state",
    "memory_inventory_ref",
    "order",
    "origins",
    "policy",
    "production",
    "request_id",
    "research_only",
    "semantic_sha256",
    "trade",
    "version",
}
ORIGIN_FIELDS: Final = {
    "closure_refs",
    "evaluation_refs",
    "factor_observation_bindings",
    "label_ref",
    "origin_id",
    "regime_binding",
    "session_byte_sha256",
    "session_relative_path",
    "universe_factor_id",
    "universe_observation_ref",
    "variant_observation_bindings",
}


class ForwardEvaluationError(IntelligenceContractError):
    """Expected request/input blocker with a stable public error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        preserved_artifact_refs: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.preserved_artifact_refs = tuple(dict(row) for row in preserved_artifact_refs)


class ImplementationIntegrityError(RuntimeError):
    """The installed I0 source manifest cannot establish implementation identity."""


def _authority_closed(row: Mapping[str, Any], *, label: str) -> None:
    if row.get("authority") != NO_AUTHORITY:
        raise IntelligenceContractError(f"{label} authority is open")
    if any(row.get(field) is not False for field in ("broker", "execution", "order", "trade")):
        raise IntelligenceContractError(f"{label} action authority is open")
    if (
        row.get("research_only") is not True
        or row.get("production") is not False
        or row.get("default_protocol_state") != "V15_DEFAULT"
        or row.get("global_activation_state") != "INACTIVE"
    ):
        raise IntelligenceContractError(f"{label} protocol state is open")


def _request_identity(row: Mapping[str, Any]) -> str:
    body = dict(row)
    body.pop("request_id", None)
    body.pop("semantic_sha256", None)
    return "forward-evaluation-request-" + hashlib.sha256(canonical_bytes(body)).hexdigest()


def _request_ref(row: Mapping[str, Any], *, relative_path: str) -> dict[str, str]:
    return {
        "artifact_id": str(row["request_id"]),
        "artifact_version": REQUEST_VERSION,
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(row)).hexdigest(),
        "relative_path": relative_path,
        "semantic_sha256": str(row["semantic_sha256"]),
    }


def _content_ref_fields(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "semantic_sha256",
    }:
        raise IntelligenceContractError(f"{label} is not a content ref")
    return {
        "artifact_id": str(value["artifact_id"]),
        "artifact_version": str(value["artifact_version"]),
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def _policy_input_ref(value: Mapping[str, Any], *, label: str, version: str) -> dict[str, str]:
    ref = research_input_ref(value, label=label)
    if ref["artifact_version"] != version:
        raise IntelligenceContractError(f"{label} version mismatch")
    return ref


def _rule_shape(value: Any, *, label: str) -> None:
    fields = {
        "aggregation",
        "factor_id",
        "label_field",
        "metric_id",
        "operator",
        "threshold",
        "window_end",
        "window_start",
    }
    if type(value) is not dict or set(value) != fields:
        raise IntelligenceContractError(f"{label} shape is invalid")
    if value["aggregation"] != "MEAN" or value["operator"] not in RULE_OPERATORS:
        raise IntelligenceContractError(f"{label} policy is invalid")
    _policy_window(value, label=label)
    identifier(value["factor_id"], label=f"{label}.factor_id")
    identifier(value["label_field"], label=f"{label}.label_field")
    if value["label_field"] != "total_return":
        raise IntelligenceContractError(f"{label}.label_field is not supported in v1")
    identifier(value["metric_id"], label=f"{label}.metric_id")
    decimal_value(value["threshold"], label=f"{label}.threshold")


def _policy_window(value: Mapping[str, Any], *, label: str) -> None:
    boundaries: dict[str, str] = {}
    for field in ("window_start", "window_end"):
        raw = value[field]
        if type(raw) is not str:
            raise IntelligenceContractError(f"{label}.{field} must be a canonical date")
        try:
            parsed = date.fromisoformat(raw)
        except ValueError as exc:
            raise IntelligenceContractError(f"{label}.{field} must be a canonical date") from exc
        if parsed.isoformat() != raw:
            raise IntelligenceContractError(f"{label}.{field} must be a canonical date")
        boundaries[field] = raw
    if boundaries["window_end"] < boundaries["window_start"]:
        raise IntelligenceContractError(f"{label} window is reversed")


def _hypothesis_specs(value: Any, *, factor_ids: set[str]) -> list[dict[str, Any]]:
    if type(value) is not list or not 1 <= len(value) <= MAX_HYPOTHESES:
        raise IntelligenceContractError("hypothesis spec cardinality is invalid")
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(value):
        label = f"hypothesis_specs[{index}]"
        fields = {
            "contrary_rules",
            "evidence_refs",
            "falsification_bindings",
            "hypothesis_ref",
            "min_coverage",
            "min_mature_origins",
            "spec_id",
            "support_rules",
        }
        if type(spec) is not dict or set(spec) != fields:
            raise IntelligenceContractError(f"{label} shape is invalid")
        identifier(spec["spec_id"], label=f"{label}.spec_id")
        _policy_input_ref(
            spec["hypothesis_ref"], label=f"{label}.hypothesis_ref", version=HYPOTHESIS_VERSION
        )
        evidence_refs = spec["evidence_refs"]
        if type(evidence_refs) is not list or not 1 <= len(evidence_refs) <= MAX_EVIDENCE:
            raise IntelligenceContractError(f"{label}.evidence_refs cardinality is invalid")
        normalized_evidence = [
            _policy_input_ref(
                ref,
                label=f"{label}.evidence_refs[{ref_index}]",
                version=EVIDENCE_VERSION,
            )
            for ref_index, ref in enumerate(evidence_refs)
        ]
        if len({ref["artifact_id"] for ref in normalized_evidence}) != len(normalized_evidence):
            raise IntelligenceContractError(f"{label}.evidence_refs contains duplicates")
        support_rules = spec["support_rules"]
        contrary_rules = spec["contrary_rules"]
        if type(support_rules) is not list or not 1 <= len(support_rules) <= MAX_RULES:
            raise IntelligenceContractError(f"{label}.support_rules must be non-empty")
        if type(contrary_rules) is not list or len(contrary_rules) > MAX_RULES:
            raise IntelligenceContractError(f"{label}.contrary_rules cardinality is invalid")
        for role, rules in (("support_rules", support_rules), ("contrary_rules", contrary_rules)):
            for rule_index, rule in enumerate(rules):
                _rule_shape(rule, label=f"{label}.{role}[{rule_index}]")
                if rule["factor_id"] not in factor_ids:
                    raise IntelligenceContractError(f"{label}.{role} references unknown factor")
        bindings = spec["falsification_bindings"]
        if type(bindings) is not list or not 1 <= len(bindings) <= MAX_RULES:
            raise IntelligenceContractError(f"{label}.falsification_bindings must be non-empty")
        for binding_index, binding in enumerate(bindings):
            binding_label = f"{label}.falsification_bindings[{binding_index}]"
            if type(binding) is not dict or set(binding) != {
                "condition_index",
                "factor_id",
                "label_field",
                "metric_id",
                "window_end",
                "window_start",
            }:
                raise IntelligenceContractError(f"{binding_label} shape is invalid")
            if (
                type(binding["condition_index"]) is not int
                or type(binding["condition_index"]) is bool
                or binding["condition_index"] < 0
            ):
                raise IntelligenceContractError(f"{binding_label}.condition_index is invalid")
            if binding["factor_id"] not in factor_ids:
                raise IntelligenceContractError(f"{binding_label} references unknown factor")
            _policy_window(binding, label=binding_label)
            identifier(binding["metric_id"], label=f"{binding_label}.metric_id")
            identifier(binding["label_field"], label=f"{binding_label}.label_field")
            if binding["label_field"] != "total_return":
                raise IntelligenceContractError(
                    f"{binding_label}.label_field is not supported in v1"
                )
        minimum = spec["min_mature_origins"]
        if type(minimum) is not int or type(minimum) is bool or not 1 <= minimum <= MAX_ORIGINS:
            raise IntelligenceContractError(f"{label}.min_mature_origins is invalid")
        decimal_value(
            spec["min_coverage"],
            label=f"{label}.min_coverage",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        rows.append(spec)
    spec_ids = [str(row["spec_id"]) for row in rows]
    if spec_ids != sorted(spec_ids, key=lambda item: item.encode("ascii")) or len(spec_ids) != len(
        set(spec_ids)
    ):
        raise IntelligenceContractError("hypothesis specs are not canonical")
    return rows


def _calibration_specs(value: Any, *, factor_ids: set[str]) -> None:
    if type(value) is not list or len(value) > MAX_EVIDENCE:
        raise IntelligenceContractError("calibration spec cardinality is invalid")
    evidence_ids: list[str] = []
    fields = {
        "evidence_id",
        "factor_id",
        "metric_id",
        "min_mature_count",
        "success_operator",
        "success_threshold",
    }
    for index, spec in enumerate(value):
        label = f"calibration_specs[{index}]"
        if type(spec) is not dict or set(spec) != fields:
            raise IntelligenceContractError(f"{label} shape is invalid")
        evidence_ids.append(sha256(spec["evidence_id"], label=f"{label}.evidence_id"))
        if spec["factor_id"] not in factor_ids:
            raise IntelligenceContractError(f"{label} references unknown factor")
        identifier(spec["metric_id"], label=f"{label}.metric_id")
        if spec["success_operator"] not in RULE_OPERATORS:
            raise IntelligenceContractError(f"{label}.success_operator is invalid")
        decimal_value(spec["success_threshold"], label=f"{label}.success_threshold")
        minimum = spec["min_mature_count"]
        if type(minimum) is not int or type(minimum) is bool or not 1 <= minimum <= MAX_ORIGINS:
            raise IntelligenceContractError(f"{label}.min_mature_count is invalid")
    if evidence_ids != sorted(evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
        raise IntelligenceContractError("calibration specs are not canonical")


def _policy(document: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_content_addressed(document, identity_field="policy_id")
    required = {
        "authority",
        "broker",
        "calibration_specs",
        "created_at",
        "default_protocol_state",
        "execution",
        "factor_specs",
        "global_activation_state",
        "horizon_sessions",
        "hypothesis_specs",
        "min_available_origins",
        "min_industry_mapping_coverage",
        "min_joint_coverage",
        "min_symbols",
        "order",
        "policy_id",
        "production",
        "regime_policy",
        "research_only",
        "semantic_sha256",
        "timestamp",
        "trade",
        "variant_policy",
        "version",
    }
    if set(row) != required or row.get("version") != POLICY_VERSION:
        raise IntelligenceContractError("evaluation policy shape/version mismatch")
    _authority_closed(row, label="policy")
    created_at = timestamp(row.get("created_at"), label="policy.created_at")
    if row.get("timestamp") != created_at:
        raise IntelligenceContractError("policy timestamp must equal created_at")
    horizon = row.get("horizon_sessions")
    if type(horizon) is not int or horizon not in {1, 5, 10, 20, 60}:
        raise IntelligenceContractError("policy horizon is invalid")
    for field, lower, upper in (
        ("min_available_origins", 1, MAX_ORIGINS),
        ("min_symbols", 5, 10000),
    ):
        value = row.get(field)
        if type(value) is not int or type(value) is bool or not lower <= value <= upper:
            raise IntelligenceContractError(f"policy {field} is invalid")
    for field in ("min_joint_coverage", "min_industry_mapping_coverage"):
        decimal_value(
            row.get(field), label=f"policy.{field}", minimum=Decimal("0"), maximum=Decimal("1")
        )
    factor_specs = row.get("factor_specs")
    if type(factor_specs) is not list or not 1 <= len(factor_specs) <= MAX_FACTORS:
        raise IntelligenceContractError("policy factor_specs cardinality is invalid")
    factor_ids: list[str] = []
    for spec in factor_specs:
        if type(spec) is not dict or set(spec) != {
            "direction",
            "expected_rank_ic_sign",
            "factor_id",
            "factor_ref",
        }:
            raise IntelligenceContractError("factor spec shape is invalid")
        factor_ids.append(identifier(spec["factor_id"], label="factor_id"))
        factor_ref = exact_ref(spec["factor_ref"], label="factor_ref")
        if factor_ref["cutoff"] > created_at:
            raise IntelligenceContractError("factor definition postdates policy creation")
        if spec["direction"] not in {"HIGHER_IS_BETTER", "LOWER_IS_BETTER"}:
            raise IntelligenceContractError("factor direction is invalid")
        if spec["expected_rank_ic_sign"] != "POSITIVE":
            raise IntelligenceContractError("oriented expected sign must be POSITIVE")
    if factor_ids != sorted(factor_ids, key=lambda value: value.encode("ascii")) or len(
        factor_ids
    ) != len(set(factor_ids)):
        raise IntelligenceContractError("factor specs are not canonical")
    _hypothesis_specs(row.get("hypothesis_specs"), factor_ids=set(factor_ids))
    _calibration_specs(row.get("calibration_specs"), factor_ids=set(factor_ids))
    variant_policy = row.get("variant_policy")
    if type(variant_policy) is not dict or set(variant_policy) != {"comparison_rules", "variants"}:
        raise IntelligenceContractError("variant policy shape is invalid")
    variants = variant_policy["variants"]
    if type(variants) is not list or any(
        type(value) is not dict or set(value) != {"required", "variant_id", "variant_ref"}
        for value in variants
    ):
        raise IntelligenceContractError("variant spec shape is invalid")
    if [value["variant_id"] for value in variants] != list(VARIANT_IDS):
        raise IntelligenceContractError("variant policy must contain the fixed variants")
    if [value["required"] for value in variants] != [True, False, False]:
        raise IntelligenceContractError("variant required flags are invalid")
    for value in variants:
        variant_ref = exact_ref(value["variant_ref"], label="variant_ref")
        if variant_ref["cutoff"] > created_at:
            raise IntelligenceContractError("variant definition postdates policy creation")
    comparison_rules = variant_policy["comparison_rules"]
    if type(comparison_rules) is not list or len(comparison_rules) != len(VARIANT_METRIC_IDS):
        raise IntelligenceContractError("variant comparison rules are incomplete")
    comparison_ids: list[str] = []
    for index, rule in enumerate(comparison_rules):
        label = f"variant comparison_rules[{index}]"
        if type(rule) is not dict or set(rule) != {
            "degradation_threshold",
            "direction",
            "improvement_threshold",
            "metric_id",
            "tolerance",
        }:
            raise IntelligenceContractError(f"{label} shape is invalid")
        metric_id = str(rule["metric_id"])
        comparison_ids.append(metric_id)
        if metric_id not in VARIANT_METRIC_IDS:
            raise IntelligenceContractError(f"{label}.metric_id is not allowlisted")
        expected_direction = (
            "LOWER_IS_BETTER" if metric_id in {"drawdown", "turnover"} else "HIGHER_IS_BETTER"
        )
        if rule["direction"] != expected_direction:
            raise IntelligenceContractError(f"{label}.direction is invalid")
        for field in ("degradation_threshold", "improvement_threshold", "tolerance"):
            decimal_value(rule[field], label=f"{label}.{field}", minimum=Decimal("0"))
    if set(comparison_ids) != VARIANT_METRIC_IDS or len(comparison_ids) != len(set(comparison_ids)):
        raise IntelligenceContractError("variant comparison rules are not canonical")
    regime_policy = row.get("regime_policy")
    if type(regime_policy) is not dict or set(regime_policy) != {
        "industry_entity_scope",
        "min_stratum_origins",
        "theme_entity_scope",
    }:
        raise IntelligenceContractError("regime policy shape is invalid")
    if (
        regime_policy["industry_entity_scope"] != "GLOBAL_BREADTH"
        or regime_policy["theme_entity_scope"] != "GLOBAL_BREADTH"
    ):
        raise IntelligenceContractError("regime v1 supports global breadth only")
    if (
        type(regime_policy["min_stratum_origins"]) is not int
        or not 1 <= regime_policy["min_stratum_origins"] <= MAX_ORIGINS
    ):
        raise IntelligenceContractError("regime stratum minimum is invalid")
    return row


def _validate_request(document: Mapping[str, Any], *, relative_path: str) -> dict[str, Any]:
    try:
        row = validate_semantic_sha(document)
    except Exception as exc:
        raise IntelligenceContractError("request semantic SHA mismatch") from exc
    if set(row) != REQUEST_FIELDS or row.get("version") != REQUEST_VERSION:
        raise IntelligenceContractError("forward evaluation request shape/version mismatch")
    _authority_closed(row, label="request")
    as_of = timestamp(row.get("as_of"), label="request.as_of")
    if timestamp(row.get("evaluated_at"), label="request.evaluated_at") != as_of:
        raise IntelligenceContractError("evaluated_at must equal as_of")
    expected_id = _request_identity(row)
    if row.get("request_id") != expected_id or REQUEST_NAME_RE.fullmatch(expected_id) is None:
        raise IntelligenceContractError("request ID is not content addressed")
    if relative_path != f"{REQUEST_PREFIX}{expected_id}.json":
        raise IntelligenceContractError("request path is not bound to request ID")
    policy = _policy(row.get("policy", {}))
    origins = row.get("origins")
    if type(origins) is not list or not 1 <= len(origins) <= MAX_ORIGINS:
        raise IntelligenceContractError("request origins cardinality is invalid")
    for origin in origins:
        if type(origin) is not dict or set(origin) != ORIGIN_FIELDS:
            raise IntelligenceContractError("origin shape is invalid")
    origin_ids = [str(value["origin_id"]) for value in origins]
    if origin_ids != sorted(origin_ids, key=lambda value: value.encode("ascii")) or len(
        origin_ids
    ) != len(set(origin_ids)):
        raise IntelligenceContractError("request origins are not canonical")
    _policy_input_ref(
        row.get("memory_inventory_ref", {}),
        label="memory_inventory_ref",
        version=MEMORY_INVENTORY_VERSION,
    )
    if policy["timestamp"] > as_of:
        raise IntelligenceContractError("policy is from the future")
    return row


def _load_request(
    reader: ExactArtifactReader, *, request_path: str, request_sha256: str
) -> tuple[dict[str, Any], bytes]:
    if not request_path.startswith(REQUEST_PREFIX):
        raise ForwardEvaluationError("path_invalid", "request is outside its allowlist")
    try:
        raw = reader.read(request_path, request_sha256)
    except IntelligenceContractError as exc:
        code = "sha_mismatch" if "SHA" in str(exc) else "path_invalid"
        raise ForwardEvaluationError(code, str(exc)) from exc
    if len(raw) > MAX_REQUEST_BYTES:
        raise ForwardEvaluationError("limit_exceeded", "request exceeds byte limit")
    try:
        payload = load_canonical_resource(raw, label=REQUEST_VERSION, max_bytes=MAX_REQUEST_BYTES)
        if type(payload) is not dict or canonical_resource_bytes(payload) != raw:
            raise IntelligenceContractError("request bytes are not canonical")
        row = _validate_request(payload, relative_path=request_path)
    except IntelligenceContractError as exc:
        raise ForwardEvaluationError("request_invalid", str(exc)) from exc
    return row, raw


def _load_research_input(
    reader: ExactArtifactReader, reference: Mapping[str, Any]
) -> dict[str, Any]:
    ref = research_input_ref(reference, label="research_input_ref")
    expected_identity = INPUT_IDENTITY_FIELDS.get(ref["artifact_version"])
    if expected_identity is None:
        raise IntelligenceContractError("research input version is not allowlisted")
    raw = reader.read(ref["relative_path"], ref["byte_sha256"])
    payload = load_canonical_resource(raw, label=ref["artifact_version"])
    if type(payload) is not dict or payload.get("version") != ref["artifact_version"]:
        raise IntelligenceContractError("research input version mismatch")
    row = validate_content_addressed(payload, identity_field=expected_identity)
    if (
        row.get(expected_identity) != ref["artifact_id"]
        or row.get("semantic_sha256") != ref["semantic_sha256"]
        or canonical_resource_bytes(row) != raw
    ):
        raise IntelligenceContractError("research input reference binding mismatch")
    return row


def _dedupe_exact_refs(values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    by_locator: dict[tuple[str, str], dict[str, str]] = {}
    path_hashes: dict[str, str] = {}
    for index, value in enumerate(values):
        ref = exact_ref(value, label=f"observation_refs[{index}]")
        prior_sha = path_hashes.setdefault(ref["relative_path"], ref["byte_sha256"])
        if prior_sha != ref["byte_sha256"]:
            raise IntelligenceContractError("same path declares different SHA")
        key = (ref["relative_path"], ref["byte_sha256"])
        if key in by_locator and by_locator[key] != ref:
            raise IntelligenceContractError("same locator has conflicting exact refs")
        by_locator[key] = ref
    return sorted_exact_refs(list(by_locator.values()), label="observation_refs")


def _metric_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["metric_id"]): dict(row) for row in receipt["metrics"]}


def _evaluation_window(origins: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    sessions = sorted(str(row["origin_session"]) for row in origins)
    return {
        "end_origin_session": sessions[-1],
        "horizon_sessions": horizon,
        "origin_count": len(sessions),
        "start_origin_session": sessions[0],
    }


def _origin_identity(origin: Mapping[str, Any]) -> str:
    return (
        "origin-"
        + hashlib.sha256(
            canonical_bytes(
                {
                    "label_ref": origin["label_ref"],
                    "session_byte_sha256": origin["session_byte_sha256"],
                    "session_relative_path": origin["session_relative_path"],
                    "universe_observation_ref": origin["universe_observation_ref"],
                }
            )
        ).hexdigest()
    )


def _binding_map(
    values: Any, *, id_field: str, label: str, expected_ids: Sequence[str]
) -> dict[str, Mapping[str, Any] | None]:
    if type(values) is not list:
        raise IntelligenceContractError(f"{label} must be a list")
    result: dict[str, Mapping[str, Any] | None] = {}
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != {id_field, "observation_ref"}:
            raise IntelligenceContractError(f"{label}[{index}] shape is invalid")
        subject_id = value[id_field]
        if subject_id in result:
            raise IntelligenceContractError(f"{label} contains duplicate subjects")
        reference = value["observation_ref"]
        if reference is not None:
            reference = exact_ref(
                reference,
                label=f"{label}[{index}].observation_ref",
                expected_versions=(FACTOR_UNIVERSE_OBSERVATION_VERSION,),
            )
        result[str(subject_id)] = reference
    if list(result) != list(expected_ids):
        raise IntelligenceContractError(f"{label} identities/order are not canonical")
    return result


def _document_by_ref(
    references: Sequence[Mapping[str, Any]], documents: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    by_semantic = {str(row["semantic_sha256"]): dict(row) for row in documents}
    for reference in references:
        ref = exact_ref(reference, label="observation_ref")
        document = by_semantic.get(ref["semantic_sha256"])
        if document is None:
            raise IntelligenceContractError("observation document was not replayed")
        result[(ref["relative_path"], ref["byte_sha256"])] = document
    return result


def _subject_origin(
    *,
    origin_id: str,
    observation: Mapping[str, Any],
    label: Mapping[str, Any],
    universe_symbols: Sequence[str],
) -> dict[str, Any]:
    validated_observation = validate_factor_observation(observation)
    validated_label = validate_factor_forward_label(label)
    score_rows = {str(row["symbol"]): row for row in validated_observation["observations"]}
    label_rows = {str(row["symbol"]): row for row in validated_label["label_rows"]}
    if set(score_rows) != set(universe_symbols) or set(label_rows) != set(universe_symbols):
        raise IntelligenceContractError("subject/label domain differs from universe anchor")
    symbol_rows = []
    for symbol in universe_symbols:
        score = score_rows[symbol]
        future = label_rows[symbol]
        symbol_rows.append(
            {
                "cost_adjusted_return": future["cost_adjusted_return"],
                "industry_adjusted_return": future["industry_adjusted_return"],
                "industry_id": future["industry_id"],
                "score": score["value"],
                "score_status": score["status"],
                "symbol": symbol,
                "total_return": future["total_return"],
            }
        )
    return {
        "label_session": validated_label["label_session"],
        "next_open_session": validated_label["shanghai_open_sessions"][1],
        "origin_id": origin_id,
        "origin_session": validated_label["decision_session"],
        "symbol_rows": symbol_rows,
    }


def _replay_origins(
    *,
    workspace_root: str,
    request: Mapping[str, Any],
    reader: ExactArtifactReader,
) -> list[dict[str, Any]]:
    policy = request["policy"]
    factor_specs = {str(row["factor_id"]): row for row in policy["factor_specs"]}
    variant_specs = {str(row["variant_id"]): row for row in policy["variant_policy"]["variants"]}
    runtime_rows: list[dict[str, Any]] = []
    duplicate_keys: set[tuple[Any, ...]] = set()
    total_refs = 0
    for origin in request["origins"]:
        if origin["origin_id"] != _origin_identity(origin):
            raise IntelligenceContractError("origin_id is not content addressed")
        factor_bindings = _binding_map(
            origin["factor_observation_bindings"],
            id_field="factor_id",
            label="factor_observation_bindings",
            expected_ids=list(factor_specs),
        )
        variant_bindings = _binding_map(
            origin["variant_observation_bindings"],
            id_field="variant_id",
            label="variant_observation_bindings",
            expected_ids=VARIANT_IDS,
        )
        if variant_bindings[VARIANT_IDS[0]] is None:
            raise IntelligenceContractError("core variant observation is required")
        universe_ref = exact_ref(
            origin["universe_observation_ref"],
            label="universe_observation_ref",
            expected_versions=(FACTOR_UNIVERSE_OBSERVATION_VERSION,),
        )
        if factor_bindings.get(origin["universe_factor_id"]) != universe_ref:
            raise IntelligenceContractError("universe anchor is not its declared factor binding")
        for reference in factor_bindings.values():
            if reference is None:
                raise IntelligenceContractError("factor observation binding cannot be null")
        observation_refs = _dedupe_exact_refs(
            [
                universe_ref,
                *[value for value in factor_bindings.values() if value is not None],
                *[value for value in variant_bindings.values() if value is not None],
            ]
        )
        label_ref = exact_ref(
            origin["label_ref"],
            label="label_ref",
            expected_versions=("myquant.v17.v4.forward-label.v1",),
        )
        evaluation_refs = sorted_exact_refs(origin["evaluation_refs"], label="evaluation_refs")
        closure_refs = sorted_exact_refs(origin["closure_refs"], label="closure_refs")
        total_refs += len(observation_refs) + 1 + len(evaluation_refs) + len(closure_refs)
        if total_refs > MAX_GLOBAL_REFS:
            raise ForwardEvaluationError("limit_exceeded", "global exact-ref limit exceeded")
        replayed = replay_forward_evaluation_inputs(
            workspace_root=workspace_root,
            session_relative_path=origin["session_relative_path"],
            session_byte_sha256=origin["session_byte_sha256"],
            observation_refs=observation_refs,
            label_refs=[label_ref],
            evaluation_refs=evaluation_refs,
            closure_refs=closure_refs,
            as_of=request["evaluated_at"],
            reader=reader,
        )
        if len(replayed["label_documents"]) != 1:
            raise IntelligenceContractError("one matured label is required per origin")
        label = validate_factor_forward_label(
            replayed["label_documents"][0],
            observation_run_ref=replayed["bundle"]["run_ref"],
        )
        if label["horizon_sessions"] != policy["horizon_sessions"]:
            raise IntelligenceContractError("label horizon differs from evaluation policy")
        document_map = _document_by_ref(observation_refs, replayed["observation_documents"])

        def observation_for(reference: Mapping[str, Any]) -> dict[str, Any]:
            ref = exact_ref(reference, label="subject_observation_ref")
            return document_map[(ref["relative_path"], ref["byte_sha256"])]

        universe_document = validate_factor_observation(observation_for(universe_ref))
        universe_symbols = [str(row["symbol"]) for row in universe_document["observations"]]
        factor_origins: dict[str, dict[str, Any]] = {}
        for factor_id, reference in factor_bindings.items():
            assert reference is not None
            document = validate_factor_observation(observation_for(reference))
            if document["factor_ref"] != factor_specs[factor_id]["factor_ref"]:
                raise IntelligenceContractError("factor observation definition mismatch")
            factor_origins[factor_id] = _subject_origin(
                origin_id=origin["origin_id"],
                observation=document,
                label=label,
                universe_symbols=universe_symbols,
            )
        variant_origins: dict[str, dict[str, Any] | None] = {}
        for variant_id, reference in variant_bindings.items():
            if reference is None:
                variant_origins[variant_id] = None
                continue
            document = validate_factor_observation(observation_for(reference))
            if document["factor_ref"] != variant_specs[variant_id]["variant_ref"]:
                raise IntelligenceContractError("variant observation definition mismatch")
            variant_origins[variant_id] = _subject_origin(
                origin_id=origin["origin_id"],
                observation=document,
                label=label,
                universe_symbols=universe_symbols,
            )
        run = replayed["run"]
        duplicate_key = (
            run["strategy_id"],
            run["decision_session"],
            replayed["bundle"]["run_ref"]["semantic_sha256"],
            label_ref["semantic_sha256"],
            label["horizon_sessions"],
        )
        if duplicate_key in duplicate_keys:
            raise IntelligenceContractError("duplicate semantic origin is rejected")
        duplicate_keys.add(duplicate_key)
        runtime_rows.append(
            {
                "bundle": replayed["bundle"],
                "evaluation_refs": evaluation_refs,
                "factor_origins": factor_origins,
                "factor_refs": {key: value for key, value in factor_bindings.items()},
                "label": label,
                "label_ref": label_ref,
                "origin_id": origin["origin_id"],
                "origin_session": label["decision_session"],
                "regime_binding": origin["regime_binding"],
                "universe_factor_id": origin["universe_factor_id"],
                "universe_observation_ref": universe_ref,
                "variant_origins": variant_origins,
                "variant_refs": variant_bindings,
            }
        )
    runtime_rows.sort(key=lambda row: (row["origin_session"], row["origin_id"]))
    earliest_origin_cutoff = min(row["bundle"]["run_ref"]["cutoff"] for row in runtime_rows)
    preregistered = policy["created_at"] <= earliest_origin_cutoff
    for row in runtime_rows:
        row["preregistered"] = preregistered
    return runtime_rows


def _unavailable_metric_rows() -> list[dict[str, Any]]:
    from .factor_evaluator import FACTOR_METRIC_FORMULA_VERSION, METRIC_IDS

    return [
        {
            "available_origin_count": 0,
            "blocker_codes": ["INSUFFICIENT_AVAILABLE_ORIGINS"],
            "formula_version": FACTOR_METRIC_FORMULA_VERSION,
            "input_origin_ids": [],
            "limitations": [],
            "metric_id": metric_id,
            "sample_count": 0,
            "status": "UNAVAILABLE",
            "unit": (
                "RATIO"
                if "coverage" in metric_id
                or metric_id
                in {
                    "drawdown",
                    "icir",
                    "icir_base",
                    "rank_ic",
                    "stability",
                    "turnover",
                }
                else "RETURN"
            ),
            "value": None,
        }
        for metric_id in METRIC_IDS
    ]


def _evaluate_subjects(
    *,
    policy: Mapping[str, Any],
    origins: Sequence[Mapping[str, Any]],
    universe_ref: Mapping[str, Any],
    evaluated_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .factor_evaluator import evaluate_factor

    evaluation_window = _evaluation_window(origins, int(policy["horizon_sessions"]))
    factor_receipts: list[dict[str, Any]] = []
    for spec in policy["factor_specs"]:
        factor_id = str(spec["factor_id"])
        result = evaluate_factor(
            factor_id=factor_id,
            origins=[row["factor_origins"][factor_id] for row in origins],
            orientation=spec["direction"],
            horizon_sessions=policy["horizon_sessions"],
            min_symbols=policy["min_symbols"],
            min_available_origins=policy["min_available_origins"],
            min_joint_coverage=policy["min_joint_coverage"],
            min_industry_mapping_coverage=policy["min_industry_mapping_coverage"],
        )
        factor_receipts.append(
            build_subject_receipt(
                subject_type="FACTOR",
                subject_id=factor_id,
                subject_ref=spec["factor_ref"],
                evaluation_window=evaluation_window,
                universe_ref=universe_ref,
                observation_refs=[row["factor_refs"][factor_id] for row in origins],
                metrics=result["metrics"],
                origin_metrics=result["origin_metrics"],
                limitations=result["limitations"],
                evaluated_at=evaluated_at,
            )
        )

    variant_receipts: list[dict[str, Any]] = []
    for spec in policy["variant_policy"]["variants"]:
        variant_id = str(spec["variant_id"])
        available_origins = [
            row["variant_origins"][variant_id]
            for row in origins
            if row["variant_origins"][variant_id] is not None
        ]
        observation_refs = [
            row["variant_refs"][variant_id]
            for row in origins
            if row["variant_refs"][variant_id] is not None
        ]
        if not available_origins:
            metrics = _unavailable_metric_rows()
            origin_metrics: Sequence[Mapping[str, Any]] = ()
            limitations = ["OPTIONAL_VARIANT_UNAVAILABLE"]
        else:
            result = evaluate_factor(
                factor_id=variant_id,
                origins=available_origins,
                orientation="HIGHER_IS_BETTER",
                horizon_sessions=policy["horizon_sessions"],
                min_symbols=policy["min_symbols"],
                min_available_origins=policy["min_available_origins"],
                min_joint_coverage=policy["min_joint_coverage"],
                min_industry_mapping_coverage=policy["min_industry_mapping_coverage"],
            )
            metrics = result["metrics"]
            origin_metrics = result["origin_metrics"]
            limitations = result["limitations"]
        variant_receipts.append(
            build_subject_receipt(
                subject_type="VARIANT",
                subject_id=variant_id,
                subject_ref=spec["variant_ref"],
                evaluation_window=evaluation_window,
                universe_ref=universe_ref,
                observation_refs=observation_refs,
                metrics=metrics,
                origin_metrics=origin_metrics,
                limitations=limitations,
                evaluated_at=evaluated_at,
            )
        )
    return factor_receipts, variant_receipts


def _variant_comparison(
    *,
    policy: Mapping[str, Any],
    variant_receipts: Sequence[Mapping[str, Any]],
    preregistered: bool,
    evaluated_at: str,
) -> dict[str, Any]:
    from .variant_evaluator import evaluate_variants

    by_id = {str(row["variant_id"]): row for row in variant_receipts}
    inputs: dict[str, Any] = {}
    for variant_id in VARIANT_IDS:
        receipt = by_id[variant_id]
        metrics = _metric_map(receipt)
        if not receipt["origin_metrics"] and variant_id != VARIANT_IDS[0]:
            inputs[variant_id] = None
            continue
        inputs[variant_id] = {
            "available_origin_ids": [row["origin_id"] for row in receipt["origin_metrics"]],
            "metrics": {
                metric_id: {
                    "input_origin_ids": metrics[metric_id]["input_origin_ids"],
                    "status": metrics[metric_id]["status"],
                    "value": metrics[metric_id]["value"],
                }
                for metric_id in (
                    "long_short_spread",
                    "rank_ic",
                    "icir",
                    "turnover",
                    "drawdown",
                    "joint_coverage",
                    "cost_adjusted_return",
                )
            },
            "status": (
                "UNAVAILABLE"
                if all(row["status"] == "UNAVAILABLE" for row in metrics.values())
                else "PARTIAL"
            ),
        }
    result = evaluate_variants(
        variants=inputs,
        rules=policy["variant_policy"]["comparison_rules"],
    )
    candidate_rows: list[dict[str, Any]] = []
    for comparison in result["comparisons"]:
        conclusion = (
            comparison["conclusion"]
            if preregistered
            else ("UNAVAILABLE" if comparison["conclusion"] == "UNAVAILABLE" else "INCONCLUSIVE")
        )
        comparison_metrics = []
        for row in comparison["metric_comparisons"]:
            comparison_metrics.append(
                {
                    "baseline_value": row["baseline_value"],
                    "blocker_codes": row["blocker_codes"],
                    "candidate_value": row["candidate_value"],
                    "delta": row["improvement_delta"],
                    "direction": row["direction"],
                    "input_origin_ids": row["input_origin_ids"],
                    "metric_id": row["metric_id"],
                    "status": row["status"],
                }
            )
        candidate_rows.append(
            {
                "blocker_codes": comparison["blockers"],
                "candidate_factor_receipt_ref": receipt_ref(
                    by_id[comparison["candidate_variant_id"]]
                ),
                "comparison_metrics": comparison_metrics,
                "dropped_origin_ids": sorted(
                    set(comparison["dropped_baseline_origin_ids"])
                    | set(comparison["dropped_candidate_origin_ids"])
                ),
                "incremental_conclusion": conclusion,
                "paired_origin_ids": comparison["paired_origin_ids"],
                "variant_id": comparison["candidate_variant_id"],
            }
        )
    limitations = list(result["limitations"])
    if not preregistered:
        limitations.append("POSTHOC_VARIANT_COMPARISON_INCONCLUSIVE")
    return build_variant_comparison_receipt(
        baseline_factor_receipt_ref=receipt_ref(by_id[VARIANT_IDS[0]]),
        candidate_rows=candidate_rows,
        limitations=limitations,
        evaluated_at=evaluated_at,
    )


def _authorized_refs(origins: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for origin in origins:
        for value in origin["bundle"]["authorized_evidence_refs"]:
            key = (value["relative_path"], value["byte_sha256"])
            if key not in seen:
                refs.append(dict(value))
                seen.add(key)
    return refs


def _hypothesis_inputs(
    *,
    policy: Mapping[str, Any],
    reader: ExactArtifactReader,
    origins: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hypotheses: list[dict[str, Any]] = []
    helper_specs: list[dict[str, Any]] = []
    all_evidence: dict[str, dict[str, Any]] = {}
    authorized = _authorized_refs(origins)
    for spec in policy["hypothesis_specs"]:
        expected_fields = {
            "contrary_rules",
            "evidence_refs",
            "falsification_bindings",
            "hypothesis_ref",
            "min_coverage",
            "min_mature_origins",
            "spec_id",
            "support_rules",
        }
        if type(spec) is not dict or set(spec) != expected_fields:
            raise IntelligenceContractError("hypothesis evaluation spec shape is invalid")
        hypothesis = _load_research_input(reader, spec["hypothesis_ref"])
        if hypothesis["version"] != HYPOTHESIS_VERSION:
            raise IntelligenceContractError("hypothesis ref version mismatch")
        evidence_docs = [_load_research_input(reader, value) for value in spec["evidence_refs"]]
        if len(all_evidence) + len(evidence_docs) > MAX_EVIDENCE:
            raise ForwardEvaluationError("limit_exceeded", "evidence limit exceeded")
        validated = [
            validate_evidence(value, as_of=hypothesis["timestamp"]) for value in evidence_docs
        ]
        validated_hypothesis = validate_hypothesis(
            hypothesis,
            evidence=validated,
            as_of=hypothesis["timestamp"],
        )
        if validated_hypothesis["timestamp"] > policy["created_at"]:
            raise IntelligenceContractError("hypothesis postdates sealed evaluation policy")
        hypothesis_evidence_refs = sorted(
            [
                *validated_hypothesis["supporting_evidence_refs"],
                *validated_hypothesis["contrary_evidence_refs"],
            ],
            key=lambda ref: (ref["artifact_id"], ref["byte_sha256"]),
        )
        supplied_evidence_refs = sorted(
            [content_ref(value, identity_field="evidence_id") for value in validated],
            key=lambda ref: (ref["artifact_id"], ref["byte_sha256"]),
        )
        if supplied_evidence_refs != hypothesis_evidence_refs:
            raise IntelligenceContractError(
                "hypothesis spec evidence must exactly match sealed hypothesis evidence"
            )
        if any(
            condition["window_sessions"] != policy["horizon_sessions"]
            for condition in validated_hypothesis["falsification_conditions"]
        ):
            raise IntelligenceContractError(
                "hypothesis falsification horizon differs from evaluation policy"
            )
        earliest_origin_cutoff = min(row["bundle"]["run_ref"]["cutoff"] for row in origins)
        if validated_hypothesis["timestamp"] > earliest_origin_cutoff:
            raise IntelligenceContractError("hypothesis was not sealed before evaluation origins")
        for evidence in validated:
            if evidence["source_ref"] not in authorized:
                raise IntelligenceContractError("hypothesis evidence source is not authorized")
            all_evidence[str(evidence["evidence_id"])] = evidence
        helper = dict(spec)
        helper["hypothesis_ref"] = content_ref(validated_hypothesis, identity_field="hypothesis_id")
        helper["evidence_refs"] = [
            content_ref(value, identity_field="evidence_id") for value in validated
        ]
        hypotheses.append(validated_hypothesis)
        helper_specs.append(helper)
    order = sorted(range(len(hypotheses)), key=lambda index: hypotheses[index]["hypothesis_id"])
    return (
        [hypotheses[index] for index in order],
        [helper_specs[index] for index in order],
        [all_evidence[key] for key in sorted(all_evidence)],
    )


def _hypothesis_receipts(
    *,
    policy: Mapping[str, Any],
    origins: Sequence[Mapping[str, Any]],
    factor_receipts: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    preregistered: bool,
    evaluated_at: str,
) -> list[dict[str, Any]]:
    from .factor_evaluator import evaluate_factor
    from .hypothesis_evaluator import evaluate_hypothesis

    factor_by_id = {str(row["factor_id"]): row for row in factor_receipts}
    factor_specs = {str(row["factor_id"]): row for row in policy["factor_specs"]}
    results: list[dict[str, Any]] = []
    window_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for hypothesis, spec in zip(hypotheses, specs):
        locators = [*spec["support_rules"], *spec["contrary_rules"]]
        for binding in spec["falsification_bindings"]:
            locators.append(binding)
        metric_lookup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for locator in locators:
            factor_id = str(locator["factor_id"])
            if factor_id not in factor_by_id:
                raise IntelligenceContractError("hypothesis metric factor is not evaluated")
            key = (
                factor_id,
                str(locator["metric_id"]),
                str(locator["window_start"]),
                str(locator["window_end"]),
                str(locator["label_field"]),
            )
            if key in metric_lookup:
                continue
            cache_key = (factor_id, str(locator["window_start"]), str(locator["window_end"]))
            if cache_key not in window_cache:
                if len(window_cache) >= MAX_DISTINCT_HYPOTHESIS_WINDOWS:
                    raise IntelligenceContractError(
                        "distinct hypothesis window evaluation limit exceeded"
                    )
                window_rows = [
                    row
                    for row in origins
                    if str(locator["window_start"])
                    <= str(row["origin_session"])
                    <= str(locator["window_end"])
                ]
                if window_rows:
                    factor_result = evaluate_factor(
                        factor_id=factor_id,
                        origins=[row["factor_origins"][factor_id] for row in window_rows],
                        orientation=factor_specs[factor_id]["direction"],
                        horizon_sessions=policy["horizon_sessions"],
                        min_symbols=policy["min_symbols"],
                        min_available_origins=policy["min_available_origins"],
                        min_joint_coverage=policy["min_joint_coverage"],
                        min_industry_mapping_coverage=policy["min_industry_mapping_coverage"],
                    )
                    window_cache[cache_key] = _metric_map(factor_result)
                else:
                    window_cache[cache_key] = {}
            metric = window_cache[cache_key].get(str(locator["metric_id"]))
            if metric is None:
                value = None
                status = "UNAVAILABLE"
                input_origin_ids: list[str] = []
            else:
                value = metric["value"]
                status = str(metric["status"])
                input_origin_ids = list(metric["input_origin_ids"])
            metric_lookup[key] = {
                "input_origin_ids": input_origin_ids,
                "status": status,
                "value": value,
            }
        relevant_origins = [
            row
            for row in origins
            if hypothesis["expected_window"]["start"][:10]
            <= row["origin_session"]
            <= hypothesis["expected_window"]["end"][:10]
        ]
        primary_factor = factor_by_id[str(spec["support_rules"][0]["factor_id"])]
        relevant_origin_ids = {str(row["origin_id"]) for row in relevant_origins}
        coverage_values = [
            Decimal(str(row["metrics"]["joint_coverage"]))
            for row in primary_factor["origin_metrics"]
            if row["origin_id"] in relevant_origin_ids
            and row["metrics"].get("joint_coverage") is not None
        ]
        window_joint_coverage = (
            Decimal("0")
            if not coverage_values
            else sum(coverage_values, Decimal("0")) / Decimal(len(coverage_values))
        )
        outcome = evaluate_hypothesis(
            hypothesis=hypothesis,
            spec=spec,
            metric_lookup=metric_lookup,
            preregistered=preregistered,
            mature_origin_count=len(relevant_origins),
            joint_coverage=decimal_text(window_joint_coverage),
        )
        detailed_summary = outcome["evidence_summary"]
        outcome["evidence_summary"] = {
            "contrary_rule_count": len(detailed_summary["contrary_results"]),
            "contrary_triggered_count": sum(
                row["outcome"] == "PASS" for row in detailed_summary["contrary_results"]
            ),
            "falsification_condition_count": len(detailed_summary["falsification_results"]),
            "mature_origin_count": detailed_summary["mature_origin_count"],
            "support_rule_count": len(detailed_summary["support_results"]),
            "support_rule_pass_count": sum(
                row["outcome"] == "PASS" for row in detailed_summary["support_results"]
            ),
        }
        results.append(build_hypothesis_receipt(evaluated_at=evaluated_at, **outcome))
    return results


def _compare_decimal(value: Decimal, operator: str, threshold_value: Decimal) -> bool:
    if operator == "EQ":
        return value == threshold_value
    if operator == "NEQ":
        return value != threshold_value
    if operator == "GT":
        return value > threshold_value
    if operator == "GTE":
        return value >= threshold_value
    if operator == "LT":
        return value < threshold_value
    if operator == "LTE":
        return value <= threshold_value
    raise IntelligenceContractError("calibration operator is invalid")


def _calibration_receipt(
    *,
    policy: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    factor_receipts: Sequence[Mapping[str, Any]],
    preregistered: bool,
    evaluated_at: str,
) -> dict[str, Any]:
    factor_by_id = {str(row["factor_id"]): row for row in factor_receipts}
    specs = {str(row["evidence_id"]): row for row in policy["calibration_specs"]}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        key = (str(item["source_type"]), str(item["direction"]), str(item["strength"]))
        group = groups.setdefault(
            key,
            {
                "available_count": 0,
                "blocker_codes": [],
                "direction": key[1],
                "evidence_refs": [],
                "mature_count": 0,
                "required_mature_count": 0,
                "source_type": key[0],
                "status": "UNAVAILABLE",
                "strength": key[2],
                "success_count": 0,
                "success_rate": None,
            },
        )
        group["evidence_refs"].append(content_ref(item, identity_field="evidence_id"))
        spec = specs.get(str(item["evidence_id"]))
        if spec is None:
            group["blocker_codes"].append("CALIBRATION_MAPPING_MISSING")
            continue
        group["required_mature_count"] = max(
            int(group["required_mature_count"]), int(spec["min_mature_count"])
        )
        receipt = factor_by_id.get(str(spec["factor_id"]))
        metric = None if receipt is None else _metric_map(receipt).get(str(spec["metric_id"]))
        if metric is None or metric["status"] != "AVAILABLE":
            group["blocker_codes"].append("CALIBRATION_METRIC_UNAVAILABLE")
            continue
        group["available_count"] += 1
        group["mature_count"] += 1
        if _compare_decimal(
            Decimal(str(metric["value"])),
            str(spec["success_operator"]),
            Decimal(str(spec["success_threshold"])),
        ):
            group["success_count"] += 1
    for group in groups.values():
        if group["mature_count"] > 0 and group["mature_count"] >= group["required_mature_count"]:
            group["status"] = "AVAILABLE"
            group["success_rate"] = decimal_text(
                Decimal(group["success_count"]) / Decimal(group["mature_count"])
            )
    limitations = ["BAYESIAN_CALIBRATION_DIAGNOSTIC_ONLY"]
    if not preregistered:
        limitations.append("POST_HOC_DIAGNOSTIC")
    for group in groups.values():
        group["blocker_codes"] = sorted(set(group["blocker_codes"]))
        group["evidence_refs"].sort(key=lambda row: row["artifact_id"])
    return build_calibration_receipt(
        group_rows=list(groups.values()),
        limitations=limitations,
        evaluated_at=evaluated_at,
    )


def _regime_receipt(
    *,
    policy: Mapping[str, Any],
    origins: Sequence[Mapping[str, Any]],
    factor_receipts: Sequence[Mapping[str, Any]],
    variant_receipts: Sequence[Mapping[str, Any]],
    reader: ExactArtifactReader,
    evaluated_at: str,
) -> dict[str, Any]:
    from .regime_evaluator import evaluate_regimes

    selected_states: dict[str, dict[str, str] | None] = {}
    for origin in origins:
        authorized = [dict(value) for value in origin["bundle"]["authorized_evidence_refs"]]
        binding = origin["regime_binding"]
        if binding is None:
            selected_states[origin["origin_id"]] = None
            continue
        if type(binding) is not dict or set(binding) != {
            "evidence_refs",
            "industry_entity_scope",
            "input_ref",
            "receipt_ref",
            "theme_entity_scope",
        }:
            raise IntelligenceContractError("regime binding shape is invalid")
        if (
            binding["industry_entity_scope"] != "GLOBAL_BREADTH"
            or binding["theme_entity_scope"] != "GLOBAL_BREADTH"
        ):
            raise IntelligenceContractError("regime entity scope mismatch")
        if (
            type(binding["evidence_refs"]) is not list
            or not 1 <= len(binding["evidence_refs"]) <= MAX_EVIDENCE
        ):
            raise IntelligenceContractError("regime evidence cardinality is invalid")
        regime_input = _load_research_input(reader, binding["input_ref"])
        regime_source_receipt = _load_research_input(reader, binding["receipt_ref"])
        evidence = [_load_research_input(reader, value) for value in binding["evidence_refs"]]
        receipt_time = timestamp(
            regime_source_receipt["timestamp"], label="regime_receipt.timestamp"
        )
        if (
            not regime_input["available_at"]
            <= receipt_time
            <= origin["bundle"]["run_ref"]["cutoff"]
        ):
            raise IntelligenceContractError("regime receipt temporal direction is invalid")
        for item in evidence:
            validated = validate_evidence(item, as_of=receipt_time)
            if validated["source_ref"] not in authorized:
                raise IntelligenceContractError("regime evidence source is not authorized")
        validated_receipt = validate_regime_receipt(
            regime_source_receipt,
            regime_input=regime_input,
            evidence=evidence,
            as_of=receipt_time,
        )
        selected_states[origin["origin_id"]] = {
            "industry": validated_receipt["industry_state"],
            "market": validated_receipt["market_state"],
            "theme": validated_receipt["theme_state"],
        }

    subject_specs: list[dict[str, str]] = []
    subject_receipts: list[tuple[str, str, Mapping[str, Any]]] = []
    for receipt in factor_receipts:
        subject_specs.append(
            {
                "scope": "GLOBAL_BREADTH",
                "subject_id": receipt["factor_id"],
                "subject_type": "factor",
            }
        )
        subject_receipts.append(("factor", str(receipt["factor_id"]), receipt))
    variant_types = {
        VARIANT_IDS[0]: "industry",
        VARIANT_IDS[1]: "industry",
        VARIANT_IDS[2]: "theme",
    }
    for receipt in variant_receipts:
        helper_type = variant_types[str(receipt["variant_id"])]
        subject_specs.append(
            {
                "scope": "GLOBAL_BREADTH",
                "subject_id": receipt["variant_id"],
                "subject_type": helper_type,
            }
        )
        subject_receipts.append((helper_type, str(receipt["variant_id"]), receipt))
    subject_specs.sort(key=lambda row: (row["subject_type"], row["subject_id"]))
    origin_rows: list[dict[str, Any]] = []
    for origin in origins:
        subjects = []
        for helper_type, subject_id, receipt in subject_receipts:
            per_origin = next(
                (
                    row
                    for row in receipt["origin_metrics"]
                    if row["origin_id"] == origin["origin_id"]
                ),
                None,
            )
            if per_origin is None:
                continue
            metrics = {
                metric_id: per_origin["metrics"].get(metric_id)
                for metric_id in (
                    "cost_adjusted_return",
                    "joint_coverage",
                    "long_short_spread",
                    "neutralized_alpha",
                    "q5_long_only_cost_adjusted_return",
                    "rank_ic",
                )
            }
            subjects.append(
                {
                    "metrics": metrics,
                    "q5_weights": per_origin["q5_weights"],
                    "scope": "GLOBAL_BREADTH",
                    "subject_id": subject_id,
                    "subject_type": helper_type,
                }
            )
        exemplar = next(iter(origin["factor_origins"].values()))
        origin_rows.append(
            {
                "label_session": exemplar["label_session"],
                "next_open_session": exemplar["next_open_session"],
                "origin_id": origin["origin_id"],
                "origin_session": origin["origin_session"],
                "states": selected_states[origin["origin_id"]],
                "subjects": subjects,
            }
        )
    result = evaluate_regimes(
        origin_rows=origin_rows,
        subject_ids=subject_specs,
        horizon_sessions=policy["horizon_sessions"],
        min_stratum_origins=policy["regime_policy"]["min_stratum_origins"],
    )
    for layer in result["layer_rows"]:
        for state in layer["state_rows"]:
            for metric in state["factor_metric_rows"]:
                metric["subject_type"] = (
                    "FACTOR" if metric["subject_type"] == "factor" else "VARIANT"
                )
    unconditional_refs = [
        *[receipt_ref(value) for value in factor_receipts],
        *[receipt_ref(value) for value in variant_receipts],
    ]
    return build_regime_receipt(
        layer_rows=result["layer_rows"],
        unconditional_factor_refs=unconditional_refs,
        limitations=result["limitations"],
        evaluated_at=evaluated_at,
    )


def _memory_proposal(
    *,
    inventory: Mapping[str, Any],
    hypotheses: Sequence[Mapping[str, Any]],
    hypothesis_receipts: Sequence[Mapping[str, Any]],
    evaluated_at: str,
) -> dict[str, Any]:
    source = validate_memory_inventory(inventory)
    chain = tuple(source["entries"])
    original_count = len(chain)
    current_tip = str(source["tip"])
    by_hypothesis = {str(row["hypothesis_ref"]["artifact_id"]): row for row in hypothesis_receipts}
    for hypothesis in sorted(hypotheses, key=lambda row: str(row["hypothesis_id"])):
        evaluation = by_hypothesis[str(hypothesis["hypothesis_id"])]
        refs = [
            content_ref(hypothesis, identity_field="hypothesis_id"),
            receipt_ref(evaluation),
        ]

        def add(event_type: str, status: str, summary: str) -> None:
            nonlocal chain, current_tip
            chain = append_memory(
                chain,
                event_type=event_type,
                status=status,
                subject_id=str(hypothesis["hypothesis_id"]),
                summary=summary,
                artifact_refs=refs,
                timestamp_value=evaluated_at,
                expected_tip=current_tip,
            )
            current_tip = memory_tip(chain)

        receipt_id = str(evaluation["receipt_id"])
        add("EVALUATED", "UNRESOLVED", f"Forward evaluation completed: {receipt_id}.")
        if evaluation["hypothesis_status"] == "SUPPORTED":
            add(
                "HYPOTHESIS_SUPPORTED",
                "SUPPORTED",
                f"Forward hypothesis supported: {receipt_id}.",
            )
        elif evaluation["hypothesis_status"] == "FAILED":
            add(
                "HYPOTHESIS_FALSIFIED",
                "FALSIFIED",
                f"Forward hypothesis falsified: {receipt_id}.",
            )
            add(
                "FAILED_CASE",
                "FAILED",
                f"Forward failed case retained: {receipt_id}.",
            )
    suffix = list(chain[original_count:])
    return build_memory_proposal(
        expected_before_tip=source["tip"],
        observed_after_tip=current_tip,
        proposed_entries=suffix,
        source_inventory_ref=content_ref(source, identity_field="inventory_id"),
        evaluated_at=evaluated_at,
    )


def run_forward_research_evaluation(
    workspace_root: str,
    *,
    request_path: str,
    request_sha256: str,
) -> dict[str, Any]:
    """Replay an exact request and return one self-contained research envelope."""

    try:
        implementation_manifest = verify_package()
    except (CanonicalContractError, IntelligenceContractError) as exc:
        raise ImplementationIntegrityError("I0 package verification failed") from exc
    try:
        reader = ExactArtifactReader(workspace_root)
        request, _ = _load_request(
            reader,
            request_path=request_path,
            request_sha256=request_sha256,
        )
        policy = request["policy"]
        origins = _replay_origins(
            workspace_root=workspace_root,
            request=request,
            reader=reader,
        )
        evaluated_at = str(request["evaluated_at"])
        preregistered = bool(origins[0]["preregistered"])
        universe_inventory = build_universe_inventory(
            rows=[
                {
                    "origin_id": row["origin_id"],
                    "universe_factor_id": row["universe_factor_id"],
                    "universe_observation_ref": row["universe_observation_ref"],
                }
                for row in origins
            ],
            evaluated_at=evaluated_at,
        )
        universe_ref = content_ref(universe_inventory, identity_field="inventory_id")
        factor_receipts, variant_receipts = _evaluate_subjects(
            policy=policy,
            origins=origins,
            universe_ref=universe_ref,
            evaluated_at=evaluated_at,
        )
        variant_receipt = _variant_comparison(
            policy=policy,
            variant_receipts=variant_receipts,
            preregistered=preregistered,
            evaluated_at=evaluated_at,
        )
        hypotheses, hypothesis_specs, evidence = _hypothesis_inputs(
            policy=policy,
            reader=reader,
            origins=origins,
        )
        hypothesis_receipts = _hypothesis_receipts(
            policy=policy,
            origins=origins,
            factor_receipts=factor_receipts,
            hypotheses=hypotheses,
            specs=hypothesis_specs,
            preregistered=preregistered,
            evaluated_at=evaluated_at,
        )
        calibration_receipt = _calibration_receipt(
            policy=policy,
            evidence=evidence,
            factor_receipts=factor_receipts,
            preregistered=preregistered,
            evaluated_at=evaluated_at,
        )
        regime_receipt = _regime_receipt(
            policy=policy,
            origins=origins,
            factor_receipts=factor_receipts,
            variant_receipts=variant_receipts,
            reader=reader,
            evaluated_at=evaluated_at,
        )
        memory_inventory = validate_memory_inventory(
            _load_research_input(reader, request["memory_inventory_ref"])
        )
        if memory_inventory["timestamp"] > policy["created_at"]:
            raise IntelligenceContractError("memory inventory postdates sealed evaluation policy")
        memory_proposal = _memory_proposal(
            inventory=memory_inventory,
            hypotheses=hypotheses,
            hypothesis_receipts=hypothesis_receipts,
            evaluated_at=evaluated_at,
        )
        request_reference = _request_ref(request, relative_path=request_path)
        factor_refs = [receipt_ref(value) for value in factor_receipts]
        variant_subject_refs = [receipt_ref(value) for value in variant_receipts]
        hypothesis_evaluation_refs = [receipt_ref(value) for value in hypothesis_receipts]
        hypothesis_refs = [
            content_ref(value, identity_field="hypothesis_id") for value in hypotheses
        ]
        evaluation_artifact_refs = sorted(
            [
                universe_ref,
                *factor_refs,
                *variant_subject_refs,
                receipt_ref(variant_receipt),
                *hypothesis_evaluation_refs,
                receipt_ref(calibration_receipt),
                receipt_ref(regime_receipt),
                receipt_ref(memory_proposal),
            ],
            key=lambda row: (
                row["artifact_version"],
                row["artifact_id"],
                row["byte_sha256"],
            ),
        )
        summary_metrics = []
        for subject_type, subject_id, receipt in [
            *[("FACTOR", value["factor_id"], value) for value in factor_receipts],
            *[("VARIANT", value["variant_id"], value) for value in variant_receipts],
        ]:
            for metric in receipt["metrics"]:
                summary_metrics.append(
                    {
                        "metric_id": metric["metric_id"],
                        "status": metric["status"],
                        "subject_id": subject_id,
                        "subject_type": subject_type,
                        "value": metric["value"],
                    }
                )
        summary_metrics.sort(
            key=lambda row: (row["subject_type"], row["subject_id"], row["metric_id"])
        )
        all_observation_refs = _dedupe_exact_refs(
            [ref for row in origins for ref in row["bundle"]["observation_refs"]]
        )
        all_label_refs = sorted_exact_refs(
            [row["label_ref"] for row in origins], label="main.label_refs"
        )
        all_evaluation_refs = sorted_exact_refs(
            [ref for row in origins for ref in row["evaluation_refs"]],
            label="main.evaluation_artifact_refs",
        )
        main_limitations = [
            "RESEARCH_ONLY_NO_PRODUCTION_AUTHORITY",
            "NO_FACTOR_WEIGHT_MUTATION",
            "NO_POSTERIOR_MUTATION",
            "CALLER_OWNS_MEMORY_PERSISTENCE",
        ]
        if not preregistered:
            main_limitations.append("POSTHOC_POLICY_CONCLUSIONS_DOWNGRADED")
        main_receipt = build_main_receipt(
            evaluated_at=evaluated_at,
            calibration_ref=receipt_ref(calibration_receipt),
            evaluation_artifact_refs=evaluation_artifact_refs,
            evaluation_window=_evaluation_window(origins, int(policy["horizon_sessions"])),
            factor_refs=factor_refs,
            hypothesis_evaluation_refs=hypothesis_evaluation_refs,
            hypothesis_refs=hypothesis_refs,
            implementation_sha=str(implementation_manifest["semantic_sha256"]),
            label_refs=all_label_refs,
            limitations=sorted(main_limitations),
            memory_proposal_ref=receipt_ref(memory_proposal),
            metrics=summary_metrics,
            observation_refs=all_observation_refs,
            policy_ref=content_ref(policy, identity_field="policy_id"),
            regime_ref=receipt_ref(regime_receipt),
            request_ref=request_reference,
            source_evaluation_refs=all_evaluation_refs,
            universe_ref=universe_ref,
            variant_ref=receipt_ref(variant_receipt),
        )
        envelope = build_envelope(
            evaluated_at=evaluated_at,
            calibration_evidence=calibration_receipt,
            factor_evaluations=factor_receipts,
            hypothesis_evaluations=hypothesis_receipts,
            main_receipt=main_receipt,
            memory_proposal=memory_proposal,
            regime_evaluation=regime_receipt,
            request_ref=request_reference,
            universe_inventory=universe_inventory,
            variant_evaluation=variant_receipt,
            variant_factor_evaluations=variant_receipts,
        )
        for artifact in [
            universe_inventory,
            *factor_receipts,
            *variant_receipts,
            variant_receipt,
            *hypothesis_receipts,
            calibration_receipt,
            regime_receipt,
            memory_proposal,
            main_receipt,
        ]:
            if len(canonical_resource_bytes(artifact)) > 8 * 1024 * 1024:
                raise ForwardEvaluationError("limit_exceeded", "subreceipt exceeds byte limit")
        if len(canonical_resource_bytes(envelope)) + 1 > MAX_ENVELOPE_BYTES:
            raise ForwardEvaluationError("limit_exceeded", "evaluation envelope exceeds byte limit")
        return envelope
    except ForwardEvaluationError:
        raise
    except (CanonicalContractError, FactorObservationError, RegimeEvaluationError) as exc:
        raise ForwardEvaluationError("artifact_invalid", str(exc)) from exc
    except IntelligenceContractError as exc:
        text = str(exc).lower()
        if "future" in text or "temporal" in text or "postdate" in text:
            code = "temporal_invalid"
        elif "closure" in text:
            code = "closure_invalid"
        elif "limit" in text or "cardinality" in text or "exceeds" in text:
            code = "limit_exceeded"
        elif "memory" in text or "tip" in text or "hash chain" in text:
            code = "memory_conflict"
        else:
            code = "artifact_invalid"
        raise ForwardEvaluationError(code, str(exc)) from exc


__all__ = [
    "ENVELOPE_VERSION",
    "ForwardEvaluationError",
    "ImplementationIntegrityError",
    "MAX_REQUEST_BYTES",
    "POLICY_VERSION",
    "REQUEST_PREFIX",
    "REQUEST_VERSION",
    "run_forward_research_evaluation",
]
