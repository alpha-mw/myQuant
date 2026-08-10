"""Deterministic frozen-v3 Fusion projection over one same-run peer set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from ...v17_v4_runtime.forward_scoring_v3 import (
    FUSION_SCORING_V3_VERSION,
    ForwardScoringV3Error,
    fuse_forward_scores_v3,
)
from ..fundamental import validate_fundamental_profile

from .._core import (
    canonical_bytes,
    common_fields,
    content_ref,
    decimal_text,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_seal,
)
from .graph import validate_evidence_graph_v2
from .models import DecisionV2ContractError, decision_contract

FUSION_PROJECTION_V2_VERSION: Final = "myquant.v17.research-intelligence-v2.fusion-projection.v1"
FUSION_IMPLEMENTATION_SHA256: Final = (
    "35fcd9ac98bb1ef51b244c95f20db4489dc6ffdf5adcd51b1bde69ab5369f417"
)
_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_PROJECTION_FIELDS: Final = _COMMON_FIELDS | {
    "as_of",
    "base_weights",
    "fusion_implementation_sha256",
    "graph_refs",
    "profile_refs",
    "projection_id",
    "projected_records",
    "raw_float_audit",
    "run_id",
    "scorer_version",
    "semantic_sha256",
    "version",
}


def _fail(message: str) -> None:
    raise DecisionV2ContractError(message)


def _exact_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact mapping")
    return dict(value)


def _documents_with_closures(
    documents: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    for value in (documents, closures):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            _fail(f"{label} documents and closures must be sequences")
    if len(documents) != len(closures) or not documents:
        _fail(f"{label} document and closure counts are invalid")
    return [
        (document, _exact_mapping(closure, label=f"{label} closure[{index}]"))
        for index, (document, closure) in enumerate(zip(documents, closures))
    ]


def _validated_graphs(
    documents: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    as_of: str,
) -> list[dict[str, Any]]:
    pairs = _documents_with_closures(documents, closures, label="evidence graph")
    rows = [validate_evidence_graph_v2(document, **closure) for document, closure in pairs]
    rows.sort(key=lambda row: row["company_code"].encode("ascii"))
    companies = [row["company_code"] for row in rows]
    if len(rows) > 500 or len(companies) != len(set(companies)):
        _fail("FusionProjectionV2 requires 1-500 unique subject graphs")
    if any(row["run_id"] != run_id or row["timestamp"] != as_of for row in rows):
        _fail("FusionProjectionV2 graph run/time mismatch")
    if any(row["fusion_ready"] is not True for row in rows):
        _fail("FusionProjectionV2 cannot fuse an incomplete graph")
    pool_refs = {canonical_bytes(row["quant_pool_ref"]) for row in rows}
    manifest_refs = {canonical_bytes(row["v2_manifest_ref"]) for row in rows}
    if len(pool_refs) != 1 or len(manifest_refs) != 1:
        _fail("FusionProjectionV2 graphs do not share one B0 pool/v2 manifest")
    return rows


def _validated_profiles(
    documents: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
    *,
    companies: Sequence[str],
    as_of: str,
) -> list[dict[str, Any]]:
    pairs = _documents_with_closures(documents, closures, label="fundamental profile")
    rows = [validate_fundamental_profile(document, **closure) for document, closure in pairs]
    rows.sort(key=lambda row: row["company_code"].encode("ascii"))
    if [row["company_code"] for row in rows] != list(companies):
        _fail("I4 profiles do not exactly cover the graph subject set")
    expected_peers = list(companies)
    if any(row["timestamp"] != as_of or row["peer_symbols"] != expected_peers for row in rows):
        _fail("I4 profiles must share the graph time and exact peer set")
    policy_refs = {canonical_bytes(row["policy_ref"]) for row in rows}
    scorer_bindings = {(row["scorer_version"], row["scorer_implementation_sha256"]) for row in rows}
    if len(policy_refs) != 1 or len(scorer_bindings) != 1:
        _fail("I4 profiles do not share one policy/scorer closure")
    return rows


def _bind_graph_profiles(
    graphs: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> None:
    profiles_by_company = {row["company_code"]: row for row in profiles}
    for graph in graphs:
        profile = profiles_by_company[graph["company_code"]]
        if graph["fundamental_profile_ref"] != content_ref(
            profile,
            identity_field="profile_id",
        ):
            _fail("EvidenceGraphV2 does not bind its same-subject I4 profile")
        if profile["score_present"] is not True or profile["raw_score"] is None:
            _fail("FusionProjectionV2 requires an available I4 subject score")


def _raw_float_audit(value: Any) -> Any:
    if type(value) is float:
        return {"binary_float_repr": repr(value)}
    if type(value) is list:
        return [_raw_float_audit(item) for item in value]
    if type(value) is dict:
        return {str(key): _raw_float_audit(item) for key, item in value.items()}
    return value


def _decimal_projection(value: Any) -> Any:
    if type(value) is float:
        return decimal_text(Decimal(str(value)))
    if type(value) is list:
        return [_decimal_projection(item) for item in value]
    if type(value) is dict:
        return {str(key): _decimal_projection(item) for key, item in value.items()}
    return value


def _run_frozen_fusion_once(
    *,
    graphs: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    companies = [row["company_code"] for row in graphs]
    profiles_by_company = {row["company_code"]: row for row in profiles}
    try:
        result = fuse_forward_scores_v3(
            symbols=companies,
            quant_scores={row["company_code"]: row["quant_score"] for row in graphs},
            fundamental_scores={
                company: profiles_by_company[company]["raw_score"] for company in companies
            },
            fundamental_coverages={
                company: profiles_by_company[company]["coverage"] for company in companies
            },
        )
    except (ForwardScoringV3Error, TypeError, ValueError) as exc:
        raise DecisionV2ContractError(f"frozen Fusion v3 replay failed: {exc}") from exc
    if result.get("version") != FUSION_SCORING_V3_VERSION:
        _fail("frozen Fusion scorer version mismatch")
    return result


def _validate_fusion_topology(result: Mapping[str, Any], *, companies: Sequence[str]) -> None:
    records = result.get("records")
    if type(records) is not list or len(records) != len(companies):
        _fail("frozen Fusion result cardinality mismatch")
    if {row.get("symbol") for row in records} != set(companies):
        _fail("frozen Fusion result subject topology mismatch")
    if sorted(row.get("rank") for row in records) != list(range(1, len(records) + 1)):
        _fail("frozen Fusion rank topology mismatch")
    if result.get("base_weights") != {"fundamental": 0.5, "quant": 0.5}:
        _fail("frozen Fusion must preserve exact 50/50 branch weights")


@decision_contract
def build_fusion_projection_v2(
    *,
    evidence_graphs: Sequence[Mapping[str, Any]],
    graph_validation_closures: Sequence[Mapping[str, Any]],
    fundamental_profiles: Sequence[Mapping[str, Any]],
    fundamental_profile_validation_closures: Sequence[Mapping[str, Any]],
    fusion_implementation_sha256: str,
    run_id: str,
    as_of: str,
) -> dict[str, Any]:
    """Call frozen Fusion v3 once and seal its Decimal authority projection."""

    issued_at = timestamp(as_of, label="as_of")
    run = identifier(run_id, label="run_id")
    graphs = _validated_graphs(
        evidence_graphs,
        graph_validation_closures,
        run_id=run,
        as_of=issued_at,
    )
    companies = [row["company_code"] for row in graphs]
    profiles = _validated_profiles(
        fundamental_profiles,
        fundamental_profile_validation_closures,
        companies=companies,
        as_of=issued_at,
    )
    _bind_graph_profiles(graphs, profiles)
    implementation_sha = sha256(
        fusion_implementation_sha256,
        label="fusion_implementation_sha256",
    )
    if implementation_sha != FUSION_IMPLEMENTATION_SHA256:
        _fail("frozen Fusion implementation SHA mismatch")
    raw_result = _run_frozen_fusion_once(graphs=graphs, profiles=profiles)
    _validate_fusion_topology(raw_result, companies=companies)
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "base_weights": _decimal_projection(raw_result["base_weights"]),
            "fusion_implementation_sha256": implementation_sha,
            "graph_refs": [content_ref(row, identity_field="graph_id") for row in graphs],
            "profile_refs": [content_ref(row, identity_field="profile_id") for row in profiles],
            "projected_records": _decimal_projection(raw_result["records"]),
            "raw_float_audit": _raw_float_audit(raw_result),
            "run_id": run,
            "scorer_version": FUSION_SCORING_V3_VERSION,
            "version": FUSION_PROJECTION_V2_VERSION,
        },
        identity_field="projection_id",
    )


@decision_contract
def validate_fusion_projection_v2(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="projection_id")
    require_exact_keys(row, _PROJECTION_FIELDS, label="FusionProjectionV2")
    try:
        expected = build_fusion_projection_v2(**closure)
    except TypeError as exc:
        raise DecisionV2ContractError("FusionProjectionV2 closure shape is invalid") from exc
    if row != expected or row["version"] != FUSION_PROJECTION_V2_VERSION:
        _fail("FusionProjectionV2 replay mismatch")
    if canonical_bytes(row) != canonical_bytes(expected):
        _fail("FusionProjectionV2 byte replay mismatch")
    return row


__all__ = [
    "FUSION_IMPLEMENTATION_SHA256",
    "FUSION_PROJECTION_V2_VERSION",
    "build_fusion_projection_v2",
    "validate_fusion_projection_v2",
]
