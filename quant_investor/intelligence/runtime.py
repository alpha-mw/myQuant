"""Pure I0 runtime receipt connecting Observation, evidence, and intelligence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    assert_no_authority,
    content_ref,
    seal_content_addressed,
    sha256,
    timestamp,
    validate_content_addressed,
)
from .bayesian.engine import BAYESIAN_RECEIPT_VERSION, validate_bayesian_receipt
from .evidence.forward_adapter import (
    BUNDLE_VERSION,
    build_observation_evidence_bundle,
    validate_observation_evidence_bundle,
)
from .evidence.models import EVIDENCE_VERSION, validate_evidence_set
from .fusion.branches import BRANCH_VERSION, validate_branch
from .fusion.engine import FUSION_RECEIPT_VERSION, validate_fusion_receipt
from .hypothesis.models import HYPOTHESIS_VERSION, validate_hypothesis
from .memory.chain import memory_tip, validate_memory_chain
from .regime.engine import REGIME_RECEIPT_VERSION, validate_regime_receipt
from .regime.input import REGIME_INPUT_VERSION, validate_regime_input

INTELLIGENCE_RUNTIME_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.runtime-receipt.v1"


def _validated(
    document: Mapping[str, Any],
    *,
    identity_field: str,
    expected_version: str,
    as_of: str,
) -> dict[str, Any]:
    row = validate_content_addressed(document, identity_field=identity_field)
    document_timestamp = timestamp(row.get("timestamp"), label=f"{expected_version}.timestamp")
    cutoff = timestamp(as_of, label="as_of")
    if row.get("version") != expected_version:
        raise IntelligenceContractError(f"{expected_version} version mismatch")
    if document_timestamp > cutoff:
        raise IntelligenceContractError(f"{expected_version} is from the future")
    assert_no_authority(row)
    return row


def _validate_hypotheses_and_bayesian(
    *,
    hypotheses: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    as_of: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hypothesis_rows = [
        validate_hypothesis(value, evidence=evidence, as_of=as_of) for value in hypotheses
    ]
    hypothesis_ids = {str(row["hypothesis_id"]) for row in hypothesis_rows}
    bayesian_rows = [
        validate_bayesian_receipt(value, evidence=evidence, as_of=as_of) for value in receipts
    ]
    if any(row.get("hypothesis_id") not in hypothesis_ids for row in bayesian_rows):
        raise IntelligenceContractError("Bayesian hypothesis closure mismatch")
    return hypothesis_rows, bayesian_rows


def _exact_ref_key(value: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(item)) for key, item in value.items()))


def _assert_sources_authorized(
    *,
    bundle: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    regime_input: Mapping[str, Any],
) -> None:
    authorized = bundle.get("authorized_evidence_refs")
    if type(authorized) is not list or not authorized:
        raise IntelligenceContractError("Observation bundle has no authorized evidence refs")
    authorized_keys = {_exact_ref_key(ref) for ref in authorized}
    evidence_source_keys = {_exact_ref_key(row.get("source_ref", {})) for row in evidence}
    regime_source_keys = {_exact_ref_key(ref) for ref in regime_input.get("source_refs", [])}
    if not evidence_source_keys.issubset(authorized_keys):
        raise IntelligenceContractError("evidence source is outside Observation closure")
    if not regime_source_keys.issubset(authorized_keys):
        raise IntelligenceContractError("regime source is outside Observation closure")


def build_intelligence_runtime_receipt(
    *,
    observation_bundle: Mapping[str, Any],
    workspace_root: str,
    session_relative_path: str,
    session_byte_sha256: str,
    observation_refs: Sequence[Mapping[str, Any]],
    closure_refs: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    bayesian_receipts: Sequence[Mapping[str, Any]],
    regime_input: Mapping[str, Any],
    regime_receipt: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
    fusion_receipt: Mapping[str, Any],
    hypotheses: Sequence[Mapping[str, Any]],
    memory_entries: Sequence[Mapping[str, Any]],
    expected_memory_tip: str,
    as_of: str,
    label_refs: Sequence[Mapping[str, Any]] = (),
    evaluation_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Verify the complete research closure and return one no-authority receipt."""

    cutoff = timestamp(as_of, label="as_of")
    rebuilt_bundle = build_observation_evidence_bundle(
        workspace_root=workspace_root,
        session_relative_path=session_relative_path,
        session_byte_sha256=session_byte_sha256,
        observation_refs=observation_refs,
        closure_refs=closure_refs,
        label_refs=label_refs,
        evaluation_refs=evaluation_refs,
        as_of=cutoff,
    )
    if rebuilt_bundle != observation_bundle:
        raise IntelligenceContractError("Observation bundle does not match exact adapter replay")
    bundle = validate_observation_evidence_bundle(rebuilt_bundle, as_of=cutoff)
    evidence_rows = validate_evidence_set(evidence, as_of=cutoff)
    if not bayesian_receipts or not branches or not hypotheses or not memory_entries:
        raise IntelligenceContractError("runtime closure is incomplete")
    regime_input_row = validate_regime_input(regime_input, as_of=cutoff)
    _assert_sources_authorized(
        bundle=bundle,
        evidence=evidence_rows,
        regime_input=regime_input_row,
    )

    hypothesis_rows, bayesian_rows = _validate_hypotheses_and_bayesian(
        hypotheses=hypotheses,
        receipts=bayesian_receipts,
        evidence=evidence_rows,
        as_of=cutoff,
    )

    regime = validate_regime_receipt(
        regime_receipt,
        regime_input=regime_input_row,
        evidence=evidence_rows,
        as_of=cutoff,
    )

    branch_rows = [
        validate_branch(value, evidence=evidence_rows, as_of=cutoff) for value in branches
    ]

    fusion = validate_fusion_receipt(
        fusion_receipt,
        branches=branch_rows,
        as_of=cutoff,
    )

    memory_rows = validate_memory_chain(memory_entries, expected_tip=expected_memory_tip)
    for entry in memory_rows:
        if entry.get("timestamp") > cutoff:
            raise IntelligenceContractError("memory entry is from the future")
        assert_no_authority(entry)
    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "component_refs": {
                "bayesian": [
                    content_ref(row, identity_field="receipt_id") for row in bayesian_rows
                ],
                "branches": [content_ref(row, identity_field="branch_id") for row in branch_rows],
                "evidence": [
                    content_ref(row, identity_field="evidence_id") for row in evidence_rows
                ],
                "fusion": content_ref(fusion, identity_field="receipt_id"),
                "hypotheses": [
                    content_ref(row, identity_field="hypothesis_id") for row in hypothesis_rows
                ],
                "observation_bundle": content_ref(bundle, identity_field="bundle_id"),
                "regime": content_ref(regime, identity_field="receipt_id"),
                "regime_input": content_ref(regime_input_row, identity_field="input_id"),
            },
            "memory_entry_count": len(memory_rows),
            "memory_tip_sha256": memory_tip(memory_rows),
            "production": False,
            "research_only": True,
            "timestamp": cutoff,
            "version": INTELLIGENCE_RUNTIME_RECEIPT_VERSION,
        },
        identity_field="runtime_receipt_id",
    )


def verify_runtime_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the sealed summary shape; full closure replay occurs in the builder."""

    row = _validated(
        document,
        identity_field="runtime_receipt_id",
        expected_version=INTELLIGENCE_RUNTIME_RECEIPT_VERSION,
        as_of=str(document.get("timestamp")),
    )
    if set(row) != {
        "authority",
        "component_refs",
        "memory_entry_count",
        "memory_tip_sha256",
        "production",
        "research_only",
        "runtime_receipt_id",
        "semantic_sha256",
        "timestamp",
        "version",
    }:
        raise IntelligenceContractError("runtime receipt shape is not closed")
    components = row.get("component_refs")
    if type(components) is not dict or set(components) != {
        "bayesian",
        "branches",
        "evidence",
        "fusion",
        "hypotheses",
        "observation_bundle",
        "regime",
        "regime_input",
    }:
        raise IntelligenceContractError("runtime receipt is incomplete")
    required_lists = ("bayesian", "branches", "evidence", "hypotheses")
    if any(
        type(components.get(name)) is not list or not components[name] for name in required_lists
    ):
        raise IntelligenceContractError("runtime receipt list closure is incomplete")
    if type(row.get("memory_entry_count")) is not int or row["memory_entry_count"] < 1:
        raise IntelligenceContractError("runtime memory count is invalid")
    sha256(row.get("memory_tip_sha256"), label="memory_tip_sha256")
    content_ref_fields = {"artifact_id", "artifact_version", "byte_sha256", "semantic_sha256"}
    component_versions = {
        "bayesian": BAYESIAN_RECEIPT_VERSION,
        "branches": BRANCH_VERSION,
        "evidence": EVIDENCE_VERSION,
        "fusion": FUSION_RECEIPT_VERSION,
        "hypotheses": HYPOTHESIS_VERSION,
        "observation_bundle": BUNDLE_VERSION,
        "regime": REGIME_RECEIPT_VERSION,
        "regime_input": REGIME_INPUT_VERSION,
    }
    flattened: list[tuple[str, Mapping[str, Any]]] = []
    for name, expected_version in component_versions.items():
        value = components[name]
        refs = value if name in required_lists else [value]
        artifact_ids: list[str] = []
        byte_hashes: list[str] = []
        for ref in refs:
            flattened.append((expected_version, ref))
            if type(ref) is dict:
                artifact_ids.append(str(ref.get("artifact_id")))
                byte_hashes.append(str(ref.get("byte_sha256")))
        if len(artifact_ids) != len(set(artifact_ids)) or len(byte_hashes) != len(set(byte_hashes)):
            raise IntelligenceContractError(f"runtime {name} refs contain duplicates")
    for expected_version, ref in flattened:
        if type(ref) is not dict or set(ref) != content_ref_fields:
            raise IntelligenceContractError("runtime component ref is malformed")
        if not all(type(ref[field]) is str and ref[field] for field in content_ref_fields):
            raise IntelligenceContractError("runtime component ref is incomplete")
        sha256(ref["byte_sha256"], label="component_ref.byte_sha256")
        sha256(ref["semantic_sha256"], label="component_ref.semantic_sha256")
        if ref["artifact_version"] != expected_version:
            raise IntelligenceContractError("runtime component ref version mismatch")
    return row


__all__ = [
    "INTELLIGENCE_RUNTIME_RECEIPT_VERSION",
    "build_intelligence_runtime_receipt",
    "verify_runtime_receipt",
]
