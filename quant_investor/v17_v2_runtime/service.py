"""Application service for the isolated v17 protocol-v2 research runtime.

The service intentionally has no broker, network, LLM, or production-routing
surface.  It verifies the frozen package and executes only the pure in-memory
pipeline.  Durable source and shadow-ledger writes remain explicit lower-level
operations guarded by :mod:`quant_investor.v17_v2_runtime.gate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from quant_investor.v17_v2_contract.canonical import canonical_resource_bytes
from quant_investor.v17_v2_contract.resources import (
    expected_ledger_implementation_bindings,
    load_package_manifest,
    load_packaged_json,
    verify_packaged_assets,
)
from quant_investor.v17_v2_contract.validators import (
    admit_runtime_source_hash_dag,
    validate_source_role_matrix,
)

from .gate import RuntimeGate
from .pipeline import PipelineInput, PipelineResult, run_deterministic_pipeline
from .sources import SourceFile, read_pinned_source_bytes
from .storage import SecureStore

PROTOCOL_VERSION = "myquant.v17.v2"
RUNTIME_AUTHORITY = False

_PIPELINE_INPUT_KEYS = frozenset(
    {
        "cutoff",
        "strategy_id",
        "fundamental_rows",
        "fundamental_history",
        "forward_observations",
        "price_history",
        "timing_observations",
        "deep_response",
        "sealed_evidence_ids",
        "holdings",
        "cash",
        "nav",
        "risk_policy",
        "cost_policy",
        "tradability",
        "risk_model",
        "clusters",
        "macro",
        "markov",
        "portfolio_candidates",
    }
)
_PIPELINE_ENVELOPE_KEYS = frozenset({"version", "authority"})
_SOURCE_BUNDLE_KEYS = frozenset(
    {
        "source_root",
        "source_objects",
        "dataset_manifests",
        "observation_dispositions",
        "source_manifest",
        "source_manifest_path",
        "generation_catalogs",
        "summaries",
        "source_binding_set",
        "source_binding_set_path",
        "source_locator",
        "source_locator_path",
    }
)


class RuntimeServiceError(ValueError):
    """A fail-closed service boundary rejection."""


@dataclass(frozen=True)
class RuntimeReadiness:
    protocol_version: str
    distribution_version: str
    matrix_status: str
    runtime_usable: bool
    pending_registry: tuple[str, ...]
    authority: bool
    packaged_asset_count: int
    implementation_binding_count: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "distribution_version": self.distribution_version,
            "matrix_status": self.matrix_status,
            "runtime_usable": self.runtime_usable,
            "pending_registry": list(self.pending_registry),
            "authority": False,
            "packaged_asset_count": self.packaged_asset_count,
            "implementation_binding_count": self.implementation_binding_count,
        }


@dataclass(frozen=True)
class SourceAdmissionResult:
    disposition: str
    locator_id: str
    locator_byte_sha256: str
    input_binding_count: int
    unavailable_required_roles: tuple[str, ...]
    committed: bool
    committed_path_count: int
    authority: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "locator_id": self.locator_id,
            "locator_byte_sha256": self.locator_byte_sha256,
            "input_binding_count": self.input_binding_count,
            "unavailable_required_roles": list(self.unavailable_required_roles),
            "committed": self.committed,
            "committed_path_count": self.committed_path_count,
            "authority": False,
        }


def verify_runtime() -> RuntimeReadiness:
    """Verify exact packaged bytes and the Phase-1 admission contract."""

    verified_assets = verify_packaged_assets()
    package = load_package_manifest()
    matrix = validate_source_role_matrix(load_packaged_json("resources/source_role_matrix.v1.json"))
    distribution = package.get("distribution")
    if type(distribution) is not dict or type(distribution.get("version")) is not str:
        raise RuntimeServiceError("package distribution version is missing")
    pending = matrix.get("pending_registry")
    if type(pending) is not list or any(type(item) is not str for item in pending):
        raise RuntimeServiceError("source role matrix pending_registry is invalid")
    readiness = RuntimeReadiness(
        protocol_version=PROTOCOL_VERSION,
        distribution_version=distribution["version"],
        matrix_status=str(matrix.get("completeness")),
        runtime_usable=matrix.get("runtime_usable") is True,
        pending_registry=tuple(pending),
        authority=False,
        packaged_asset_count=len(verified_assets),
        implementation_binding_count=len(expected_ledger_implementation_bindings()),
    )
    if (
        readiness.matrix_status != "COMPLETE"
        or not readiness.runtime_usable
        or readiness.pending_registry
        or package.get("runtime_usable") is not True
        or package.get("authority") is not False
        or matrix.get("authority") is not False
    ):
        raise RuntimeServiceError("Phase-1 runtime admission contract is not complete")
    return readiness


def _tuple_of_mappings(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise RuntimeServiceError(f"{label} must be an array of objects")
    return tuple(value)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise RuntimeServiceError(f"{label} must be an object")
    return value


def pipeline_input_from_mapping(payload: Mapping[str, Any]) -> PipelineInput:
    """Create a typed pipeline input without fallback or implicit defaults."""

    if type(payload) is not dict:
        raise RuntimeServiceError("analysis input root must be an object")
    keys = frozenset(payload)
    allowed_keys = _PIPELINE_INPUT_KEYS | _PIPELINE_ENVELOPE_KEYS
    if not _PIPELINE_INPUT_KEYS.issubset(keys) or not keys.issubset(allowed_keys):
        missing = sorted(_PIPELINE_INPUT_KEYS.difference(payload))
        extra = sorted(set(payload).difference(allowed_keys))
        raise RuntimeServiceError(f"analysis input keys mismatch; missing={missing}, extra={extra}")
    if "version" in payload and payload["version"] != f"{PROTOCOL_VERSION}.pipeline-input.v1":
        raise RuntimeServiceError("analysis input version mismatch")
    if "authority" in payload and payload["authority"] is not False:
        raise RuntimeServiceError("analysis input authority must be false")
    price_history = _mapping(payload["price_history"], label="price_history")
    normalized_prices: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for symbol, rows in price_history.items():
        if type(symbol) is not str:
            raise RuntimeServiceError("price_history keys must be strings")
        normalized_prices[symbol] = _tuple_of_mappings(
            rows,
            label=f"price_history.{symbol}",
        )
    evidence = _mapping(payload["sealed_evidence_ids"], label="sealed_evidence_ids")
    normalized_evidence: dict[str, tuple[str, ...]] = {}
    for symbol, ids in evidence.items():
        if (
            type(symbol) is not str
            or type(ids) is not list
            or any(type(item) is not str for item in ids)
        ):
            raise RuntimeServiceError("sealed_evidence_ids must map symbols to string arrays")
        normalized_evidence[symbol] = tuple(ids)
    return PipelineInput(
        cutoff=payload["cutoff"],
        strategy_id=payload["strategy_id"],
        fundamental_rows=_tuple_of_mappings(payload["fundamental_rows"], label="fundamental_rows"),
        fundamental_history=_tuple_of_mappings(
            payload["fundamental_history"], label="fundamental_history"
        ),
        forward_observations=_tuple_of_mappings(
            payload["forward_observations"], label="forward_observations"
        ),
        price_history=normalized_prices,
        timing_observations=_tuple_of_mappings(
            payload["timing_observations"], label="timing_observations"
        ),
        deep_response=_mapping(payload["deep_response"], label="deep_response"),
        sealed_evidence_ids=normalized_evidence,
        holdings=_mapping(payload["holdings"], label="holdings"),
        cash=payload["cash"],
        nav=payload["nav"],
        risk_policy=_mapping(payload["risk_policy"], label="risk_policy"),
        cost_policy=_mapping(payload["cost_policy"], label="cost_policy"),
        tradability=_mapping(payload["tradability"], label="tradability"),
        risk_model=_mapping(payload["risk_model"], label="risk_model"),
        clusters=_mapping(payload["clusters"], label="clusters"),
        macro=_mapping(payload["macro"], label="macro"),
        markov=_mapping(payload["markov"], label="markov"),
        portfolio_candidates=_tuple_of_mappings(
            payload["portfolio_candidates"], label="portfolio_candidates"
        ),
    )


def analyze_mapping(payload: Mapping[str, Any]) -> PipelineResult:
    """Verify runtime admission and execute one deterministic shadow analysis."""

    verify_runtime()
    return run_deterministic_pipeline(pipeline_input_from_mapping(payload))


def _document_mapping(value: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    mapping = _mapping(value, label=label)
    result: dict[str, Mapping[str, Any]] = {}
    for path, document in mapping.items():
        if type(path) is not str or type(document) is not dict:
            raise RuntimeServiceError(f"{label} must map paths to objects")
        result[path] = document
    return result


def admit_source_bundle(
    payload: Mapping[str, Any],
    *,
    workspace_root: Path,
    commit: bool,
) -> SourceAdmissionResult:
    """Validate and optionally publish one exact externally-built source DAG."""

    verify_runtime()
    if type(payload) is not dict or frozenset(payload) != _SOURCE_BUNDLE_KEYS:
        raise RuntimeServiceError("source admission bundle keys mismatch")
    source_root_value = payload["source_root"]
    if type(source_root_value) is not str:
        raise RuntimeServiceError("source_root must be an absolute path string")
    source_root = Path(source_root_value)
    objects_value = payload["source_objects"]
    if type(objects_value) is not list or not objects_value:
        raise RuntimeServiceError("source_objects must be a nonempty array")
    source_objects: dict[str, bytes] = {}
    for index, item in enumerate(objects_value):
        if type(item) is not dict or set(item) != {
            "relative_path",
            "absolute_path",
            "expected_sha256",
        }:
            raise RuntimeServiceError(f"source_objects[{index}] keys mismatch")
        relative_path = item["relative_path"]
        absolute_path = item["absolute_path"]
        expected_sha256 = item["expected_sha256"]
        values = (relative_path, absolute_path, expected_sha256)
        if any(type(value) is not str for value in values):
            raise RuntimeServiceError(f"source_objects[{index}] fields must be strings")
        if relative_path in source_objects:
            raise RuntimeServiceError(f"source object path is duplicated: {relative_path}")
        source_objects[relative_path] = read_pinned_source_bytes(
            source_root=source_root,
            source=SourceFile(
                path=Path(absolute_path),
                expected_sha256=expected_sha256,
                role=relative_path,
            ),
        )

    dataset_manifests = _document_mapping(payload["dataset_manifests"], label="dataset_manifests")
    dispositions = _document_mapping(
        payload["observation_dispositions"], label="observation_dispositions"
    )
    catalogs = _document_mapping(payload["generation_catalogs"], label="generation_catalogs")
    summaries = _document_mapping(payload["summaries"], label="summaries")
    source_manifest = _mapping(payload["source_manifest"], label="source_manifest")
    source_binding_set = _mapping(payload["source_binding_set"], label="source_binding_set")
    source_locator = _mapping(payload["source_locator"], label="source_locator")
    source_manifest_path = payload["source_manifest_path"]
    source_binding_set_path = payload["source_binding_set_path"]
    source_locator_path = payload["source_locator_path"]
    document_paths = (
        source_manifest_path,
        source_binding_set_path,
        source_locator_path,
    )
    if any(type(value) is not str for value in document_paths):
        raise RuntimeServiceError("source document paths must be strings")

    stored_documents: dict[str, bytes] = {}
    for path, document in (
        *dataset_manifests.items(),
        *dispositions.items(),
        *catalogs.items(),
        *summaries.items(),
        (source_manifest_path, source_manifest),
        (source_binding_set_path, source_binding_set),
        (source_locator_path, source_locator),
    ):
        if path in stored_documents:
            raise RuntimeServiceError(f"source document path is duplicated: {path}")
        stored_documents[path] = canonical_resource_bytes(document)

    matrix = load_packaged_json("resources/source_role_matrix.v1.json")
    outcome = admit_runtime_source_hash_dag(
        source_role_matrix=matrix,
        source_objects=source_objects,
        dataset_manifests=dataset_manifests,
        observation_dispositions=dispositions,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        generation_catalogs=catalogs,
        summaries=summaries,
        source_binding_set=source_binding_set,
        source_binding_set_path=source_binding_set_path,
        source_locator=source_locator,
        source_locator_path=source_locator_path,
        stored_document_bytes=stored_documents,
    )

    committed_count = 0
    if commit:
        decision = RuntimeGate(workspace_root).classify(
            "SOURCE_MAINTAIN",
            str(outcome.locator["locator_id"]),
            version="ABSENT",
            state="MISSING",
            checkpoint="INITIALIZED",
        )
        expected_namespaces = {
            "SOURCE_OBJECTS",
            "SOURCE_MANIFESTS",
            "SOURCE_LOCATORS",
        }
        if not decision.allowed or set(decision.allowed_write_namespaces) != expected_namespaces:
            raise RuntimeServiceError(f"source publication gate rejected: {decision.detail}")
        store = SecureStore(workspace_root, max_read_bytes=2**31)
        write_items = [
            *sorted(source_objects.items()),
            *sorted(
                (path, raw) for path, raw in stored_documents.items() if path != source_locator_path
            ),
            (source_locator_path, stored_documents[source_locator_path]),
        ]
        for path, raw in write_items:
            store.write_exact_once(path, raw)
            committed_count += 1
    return SourceAdmissionResult(
        disposition=outcome.disposition.value,
        locator_id=str(outcome.locator["locator_id"]),
        locator_byte_sha256=outcome.locator_byte_sha256,
        input_binding_count=len(outcome.input_bindings),
        unavailable_required_roles=outcome.unavailable_required_roles,
        committed=commit,
        committed_path_count=committed_count,
    )


__all__ = [
    "RUNTIME_AUTHORITY",
    "RuntimeReadiness",
    "RuntimeServiceError",
    "SourceAdmissionResult",
    "admit_source_bundle",
    "analyze_mapping",
    "pipeline_input_from_mapping",
    "verify_runtime",
]
