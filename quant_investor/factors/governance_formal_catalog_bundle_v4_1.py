"""Private readback contract for v4.1 classification-only formal catalogs.

"Formal" in this module means only that the ontology and candidate catalog are
readable by the exact FactorGovernance v4 schemas.  It does not mean that a
signal is executable, screening-eligible, statistically qualified, admitted,
or production-authorized.  The module consumes only caller-supplied mappings;
the shared private-bundle I/O helper owns the optional filesystem transaction.
Protected registry/CN control hashes are immutable audit sentinels for the
build and locked precommit revalidation points only; external maintenance is
not locked and postcommit stability is explicitly outside bundle acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors import governance_formal_catalog_adapter_v4_1 as adapter
from quant_investor.factors import (
    governance_formal_catalog_materialization_v4_1 as materialization,
)
from quant_investor.factors.governance_private_bundle_io import (
    PrivateBundleContract,
)


PROTOCOL_VERSION = "v4.1"
READINESS = "EXPLORATORY_FORMAL_CATALOG_CLASSIFICATION_ONLY"
LIFECYCLE_STATE = "DISCOVERY"
FORMAL_CATALOG_READBACK_REPORT_SCHEMA_VERSION = (
    "factor-governance-formal-catalog-materialization-readback.v4.1"
)
FORMAL_CATALOG_READBACK_REPORT_FILENAME = (
    "formal_catalog_materialization_readback.v4_1.json"
)
FORMAL_CATALOG_INPUT_FILENAMES = (
    *materialization.FORMAL_CATALOG_MATERIALIZATION_FILENAMES,
    adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME,
)
FORMAL_CATALOG_BUNDLE_FILENAMES = (
    *FORMAL_CATALOG_INPUT_FILENAMES,
    FORMAL_CATALOG_READBACK_REPORT_FILENAME,
)
FORMAL_CATALOG_PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_formal_catalog",
)
PROTECTED_CONTROL_PATH_SUFFIXES = (
    "/quant_investor/factor_registry/mined_factors.json",
    "/data/parquet/cn/_latest.json",
    "/data/parquet/cn/_catalog.json",
    "/data/parquet/cn/_fundamental_latest.json",
    "/data/parquet/cn/latest_manifest.json",
)

MEASUREMENT_STATUS_FIELDS = (
    "admission_duplicate_primitive",
    "cost",
    "family_bh",
    "high_correlation_dedup",
    "maturity",
    "neutralization",
    "statistics",
    "transaction_plan",
    "verified_v4_replay",
    "walk_forward",
)
SIDE_EFFECT_FIELDS = (
    "apply_performed",
    "broker_called",
    "budget_write_performed",
    "filesystem_input_read_performed",
    "holdout_access_performed",
    "live_provider_called",
    "market_data_access_performed",
    "network_called",
    "order_created",
    "portfolio_constructed",
    "private_readback_report_created",
    "private_research_bundle_created",
    "production_pointer_mutated",
    "production_receipt_created",
    "proposal_created",
    "registry_write_performed",
    "replay_artifact_created",
    "research_head_created",
    "statistics_performed",
    "trade_created",
    "transaction_plan_created",
    "wal_write_performed",
)
BLOCKERS = (
    "admission_duplicate_primitive_not_run",
    "classification_only",
    "cost_not_run",
    "family_bh_not_run",
    "high_correlation_dedup_not_run",
    "maturity_not_authoritative",
    "neutralization_not_run",
    "runtime_equivalence_not_verified",
    "signal_computability_not_proven",
    "statistics_not_run",
    "transaction_plan_not_run",
    "verified_v4_replay_not_run",
    "walk_forward_not_run",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_ARTIFACT_BINDING_FIELDS = frozenset(
    {
        "filename",
        "byte_sha256",
        "semantic_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
)
_PROTECTED_BINDING_FIELDS = frozenset({"absolute_path", "byte_sha256"})
_SOURCE_ACCOUNTING_FIELDS = frozenset(
    {
        "source_candidate_count",
        "new_candidate_count",
        "structural_alias_count",
        "incompatible_count",
    }
)
_CATALOG_ACCOUNTING_FIELDS = frozenset(
    {"base_candidate_count", "new_candidate_count", "candidate_count"}
)
_ONTOLOGY_ACCOUNTING_FIELDS = frozenset(
    {"base_primitive_count", "new_primitive_count", "primitive_count"}
)
_READBACK_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "readiness",
        "lifecycle_state",
        "artifact_bindings",
        "source_accounting",
        "catalog_accounting",
        "ontology_accounting",
        "mapping_proof_sha256",
        "formal_catalog_sha256",
        "adapter_validation_sha256",
        "protected_bindings",
        "protected_bindings_semantic_sha256",
        "protected_controls_bound_at_build_and_precommit",
        "postcommit_protected_stability_part_of_bundle_acceptance",
        "protected_stability_scope",
        "source_authenticity_recomputed_by_materializer",
        "adapter_source_authenticity_recomputed",
        "classification_only",
        "runtime_equivalence_verified",
        "signal_computability_proven",
        "screening_eligible",
        "proposal_eligible",
        "registry_entry_created",
        "initial_weight_policy",
        "qualification",
        "formal_admission_authority",
        "production_apply_enabled",
        "new_risk_authorized",
        "measurement_status",
        "blockers",
        "side_effects",
        "report_semantic_sha256",
    }
)


class FactorGovernanceFormalCatalogBundleV4_1Error(ValueError):
    """Raised when a classification-only formal bundle cannot be proven."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_1(value) + b"\n"


def semantic_sha256_v4_1(
    value: Any,
    *,
    exclude_fields: Sequence[str] = (),
) -> str:
    normalized = copy.deepcopy(value)
    if exclude_fields:
        if not isinstance(normalized, Mapping):
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                "exclude_fields requires a top-level object"
            )
        normalized = dict(normalized)
        for field in exclude_fields:
            normalized.pop(field, None)
    return hashlib.sha256(canonical_json_bytes_v4_1(normalized)).hexdigest()


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"{label} must be an object"
        )
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"{label} field names must be strings"
        )
    if set(payload) != set(fields):
        missing = sorted(set(fields) - set(payload))
        extra = sorted(set(payload) - set(fields))
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"{label} fields mismatch: missing={missing};extra={extra}"
        )
    return payload


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _artifact_semantic_sha256(filename: str, artifact: Mapping[str, Any]) -> str:
    field_by_filename = {
        materialization.PRIMITIVE_MAPPING_POLICY_FILENAME: (
            "policy_semantic_sha256"
        ),
        materialization.PRIMITIVE_MAPPING_PROOF_FILENAME: (
            "proof_semantic_sha256"
        ),
        materialization.FORMAL_ONTOLOGY_FILENAME: "semantic_sha256",
        materialization.FORMAL_CATALOG_FILENAME: "semantic_sha256",
        materialization.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME: (
            "manifest_semantic_sha256"
        ),
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME: (
            "validation_semantic_sha256"
        ),
    }
    try:
        field = field_by_filename[filename]
        value = artifact[field]
    except KeyError as exc:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            f"unknown artifact semantic binding: {filename}"
        ) from exc
    return _sha(value, f"{filename}.{field}")


def _normalize_artifact_bindings(
    artifact_bindings: Sequence[Mapping[str, Any]],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(artifact_bindings, Sequence) or isinstance(
        artifact_bindings, (str, bytes, bytearray)
    ):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "artifact_bindings must be a sequence"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(artifact_bindings):
        row = _exact(raw, _ARTIFACT_BINDING_FIELDS - {"semantic_sha256"}, f"artifact_bindings[{index}]")
        filename = row["filename"]
        if type(filename) is not str or filename not in FORMAL_CATALOG_INPUT_FILENAMES:
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"artifact binding filename is not canonical: {filename!r}"
            )
        artifact = artifacts.get(filename)
        if not isinstance(artifact, Mapping):
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"artifact binding has no matching value: {filename}"
            )
        canonical = canonical_file_bytes_v4_1(artifact)
        normalized = {
            "filename": filename,
            "byte_sha256": _sha(
                row["byte_sha256"], f"artifact_bindings[{index}].byte_sha256"
            ),
            "semantic_sha256": _artifact_semantic_sha256(filename, artifact),
            "size_bytes": _integer(
                row["size_bytes"], f"artifact_bindings[{index}].size_bytes", minimum=1
            ),
            "mode": _integer(row["mode"], f"artifact_bindings[{index}].mode"),
            "uid": _integer(row["uid"], f"artifact_bindings[{index}].uid"),
            "nlink": _integer(
                row["nlink"], f"artifact_bindings[{index}].nlink", minimum=1
            ),
        }
        if normalized["byte_sha256"] != hashlib.sha256(canonical).hexdigest():
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"artifact binding byte SHA mismatch: {filename}"
            )
        if normalized["size_bytes"] != len(canonical):
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"artifact binding size mismatch: {filename}"
            )
        if normalized["mode"] != 0o600 or normalized["nlink"] != 1:
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"artifact binding is not owner-private: {filename}"
            )
        rows.append(normalized)
    if [row["filename"] for row in rows] != list(FORMAL_CATALOG_INPUT_FILENAMES):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "artifact bindings must follow the exact canonical input order"
        )
    return rows


def _normalize_protected_bindings(value: Any) -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        raw_rows = [
            {"absolute_path": path, "byte_sha256": sha256}
            for path, sha256 in value.items()
        ]
    elif isinstance(value, list):
        raw_rows = list(value)
    else:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "protected_bindings must be an object or list"
        )
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(raw_rows):
        row = _exact(raw, _PROTECTED_BINDING_FIELDS, f"protected_bindings[{index}]")
        path = row["absolute_path"]
        if (
            type(path) is not str
            or not path.startswith("/")
            or "\x00" in path
            or any(part in {"", ".", ".."} for part in path.split("/")[1:])
        ):
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                "protected binding paths must be normalized absolute paths"
            )
        rows.append(
            {
                "absolute_path": path,
                "byte_sha256": _sha(
                    row["byte_sha256"],
                    f"protected_bindings[{index}].byte_sha256",
                ),
            }
        )
    rows.sort(key=lambda row: row["absolute_path"])
    paths = [row["absolute_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "protected binding paths must be distinct"
        )
    matched_suffixes = []
    for path in paths:
        matches = [
            suffix
            for suffix in PROTECTED_CONTROL_PATH_SUFFIXES
            if path.endswith(suffix)
        ]
        if len(matches) != 1:
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"protected binding path is outside the exact allowlist: {path}"
            )
        matched_suffixes.extend(matches)
    if set(matched_suffixes) != set(PROTECTED_CONTROL_PATH_SUFFIXES) or len(
        matched_suffixes
    ) != len(PROTECTED_CONTROL_PATH_SUFFIXES):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "protected bindings must contain the exact five control files"
        )
    return rows


def _authority_constants() -> dict[str, Any]:
    return {
        "protected_controls_bound_at_build_and_precommit": True,
        "postcommit_protected_stability_part_of_bundle_acceptance": False,
        "protected_stability_scope": (
            "build_and_precommit_only_external_controls_are_not_locked"
        ),
        "source_authenticity_recomputed_by_materializer": True,
        "adapter_source_authenticity_recomputed": False,
        "classification_only": True,
        "runtime_equivalence_verified": False,
        "signal_computability_proven": False,
        "screening_eligible": False,
        "proposal_eligible": False,
        "registry_entry_created": False,
        "initial_weight_policy": "zero_only",
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "new_risk_authorized": False,
    }


def _expected_readback_report(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
    protected_bindings: Any,
) -> dict[str, Any]:
    if type(run_id) is not str or _SAFE_RUN_ID_RE.fullmatch(run_id) is None or ".." in run_id:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "run_id must be one safe non-empty path segment"
        )
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        FORMAL_CATALOG_INPUT_FILENAMES
    ):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "formal readback inputs must be the exact six canonical artifacts"
        )
    manifest = artifacts[
        materialization.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME
    ]
    proof = artifacts[materialization.PRIMITIVE_MAPPING_PROOF_FILENAME]
    ontology = artifacts[materialization.FORMAL_ONTOLOGY_FILENAME]
    catalog = artifacts[materialization.FORMAL_CATALOG_FILENAME]
    adapter_validation = artifacts[
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME
    ]
    cycle_ids = {manifest.get("cycle_id"), proof.get("cycle_id")}
    if len(cycle_ids) != 1 or not all(type(value) is str and value for value in cycle_ids):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "formal artifacts do not share one cycle_id"
        )
    if len(ontology.get("primitives", ())) != 18:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "formal ontology must contain exactly 18 primitives"
        )
    if len(catalog.get("candidates", ())) != 267:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "formal catalog must contain exactly 267 candidates"
        )
    if adapter_validation.get("candidate_count") != 267:
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "adapter validation must account for exactly 267 candidates"
        )
    normalized_bindings = _normalize_artifact_bindings(
        artifact_bindings,
        artifacts=artifacts,
    )
    normalized_protected = _normalize_protected_bindings(protected_bindings)
    measurement_status = {
        field: "not_run" for field in MEASUREMENT_STATUS_FIELDS
    }
    side_effects = {field: False for field in SIDE_EFFECT_FIELDS}
    side_effects.update(
        {
            "filesystem_input_read_performed": True,
            "private_readback_report_created": True,
            "private_research_bundle_created": True,
        }
    )
    payload = {
        "schema_version": FORMAL_CATALOG_READBACK_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": next(iter(cycle_ids)),
        "run_id": run_id,
        "readiness": READINESS,
        "lifecycle_state": LIFECYCLE_STATE,
        "artifact_bindings": normalized_bindings,
        "source_accounting": {
            "source_candidate_count": 100,
            "new_candidate_count": 37,
            "structural_alias_count": 6,
            "incompatible_count": 57,
        },
        "catalog_accounting": {
            "base_candidate_count": 230,
            "new_candidate_count": 37,
            "candidate_count": 267,
        },
        "ontology_accounting": {
            "base_primitive_count": 13,
            "new_primitive_count": 5,
            "primitive_count": 18,
        },
        "mapping_proof_sha256": _artifact_semantic_sha256(
            materialization.PRIMITIVE_MAPPING_PROOF_FILENAME,
            proof,
        ),
        "formal_catalog_sha256": _artifact_semantic_sha256(
            materialization.FORMAL_CATALOG_FILENAME,
            catalog,
        ),
        "adapter_validation_sha256": _artifact_semantic_sha256(
            adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME,
            adapter_validation,
        ),
        "protected_bindings": normalized_protected,
        "protected_bindings_semantic_sha256": semantic_sha256_v4_1(
            normalized_protected
        ),
        **_authority_constants(),
        "measurement_status": measurement_status,
        "blockers": list(BLOCKERS),
        "side_effects": side_effects,
    }
    payload["report_semantic_sha256"] = semantic_sha256_v4_1(payload)
    return payload


def build_formal_catalog_readback_report_v4_1(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
    protected_bindings: Any,
) -> dict[str, Any]:
    """Build the exact non-authoritative private-bundle readback report."""

    return _expected_readback_report(
        run_id=run_id,
        artifacts=artifacts,
        artifact_bindings=artifact_bindings,
        protected_bindings=protected_bindings,
    )


def validate_formal_catalog_readback_report_v4_1(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    protected_bindings: Any,
) -> dict[str, Any]:
    """Validate one report against its six bound input artifact values."""

    payload = _exact(value, _READBACK_FIELDS, "formal catalog readback report")
    bindings_raw = payload["artifact_bindings"]
    if not isinstance(bindings_raw, list):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "readback artifact_bindings must be a list"
        )
    expected = _expected_readback_report(
        run_id=payload["run_id"],
        artifacts=artifacts,
        artifact_bindings=[
            {
                key: row[key]
                for key in _ARTIFACT_BINDING_FIELDS - {"semantic_sha256"}
            }
            for row in bindings_raw
        ],
        protected_bindings=protected_bindings,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "formal catalog readback report differs from exact recomputation"
        )
    return expected


def validate_formal_catalog_bundle_values_v4_1(
    values: Mapping[str, Mapping[str, Any]],
    *,
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    source_bindings: Any,
    code_bindings: Any,
    protected_bindings: Any,
) -> dict[str, dict[str, Any]]:
    """Cross-validate all seven artifacts through complete recomputation."""

    if not isinstance(values, Mapping) or set(values) != set(
        FORMAL_CATALOG_BUNDLE_FILENAMES
    ):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "formal catalog bundle must contain exactly seven canonical artifacts"
        )
    adapter_validation = values[
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME
    ]
    materialized = {
        filename: values[filename]
        for filename in materialization.FORMAL_CATALOG_MATERIALIZATION_FILENAMES
    }
    normalized_materialized = (
        materialization.validate_formal_catalog_materialization_v4_1(
            materialized,
            discovery_values=discovery_values,
            base_ontology=base_ontology,
            base_catalog=base_catalog,
            source_bindings=source_bindings,
            code_bindings=code_bindings,
            adapter_validation=adapter_validation,
        )
    )
    normalized_adapter = adapter.validate_formal_catalog_adapter_validation_v4_1(
        adapter_validation,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
        ontology=normalized_materialized[materialization.FORMAL_ONTOLOGY_FILENAME],
        catalog=normalized_materialized[materialization.FORMAL_CATALOG_FILENAME],
        mapping_proof=normalized_materialized[
            materialization.PRIMITIVE_MAPPING_PROOF_FILENAME
        ],
    )
    normalized_inputs = {
        **normalized_materialized,
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME: normalized_adapter,
    }
    normalized_report = validate_formal_catalog_readback_report_v4_1(
        values[FORMAL_CATALOG_READBACK_REPORT_FILENAME],
        artifacts=normalized_inputs,
        protected_bindings=protected_bindings,
    )
    return {
        **normalized_inputs,
        FORMAL_CATALOG_READBACK_REPORT_FILENAME: normalized_report,
    }


def build_formal_catalog_bundle_contract_v4_1(
    *,
    expected_artifacts: Mapping[str, Mapping[str, Any]],
    discovery_values: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    source_bindings: Any,
    code_bindings: Any,
    protected_bindings: Any,
) -> PrivateBundleContract:
    """Create a strict I/O contract closed over one exact recomputable bundle."""

    if not isinstance(expected_artifacts, Mapping) or set(expected_artifacts) != set(
        FORMAL_CATALOG_INPUT_FILENAMES
    ):
        raise FactorGovernanceFormalCatalogBundleV4_1Error(
            "expected_artifacts must contain exactly six input artifacts"
        )
    expected = copy.deepcopy(dict(expected_artifacts))
    discovery = copy.deepcopy(dict(discovery_values))
    ontology = copy.deepcopy(dict(base_ontology))
    catalog = copy.deepcopy(dict(base_catalog))
    sources = copy.deepcopy(source_bindings)
    code = copy.deepcopy(code_bindings)
    protected = _normalize_protected_bindings(protected_bindings)

    def validate_artifact(filename: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if filename == FORMAL_CATALOG_READBACK_REPORT_FILENAME:
            return validate_formal_catalog_readback_report_v4_1(
                value,
                artifacts=expected,
                protected_bindings=protected,
            )
        if filename not in expected:
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"unexpected formal catalog artifact: {filename}"
            )
        if canonical_json_bytes_v4_1(value) != canonical_json_bytes_v4_1(
            expected[filename]
        ):
            raise FactorGovernanceFormalCatalogBundleV4_1Error(
                f"formal catalog artifact differs from expected recomputation: {filename}"
            )
        return copy.deepcopy(expected[filename])

    def validate_complete(
        values: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        return validate_formal_catalog_bundle_values_v4_1(
            values,
            discovery_values=discovery,
            base_ontology=ontology,
            base_catalog=catalog,
            source_bindings=sources,
            code_bindings=code,
            protected_bindings=protected,
        )

    def build_readback_report(
        *,
        run_id: str,
        artifacts: Mapping[str, Mapping[str, Any]],
        artifact_bindings: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return build_formal_catalog_readback_report_v4_1(
            run_id=run_id,
            artifacts=artifacts,
            artifact_bindings=artifact_bindings,
            protected_bindings=protected,
        )

    return PrivateBundleContract(
        root_suffix=FORMAL_CATALOG_PRIVATE_ROOT_SUFFIX,
        input_filenames=FORMAL_CATALOG_INPUT_FILENAMES,
        readback_report_filename=FORMAL_CATALOG_READBACK_REPORT_FILENAME,
        canonicalize=canonical_file_bytes_v4_1,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_readback_report,
    )


__all__ = [
    "BLOCKERS",
    "FORMAL_CATALOG_BUNDLE_FILENAMES",
    "FORMAL_CATALOG_INPUT_FILENAMES",
    "FORMAL_CATALOG_PRIVATE_ROOT_SUFFIX",
    "FORMAL_CATALOG_READBACK_REPORT_FILENAME",
    "FORMAL_CATALOG_READBACK_REPORT_SCHEMA_VERSION",
    "FactorGovernanceFormalCatalogBundleV4_1Error",
    "LIFECYCLE_STATE",
    "MEASUREMENT_STATUS_FIELDS",
    "PROTOCOL_VERSION",
    "PROTECTED_CONTROL_PATH_SUFFIXES",
    "READINESS",
    "SIDE_EFFECT_FIELDS",
    "build_formal_catalog_bundle_contract_v4_1",
    "build_formal_catalog_readback_report_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "semantic_sha256_v4_1",
    "validate_formal_catalog_bundle_values_v4_1",
    "validate_formal_catalog_readback_report_v4_1",
]
