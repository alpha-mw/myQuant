"""Hash-bound package, runtime, and predecessor resources for V17 v5."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping

from quant_investor.v17_v4_contract import (
    verify_package as verify_v4_package,
    verify_runtime_build as verify_v4_runtime_build,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field as v4_identity_field,
    schema_path_for_version as v4_schema_path,
)
from quant_investor.v17_v4_contract.resources import load_packaged_json as load_v4_json

from .canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)
from .identities import (
    IdentityContractError,
    require_git_commit,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from .limits import LIMITS
from .validators import (
    FACTOR_DIAGNOSTIC_POLICY_ID,
    FACTOR_DIAGNOSTIC_POLICY_VERSION,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_VERSION,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_VERSION,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    NO_AUTHORITY,
    V4_COMPATIBILITY_POLICY_ID,
    V4_COMPATIBILITY_POLICY_V1_ID,
    V4_COMPATIBILITY_POLICY_V1_VERSION,
    V4_COMPATIBILITY_POLICY_V2_ID,
    V4_COMPATIBILITY_POLICY_V2_VERSION,
    V4_COMPATIBILITY_POLICY_V3_ID,
    V4_COMPATIBILITY_POLICY_V3_VERSION,
    V4_COMPATIBILITY_POLICY_VERSION,
    V4_FACTOR_EVIDENCE_ADAPTER_POLICY_ID,
    V4_FACTOR_EVIDENCE_ADAPTER_POLICY_VERSION,
    V4_V2_PUBLICATION_BLOCK_CLI_SHA256,
)

PROTOCOL_VERSION: Final = "myquant.v17.v5"
PACKAGE_MANIFEST_PATH: Final = "resources/package_manifest.v1.json"
RUNTIME_BUILD_MANIFEST_PATH: Final = "resources/runtime_build_manifest.v1.json"
COMPATIBILITY_POLICY_V1_PATH: Final = "resources/v4_compatibility_policy.v1.json"
COMPATIBILITY_POLICY_V2_PATH: Final = "resources/v4_compatibility_policy.v2.json"
COMPATIBILITY_POLICY_V3_PATH: Final = "resources/v4_compatibility_policy.v3.json"
COMPATIBILITY_POLICY_PATH: Final = "resources/v4_compatibility_policy.v4.json"
FACTOR_DIAGNOSTIC_POLICY_PATH: Final = "resources/factor_diagnostic_policy.v1.json"
FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_PATH: Final = "resources/factor_regime_diagnostic_policy.v1.json"
FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_PATH: Final = "resources/factor_regime_diagnostic_policy.v2.json"
FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH: Final = "resources/factor_regime_diagnostic_policy.v3.json"
V4_FACTOR_EVIDENCE_ADAPTER_POLICY_PATH: Final = (
    "resources/v4_factor_evidence_adapter_policy.v1.json"
)
PACKAGE_MANIFEST_SHA256: Final = "c26de7e7348e3a1b56258203260bea90e04e8be7ec65fdce1b87407fe099318b"
_PACKAGE_ROOT: Final = Path(__file__).resolve().parent


class PackageResourceError(RuntimeError):
    """Raised when a sealed V17 v5 package resource drifts."""

    exit_code = 2


def _read_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = load_canonical_resource(raw, label=label)
    except (OSError, CanonicalContractError) as exc:
        raise PackageResourceError(f"{label} is unreadable or invalid") from exc
    if type(payload) is not dict:
        raise PackageResourceError(f"{label} root must be an object")
    return raw, dict(payload)


def load_package_manifest(*, package_root: Path | None = None) -> dict[str, Any]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    raw, manifest = _read_object(root / PACKAGE_MANIFEST_PATH, label="v5 package manifest")
    try:
        require_sha256(PACKAGE_MANIFEST_SHA256, label="package manifest SHA-256")
        validate_semantic_sha(manifest)
    except (IdentityContractError, CanonicalContractError) as exc:
        raise PackageResourceError("v5 package manifest seal is invalid") from exc
    if hashlib.sha256(raw).hexdigest() != PACKAGE_MANIFEST_SHA256:
        raise PackageResourceError("v5 package manifest byte SHA-256 mismatch")
    expected_keys = {
        "array_order_semantics",
        "assets",
        "authority",
        "protocol_version",
        "self_binding",
        "semantic_sha256",
        "source_paths",
        "version",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("version") != "myquant.v17.v5.package-manifest.v1"
        or manifest.get("authority") != NO_AUTHORITY
        or manifest.get("self_binding")
        != {
            "byte_sha256_source": (
                "quant_investor.v17_v5_contract.resources.PACKAGE_MANIFEST_SHA256"
            ),
            "relative_path": PACKAGE_MANIFEST_PATH,
        }
    ):
        raise PackageResourceError("v5 package manifest identity mismatch")
    return manifest


def _asset_rows(manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    rows = manifest.get("assets")
    if type(rows) is not list:
        raise PackageResourceError("v5 package assets must be an array")
    result: list[dict[str, str]] = []
    paths: set[str] = set()
    identities: set[str] = set()
    previous: str | None = None
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "artifact_id",
            "byte_sha256",
            "relative_path",
        }:
            raise PackageResourceError(f"v5 package asset row {index} shape mismatch")
        relative_path = row["relative_path"]
        artifact_id = row["artifact_id"]
        try:
            require_relative_path(relative_path)
            require_identifier(artifact_id, label="package artifact_id")
            require_sha256(row["byte_sha256"])
        except IdentityContractError as exc:
            raise PackageResourceError(f"v5 package asset row {index} invalid") from exc
        if (
            not relative_path.startswith(("resources/", "schemas/"))
            or not relative_path.endswith(".json")
            or relative_path == PACKAGE_MANIFEST_PATH
            or (previous is not None and relative_path <= previous)
            or relative_path.casefold() in paths
            or artifact_id.casefold() in identities
        ):
            raise PackageResourceError(f"v5 package asset row {index} noncanonical")
        previous = relative_path
        paths.add(relative_path.casefold())
        identities.add(artifact_id.casefold())
        result.append(dict(row))
    return tuple(result)


def _source_paths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = manifest.get("source_paths")
    if (
        type(values) is not list
        or values != sorted(values)
        or len(values) != len({value.casefold() for value in values if type(value) is str})
        or any(
            type(value) is not str or "/" in value or not value.endswith(".py") for value in values
        )
    ):
        raise PackageResourceError("v5 source path inventory is invalid")
    return tuple(values)


def read_packaged_asset(
    relative_path: str,
    *,
    package_root: Path | None = None,
) -> bytes:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    expected = {
        row["relative_path"]: row["byte_sha256"]
        for row in _asset_rows(load_package_manifest(package_root=root))
    }
    if relative_path not in expected:
        raise PackageResourceError(f"unknown V17 v5 package asset: {relative_path!r}")
    try:
        raw = (root / relative_path).read_bytes()
    except OSError as exc:
        raise PackageResourceError(f"v5 package asset unreadable: {relative_path}") from exc
    if hashlib.sha256(raw).hexdigest() != expected[relative_path]:
        raise PackageResourceError(f"v5 package asset SHA mismatch: {relative_path}")
    try:
        payload = load_canonical_resource(raw, label=relative_path)
        if type(payload) is not dict:
            raise PackageResourceError(f"v5 package asset is not an object: {relative_path}")
        if relative_path.startswith("resources/"):
            validate_semantic_sha(payload)
    except CanonicalContractError as exc:
        raise PackageResourceError(f"v5 package asset invalid: {relative_path}") from exc
    return raw


def load_packaged_json(
    relative_path: str,
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    payload = load_canonical_resource(
        read_packaged_asset(relative_path, package_root=package_root),
        label=relative_path,
    )
    if type(payload) is not dict:
        raise PackageResourceError(f"{relative_path} root must be an object")
    return deepcopy(payload)


def _load_compatibility_policy(
    *,
    package_root: Path | None = None,
    relative_path: str,
    expected_version: str,
    expected_artifact_id: str,
) -> dict[str, Any]:
    policy = load_packaged_json(relative_path, package_root=package_root)
    expected_keys = {
        "allowed_artifacts",
        "array_order_semantics",
        "artifact_id",
        "authority",
        "closure_limits",
        "forbidden_import_prefixes",
        "forbidden_v4_writer_modules",
        "predecessor",
        "protocol_version",
        "semantic_sha256",
        "version",
    }
    if (
        set(policy) != expected_keys
        or policy["protocol_version"] != PROTOCOL_VERSION
        or policy["version"] != expected_version
        or policy["artifact_id"] != expected_artifact_id
        or policy["authority"] != NO_AUTHORITY
        or policy["forbidden_import_prefixes"] != ["quant_investor.v17_v4_runtime"]
    ):
        raise PackageResourceError("v4 compatibility policy identity mismatch")
    limits = policy["closure_limits"]
    if limits != {
        "max_artifact_bytes": LIMITS["compat_max_artifact_bytes"],
        "max_closure_bytes": LIMITS["compat_max_closure_bytes"],
        "max_depth": LIMITS["compat_max_closure_depth"],
        "max_nodes": LIMITS["compat_max_closure_nodes"],
        "max_parquet_row_groups": LIMITS["compat_max_parquet_row_groups"],
        "max_parquet_rows": LIMITS["compat_max_parquet_rows"],
    }:
        raise PackageResourceError("v4 compatibility closure limits drift")
    rows = policy["allowed_artifacts"]
    if type(rows) is not list or not rows:
        raise PackageResourceError("v4 compatibility artifact allowlist is empty")
    versions: list[str] = []
    referenced_versions: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "allowed_path_prefixes",
            "identity_field",
            "root_admissible",
            "schema_id",
            "transitive_edges",
            "validation_mode",
            "version",
        }:
            raise PackageResourceError(f"v4 compatibility row {index} shape mismatch")
        version = row["version"]
        try:
            require_identifier(version, label="v4 artifact version")
        except IdentityContractError as exc:
            raise PackageResourceError(f"v4 compatibility row {index} invalid") from exc
        validation_mode = row["validation_mode"]
        if validation_mode == "V4_REGISTERED_JSON":
            try:
                schema = load_v4_json(v4_schema_path(version))
                expected_identity = v4_identity_field(version)
                expected_schema_id: str | None = str(schema.get("$id"))
            except Exception as exc:
                raise PackageResourceError(
                    f"v4 compatibility row {index} registered binding absent"
                ) from exc
        elif validation_mode in {
            "V4_GENERIC_CANONICAL_TERMINAL",
            "V4_PARQUET_METADATA",
        }:
            expected_identity = "artifact_id"
            expected_schema_id = None
        else:
            raise PackageResourceError(f"v4 compatibility row {index} validation mode invalid")
        prefixes = row["allowed_path_prefixes"]
        edges = row["transitive_edges"]
        if (
            row["identity_field"] != expected_identity
            or row["schema_id"] != expected_schema_id
            or type(row["root_admissible"]) is not bool
            or type(prefixes) is not list
            or not prefixes
            or prefixes != sorted(prefixes)
            or any(
                type(prefix) is not str
                or not prefix.startswith(
                    (
                        "data/private/v17_v4_",
                        "reports/factor_governance/private/monthly_factor_research",
                        "results/v17_v4_",
                    )
                )
                for prefix in prefixes
            )
            or type(edges) is not list
            or edges
            != sorted(
                edges,
                key=lambda edge: edge["json_pointer"] if type(edge) is dict else "",
            )
        ):
            raise PackageResourceError(f"v4 compatibility row {index} binding mismatch")
        for edge_index, edge in enumerate(edges):
            if type(edge) is not dict or set(edge) != {
                "cardinality",
                "json_pointer",
                "mode",
                "target_versions",
            }:
                raise PackageResourceError(
                    f"v4 compatibility row {index} edge {edge_index} shape mismatch"
                )
            targets = edge["target_versions"]
            if (
                edge["cardinality"]
                not in {
                    "EXACT_ONE",
                    "ONE_OR_MORE",
                    "ONE_PER_PARENT_ROW",
                    "ZERO_OR_MORE",
                    "ZERO_OR_ONE",
                }
                or edge["mode"]
                not in {
                    "DECODED_REF_SCAN",
                    "FOLLOW",
                    "TERMINAL_SOURCE_BINDING",
                }
                or type(edge["json_pointer"]) is not str
                or not edge["json_pointer"].startswith("/")
                or type(targets) is not list
                or targets != sorted(targets)
                or len(targets) != len(set(targets))
                or (edge["mode"] == "TERMINAL_SOURCE_BINDING" and targets)
                or (edge["mode"] != "TERMINAL_SOURCE_BINDING" and not targets)
                or any(type(target) is not str for target in targets)
            ):
                raise PackageResourceError(
                    f"v4 compatibility row {index} edge {edge_index} invalid"
                )
            referenced_versions.update(targets)
        if validation_mode != "V4_REGISTERED_JSON" and edges:
            raise PackageResourceError(
                f"v4 compatibility row {index} terminal has transitive edges"
            )
        versions.append(version)
    if versions != sorted(versions) or len(versions) != len(set(versions)):
        raise PackageResourceError("v4 compatibility versions are not uniquely ordered")
    if not referenced_versions.issubset(versions):
        raise PackageResourceError("v4 compatibility edge target is not allowlisted")
    predecessor = policy["predecessor"]
    try:
        require_git_commit(predecessor["source_git_commit"])
        require_sha256(predecessor["package_manifest_byte_sha256"])
        require_sha256(predecessor["runtime_manifest_byte_sha256"])
        require_relative_path(predecessor["package_manifest_relative_path"])
        require_relative_path(predecessor["runtime_manifest_relative_path"])
    except (IdentityContractError, KeyError, TypeError) as exc:
        raise PackageResourceError("v4 predecessor policy binding invalid") from exc
    if predecessor["protocol_version"] != "myquant.v17.v4":
        raise PackageResourceError("v4 predecessor protocol mismatch")
    forbidden = policy["forbidden_v4_writer_modules"]
    if (
        type(forbidden) is not list
        or forbidden != sorted(forbidden)
        or any(
            type(value) is not str or not value.startswith("quant_investor.v17_v4_runtime.")
            for value in forbidden
        )
    ):
        raise PackageResourceError("v4 forbidden writer inventory invalid")
    return policy


def load_compatibility_policy_v1(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_compatibility_policy(
        package_root=package_root,
        relative_path=COMPATIBILITY_POLICY_V1_PATH,
        expected_version=V4_COMPATIBILITY_POLICY_V1_VERSION,
        expected_artifact_id=V4_COMPATIBILITY_POLICY_V1_ID,
    )


def load_compatibility_policy_v2(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_compatibility_policy(
        package_root=package_root,
        relative_path=COMPATIBILITY_POLICY_V2_PATH,
        expected_version=V4_COMPATIBILITY_POLICY_V2_VERSION,
        expected_artifact_id=V4_COMPATIBILITY_POLICY_V2_ID,
    )


def load_compatibility_policy_v3(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_compatibility_policy(
        package_root=package_root,
        relative_path=COMPATIBILITY_POLICY_V3_PATH,
        expected_version=V4_COMPATIBILITY_POLICY_V3_VERSION,
        expected_artifact_id=V4_COMPATIBILITY_POLICY_V3_ID,
    )


def load_compatibility_policy(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_compatibility_policy(
        package_root=package_root,
        relative_path=COMPATIBILITY_POLICY_PATH,
        expected_version=V4_COMPATIBILITY_POLICY_VERSION,
        expected_artifact_id=V4_COMPATIBILITY_POLICY_ID,
    )


def load_factor_diagnostic_policy(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    policy = load_packaged_json(
        FACTOR_DIAGNOSTIC_POLICY_PATH,
        package_root=package_root,
    )
    if (
        set(policy)
        != {
            "artifact_id",
            "authority",
            "input_contract",
            "limits",
            "protocol_version",
            "rank_ic",
            "sample_policy",
            "semantic_sha256",
            "statuses",
            "version",
        }
        or policy["artifact_id"] != FACTOR_DIAGNOSTIC_POLICY_ID
        or policy["authority"] != NO_AUTHORITY
        or policy["protocol_version"] != PROTOCOL_VERSION
        or policy["version"] != FACTOR_DIAGNOSTIC_POLICY_VERSION
        or policy["statuses"] != ["ACCUMULATING", "UNAVAILABLE", "UNOBSERVED"]
        or policy["limits"]
        != {
            "max_origins": 4_096,
            "max_symbols_per_origin": 10_000,
            "max_total_symbol_rows": 2_000_000,
        }
        or policy["sample_policy"]
        != {
            "descriptive_coverage_minimum_origins": 60,
            "descriptive_coverage_minimum_symbols_per_origin": 100,
            "horizon_sessions": 20,
            "inference_gate_passed": False,
            "naturally_matured_only": True,
        }
        or policy["rank_ic"]
        != {
            "negative_zero": "ZERO",
            "output_decimal_places": 12,
            "rounding": "ROUND_HALF_EVEN",
            "symbol_order": "ASCII_ASCENDING",
            "ties": "AVERAGE_RANK_EXACT_DECIMAL",
        }
        or policy["input_contract"]
        != {
            "canonical_decimal": (
                "no_exponent_no_plus_no_trailing_fractional_zero_" "negative_zero_forbidden"
            ),
            "constant_input": "ORIGIN_RANK_IC_UNAVAILABLE",
            "duplicate_conflict": "MALFORMED_EXIT_2",
            "duplicate_identical": "DEDUPLICATE",
            "maturity": ("EXACT_SHANGHAI_OPEN_SESSION_DISTANCE_AND_" "LABEL_AVAILABLE_AT_CUTOFF"),
            "nonfinite": "MALFORMED_EXIT_2",
            "symbol_intersection": "FACTOR_AND_FORWARD_RETURN",
        }
    ):
        raise PackageResourceError("factor diagnostic policy identity mismatch")
    return policy


def _load_factor_regime_diagnostic_policy(
    *,
    package_root: Path | None = None,
    relative_path: str,
    expected_version: str,
    expected_artifact_id: str,
) -> dict[str, Any]:
    policy = load_packaged_json(
        relative_path,
        package_root=package_root,
    )
    v1_expected_keys = {
        "accepted_factor_evidence_versions",
        "accepted_label_versions",
        "artifact_id",
        "authority",
        "conditioning_dimension",
        "decimal_quantization",
        "deterministic_ordering",
        "horizon_sessions",
        "minimum_descriptive_origins",
        "minimum_stability_origins",
        "missing_evidence_statuses",
        "newey_west_lag",
        "no_governance_action",
        "origin_binding_policy",
        "overlap_adjustment_policy",
        "protocol_version",
        "publication_causality_policy",
        "regime_source_versions",
        "semantic_sha256",
        "status_contract",
        "version",
    }
    v2_expected_keys = {
        "accepted_factor_evidence_versions",
        "accepted_label_versions",
        "accepted_regime_source_versions",
        "artifact_id",
        "authority",
        "conditioning_dimension",
        "conditioning_ineligible_states",
        "conditioning_states",
        "decimal_quantization",
        "deterministic_ordering",
        "horizon_sessions",
        "minimum_descriptive_origins",
        "minimum_stability_origins",
        "missing_evidence_statuses",
        "newey_west_lag",
        "no_governance_action",
        "origin_binding_policy",
        "overlap_adjustment_policy",
        "protocol_version",
        "publication_causality_policy",
        "regime_source_requirements",
        "required_hard_state_derivation",
        "required_inference_kind",
        "required_no_retroactive_causal_backfill",
        "required_publication_phase",
        "required_scope_kind",
        "required_smoothing_used",
        "semantic_sha256",
        "status_contract",
        "version",
    }
    v3_expected_keys = v2_expected_keys | {
        "conditioning_eligible_continuity",
        "conditioning_ineligible_continuity",
        "segment_sequence_rules",
    }
    is_v2 = expected_version == FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_VERSION
    is_v3 = expected_version == FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION
    if (
        set(policy)
        != (v3_expected_keys if is_v3 else v2_expected_keys if is_v2 else v1_expected_keys)
        or policy["artifact_id"] != expected_artifact_id
        or policy["authority"] != NO_AUTHORITY
        or policy["protocol_version"] != PROTOCOL_VERSION
        or policy["version"] != expected_version
        or policy["conditioning_dimension"] != "ORIGIN_REGIME"
        or policy["horizon_sessions"] != 20
        or policy["minimum_descriptive_origins"] != 20
        or policy["minimum_stability_origins"] != 60
        or policy["newey_west_lag"] != 19
        or policy["missing_evidence_statuses"] != ["UNAVAILABLE", "UNOBSERVED"]
        or policy["status_contract"] != ["ACCUMULATING", "UNAVAILABLE", "UNOBSERVED"]
        or policy["accepted_factor_evidence_versions"]
        != [
            "myquant.v17.v4.factor-universe-observation.v1",
            "myquant.v17.v4.forward-evaluation-receipt.v1",
        ]
        or policy["accepted_label_versions"] != ["myquant.v17.v4.forward-label.v1"]
        or policy["decimal_quantization"]
        != {
            "decimal_places": 12,
            "negative_zero": "ZERO",
            "rounding": "ROUND_HALF_EVEN",
        }
        or policy["overlap_adjustment_policy"]
        != {
            "bartlett_weight": "1-k/(lag+1)",
            "gamma_denominator": "n",
            "newey_west_variance_of_mean": "long_run_variance/n",
            "overlapping_label_sessions": 20,
        }
        or policy["publication_causality_policy"]
        != {
            "available_at": "NOT_AFTER_FACTOR_ORIGIN_CUTOFF",
            "created_at_is_not_published_at": True,
            "future_or_smoothed_state": "FORBIDDEN",
            "published_at": "REQUIRED_AND_NOT_AFTER_FACTOR_ORIGIN_CUTOFF",
            "raw_or_mutable_regime_reconstruction": "FORBIDDEN",
        }
    ):
        raise PackageResourceError("factor regime diagnostic policy identity mismatch")
    if is_v2 or is_v3:
        if (
            policy["accepted_regime_source_versions"]
            != [
                (
                    "myquant.v17.v4.regime-evidence.v3"
                    if is_v3
                    else "myquant.v17.v4.regime-evidence.v2"
                )
            ]
            or policy["required_inference_kind"] != "FILTERED_CAUSAL"
            or policy["required_smoothing_used"] is not False
            or policy["required_publication_phase"] != "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
            or policy["required_scope_kind"] != "FULL_MARKET"
            or policy["required_hard_state_derivation"] != "SEALED_ARGMAX_POLICY_V1"
            or policy["required_no_retroactive_causal_backfill"] is not True
            or policy["conditioning_states"] != ["趋势上涨", "震荡低波", "震荡高波", "趋势下跌"]
            or policy["conditioning_ineligible_states"] != ["未知"]
            or (
                not is_v3
                and policy["regime_source_requirements"]
                != {
                    "hard_state_derivation": "SEALED_ARGMAX_POLICY_V1",
                    "inference_kind": "FILTERED_CAUSAL",
                    "no_retroactive_causal_backfill": True,
                    "publication_phase": "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
                    "scope_kind": "FULL_MARKET",
                    "smoothing_used": False,
                    "state_probabilities_sum": "1.000000000000",
                }
            )
        ):
            raise PackageResourceError(
                f"factor regime diagnostic policy {'v3' if is_v3 else 'v2'} mismatch"
            )
        if is_v3 and (
            policy["conditioning_eligible_continuity"] != ["CONTIGUOUS", "ROLLOVER"]
            or policy["conditioning_ineligible_continuity"] != ["GENESIS", "RECOVERY"]
            or policy["segment_sequence_rules"]
            != {
                "CONTIGUOUS": "SEQUENCE_GT_ZERO_ELIGIBLE",
                "GENESIS": "SEQUENCE_ZERO_INELIGIBLE",
                "RECOVERY": "SEQUENCE_ZERO_INELIGIBLE",
                "ROLLOVER": "ELIGIBLE_AFTER_V4_COMPOSITE_FINALITY",
            }
            or policy["regime_source_requirements"]
            != {
                "bounded_replay": "ROOT_AND_CURRENT_DIRECT_CLOSURE_ONLY",
                "finalized": "EVIDENCE_CURRENT_CHECKPOINT_COMPOSITE",
                "hard_state_derivation": "SEALED_ARGMAX_POLICY_V1",
                "inference_kind": "FILTERED_CAUSAL",
                "no_retroactive_causal_backfill": True,
                "publication_phase": "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
                "scope_kind": "FULL_MARKET",
                "smoothing_used": False,
                "state_probabilities_sum": "1.000000000000",
            }
        ):
            raise PackageResourceError("factor regime diagnostic policy v3 mismatch")
    else:
        if policy["regime_source_versions"] != {
            "capabilities": [
                {
                    "capability_status": ("CONDITIONING_INELIGIBLE_HARD_STATE_ABSENT"),
                    "decision_session": "ABSENT",
                    "effective_session": "ABSENT",
                    "hard_state": "ABSENT",
                    "posterior": "ABSENT",
                    "published_at": "ABSENT",
                    "source_refs": "ABSENT",
                    "version": "myquant.v17.v4.regime-evidence.v1",
                }
            ],
            "conditioning_eligible": [],
            "registered": ["myquant.v17.v4.regime-evidence.v1"],
        }:
            raise PackageResourceError("factor regime diagnostic policy v1 mismatch")
    return policy


def load_factor_regime_diagnostic_policy_v1(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_factor_regime_diagnostic_policy(
        package_root=package_root,
        relative_path=FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_PATH,
        expected_version=FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_VERSION,
        expected_artifact_id=FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_ID,
    )


def load_factor_regime_diagnostic_policy_v2(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_factor_regime_diagnostic_policy(
        package_root=package_root,
        relative_path=FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_PATH,
        expected_version=FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_VERSION,
        expected_artifact_id=FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_ID,
    )


def load_factor_regime_diagnostic_policy(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    return _load_factor_regime_diagnostic_policy(
        package_root=package_root,
        relative_path=FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
        expected_version=FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
        expected_artifact_id=FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    )


def load_v4_factor_evidence_adapter_policy(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    policy = load_packaged_json(
        V4_FACTOR_EVIDENCE_ADAPTER_POLICY_PATH,
        package_root=package_root,
    )
    if (
        set(policy)
        != {
            "artifact_id",
            "authority",
            "evidence_contract",
            "failure_contract",
            "lineage_contract",
            "output_contract",
            "protocol_version",
            "semantic_sha256",
            "version",
        }
        or policy["artifact_id"] != V4_FACTOR_EVIDENCE_ADAPTER_POLICY_ID
        or policy["authority"] != NO_AUTHORITY
        or policy["protocol_version"] != PROTOCOL_VERSION
        or policy["version"] != V4_FACTOR_EVIDENCE_ADAPTER_POLICY_VERSION
        or policy["evidence_contract"]
        != {
            "evaluation_receipt_completeness": "COMPLETE",
            "evaluation_receipt_execution_outcome": "SUCCEEDED",
            "evaluation_receipt_type": "factor_evaluation_receipt",
            "factor_inventory_rows": "EXACTLY_ONE_MATCHING_FACTOR",
            "horizon_sessions": 20,
            "label_completeness": "COMPLETE",
            "label_return_field": "total_return",
            "observation_completeness": "COMPLETE",
            "origin_rows_per_receipt": "EXACTLY_ONE_MATCHING_LINEAGE",
            "request_profile": "FORWARD_EVIDENCE",
            "run_state": "FORWARD_EVIDENCE_ACTIVE",
        }
        or policy["failure_contract"]
        != {
            "contradictory_or_tampered_closure": "MALFORMED_EXIT_2_NO_ARTIFACT",
            "missing_registered_dependency": "UNAVAILABLE",
            "mixed_stratum": "MALFORMED_EXIT_2_NO_ARTIFACT",
            "no_matching_factor_or_label": "UNAVAILABLE",
        }
        or policy["lineage_contract"]
        != {
            "origin_evidence_preimage": [
                "decision_session",
                "factor_observation_ref",
                "factor_set_ref",
                "forward_label_ref",
                "observation_run_ref",
                "request_ref",
                "source_locator_ref",
            ],
            "source_series_preimage": [
                "factor_input_bundle_version",
                "factor_slice_fields",
                "neutralizer_fields",
                "required_fields",
                "source_locator_version",
            ],
        }
        or policy["output_contract"]
        != {
            "effectiveness_claimed": False,
            "factor_governance_write": False,
            "factor_tier_change_eligible": False,
            "factor_weight_change_eligible": False,
            "inference_implemented": False,
            "promotion_eligible": False,
        }
    ):
        raise PackageResourceError("V4 factor evidence adapter policy identity mismatch")
    return policy


def verify_runtime_build(
    *,
    package_root: Path | None = None,
) -> dict[str, str]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_packaged_json(RUNTIME_BUILD_MANIFEST_PATH, package_root=root)
    if (
        set(manifest)
        != {
            "array_order_semantics",
            "authority",
            "protocol_version",
            "semantic_sha256",
            "sources",
            "version",
        }
        or manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["version"] != "myquant.v17.v5.runtime-build-manifest.v1"
        or manifest["authority"] != NO_AUTHORITY
    ):
        raise PackageResourceError("v5 runtime manifest identity mismatch")
    rows = manifest["sources"]
    if type(rows) is not list or not rows:
        raise PackageResourceError("v5 runtime source inventory is empty")
    expected: list[str] = []
    result: dict[str, str] = {}
    previous: str | None = None
    quant_root = root.parent
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"byte_sha256", "relative_path"}:
            raise PackageResourceError(f"v5 runtime source row {index} shape mismatch")
        relative = row["relative_path"]
        try:
            require_relative_path(relative)
            require_sha256(row["byte_sha256"])
        except IdentityContractError as exc:
            raise PackageResourceError(f"v5 runtime source row {index} invalid") from exc
        if (
            not relative.startswith("v17_v5_runtime/")
            or not relative.endswith(".py")
            or (previous is not None and relative <= previous)
        ):
            raise PackageResourceError(f"v5 runtime source row {index} noncanonical")
        previous = relative
        expected.append(relative)
        raw = (quant_root / relative).read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != row["byte_sha256"]:
            raise PackageResourceError(f"v5 runtime source SHA mismatch: {relative}")
        result[relative] = observed
    discovered = sorted(
        path.relative_to(quant_root).as_posix()
        for path in (quant_root / "v17_v5_runtime").glob("*.py")
    )
    if discovered != expected:
        raise PackageResourceError("v5 runtime Python inventory differs from manifest")
    return result


def verify_package(
    *,
    package_root: Path | None = None,
) -> dict[str, str]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    manifest = load_package_manifest(package_root=root)
    rows = _asset_rows(manifest)
    expected_paths = [row["relative_path"] for row in rows]
    discovered = sorted(
        path.relative_to(root).as_posix()
        for directory in ("resources", "schemas")
        for path in (root / directory).glob("*.json")
        if path.relative_to(root).as_posix() != PACKAGE_MANIFEST_PATH
    )
    if discovered != expected_paths:
        raise PackageResourceError("v5 packaged JSON inventory differs from manifest")
    sources = sorted(path.name for path in root.glob("*.py"))
    if sources != list(_source_paths(manifest)):
        raise PackageResourceError("v5 contract Python inventory differs from manifest")
    result: dict[str, str] = {}
    for row in rows:
        raw = read_packaged_asset(row["relative_path"], package_root=root)
        result[row["relative_path"]] = hashlib.sha256(raw).hexdigest()
    load_compatibility_policy_v1(package_root=root)
    load_compatibility_policy_v2(package_root=root)
    load_compatibility_policy(package_root=root)
    load_factor_diagnostic_policy(package_root=root)
    load_factor_regime_diagnostic_policy_v1(package_root=root)
    load_factor_regime_diagnostic_policy_v2(package_root=root)
    load_factor_regime_diagnostic_policy(package_root=root)
    load_v4_factor_evidence_adapter_policy(package_root=root)
    verify_runtime_build(package_root=root)
    return result


def verify_predecessor(
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    root = _PACKAGE_ROOT if package_root is None else Path(package_root)
    policy = load_compatibility_policy(package_root=root)
    predecessor = policy["predecessor"]
    workspace_root = root.parent.parent
    package_path = workspace_root / predecessor["package_manifest_relative_path"]
    runtime_path = workspace_root / predecessor["runtime_manifest_relative_path"]
    try:
        package_sha = hashlib.sha256(package_path.read_bytes()).hexdigest()
        runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PackageResourceError("pinned V17 v4 manifests are unreadable") from exc
    if (
        package_sha != predecessor["package_manifest_byte_sha256"]
        or runtime_sha != predecessor["runtime_manifest_byte_sha256"]
    ):
        raise PackageResourceError("pinned V17 v4 predecessor manifest drift")
    v4_assets = verify_v4_package()
    v4_sources = verify_v4_runtime_build()
    evidence_schema_sha = hashlib.sha256(
        (
            workspace_root / "quant_investor/v17_v4_contract/schemas/regime_evidence.v3.schema.json"
        ).read_bytes()
    ).hexdigest()
    inference_policy_sha = hashlib.sha256(
        (
            workspace_root
            / "quant_investor/v17_v4_contract/resources/regime_inference_policy.v2.json"
        ).read_bytes()
    ).hexdigest()
    producer_sha = hashlib.sha256(
        (workspace_root / "quant_investor/v17_v4_runtime/regime_evidence_v3.py").read_bytes()
    ).hexdigest()
    cli_raw = (workspace_root / "quant_investor/v17_v4_runtime/cli.py").read_bytes()
    cli_sha = hashlib.sha256(cli_raw).hexdigest()
    if (
        len(v4_assets) != 114
        or len(v4_sources) != 34
        or evidence_schema_sha != "429c9ed6f664ae70f0a34d92e0a94bc10293291217d58eb22f2fb2e36e83ab80"
        or inference_policy_sha
        != "46733a14377476c43ed230f9167dd786795c9b01159755cf91f358d07d44a3c1"
        or producer_sha != "b9819326d32df1f094ecc5954f3664c36f060d9e5e3044adaaf17c4abb8b4180"
        or cli_sha != V4_V2_PUBLICATION_BLOCK_CLI_SHA256
        or b"REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE" not in cli_raw
    ):
        raise PackageResourceError("pinned V17 v4 bounded-regime predecessor drift")
    return {
        "regime_evidence_v3_runtime_sha256": producer_sha,
        "regime_evidence_v3_schema_sha256": evidence_schema_sha,
        "regime_inference_policy_v2_sha256": inference_policy_sha,
        "package_asset_count": len(v4_assets),
        "package_manifest_byte_sha256": package_sha,
        "protocol_version": predecessor["protocol_version"],
        "runtime_manifest_byte_sha256": runtime_sha,
        "runtime_source_count": len(v4_sources),
        "source_git_commit": predecessor["source_git_commit"],
        "status": "PINNED_AND_VERIFIED",
        "v2_cli_source_sha256": cli_sha,
        "v2_publication_status": "REGIME_EVIDENCE_V2_CHAIN_NON_DEPLOYABLE",
    }


__all__ = [
    "COMPATIBILITY_POLICY_PATH",
    "COMPATIBILITY_POLICY_V2_PATH",
    "COMPATIBILITY_POLICY_V3_PATH",
    "FACTOR_DIAGNOSTIC_POLICY_PATH",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V1_PATH",
    "FACTOR_REGIME_DIAGNOSTIC_POLICY_V2_PATH",
    "PACKAGE_MANIFEST_PATH",
    "PACKAGE_MANIFEST_SHA256",
    "PackageResourceError",
    "RUNTIME_BUILD_MANIFEST_PATH",
    "load_compatibility_policy",
    "load_compatibility_policy_v1",
    "load_compatibility_policy_v2",
    "load_compatibility_policy_v3",
    "load_factor_diagnostic_policy",
    "load_factor_regime_diagnostic_policy",
    "load_factor_regime_diagnostic_policy_v1",
    "load_factor_regime_diagnostic_policy_v2",
    "load_package_manifest",
    "load_packaged_json",
    "read_packaged_asset",
    "verify_package",
    "verify_predecessor",
    "verify_runtime_build",
]
