"""Owner-private exact-once bundle for Factor v4.4 preregistration.

The fixed bundle contains 26 inputs and one generated readback report.  The
fourteen prefixed v4.2 files and three v4.3 diagnostic files must be supplied
as their original canonical bytes; parsing and re-serialization is never used
as proof of original-byte preservation.  Durable no-clobber publication is
delegated to :mod:`governance_private_bundle_io`.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_2 as v42_bundle,
)
from quant_investor.factors import governance_candidate_preregistration_v4_2 as v42
from quant_investor.factors import governance_candidate_preregistration_v4_4 as core
from quant_investor.factors import (
    governance_prior_diagnostic_nomination_bundle_v4_3 as diagnostic_bundle,
)
from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    validate_cycle_state_v4_1,
)
from quant_investor.factors.governance_private_bundle_io import (
    PrivateBundleContract,
    publish_private_bundle,
    readback_private_bundle,
)


ROOT_SUFFIX_V4_4 = (
    "reports",
    "factor_governance",
    "private",
    "v4_4_candidate_preregistration",
)

CODE_BINDING_SET_FILENAME_V4_4 = "code_binding_set.v4_4.json"
EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4 = (
    "expanded_candidate_selection.v4_4.json"
)
DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4 = (
    "definition_identity_collision_audit.v4_4.json"
)
CYCLE_ROOT_FILENAME_V4_4 = "cycle_root.v4_4.json"
FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4 = "future_source_envelope.v4_4.json"
PRECOMMITTED_STATE_FILENAME_V4_4 = "cycle_state.precommitted.v4_1.json"
DISCOVERY_SOURCE_NODE_FILENAME_V4_4 = "discovery_source_node.v4_4.json"
DISCOVERY_STATE_FILENAME_V4_4 = "cycle_state.discovery.v4_1.json"
PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4 = (
    "prereg_discovery_orchestration.v4_4.json"
)
READBACK_REPORT_FILENAME_V4_4 = "candidate_preregistration_readback.v4_4.json"

V4_2_PREDECESSOR_FILENAMES_V4_4 = core.V4_2_PREDECESSOR_FILENAMES
PRIOR_DIAGNOSTIC_FILENAMES_V4_4 = core.PRIOR_DIAGNOSTIC_FILENAMES
INPUT_FILENAMES_V4_4 = (
    *V4_2_PREDECESSOR_FILENAMES_V4_4,
    CODE_BINDING_SET_FILENAME_V4_4,
    *PRIOR_DIAGNOSTIC_FILENAMES_V4_4,
    EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4,
    DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4,
    CYCLE_ROOT_FILENAME_V4_4,
    FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4,
    PRECOMMITTED_STATE_FILENAME_V4_4,
    DISCOVERY_SOURCE_NODE_FILENAME_V4_4,
    DISCOVERY_STATE_FILENAME_V4_4,
    PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4,
)
COLLECTED_RAW_FILENAMES_V4_4 = (
    *V4_2_PREDECESSOR_FILENAMES_V4_4,
    *PRIOR_DIAGNOSTIC_FILENAMES_V4_4,
)

READBACK_REPORT_SCHEMA_VERSION_V4_4 = (
    "factor-governance-candidate-preregistration-readback.v4.4"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class FactorGovernanceCandidatePreregistrationBundleV4_4Error(ValueError):
    """Raised when the v4.4 private bundle fails closed."""


def _error(message: str) -> FactorGovernanceCandidatePreregistrationBundleV4_4Error:
    return FactorGovernanceCandidatePreregistrationBundleV4_4Error(message)


def _exact(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(set(fields) - set(payload))
    unknown = sorted(set(payload) - set(fields))
    if missing or unknown:
        raise _error(
            f"{label} fields invalid: missing={','.join(missing) or '-'};"
            f"unknown={','.join(unknown) or '-'}"
        )
    core.canonical_json_bytes_v4_4(payload)
    return payload


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["artifact_semantic_sha256"] = core.semantic_sha256_v4_4(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _validate_self(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    supplied = _sha256(
        payload.get("artifact_semantic_sha256"),
        f"{label} artifact semantic SHA",
    )
    if supplied != core.semantic_sha256_v4_4(
        {key: item for key, item in payload.items() if key != "artifact_semantic_sha256"}
    ):
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return payload


def _semantic(value: Mapping[str, Any], label: str) -> str:
    if "artifact_semantic_sha256" in value:
        return _sha256(
            value["artifact_semantic_sha256"],
            f"{label} artifact semantic SHA",
        )
    if "state_semantic_sha256" in value:
        return _sha256(
            value["state_semantic_sha256"],
            f"{label} state semantic SHA",
        )
    raise _error(f"{label} has no accepted semantic identity")


def _canonical_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return core.canonical_file_bytes_v4_4(left) == core.canonical_file_bytes_v4_4(
        right
    )


def _prefixed_v42(values: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        filename: copy.deepcopy(
            dict(values[core.V4_2_PREDECESSOR_PREFIX + filename])
        )
        for filename in v42_bundle.INPUT_FILENAMES_V4_2
    }


def _normalized_v42_graph(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    prefixed = {
        filename: values[filename] for filename in V4_2_PREDECESSOR_FILENAMES_V4_4
    }
    normalized_prefixed = core.validate_v4_2_predecessor_graph_v4_4(prefixed)
    return _prefixed_v42(normalized_prefixed)


def _fully_validate_v42_future_envelope(
    graph: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = graph[v42_bundle.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2]
    envelope = graph[v42_bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2]
    try:
        return v42.validate_future_source_envelope_v4_2(
            envelope,
            selection_spec=graph[v42_bundle.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2],
            aquant_receipt=graph[
                v42_bundle.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
            ],
            myquant_receipt=graph[
                v42_bundle.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
            ],
            operator_semantics=graph[v42_bundle.OPERATOR_SEMANTICS_FILENAME_V4_2],
            comparison_catalog_receipt=graph[
                v42_bundle.COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2
            ],
            code_binding_set_semantic_sha256=graph[
                v42_bundle.CODE_BINDING_SET_FILENAME_V4_2
            ]["artifact_semantic_sha256"],
            strict_source_binding_semantic_sha256=source[
                "artifact_semantic_sha256"
            ],
            full_a_scope_sha256=source["full_a_scope_sha256"],
            full_a_scope_count=source["expected_scope_count"],
            serving_inventory_count=source["serving_inventory_count"],
        )
    except Exception as exc:
        raise _error(f"embedded v4.2 future envelope validation failed: {exc}") from exc


def _diagnostic_graph(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return core.validate_prior_diagnostic_graph_v4_4(
        {filename: values[filename] for filename in PRIOR_DIAGNOSTIC_FILENAMES_V4_4}
    )


def _artifact_bindings(
    filenames: Sequence[str], values: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        core.build_artifact_binding_v4_4(filename=filename, artifact=values[filename])
        for filename in filenames
    ]


def _build_from_normalized(
    *,
    v42_graph: Mapping[str, Mapping[str, Any]],
    diagnostic_graph: Mapping[str, Mapping[str, Any]],
    code_binding_set: Mapping[str, Any],
    publication_at: str,
) -> dict[str, dict[str, Any]]:
    # Every v4.2 leaf, crosslink, and deterministic builder has already been
    # checked before any v4.4 artifact is derived.
    _fully_validate_v42_future_envelope(v42_graph)
    prefixed = {
        core.V4_2_PREDECESSOR_PREFIX + filename: copy.deepcopy(v42_graph[filename])
        for filename in v42_bundle.INPUT_FILENAMES_V4_2
    }
    diagnostic_values = {
        filename: copy.deepcopy(diagnostic_graph[filename])
        for filename in PRIOR_DIAGNOSTIC_FILENAMES_V4_4
    }
    diagnostic_bindings = _artifact_bindings(
        PRIOR_DIAGNOSTIC_FILENAMES_V4_4, diagnostic_values
    )
    selection_v42 = v42_graph[v42_bundle.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2]
    comparison_v42 = v42_graph[
        v42_bundle.COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2
    ]
    nomination = diagnostic_values[
        diagnostic_bundle.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3
    ]
    expanded = core.build_expanded_candidate_selection_v4_4(
        v4_2_selection_spec=selection_v42,
        v4_2_aquant_receipt=v42_graph[
            v42_bundle.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        v4_2_myquant_receipt=v42_graph[
            v42_bundle.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        v4_2_operator_semantics=v42_graph[
            v42_bundle.OPERATOR_SEMANTICS_FILENAME_V4_2
        ],
        v4_2_comparison_catalog_receipt=comparison_v42,
        prior_diagnostic_nomination=nomination,
        diagnostic_artifact_bindings=diagnostic_bindings,
    )
    collision = core.build_definition_identity_collision_audit_v4_4(
        expanded_candidate_selection=expanded,
        v4_2_selection_spec=selection_v42,
        v4_2_aquant_receipt=v42_graph[
            v42_bundle.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        v4_2_myquant_receipt=v42_graph[
            v42_bundle.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
        ],
        v4_2_operator_semantics=v42_graph[
            v42_bundle.OPERATOR_SEMANTICS_FILENAME_V4_2
        ],
        prior_diagnostic_nomination=nomination,
        diagnostic_artifact_bindings=diagnostic_bindings,
        comparison_catalog_receipt=comparison_v42,
    )
    future = core.build_future_source_envelope_v4_4(
        v4_2_predecessor_artifacts=prefixed,
        expanded_candidate_selection=expanded,
        publication_at=publication_at,
    )
    root = core.build_cycle_root_v4_4(
        v4_2_artifact_bindings=_artifact_bindings(
            V4_2_PREDECESSOR_FILENAMES_V4_4, prefixed
        ),
        diagnostic_artifact_bindings=diagnostic_bindings,
        code_binding_set=code_binding_set,
        expanded_candidate_selection=expanded,
        definition_identity_collision_audit=collision,
        future_source_envelope=future,
    )
    states = core.build_preregistration_discovery_cycle_v4_4(
        cycle_root=root,
        future_source_envelope=future,
        expanded_candidate_selection=expanded,
        definition_identity_collision_audit=collision,
        code_binding_set=code_binding_set,
    )
    result: dict[str, dict[str, Any]] = {}
    for filename in V4_2_PREDECESSOR_FILENAMES_V4_4:
        result[filename] = copy.deepcopy(dict(prefixed[filename]))
    result[CODE_BINDING_SET_FILENAME_V4_4] = copy.deepcopy(dict(code_binding_set))
    for filename in PRIOR_DIAGNOSTIC_FILENAMES_V4_4:
        result[filename] = copy.deepcopy(dict(diagnostic_values[filename]))
    result[EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4] = expanded
    result[DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4] = collision
    result[CYCLE_ROOT_FILENAME_V4_4] = root
    result[FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4] = future
    result[PRECOMMITTED_STATE_FILENAME_V4_4] = states[
        PRECOMMITTED_STATE_FILENAME_V4_4
    ]
    result[DISCOVERY_SOURCE_NODE_FILENAME_V4_4] = states[
        DISCOVERY_SOURCE_NODE_FILENAME_V4_4
    ]
    result[DISCOVERY_STATE_FILENAME_V4_4] = states[
        DISCOVERY_STATE_FILENAME_V4_4
    ]
    result[PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4] = states[
        PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4
    ]
    return result


def _validate_raw_input_bindings(
    value: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error("raw_input_bindings must be the exact ordered sequence")
    if len(value) != len(INPUT_FILENAMES_V4_4):
        raise _error("raw_input_bindings exact inventory mismatch")
    normalized: list[dict[str, Any]] = []
    for index, (filename, item) in enumerate(
        zip(INPUT_FILENAMES_V4_4, value, strict=True)
    ):
        row = _exact(
            item,
            {"filename", "byte_sha256", "size_bytes"},
            f"raw input binding[{index}]",
        )
        raw = core.canonical_file_bytes_v4_4(artifacts[filename])
        expected = {
            "filename": filename,
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        if row != expected:
            raise _error(f"raw input binding mismatch: {filename}")
        normalized.append(expected)
    return normalized


def _validate_collected_raw_bytes(
    value: Mapping[str, bytes],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise _error("collected_raw_bytes must be a filename-to-bytes mapping")
    filenames = tuple(value)
    if filenames not in (COLLECTED_RAW_FILENAMES_V4_4, INPUT_FILENAMES_V4_4):
        raise _error("collected_raw_bytes inventory/order mismatch")
    normalized: dict[str, bytes] = {}
    for filename in filenames:
        raw = value[filename]
        if type(raw) is not bytes:
            raise _error(f"collected raw input must be bytes: {filename}")
        canonical = core.canonical_file_bytes_v4_4(artifacts[filename])
        if raw != canonical:
            raise _error(f"collected raw bytes are not exact canonical bytes: {filename}")
        normalized[filename] = bytes(raw)
    return normalized


def raw_input_bindings_v4_4(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if tuple(artifacts) != INPUT_FILENAMES_V4_4:
        raise _error("artifact inventory/order mismatch for raw bindings")
    return [
        {
            "filename": filename,
            "byte_sha256": hashlib.sha256(
                core.canonical_file_bytes_v4_4(artifacts[filename])
            ).hexdigest(),
            "size_bytes": len(core.canonical_file_bytes_v4_4(artifacts[filename])),
        }
        for filename in INPUT_FILENAMES_V4_4
    ]


def build_candidate_preregistration_bundle_artifacts_v4_4(
    *,
    v4_2_predecessor_artifacts: Mapping[str, Mapping[str, Any]],
    prior_diagnostic_artifacts: Mapping[str, Mapping[str, Any]],
    code_binding_set: Mapping[str, Any],
    publication_at: str,
    raw_input_bindings: Sequence[Mapping[str, Any]] | None = None,
    collected_raw_bytes: Mapping[str, bytes] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_v42_prefixed = core.validate_v4_2_predecessor_graph_v4_4(
        v4_2_predecessor_artifacts
    )
    v42_graph = _prefixed_v42(normalized_v42_prefixed)
    diagnostic_graph = core.validate_prior_diagnostic_graph_v4_4(
        prior_diagnostic_artifacts
    )
    code = core.validate_code_binding_set_v4_4(code_binding_set)
    artifacts = _build_from_normalized(
        v42_graph=v42_graph,
        diagnostic_graph=diagnostic_graph,
        code_binding_set=code,
        publication_at=publication_at,
    )
    if tuple(artifacts) != INPUT_FILENAMES_V4_4:
        raise _error("internal v4.4 builder inventory/order mismatch")
    if raw_input_bindings is not None:
        _validate_raw_input_bindings(raw_input_bindings, artifacts)
    if collected_raw_bytes is not None:
        _validate_collected_raw_bytes(collected_raw_bytes, artifacts)
    return artifacts


def validate_candidate_preregistration_bundle_inputs_v4_4(
    values: Mapping[str, Mapping[str, Any]],
    *,
    raw_input_bindings: Sequence[Mapping[str, Any]] | None = None,
    collected_raw_bytes: Mapping[str, bytes] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or tuple(values) != INPUT_FILENAMES_V4_4:
        raise _error("bundle input inventory/order mismatch")
    v42_graph = _normalized_v42_graph(values)
    _fully_validate_v42_future_envelope(v42_graph)
    diagnostic_graph = _diagnostic_graph(values)
    code = core.validate_code_binding_set_v4_4(
        values[CODE_BINDING_SET_FILENAME_V4_4]
    )
    publication_at = values[FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4].get(
        "publication", {}
    ).get("recorded_local_time")
    if type(publication_at) is not str:
        raise _error("future source publication_at is missing")
    rebuilt = _build_from_normalized(
        v42_graph=v42_graph,
        diagnostic_graph=diagnostic_graph,
        code_binding_set=code,
        publication_at=publication_at,
    )
    for filename in INPUT_FILENAMES_V4_4:
        if not _canonical_equal(values[filename], rebuilt[filename]):
            raise _error(f"cross-artifact graph mismatch: {filename}")
    if raw_input_bindings is not None:
        _validate_raw_input_bindings(raw_input_bindings, rebuilt)
    if collected_raw_bytes is not None:
        _validate_collected_raw_bytes(collected_raw_bytes, rebuilt)
    return rebuilt


def _validate_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if filename.startswith(core.V4_2_PREDECESSOR_PREFIX):
        original = filename.removeprefix(core.V4_2_PREDECESSOR_PREFIX)
        if original not in v42_bundle.INPUT_FILENAMES_V4_2:
            raise _error(f"unknown prefixed v4.2 artifact: {filename}")
        try:
            return getattr(v42_bundle, "_validate_artifact")(original, value)
        except Exception as exc:
            raise _error(f"prefixed v4.2 artifact validation failed: {exc}") from exc
    if filename in PRIOR_DIAGNOSTIC_FILENAMES_V4_4:
        try:
            return getattr(diagnostic_bundle, "_validate_artifact")(filename, value)
        except Exception as exc:
            raise _error(f"diagnostic artifact validation failed: {exc}") from exc
    if filename == CODE_BINDING_SET_FILENAME_V4_4:
        return core.validate_code_binding_set_v4_4(value)
    if filename == CYCLE_ROOT_FILENAME_V4_4:
        return core.validate_cycle_root_v4_4(value)
    if filename == PRECOMMITTED_STATE_FILENAME_V4_4:
        return validate_cycle_state_v4_1(value, expected_state=PRECOMMITTED)
    if filename == DISCOVERY_STATE_FILENAME_V4_4:
        return validate_cycle_state_v4_1(value, expected_state=DISCOVERY)
    if filename == READBACK_REPORT_FILENAME_V4_4:
        return validate_candidate_preregistration_readback_v4_4(value)
    if filename in {
        EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4,
        DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4,
        FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4,
        DISCOVERY_SOURCE_NODE_FILENAME_V4_4,
        PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4,
    }:
        return _validate_self(value, filename)
    raise _error(f"unknown v4.4 artifact: {filename}")


def validate_candidate_preregistration_readback_v4_4(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "evidence_contract_version",
            "filename",
            "cycle_id",
            "cycle_root_sha256",
            "publication_evidence_scope",
            "intended_destination",
            "required_commit_primitive",
            "commit_success_claimed",
            "no_clobber_success_claimed",
            "fsync_success_claimed",
            "durability_success_claimed",
            "state_contract",
            "raw_byte_preservation_contract",
            "artifact_bindings",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "v4.4 readback report",
    )
    if (
        payload["schema_version"] != READBACK_REPORT_SCHEMA_VERSION_V4_4
        or payload["protocol_version"] != core.PROTOCOL_VERSION
        or payload["evidence_contract_version"] != core.EVIDENCE_CONTRACT_VERSION
        or payload["filename"] != READBACK_REPORT_FILENAME_V4_4
    ):
        raise _error("v4.4 readback schema/protocol/filename mismatch")
    if payload["intended_destination"] != {
        "root_suffix": list(ROOT_SUFFIX_V4_4),
        "directory_name": payload["cycle_id"],
    }:
        raise _error("v4.4 readback destination mismatch")
    if (
        payload["publication_evidence_scope"] != "PRECOMMIT_INTENT_ONLY"
        or payload["required_commit_primitive"] != "renameatx_np(RENAME_EXCL)"
        or any(
            payload[field] is not False
            for field in (
                "commit_success_claimed",
                "no_clobber_success_claimed",
                "fsync_success_claimed",
                "durability_success_claimed",
            )
        )
    ):
        raise _error("staged readback may not claim publication success")
    if payload["state_contract"] != {
        "precommitted_persisted": True,
        "precommitted_role": "V4_4_INTRA_BUNDLE_LINEAGE_ONLY",
        "discovery_persisted": True,
        "sole_final_current_state": DISCOVERY,
        "v4_2_cross_cycle_predecessor": False,
        "external_pointer_mutation": False,
    }:
        raise _error("v4.4 readback state contract mismatch")
    if payload["raw_byte_preservation_contract"] != {
        "v4_2_original_canonical_bytes_required": True,
        "prior_diagnostic_original_canonical_bytes_required": True,
        "parse_reserialize_is_not_original_byte_proof": True,
        "raw_bytes_embedded_in_json": False,
    }:
        raise _error("v4.4 raw-byte preservation contract mismatch")
    bindings = payload["artifact_bindings"]
    if not isinstance(bindings, list) or [
        row.get("filename") for row in bindings if isinstance(row, Mapping)
    ] != list(INPUT_FILENAMES_V4_4):
        raise _error("readback exact 26-artifact inventory/order mismatch")
    for index, row_value in enumerate(bindings):
        row = _exact(
            row_value,
            {
                "filename",
                "byte_sha256",
                "semantic_sha256",
                "size_bytes",
                "mode",
                "uid",
                "nlink",
            },
            f"readback binding[{index}]",
        )
        _sha256(row["byte_sha256"], f"readback binding[{index}] byte SHA")
        _sha256(
            row["semantic_sha256"],
            f"readback binding[{index}] semantic SHA",
        )
        if (
            type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
            or row["mode"] != 0o600
            or row["uid"] != os.getuid()
            or row["nlink"] != 1
        ):
            raise _error("readback binding must be owner 0600/nlink1 with exact hashes")
    if payload["measurement"] != core.MEASUREMENT_FLAGS:
        raise _error("readback measurement flags must remain not_run")
    if payload["authority"] != core.AUTHORITY_FLAGS:
        raise _error("readback authority flags must remain false")
    if payload["side_effects"] != core.SIDE_EFFECT_FLAGS:
        raise _error("readback side effects must remain false")
    _sha256(payload["cycle_root_sha256"], "readback cycle root SHA")
    _validate_self(payload, "v4.4 readback report")
    return copy.deepcopy(payload)


def _build_readback_report(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(artifacts) != INPUT_FILENAMES_V4_4:
        raise _error("readback builder input inventory/order mismatch")
    if len(artifact_bindings) != len(INPUT_FILENAMES_V4_4):
        raise _error("readback builder binding count mismatch")
    root = core.validate_cycle_root_v4_4(artifacts[CYCLE_ROOT_FILENAME_V4_4])
    if run_id != root["cycle_id"]:
        raise _error("publication directory must equal deterministic cycle_id")
    rows: list[dict[str, Any]] = []
    for filename, item in zip(INPUT_FILENAMES_V4_4, artifact_bindings, strict=True):
        row = _exact(
            item,
            {"filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink"},
            f"private I/O binding {filename}",
        )
        if row["filename"] != filename:
            raise _error("private I/O binding order mismatch")
        rows.append(
            {
                **copy.deepcopy(row),
                "semantic_sha256": _semantic(artifacts[filename], filename),
            }
        )
    return validate_candidate_preregistration_readback_v4_4(
        _seal(
            {
                "schema_version": READBACK_REPORT_SCHEMA_VERSION_V4_4,
                "protocol_version": core.PROTOCOL_VERSION,
                "evidence_contract_version": core.EVIDENCE_CONTRACT_VERSION,
                "filename": READBACK_REPORT_FILENAME_V4_4,
                "cycle_id": root["cycle_id"],
                "cycle_root_sha256": root["cycle_root_sha256"],
                "publication_evidence_scope": "PRECOMMIT_INTENT_ONLY",
                "intended_destination": {
                    "root_suffix": list(ROOT_SUFFIX_V4_4),
                    "directory_name": root["cycle_id"],
                },
                "required_commit_primitive": "renameatx_np(RENAME_EXCL)",
                "commit_success_claimed": False,
                "no_clobber_success_claimed": False,
                "fsync_success_claimed": False,
                "durability_success_claimed": False,
                "state_contract": {
                    "precommitted_persisted": True,
                    "precommitted_role": "V4_4_INTRA_BUNDLE_LINEAGE_ONLY",
                    "discovery_persisted": True,
                    "sole_final_current_state": DISCOVERY,
                    "v4_2_cross_cycle_predecessor": False,
                    "external_pointer_mutation": False,
                },
                "raw_byte_preservation_contract": {
                    "v4_2_original_canonical_bytes_required": True,
                    "prior_diagnostic_original_canonical_bytes_required": True,
                    "parse_reserialize_is_not_original_byte_proof": True,
                    "raw_bytes_embedded_in_json": False,
                },
                "artifact_bindings": rows,
                "measurement": copy.deepcopy(core.MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(core.AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(core.SIDE_EFFECT_FLAGS),
            }
        )
    )


def validate_candidate_preregistration_bundle_artifacts_v4_4(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = (*INPUT_FILENAMES_V4_4, READBACK_REPORT_FILENAME_V4_4)
    if not isinstance(values, Mapping) or tuple(values) != expected:
        raise _error("complete bundle inventory/order mismatch")
    normalized = validate_candidate_preregistration_bundle_inputs_v4_4(
        {filename: values[filename] for filename in INPUT_FILENAMES_V4_4}
    )
    report = validate_candidate_preregistration_readback_v4_4(
        values[READBACK_REPORT_FILENAME_V4_4]
    )
    root = normalized[CYCLE_ROOT_FILENAME_V4_4]
    if (
        report["cycle_id"] != root["cycle_id"]
        or report["cycle_root_sha256"] != root["cycle_root_sha256"]
    ):
        raise _error("readback cycle identity mismatch")
    for filename, binding in zip(
        INPUT_FILENAMES_V4_4, report["artifact_bindings"], strict=True
    ):
        raw = core.canonical_file_bytes_v4_4(normalized[filename])
        if (
            binding["byte_sha256"] != hashlib.sha256(raw).hexdigest()
            or binding["size_bytes"] != len(raw)
            or binding["semantic_sha256"] != _semantic(normalized[filename], filename)
        ):
            raise _error(f"readback artifact binding mismatch: {filename}")
    base_bindings = [
        {
            key: row[key]
            for key in ("filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink")
        }
        for row in report["artifact_bindings"]
    ]
    rebuilt = _build_readback_report(
        run_id=root["cycle_id"],
        artifacts=normalized,
        artifact_bindings=base_bindings,
    )
    if not _canonical_equal(report, rebuilt):
        raise _error("readback report deterministic reconstruction mismatch")
    return {**normalized, READBACK_REPORT_FILENAME_V4_4: report}


def candidate_preregistration_bundle_contract_v4_4() -> PrivateBundleContract:
    return PrivateBundleContract(
        root_suffix=ROOT_SUFFIX_V4_4,
        input_filenames=INPUT_FILENAMES_V4_4,
        readback_report_filename=READBACK_REPORT_FILENAME_V4_4,
        canonicalize=core.canonical_file_bytes_v4_4,
        validate_artifact=_validate_artifact,
        validate_complete=validate_candidate_preregistration_bundle_artifacts_v4_4,
        build_readback_report=_build_readback_report,
    )


def publish_candidate_preregistration_bundle_v4_4(
    *,
    private_root: str | os.PathLike[str],
    artifacts: Mapping[str, Mapping[str, Any]],
    raw_input_bindings: Sequence[Mapping[str, Any]],
    collected_raw_bytes: Mapping[str, bytes],
    revalidate_inputs: Any,
    _test_fault_hook: Any = None,
    _test_race_hook: Any = None,
) -> dict[str, Any]:
    normalized = validate_candidate_preregistration_bundle_inputs_v4_4(
        artifacts,
        raw_input_bindings=raw_input_bindings,
        collected_raw_bytes=collected_raw_bytes,
    )
    if not callable(revalidate_inputs):
        raise _error("revalidate_inputs must be callable")
    root = normalized[CYCLE_ROOT_FILENAME_V4_4]

    def locked_revalidation() -> None:
        validate_candidate_preregistration_bundle_inputs_v4_4(
            normalized,
            raw_input_bindings=raw_input_bindings,
            collected_raw_bytes=collected_raw_bytes,
        )
        revalidate_inputs()

    return publish_private_bundle(
        private_root=private_root,
        run_id=root["cycle_id"],
        artifacts=normalized,
        contract=candidate_preregistration_bundle_contract_v4_4(),
        revalidate_inputs=locked_revalidation,
        _test_fault_hook=_test_fault_hook,
        _test_race_hook=_test_race_hook,
    )


def readback_candidate_preregistration_bundle_files_v4_4(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    return readback_private_bundle(
        bundle_path, contract=candidate_preregistration_bundle_contract_v4_4()
    )


def readback_candidate_preregistration_bundle_v4_4(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    return readback_candidate_preregistration_bundle_files_v4_4(bundle_path)


__all__ = [
    "CODE_BINDING_SET_FILENAME_V4_4",
    "COLLECTED_RAW_FILENAMES_V4_4",
    "CYCLE_ROOT_FILENAME_V4_4",
    "DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_4",
    "DISCOVERY_SOURCE_NODE_FILENAME_V4_4",
    "DISCOVERY_STATE_FILENAME_V4_4",
    "EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4",
    "FUTURE_SOURCE_ENVELOPE_FILENAME_V4_4",
    "FactorGovernanceCandidatePreregistrationBundleV4_4Error",
    "INPUT_FILENAMES_V4_4",
    "PRECOMMITTED_STATE_FILENAME_V4_4",
    "PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_4",
    "PRIOR_DIAGNOSTIC_FILENAMES_V4_4",
    "READBACK_REPORT_FILENAME_V4_4",
    "READBACK_REPORT_SCHEMA_VERSION_V4_4",
    "ROOT_SUFFIX_V4_4",
    "V4_2_PREDECESSOR_FILENAMES_V4_4",
    "build_candidate_preregistration_bundle_artifacts_v4_4",
    "candidate_preregistration_bundle_contract_v4_4",
    "publish_candidate_preregistration_bundle_v4_4",
    "raw_input_bindings_v4_4",
    "readback_candidate_preregistration_bundle_files_v4_4",
    "readback_candidate_preregistration_bundle_v4_4",
    "validate_candidate_preregistration_bundle_artifacts_v4_4",
    "validate_candidate_preregistration_bundle_inputs_v4_4",
    "validate_candidate_preregistration_readback_v4_4",
]
