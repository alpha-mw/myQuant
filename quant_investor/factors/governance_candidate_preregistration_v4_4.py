"""Pure FactorGovernanceProtocol v4.4 five-candidate preregistration.

The governance protocol remains ``v4``.  Evidence contract v4.4 embeds one
already-valid v4.2 four-candidate graph and the complete v4.3 prior diagnostic
bundle, but neither input is a cross-cycle state predecessor.  The fifth idea
is explicitly outcome-informed and every formal measurement, health,
admission, activation, and production authority remains unavailable.

This module is intentionally filesystem-free.  Callers supply parsed artifacts
and exact byte descriptors; the owner-private bundle layer owns publication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_2 as v42_bundle,
)
from quant_investor.factors import governance_candidate_preregistration_v4_2 as v42
from quant_investor.factors import (
    governance_prior_diagnostic_nomination_bundle_v4_3 as diagnostic_bundle,
)
from quant_investor.factors import (
    governance_prior_diagnostic_nomination_v4_3 as diagnostic,
)
from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    build_genesis_cycle_state_v4_1,
    build_next_cycle_state_v4_1,
    byte_sha256 as cycle_state_byte_sha256_v4_1,
    validate_cycle_state_v4_1,
)


PROTOCOL_VERSION = "v4"
EVIDENCE_CONTRACT_VERSION = "v4.4"
SCHEMA_VERSION = "factor-governance-candidate-preregistration.v4.4"
CODE_BINDING_SET_SCHEMA_VERSION = "factor-governance-code-binding-set.v4.4"
EXPANDED_SELECTION_SCHEMA_VERSION = (
    "factor-governance-expanded-candidate-selection.v4.4"
)
DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION = (
    "factor-governance-definition-identity-collision-audit.v4.4"
)
CYCLE_ROOT_SCHEMA_VERSION = "factor-governance-candidate-cycle-root.v4.4"
SOURCE_ENVELOPE_SCHEMA_VERSION = "factor-governance-future-source-envelope.v4.4"
DISCOVERY_SOURCE_NODE_SCHEMA_VERSION = (
    "factor-governance-prereg-discovery-source-node.v4.4"
)
ORCHESTRATION_SCHEMA_VERSION = (
    "factor-governance-prereg-discovery-orchestration.v4.4"
)

FROZEN_PREVIOUS_CUTOFF = "2026-07-19"
PUBLICATION_TIME_AUTHORITY = "LOCAL_UNVERIFIED"
V4_2_PREDECESSOR_PREFIX = "v4_2_predecessor."

V4_2_PREDECESSOR_FILENAMES = tuple(
    V4_2_PREDECESSOR_PREFIX + filename
    for filename in v42_bundle.INPUT_FILENAMES_V4_2
)
PRIOR_DIAGNOSTIC_FILENAMES = (
    diagnostic_bundle.PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3,
    diagnostic_bundle.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3,
    diagnostic_bundle.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
)

# These are the exact accepted owner-private diagnostic files.  They are not
# inferred from a mutable path and are never treated as formal measurements.
EXPECTED_PRIOR_DIAGNOSTIC_BINDINGS = (
    {
        "filename": PRIOR_DIAGNOSTIC_FILENAMES[0],
        "byte_sha256": (
            "3f1ea68884f49c42b9641070e1acf23c2501e50ecb3f9a0c2434c75ca64d2471"
        ),
        "semantic_sha256": (
            "5d3ef2d66e6236f760d4c0ad127f49676e7ab4c260807808bb625f89637ffab8"
        ),
        "size_bytes": 662152,
    },
    {
        "filename": PRIOR_DIAGNOSTIC_FILENAMES[1],
        "byte_sha256": (
            "5a567fb7b462259196c4c920bc78dd7c31d02b2b63bd4c14a05d29f92e25db29"
        ),
        "semantic_sha256": (
            "d08496ec40da1a3a29a4f6d9d1a549f2f736c1694fba4e3b8f1d274050089023"
        ),
        "size_bytes": 46837,
    },
    {
        "filename": PRIOR_DIAGNOSTIC_FILENAMES[2],
        "byte_sha256": (
            "ddff8247913af13d186acb53ff3e72b1197d5c76fc95b4564a409f8dd1575f98"
        ),
        "semantic_sha256": (
            "0a2f2438bcb2ed1f7261d348241394da7b0c2d524f40c576e261bacebe53ff5e"
        ),
        "size_bytes": 1985,
    },
)

EXPECTED_CANDIDATE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "order": 1,
        "name": "alpha_range_position_momentum_20d",
        "definition_identity_sha256": (
            "8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f"
        ),
        "family": "price_momentum",
        "slot": "primitive:price_momentum",
        "initial_weight": 0,
        "selection_origin": "embedded_v4_2_prospective_selection",
    },
    {
        "order": 2,
        "name": "pv_low_overnight_gap_20d",
        "definition_identity_sha256": (
            "a060bd0a52353b218bb963658073e20b1b9bc5cd598c7c4207263c7f45d7dd4e"
        ),
        "family": "overnight_gap",
        "slot": "primitive:overnight_gap",
        "initial_weight": 0,
        "selection_origin": "embedded_v4_2_prospective_selection",
    },
    {
        "order": 3,
        "name": "pv_low_vol_ratio_10_60",
        "definition_identity_sha256": (
            "b8672e8996696c4f820f30cf6c4b97b2641cefe8b6e2ecd72ba1874685f87ac7"
        ),
        "family": "realized_volatility_ratio",
        "slot": "primitive:realized_volatility_ratio",
        "initial_weight": 0,
        "selection_origin": "embedded_v4_2_prospective_selection",
    },
    {
        "order": 4,
        "name": "pv_price_volume_consistency_20d",
        "definition_identity_sha256": (
            "fe70f67577bc2bcd4d7bb4275d2b7aac3f4e2671ffd618cd9400d1f02145a41d"
        ),
        "family": "price_volume_consistency",
        "slot": "primitive:price_volume_consistency",
        "initial_weight": 0,
        "selection_origin": "embedded_v4_2_prospective_selection",
    },
    {
        "order": 5,
        "name": "pv_low_vol_of_vol_20d",
        "definition_identity_sha256": diagnostic.DEFINITION_IDENTITY_SHA256,
        "family": "volatility_of_volatility",
        "slot": "primitive:volatility_of_volatility",
        "initial_weight": 0,
        "selection_origin": "v4_3_outcome_informed_prior_diagnostic_nomination",
    },
)

CODE_BINDING_PATHS_V4_4 = (
    "scripts/build_factor_v4_4_candidate_preregistration.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
    "scripts/build_factor_v4_2_candidate_preregistration.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_bundle_v4_3.py",
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/codex_review/storage.py",
    "quant_investor/market/pit_universe.py",
    "quant_investor/factors/governance_source_v4_1.py",
)

MEASUREMENT_FLAGS = {
    "runtime_equivalence": "not_run",
    "signal_computability": "not_run",
    "measurement": "not_run",
    "statistics": "not_run",
    "family_bh": "not_run",
    "maturity": "not_run",
    "walk_forward": "not_run",
    "cost": "not_run",
    "neutralization": "not_run",
    "stability": "not_run",
    "structural_dedup": "not_run",
    "formal_dedup": "not_run",
    "high_correlation_dedup": "not_run",
    "verified_v4_replay": "not_run",
    "transaction_plan": "not_run",
    "readiness": "PROSPECTIVE_PREREGISTRATION_ONLY",
    "status": "measurement_not_run",
}

AUTHORITY_FLAGS = {
    "healthy_source_receipt": False,
    "healthy_factor_authorized": False,
    "measurement_authorized": False,
    "screening_authorized": False,
    "family_bh_authorized": False,
    "maturity_authorized": False,
    "walk_forward_authorized": False,
    "dedup_authorized": False,
    "replay_authorized": False,
    "candidate_qualified": False,
    "qualification_authorized": False,
    "admission_authorized": False,
    "registry_write_authorized": False,
    "production_proposal_authorized": False,
    "apply_authorized": False,
    "activation_receipt_authorized": False,
    "production_activation_authorized": False,
    "production_candidate_authorized": False,
    "production_new_risk_authorized": False,
}

SIDE_EFFECT_FLAGS = {
    "registry": False,
    "wal": False,
    "budget": False,
    "production_receipt": False,
    "production_pointer": False,
    "proposal": False,
    "apply": False,
    "activation": False,
    "provider": False,
    "network": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
    "transaction": False,
}

SELECTION_CLAIMS = {
    "outcome_informed_selection": True,
    "external_label_independence": False,
    "prior_statistics_nomination_only": True,
    "prior_statistics_inherited_as_formal_evidence": False,
    "authoritative_evidence_route": (
        "independent_post_v4_4_publication_embargo_and_holdout_only"
    ),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_ID_RE = re.compile(r"\d{8}T\d{6}Z")
_DIAGNOSTIC_STATISTIC_KEYS = frozenset(
    {
        "rank_ic",
        "raw_mean_ic",
        "adjusted_mean_ic",
        "ic_std_ddof1",
        "icir",
        "t_statistic",
        "p_value",
        "bonferroni_p",
        "bh_q_value",
        "coverage",
        "coverage_rate",
        "valid_period_count",
        "effective_start",
    }
)


class FactorGovernanceCandidatePreregistrationV4_4Error(ValueError):
    """Raised when v4.4 evidence fails closed."""


FactorGovernanceCandidatePreregistrationV44Error = (
    FactorGovernanceCandidatePreregistrationV4_4Error
)


def _error(message: str) -> FactorGovernanceCandidatePreregistrationV4_4Error:
    return FactorGovernanceCandidatePreregistrationV4_4Error(message)


def canonical_json_bytes_v4_4(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise _error(f"value is not canonical finite JSON: {exc}") from exc


def canonical_file_bytes_v4_4(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_4(value) + b"\n"


def semantic_sha256_v4_4(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_4(value)).hexdigest()


def byte_sha256_v4_4(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_file_bytes_v4_4(value)).hexdigest()


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
    canonical_json_bytes_v4_4(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _date(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    return value


def _snapshot_id(value: Any, *, cutoff: str) -> str:
    if type(value) is not str or _SNAPSHOT_ID_RE.fullmatch(value) is None:
        raise _error("snapshot_id must be canonical YYYYMMDDTHHMMSSZ")
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise _error("snapshot_id must be a real UTC timestamp") from exc
    if parsed.date().isoformat() != cutoff:
        raise _error("snapshot_id calendar date must exactly equal v4.4 cutoff")
    return value


def _publication_at(value: Any, *, cutoff: str) -> str:
    if type(value) is not str:
        raise _error("publication_at must be an offset-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _error("publication_at must be an offset-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _error("publication_at must be offset-aware")
    if parsed.isoformat() != value:
        raise _error("publication_at must be canonical ISO format")
    if parsed.date() < date.fromisoformat(cutoff):
        raise _error("publication_at must not precede the strict cutoff")
    return value


def deterministic_cycle_id_v4_4(*, cutoff: str, snapshot_id: str) -> str:
    normalized_cutoff = _date(cutoff, "cutoff")
    normalized_snapshot = _snapshot_id(snapshot_id, cutoff=normalized_cutoff)
    return (
        f"cn_full_a_v4_4_{normalized_cutoff.replace('-', '')}_"
        f"{normalized_snapshot}"
    )


def _self_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "artifact_semantic_sha256"
    }


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["artifact_semantic_sha256"] = semantic_sha256_v4_4(payload)
    return payload


def _validate_self(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    supplied = _sha256(
        payload.get("artifact_semantic_sha256"),
        f"{label} artifact semantic SHA",
    )
    if supplied != semantic_sha256_v4_4(_self_payload(payload)):
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return payload


def _artifact_semantic(value: Mapping[str, Any], label: str) -> str:
    if "artifact_semantic_sha256" in value:
        return _sha256(value["artifact_semantic_sha256"], f"{label} semantic SHA")
    if "state_semantic_sha256" in value:
        return _sha256(value["state_semantic_sha256"], f"{label} state SHA")
    raise _error(f"{label} has no accepted semantic identity")


def build_artifact_binding_v4_4(
    *, filename: str, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    if type(filename) is not str or not filename:
        raise _error("artifact filename must be non-empty")
    raw = canonical_file_bytes_v4_4(artifact)
    return {
        "filename": filename,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": _artifact_semantic(artifact, filename),
        "size_bytes": len(raw),
    }


def validate_artifact_binding_v4_4(
    value: Mapping[str, Any], *, expected_filename: str | None = None
) -> dict[str, Any]:
    row = _exact(
        value,
        {"filename", "byte_sha256", "semantic_sha256", "size_bytes"},
        "artifact binding",
    )
    if type(row["filename"]) is not str or not row["filename"]:
        raise _error("artifact binding filename must be non-empty")
    if expected_filename is not None and row["filename"] != expected_filename:
        raise _error("artifact binding filename/order mismatch")
    return {
        "filename": row["filename"],
        "byte_sha256": _sha256(row["byte_sha256"], "artifact byte SHA"),
        "semantic_sha256": _sha256(
            row["semantic_sha256"], "artifact semantic SHA"
        ),
        "size_bytes": _positive_int(row["size_bytes"], "artifact size_bytes"),
    }


def _reject_diagnostic_statistics(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(
            key for key in value if type(key) is str and key in _DIAGNOSTIC_STATISTIC_KEYS
        )
        if forbidden:
            raise _error(
                f"{label} may not inherit diagnostic statistics: {','.join(forbidden)}"
            )
        for key, item in value.items():
            _reject_diagnostic_statistics(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_diagnostic_statistics(item, f"{label}[{index}]")


def validate_v4_2_predecessor_graph_v4_4(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the exact embedded v4.2 graph without creating a state edge."""

    if not isinstance(values, Mapping) or tuple(values) != V4_2_PREDECESSOR_FILENAMES:
        raise _error("embedded v4.2 predecessor inventory/order mismatch")
    unprefixed = {
        filename: values[V4_2_PREDECESSOR_PREFIX + filename]
        for filename in v42_bundle.INPUT_FILENAMES_V4_2
    }
    try:
        normalized = v42_bundle.validate_candidate_preregistration_bundle_inputs_v4_2(
            unprefixed
        )
    except Exception as exc:
        raise _error(f"embedded v4.2 graph validation failed: {exc}") from exc
    return {
        V4_2_PREDECESSOR_PREFIX + filename: copy.deepcopy(normalized[filename])
        for filename in v42_bundle.INPUT_FILENAMES_V4_2
    }


def validate_prior_diagnostic_graph_v4_4(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or tuple(values) != PRIOR_DIAGNOSTIC_FILENAMES:
        raise _error("prior diagnostic inventory/order mismatch")
    try:
        normalized = (
            diagnostic_bundle.validate_prior_diagnostic_nomination_bundle_artifacts_v4_3(
                values
            )
        )
    except Exception as exc:
        raise _error(f"prior diagnostic graph validation failed: {exc}") from exc
    return {
        filename: copy.deepcopy(normalized[filename])
        for filename in PRIOR_DIAGNOSTIC_FILENAMES
    }


def validate_code_binding_set_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "evidence_contract_version",
            "path_count",
            "ordered_bindings",
            "artifact_semantic_sha256",
        },
        "v4.4 code binding set",
    )
    if (
        payload["schema_version"] != CODE_BINDING_SET_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("code binding set schema/protocol mismatch")
    rows = payload["ordered_bindings"]
    if not isinstance(rows, list) or len(rows) != len(CODE_BINDING_PATHS_V4_4):
        raise _error("code binding set exact inventory mismatch")
    normalized: list[dict[str, Any]] = []
    for index, (item, expected_path) in enumerate(
        zip(rows, CODE_BINDING_PATHS_V4_4, strict=True), start=1
    ):
        row = _exact(
            item,
            {"order", "relative_path", "byte_sha256", "size_bytes"},
            f"code binding[{index}]",
        )
        if row["order"] != index or row["relative_path"] != expected_path:
            raise _error("code binding path/order mismatch")
        normalized.append(
            {
                "order": index,
                "relative_path": expected_path,
                "byte_sha256": _sha256(row["byte_sha256"], "code byte SHA"),
                "size_bytes": _positive_int(row["size_bytes"], "code size"),
            }
        )
    if payload["path_count"] != len(normalized):
        raise _error("code binding path_count mismatch")
    _validate_self(payload, "code binding set")
    return copy.deepcopy(payload)


def build_code_binding_set_v4_4(
    *, ordered_bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(ordered_bindings, Sequence) or isinstance(
        ordered_bindings, (str, bytes, bytearray)
    ):
        raise _error("ordered_bindings must be a sequence")
    return validate_code_binding_set_v4_4(
        _seal(
            {
                "schema_version": CODE_BINDING_SET_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                "path_count": len(ordered_bindings),
                "ordered_bindings": [copy.deepcopy(dict(row)) for row in ordered_bindings],
            }
        )
    )


def _validate_diagnostic_bindings(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(PRIOR_DIAGNOSTIC_FILENAMES):
        raise _error("diagnostic artifact binding inventory mismatch")
    normalized = [
        validate_artifact_binding_v4_4(row, expected_filename=filename)
        for row, filename in zip(value, PRIOR_DIAGNOSTIC_FILENAMES, strict=True)
    ]
    if normalized != [copy.deepcopy(row) for row in EXPECTED_PRIOR_DIAGNOSTIC_BINDINGS]:
        raise _error("diagnostic artifact bindings differ from accepted v4.3 bytes")
    return normalized


def _validate_selection_sources(
    *,
    v4_2_selection_spec: Mapping[str, Any],
    v4_2_aquant_receipt: Mapping[str, Any],
    v4_2_myquant_receipt: Mapping[str, Any],
    v4_2_operator_semantics: Mapping[str, Any],
    v4_2_comparison_catalog_receipt: Mapping[str, Any],
    prior_diagnostic_nomination: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        selection = v42.validate_selection_spec_v4_2(
            v4_2_selection_spec,
            aquant_receipt=v4_2_aquant_receipt,
            myquant_receipt=v4_2_myquant_receipt,
            operator_semantics=v4_2_operator_semantics,
            comparison_catalog_receipt=v4_2_comparison_catalog_receipt,
        )
    except Exception as exc:
        raise _error(f"embedded v4.2 selection validation failed: {exc}") from exc
    rows = selection.get("candidates")
    if not isinstance(rows, list) or len(rows) != 4:
        raise _error("v4.2 selection must contain the exact four candidates")
    for expected, row in zip(EXPECTED_CANDIDATE_ROWS[:4], rows, strict=True):
        if (
            row.get("order") != expected["order"]
            or row.get("name") != expected["name"]
            or row.get("definition_identity_sha256")
            != expected["definition_identity_sha256"]
            or row.get("family") != expected["family"]
            or row.get("initial_weight") != 0
        ):
            raise _error("first four candidates differ from embedded v4.2 selection")
    try:
        nomination = diagnostic.validate_prior_diagnostic_nomination_v4_3(
            prior_diagnostic_nomination
        )
    except Exception as exc:
        raise _error(f"prior diagnostic nomination validation failed: {exc}") from exc
    winner = nomination["winner_candidate"]
    expected_fifth = EXPECTED_CANDIDATE_ROWS[4]
    if (
        winner.get("name") != expected_fifth["name"]
        or nomination.get("definition_identity_sha256")
        != expected_fifth["definition_identity_sha256"]
        or winner.get("family") != expected_fifth["family"]
        or winner.get("slot") != expected_fifth["slot"]
        or winner.get("initial_weight") != 0
        or winner.get("outcome_informed_selection") is not True
        or winner.get("external_label_independence") is not False
        or winner.get("prior_statistics_nomination_only") is not True
    ):
        raise _error("fifth candidate differs from the diagnostic winner identity")
    return selection, nomination


def validate_expanded_candidate_selection_v4_4(
    value: Mapping[str, Any],
    *,
    v4_2_selection_spec: Mapping[str, Any],
    v4_2_aquant_receipt: Mapping[str, Any],
    v4_2_myquant_receipt: Mapping[str, Any],
    v4_2_operator_semantics: Mapping[str, Any],
    v4_2_comparison_catalog_receipt: Mapping[str, Any],
    prior_diagnostic_nomination: Mapping[str, Any],
    diagnostic_artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _selection, nomination = _validate_selection_sources(
        v4_2_selection_spec=v4_2_selection_spec,
        v4_2_aquant_receipt=v4_2_aquant_receipt,
        v4_2_myquant_receipt=v4_2_myquant_receipt,
        v4_2_operator_semantics=v4_2_operator_semantics,
        v4_2_comparison_catalog_receipt=v4_2_comparison_catalog_receipt,
        prior_diagnostic_nomination=prior_diagnostic_nomination,
    )
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "evidence_contract_version",
            "preregistered_candidate_count",
            "candidates",
            "embedded_v4_2_selection_semantic_sha256",
            "prior_diagnostic_provenance",
            "selection_claims",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "expanded candidate selection",
    )
    _reject_diagnostic_statistics(payload, "expanded candidate selection")
    if (
        payload["schema_version"] != EXPANDED_SELECTION_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
        or payload["preregistered_candidate_count"] != 5
        or payload["candidates"] != list(EXPECTED_CANDIDATE_ROWS)
    ):
        raise _error("expanded candidate selection fixed identity/oracle mismatch")
    if payload["embedded_v4_2_selection_semantic_sha256"] != _artifact_semantic(
        v4_2_selection_spec, "v4.2 selection"
    ):
        raise _error("expanded selection v4.2 semantic binding mismatch")
    expected_bindings = _validate_diagnostic_bindings(
        [copy.deepcopy(dict(row)) for row in diagnostic_artifact_bindings]
    )
    expected_provenance = {
        "run_id": diagnostic.RUN_ID,
        "purpose": diagnostic.PURPOSE,
        "artifact_bindings": expected_bindings,
        "winner_name": EXPECTED_CANDIDATE_ROWS[4]["name"],
        "winner_definition_identity_sha256": diagnostic.DEFINITION_IDENTITY_SHA256,
        "winner_source": nomination["winner"]["source_name"],
        "provenance_role": "NOMINATION_ONLY_NOT_FORMAL_EVIDENCE",
    }
    if payload["prior_diagnostic_provenance"] != expected_provenance:
        raise _error("expanded selection diagnostic provenance mismatch")
    if payload["selection_claims"] != SELECTION_CLAIMS:
        raise _error("expanded selection anti-laundering claims mismatch")
    if payload["measurement"] != MEASUREMENT_FLAGS:
        raise _error("expanded selection measurement flags must remain not_run")
    if payload["authority"] != AUTHORITY_FLAGS:
        raise _error("expanded selection authority flags must remain false")
    if payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("expanded selection side effects must remain false")
    names = [row["name"] for row in payload["candidates"]]
    identities = [row["definition_identity_sha256"] for row in payload["candidates"]]
    families = [row["family"] for row in payload["candidates"]]
    slots = [row["slot"] for row in payload["candidates"]]
    if any(len(set(items)) != 5 for items in (names, identities, families, slots)):
        raise _error("candidate name/identity/family/slot must each be five-unique")
    if any(row["initial_weight"] != 0 for row in payload["candidates"]):
        raise _error("all five initial weights must remain zero")
    _validate_self(payload, "expanded candidate selection")
    return copy.deepcopy(payload)


def build_expanded_candidate_selection_v4_4(
    *,
    v4_2_selection_spec: Mapping[str, Any],
    v4_2_aquant_receipt: Mapping[str, Any],
    v4_2_myquant_receipt: Mapping[str, Any],
    v4_2_operator_semantics: Mapping[str, Any],
    v4_2_comparison_catalog_receipt: Mapping[str, Any],
    prior_diagnostic_nomination: Mapping[str, Any],
    diagnostic_artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _selection, nomination = _validate_selection_sources(
        v4_2_selection_spec=v4_2_selection_spec,
        v4_2_aquant_receipt=v4_2_aquant_receipt,
        v4_2_myquant_receipt=v4_2_myquant_receipt,
        v4_2_operator_semantics=v4_2_operator_semantics,
        v4_2_comparison_catalog_receipt=v4_2_comparison_catalog_receipt,
        prior_diagnostic_nomination=prior_diagnostic_nomination,
    )
    bindings = _validate_diagnostic_bindings(
        [copy.deepcopy(dict(row)) for row in diagnostic_artifact_bindings]
    )
    return validate_expanded_candidate_selection_v4_4(
        _seal(
            {
                "schema_version": EXPANDED_SELECTION_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                "preregistered_candidate_count": 5,
                "candidates": list(copy.deepcopy(EXPECTED_CANDIDATE_ROWS)),
                "embedded_v4_2_selection_semantic_sha256": _artifact_semantic(
                    v4_2_selection_spec, "v4.2 selection"
                ),
                "prior_diagnostic_provenance": {
                    "run_id": diagnostic.RUN_ID,
                    "purpose": diagnostic.PURPOSE,
                    "artifact_bindings": bindings,
                    "winner_name": EXPECTED_CANDIDATE_ROWS[4]["name"],
                    "winner_definition_identity_sha256": (
                        diagnostic.DEFINITION_IDENTITY_SHA256
                    ),
                    "winner_source": nomination["winner"]["source_name"],
                    "provenance_role": "NOMINATION_ONLY_NOT_FORMAL_EVIDENCE",
                },
                "selection_claims": copy.deepcopy(SELECTION_CLAIMS),
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        v4_2_selection_spec=v4_2_selection_spec,
        v4_2_aquant_receipt=v4_2_aquant_receipt,
        v4_2_myquant_receipt=v4_2_myquant_receipt,
        v4_2_operator_semantics=v4_2_operator_semantics,
        v4_2_comparison_catalog_receipt=v4_2_comparison_catalog_receipt,
        prior_diagnostic_nomination=prior_diagnostic_nomination,
        diagnostic_artifact_bindings=bindings,
    )


def _comparison_identity_inventory(
    comparison_catalog_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        comparison = v42.validate_comparison_catalog_receipt_v4_2(
            comparison_catalog_receipt
        )
    except Exception as exc:
        raise _error(f"v4.2 comparison catalog validation failed: {exc}") from exc
    inventory = {
        row["name"]: row["definition_identity_sha256"]
        for row in comparison["definition_identity_inventory"]
    }
    return comparison, inventory


def validate_definition_identity_collision_audit_v4_4(
    value: Mapping[str, Any],
    *,
    expanded_candidate_selection: Mapping[str, Any],
    v4_2_selection_spec: Mapping[str, Any],
    v4_2_aquant_receipt: Mapping[str, Any],
    v4_2_myquant_receipt: Mapping[str, Any],
    v4_2_operator_semantics: Mapping[str, Any],
    prior_diagnostic_nomination: Mapping[str, Any],
    diagnostic_artifact_bindings: Sequence[Mapping[str, Any]],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    selection = validate_expanded_candidate_selection_v4_4(
        expanded_candidate_selection,
        v4_2_selection_spec=v4_2_selection_spec,
        v4_2_aquant_receipt=v4_2_aquant_receipt,
        v4_2_myquant_receipt=v4_2_myquant_receipt,
        v4_2_operator_semantics=v4_2_operator_semantics,
        v4_2_comparison_catalog_receipt=comparison_catalog_receipt,
        prior_diagnostic_nomination=prior_diagnostic_nomination,
        diagnostic_artifact_bindings=diagnostic_artifact_bindings,
    )
    comparison, inventory = _comparison_identity_inventory(
        comparison_catalog_receipt
    )
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "expanded_selection_semantic_sha256",
            "comparison_catalog_semantic_sha256",
            "method",
            "candidate_uniqueness",
            "selected_vs_comparison",
            "declared_slot_collision",
            "definition_identity_collision_detected",
            "structural_dedup",
            "formal_dedup",
            "high_correlation_dedup",
            "authority",
            "artifact_semantic_sha256",
        },
        "v4.4 definition identity collision audit",
    )
    if (
        payload["schema_version"]
        != DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["expanded_selection_semantic_sha256"]
        != selection["artifact_semantic_sha256"]
        or payload["comparison_catalog_semantic_sha256"]
        != comparison["artifact_semantic_sha256"]
        or payload["method"] != "exact_name_identity_family_slot_uniqueness.v4.4"
    ):
        raise _error("v4.4 collision audit identity/binding mismatch")
    rows = selection["candidates"]
    expected_uniqueness = {
        key: {
            "count": 5,
            "unique_count": 5,
            "values": [row[field] for row in rows],
        }
        for key, field in (
            ("names", "name"),
            ("definition_identities", "definition_identity_sha256"),
            ("families", "family"),
            ("slots", "slot"),
        )
    }
    selected_names = {row["name"] for row in rows}
    selected_identities = {row["definition_identity_sha256"] for row in rows}
    collisions = [
        {
            "selected_name": row["name"],
            "comparison_name": comparison_name,
            "reason": "name" if row["name"] == comparison_name else "identity",
        }
        for row in rows
        for comparison_name, comparison_identity in inventory.items()
        if row["name"] == comparison_name
        or row["definition_identity_sha256"] == comparison_identity
    ]
    if collisions or selected_names.intersection(inventory) or selected_identities.intersection(
        inventory.values()
    ):
        raise _error("selected candidates collide with comparison catalog")
    if payload["candidate_uniqueness"] != expected_uniqueness:
        raise _error("five-dimensional uniqueness audit mismatch")
    if payload["selected_vs_comparison"] != {
        "catalog_id": comparison["catalog_id"],
        "candidate_count": comparison["candidate_count"],
        "checked": True,
        "collisions": [],
    }:
        raise _error("comparison collision audit mismatch")
    if (
        payload["declared_slot_collision"] is not False
        or payload["definition_identity_collision_detected"] is not False
        or payload["structural_dedup"] != "not_run"
        or payload["formal_dedup"] != "not_run"
        or payload["high_correlation_dedup"] != "not_run"
    ):
        raise _error("collision audit may not launder formal dedup evidence")
    if payload["authority"] != AUTHORITY_FLAGS:
        raise _error("collision audit authority must remain false")
    _validate_self(payload, "definition identity collision audit")
    return copy.deepcopy(payload)


def build_definition_identity_collision_audit_v4_4(
    *,
    expanded_candidate_selection: Mapping[str, Any],
    v4_2_selection_spec: Mapping[str, Any],
    v4_2_aquant_receipt: Mapping[str, Any],
    v4_2_myquant_receipt: Mapping[str, Any],
    v4_2_operator_semantics: Mapping[str, Any],
    prior_diagnostic_nomination: Mapping[str, Any],
    diagnostic_artifact_bindings: Sequence[Mapping[str, Any]],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    selection = validate_expanded_candidate_selection_v4_4(
        expanded_candidate_selection,
        v4_2_selection_spec=v4_2_selection_spec,
        v4_2_aquant_receipt=v4_2_aquant_receipt,
        v4_2_myquant_receipt=v4_2_myquant_receipt,
        v4_2_operator_semantics=v4_2_operator_semantics,
        v4_2_comparison_catalog_receipt=comparison_catalog_receipt,
        prior_diagnostic_nomination=prior_diagnostic_nomination,
        diagnostic_artifact_bindings=diagnostic_artifact_bindings,
    )
    comparison, inventory = _comparison_identity_inventory(
        comparison_catalog_receipt
    )
    rows = selection["candidates"]
    if any(
        row["name"] in inventory
        or row["definition_identity_sha256"] in set(inventory.values())
        for row in rows
    ):
        raise _error("selected candidates collide with comparison catalog")
    uniqueness = {
        key: {
            "count": 5,
            "unique_count": 5,
            "values": [row[field] for row in rows],
        }
        for key, field in (
            ("names", "name"),
            ("definition_identities", "definition_identity_sha256"),
            ("families", "family"),
            ("slots", "slot"),
        )
    }
    return validate_definition_identity_collision_audit_v4_4(
        _seal(
            {
                "schema_version": (
                    DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
                ),
                "protocol_version": PROTOCOL_VERSION,
                "expanded_selection_semantic_sha256": selection[
                    "artifact_semantic_sha256"
                ],
                "comparison_catalog_semantic_sha256": comparison[
                    "artifact_semantic_sha256"
                ],
                "method": "exact_name_identity_family_slot_uniqueness.v4.4",
                "candidate_uniqueness": uniqueness,
                "selected_vs_comparison": {
                    "catalog_id": comparison["catalog_id"],
                    "candidate_count": comparison["candidate_count"],
                    "checked": True,
                    "collisions": [],
                },
                "declared_slot_collision": False,
                "definition_identity_collision_detected": False,
                "structural_dedup": "not_run",
                "formal_dedup": "not_run",
                "high_correlation_dedup": "not_run",
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
            }
        ),
        expanded_candidate_selection=selection,
        v4_2_selection_spec=v4_2_selection_spec,
        v4_2_aquant_receipt=v4_2_aquant_receipt,
        v4_2_myquant_receipt=v4_2_myquant_receipt,
        v4_2_operator_semantics=v4_2_operator_semantics,
        prior_diagnostic_nomination=prior_diagnostic_nomination,
        diagnostic_artifact_bindings=diagnostic_artifact_bindings,
        comparison_catalog_receipt=comparison,
    )


def _source_summary(
    *,
    v4_2_predecessor_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_prefixed = validate_v4_2_predecessor_graph_v4_4(
        v4_2_predecessor_artifacts
    )
    graph = {
        filename: normalized_prefixed[V4_2_PREDECESSOR_PREFIX + filename]
        for filename in v42_bundle.INPUT_FILENAMES_V4_2
    }
    strict_source_binding = graph[
        v42_bundle.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    ]
    v4_2_future_source_envelope = graph[
        v42_bundle.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2
    ]
    try:
        source = v42_bundle.validate_strict_full_a_source_binding_v4_2(
            strict_source_binding
        )
    except Exception as exc:
        raise _error(f"v4.2 strict source validation failed: {exc}") from exc
    try:
        envelope = v42.validate_future_source_envelope_v4_2(
            v4_2_future_source_envelope,
            selection_spec=graph[
                v42_bundle.CANDIDATE_SELECTION_SPEC_FILENAME_V4_2
            ],
            aquant_receipt=graph[
                v42_bundle.AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2
            ],
            myquant_receipt=graph[
                v42_bundle.MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2
            ],
            operator_semantics=graph[
                v42_bundle.OPERATOR_SEMANTICS_FILENAME_V4_2
            ],
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
        raise _error(
            f"embedded v4.2 future source envelope validation failed: {exc}"
        ) from exc
    cutoff = _date(source["cutoff"], "source cutoff")
    if date.fromisoformat(cutoff) <= date.fromisoformat(FROZEN_PREVIOUS_CUTOFF):
        raise _error("v4.4 cutoff must be strictly later than 2026-07-19")
    snapshot = _snapshot_id(source["snapshot_id"], cutoff=cutoff)
    if (
        envelope.get("cutoff") != cutoff
        or envelope.get("snapshot_id") != snapshot
        or envelope.get("snapshot_date") != cutoff
        or envelope.get("latest_trade_date") != cutoff
        or envelope.get("latest_complete_trade_date") != cutoff
        or envelope.get("analysis_start") != source["analysis_start"]
        or envelope.get("full_a_scope_sha256") != source["full_a_scope_sha256"]
        or envelope.get("full_a_scope_count") != source["expected_scope_count"]
        or envelope.get("serving_inventory_count")
        != source["serving_inventory_count"]
        or envelope.get("strict_source_binding_semantic_sha256")
        != source["artifact_semantic_sha256"]
    ):
        raise _error("embedded v4.2 strict source/envelope crosslink mismatch")
    if envelope.get("artifact_semantic_sha256") != semantic_sha256_v4_4(
        {
            key: item
            for key, item in envelope.items()
            if key != "artifact_semantic_sha256"
        }
    ):
        raise _error("embedded v4.2 future source envelope self SHA mismatch")
    table = source["backend_binding"]["table"]
    serving = source["backend_binding"]["eligibility_boundary"][
        "serving_inventory"
    ]
    return {
        "analysis_start": source["analysis_start"],
        "cutoff": cutoff,
        "snapshot_id": snapshot,
        "snapshot_date": cutoff,
        "latest_trade_date": cutoff,
        "latest_complete_trade_date": cutoff,
        "market": "CN",
        "universe": "full_a",
        "storage_mode": "strict_parquet",
        "full_a_scope_count": source["expected_scope_count"],
        "full_a_scope_sha256": source["full_a_scope_sha256"],
        "table_inventory_sha256": table["inventory_sha256"],
        "table_regular_file_count": table["regular_file_count"],
        "table_parquet_file_count": table["parquet_file_count"],
        "serving_inventory_count": source["serving_inventory_count"],
        "serving_inventory_descriptor_semantic_sha256": semantic_sha256_v4_4(
            serving
        ),
        "strict_source_binding_semantic_sha256": source[
            "artifact_semantic_sha256"
        ],
        "embedded_v4_2_future_envelope_semantic_sha256": envelope[
            "artifact_semantic_sha256"
        ],
    }


def _validate_source_summary_v4_4(
    value: Mapping[str, Any],
    *,
    cutoff: str,
    snapshot_id: str,
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "analysis_start",
            "cutoff",
            "snapshot_id",
            "snapshot_date",
            "latest_trade_date",
            "latest_complete_trade_date",
            "market",
            "universe",
            "storage_mode",
            "full_a_scope_count",
            "full_a_scope_sha256",
            "table_inventory_sha256",
            "table_regular_file_count",
            "table_parquet_file_count",
            "serving_inventory_count",
            "serving_inventory_descriptor_semantic_sha256",
            "strict_source_binding_semantic_sha256",
            "embedded_v4_2_future_envelope_semantic_sha256",
        },
        "v4.4 source summary",
    )
    analysis_start = _date(payload["analysis_start"], "source analysis_start")
    summary_cutoff = _date(payload["cutoff"], "source cutoff")
    summary_snapshot = _snapshot_id(payload["snapshot_id"], cutoff=summary_cutoff)
    if (
        summary_cutoff != cutoff
        or summary_snapshot != snapshot_id
        or payload["snapshot_date"] != cutoff
        or payload["latest_trade_date"] != cutoff
        or payload["latest_complete_trade_date"] != cutoff
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
        or payload["storage_mode"] != "strict_parquet"
    ):
        raise _error("v4.4 source summary root crosslink mismatch")
    for field in (
        "snapshot_date",
        "latest_trade_date",
        "latest_complete_trade_date",
    ):
        _date(payload[field], f"source {field}")
    if date.fromisoformat(analysis_start) > date.fromisoformat(cutoff):
        raise _error("source analysis_start must not follow cutoff")
    full_a_scope_count = _positive_int(
        payload["full_a_scope_count"], "source full_a_scope_count"
    )
    table_regular_file_count = _positive_int(
        payload["table_regular_file_count"], "source table_regular_file_count"
    )
    table_parquet_file_count = _positive_int(
        payload["table_parquet_file_count"], "source table_parquet_file_count"
    )
    serving_inventory_count = _positive_int(
        payload["serving_inventory_count"], "source serving_inventory_count"
    )
    if table_parquet_file_count > table_regular_file_count:
        raise _error("source Parquet file count exceeds regular file count")
    if serving_inventory_count < full_a_scope_count:
        raise _error("source serving inventory is smaller than full-A scope")
    for field in (
        "full_a_scope_sha256",
        "table_inventory_sha256",
        "serving_inventory_descriptor_semantic_sha256",
        "strict_source_binding_semantic_sha256",
        "embedded_v4_2_future_envelope_semantic_sha256",
    ):
        _sha256(payload[field], f"source {field}")
    return copy.deepcopy(payload)


def validate_future_source_envelope_v4_4(
    value: Mapping[str, Any],
    *,
    v4_2_predecessor_artifacts: Mapping[str, Mapping[str, Any]],
    expanded_candidate_selection: Mapping[str, Any],
    publication_at: str,
) -> dict[str, Any]:
    source = _source_summary(
        v4_2_predecessor_artifacts=v4_2_predecessor_artifacts,
    )
    normalized_publication = _publication_at(publication_at, cutoff=source["cutoff"])
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "evidence_contract_version",
            "cycle_id",
            "source_summary",
            "expanded_selection_semantic_sha256",
            "publication",
            "future_measurement_policy",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "v4.4 future source envelope",
    )
    cycle_id = deterministic_cycle_id_v4_4(
        cutoff=source["cutoff"], snapshot_id=source["snapshot_id"]
    )
    if (
        payload["schema_version"] != SOURCE_ENVELOPE_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
        or payload["cycle_id"] != cycle_id
        or payload["source_summary"] != source
        or payload["expanded_selection_semantic_sha256"]
        != _artifact_semantic(expanded_candidate_selection, "expanded selection")
    ):
        raise _error("v4.4 future source fixed identity/binding mismatch")
    if payload["publication"] != {
        "recorded_local_time": normalized_publication,
        "publication_time_authority": PUBLICATION_TIME_AUTHORITY,
        "publication_proven": False,
        "separate_post_commit_proof_required": True,
    }:
        raise _error("v4.4 publication authority contract mismatch")
    if payload["future_measurement_policy"] != {
        "prior_diagnostic_window_inherited": False,
        "prior_statistics_inherited_as_formal_evidence": False,
        "measurement_anchor": "SEPARATELY_PROVEN_V4_4_PUBLICATION",
        "publication_session_in_measurement_sample": False,
        "embargo_open_sessions": 30,
        "first_eligible_measurement_session": 31,
        "minimum_post_embargo_open_sessions": 240,
        "minimum_distinct_closed_month_ends": 12,
        "measurement_authorized": False,
    }:
        raise _error("v4.4 future measurement policy mismatch")
    if payload["measurement"] != MEASUREMENT_FLAGS:
        raise _error("v4.4 future measurement flags must remain not_run")
    if payload["authority"] != AUTHORITY_FLAGS:
        raise _error("v4.4 future source authority must remain false")
    if payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("v4.4 future source side effects must remain false")
    _validate_self(payload, "future source envelope")
    return copy.deepcopy(payload)


def build_future_source_envelope_v4_4(
    *,
    v4_2_predecessor_artifacts: Mapping[str, Mapping[str, Any]],
    expanded_candidate_selection: Mapping[str, Any],
    publication_at: str,
) -> dict[str, Any]:
    source = _source_summary(
        v4_2_predecessor_artifacts=v4_2_predecessor_artifacts,
    )
    normalized_publication = _publication_at(publication_at, cutoff=source["cutoff"])
    return validate_future_source_envelope_v4_4(
        _seal(
            {
                "schema_version": SOURCE_ENVELOPE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                "cycle_id": deterministic_cycle_id_v4_4(
                    cutoff=source["cutoff"], snapshot_id=source["snapshot_id"]
                ),
                "source_summary": source,
                "expanded_selection_semantic_sha256": _artifact_semantic(
                    expanded_candidate_selection, "expanded selection"
                ),
                "publication": {
                    "recorded_local_time": normalized_publication,
                    "publication_time_authority": PUBLICATION_TIME_AUTHORITY,
                    "publication_proven": False,
                    "separate_post_commit_proof_required": True,
                },
                "future_measurement_policy": {
                    "prior_diagnostic_window_inherited": False,
                    "prior_statistics_inherited_as_formal_evidence": False,
                    "measurement_anchor": "SEPARATELY_PROVEN_V4_4_PUBLICATION",
                    "publication_session_in_measurement_sample": False,
                    "embargo_open_sessions": 30,
                    "first_eligible_measurement_session": 31,
                    "minimum_post_embargo_open_sessions": 240,
                    "minimum_distinct_closed_month_ends": 12,
                    "measurement_authorized": False,
                },
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        v4_2_predecessor_artifacts=v4_2_predecessor_artifacts,
        expanded_candidate_selection=expanded_candidate_selection,
        publication_at=normalized_publication,
    )


def validate_cycle_root_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "evidence_contract_version",
            "market",
            "universe",
            "cutoff",
            "snapshot_id",
            "cycle_id",
            "embedded_v4_2_evidence_graph",
            "prior_diagnostic_evidence",
            "ordered_current_cycle_root_bindings",
            "source_summary",
            "lineage_contract",
            "cycle_root_sha256",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "v4.4 cycle root",
    )
    cutoff = _date(payload["cutoff"], "cycle root cutoff")
    if date.fromisoformat(cutoff) <= date.fromisoformat(FROZEN_PREVIOUS_CUTOFF):
        raise _error("v4.4 cutoff must be strictly later than 2026-07-19")
    snapshot = _snapshot_id(payload["snapshot_id"], cutoff=cutoff)
    _validate_source_summary_v4_4(
        payload["source_summary"], cutoff=cutoff, snapshot_id=snapshot
    )
    cycle_id = deterministic_cycle_id_v4_4(cutoff=cutoff, snapshot_id=snapshot)
    if (
        payload["schema_version"] != CYCLE_ROOT_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
        or payload["cycle_id"] != cycle_id
    ):
        raise _error("v4.4 cycle root fixed identity mismatch")
    embedded = _exact(
        payload["embedded_v4_2_evidence_graph"],
        {"role", "artifact_count", "artifact_bindings", "cross_cycle_state_edge"},
        "embedded v4.2 evidence graph",
    )
    if (
        embedded["role"] != "EMBEDDED_SEALED_EVIDENCE_GRAPH"
        or embedded["artifact_count"] != 14
        or embedded["cross_cycle_state_edge"] is not False
        or not isinstance(embedded["artifact_bindings"], list)
        or [row.get("filename") for row in embedded["artifact_bindings"]]
        != list(V4_2_PREDECESSOR_FILENAMES)
    ):
        raise _error("embedded v4.2 lineage contract mismatch")
    for row, filename in zip(
        embedded["artifact_bindings"], V4_2_PREDECESSOR_FILENAMES, strict=True
    ):
        validate_artifact_binding_v4_4(row, expected_filename=filename)
    prior = _exact(
        payload["prior_diagnostic_evidence"],
        {
            "role",
            "artifact_count",
            "artifact_bindings",
            "statistics_inherited_as_formal_evidence",
        },
        "prior diagnostic evidence",
    )
    if (
        prior["role"] != "OUTCOME_INFORMED_NOMINATION_PROVENANCE_ONLY"
        or prior["artifact_count"] != 3
        or prior["statistics_inherited_as_formal_evidence"] is not False
    ):
        raise _error("prior diagnostic lineage contract mismatch")
    _validate_diagnostic_bindings(prior["artifact_bindings"])
    current_names = (
        "code_binding_set.v4_4.json",
        "expanded_candidate_selection.v4_4.json",
        "definition_identity_collision_audit.v4_4.json",
        "future_source_envelope.v4_4.json",
    )
    current = payload["ordered_current_cycle_root_bindings"]
    if not isinstance(current, list) or len(current) != len(current_names):
        raise _error("cycle root current binding inventory mismatch")
    for row, filename in zip(current, current_names, strict=True):
        validate_artifact_binding_v4_4(row, expected_filename=filename)
    if payload["lineage_contract"] != {
        "genesis_is_v4_4_local": True,
        "precommitted_predecessor_kind": "genesis",
        "v4_2_cycle_state_used_as_predecessor": False,
        "v4_3_diagnostic_cycle_state_used_as_predecessor": False,
        "allowed_state_transition": "PRECOMMITTED->DISCOVERY",
    }:
        raise _error("cycle root independent state-lineage contract mismatch")
    base = {
        key: copy.deepcopy(item)
        for key, item in payload.items()
        if key not in {"cycle_root_sha256", "artifact_semantic_sha256"}
    }
    if payload["cycle_root_sha256"] != semantic_sha256_v4_4(base):
        raise _error("cycle_root_sha256 mismatch")
    if payload["authority"] != AUTHORITY_FLAGS or payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("cycle root authority/side effects must remain false")
    _validate_self(payload, "cycle root")
    return copy.deepcopy(payload)


def build_cycle_root_v4_4(
    *,
    v4_2_artifact_bindings: Sequence[Mapping[str, Any]],
    diagnostic_artifact_bindings: Sequence[Mapping[str, Any]],
    code_binding_set: Mapping[str, Any],
    expanded_candidate_selection: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(v4_2_artifact_bindings, Sequence) or len(
        v4_2_artifact_bindings
    ) != len(V4_2_PREDECESSOR_FILENAMES):
        raise _error("v4.2 root binding inventory mismatch")
    v42_bindings = [
        validate_artifact_binding_v4_4(row, expected_filename=filename)
        for row, filename in zip(
            v4_2_artifact_bindings, V4_2_PREDECESSOR_FILENAMES, strict=True
        )
    ]
    diagnostic_bindings = _validate_diagnostic_bindings(
        [copy.deepcopy(dict(row)) for row in diagnostic_artifact_bindings]
    )
    future = copy.deepcopy(dict(future_source_envelope))
    source = copy.deepcopy(future["source_summary"])
    current = [
        build_artifact_binding_v4_4(
            filename=filename, artifact=artifact
        )
        for filename, artifact in (
            ("code_binding_set.v4_4.json", code_binding_set),
            (
                "expanded_candidate_selection.v4_4.json",
                expanded_candidate_selection,
            ),
            (
                "definition_identity_collision_audit.v4_4.json",
                definition_identity_collision_audit,
            ),
            ("future_source_envelope.v4_4.json", future_source_envelope),
        )
    ]
    base = {
        "schema_version": CYCLE_ROOT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "market": "CN",
        "universe": "full_a",
        "cutoff": source["cutoff"],
        "snapshot_id": source["snapshot_id"],
        "cycle_id": future["cycle_id"],
        "embedded_v4_2_evidence_graph": {
            "role": "EMBEDDED_SEALED_EVIDENCE_GRAPH",
            "artifact_count": 14,
            "artifact_bindings": v42_bindings,
            "cross_cycle_state_edge": False,
        },
        "prior_diagnostic_evidence": {
            "role": "OUTCOME_INFORMED_NOMINATION_PROVENANCE_ONLY",
            "artifact_count": 3,
            "artifact_bindings": diagnostic_bindings,
            "statistics_inherited_as_formal_evidence": False,
        },
        "ordered_current_cycle_root_bindings": current,
        "source_summary": source,
        "lineage_contract": {
            "genesis_is_v4_4_local": True,
            "precommitted_predecessor_kind": "genesis",
            "v4_2_cycle_state_used_as_predecessor": False,
            "v4_3_diagnostic_cycle_state_used_as_predecessor": False,
            "allowed_state_transition": "PRECOMMITTED->DISCOVERY",
        },
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    return validate_cycle_root_v4_4(
        _seal({**base, "cycle_root_sha256": semantic_sha256_v4_4(base)})
    )


def validate_discovery_source_node_v4_4(
    value: Mapping[str, Any],
    *,
    cycle_root: Mapping[str, Any],
    precommitted_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    expanded_candidate_selection: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> dict[str, Any]:
    root = validate_cycle_root_v4_4(cycle_root)
    predecessor = validate_cycle_state_v4_1(
        precommitted_state,
        expected_cycle_id=root["cycle_id"],
        expected_cycle_root_sha256=root["cycle_root_sha256"],
        expected_state=PRECOMMITTED,
    )
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "cycle_root_sha256",
            "predecessor_state_binding",
            "ordered_evidence_bindings",
            "lineage_contract",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "v4.4 discovery source node",
    )
    expected_bindings = [
        build_artifact_binding_v4_4(filename=filename, artifact=artifact)
        for filename, artifact in (
            ("cycle_root.v4_4.json", root),
            ("future_source_envelope.v4_4.json", future_source_envelope),
            (
                "expanded_candidate_selection.v4_4.json",
                expanded_candidate_selection,
            ),
            (
                "definition_identity_collision_audit.v4_4.json",
                definition_identity_collision_audit,
            ),
            ("code_binding_set.v4_4.json", code_binding_set),
        )
    ]
    if (
        payload["schema_version"] != DISCOVERY_SOURCE_NODE_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["cycle_id"] != root["cycle_id"]
        or payload["cycle_root_sha256"] != root["cycle_root_sha256"]
        or payload["predecessor_state_binding"]
        != {
            "filename": "cycle_state.precommitted.v4_1.json",
            "byte_sha256": cycle_state_byte_sha256_v4_1(predecessor),
            "semantic_sha256": predecessor["state_semantic_sha256"],
        }
        or payload["ordered_evidence_bindings"] != expected_bindings
    ):
        raise _error("discovery source node binding mismatch")
    if payload["lineage_contract"] != {
        "embedded_v4_2_graph_is_evidence_only": True,
        "prior_diagnostic_is_nomination_only": True,
        "cross_cycle_predecessor_forbidden": True,
        "current_transition": "PRECOMMITTED->DISCOVERY",
    }:
        raise _error("discovery source node lineage mismatch")
    if payload["measurement"] != MEASUREMENT_FLAGS or payload["authority"] != AUTHORITY_FLAGS:
        raise _error("discovery source node measurement/authority mismatch")
    if payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("discovery source node side effects mismatch")
    _validate_self(payload, "discovery source node")
    return copy.deepcopy(payload)


def build_discovery_source_node_v4_4(
    *,
    cycle_root: Mapping[str, Any],
    precommitted_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    expanded_candidate_selection: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> dict[str, Any]:
    root = validate_cycle_root_v4_4(cycle_root)
    predecessor = validate_cycle_state_v4_1(
        precommitted_state,
        expected_cycle_id=root["cycle_id"],
        expected_cycle_root_sha256=root["cycle_root_sha256"],
        expected_state=PRECOMMITTED,
    )
    return validate_discovery_source_node_v4_4(
        _seal(
            {
                "schema_version": DISCOVERY_SOURCE_NODE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "cycle_id": root["cycle_id"],
                "cycle_root_sha256": root["cycle_root_sha256"],
                "predecessor_state_binding": {
                    "filename": "cycle_state.precommitted.v4_1.json",
                    "byte_sha256": cycle_state_byte_sha256_v4_1(predecessor),
                    "semantic_sha256": predecessor["state_semantic_sha256"],
                },
                "ordered_evidence_bindings": [
                    build_artifact_binding_v4_4(filename=filename, artifact=artifact)
                    for filename, artifact in (
                        ("cycle_root.v4_4.json", root),
                        (
                            "future_source_envelope.v4_4.json",
                            future_source_envelope,
                        ),
                        (
                            "expanded_candidate_selection.v4_4.json",
                            expanded_candidate_selection,
                        ),
                        (
                            "definition_identity_collision_audit.v4_4.json",
                            definition_identity_collision_audit,
                        ),
                        ("code_binding_set.v4_4.json", code_binding_set),
                    )
                ],
                "lineage_contract": {
                    "embedded_v4_2_graph_is_evidence_only": True,
                    "prior_diagnostic_is_nomination_only": True,
                    "cross_cycle_predecessor_forbidden": True,
                    "current_transition": "PRECOMMITTED->DISCOVERY",
                },
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        cycle_root=root,
        precommitted_state=predecessor,
        future_source_envelope=future_source_envelope,
        expanded_candidate_selection=expanded_candidate_selection,
        definition_identity_collision_audit=definition_identity_collision_audit,
        code_binding_set=code_binding_set,
    )


def validate_preregistration_discovery_cycle_v4_4(
    value: Mapping[str, Any],
    *,
    cycle_root: Mapping[str, Any],
    precommitted_state: Mapping[str, Any],
    discovery_source_node: Mapping[str, Any],
    discovery_state: Mapping[str, Any],
) -> dict[str, Any]:
    root = validate_cycle_root_v4_4(cycle_root)
    predecessor = validate_cycle_state_v4_1(
        precommitted_state,
        expected_cycle_id=root["cycle_id"],
        expected_cycle_root_sha256=root["cycle_root_sha256"],
        expected_state=PRECOMMITTED,
    )
    node = copy.deepcopy(dict(discovery_source_node))
    state = validate_cycle_state_v4_1(
        discovery_state,
        expected_cycle_id=root["cycle_id"],
        expected_cycle_root_sha256=root["cycle_root_sha256"],
        expected_state=DISCOVERY,
    )
    expected_predecessor = {
        "kind": "cycle_state",
        "byte_sha256": cycle_state_byte_sha256_v4_1(predecessor),
        "semantic_sha256": predecessor["state_semantic_sha256"],
    }
    if state["predecessor"] != expected_predecessor:
        raise _error("discovery state must descend from v4.4 PRECOMMITTED only")
    if state["source_chain_node_sha256"] != _artifact_semantic(
        node, "discovery source node"
    ):
        raise _error("discovery state source-node binding mismatch")
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "cycle_root_sha256",
            "precommitted_state_binding",
            "discovery_source_node_binding",
            "discovery_state_binding",
            "transition",
            "current_state",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "v4.4 preregistration discovery orchestration",
    )
    if (
        payload["schema_version"] != ORCHESTRATION_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["cycle_id"] != root["cycle_id"]
        or payload["cycle_root_sha256"] != root["cycle_root_sha256"]
        or payload["precommitted_state_binding"]
        != {
            "byte_sha256": cycle_state_byte_sha256_v4_1(predecessor),
            "semantic_sha256": predecessor["state_semantic_sha256"],
        }
        or payload["discovery_source_node_binding"]
        != build_artifact_binding_v4_4(
            filename="discovery_source_node.v4_4.json", artifact=node
        )
        or payload["discovery_state_binding"]
        != {
            "byte_sha256": cycle_state_byte_sha256_v4_1(state),
            "semantic_sha256": state["state_semantic_sha256"],
        }
        or payload["transition"]
        != {
            "from": PRECOMMITTED,
            "to": DISCOVERY,
            "predecessor_kind": "cycle_state",
            "cross_cycle_edge": False,
        }
        or payload["current_state"] != DISCOVERY
    ):
        raise _error("v4.4 discovery orchestration binding mismatch")
    if payload["measurement"] != MEASUREMENT_FLAGS or payload["authority"] != AUTHORITY_FLAGS:
        raise _error("orchestration measurement/authority mismatch")
    if payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("orchestration side effects mismatch")
    _validate_self(payload, "preregistration discovery orchestration")
    return copy.deepcopy(payload)


def build_preregistration_discovery_cycle_v4_4(
    *,
    cycle_root: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    expanded_candidate_selection: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    root = validate_cycle_root_v4_4(cycle_root)
    precommit_source = semantic_sha256_v4_4(
        {
            "domain": "v4.4-precommit-source-chain",
            "cycle_root_sha256": root["cycle_root_sha256"],
            "future_source_envelope_semantic_sha256": _artifact_semantic(
                future_source_envelope, "future source envelope"
            ),
        }
    )
    predecessor = build_genesis_cycle_state_v4_1(
        cycle_id=root["cycle_id"],
        cycle_root_sha256=root["cycle_root_sha256"],
        source_chain_node_sha256=precommit_source,
    )
    node = build_discovery_source_node_v4_4(
        cycle_root=root,
        precommitted_state=predecessor,
        future_source_envelope=future_source_envelope,
        expanded_candidate_selection=expanded_candidate_selection,
        definition_identity_collision_audit=definition_identity_collision_audit,
        code_binding_set=code_binding_set,
    )
    predecessor_byte = cycle_state_byte_sha256_v4_1(predecessor)
    discovery_state = build_next_cycle_state_v4_1(
        predecessor=predecessor,
        predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_semantic_sha256=predecessor[
            "state_semantic_sha256"
        ],
        cycle_id=root["cycle_id"],
        cycle_root_sha256=root["cycle_root_sha256"],
        next_state=DISCOVERY,
        source_chain_node_sha256=node["artifact_semantic_sha256"],
    )
    orchestration = _seal(
        {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "cycle_id": root["cycle_id"],
            "cycle_root_sha256": root["cycle_root_sha256"],
            "precommitted_state_binding": {
                "byte_sha256": predecessor_byte,
                "semantic_sha256": predecessor["state_semantic_sha256"],
            },
            "discovery_source_node_binding": build_artifact_binding_v4_4(
                filename="discovery_source_node.v4_4.json", artifact=node
            ),
            "discovery_state_binding": {
                "byte_sha256": cycle_state_byte_sha256_v4_1(discovery_state),
                "semantic_sha256": discovery_state["state_semantic_sha256"],
            },
            "transition": {
                "from": PRECOMMITTED,
                "to": DISCOVERY,
                "predecessor_kind": "cycle_state",
                "cross_cycle_edge": False,
            },
            "current_state": DISCOVERY,
            "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
            "authority": copy.deepcopy(AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
        }
    )
    orchestration = validate_preregistration_discovery_cycle_v4_4(
        orchestration,
        cycle_root=root,
        precommitted_state=predecessor,
        discovery_source_node=node,
        discovery_state=discovery_state,
    )
    return {
        "cycle_state.precommitted.v4_1.json": predecessor,
        "discovery_source_node.v4_4.json": node,
        "cycle_state.discovery.v4_1.json": discovery_state,
        "prereg_discovery_orchestration.v4_4.json": orchestration,
    }


__all__ = [
    "AUTHORITY_FLAGS",
    "CODE_BINDING_PATHS_V4_4",
    "CODE_BINDING_SET_SCHEMA_VERSION",
    "CYCLE_ROOT_SCHEMA_VERSION",
    "DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION",
    "DISCOVERY_SOURCE_NODE_SCHEMA_VERSION",
    "EVIDENCE_CONTRACT_VERSION",
    "EXPECTED_CANDIDATE_ROWS",
    "EXPECTED_PRIOR_DIAGNOSTIC_BINDINGS",
    "EXPANDED_SELECTION_SCHEMA_VERSION",
    "FROZEN_PREVIOUS_CUTOFF",
    "FactorGovernanceCandidatePreregistrationV4_4Error",
    "MEASUREMENT_FLAGS",
    "ORCHESTRATION_SCHEMA_VERSION",
    "PRIOR_DIAGNOSTIC_FILENAMES",
    "PROTOCOL_VERSION",
    "PUBLICATION_TIME_AUTHORITY",
    "SCHEMA_VERSION",
    "SELECTION_CLAIMS",
    "SIDE_EFFECT_FLAGS",
    "SOURCE_ENVELOPE_SCHEMA_VERSION",
    "V4_2_PREDECESSOR_FILENAMES",
    "V4_2_PREDECESSOR_PREFIX",
    "build_artifact_binding_v4_4",
    "build_code_binding_set_v4_4",
    "build_cycle_root_v4_4",
    "build_definition_identity_collision_audit_v4_4",
    "build_discovery_source_node_v4_4",
    "build_expanded_candidate_selection_v4_4",
    "build_future_source_envelope_v4_4",
    "build_preregistration_discovery_cycle_v4_4",
    "byte_sha256_v4_4",
    "canonical_file_bytes_v4_4",
    "canonical_json_bytes_v4_4",
    "deterministic_cycle_id_v4_4",
    "semantic_sha256_v4_4",
    "validate_artifact_binding_v4_4",
    "validate_code_binding_set_v4_4",
    "validate_cycle_root_v4_4",
    "validate_definition_identity_collision_audit_v4_4",
    "validate_discovery_source_node_v4_4",
    "validate_expanded_candidate_selection_v4_4",
    "validate_future_source_envelope_v4_4",
    "validate_preregistration_discovery_cycle_v4_4",
    "validate_prior_diagnostic_graph_v4_4",
    "validate_v4_2_predecessor_graph_v4_4",
]
