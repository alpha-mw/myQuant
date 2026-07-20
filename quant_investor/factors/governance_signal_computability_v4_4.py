"""Pure additive exact-five signal-computability evidence contract for v4.4."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import PurePath
from typing import Any

from quant_investor.factors import governance_candidate_preregistration_v4_2 as v42
from quant_investor.factors import governance_candidate_preregistration_v4_4 as prereg
from quant_investor.factors import governance_exact_five_no_label_eval_v4_4 as evaluator
from quant_investor.factors import governance_prior_diagnostic_nomination_v4_3 as v43diag


PROTOCOL_VERSION = "v4"
EVIDENCE_CONTRACT_VERSION = "v4.4"
SCHEMA_VERSION = "factor-governance-exact-five-signal-computability.v4.4"
READBACK_SCHEMA_VERSION = (
    "factor-governance-exact-five-signal-computability-readback.v4.4"
)
SNAPSHOT_BINDING_SCHEMA_VERSION = "factor-governance-exact-five-data-envelope.v4.4"
PREREG_BINDING_SCHEMA_VERSION = (
    "factor-governance-exact-five-preregistration-binding.v4.4"
)
SYNTHETIC_SCOPE = "SYNTHETIC_VALIDATION_ONLY"
STRICT_SCOPE = "FUTURE_STRICT_FULL_A_SNAPSHOT"
FROZEN_PREVIOUS_CUTOFF = prereg.FROZEN_PREVIOUS_CUTOFF

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT = re.compile(r"\d{8}T\d{6}Z")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")

SOURCE_BINDINGS_V4_4 = {
    "aquant": {
        "commit": v42.AQUANT_COMMIT,
        "relative_path": v42.AQUANT_PATH,
        "blob_oid": v42.AQUANT_BLOB_OID,
        "raw_sha256": v42.AQUANT_RAW_SHA256,
        "file_mode": v42.AQUANT_MODE,
        "range_definition_identity_sha256": v42.AQUANT_RANGE_DEFINITION_SHA256,
    },
    "myquant": {
        "commit": v42.MYQUANT_COMMIT,
        "relative_path": v42.MYQUANT_PATH,
        "blob_oid": v42.MYQUANT_BLOB_OID,
        "raw_sha256": v42.MYQUANT_FULL_SHA256,
        "alias_rows": copy.deepcopy(list(v42.MYQUANT_ALIAS_ROWS)),
        "vol_of_vol_definition_identity_sha256": v43diag.DEFINITION_IDENTITY_SHA256,
    },
    "restricted_source_programs_semantic_sha256": (
        evaluator.source_programs_semantic_sha256_v4_4()
    ),
}

FIELD_ADAPTERS_V4_4 = (
    {
        "candidate": "alpha_range_position_momentum_20d",
        "source_fields": ["close"],
        "physical_columns": ["close"],
        "evaluator_inputs": ["raw_close"],
        "adjustment": "raw_unadjusted",
        "unit_transform": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nan_then_node_level_pit_mask",
        "fallback": False,
    },
    {
        "candidate": "pv_low_overnight_gap_20d",
        "source_fields": ["open", "close"],
        "physical_columns": ["open", "close"],
        "evaluator_inputs": ["raw_open", "raw_close"],
        "adjustment": "raw_unadjusted",
        "unit_transform": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nan_then_node_level_pit_mask",
        "fallback": False,
    },
    {
        "candidate": "pv_low_vol_ratio_10_60",
        "source_fields": ["close"],
        "physical_columns": ["close"],
        "evaluator_inputs": ["raw_close"],
        "adjustment": "raw_unadjusted",
        "unit_transform": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nan_then_node_level_pit_mask",
        "fallback": False,
    },
    {
        "candidate": "pv_price_volume_consistency_20d",
        "source_fields": ["close", "volume"],
        "physical_columns": ["close", "vol"],
        "evaluator_inputs": ["raw_close", "vol"],
        "adjustment": "raw_unadjusted",
        "unit_transform": "canonical_vol_exposed_as_volume_without_scaling",
        "dtype": "float64",
        "missing_policy": "preserve_nan_then_node_level_pit_mask",
        "fallback": False,
    },
    {
        "candidate": "pv_low_vol_of_vol_20d",
        "source_fields": ["close"],
        "physical_columns": ["adj_close"],
        "evaluator_inputs": ["adj_close"],
        "adjustment": "exact_adjusted_close",
        "unit_transform": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nan_then_node_level_pit_mask",
        "fallback": False,
    },
)

MEASUREMENT_FLAGS = {
    "statistics": "not_run",
    "family_bh": "not_run",
    "maturity": "not_run",
    "walk_forward": "not_run",
    "cost": "not_run",
    "neutralization": "not_run",
    "stability": "not_run",
    "formal_dedup": "not_run",
    "high_correlation_dedup": "not_run",
    "verified_v4_replay": "not_run",
    "transaction_plan": "not_run",
}
AUTHORITY_FLAGS = {
    "healthy_source_receipt": False,
    "screening_authorized": False,
    "family_bh_authorized": False,
    "maturity_authorized": False,
    "candidate_qualified": False,
    "qualification_authorized": False,
    "admission_authorized": False,
    "production_new_risk_authorized": False,
    "production_candidate_authorized": False,
    "registry_write_authorized": False,
    "production_proposal_authorized": False,
    "activation_authorized": False,
    "apply_authorized": False,
}
SIDE_EFFECT_FLAGS = {
    "registry": False,
    "wal": False,
    "budget": False,
    "production_receipt": False,
    "production_pointer": False,
    "proposal": False,
    "apply": False,
    "provider": False,
    "network": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
    "transaction": False,
}

_BANNED_INPUT_KEYS = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "forward_return",
        "forward_returns",
        "realized_return",
        "realized_returns",
        "ic",
        "rank_ic",
        "p_value",
        "q_value",
        "bh",
        "maturity",
        "cost",
        "replay",
        "pnl",
        "performance",
        "outcome",
        "verdict",
        "registry",
        "provider",
        "broker",
        "order",
        "trade",
    }
)


class FactorGovernanceSignalComputabilityV4_4Error(ValueError):
    """Raised when additive v4.4 computability evidence fails closed."""


def _error(message: str) -> FactorGovernanceSignalComputabilityV4_4Error:
    return FactorGovernanceSignalComputabilityV4_4Error(message)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _date(value: Any, label: str) -> str:
    if type(value) is not str or _DATE.fullmatch(value) is None:
        raise _error(f"{label} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} is not a calendar date") from exc
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = copy.deepcopy(dict(value))
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    if set(payload) != fields:
        raise _error(f"{label} fields are not exact")
    evaluator.canonical_json_bytes_v4_4(payload)
    return payload


def _reject_banned_input_keys(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise _error(f"{label} contains a non-string field")
            if key.lower() in _BANNED_INPUT_KEYS:
                raise _error(f"{label} contains prohibited outcome/authority field: {key}")
            _reject_banned_input_keys(child, label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_banned_input_keys(child, label)


def field_adapters_semantic_sha256_v4_4() -> str:
    return evaluator.semantic_sha256_v4_4(list(FIELD_ADAPTERS_V4_4))


def candidate_rows_semantic_sha256_v4_4() -> str:
    return evaluator.semantic_sha256_v4_4(list(prereg.EXPECTED_CANDIDATE_ROWS))


def synthetic_fixture_collection_sha256_v4_4() -> str:
    return evaluator.semantic_sha256_v4_4(
        {
            "source_bindings": SOURCE_BINDINGS_V4_4,
            "field_adapters": list(FIELD_ADAPTERS_V4_4),
            "synthetic_fixture_binding": evaluator.synthetic_fixture_binding_v4_4(),
        }
    )


def recompute_synthetic_passes_v4_4() -> list[dict[str, Any]]:
    fixture_binding = evaluator.synthetic_fixture_binding_v4_4()
    collection_sha256 = synthetic_fixture_collection_sha256_v4_4()
    passes: list[dict[str, Any]] = []
    for pass_index in (1, 2):
        matrices, pit = evaluator.build_synthetic_fixture_v4_4()
        current_inputs = {
            name: evaluator.matrix_hash_descriptor_v4_4(matrices[name])
            for name in evaluator.INPUT_FIELDS
        }
        current_pit = evaluator.matrix_hash_descriptor_v4_4(pit.astype(float))
        if (
            current_inputs != fixture_binding["input_matrix_descriptors"]
            or current_pit != fixture_binding["pit_mask_descriptor"]
        ):
            raise _error("deterministic synthetic fixture drifted during recomputation")
        pass_id = f"fresh_pass_{pass_index}"
        source_outputs = evaluator.evaluate_source_dag_v4_4(matrices, pit)
        local_outputs = evaluator.evaluate_local_formulas_v4_4(matrices, pit)
        passes.append(
            {
                "pass_id": pass_id,
                "collection_sha256": collection_sha256,
                "engines": [
                    evaluator.build_engine_pass_result_v4_4(
                        engine_id=evaluator.SOURCE_ENGINE_ID,
                        pass_id=pass_id,
                        collection_sha256=collection_sha256,
                        outputs=source_outputs,
                        pit_mask=pit,
                    ),
                    evaluator.build_engine_pass_result_v4_4(
                        engine_id=evaluator.LOCAL_ENGINE_ID,
                        pass_id=pass_id,
                        collection_sha256=collection_sha256,
                        outputs=local_outputs,
                        pit_mask=pit,
                    ),
                ],
            }
        )
    return passes


def validate_snapshot_binding_v4_4(
    value: Mapping[str, Any], *, evidence_scope: str
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "source_kind",
            "market",
            "universe",
            "snapshot_id",
            "analysis_start",
            "cutoff",
            "latest_trade_date",
            "complete_trade_date",
            "full_a_count",
            "covered_count",
            "coverage_ratio",
            "full_a_semantic_sha256",
            "snapshot_manifest_sha256",
            "table_inventory_sha256",
            "pit_membership_sha256",
            "pit_manifest_sha256",
            "date_axis_sha256",
            "symbol_axis_sha256",
            "eligibility_matrix_sha256",
            "pit_mask_descriptor",
            "fallbacks",
            "strict_full_a_proven",
        },
        "snapshot binding",
    )
    if (
        payload["schema_version"] != SNAPSHOT_BINDING_SCHEMA_VERSION
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
        or type(payload["snapshot_id"]) is not str
        or _SNAPSHOT.fullmatch(payload["snapshot_id"]) is None
    ):
        raise _error("snapshot binding identity mismatch")
    cutoff = _date(payload["cutoff"], "cutoff")
    analysis_start = _date(payload["analysis_start"], "analysis_start")
    if (
        cutoff <= FROZEN_PREVIOUS_CUTOFF
        or payload["snapshot_id"][:8] != cutoff.replace("-", "")
        or payload["latest_trade_date"] != cutoff
        or payload["complete_trade_date"] != cutoff
        or analysis_start > cutoff
    ):
        raise _error("snapshot cutoff must be strictly later than 2026-07-19")
    if (
        type(payload["full_a_count"]) is not int
        or payload["full_a_count"] <= 0
        or payload["covered_count"] != payload["full_a_count"]
        or payload["coverage_ratio"] != 1.0
    ):
        raise _error("snapshot binding must prove exact full-A coverage")
    for field in (
        "full_a_semantic_sha256",
        "snapshot_manifest_sha256",
        "table_inventory_sha256",
        "pit_membership_sha256",
        "pit_manifest_sha256",
        "date_axis_sha256",
        "symbol_axis_sha256",
        "eligibility_matrix_sha256",
    ):
        _sha(payload[field], f"snapshot {field}")
    if payload["fallbacks"] != {
        "csv": False,
        "mock": False,
        "serving": False,
        "stale_pointer": False,
    }:
        raise _error("snapshot fallbacks must all be false")
    if evidence_scope == STRICT_SCOPE:
        if (
            payload["source_kind"] != "strict_parquet"
            or payload["strict_full_a_proven"] is not True
        ):
            raise _error("strict scope requires a proven strict-Parquet full-A source")
    elif evidence_scope == SYNTHETIC_SCOPE:
        if (
            payload["source_kind"] != "synthetic_fixture"
            or payload["strict_full_a_proven"] is not False
        ):
            raise _error("synthetic scope must remain explicitly non-strict")
    else:
        raise _error("evidence_scope is not accepted")
    pit = payload["pit_mask_descriptor"]
    if not isinstance(pit, Mapping):
        raise _error("snapshot PIT mask descriptor must be an object")
    if (
        pit.get("matrix_sha256") != payload["eligibility_matrix_sha256"]
        or pit.get("date_axis", {}).get("sha256") != payload["date_axis_sha256"]
        or pit.get("date_axis", {}).get("first") != payload["analysis_start"]
        or pit.get("date_axis", {}).get("last") != payload["cutoff"]
        or pit.get("symbol_axis", {}).get("sha256") != payload["symbol_axis_sha256"]
        or pit.get("column_count") != payload["full_a_count"]
    ):
        raise _error("snapshot PIT mask descriptor bindings mismatch")
    _reject_banned_input_keys(payload, "snapshot binding")
    return payload


def validate_preregistration_binding_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "binding_scope",
            "cycle_id",
            "bundle_path",
            "artifact_count",
            "readback_byte_sha256",
            "readback_semantic_sha256",
            "candidate_rows_semantic_sha256",
            "existing_signal_computability",
            "existing_authority_false",
            "immutable_readback_accepted",
        },
        "v4.4 preregistration binding",
    )
    path = payload["bundle_path"]
    if (
        payload["schema_version"] != PREREG_BINDING_SCHEMA_VERSION
        or payload["binding_scope"] not in {
            "synthetic_fixture",
            "immutable_private_readback",
        }
        or type(payload["cycle_id"]) is not str
        or _SAFE_SEGMENT.fullmatch(payload["cycle_id"]) is None
        or not payload["cycle_id"].startswith("cn_full_a_v4_4_")
        or type(path) is not str
        or not PurePath(path).is_absolute()
        or path != str(PurePath(path))
        or ".." in PurePath(path).parts
        or "\x00" in path
        or payload["artifact_count"] != 27
        or payload["candidate_rows_semantic_sha256"]
        != candidate_rows_semantic_sha256_v4_4()
        or payload["existing_signal_computability"] != "not_run"
        or payload["existing_authority_false"] is not True
    ):
        raise _error("v4.4 preregistration binding mismatch")
    if (
        payload["binding_scope"] == "synthetic_fixture"
        and payload["immutable_readback_accepted"] is not False
    ) or (
        payload["binding_scope"] == "immutable_private_readback"
        and payload["immutable_readback_accepted"] is not True
    ):
        raise _error("v4.4 preregistration readback scope mismatch")
    _sha(payload["readback_byte_sha256"], "preregistration readback byte SHA")
    _sha(
        payload["readback_semantic_sha256"],
        "preregistration readback semantic SHA",
    )
    _reject_banned_input_keys(payload, "preregistration binding")
    return payload


def _validate_source_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    if payload != SOURCE_BINDINGS_V4_4:
        raise _error("source bindings differ from the exact pinned source identities")
    _reject_banned_input_keys(payload, "source bindings")
    return payload


def _validate_field_adapters(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error("field adapters must be a sequence")
    normalized = [copy.deepcopy(dict(row)) for row in value]
    if normalized != list(FIELD_ADAPTERS_V4_4):
        raise _error("field adapters differ from the frozen candidate-specific mappings")
    return normalized


def _validate_passes(
    value: Sequence[Mapping[str, Any]], *, snapshot: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise _error("exactly two fresh computation passes are required")
    normalized: list[dict[str, Any]] = []
    pass_equivalence: list[dict[str, Any]] = []
    collection_sha: str | None = None
    for index, raw_pass in enumerate(value, start=1):
        payload = _exact(
            raw_pass,
            {"pass_id", "collection_sha256", "engines"},
            f"fresh pass {index}",
        )
        expected_pass_id = f"fresh_pass_{index}"
        if payload["pass_id"] != expected_pass_id:
            raise _error("fresh pass order or identity mismatch")
        current_collection = _sha(
            payload["collection_sha256"], f"fresh pass {index} collection SHA"
        )
        if collection_sha is None:
            collection_sha = current_collection
        elif collection_sha != current_collection:
            raise _error("fresh source/data collections differ across passes")
        engines = payload["engines"]
        if not isinstance(engines, list) or len(engines) != 2:
            raise _error("each pass must contain exactly two independent engines")
        try:
            validated = [
                evaluator.validate_engine_pass_result_v4_4(row) for row in engines
            ]
        except evaluator.FactorGovernanceExactFiveEvalV4_4Error as exc:
            raise _error(f"invalid independent engine result: {exc}") from exc
        if [row["engine_id"] for row in validated] != [
            evaluator.SOURCE_ENGINE_ID,
            evaluator.LOCAL_ENGINE_ID,
        ]:
            raise _error("independent engine inventory/order mismatch")
        if any(
            row["pass_id"] != expected_pass_id
            or row["collection_sha256"] != current_collection
            for row in validated
        ):
            raise _error("engine result does not bind its fresh pass collection")
        if any(row["pit_mask"] != snapshot["pit_mask_descriptor"] for row in validated):
            raise _error("engine PIT mask differs from the snapshot binding")
        source_payload = evaluator.engine_equivalence_payload_v4_4(validated[0])
        local_payload = evaluator.engine_equivalence_payload_v4_4(validated[1])
        if source_payload != local_payload:
            raise _error("independent engines differ within a fresh pass")
        pass_equivalence.append(source_payload)
        payload["engines"] = validated
        normalized.append(payload)
    if pass_equivalence[0] != pass_equivalence[1]:
        raise _error("exact-five outputs drift across fresh passes")
    return normalized


def build_signal_computability_proof_v4_4(
    *,
    evidence_scope: str,
    source_bindings: Mapping[str, Any],
    field_adapters: Sequence[Mapping[str, Any]],
    snapshot_binding: Mapping[str, Any],
    preregistration_binding: Mapping[str, Any],
    passes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if evidence_scope != SYNTHETIC_SCOPE:
        raise _error(
            "strict snapshot proof construction is unavailable in the synthetic-only slice"
        )
    source = _validate_source_bindings(source_bindings)
    adapters = _validate_field_adapters(field_adapters)
    snapshot = validate_snapshot_binding_v4_4(
        snapshot_binding, evidence_scope=evidence_scope
    )
    preregistration = validate_preregistration_binding_v4_4(
        preregistration_binding
    )
    if preregistration["binding_scope"] != "synthetic_fixture":
        raise _error("synthetic-only proof requires a synthetic preregistration fixture")
    fixture_binding = evaluator.synthetic_fixture_binding_v4_4()
    if snapshot["pit_mask_descriptor"] != fixture_binding["pit_mask_descriptor"]:
        raise _error("snapshot PIT mask differs from the deterministic synthetic fixture")
    normalized_passes = _validate_passes(passes, snapshot=snapshot)
    recomputed_passes = recompute_synthetic_passes_v4_4()
    if normalized_passes != recomputed_passes:
        raise _error(
            "supplied passes differ from independent deterministic fixture recomputation"
        )
    proof = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "evidence_scope": evidence_scope,
        "source_binding_verification": "CONTRACT_CONSTANTS_ONLY_NOT_GIT_REOPENED",
        "source_bindings": source,
        "field_adapters": adapters,
        "field_adapters_semantic_sha256": field_adapters_semantic_sha256_v4_4(),
        "synthetic_fixture_binding": fixture_binding,
        "snapshot_binding": snapshot,
        "preregistration_binding": preregistration,
        "passes": normalized_passes,
        "candidate_count": 5,
        "atomic_exact_five_passed": True,
        "independent_engine_equivalence_proven": True,
        "double_collection_reproducibility_proven": True,
        "synthetic_validation_passed": True,
        "strict_snapshot_signal_computability_proven": False,
        "readiness": "NON_AUTHORIZING_COMPUTABILITY_ONLY",
        "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    proof["artifact_semantic_sha256"] = evaluator.semantic_sha256_v4_4(proof)
    return proof


def validate_signal_computability_proof_v4_4(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "evidence_contract_version",
            "evidence_scope",
            "source_binding_verification",
            "source_bindings",
            "field_adapters",
            "field_adapters_semantic_sha256",
            "synthetic_fixture_binding",
            "snapshot_binding",
            "preregistration_binding",
            "passes",
            "candidate_count",
            "atomic_exact_five_passed",
            "independent_engine_equivalence_proven",
            "double_collection_reproducibility_proven",
            "synthetic_validation_passed",
            "strict_snapshot_signal_computability_proven",
            "readiness",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "exact-five signal-computability proof",
    )
    supplied = _sha(
        payload.pop("artifact_semantic_sha256"), "proof artifact semantic SHA"
    )
    rebuilt = build_signal_computability_proof_v4_4(
        evidence_scope=payload["evidence_scope"],
        source_bindings=payload["source_bindings"],
        field_adapters=payload["field_adapters"],
        snapshot_binding=payload["snapshot_binding"],
        preregistration_binding=payload["preregistration_binding"],
        passes=payload["passes"],
    )
    if (
        supplied != evaluator.semantic_sha256_v4_4(payload)
        or supplied != rebuilt["artifact_semantic_sha256"]
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
        or payload["source_binding_verification"]
        != "CONTRACT_CONSTANTS_ONLY_NOT_GIT_REOPENED"
        or payload["field_adapters_semantic_sha256"]
        != field_adapters_semantic_sha256_v4_4()
        or payload["synthetic_fixture_binding"]
        != evaluator.synthetic_fixture_binding_v4_4()
        or payload["candidate_count"] != 5
        or payload["atomic_exact_five_passed"] is not True
        or payload["independent_engine_equivalence_proven"] is not True
        or payload["double_collection_reproducibility_proven"] is not True
        or payload["synthetic_validation_passed"] is not True
        or payload["strict_snapshot_signal_computability_proven"] is not False
        or payload["readiness"] != "NON_AUTHORIZING_COMPUTABILITY_ONLY"
        or payload["measurement"] != MEASUREMENT_FLAGS
        or payload["authority"] != AUTHORITY_FLAGS
        or payload["side_effects"] != SIDE_EFFECT_FLAGS
    ):
        raise _error("exact-five signal-computability proof mismatch")
    return rebuilt


def build_signal_computability_readback_v4_4(
    *, proof: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_signal_computability_proof_v4_4(proof)
    proof_bytes = evaluator.canonical_json_bytes_v4_4(normalized) + b"\n"
    readback = {
        "schema_version": READBACK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "proof_byte_sha256": hashlib.sha256(proof_bytes).hexdigest(),
        "proof_semantic_sha256": normalized["artifact_semantic_sha256"],
        "evidence_scope": normalized["evidence_scope"],
        "candidate_count": 5,
        "atomic_exact_five_passed": True,
        "strict_snapshot_signal_computability_proven": normalized[
            "strict_snapshot_signal_computability_proven"
        ],
        "readiness": "NON_AUTHORIZING_COMPUTABILITY_ONLY",
        "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    readback["artifact_semantic_sha256"] = evaluator.semantic_sha256_v4_4(readback)
    return readback


def validate_signal_computability_readback_v4_4(
    value: Mapping[str, Any], *, proof: Mapping[str, Any]
) -> dict[str, Any]:
    normalized_proof = validate_signal_computability_proof_v4_4(proof)
    expected = build_signal_computability_readback_v4_4(proof=normalized_proof)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise _error("signal-computability readback differs from independent rebuild")
    return copy.deepcopy(expected)


__all__ = [
    "AUTHORITY_FLAGS",
    "EVIDENCE_CONTRACT_VERSION",
    "FIELD_ADAPTERS_V4_4",
    "FROZEN_PREVIOUS_CUTOFF",
    "FactorGovernanceSignalComputabilityV4_4Error",
    "MEASUREMENT_FLAGS",
    "PREREG_BINDING_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "READBACK_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SIDE_EFFECT_FLAGS",
    "SNAPSHOT_BINDING_SCHEMA_VERSION",
    "SOURCE_BINDINGS_V4_4",
    "STRICT_SCOPE",
    "SYNTHETIC_SCOPE",
    "build_signal_computability_proof_v4_4",
    "build_signal_computability_readback_v4_4",
    "candidate_rows_semantic_sha256_v4_4",
    "field_adapters_semantic_sha256_v4_4",
    "recompute_synthetic_passes_v4_4",
    "synthetic_fixture_collection_sha256_v4_4",
    "validate_preregistration_binding_v4_4",
    "validate_signal_computability_proof_v4_4",
    "validate_signal_computability_readback_v4_4",
    "validate_snapshot_binding_v4_4",
]
