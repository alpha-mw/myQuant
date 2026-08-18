"""Non-authorizing evidence for the user-approved bootstrap declaration."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any, Final

from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    seal_artifact,
)

from .bootstrap import (
    BLEND_W75_CONTROL,
    NOT_CLAIMED,
    bootstrap_factor_definitions,
)
from .common import (
    artifact_ref,
    business_identity,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .errors import FactorGovernanceError

BOOTSTRAP_EVIDENCE_KIND: Final = "factor.bootstrap_exception_evidence"
BOOTSTRAP_ADMISSION_ROUTE: Final = "BOOTSTRAP_EXCEPTION"
BOOTSTRAP_DECISION_SOURCE_ID: Final = "user-approved-unified-runtime-cutover"

_EVIDENCE_FIELDS: Final = {
    "bootstrap_evidence_id",
    "admission_route",
    "producer_identity",
    "decision_source_id",
    "decision_source_sha256",
    "factor_rows",
    "reader_contract",
    "source_refs",
    "factor_set_sha256",
    "weight_total",
    "authorizes_readiness",
    "authorizes_selectability",
}
_SOURCE_ROLE_KINDS: Final = {
    "code": "system.release",
    "decision_source": "system.source_bundle",
    "exchange_calendar": "system.source_bundle",
    "implementation": "system.source_bundle",
    "market": "system.source_bundle",
    "pit_universe": "system.source_bundle",
    "recomputation": "system.source_bundle",
    "source_generation": "system.source_bundle",
}
_DECISION_DOCUMENT: Final = {
    "kind": "factor.bootstrap_decision",
    "decision_source_id": BOOTSTRAP_DECISION_SOURCE_ID,
    "admission_route": BOOTSTRAP_ADMISSION_ROUTE,
    "producer_identity": NOT_CLAIMED,
    "factor_weights": [
        {
            "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
            "weight": "0.500000000000",
        },
        {
            "factor_id": "pv_low_dollar_volume_5d",
            "weight": "0.500000000000",
        },
    ],
    "control_factor_ids": [BLEND_W75_CONTROL],
    "prospective_evidence_claimed": False,
    "activation_authorized": False,
}
_READER_CONTRACT: Final = {
    "reader": "MarketDataReader",
    "market": "CN",
    "mode_policy": "strict",
    "source_format": "PARQUET",
    "fallback_allowed": False,
}


def _definition_parts(row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Read the canonical bootstrap definition without accepting synonyms."""

    required = {
        "factor_id",
        "spec_id",
        "family",
        "formula",
        "parameters",
        "direction",
        "input_fields",
        "required_source_roles",
        "role",
        "selectable",
        "bootstrap_weight",
        "producer_identity",
    }
    if type(row) is not dict or set(row) != required:
        raise FactorGovernanceError("bootstrap factor definition fields are not exact")
    return dict(row), str(row["bootstrap_weight"])


def _canonical_factor_rows(
    source_refs: list[dict[str, Any]],
    *,
    code_sha256: str,
    implementation_sha256: str,
) -> list[dict[str, Any]]:
    del source_refs
    code_sha = require_sha256(code_sha256, label="code_sha256")
    implementation_sha = require_sha256(implementation_sha256, label="implementation_sha256")
    result: list[dict[str, Any]] = []
    for definition in bootstrap_factor_definitions():
        spec, weight = _definition_parts(definition)
        if spec.get("role") == "CONTROL_ONLY":
            continue
        factor_id = str(spec["factor_id"])
        result.append(
            {
                "factor_id": factor_id,
                "spec_id": str(spec["spec_id"]),
                "formula": str(spec["formula"]),
                "parameters": dict(spec["parameters"]),
                "direction": str(spec["direction"]),
                "input_fields": sorted(
                    list(spec["input_fields"]),
                    key=lambda value: value.encode("utf-8"),
                ),
                "required_source_roles": list(spec["required_source_roles"]),
                "implementation_sha256": implementation_sha,
                "code_sha256": code_sha,
                "weight": weight,
                "role": "BOOTSTRAP",
                "selectable": True,
            }
        )
    return sorted(result, key=lambda row: row["factor_id"].encode("utf-8"))


def _factor_set_sha256(factor_rows: list[dict[str, Any]]) -> str:
    definitions = []
    for definition in bootstrap_factor_definitions():
        spec, weight = _definition_parts(definition)
        role = str(spec["role"])
        definitions.append(
            {
                "factor_id": str(spec["factor_id"]),
                "spec_id": str(spec["spec_id"]),
                "family": str(spec["family"]),
                "formula": str(spec["formula"]),
                "parameters": dict(spec["parameters"]),
                "direction": str(spec["direction"]),
                "input_fields": sorted(
                    list(spec["input_fields"]),
                    key=lambda value: value.encode("utf-8"),
                ),
                "required_source_roles": list(spec["required_source_roles"]),
                "role": role,
                "selectable": role != "CONTROL_ONLY",
                "bootstrap_weight": weight,
                "producer_identity": NOT_CLAIMED,
            }
        )
    definitions.sort(key=lambda row: str(row["factor_id"]).encode("utf-8"))
    control_rows = [
        {
            "factor_id": row["factor_id"],
            "spec_id": row["spec_id"],
            "direction": row["direction"],
            "required_source_roles": row["required_source_roles"],
            "weight": row["bootstrap_weight"],
            "role": row["role"],
            "selectable": row["selectable"],
        }
        for row in definitions
        if row["role"] == "CONTROL_ONLY"
    ]
    content = {
        "factor_definitions": definitions,
        "factor_rows": [
            {
                key: row[key]
                for key in (
                    "factor_id",
                    "spec_id",
                    "direction",
                    "required_source_roles",
                    "weight",
                    "role",
                    "selectable",
                )
            }
            for row in factor_rows
        ],
        "control_rows": control_rows,
        "weighting_method": "EQUAL_WEIGHT",
        "weight_total": "1.000000000000",
    }
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _source_ref_rows(
    source_artifacts: Mapping[str, Mapping[str, Any] | bytes],
) -> list[dict[str, Any]]:
    if type(source_artifacts) is not dict or set(source_artifacts) != set(_SOURCE_ROLE_KINDS):
        raise FactorGovernanceError("bootstrap source artifact roles are not exact")
    rows = []
    for role in sorted(_SOURCE_ROLE_KINDS, key=lambda value: value.encode("utf-8")):
        artifact = validate_governance_artifact(
            source_artifacts[role], expected_kind=_SOURCE_ROLE_KINDS[role]
        )
        required_inner_role = {
            "decision_source": "bootstrap_decision",
            "implementation": "implementation_tree_manifest",
        }.get(role)
        if required_inner_role is not None:
            sources = artifact["payload"]["sources"]
            if type(sources) is not list:
                raise FactorGovernanceError(f"{role} source bundle sources must be a list")
            matches = [
                row
                for row in sources
                if type(row) is dict and row.get("role") == required_inner_role
            ]
            if len(matches) != 1 or set(matches[0]) != {"role", "source_ref"}:
                raise FactorGovernanceError(
                    f"{role} source bundle lacks unique {required_inner_role}"
                )
            validate_artifact_ref(
                matches[0]["source_ref"],
                label=f"{role}.{required_inner_role}",
                expected_kind="system.source_object",
            )
        rows.append({"role": role, "ref": artifact_ref(artifact)})
    return rows


def _decision_sha256(decision_source_bytes: bytes) -> str:
    if type(decision_source_bytes) is not bytes:
        raise FactorGovernanceError("decision source must be exact canonical bytes")
    try:
        decision = parse_canonical_json_bytes(decision_source_bytes)
    except Exception as exc:
        raise FactorGovernanceError("decision source is not canonical JSON") from exc
    if decision != _DECISION_DOCUMENT:
        raise FactorGovernanceError("decision source differs from the approved cutover")
    return hashlib.sha256(decision_source_bytes).hexdigest()


def _payload(
    *,
    decision_source_sha256: str,
    source_refs: list[dict[str, Any]],
    code_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    normalized_refs = []
    if type(source_refs) is not list or len(source_refs) != len(_SOURCE_ROLE_KINDS):
        raise FactorGovernanceError("bootstrap source refs are incomplete")
    for index, row in enumerate(source_refs):
        if type(row) is not dict or set(row) != {"role", "ref"}:
            raise FactorGovernanceError("bootstrap source ref row fields are not exact")
        role = row["role"]
        if role not in _SOURCE_ROLE_KINDS:
            raise FactorGovernanceError("bootstrap source ref role is invalid")
        normalized_refs.append(
            {
                "role": role,
                "ref": validate_artifact_ref(
                    row["ref"],
                    label=f"source_refs[{index}].ref",
                    expected_kind=_SOURCE_ROLE_KINDS[role],
                ),
            }
        )
    normalized_refs.sort(key=lambda row: row["role"].encode("utf-8"))
    if [row["role"] for row in normalized_refs] != sorted(_SOURCE_ROLE_KINDS):
        raise FactorGovernanceError("bootstrap source ref roles are duplicated")
    factor_rows = _canonical_factor_rows(
        normalized_refs,
        code_sha256=code_sha256,
        implementation_sha256=implementation_sha256,
    )
    factor_set_sha = _factor_set_sha256(factor_rows)
    decision_sha = require_sha256(decision_source_sha256, label="decision_source_sha256")
    identity_inputs = {
        "decision_source_sha256": decision_sha,
        "factor_set_sha256": factor_set_sha,
        "source_refs": normalized_refs,
    }
    return {
        "bootstrap_evidence_id": business_identity("bootstrap-evidence", identity_inputs),
        "admission_route": BOOTSTRAP_ADMISSION_ROUTE,
        "producer_identity": NOT_CLAIMED,
        "decision_source_id": BOOTSTRAP_DECISION_SOURCE_ID,
        "decision_source_sha256": decision_sha,
        "factor_rows": factor_rows,
        "reader_contract": dict(_READER_CONTRACT),
        "source_refs": normalized_refs,
        "factor_set_sha256": factor_set_sha,
        "weight_total": "1.000000000000",
        "authorizes_readiness": False,
        "authorizes_selectability": False,
    }


def build_bootstrap_exception_evidence(
    *,
    decision_source_bytes: bytes,
    source_artifacts: Mapping[str, Mapping[str, Any] | bytes],
    implementation_source_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Seal immutable evidence that grants no readiness or activation authority."""

    if "code" not in source_artifacts:
        raise FactorGovernanceError("source artifact role code is required")
    code_artifact = validate_governance_artifact(
        source_artifacts["code"], expected_kind="system.release"
    )
    payload = _payload(
        decision_source_sha256=_decision_sha256(decision_source_bytes),
        source_refs=_source_ref_rows(source_artifacts),
        code_sha256=code_artifact["payload"]["code_sha256"],
        implementation_sha256=implementation_source_sha256,
    )
    return seal_artifact(BOOTSTRAP_EVIDENCE_KIND, payload, created_at=created_at)


def validate_bootstrap_exception_evidence(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    envelope, payload = exact_payload(
        document, kind=BOOTSTRAP_EVIDENCE_KIND, fields=_EVIDENCE_FIELDS
    )
    expected = seal_artifact(
        BOOTSTRAP_EVIDENCE_KIND,
        _payload(
            decision_source_sha256=payload["decision_source_sha256"],
            source_refs=payload["source_refs"],
            code_sha256=payload["factor_rows"][0]["code_sha256"],
            implementation_sha256=payload["factor_rows"][0]["implementation_sha256"],
        ),
        created_at=envelope["created_at"],
    )
    if expected != envelope:
        raise FactorGovernanceError("bootstrap exception evidence does not replay exactly")
    return envelope


__all__ = [
    "BOOTSTRAP_ADMISSION_ROUTE",
    "BOOTSTRAP_DECISION_SOURCE_ID",
    "BOOTSTRAP_EVIDENCE_KIND",
    "build_bootstrap_exception_evidence",
    "validate_bootstrap_exception_evidence",
]
