"""Immutable, nonauthorizing Factor-v4 production-set evidence carrier.

The carrier binds exact legacy registry bytes without extending or mutating the
``mined-factor-registry.v1`` schema.  It deliberately does not infer v4 family,
slot, maturity, health, or receipt validity from legacy metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors.governance_canonical_replay_v4 import (
    REPLAY_SCHEMA_VERSION,
    validate_canonical_replay_v4,
)
from quant_investor.factors.governance_protocol_v4 import (
    PROTOCOL_VERSION,
    protocol_hash,
)
from quant_investor.factors.governance_transaction_v4 import (
    ACTIVATION_RECEIPT_SCHEMA_VERSION,
)
from quant_investor.factors.registry_store import (
    FactorRegistryStoreError,
    parse_registry_snapshot_bytes_strict,
    registry_payload_semantic_sha256,
)
from quant_investor.factors.runtime import production_factor_set_sha256
from quant_investor.factors.evidence_contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    encode_f64,
    seal_semantic,
    semantic_sha256,
    sha256_bytes,
    validate_semantic_seal,
)

FACTOR_PRODUCTION_SET_CARRIER_SCHEMA = "factor-production-set-carrier.v4.foundation"
FACTOR_PRODUCTION_SET_READBACK_SCHEMA = (
    "factor-production-set-carrier-readback.v4.foundation"
)
LEGACY_REGISTRY_SCHEMA = "mined-factor-registry.v1"
PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"
GOVERNED_ROOT_POLICY = "v16.governed-data-root.v2"
MIN_PRODUCTION_FACTORS = 5
MAX_PRODUCTION_FACTORS = 10


class FactorProductionSetCarrierV4Error(ValueError):
    """Raised when the immutable Factor-v4 carrier fails closed."""


def _exact(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FactorProductionSetCarrierV4Error(f"{label} fields mismatch")
    return dict(value)


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        normalized = date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise FactorProductionSetCarrierV4Error(f"{label} must be ISO date") from exc
    if normalized != text:
        raise FactorProductionSetCarrierV4Error(f"{label} must be canonical ISO date")
    return text


def _refs(
    values: Sequence[EvidenceRef],
    *,
    schema: str,
    label: str,
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorProductionSetCarrierV4Error(f"{label} must be a sequence")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        if not isinstance(value, EvidenceRef):
            raise FactorProductionSetCarrierV4Error(f"{label} must contain EvidenceRef")
        if value.artifact_schema != schema or value.root_policy != PRIVATE_ROOT_POLICY:
            raise FactorProductionSetCarrierV4Error(f"{label} schema/root policy mismatch")
        identity = tuple(value.to_dict()[key] for key in sorted(value.to_dict()))
        if identity in seen:
            raise FactorProductionSetCarrierV4Error(f"{label} contains duplicate refs")
        seen.add(identity)
        normalized.append(value.to_dict())
    return normalized


def build_factor_production_set_carrier_v4(
    *,
    as_of: str,
    legacy_registry_ref: EvidenceRef,
    canonical_replay_refs: Sequence[EvidenceRef] = (),
    activation_receipt_ref: EvidenceRef | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(legacy_registry_ref, EvidenceRef)
        or legacy_registry_ref.artifact_schema != LEGACY_REGISTRY_SCHEMA
        or legacy_registry_ref.root_policy != GOVERNED_ROOT_POLICY
    ):
        raise FactorProductionSetCarrierV4Error(
            "legacy registry ref schema/root policy mismatch"
        )
    replay_refs = _refs(
        canonical_replay_refs,
        schema=REPLAY_SCHEMA_VERSION,
        label="canonical_replay_refs",
    )
    if activation_receipt_ref is not None and (
        not isinstance(activation_receipt_ref, EvidenceRef)
        or activation_receipt_ref.artifact_schema != ACTIVATION_RECEIPT_SCHEMA_VERSION
        or activation_receipt_ref.root_policy != PRIVATE_ROOT_POLICY
    ):
        raise FactorProductionSetCarrierV4Error(
            "activation receipt ref schema/root policy mismatch"
        )
    return seal_semantic(
        {
            "schema_version": FACTOR_PRODUCTION_SET_CARRIER_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "as_of": _iso_date(as_of, label="as_of"),
            "legacy_registry_ref": legacy_registry_ref.to_dict(),
            "canonical_replay_refs": replay_refs,
            "activation_receipt_ref": (
                None
                if activation_receipt_ref is None
                else activation_receipt_ref.to_dict()
            ),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_factor_production_set_carrier_v4(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = validate_semantic_seal(value)
    except EvidenceV2Error as exc:
        raise FactorProductionSetCarrierV4Error(str(exc)) from exc
    payload = _exact(
        payload,
        {
            "schema_version",
            "protocol_version",
            "protocol_hash",
            "as_of",
            "legacy_registry_ref",
            "canonical_replay_refs",
            "activation_receipt_ref",
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
            "semantic_sha256",
        },
        label="Factor production-set carrier",
    )
    if (
        payload["schema_version"] != FACTOR_PRODUCTION_SET_CARRIER_SCHEMA
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["protocol_hash"] != protocol_hash()
    ):
        raise FactorProductionSetCarrierV4Error("Factor carrier identity mismatch")
    _iso_date(payload["as_of"], label="as_of")
    registry_ref = EvidenceRef.from_dict(payload["legacy_registry_ref"])
    if (
        registry_ref.artifact_schema != LEGACY_REGISTRY_SCHEMA
        or registry_ref.root_policy != GOVERNED_ROOT_POLICY
    ):
        raise FactorProductionSetCarrierV4Error("legacy registry ref is invalid")
    replay_values = payload["canonical_replay_refs"]
    if not isinstance(replay_values, list):
        raise FactorProductionSetCarrierV4Error("canonical_replay_refs must be a list")
    replay_refs = [EvidenceRef.from_dict(item) for item in replay_values]
    if _refs(
        replay_refs,
        schema=REPLAY_SCHEMA_VERSION,
        label="canonical_replay_refs",
    ) != replay_values:
        raise FactorProductionSetCarrierV4Error("canonical replay refs are not canonical")
    receipt_value = payload["activation_receipt_ref"]
    if receipt_value is not None:
        receipt_ref = EvidenceRef.from_dict(receipt_value)
        if (
            receipt_ref.artifact_schema != ACTIVATION_RECEIPT_SCHEMA_VERSION
            or receipt_ref.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise FactorProductionSetCarrierV4Error("activation receipt ref is invalid")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise FactorProductionSetCarrierV4Error(
            "Factor production-set carrier must be nonauthorizing"
        )
    return payload


def bind_factor_production_set_carrier_v4(
    value: Mapping[str, Any],
    *,
    absolute_path: str,
) -> BoundCanonicalArtifact:
    payload = validate_factor_production_set_carrier_v4(value)
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=FACTOR_PRODUCTION_SET_CARRIER_SCHEMA,
            absolute_path=absolute_path,
            byte_sha256=sha256_bytes(raw),
            semantic_sha256=semantic_sha256(payload),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_domain_json(artifact: BoundRawArtifact, *, label: str) -> dict[str, Any]:
    try:
        text = artifact.payload.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise FactorProductionSetCarrierV4Error(f"{label}: {exc}") from exc
    if canonical != artifact.payload:
        raise FactorProductionSetCarrierV4Error(f"{label} bytes are not canonical JSON")
    if not isinstance(value, Mapping):
        raise FactorProductionSetCarrierV4Error(f"{label} must contain an object")
    payload = dict(value)
    if payload.get("schema_version") != artifact.reference.artifact_schema:
        raise FactorProductionSetCarrierV4Error(f"{label} schema mismatch")
    if hashlib.sha256(canonical).hexdigest() != artifact.reference.semantic_sha256:
        raise FactorProductionSetCarrierV4Error(f"{label} semantic SHA mismatch")
    return payload


@dataclass(frozen=True)
class FactorProductionSetEvidenceBundleV4:
    carrier: BoundCanonicalArtifact
    legacy_registry: BoundRawArtifact
    canonical_replays: tuple[BoundRawArtifact, ...] = ()
    activation_receipt: BoundRawArtifact | None = None

    def read(self) -> dict[str, Any]:
        if not isinstance(self.carrier, BoundCanonicalArtifact) or not isinstance(
            self.legacy_registry, BoundRawArtifact
        ):
            raise FactorProductionSetCarrierV4Error(
                "Factor production-set bundle fields have invalid types"
            )
        carrier = validate_factor_production_set_carrier_v4(self.carrier.read())
        if carrier["legacy_registry_ref"] != self.legacy_registry.reference.to_dict():
            raise FactorProductionSetCarrierV4Error("legacy registry ref drift")
        if (
            self.legacy_registry.reference.artifact_schema != LEGACY_REGISTRY_SCHEMA
            or self.legacy_registry.reference.root_policy != GOVERNED_ROOT_POLICY
        ):
            raise FactorProductionSetCarrierV4Error("legacy registry artifact is invalid")
        try:
            snapshot = parse_registry_snapshot_bytes_strict(
                self.legacy_registry.payload,
                source=self.legacy_registry.reference.absolute_path,
            )
        except FactorRegistryStoreError as exc:
            raise FactorProductionSetCarrierV4Error(str(exc)) from exc
        if snapshot.registry_sha256 != self.legacy_registry.reference.byte_sha256:
            raise FactorProductionSetCarrierV4Error("legacy registry byte SHA drift")
        if (
            registry_payload_semantic_sha256(snapshot.payload)
            != self.legacy_registry.reference.semantic_sha256
        ):
            raise FactorProductionSetCarrierV4Error("legacy registry semantic SHA drift")

        expected_replay_refs = carrier["canonical_replay_refs"]
        actual_replay_refs = [item.reference.to_dict() for item in self.canonical_replays]
        if actual_replay_refs != expected_replay_refs:
            raise FactorProductionSetCarrierV4Error("canonical replay ref inventory drift")
        replay_payloads: list[dict[str, Any]] = []
        for artifact in self.canonical_replays:
            if (
                not isinstance(artifact, BoundRawArtifact)
                or artifact.reference.artifact_schema != REPLAY_SCHEMA_VERSION
                or artifact.reference.root_policy != PRIVATE_ROOT_POLICY
            ):
                raise FactorProductionSetCarrierV4Error("canonical replay artifact is invalid")
            try:
                replay = validate_canonical_replay_v4(
                    _read_domain_json(artifact, label="canonical replay")
                )
            except (TypeError, ValueError) as exc:
                raise FactorProductionSetCarrierV4Error(str(exc)) from exc
            replay_payloads.append(replay)

        expected_receipt_ref = carrier["activation_receipt_ref"]
        actual_receipt_ref = (
            None
            if self.activation_receipt is None
            else self.activation_receipt.reference.to_dict()
        )
        if actual_receipt_ref != expected_receipt_ref:
            raise FactorProductionSetCarrierV4Error("activation receipt ref drift")
        if self.activation_receipt is not None:
            if (
                not isinstance(self.activation_receipt, BoundRawArtifact)
                or self.activation_receipt.reference.artifact_schema
                != ACTIVATION_RECEIPT_SCHEMA_VERSION
                or self.activation_receipt.reference.root_policy != PRIVATE_ROOT_POLICY
            ):
                raise FactorProductionSetCarrierV4Error("activation receipt artifact is invalid")
            _read_domain_json(self.activation_receipt, label="activation receipt")

        production_rows = sorted(
            (
                record
                for record in snapshot.record_payloads.values()
                if record.get("state") == "production_factor"
            ),
            key=lambda item: str(item.get("name") or ""),
        )
        names = [str(item["name"]) for item in production_rows]
        set_sha = production_factor_set_sha256(names)
        metadata = snapshot.metadata_payload
        blockers: list[str] = []
        registry_weights: dict[str, float] = {}
        for item in production_rows:
            name = str(item["name"])
            raw_weight = item.get("weight")
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or abs(float(raw_weight)) <= 1e-15
            ):
                blockers.append(f"legacy_registry_production_weight_invalid:{name}")
                continue
            registry_weights[name] = abs(float(raw_weight))
        total_weight = sum(registry_weights.values())
        normalized_weights = (
            {
                name: weight / total_weight
                for name, weight in sorted(registry_weights.items())
            }
            if total_weight > 1e-15
            else {}
        )
        for name, weight in normalized_weights.items():
            if weight > 0.20 + 1e-12:
                blockers.append(f"factor_abs_weight_above_0.20:{name}")
        if metadata.get("production_factor_count") != len(names):
            blockers.append("legacy_registry_metadata_production_factor_count_mismatch")
        if metadata.get("production_factor_names") != names:
            blockers.append("legacy_registry_metadata_production_factor_names_mismatch")
        if metadata.get("production_factor_set_sha256") != set_sha:
            blockers.append("legacy_registry_metadata_production_factor_set_sha256_mismatch")
        if len(names) < MIN_PRODUCTION_FACTORS:
            blockers.append("production_factor_count_below_5")
        if len(names) > MAX_PRODUCTION_FACTORS:
            blockers.append("production_factor_count_above_target_10")
        if not replay_payloads:
            blockers.append("factor_v4_canonical_replay_evidence_missing")
        else:
            current_replays = [
                item
                for item in replay_payloads
                if item["registry_file_sha256"] == snapshot.registry_sha256
                and item["production_factor_set_sha256"] == set_sha
            ]
            if len(current_replays) != len(replay_payloads):
                blockers.append("factor_v4_canonical_replay_current_set_binding_mismatch")
        blockers.extend(
            (
                "factor_v4_authoritative_record_projection_not_integrated",
                "factor_v4_family_and_slot_evidence_not_integrated",
                "factor_v4_fresh_health_evidence_not_integrated",
                "factor_v4_maturity_and_bh_evidence_not_integrated",
                "factor_v4_activation_receipt_not_validated",
            )
        )
        readback = seal_semantic(
            {
                "schema_version": FACTOR_PRODUCTION_SET_READBACK_SCHEMA,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(),
                "as_of": carrier["as_of"],
                "carrier_ref": self.carrier.reference.to_dict(),
                "legacy_registry_ref": self.legacy_registry.reference.to_dict(),
                "legacy_registry_metadata_sha256": snapshot.metadata_sha256,
                "production_factor_names": names,
                "production_factor_count": len(names),
                "production_factor_set_sha256": set_sha,
                "registry_record_sha256s": {
                    name: snapshot.record_sha256s[name] for name in names
                },
                "registry_abs_weights": {
                    name: encode_f64(weight)
                    for name, weight in sorted(registry_weights.items())
                },
                "normalized_abs_weights": {
                    name: encode_f64(weight)
                    for name, weight in sorted(normalized_weights.items())
                },
                "canonical_replay_refs": expected_replay_refs,
                "activation_receipt_ref": expected_receipt_ref,
                "production_family_count": None,
                "healthy_factor_count": 0,
                "factor_governance_ready": False,
                "new_risk_eligible": False,
                "activation_candidate": False,
                "new_risk_authorized": False,
                "production_apply_enabled": False,
                "blockers": sorted(set(blockers)),
            }
        )
        return readback


__all__ = [
    "FACTOR_PRODUCTION_SET_CARRIER_SCHEMA",
    "FACTOR_PRODUCTION_SET_READBACK_SCHEMA",
    "FactorProductionSetCarrierV4Error",
    "FactorProductionSetEvidenceBundleV4",
    "bind_factor_production_set_carrier_v4",
    "build_factor_production_set_carrier_v4",
    "validate_factor_production_set_carrier_v4",
]
