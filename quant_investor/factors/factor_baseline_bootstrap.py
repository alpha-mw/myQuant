"""Build a deterministic, staging-only Factor v3 baseline plan.

The builder has no write/apply mode.  It validates an explicit candidate
manifest, computes bounded normalized weights, and asks the registry store for
its existing dry-run CAS/inverse-patch manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quant_investor.factors.governance_canonical_replay_v3 import (
    EVIDENCE_SCHEMA_VERSION,
    semantic_sha256,
    validate_v3_evidence,
)
from quant_investor.factors.governance_protocol_v3 import (
    MAX_FACTOR_ABS_WEIGHT,
    MAX_FAMILY_ABS_WEIGHT,
    PROTOCOL_VERSION,
    protocol_hash,
)
from quant_investor.factors.registry_store import (
    apply_factor_record_patch,
    load_registry_snapshot_strict,
)
from quant_investor.factors.runtime import production_factor_set_sha256


BOOTSTRAP_CANDIDATE_SCHEMA_VERSION = "factor-baseline-bootstrap-candidates.v1"
BOOTSTRAP_PLAN_SCHEMA_VERSION = "factor-baseline-bootstrap-plan.v1"
MIN_BOOTSTRAP_FACTORS = 5
MIN_BOOTSTRAP_FAMILIES = 3


class FactorBaselineBootstrapError(ValueError):
    """Raised when a bootstrap plan cannot be built safely."""


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise FactorBaselineBootstrapError(f"{label} must contain exact fields")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorBaselineBootstrapError(f"{label} must be an exact non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise FactorBaselineBootstrapError(f"{label} must be lowercase SHA-256")
    return text


def validate_bootstrap_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate candidate diversity/slots and return normalized signed weights."""

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise FactorBaselineBootstrapError("candidates must be a sequence")
    if len(candidates) < MIN_BOOTSTRAP_FACTORS:
        raise FactorBaselineBootstrapError("bootstrap requires at least 5 factors")
    fields = {
        "name",
        "family",
        "slot",
        "direction",
        "raw_weight",
        "registry_record_sha256",
        "runtime_contract",
        "evidence",
    }
    normalized_rows: list[dict[str, Any]] = []
    names: set[str] = set()
    slots: set[str] = set()
    for index, raw in enumerate(candidates):
        row = _exact(dict(raw), fields, f"candidate[{index}]")
        name = _text(row["name"], "candidate name")
        family = _text(row["family"], "candidate family")
        slot = _text(row["slot"], "candidate slot")
        if name in names:
            raise FactorBaselineBootstrapError("candidate names must be unique")
        if slot in slots:
            raise FactorBaselineBootstrapError("candidate slots must be unique")
        names.add(name)
        slots.add(slot)
        if isinstance(row["direction"], bool) or not isinstance(
            row["direction"], (int, float)
        ):
            raise FactorBaselineBootstrapError("candidate direction must be numeric")
        direction = float(row["direction"])
        if not math.isfinite(direction) or abs(direction) <= 1e-15:
            raise FactorBaselineBootstrapError("candidate direction must be finite non-zero")
        if isinstance(row["raw_weight"], bool) or not isinstance(
            row["raw_weight"], (int, float)
        ):
            raise FactorBaselineBootstrapError("candidate raw_weight must be numeric")
        raw_weight = abs(float(row["raw_weight"]))
        if not math.isfinite(raw_weight) or raw_weight <= 1e-15:
            raise FactorBaselineBootstrapError("candidate raw_weight must be positive")
        record_sha = _sha(row["registry_record_sha256"], "registry record SHA")
        contract = row["runtime_contract"]
        if not isinstance(contract, Mapping):
            raise FactorBaselineBootstrapError("runtime contract must be an object")
        contract = copy.deepcopy(dict(contract))
        if (
            contract.get("schema_version") != "factor-production-runtime-contract.v1"
            or contract.get("factor_name") != name
        ):
            raise FactorBaselineBootstrapError("runtime contract identity mismatch")
        contract_sha = semantic_sha256(contract)
        try:
            evidence = validate_v3_evidence(dict(row["evidence"]))
        except (TypeError, ValueError) as exc:
            raise FactorBaselineBootstrapError(str(exc)) from exc
        if evidence["factor_name"] != name:
            raise FactorBaselineBootstrapError("evidence factor identity mismatch")
        if evidence["runtime_contract_sha256"] != contract_sha:
            raise FactorBaselineBootstrapError("evidence/runtime contract SHA mismatch")
        normalized_rows.append(
            {
                "name": name,
                "family": family,
                "slot": slot,
                "direction": direction,
                "raw_weight": raw_weight,
                "registry_record_sha256": record_sha,
                "runtime_contract": contract,
                "runtime_contract_sha256": contract_sha,
                "evidence": evidence,
            }
        )
    families = {row["family"] for row in normalized_rows}
    if len(families) < MIN_BOOTSTRAP_FAMILIES:
        raise FactorBaselineBootstrapError("bootstrap requires at least 3 families")
    total = sum(row["raw_weight"] for row in normalized_rows)
    family_totals: dict[str, float] = {}
    signed_weights: dict[str, float] = {}
    absolute_weights: dict[str, float] = {}
    for row in normalized_rows:
        weight = row["raw_weight"] / total
        absolute_weights[row["name"]] = weight
        signed_weights[row["name"]] = weight * (1.0 if row["direction"] > 0 else -1.0)
        family_totals[row["family"]] = family_totals.get(row["family"], 0.0) + weight
        if weight > MAX_FACTOR_ABS_WEIGHT + 1e-12:
            raise FactorBaselineBootstrapError("factor normalized weight exceeds 20%")
    if any(weight > MAX_FAMILY_ABS_WEIGHT + 1e-12 for weight in family_totals.values()):
        raise FactorBaselineBootstrapError("family normalized weight exceeds 35%")
    return {
        "candidates": sorted(normalized_rows, key=lambda row: row["name"]),
        "absolute_weights": dict(sorted(absolute_weights.items())),
        "signed_weights": dict(sorted(signed_weights.items())),
        "family_absolute_weights": dict(sorted(family_totals.items())),
        "factor_count": len(normalized_rows),
        "family_count": len(families),
    }


def build_factor_baseline_bootstrap_plan(
    *,
    registry_path: str | Path,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic dry-run plan without changing registry bytes."""

    manifest = _exact(
        dict(candidate_manifest),
        {
            "schema_version",
            "as_of",
            "expected_registry_sha256",
            "calendar_sha256",
            "pit_sha256",
            "candidates",
        },
        "candidate manifest",
    )
    if manifest["schema_version"] != BOOTSTRAP_CANDIDATE_SCHEMA_VERSION:
        raise FactorBaselineBootstrapError("unsupported bootstrap candidate schema")
    as_of = _text(manifest["as_of"], "as_of")
    expected_registry_sha = _sha(
        manifest["expected_registry_sha256"], "expected_registry_sha256"
    )
    calendar_sha = _sha(manifest["calendar_sha256"], "calendar_sha256")
    pit_sha = _sha(manifest["pit_sha256"], "pit_sha256")
    snapshot = load_registry_snapshot_strict(registry_path)
    if snapshot.registry_sha256 != expected_registry_sha:
        raise FactorBaselineBootstrapError("expected registry SHA CAS mismatch")
    validated = validate_bootstrap_candidates(manifest["candidates"])
    records_by_name = {record.name: record for record in snapshot.registry.factors}
    selected_names = set(validated["absolute_weights"])
    for row in validated["candidates"]:
        record = records_by_name.get(row["name"])
        if record is None:
            raise FactorBaselineBootstrapError(f"candidate absent from registry: {row['name']}")
        if snapshot.record_sha256s.get(row["name"]) != row["registry_record_sha256"]:
            raise FactorBaselineBootstrapError("candidate registry record SHA mismatch")
        metadata = dict(record.metadata or {})
        family = str(
            metadata.get("factor_family")
            or metadata.get("governance_family")
            or record.category
            or ""
        ).strip()
        cluster = str(metadata.get("dominant_primitive_cluster") or "").strip()
        if not cluster:
            dominant = metadata.get("dominant_primitives", []) or []
            if isinstance(dominant, list):
                cluster = "+".join(sorted(str(item) for item in dominant if str(item)))
        if family != row["family"] or f"{family}::{cluster}" != row["slot"]:
            raise FactorBaselineBootstrapError("candidate family/slot registry mismatch")
        if not math.isclose(float(record.direction), row["direction"], rel_tol=0.0, abs_tol=1e-12):
            raise FactorBaselineBootstrapError("candidate direction registry mismatch")
        evidence = row["evidence"]
        if evidence["registry_file_sha256"] != expected_registry_sha:
            raise FactorBaselineBootstrapError("candidate evidence registry SHA mismatch")
        if evidence["calendar_sha256"] != calendar_sha or evidence["pit_sha256"] != pit_sha:
            raise FactorBaselineBootstrapError("candidate PIT/calendar evidence mismatch")

    patches: dict[str, dict[str, Any]] = {}
    expected_record_sha256s: dict[str, str | None] = {}
    for record in snapshot.registry.selectable_factors():
        if record.name not in selected_names:
            payload = record.to_dict()
            payload["state"] = "reduced"
            payload["weight"] = 0.0
            patches[record.name] = payload
            expected_record_sha256s[record.name] = snapshot.record_sha256s[record.name]
    for row in validated["candidates"]:
        payload = records_by_name[row["name"]].to_dict()
        payload["state"] = "production_factor"
        payload["weight"] = validated["absolute_weights"][row["name"]]
        patches[row["name"]] = payload
        expected_record_sha256s[row["name"]] = snapshot.record_sha256s[row["name"]]

    manifest_semantic_sha = semantic_sha256(manifest)
    dry_run = apply_factor_record_patch(
        registry_path,
        patches,
        expected_registry_sha256=expected_registry_sha,
        expected_record_sha256s=expected_record_sha256s,
        mutation_id=f"factor-v3-bootstrap:{as_of}:{manifest_semantic_sha[:12]}",
        reason="Factor v3 baseline bootstrap plan only",
        write=False,
    )
    production_names = sorted(selected_names)
    required_metadata_updates = {
        "factor_governance_protocol_version": PROTOCOL_VERSION,
        "factor_governance_protocol_hash": protocol_hash(),
        "factor_governance_evidence_schema": EVIDENCE_SCHEMA_VERSION,
        "production_factor_count": len(production_names),
        "production_factor_names": production_names,
        "production_factor_set_sha256": production_factor_set_sha256(production_names),
        "production_factor_runtime_contracts": {
            row["name"]: row["runtime_contract"] for row in validated["candidates"]
        },
    }
    plan = {
        "schema_version": BOOTSTRAP_PLAN_SCHEMA_VERSION,
        "status": "ready_plan_only",
        "apply_authorized": False,
        "registry_mutation_performed": False,
        "as_of": as_of,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "registry_path": str(Path(registry_path).expanduser().resolve()),
        "registry_file_sha256": snapshot.registry_sha256,
        "production_factor_set_sha256": production_factor_set_sha256(production_names),
        "candidate_manifest_semantic_sha256": manifest_semantic_sha,
        "calendar_sha256": calendar_sha,
        "pit_sha256": pit_sha,
        "absolute_weights": validated["absolute_weights"],
        "signed_weights": validated["signed_weights"],
        "family_absolute_weights": validated["family_absolute_weights"],
        "slots": {row["name"]: row["slot"] for row in validated["candidates"]},
        "evidence_sha256s": {
            row["name"]: row["evidence"]["replay_semantic_sha256"]
            for row in validated["candidates"]
        },
        "runtime_contract_sha256s": {
            row["name"]: row["runtime_contract_sha256"]
            for row in validated["candidates"]
        },
        "required_metadata_updates": required_metadata_updates,
        "cas_wal_inverse_dry_run": dry_run,
    }
    plan["plan_sha256"] = semantic_sha256(plan)
    return plan


__all__ = [
    "BOOTSTRAP_CANDIDATE_SCHEMA_VERSION",
    "BOOTSTRAP_PLAN_SCHEMA_VERSION",
    "FactorBaselineBootstrapError",
    "MIN_BOOTSTRAP_FACTORS",
    "MIN_BOOTSTRAP_FAMILIES",
    "build_factor_baseline_bootstrap_plan",
    "validate_bootstrap_candidates",
]
