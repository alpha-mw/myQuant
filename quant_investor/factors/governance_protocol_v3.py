"""FactorGovernanceProtocol v3 production-readiness contract.

This module is intentionally fail closed.  It validates the current registry,
v3 evidence, factor/runtime risk budgets and the existing hash-bound activation
receipt contract.  It contains no registry mutation or bootstrap apply path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from quant_investor.factors.governance_canonical_replay_v3 import (
    EVIDENCE_SCHEMA_VERSION,
    CanonicalReplayV3Error,
    readback_v3_evidence,
    validate_v3_evidence,
)
from quant_investor.factors.runtime_contract import (
    validate_production_runtime_contracts,
    validate_quant_production_activation,
)


PROTOCOL_VERSION = "v3"
PROTOCOL_SCHEMA_VERSION = "factor-governance-protocol.v3"
MIN_MONTH_END_RANKIC_COUNT = 12
MIN_NONOVERLAP_30D_COHORT_COUNT = 8
MIN_PURGE_DAYS = 30
REQUIRED_EMBARGO_DAYS = 30
FDR_Q = 0.10
MAX_FACTOR_ABS_WEIGHT = 0.20
MAX_FAMILY_ABS_WEIGHT = 0.35
FORWARD_PRODUCTION_APPLY_ENABLED = False
FORWARD_PRODUCTION_APPLY_BLOCKER = "forward_factor_apply_not_authorized_pr4"
CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER = "factor_v3_canonical_evidence_not_verified"
CANONICAL_PRODUCER_AUTHENTICATION_BLOCKER = (
    "factor_v3_canonical_producer_not_authenticated"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def protocol_policy() -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "canonical_chain": [
            "quant",
            "deterministic_funnel",
            "bayesian",
            "risk_guard",
            "ic_coordinator",
            "portfolio_constructor",
        ],
        "likelihood_branches": ["fundamental", "quant"],
        "context_branches": ["fundamental", "macro", "quant"],
        "candidate_maturity": {
            "month_end_rankic_count": MIN_MONTH_END_RANKIC_COUNT,
            "nonoverlap_30d_cohort_count": MIN_NONOVERLAP_30D_COHORT_COUNT,
            "ninety_day_check_authoritative": False,
        },
        "multiple_testing": {"method": "benjamini_hochberg", "q": FDR_Q},
        "risk_budget": {
            "max_factor_normalized_abs_weight": MAX_FACTOR_ABS_WEIGHT,
            "max_family_normalized_abs_weight": MAX_FAMILY_ABS_WEIGHT,
        },
        "evidence_schema": EVIDENCE_SCHEMA_VERSION,
        "legacy_v2_evidence": "reject",
        "registry_mutation": {
            "enabled": False,
            "blocker": FORWARD_PRODUCTION_APPLY_BLOCKER,
        },
    }


def protocol_hash() -> str:
    return _sha256(protocol_policy())


PROTOCOL_HASH = protocol_hash()


def canonical_replay_producer_control(
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    control = {
        "producer_implemented": True,
        "producer_available": True,
        "local_bytes_readback_verified": False,
        "ic_input_output_hash_binding_verified": False,
        "canonical_producer_authenticated": False,
        "production_apply_authorized": False,
        "production_apply_eligible": False,
        "blocker": CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
    }
    if evidence is None:
        return control
    try:
        readback = readback_v3_evidence(evidence)
    except (CanonicalReplayV3Error, OSError, TypeError, ValueError):
        return control
    return {
        **control,
        "local_bytes_readback_verified": True,
        "ic_input_output_hash_binding_verified": bool(
            readback["ic_input_output_hash_binding_verified"]
        ),
        "replay_file_sha256": str(readback["replay_file_sha256"]),
        "replay_semantic_sha256": str(
            readback["replay"]["replay_semantic_sha256"]
        ),
        "replay_registry_file_sha256": str(
            readback["replay"]["registry_file_sha256"]
        ),
        "replay_production_factor_set_sha256": str(
            readback["replay"]["production_factor_set_sha256"]
        ),
        "canonical_producer_authenticated": False,
        "blocker": CANONICAL_PRODUCER_AUTHENTICATION_BLOCKER,
    }


def assess_candidate_maturity(
    *,
    month_end_rankic_dates: Sequence[str],
    forward_cohorts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    month_ends: list[str] = []
    for raw in month_end_rankic_dates:
        try:
            observed = date.fromisoformat(str(raw)).isoformat()
        except ValueError:
            continue
        month_key = observed[:7]
        if month_key not in {item[:7] for item in month_ends}:
            month_ends.append(observed)
    valid_cohorts: list[tuple[date, date, str]] = []
    for cohort in forward_cohorts:
        try:
            start = date.fromisoformat(str(cohort.get("start")))
            end = date.fromisoformat(str(cohort.get("end")))
            cohort_id = str(cohort.get("cohort_id") or "").strip()
            horizon = int(cohort.get("horizon_days", 0))
        except (TypeError, ValueError):
            continue
        if cohort_id and horizon == 30 and end > start:
            valid_cohorts.append((start, end, cohort_id))
    nonoverlap: list[tuple[date, date, str]] = []
    last_end: date | None = None
    seen: set[str] = set()
    for start, end, cohort_id in sorted(valid_cohorts):
        if cohort_id in seen or (last_end is not None and start <= last_end):
            continue
        nonoverlap.append((start, end, cohort_id))
        seen.add(cohort_id)
        last_end = end
    by_month = len(month_ends) >= MIN_MONTH_END_RANKIC_COUNT
    by_cohort = len(nonoverlap) >= MIN_NONOVERLAP_30D_COHORT_COUNT
    return {
        "mature": by_month or by_cohort,
        "maturity_route": (
            "month_end_rankic"
            if by_month
            else "nonoverlap_30d_forward_cohort"
            if by_cohort
            else "insufficient"
        ),
        "month_end_rankic_count": len(month_ends),
        "nonoverlap_30d_cohort_count": len(nonoverlap),
        "ninety_day_check_authoritative": False,
    }


def benjamini_hochberg_by_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    q: float = FDR_Q,
) -> list[dict[str, Any]]:
    if not math.isfinite(float(q)) or not 0.0 < float(q) <= 1.0:
        raise ValueError("q must be in (0, 1]")
    families: dict[str, list[dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = dict(raw)
        name = str(item.get("name") or "").strip()
        family = str(item.get("family") or "").strip()
        p_value_raw = item.get("p_value")
        if isinstance(p_value_raw, bool) or not isinstance(
            p_value_raw, (int, float)
        ):
            raise ValueError("p_value must be numeric")
        p_value = float(p_value_raw)
        if not name or not family or not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("name/family/p_value are invalid")
        item.update(
            {"name": name, "family": family, "p_value": p_value, "_input_index": index}
        )
        families.setdefault(family, []).append(item)
    for members in families.values():
        ordered = sorted(members, key=lambda item: (item["p_value"], item["name"]))
        count = len(ordered)
        last_passing = 0
        for rank, item in enumerate(ordered, start=1):
            if item["p_value"] <= q * rank / count:
                last_passing = rank
        adjusted = [1.0] * count
        running = 1.0
        for position in range(count - 1, -1, -1):
            rank = position + 1
            running = min(running, ordered[position]["p_value"] * count / rank)
            adjusted[position] = min(1.0, running)
        for rank, (item, q_value) in enumerate(zip(ordered, adjusted), start=1):
            item.update(
                {
                    "bh_family_test_count": count,
                    "bh_rank": rank,
                    "bh_q": q,
                    "bh_q_value": q_value,
                    "fdr_passed": rank <= last_passing,
                    "fdr_method": "benjamini_hochberg_by_family",
                }
            )
            output.append(item)
    output.sort(key=lambda item: item.pop("_input_index"))
    return output


def _slot_identity(record: Any) -> tuple[str, str]:
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
    return family, cluster


def governance_runtime_status(registry: Any) -> dict[str, Any]:
    """Return exact v3 readiness; legacy metadata/evidence always blocks."""

    selectable = registry.selectable_factors()
    manifest = registry.selectable_manifest()
    metadata = dict(registry.metadata or {})
    blockers: list[str] = []
    if metadata.get("missing"):
        blockers.append("registry_missing")
    if metadata.get("load_error") or metadata.get("strict_load_error"):
        blockers.append("registry_load_error")
    if metadata.get("strict_loader") is not True:
        blockers.append("registry_not_strictly_loaded")
    if not selectable:
        blockers.append("no_selectable_production_factors")
    if metadata.get("factor_governance_protocol_version") != PROTOCOL_VERSION:
        blockers.append("registry_protocol_version_mismatch")
    if metadata.get("factor_governance_protocol_hash") != protocol_hash():
        blockers.append("registry_protocol_hash_mismatch")
    for key in (
        "production_factor_count",
        "production_factor_names",
        "production_factor_set_sha256",
    ):
        if metadata.get(key) != manifest[key]:
            blockers.append(f"registry_{key}_mismatch")

    registry_file_sha = str(metadata.get("registry_sha256") or "")
    evidence_value = metadata.get("factor_governance_v3_evidence")
    producer_control = canonical_replay_producer_control(
        evidence_value if isinstance(evidence_value, Mapping) else None
    )
    if not isinstance(evidence_value, Mapping):
        blockers.append("registry_v3_evidence_missing")
    else:
        try:
            evidence = validate_v3_evidence(evidence_value)
            if evidence["registry_file_sha256"] != registry_file_sha:
                blockers.append("registry_v3_evidence_registry_sha_mismatch")
        except (CanonicalReplayV3Error, TypeError, ValueError):
            blockers.append("registry_v3_evidence_invalid")
    if producer_control["local_bytes_readback_verified"]:
        if producer_control.get("replay_registry_file_sha256") != registry_file_sha:
            blockers.append("registry_v3_replay_registry_sha_mismatch")
        if (
            producer_control.get("replay_production_factor_set_sha256")
            != manifest["production_factor_set_sha256"]
        ):
            blockers.append("registry_v3_replay_factor_set_sha_mismatch")
    else:
        blockers.append("canonical_evidence_not_readback_bound")
    if not producer_control["ic_input_output_hash_binding_verified"]:
        blockers.append("canonical_ic_input_output_not_readback_bound")
    if not producer_control["canonical_producer_authenticated"]:
        blockers.append("canonical_producer_not_authenticated")
    if not producer_control["production_apply_authorized"]:
        blockers.append(FORWARD_PRODUCTION_APPLY_BLOCKER)

    total = 0.0
    numeric_weights: dict[str, float] = {}
    families: dict[str, str] = {}
    slots: dict[str, list[str]] = {}
    for record in selectable:
        try:
            weight = abs(float(record.weight))
        except (TypeError, ValueError):
            weight = math.nan
        if not math.isfinite(weight) or weight <= 1e-15:
            blockers.append(f"factor_weight_invalid:{record.name}")
            continue
        family, cluster = _slot_identity(record)
        if not family or not cluster:
            blockers.append(f"factor_slot_identity_missing:{record.name}")
        else:
            slots.setdefault(f"{family}::{cluster}", []).append(record.name)
            families[record.name] = family
        numeric_weights[record.name] = weight
        total += weight
    if total <= 1e-15:
        blockers.append("production_factor_total_abs_weight_zero")
    normalized = {
        name: value / total for name, value in numeric_weights.items()
    } if total > 1e-15 else {}
    family_weights: dict[str, float] = {}
    for name, weight in normalized.items():
        if weight > MAX_FACTOR_ABS_WEIGHT + 1e-12:
            blockers.append(f"factor_abs_weight_above_0.20:{name}")
        factor_family = families.get(name)
        if factor_family:
            family_weights[factor_family] = (
                family_weights.get(factor_family, 0.0) + weight
            )
    for slot, names in slots.items():
        if len(names) != 1:
            blockers.append(f"factor_slot_multiple_incumbents:{slot}")
    for family, weight in family_weights.items():
        if weight > MAX_FAMILY_ABS_WEIGHT + 1e-12:
            blockers.append(f"family_abs_weight_above_0.35:{family}")

    runtime_contract_status = validate_production_runtime_contracts(selectable, metadata)
    blockers.extend(runtime_contract_status.get("blockers", []))
    activation = validate_quant_production_activation(
        metadata,
        manifest,
        str(runtime_contract_status.get("contracts_sha256") or ""),
        implementation_code_sha256s=dict(
            runtime_contract_status.get("implementation_code_sha256s", {}) or {}
        ),
        protocol_version=PROTOCOL_VERSION,
        protocol_hash_value=protocol_hash(),
    )
    blockers.extend(activation.get("blockers", []))
    blockers = list(dict.fromkeys(blockers))
    ready = not blockers
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "status": "ready" if ready else "governance_blocked",
        "factor_mode": "governed_mined_factors" if ready else "governance_blocked",
        "confidence_multiplier": 1.0 if ready else 0.0,
        "production_eligible": ready,
        "legacy_fallback_allowed": False,
        "production_factor_count": len(selectable),
        "production_factor_names": manifest["production_factor_names"],
        "registry_file_sha256": registry_file_sha,
        "production_factor_set_sha256": manifest["production_factor_set_sha256"],
        "normalized_abs_weights": normalized,
        "family_normalized_abs_weights": family_weights,
        "slot_incumbents": slots,
        "canonical_replay_producer_control": producer_control,
        "factor_runtime_contracts": dict(runtime_contract_status.get("contracts", {}) or {}),
        "factor_runtime_contracts_sha256": str(
            runtime_contract_status.get("contracts_sha256") or ""
        ),
        "factor_runtime_implementation_code_sha256s": dict(
            runtime_contract_status.get("implementation_code_sha256s", {}) or {}
        ),
        "quant_production_activation": dict(activation),
        "blockers": blockers,
    }


__all__ = [
    "CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER",
    "CANONICAL_PRODUCER_AUTHENTICATION_BLOCKER",
    "FDR_Q",
    "FORWARD_PRODUCTION_APPLY_BLOCKER",
    "MAX_FACTOR_ABS_WEIGHT",
    "MAX_FAMILY_ABS_WEIGHT",
    "MIN_MONTH_END_RANKIC_COUNT",
    "MIN_NONOVERLAP_30D_COHORT_COUNT",
    "PROTOCOL_HASH",
    "PROTOCOL_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "assess_candidate_maturity",
    "benjamini_hochberg_by_family",
    "canonical_replay_producer_control",
    "governance_runtime_status",
    "protocol_hash",
    "protocol_policy",
]
