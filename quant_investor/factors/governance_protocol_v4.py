"""Pure-offline FactorGovernanceProtocol v4 readiness policy.

The v4 module is deliberately isolated from the current v2/v3 runtime.  It
validates proposed v4 records and receipts, but it cannot mutate a registry,
activate a factor set, or authorize new risk.  Those side effects remain the
responsibility of a separately authorized production workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

PROTOCOL_VERSION = "v4"
PROTOCOL_SCHEMA_VERSION = "factor-governance-protocol.v4"
READINESS_SCHEMA_VERSION = "factor-governance-readiness.v4"
FACTOR_EVIDENCE_SCHEMA_VERSION = "factor-governance-replay-evidence.v4"
OPEN_SESSION_CALENDAR_SCHEMA_VERSION = "factor-governance-open-session-calendar.v4"
OPEN_SESSION_CALENDAR_SOURCE = "strict_parquet_observed_trade_dates"
TARGET_PRODUCTION_FACTOR_COUNT = 10
MIN_NEW_RISK_FACTOR_COUNT = 5
MIN_NEW_RISK_FAMILY_COUNT = 3
MIN_MONTH_END_RANKIC_COUNT = 12
MIN_NONOVERLAP_30D_COHORT_COUNT = 8
REQUIRED_GATE_IDS = tuple(range(1, 9))
FDR_Q = 0.10
MAX_FACTOR_ABS_WEIGHT = 0.20
MAX_FAMILY_ABS_WEIGHT = 0.35
MAX_MONTH_END_PROPOSALS = 2
PRODUCTION_APPLY_ENABLED = False
PRODUCTION_APPLY_BLOCKER = "factor_v4_production_apply_not_authorized"

CONTROL_CHAIN_STAGES = (
    "eligibility",
    "quant",
    "funnel",
    "codex_s1",
    "bayesian",
    "risk_advisor",
    "codex_ic",
    "portfolio_constructor",
)


class FactorGovernanceV4Error(ValueError):
    """Raised when a v4 planning/readiness payload is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceV4Error(f"value is not canonical JSON: {exc}") from exc


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def protocol_policy() -> dict[str, Any]:
    """Return the immutable v4 policy that all v4 plans bind."""

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "production_factor_target": TARGET_PRODUCTION_FACTOR_COUNT,
        "minimum_new_risk_baseline": {
            "factor_count": MIN_NEW_RISK_FACTOR_COUNT,
            "family_count": MIN_NEW_RISK_FAMILY_COUNT,
            "exactly_five_factor_consequence": "five_distinct_families_required",
        },
        "underfilled": {
            "factor_count_min": MIN_NEW_RISK_FACTOR_COUNT,
            "factor_count_max": TARGET_PRODUCTION_FACTOR_COUNT - 1,
            "accelerated_mining_required": True,
        },
        "healthy_factor": {
            "state": "production_factor",
            "positive_weight_required": True,
            "required_gate_ids": list(REQUIRED_GATE_IDS),
            "month_end_rankic_count": MIN_MONTH_END_RANKIC_COUNT,
            "nonoverlap_30d_cohort_count": MIN_NONOVERLAP_30D_COHORT_COUNT,
            "maturity_calendar": {
                "schema_version": OPEN_SESSION_CALENDAR_SCHEMA_VERSION,
                "source": OPEN_SESSION_CALENDAR_SOURCE,
                "strict_pointer_manifest_binding_required": True,
                "cohort_open_session_count": 30,
            },
            "multiple_testing": {
                "method": "benjamini_hochberg_by_family",
                "q_lte": FDR_Q,
            },
            "unique_slot_required": True,
            "runtime_contract_required": True,
            "fresh_health_required": True,
            "activation_receipt_required": True,
        },
        "candidate_admission": {
            "strict_parquet_source_receipt_required": True,
            "full_a_universe_required": True,
            "screening_exact_readback_required": {
                "schema_version": "factor-governance-screening-readback.v4",
                "artifacts": [
                    "factor-primitive-ontology.v4",
                    "factor-candidate-catalog.v4",
                    "factor-screening-evidence.v4",
                ],
                "candidate_row": "exactly_one_bh_pass_row",
                "caller_bh_fields_authoritative": False,
            },
            "dedup_exact_readback_required": {
                "schema_version": "factor-candidate-dedup-readback.v4",
                "evidence_schema": "factor-candidate-dedup-evidence.v4",
                "metric": "median_monthly_cross_sectional_abs_spearman",
                "threshold": 0.70,
                "duplicate_primitive_allowed": False,
                "caller_dedup_booleans_authoritative": False,
            },
            "canonical_replay_hash_bindings": [
                "market_data_input_sha256",
                "candidate_catalog_sha256",
                "screening_evidence_sha256",
                "dedup_evidence_sha256",
                "quantitative_evidence_sha256",
            ],
            "initial_weight": 0.0,
            "maturity_required": True,
            "calendar_verified_maturity_required": True,
            "bh_q_lte": FDR_Q,
            "required_gate_ids": list(REQUIRED_GATE_IDS),
            "registry_write_enabled": False,
        },
        "risk_budget": {
            "max_factor_normalized_abs_weight": MAX_FACTOR_ABS_WEIGHT,
            "max_family_normalized_abs_weight": MAX_FAMILY_ABS_WEIGHT,
        },
        "cadence": {
            "weekly": ["mining_report", "health_report"],
            "weekly_mutation_allowed": False,
            "month_end_max_proposals": MAX_MONTH_END_PROPOSALS,
            "watch_reduce_deprecate": "proposal_only",
        },
        "target_maintenance": {
            "mode": "one_in_one_out",
            "incremental_edge": "paired_95pct_lower_bound_gt_zero",
        },
        "data_blocked": {
            "counts_as_alpha_failure": False,
            "blocks_readiness": True,
        },
        "canonical_chain": list(CONTROL_CHAIN_STAGES),
        "risk_advisor": {
            "advisory_only": True,
            "positive_weight_requires_approval": False,
        },
        "legacy_evidence": {"v2": "reject", "v3": "reject", "auto_upgrade": False},
        "transaction": {
            "wal_required": True,
            "cas_required": True,
            "inverse_rollback_plan_required": True,
            "activation_receipt_required": True,
            "production_apply_enabled": PRODUCTION_APPLY_ENABLED,
            "blocker": PRODUCTION_APPLY_BLOCKER,
        },
    }


def protocol_hash() -> str:
    return semantic_sha256(protocol_policy())


PROTOCOL_HASH = protocol_hash()


def validate_open_session_calendar_v4(
    value: Mapping[str, Any],
    *,
    expected_calendar_sha256: str | None = None,
    expected_latest_pointer_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact strict-Parquet-derived CN open-session calendar."""

    fields = {
        "schema_version",
        "market",
        "source",
        "latest_pointer_sha256",
        "manifest_sha256",
        "open_session_dates",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FactorGovernanceV4Error("open-session calendar fields invalid")
    payload = dict(value)
    if payload["schema_version"] != OPEN_SESSION_CALENDAR_SCHEMA_VERSION:
        raise FactorGovernanceV4Error("open-session calendar schema mismatch")
    if payload["market"] != "CN":
        raise FactorGovernanceV4Error("open-session calendar market must be CN")
    if payload["source"] != OPEN_SESSION_CALENDAR_SOURCE:
        raise FactorGovernanceV4Error("open-session calendar source mismatch")
    for key in ("latest_pointer_sha256", "manifest_sha256"):
        if not _is_sha256(payload[key]):
            raise FactorGovernanceV4Error(f"open-session calendar {key} invalid")
    raw_dates = payload["open_session_dates"]
    if not isinstance(raw_dates, list) or not raw_dates:
        raise FactorGovernanceV4Error("open-session calendar dates missing")
    normalized_dates: list[str] = []
    for raw in raw_dates:
        if type(raw) is not str:
            raise FactorGovernanceV4Error("open-session calendar date must be a string")
        try:
            observed = date.fromisoformat(raw)
        except ValueError as exc:
            raise FactorGovernanceV4Error("open-session calendar date is invalid") from exc
        if observed.isoformat() != raw:
            raise FactorGovernanceV4Error("open-session calendar date is not canonical")
        if observed.weekday() >= 5:
            raise FactorGovernanceV4Error("open-session calendar contains a weekend")
        normalized_dates.append(raw)
    if normalized_dates != sorted(normalized_dates) or len(normalized_dates) != len(
        set(normalized_dates)
    ):
        raise FactorGovernanceV4Error(
            "open-session calendar dates must be sorted and distinct"
        )
    normalized = {**payload, "open_session_dates": normalized_dates}
    calendar_sha256 = semantic_sha256(normalized)
    if (
        expected_calendar_sha256 is not None
        and calendar_sha256 != expected_calendar_sha256
    ):
        raise FactorGovernanceV4Error("open-session calendar SHA mismatch")
    if (
        expected_latest_pointer_sha256 is not None
        and payload["latest_pointer_sha256"] != expected_latest_pointer_sha256
    ):
        raise FactorGovernanceV4Error("open-session calendar pointer SHA mismatch")
    if (
        expected_manifest_sha256 is not None
        and payload["manifest_sha256"] != expected_manifest_sha256
    ):
        raise FactorGovernanceV4Error("open-session calendar manifest SHA mismatch")
    return {**normalized, "calendar_sha256": calendar_sha256}


def assess_candidate_maturity(
    *,
    month_end_rankic_dates: Sequence[str],
    forward_cohorts: Sequence[Mapping[str, Any]],
    calendar: Mapping[str, Any] | None = None,
    expected_calendar_sha256: str | None = None,
    expected_latest_pointer_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Assess either authoritative v4 maturity route."""

    try:
        normalized_calendar = validate_open_session_calendar_v4(
            calendar or {},
            expected_calendar_sha256=expected_calendar_sha256,
            expected_latest_pointer_sha256=expected_latest_pointer_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except (TypeError, ValueError) as exc:
        return {
            "mature": False,
            "maturity_route": "insufficient",
            "month_end_rankic_count": 0,
            "nonoverlap_30d_cohort_count": 0,
            "calendar_verified": False,
            "calendar_sha256": "",
            "maturity_blockers": [f"open_session_calendar_invalid:{exc}"],
            "ninety_day_check_authoritative": False,
        }

    calendar_dates = list(normalized_calendar["open_session_dates"])
    calendar_index = {item: index for index, item in enumerate(calendar_dates)}
    actual_month_ends: dict[str, str] = {}
    for item in calendar_dates:
        actual_month_ends[item[:7]] = item
    month_ends: dict[str, str] = {}
    for raw in month_end_rankic_dates:
        try:
            observed = date.fromisoformat(str(raw))
        except ValueError:
            continue
        canonical = observed.isoformat()
        month = canonical[:7]
        if actual_month_ends.get(month) == canonical:
            month_ends.setdefault(month, canonical)

    candidates: list[tuple[int, int, str]] = []
    for raw in forward_cohorts:
        try:
            raw_dates = raw.get("open_session_dates")
            if not isinstance(raw_dates, list) or len(raw_dates) != 30:
                continue
            dates = [date.fromisoformat(str(item)).isoformat() for item in raw_dates]
            if dates != sorted(dates) or len(dates) != len(set(dates)):
                continue
            indexes = [calendar_index[item] for item in dates]
            if indexes != list(range(indexes[0], indexes[0] + 30)):
                continue
            start = date.fromisoformat(str(raw.get("start"))).isoformat()
            end = date.fromisoformat(str(raw.get("end"))).isoformat()
            horizon = int(raw.get("horizon_days", 0))
        except (KeyError, TypeError, ValueError):
            continue
        cohort_id = str(raw.get("cohort_id") or "").strip()
        if (
            cohort_id
            and horizon == 30
            and start == dates[0]
            and end == dates[-1]
            and raw.get("calendar_sha256") == normalized_calendar["calendar_sha256"]
        ):
            candidates.append((indexes[0], indexes[-1], cohort_id))

    nonoverlap: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    last_end: int | None = None
    for start, end, cohort_id in sorted(candidates):
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
            else "nonoverlap_30d_forward_cohort" if by_cohort else "insufficient"
        ),
        "month_end_rankic_count": len(month_ends),
        "nonoverlap_30d_cohort_count": len(nonoverlap),
        "calendar_verified": True,
        "calendar_sha256": normalized_calendar["calendar_sha256"],
        "maturity_blockers": [],
        "ninety_day_check_authoritative": False,
    }


def _gate_status(value: Any) -> dict[int, bool]:
    if isinstance(value, Mapping):
        result: dict[int, bool] = {}
        for key, passed in value.items():
            try:
                gate_id = int(key)
            except (TypeError, ValueError):
                continue
            if type(passed) is bool:
                result[gate_id] = passed
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = {}
        for raw in value:
            if not isinstance(raw, Mapping):
                continue
            try:
                gate_id = int(raw.get("gate_id", 0))
            except (TypeError, ValueError):
                continue
            passed = raw.get("passed")
            if type(passed) is bool:
                result[gate_id] = passed
        return result
    return {}


def _factor_maturity(record: Mapping[str, Any]) -> dict[str, Any]:
    maturity = record.get("maturity")
    source = maturity if isinstance(maturity, Mapping) else record
    data_source_receipt = record.get("data_source_receipt")
    receipt = data_source_receipt if isinstance(data_source_receipt, Mapping) else {}
    return assess_candidate_maturity(
        month_end_rankic_dates=list(source.get("month_end_rankic_dates", []) or []),
        forward_cohorts=[
            dict(item)
            for item in source.get("forward_cohorts", []) or []
            if isinstance(item, Mapping)
        ],
        calendar=(
            dict(source["calendar"])
            if isinstance(source.get("calendar"), Mapping)
            else None
        ),
        expected_calendar_sha256=(
            str(record.get("calendar_sha256"))
            if _is_sha256(record.get("calendar_sha256"))
            else None
        ),
        expected_latest_pointer_sha256=(
            str(receipt.get("latest_pointer_sha256"))
            if _is_sha256(receipt.get("latest_pointer_sha256"))
            else None
        ),
        expected_manifest_sha256=(
            str(receipt.get("manifest_sha256"))
            if _is_sha256(receipt.get("manifest_sha256"))
            else None
        ),
    )


def assess_factor_record_v4(
    record: Mapping[str, Any],
    *,
    activation_receipt_valid: bool,
) -> dict[str, Any]:
    """Validate one proposed production record without changing it."""

    name = str(record.get("name") or "").strip()
    family = str(record.get("family") or "").strip()
    slot = str(record.get("slot") or "").strip()
    state = str(record.get("state") or "").strip()
    blockers: list[str] = []
    if not name:
        blockers.append("factor_name_missing")
        name = "<missing>"
    if not family:
        blockers.append("factor_family_missing")
    if not slot:
        blockers.append("factor_slot_missing")
    if state != "production_factor":
        blockers.append("factor_state_not_production_factor")

    raw_weight = record.get("weight")
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
        weight = math.nan
    else:
        weight = float(raw_weight)
    if not math.isfinite(weight) or abs(weight) <= 1e-15:
        blockers.append("factor_weight_not_finite_positive_magnitude")

    gates = _gate_status(record.get("gate_results"))
    missing_gates = [gate_id for gate_id in REQUIRED_GATE_IDS if gate_id not in gates]
    failed_gates = [gate_id for gate_id in REQUIRED_GATE_IDS if gates.get(gate_id) is False]
    extra_gates = sorted(set(gates) - set(REQUIRED_GATE_IDS))
    if missing_gates:
        blockers.append("factor_eight_gates_missing:" + ",".join(map(str, missing_gates)))
    if failed_gates:
        blockers.append("factor_eight_gates_failed:" + ",".join(map(str, failed_gates)))
    if extra_gates:
        blockers.append("factor_gate_ids_unexpected:" + ",".join(map(str, extra_gates)))

    maturity = _factor_maturity(record)
    if not maturity["mature"]:
        blockers.append("factor_maturity_not_met")

    bh_q_value = record.get("bh_q_value")
    if isinstance(bh_q_value, bool) or not isinstance(bh_q_value, (int, float)):
        blockers.append("factor_bh_q_value_missing")
        normalized_q = math.nan
    else:
        normalized_q = float(bh_q_value)
        if not math.isfinite(normalized_q) or not 0.0 <= normalized_q <= FDR_Q:
            blockers.append("factor_bh_q_value_above_0.10")
    if record.get("fdr_method") != "benjamini_hochberg_by_family":
        blockers.append("factor_bh_method_mismatch")

    runtime_contract = record.get("runtime_contract")
    runtime_contract_sha = record.get("runtime_contract_sha256")
    if not isinstance(runtime_contract, Mapping) or not runtime_contract:
        blockers.append("factor_runtime_contract_missing")
    elif not _is_sha256(runtime_contract_sha):
        blockers.append("factor_runtime_contract_sha256_invalid")
    elif semantic_sha256(dict(runtime_contract)) != runtime_contract_sha:
        blockers.append("factor_runtime_contract_sha256_mismatch")
    if record.get("runtime_contract_status") != "verified":
        blockers.append("factor_runtime_contract_not_verified")

    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        blockers.append("factor_v4_evidence_missing")
    else:
        if evidence.get("schema_version") != FACTOR_EVIDENCE_SCHEMA_VERSION:
            blockers.append("factor_v4_evidence_schema_mismatch")
        if evidence.get("status") != "verified":
            blockers.append("factor_v4_evidence_not_verified")
        if not _is_sha256(evidence.get("replay_semantic_sha256")):
            blockers.append("factor_v4_evidence_replay_sha_invalid")

    health = record.get("health")
    data_blocked = False
    if not isinstance(health, Mapping):
        blockers.append("factor_fresh_health_missing")
    else:
        data_blocked = health.get("status") == "data_blocked" or bool(
            health.get("data_blocked", False)
        )
        if data_blocked:
            blockers.append("factor_health_data_blocked")
        if health.get("status") != "healthy":
            blockers.append("factor_health_not_healthy")
        if health.get("fresh") is not True:
            blockers.append("factor_health_not_fresh")

    if not activation_receipt_valid:
        blockers.append("factor_activation_receipt_missing_or_invalid")
    blockers = list(dict.fromkeys(blockers))
    return {
        "name": name,
        "family": family,
        "slot": slot,
        "state": state,
        "weight": weight,
        "bh_q_value": normalized_q,
        "maturity": maturity,
        "gate_status": {str(key): value for key, value in sorted(gates.items())},
        "runtime_contract_sha256": (
            str(runtime_contract_sha) if _is_sha256(runtime_contract_sha) else ""
        ),
        "data_blocked": data_blocked,
        "data_blocked_counts_as_alpha_failure": False,
        "activation_receipt_valid": activation_receipt_valid,
        "healthy": not blockers,
        "blockers": blockers,
    }


def assess_candidate_admission_v4(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Assess a zero-weight v4 candidate after exact local evidence readback.

    This is the only candidate-admission assessment path.  It deliberately
    performs file readback rather than trusting caller-supplied status flags or
    hashes.  A readback failure is a blocker; there is no shallow-evidence
    fallback.
    """

    name = str(candidate.get("name") or "").strip()
    family = str(candidate.get("family") or "").strip()
    slot = str(candidate.get("slot") or "").strip()
    blockers: list[str] = []
    if not name:
        blockers.append("candidate_name_missing")
    if not family:
        blockers.append("candidate_family_missing")
    if not slot:
        blockers.append("candidate_slot_missing")

    source = candidate.get("data_source_receipt")
    if not isinstance(source, Mapping):
        blockers.append("strict_parquet_source_receipt_missing")
    else:
        if source.get("schema_version") != "factor-v4-strict-parquet-source-receipt.v1":
            blockers.append("strict_parquet_source_receipt_schema_mismatch")
        if source.get("backend") != "parquet" or source.get("mode_policy") != "strict":
            blockers.append("strict_parquet_source_contract_mismatch")
        if source.get("snapshot_healthy") is not True:
            blockers.append("strict_parquet_snapshot_not_healthy")
        if source.get("universe_scope") != "full_a":
            blockers.append("strict_parquet_source_not_full_a")
        for key in (
            "snapshot_id",
            "latest_pointer_sha256",
            "manifest_sha256",
            "market_data_input_sha256",
        ):
            if key == "snapshot_id":
                if not str(source.get(key) or "").strip():
                    blockers.append("strict_parquet_snapshot_id_missing")
            elif not _is_sha256(source.get(key)):
                blockers.append(f"strict_parquet_{key}_invalid")

    binding_fields = (
        "registry_file_sha256",
        "calendar_sha256",
        "pit_sha256",
        "candidate_catalog_sha256",
        "screening_evidence_sha256",
        "dedup_evidence_sha256",
        "quantitative_evidence_sha256",
    )
    for key in binding_fields:
        if not _is_sha256(candidate.get(key)):
            blockers.append(f"candidate_{key}_invalid")

    runtime_contract = candidate.get("runtime_contract")
    runtime_contract_sha = candidate.get("runtime_contract_sha256")
    if not isinstance(runtime_contract, Mapping) or not runtime_contract:
        blockers.append("candidate_runtime_contract_missing")
    elif not _is_sha256(runtime_contract_sha):
        blockers.append("candidate_runtime_contract_sha256_invalid")
    elif semantic_sha256(dict(runtime_contract)) != runtime_contract_sha:
        blockers.append("candidate_runtime_contract_sha256_mismatch")
    elif runtime_contract.get("factor_name") != name:
        blockers.append("candidate_runtime_contract_factor_identity_mismatch")
    if candidate.get("runtime_contract_status") != "verified":
        blockers.append("candidate_runtime_contract_not_verified")
    if candidate.get("full_a_universe_verified") is not True:
        blockers.append("candidate_full_a_universe_not_verified")
    initial_weight = candidate.get("initial_weight")
    if (
        isinstance(initial_weight, bool)
        or not isinstance(initial_weight, (int, float))
        or not math.isfinite(float(initial_weight))
        or abs(float(initial_weight)) > 1e-15
    ):
        blockers.append("candidate_initial_weight_must_be_zero")

    maturity = _factor_maturity(candidate)
    if not maturity["mature"]:
        blockers.append("candidate_maturity_not_met")

    screening_readback: Mapping[str, Any] | None = None
    q_value: float | None = None
    screening_descriptor = candidate.get("screening_evidence")
    if not isinstance(screening_descriptor, Mapping):
        blockers.append("candidate_screening_evidence_readback_missing")
    else:
        try:
            from quant_investor.factors.governance_candidate_admission_evidence_v4 import (
                readback_screening_evidence_v4,
            )

            screening_readback = readback_screening_evidence_v4(
                dict(screening_descriptor),
                candidate_name=name,
            )
            q_value = float(screening_readback["bh_q_value"])
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"candidate_screening_exact_readback_failed:{exc}")
    if screening_readback is not None:
        if screening_readback.get("candidate_catalog_sha256") != candidate.get(
            "candidate_catalog_sha256"
        ):
            blockers.append("candidate_screening_catalog_sha_mismatch")
        if screening_readback.get("screening_evidence_sha256") != candidate.get(
            "screening_evidence_sha256"
        ):
            blockers.append("candidate_screening_evidence_sha_mismatch")
        if screening_readback.get("family") != family:
            blockers.append("candidate_screening_family_mismatch")
        if screening_readback.get("bh_pass") is not True:
            blockers.append("candidate_screening_bh_not_passed")
        if q_value is None or not 0.0 <= q_value <= FDR_Q:
            blockers.append("candidate_screening_bh_q_value_above_0.10")
        source_bindings = screening_readback["screening_evidence"].get(
            "source_bindings", {}
        )
        if isinstance(source_bindings, Mapping):
            screening_source_bindings = {
                "registry_file_sha256": candidate.get("registry_file_sha256"),
                "latest_pointer_sha256": (
                    source.get("latest_pointer_sha256")
                    if isinstance(source, Mapping)
                    else None
                ),
                "manifest_sha256": (
                    source.get("manifest_sha256") if isinstance(source, Mapping) else None
                ),
                "market_data_input_sha256": (
                    source.get("market_data_input_sha256")
                    if isinstance(source, Mapping)
                    else None
                ),
                "pit_sha256": candidate.get("pit_sha256"),
                "calendar_sha256": candidate.get("calendar_sha256"),
            }
            for key, expected in screening_source_bindings.items():
                if source_bindings.get(key) != expected:
                    blockers.append(f"candidate_screening_{key}_mismatch")
        supplied_q = candidate.get("bh_q_value")
        if (
            supplied_q is not None
            and (
                isinstance(supplied_q, bool)
                or not isinstance(supplied_q, (int, float))
                or not math.isfinite(float(supplied_q))
                or float(supplied_q) != q_value
            )
        ):
            blockers.append("candidate_bh_q_value_caller_mismatch")
        supplied_fdr = candidate.get("fdr_method")
        if (
            supplied_fdr is not None
            and supplied_fdr != screening_readback.get("fdr_method")
        ):
            blockers.append("candidate_fdr_method_caller_mismatch")

    dedup_readback: Mapping[str, Any] | None = None
    dedup_descriptor = candidate.get("dedup_evidence")
    if not isinstance(dedup_descriptor, Mapping):
        blockers.append("candidate_dedup_evidence_readback_missing")
    elif screening_readback is not None:
        try:
            from quant_investor.factors.governance_candidate_admission_evidence_v4 import (
                readback_candidate_dedup_evidence_v4,
            )

            dedup_readback = readback_candidate_dedup_evidence_v4(
                dict(dedup_descriptor),
                catalog=screening_readback["candidate_catalog"],
                candidate_name=name,
                screening_evidence_sha256=screening_readback[
                    "screening_evidence_sha256"
                ],
                source_bindings=screening_readback["screening_evidence"][
                    "source_bindings"
                ],
            )
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"candidate_dedup_exact_readback_failed:{exc}")
    if dedup_descriptor is not None and screening_readback is None:
        blockers.append("candidate_dedup_requires_screening_catalog_readback")
    if dedup_readback is not None:
        if dedup_readback.get("dedup_evidence_sha256") != candidate.get(
            "dedup_evidence_sha256"
        ):
            blockers.append("candidate_dedup_evidence_sha_mismatch")
        if dedup_readback.get("evidence_complete") is not True:
            blockers.append("candidate_dedup_evidence_incomplete")
        if dedup_readback.get("duplicate_primitive") is not False:
            blockers.append("candidate_duplicate_primitive_not_false")
        if dedup_readback.get("high_correlation_dedup_passed") is not True:
            blockers.append("candidate_high_correlation_dedup_not_passed")
        supplied_duplicate = candidate.get("duplicate_primitive")
        if (
            supplied_duplicate is not None
            and supplied_duplicate != dedup_readback.get("duplicate_primitive")
        ):
            blockers.append("candidate_duplicate_primitive_caller_mismatch")
        supplied_dedup_pass = candidate.get("high_correlation_dedup_passed")
        if (
            supplied_dedup_pass is not None
            and supplied_dedup_pass
            != dedup_readback.get("high_correlation_dedup_passed")
        ):
            blockers.append("candidate_high_correlation_dedup_caller_mismatch")
    gates = _gate_status(candidate.get("gate_results"))
    if set(gates) != set(REQUIRED_GATE_IDS):
        blockers.append("candidate_eight_gate_domain_mismatch")
    if any(gates.get(gate_id) is not True for gate_id in REQUIRED_GATE_IDS):
        blockers.append("candidate_eight_gates_not_all_passed")

    evidence = candidate.get("evidence")
    evidence_readback: Mapping[str, Any] | None = None
    if not isinstance(evidence, Mapping):
        blockers.append("candidate_v4_evidence_missing")
    else:
        try:
            from quant_investor.factors.governance_canonical_replay_v4 import (
                readback_v4_evidence,
            )

            evidence_readback = readback_v4_evidence(dict(evidence))
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"candidate_v4_evidence_exact_readback_failed:{exc}")

    if evidence_readback is not None:
        if evidence_readback.get("local_bytes_readback_verified") is not True:
            blockers.append("candidate_v4_evidence_local_bytes_not_verified")
        if evidence_readback.get("complete_chain_hash_binding_verified") is not True:
            blockers.append("candidate_v4_evidence_chain_binding_not_verified")
        replay = evidence_readback.get("replay")
        normalized_evidence = evidence_readback.get("evidence")
        if not isinstance(replay, Mapping) or not isinstance(normalized_evidence, Mapping):
            blockers.append("candidate_v4_evidence_readback_shape_invalid")
        else:
            context = replay.get("context")
            comparison = replay.get("comparison")
            arms = replay.get("arms")
            if not isinstance(context, Mapping):
                blockers.append("candidate_v4_replay_context_missing")
            if not isinstance(comparison, Mapping):
                blockers.append("candidate_v4_replay_comparison_missing")
            if normalized_evidence.get("factor_name") != name:
                blockers.append("candidate_v4_evidence_factor_identity_mismatch")
            if replay.get("registry_file_sha256") != candidate.get(
                "registry_file_sha256"
            ):
                blockers.append("candidate_v4_registry_sha_mismatch")
            if isinstance(comparison, Mapping):
                if comparison.get("challenger") != name:
                    blockers.append("candidate_v4_replay_challenger_mismatch")
                if comparison.get("slot") != slot:
                    blockers.append("candidate_v4_replay_slot_mismatch")
            if isinstance(context, Mapping):
                context_bindings = {
                    "calendar_sha256": candidate.get("calendar_sha256"),
                    "pit_sha256": candidate.get("pit_sha256"),
                    "runtime_contract_sha256": runtime_contract_sha,
                    "candidate_catalog_sha256": candidate.get(
                        "candidate_catalog_sha256"
                    ),
                    "screening_evidence_sha256": candidate.get(
                        "screening_evidence_sha256"
                    ),
                    "dedup_evidence_sha256": candidate.get(
                        "dedup_evidence_sha256"
                    ),
                    "quantitative_evidence_sha256": candidate.get(
                        "quantitative_evidence_sha256"
                    ),
                    "latest_pointer_sha256": (
                        source.get("latest_pointer_sha256")
                        if isinstance(source, Mapping)
                        else None
                    ),
                    "manifest_sha256": (
                        source.get("manifest_sha256")
                        if isinstance(source, Mapping)
                        else None
                    ),
                    "market_data_input_sha256": (
                        source.get("market_data_input_sha256")
                        if isinstance(source, Mapping)
                        else None
                    ),
                }
                for key, expected in context_bindings.items():
                    if context.get(key) != expected:
                        blockers.append(f"candidate_v4_{key}_mismatch")
            challenger_record: Mapping[str, Any] | None = None
            if isinstance(arms, Mapping):
                transition_mode = (
                    comparison.get("transition_mode")
                    if isinstance(comparison, Mapping)
                    else None
                )
                comparison_arm_name = (
                    "A"
                    if transition_mode == "add"
                    else "B" if transition_mode == "replace" else ""
                )
                if dedup_readback is not None and comparison_arm_name:
                    comparison_arm = arms.get(comparison_arm_name)
                    expected_factor_names: list[str] | None = None
                    if isinstance(comparison_arm, Mapping):
                        quant = comparison_arm.get("quant")
                        if isinstance(quant, Mapping) and isinstance(
                            quant.get("selected_factors"), list
                        ):
                            expected_factor_names = [
                                str(item) for item in quant["selected_factors"]
                            ]
                    if expected_factor_names is None:
                        blockers.append("candidate_dedup_replay_factor_set_missing")
                    else:
                        expected_sorted = sorted(expected_factor_names)
                        observed = dedup_readback.get("comparison_factor_names")
                        if observed != expected_sorted:
                            blockers.append(
                                "candidate_dedup_comparison_factor_set_mismatch"
                            )
                arm_c = arms.get("C")
                if isinstance(arm_c, Mapping):
                    quant = arm_c.get("quant")
                    if isinstance(quant, Mapping):
                        records = quant.get("factor_records")
                        if isinstance(records, Mapping):
                            raw_record = records.get(name)
                            if isinstance(raw_record, Mapping):
                                challenger_record = raw_record
            if challenger_record is None:
                blockers.append("candidate_v4_replay_challenger_record_missing")
            else:
                if challenger_record.get("family") != family:
                    blockers.append("candidate_v4_replay_family_mismatch")
                if challenger_record.get("slot") != slot:
                    blockers.append("candidate_v4_replay_record_slot_mismatch")
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": "factor-governance-candidate-admission.v4",
        "protocol_version": PROTOCOL_VERSION,
        "name": name,
        "family": family,
        "slot": slot,
        "initial_weight": (
            float(initial_weight)
            if isinstance(initial_weight, (int, float)) and not isinstance(initial_weight, bool)
            else None
        ),
        "maturity": maturity,
        "bh_q_value": (
            q_value
            if q_value is not None and math.isfinite(q_value)
            else None
        ),
        "candidate_registry_proposal_allowed": not blockers,
        "registry_write_enabled": False,
        "candidate_catalog_sha256": (
            screening_readback.get("candidate_catalog_sha256")
            if screening_readback is not None
            else ""
        ),
        "screening_evidence_sha256": (
            screening_readback.get("screening_evidence_sha256")
            if screening_readback is not None
            else ""
        ),
        "dedup_evidence_sha256": (
            dedup_readback.get("dedup_evidence_sha256")
            if dedup_readback is not None
            else ""
        ),
        "screening_exact_readback_verified": screening_readback is not None,
        "dedup_exact_readback_verified": dedup_readback is not None,
        "evidence_exact_readback_verified": evidence_readback is not None
        and not any(item.startswith("candidate_v4_") for item in blockers),
        "blockers": blockers,
    }


def validate_candidate_admission_v4(candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = assess_candidate_admission_v4(candidate)
    if result["blockers"]:
        raise FactorGovernanceV4Error(
            "candidate admission blocked: " + ";".join(result["blockers"])
        )
    return result


def build_health_action_proposal_v4(
    *,
    factor_name: str,
    failure_window_ids: Sequence[str],
    current_weight: float,
    data_blocked_window_ids: Sequence[str] = (),
    cadence: str = "month_end",
) -> dict[str, Any]:
    """Map distinct alpha failures to a month-end proposal, never an apply."""

    name = str(factor_name or "").strip()
    if not name:
        raise FactorGovernanceV4Error("factor_name must be non-empty")
    if isinstance(current_weight, bool) or not isinstance(current_weight, (int, float)):
        raise FactorGovernanceV4Error("current_weight must be finite numeric")
    weight = float(current_weight)
    if not math.isfinite(weight):
        raise FactorGovernanceV4Error("current_weight must be finite numeric")
    failures = sorted({str(item).strip() for item in failure_window_ids if str(item).strip()})
    data_blocked = sorted(
        {str(item).strip() for item in data_blocked_window_ids if str(item).strip()}
    )
    failure_count = len(failures)
    if failure_count >= 3:
        action = "deprecate_proposal"
        proposed_weight = 0.0
    elif failure_count == 2:
        action = "reduce_proposal"
        proposed_weight = weight * 0.5
    elif failure_count == 1:
        action = "watch_proposal"
        proposed_weight = weight
    else:
        action = None
        proposed_weight = weight
    normalized_cadence = str(cadence).strip().lower().replace("-", "_")
    blockers: list[str] = []
    if normalized_cadence != "month_end" and action is not None:
        blockers.append("health_action_proposals_are_month_end_only")
    if data_blocked:
        blockers.append("fresh_data_blocked_blocks_factor_set_new_risk")
    return {
        "schema_version": "factor-governance-health-action-proposal.v4",
        "protocol_version": PROTOCOL_VERSION,
        "factor_name": name,
        "cadence": normalized_cadence,
        "action": action,
        "apply": False,
        "proposal_only": True,
        "independent_alpha_failure_count": failure_count,
        "distinct_alpha_failure_window_ids": failures,
        "data_blocked_window_count": len(data_blocked),
        "data_blocked_window_ids": data_blocked,
        "data_blocked_counts_as_alpha_failure": False,
        "factor_set_new_risk_blocked": bool(data_blocked),
        "blocking_factor_names": [name] if data_blocked else [],
        "current_weight": weight,
        "proposed_weight": proposed_weight,
        "proposal_emitted": action is not None and normalized_cadence == "month_end",
        "blockers": blockers,
    }


def _activation_receipt_status(
    receipt: Mapping[str, Any] | None,
    *,
    as_of: str,
    registry_file_sha256: str,
    production_factor_set_sha256: str,
    runtime_contracts_sha256: str,
) -> dict[str, Any]:
    if receipt is None:
        return {
            "valid": False,
            "receipt": None,
            "blockers": ["activation_receipt_missing"],
        }
    try:
        from quant_investor.factors.governance_transaction_v4 import (
            validate_activation_receipt_v4,
        )

        normalized = validate_activation_receipt_v4(
            receipt,
            expected_as_of=as_of,
            expected_protocol_hash=protocol_hash(),
            expected_registry_file_sha256=registry_file_sha256,
            expected_production_factor_set_sha256=production_factor_set_sha256,
            expected_runtime_contracts_sha256=runtime_contracts_sha256,
        )
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "receipt": None,
            "blockers": [f"activation_receipt_invalid:{exc}"],
        }
    return {
        "valid": True,
        "receipt_sha256": normalized["receipt_sha256"],
        "receipt": normalized,
        "blockers": [],
    }


def assess_factor_governance_readiness_v4(
    factor_records: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    registry_file_sha256: str,
    production_factor_set_sha256: str,
    activation_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assess proposed v4 readiness and always leave apply authority false."""

    try:
        normalized_as_of = date.fromisoformat(str(as_of)).isoformat()
    except ValueError as exc:
        raise FactorGovernanceV4Error("as_of must be ISO YYYY-MM-DD") from exc
    if not _is_sha256(registry_file_sha256):
        raise FactorGovernanceV4Error("registry_file_sha256 must be lowercase SHA-256")
    if not _is_sha256(production_factor_set_sha256):
        raise FactorGovernanceV4Error("production_factor_set_sha256 must be lowercase SHA-256")

    runtime_hashes = sorted(
        str(record.get("runtime_contract_sha256") or "") for record in factor_records
    )
    runtime_contracts_sha256 = semantic_sha256(runtime_hashes)
    receipt_status = _activation_receipt_status(
        activation_receipt,
        as_of=normalized_as_of,
        registry_file_sha256=registry_file_sha256,
        production_factor_set_sha256=production_factor_set_sha256,
        runtime_contracts_sha256=runtime_contracts_sha256,
    )
    assessments = [
        assess_factor_record_v4(
            dict(record), activation_receipt_valid=bool(receipt_status["valid"])
        )
        for record in factor_records
    ]

    blockers: list[str] = []
    names = [row["name"] for row in assessments]
    if len(names) != len(set(names)):
        blockers.append("factor_names_not_unique")
    slots: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    weights: dict[str, float] = {}
    total = 0.0
    for row in assessments:
        name = row["name"]
        family = row["family"]
        slot = row["slot"]
        if slot:
            slots.setdefault(slot, []).append(name)
        if family:
            families[name] = family
        weight = row["weight"]
        if math.isfinite(weight) and abs(weight) > 1e-15:
            weights[name] = abs(weight)
            total += abs(weight)
        blockers.extend(f"{name}:{item}" for item in row["blockers"])
    for slot, members in slots.items():
        if len(members) != 1:
            blockers.append(f"factor_slot_multiple_incumbents:{slot}")
            for row in assessments:
                if row["slot"] == slot:
                    row["healthy"] = False
                    row["blockers"] = list(
                        dict.fromkeys([*row["blockers"], "factor_slot_not_unique"])
                    )

    normalized_weights = (
        {name: value / total for name, value in weights.items()} if total > 1e-15 else {}
    )
    family_weights: dict[str, float] = {}
    for name, weight in normalized_weights.items():
        if weight > MAX_FACTOR_ABS_WEIGHT + 1e-12:
            blockers.append(f"factor_abs_weight_above_0.20:{name}")
            for row in assessments:
                if row["name"] == name:
                    row["healthy"] = False
                    row["blockers"] = list(
                        dict.fromkeys([*row["blockers"], "factor_abs_weight_above_0.20"])
                    )
        family = families.get(name)
        if family:
            family_weights[family] = family_weights.get(family, 0.0) + weight
    for family, weight in family_weights.items():
        if weight > MAX_FAMILY_ABS_WEIGHT + 1e-12:
            blockers.append(f"family_abs_weight_above_0.35:{family}")
            for row in assessments:
                if row["family"] == family:
                    row["healthy"] = False
                    row["blockers"] = list(
                        dict.fromkeys([*row["blockers"], "family_abs_weight_above_0.35"])
                    )

    record_count = len(assessments)
    family_count = len(set(families.values()))
    if record_count < MIN_NEW_RISK_FACTOR_COUNT:
        blockers.append("production_factor_count_below_5")
    if family_count < MIN_NEW_RISK_FAMILY_COUNT:
        blockers.append("production_family_count_below_3")
    if record_count == MIN_NEW_RISK_FACTOR_COUNT and family_count < record_count:
        blockers.append("exact_five_requires_five_distinct_families")
    if record_count > TARGET_PRODUCTION_FACTOR_COUNT:
        blockers.append("production_factor_count_above_target_10")
    blockers.extend(receipt_status["blockers"])
    blockers = list(dict.fromkeys(blockers))

    healthy_count = sum(1 for row in assessments if row["healthy"])
    minimum_ready = (
        not blockers
        and healthy_count == record_count
        and MIN_NEW_RISK_FACTOR_COUNT <= record_count <= TARGET_PRODUCTION_FACTOR_COUNT
        and family_count >= MIN_NEW_RISK_FAMILY_COUNT
    )
    target_ready = minimum_ready and record_count == TARGET_PRODUCTION_FACTOR_COUNT
    underfilled = minimum_ready and record_count < TARGET_PRODUCTION_FACTOR_COUNT
    status = (
        "ready_target_10"
        if target_ready
        else "underfilled_accelerated_mining" if underfilled else "no_new_risk"
    )
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "as_of": normalized_as_of,
        "status": status,
        "factor_governance_ready": minimum_ready,
        "target_ready": target_ready,
        "underfilled": underfilled,
        "accelerated_mining_required": underfilled,
        "new_risk_eligible": minimum_ready,
        "new_risk_authorized": False,
        "production_apply_enabled": PRODUCTION_APPLY_ENABLED,
        "production_apply_blocker": PRODUCTION_APPLY_BLOCKER,
        "production_factor_target": TARGET_PRODUCTION_FACTOR_COUNT,
        "production_factor_count": record_count,
        "healthy_factor_count": healthy_count,
        "production_family_count": family_count,
        "registry_file_sha256": registry_file_sha256,
        "production_factor_set_sha256": production_factor_set_sha256,
        "runtime_contracts_sha256": runtime_contracts_sha256,
        "activation_receipt": receipt_status,
        "normalized_abs_weights": normalized_weights,
        "family_normalized_abs_weights": family_weights,
        "slot_incumbents": slots,
        "factors": assessments,
        "blockers": blockers,
    }


def assess_governance_cycle_v4(
    *,
    cadence: str,
    production_factor_count: int,
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate weekly report-only or month-end proposal-only behavior."""

    normalized_cadence = str(cadence).strip().lower().replace("-", "_")
    if normalized_cadence not in {"weekly", "month_end"}:
        raise FactorGovernanceV4Error("cadence must be weekly or month_end")
    if isinstance(production_factor_count, bool) or production_factor_count < 0:
        raise FactorGovernanceV4Error("production_factor_count must be non-negative")
    rows = [dict(item) for item in proposals]
    blockers: list[str] = []
    allowed = {
        "add_proposal",
        "replace_proposal",
        "watch_proposal",
        "reduce_proposal",
        "deprecate_proposal",
    }
    if normalized_cadence == "weekly" and rows:
        blockers.append("weekly_cycle_is_report_only")
    if normalized_cadence == "month_end" and len(rows) > MAX_MONTH_END_PROPOSALS:
        blockers.append("month_end_proposal_count_above_2")
    for index, proposal in enumerate(rows):
        action = str(proposal.get("action") or "")
        if action not in allowed:
            blockers.append(f"proposal_{index}_action_not_proposal_only")
            continue
        if proposal.get("apply") not in {None, False}:
            blockers.append(f"proposal_{index}_apply_must_be_false")
        if production_factor_count >= TARGET_PRODUCTION_FACTOR_COUNT:
            if action == "add_proposal":
                blockers.append(f"proposal_{index}_target_requires_one_in_one_out")
            if action == "replace_proposal":
                incumbent = str(proposal.get("incumbent") or "").strip()
                challenger = str(proposal.get("challenger") or "").strip()
                if not incumbent or not challenger or incumbent == challenger:
                    blockers.append(f"proposal_{index}_replacement_identity_invalid")
                lower = proposal.get("incremental_edge_ci95_lower")
                if (
                    isinstance(lower, bool)
                    or not isinstance(lower, (int, float))
                    or not math.isfinite(float(lower))
                    or float(lower) <= 0.0
                ):
                    blockers.append(f"proposal_{index}_incremental_edge_ci95_lower_not_positive")
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": "factor-governance-cycle-assessment.v4",
        "protocol_version": PROTOCOL_VERSION,
        "cadence": normalized_cadence,
        "status": "proposal_ready" if not blockers else "proposal_blocked",
        "report_only": normalized_cadence == "weekly",
        "proposal_only": True,
        "production_apply_enabled": False,
        "proposal_count": len(rows),
        "proposals": rows,
        "blockers": blockers,
    }


__all__ = [
    "CONTROL_CHAIN_STAGES",
    "FACTOR_EVIDENCE_SCHEMA_VERSION",
    "FDR_Q",
    "FactorGovernanceV4Error",
    "MAX_FACTOR_ABS_WEIGHT",
    "MAX_FAMILY_ABS_WEIGHT",
    "MAX_MONTH_END_PROPOSALS",
    "MIN_MONTH_END_RANKIC_COUNT",
    "MIN_NEW_RISK_FACTOR_COUNT",
    "MIN_NEW_RISK_FAMILY_COUNT",
    "MIN_NONOVERLAP_30D_COHORT_COUNT",
    "OPEN_SESSION_CALENDAR_SCHEMA_VERSION",
    "OPEN_SESSION_CALENDAR_SOURCE",
    "PRODUCTION_APPLY_BLOCKER",
    "PRODUCTION_APPLY_ENABLED",
    "PROTOCOL_HASH",
    "PROTOCOL_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "READINESS_SCHEMA_VERSION",
    "REQUIRED_GATE_IDS",
    "TARGET_PRODUCTION_FACTOR_COUNT",
    "assess_candidate_admission_v4",
    "assess_candidate_maturity",
    "assess_factor_governance_readiness_v4",
    "assess_factor_record_v4",
    "assess_governance_cycle_v4",
    "build_health_action_proposal_v4",
    "canonical_json_bytes",
    "protocol_hash",
    "protocol_policy",
    "semantic_sha256",
    "validate_candidate_admission_v4",
    "validate_open_session_calendar_v4",
]
