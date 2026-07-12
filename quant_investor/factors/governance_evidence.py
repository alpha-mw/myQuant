"""Report-only normalizer for FactorGovernanceProtocol v2 replay evidence.

This module makes caller-supplied replay data deterministic and
content-addressed.  It is not yet a canonical production producer because it
does not read back and hash the actual bytes of every v13 DAG stage artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
)
from quant_investor.factors.governance_protocol_v2 import (
    CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
    PROTOCOL_VERSION,
    FactorEvidenceWindow,
    FactorSlot,
    FactorTransitionPlan,
    RegistryMutationPlan,
    canonical_replay_producer_control,
    protocol_hash,
    validate_purged_walk_forward,
)
from quant_investor.factors.registry_store import (
    load_registry_snapshot_strict,
)


RAW_REPLAY_SCHEMA_VERSION = "factor-full-chain-replay-input.v1"
EVIDENCE_SCHEMA_VERSION = "factor-governance-replay-evidence.v2"
SNAPSHOT_EVIDENCE_SCHEMA_VERSION = "strict-parquet-snapshot-evidence.v1"
PRODUCER_ID = "myquant.factor_governance_replay_evidence"
PRODUCER_VERSION = "v2-report-only-normalizer"
CONTROL_CHAIN_STAGES = (
    "quant",
    "theme",
    "bayesian",
    "risk_guard",
    "portfolio_constructor",
)
ARM_NAMES = ("A", "B", "C", "D")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_hex(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def producer_code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _strict_dates(values: Sequence[Any], label: str) -> list[str]:
    dates: list[str] = []
    for raw in values:
        value = str(raw or "").strip()[:10]
        if len(value) != 10:
            raise ValueError(f"{label} contains an invalid date")
        try:
            from datetime import date

            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} contains an invalid date") from exc
        dates.append(parsed.isoformat())
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError(f"{label} must be sorted and distinct")
    return dates


def _return_series(values: Sequence[Any], label: str) -> list[float | None]:
    result: list[float | None] = []
    for raw in values:
        if raw is None:
            result.append(None)
            continue
        if isinstance(raw, bool):
            raise ValueError(f"{label} contains a boolean")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-numeric return") from exc
        if not math.isfinite(number) or number <= -1.0:
            raise ValueError(f"{label} contains an invalid return")
        result.append(number)
    return result


def _stage_hashes(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.stage_artifact_hashes must be an object")
    result = {str(key): str(item or "").strip() for key, item in value.items()}
    if set(result) != set(CONTROL_CHAIN_STAGES) or any(not item for item in result.values()):
        raise ValueError(
            f"{label}.stage_artifact_hashes must contain all control-chain stages"
        )
    return {
        key: _sha256_hex(result[key], f"{label}.{key}")
        for key in CONTROL_CHAIN_STAGES
    }


def replay_arm_hash(arm: Mapping[str, Any]) -> str:
    return _hash(
        {
            "trading_dates": list(arm.get("trading_dates", []) or []),
            "after_cost_daily_returns": list(
                arm.get("after_cost_daily_returns", []) or []
            ),
            "stage_artifact_hashes": dict(
                arm.get("stage_artifact_hashes", {}) or {}
            ),
        }
    )


def _normalize_arm(
    raw: Mapping[str, Any],
    *,
    arm_name: str,
    valid_trading_days: set[str],
) -> dict[str, Any]:
    dates = _strict_dates(
        list(raw.get("trading_dates", []) or []),
        f"arm_{arm_name}.trading_dates",
    )
    returns = _return_series(
        list(raw.get("after_cost_daily_returns", []) or []),
        f"arm_{arm_name}.after_cost_daily_returns",
    )
    if not dates or len(dates) != len(returns):
        raise ValueError(f"arm_{arm_name} dates/returns must align and be non-empty")
    if any(item not in valid_trading_days for item in dates):
        raise ValueError(f"arm_{arm_name} contains a non-snapshot trading date")
    arm = {
        "trading_dates": dates,
        "after_cost_daily_returns": returns,
        "stage_artifact_hashes": _stage_hashes(
            raw.get("stage_artifact_hashes"),
            f"arm_{arm_name}",
        ),
    }
    arm["arm_hash"] = replay_arm_hash(arm)
    return arm


def _normalize_limit(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} limit evidence must be an object")
    try:
        measured = float(raw.get("measured"))
        limit = float(raw.get("limit"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} limit evidence must be numeric") from exc
    artifact_hash = _sha256_hex(
        raw.get("artifact_hash"),
        f"{label}.artifact_hash",
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in (measured, limit)):
        raise ValueError(f"{label} limit evidence must be finite and non-negative")
    return {
        "measured": measured,
        "limit": limit,
        "artifact_hash": artifact_hash,
    }


def produce_governance_replay_evidence(
    raw_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one caller-supplied replay into report-only evidence."""

    raw = copy.deepcopy(dict(raw_replay))
    if raw.get("schema_version") != RAW_REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported full-chain replay input schema")
    snapshot = dict(raw.get("snapshot_evidence", {}) or {})
    if snapshot.get("schema_version") != SNAPSHOT_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("strict snapshot evidence schema is missing")
    if snapshot.get("source") != "strict_parquet_snapshot":
        raise ValueError("snapshot evidence is not strict Parquet")
    for key in ("snapshot_id", "manifest_sha256", "latest_complete_trade_date"):
        if not str(snapshot.get(key, "") or "").strip():
            raise ValueError(f"snapshot evidence {key} is missing")
    valid_days = _strict_dates(
        list(snapshot.get("valid_trading_days", []) or []),
        "snapshot_evidence.valid_trading_days",
    )
    if not valid_days:
        raise ValueError("snapshot valid_trading_days is empty")
    snapshot = {
        "schema_version": SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
        "source": "strict_parquet_snapshot",
        "snapshot_id": str(snapshot["snapshot_id"]),
        "manifest_sha256": _sha256_hex(
            snapshot["manifest_sha256"],
            "snapshot_evidence.manifest_sha256",
        ),
        "latest_complete_trade_date": str(
            snapshot["latest_complete_trade_date"]
        )[:10],
        "valid_trading_days": valid_days,
        "valid_trading_days_sha256": _hash(valid_days),
    }
    snapshot["snapshot_evidence_hash"] = _hash(snapshot)

    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARM_NAMES):
        raise ValueError("replay input must contain exactly A/B/C/D arms")
    arms = {
        name: _normalize_arm(
            dict(raw_arms[name]),
            arm_name=name,
            valid_trading_days=set(valid_days),
        )
        for name in ARM_NAMES
    }
    date_contract = arms["A"]["trading_dates"]
    if any(arms[name]["trading_dates"] != date_contract for name in ARM_NAMES):
        raise ValueError("A/B/C/D arms must use the same trading dates")

    slot = FactorSlot.from_dict(dict(raw.get("slot", {}) or {}))
    challenger = FactorRecord.from_dict(
        dict(raw.get("challenger_record", {}) or {})
    )
    if challenger.name != slot.reserve:
        raise ValueError("challenger_record does not match slot reserve")
    if challenger.state not in {
        FactorLifecycleState.SHADOW,
        FactorLifecycleState.MATURE_CANDIDATE,
        FactorLifecycleState.PRODUCTION_CANDIDATE,
    }:
        raise ValueError("challenger_record lifecycle is not eligible")
    if not challenger.all_gates_passed():
        raise ValueError("challenger_record does not have complete 8-gate evidence")
    evidence_window = FactorEvidenceWindow.from_dict(
        dict(raw.get("evidence_window", {}) or {})
    )
    if evidence_window.snapshot_id != snapshot["snapshot_id"]:
        raise ValueError("evidence window snapshot_id mismatch")

    health = dict(raw.get("health_evidence", {}) or {})
    failure_windows = list(
        dict.fromkeys(
            str(item).strip()
            for item in health.get("failure_window_ids", []) or []
            if str(item).strip()
        )
    )
    health_hash = _sha256_hex(
        health.get("artifact_hash"),
        "health_evidence.artifact_hash",
    )
    selection = dict(raw.get("selection_evidence", {}) or {})
    selection_hash = _sha256_hex(
        selection.get("artifact_hash"),
        "selection_evidence.artifact_hash",
    )
    try:
        family_fdr_q_value = float(selection.get("family_fdr_q_value"))
    except (TypeError, ValueError) as exc:
        raise ValueError("selection family_fdr_q_value is missing") from exc
    if not math.isfinite(family_fdr_q_value):
        raise ValueError("selection evidence is incomplete")
    limits = dict(raw.get("limits_evidence", {}) or {})

    walk_forward_evidence = copy.deepcopy(
        dict(raw.get("walk_forward_evidence", {}) or {})
    )
    walk_forward_result = validate_purged_walk_forward(
        walk_forward_evidence
    )
    if not walk_forward_result["passed"]:
        raise ValueError(
            "walk-forward evidence invalid: "
            + ",".join(walk_forward_result["blockers"])
        )
    normalized_folds: list[dict[str, Any]] = []
    for index, raw_fold in enumerate(walk_forward_evidence["folds"]):
        fold = dict(raw_fold)
        fold["evidence_hash"] = _sha256_hex(
            fold.get("evidence_hash"),
            f"walk_forward_evidence.folds[{index}].evidence_hash",
        )
        normalized_folds.append(fold)
    walk_forward_evidence["folds"] = normalized_folds

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "producer": {
            "producer_id": PRODUCER_ID,
            "producer_version": PRODUCER_VERSION,
            "producer_code_hash": producer_code_hash(),
            "artifact_bytes_readback_bound": False,
            "production_apply_eligible": False,
            "production_apply_blocker": (
                CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER
            ),
        },
        "source_replay_sha256": _hash(raw),
        "as_of": str(raw.get("as_of", ""))[:10],
        "snapshot_evidence": snapshot,
        "slot": slot.to_dict(),
        "incumbent": slot.incumbent,
        "challenger_record": challenger.to_dict(),
        "evidence_window": evidence_window.to_dict(),
        "arms": arms,
        "health_evidence": {
            "failure_window_ids": failure_windows,
            "artifact_hash": health_hash,
        },
        "selection_evidence": {
            "family_fdr_q_value": family_fdr_q_value,
            "artifact_hash": selection_hash,
        },
        "limits_evidence": {
            key: _normalize_limit(limits.get(key), key)
            for key in ("turnover", "slippage", "tail_risk")
        },
        "walk_forward_evidence": walk_forward_evidence,
    }
    if evidence["as_of"] not in valid_days:
        raise ValueError("as_of is not in strict snapshot valid_trading_days")
    evidence["evidence_hash"] = _hash(evidence)
    verify_governance_replay_evidence(evidence)
    return evidence


def verify_governance_replay_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = copy.deepcopy(dict(payload))
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported governance replay evidence schema")
    if evidence.get("protocol_hash") != protocol_hash():
        raise ValueError("governance replay protocol hash mismatch")
    producer = dict(evidence.get("producer", {}) or {})
    if producer != {
        "producer_id": PRODUCER_ID,
        "producer_version": PRODUCER_VERSION,
        "producer_code_hash": producer_code_hash(),
        "artifact_bytes_readback_bound": False,
        "production_apply_eligible": False,
        "production_apply_blocker": CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
    }:
        raise ValueError("governance replay evidence producer mismatch")
    supplied_hash = _sha256_hex(
        evidence.pop("evidence_hash", ""),
        "evidence_hash",
    )
    if supplied_hash != _hash(evidence):
        raise ValueError("governance replay evidence hash mismatch")
    evidence["evidence_hash"] = supplied_hash
    snapshot = dict(evidence.get("snapshot_evidence", {}) or {})
    _sha256_hex(
        snapshot.get("manifest_sha256"),
        "snapshot_evidence.manifest_sha256",
    )
    valid_days = _strict_dates(
        list(snapshot.get("valid_trading_days", []) or []),
        "snapshot_evidence.valid_trading_days",
    )
    if snapshot.get("valid_trading_days_sha256") != _hash(valid_days):
        raise ValueError("valid trading days hash mismatch")
    snapshot_without_hash = dict(snapshot)
    supplied_snapshot_hash = snapshot_without_hash.pop(
        "snapshot_evidence_hash", ""
    )
    if supplied_snapshot_hash != _hash(snapshot_without_hash):
        raise ValueError("snapshot evidence hash mismatch")
    arms = dict(evidence.get("arms", {}) or {})
    if set(arms) != set(ARM_NAMES):
        raise ValueError("governance replay evidence arms are incomplete")
    for name in ARM_NAMES:
        arm = dict(arms[name])
        _stage_hashes(arm.get("stage_artifact_hashes"), f"arm_{name}")
        supplied_arm_hash = arm.pop("arm_hash", "")
        if supplied_arm_hash != replay_arm_hash(arm):
            raise ValueError(f"arm {name} hash mismatch")
    health = dict(evidence.get("health_evidence", {}) or {})
    selection = dict(evidence.get("selection_evidence", {}) or {})
    _sha256_hex(health.get("artifact_hash"), "health_evidence.artifact_hash")
    _sha256_hex(
        selection.get("artifact_hash"),
        "selection_evidence.artifact_hash",
    )
    for key, item in dict(evidence.get("limits_evidence", {}) or {}).items():
        _sha256_hex(
            dict(item).get("artifact_hash"),
            f"limits_evidence.{key}.artifact_hash",
        )
    return evidence


def load_governance_replay_evidence(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"governance replay evidence unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("governance replay evidence must be an object")
    return verify_governance_replay_evidence(payload)


def write_governance_replay_evidence(
    path: str | Path,
    evidence: Mapping[str, Any],
) -> None:
    verified = verify_governance_replay_evidence(evidence)
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        verified,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        dir=resolved.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, resolved)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_registry_mutation_plan_from_evidence(
    *,
    registry_path: str | Path,
    evidence: Mapping[str, Any],
    wal_path: str | Path,
    budget_ledger_path: str | Path,
) -> tuple[RegistryMutationPlan, list[str]]:
    verified = verify_governance_replay_evidence(evidence)
    snapshot = load_registry_snapshot_strict(registry_path)
    slot = FactorSlot.from_dict(dict(verified["slot"]))
    incumbent = next(
        (
            record
            for record in snapshot.registry.factors
            if record.name == slot.incumbent
        ),
        None,
    )
    if incumbent is None:
        raise ValueError("evidence incumbent is missing from registry")
    challenger = FactorRecord.from_dict(dict(verified["challenger_record"]))
    evidence_hash = str(verified["evidence_hash"])
    transition = FactorTransitionPlan(
        transition_id=(
            f"protocol-v2:{verified['as_of']}:{slot.slot_id}:"
            f"{evidence_hash[:12]}"
        ),
        as_of=str(verified["as_of"]),
        slot=slot,
        incumbent=slot.incumbent,
        challenger=slot.reserve,
        evidence_window=FactorEvidenceWindow.from_dict(
            dict(verified["evidence_window"])
        ),
        # Preserve the complete content-addressed producer artifact.  The
        # transition engine re-runs its verifier; extracted/self-reported
        # fields are never sufficient to authorize a mutation.
        arm_evidence=copy.deepcopy(verified),
        before_weights={
            incumbent.name: float(incumbent.weight),
            challenger.name: float(challenger.weight),
        },
        after_weights={
            incumbent.name: 0.0,
            challenger.name: abs(float(incumbent.weight)),
        },
        blockers=[],
        rollback={"mode": "inverse_wal"},
        evidence_hash=evidence_hash,
    )
    plan = RegistryMutationPlan(
        transition=transition,
        expected_registry_sha256=snapshot.registry_sha256,
        target_record_names=[incumbent.name, challenger.name],
        metadata_updates={
            "factor_governance_evidence_schema": EVIDENCE_SCHEMA_VERSION,
            "factor_governance_evidence_hash": evidence_hash,
            "factor_governance_production_apply_eligible": False,
            "factor_governance_production_apply_blocker": (
                canonical_replay_producer_control()["blocker"]
            ),
        },
        wal_path=str(Path(wal_path).expanduser()),
        budget_ledger_path=str(Path(budget_ledger_path).expanduser()),
        evidence_hash=evidence_hash,
        challenger_record_payload=challenger.to_dict(),
        inverse_patch_required=True,
    )
    return plan, list(verified["snapshot_evidence"]["valid_trading_days"])


__all__ = [
    "ARM_NAMES",
    "CONTROL_CHAIN_STAGES",
    "EVIDENCE_SCHEMA_VERSION",
    "PRODUCER_ID",
    "PRODUCER_VERSION",
    "RAW_REPLAY_SCHEMA_VERSION",
    "SNAPSHOT_EVIDENCE_SCHEMA_VERSION",
    "build_registry_mutation_plan_from_evidence",
    "load_governance_replay_evidence",
    "produce_governance_replay_evidence",
    "producer_code_hash",
    "replay_arm_hash",
    "verify_governance_replay_evidence",
    "write_governance_replay_evidence",
]
