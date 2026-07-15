"""FactorGovernanceProtocol v2 contracts and fail-closed transition engine.

The module is deliberately deterministic and offline.  It does not mine a
factor or manufacture portfolio evidence.  It validates evidence produced by
the existing health, selection-shadow and full v13 control-chain replays, then
builds one record-scoped CAS/WAL mutation for one factor slot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import fcntl
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.governance import (
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
)
from quant_investor.factors.registry_store import (
    METADATA_ABSENT,
    apply_factor_record_patch,
    factor_record_sha256,
    load_registry_snapshot_strict,
)
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    production_factor_set_sha256,
)
from quant_investor.factors.runtime_contract import (
    validate_production_runtime_contracts,
    validate_quant_production_activation,
)


PROTOCOL_VERSION = "v2"
PROTOCOL_SCHEMA_VERSION = "factor-governance-protocol.v2"
FACTOR_SLOT_SCHEMA_VERSION = "factor-slot.v2"
EVIDENCE_WINDOW_SCHEMA_VERSION = "factor-evidence-window.v2"
TRANSITION_PLAN_SCHEMA_VERSION = "factor-transition-plan.v2"
REGISTRY_MUTATION_PLAN_SCHEMA_VERSION = "registry-mutation-plan.v2"
MUTATION_BUDGET_LEDGER_SCHEMA_VERSION = "factor-mutation-budget-ledger-entry.v1"

MIN_MONTH_END_RANKIC_COUNT = 12
MIN_NONOVERLAP_30D_COHORT_COUNT = 8
MIN_PURGE_DAYS = 30
REQUIRED_EMBARGO_DAYS = 30
FDR_Q = 0.10
MIN_ANNUALIZED_NET_EXCESS_IMPROVEMENT = 0.01
MAX_DRAWDOWN_WORSENING = 0.02
MIN_COVERAGE_RATIO_TO_INCUMBENT = 0.95
MAX_FACTOR_ABS_WEIGHT = 0.20
MAX_FAMILY_ABS_WEIGHT = 0.35
BOOTSTRAP_MIN_SAMPLES = 60
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1729
CANONICAL_FULL_CHAIN_PRODUCER_AVAILABLE = False
CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER = (
    "canonical_full_chain_replay_producer_unavailable"
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


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _sha256_hex(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _date(value: Any, label: str) -> date:
    text = _nonempty(value, label)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO YYYY-MM-DD") from exc


def canonical_replay_producer_control() -> dict[str, Any]:
    """Describe the hard production boundary for canonical replay evidence.

    The current JSON normalizer can make evidence deterministic and
    content-addressed, but it cannot prove that caller-supplied artifact hashes
    were read back from actual v13 DAG outputs.  Forward production mutation
    therefore remains unavailable until that readback-bound producer exists.
    """

    return {
        "producer_available": CANONICAL_FULL_CHAIN_PRODUCER_AVAILABLE,
        "artifact_bytes_readback_bound": False,
        "production_apply_eligible": False,
        "blocker": CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
    }


def protocol_policy() -> dict[str, Any]:
    """Return the immutable policy payload hashed by every apply command."""

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "lifecycle": [
            "research_candidate",
            "shadow",
            "mature_candidate",
            "production_candidate",
            "production_factor",
            "watch",
            "reduced",
            "deprecated",
        ],
        "candidate_maturity": {
            "month_end_rankic_count": MIN_MONTH_END_RANKIC_COUNT,
            "nonoverlap_30d_cohort_count": MIN_NONOVERLAP_30D_COHORT_COUNT,
        },
        "health": {
            "reduce_after_independent_failures": 2,
            "deprecate_after_independent_failures": 3,
            "data_blocked_changes_streak": False,
        },
        "multiple_testing": {"method": "benjamini_hochberg", "q": FDR_Q},
        "walk_forward": {
            "purged": True,
            "min_purge_days": MIN_PURGE_DAYS,
            "embargo_days": REQUIRED_EMBARGO_DAYS,
        },
        "c_arm": {
            "evidence_schema": "factor-governance-replay-evidence.v2",
            "evidence_producer": "myquant.factor_governance_replay_evidence",
            "recompute_from_after_cost_arm_arrays": True,
            "strict_snapshot_trading_calendar_hash_required": True,
            "paired_delta_ci95_lower_gt": 0.0,
            "annualized_net_excess_improvement_gte": (
                MIN_ANNUALIZED_NET_EXCESS_IMPROVEMENT
            ),
            "max_drawdown_worsening_lte": MAX_DRAWDOWN_WORSENING,
            "coverage_ratio_to_incumbent_gte": MIN_COVERAGE_RATIO_TO_INCUMBENT,
            "full_control_chain": [
                "quant",
                "theme",
                "bayesian",
                "risk_guard",
                "portfolio_constructor",
            ],
            "paired_delta_bootstrap": {
                "method": "moving_block_bootstrap",
                "minimum_sample_count": BOOTSTRAP_MIN_SAMPLES,
                "confidence": 0.95,
                "block_length": BOOTSTRAP_BLOCK_LENGTH,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
        },
        "canonical_replay_producer_control": (
            canonical_replay_producer_control()
        ),
        "mutation": {
            "last_valid_trading_day_only": True,
            "max_targeted_swaps_per_month": 1,
            "independent_append_only_budget_ledger": True,
            "rollback_refunds_monthly_budget": False,
            "cas": True,
            "wal": True,
            "inverse_patch": True,
        },
        "risk_budget": {
            "max_factor_normalized_abs_weight": MAX_FACTOR_ABS_WEIGHT,
            "max_family_normalized_abs_weight": MAX_FAMILY_ABS_WEIGHT,
            "insufficient_evidence": "retain_previous_weights",
        },
        "zero_production": {
            "status": "governance_blocked",
            "legacy_proxy_fallback": False,
        },
    }


def protocol_hash() -> str:
    return _sha256(protocol_policy())


PROTOCOL_HASH = protocol_hash()


@dataclass(frozen=True)
class FactorSlot:
    family: str
    dominant_primitive_cluster: str
    incumbent: str
    reserve: str
    schema_version: str = FACTOR_SLOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FACTOR_SLOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported FactorSlot schema {self.schema_version!r}")
        object.__setattr__(self, "family", _nonempty(self.family, "family"))
        object.__setattr__(
            self,
            "dominant_primitive_cluster",
            _nonempty(
                self.dominant_primitive_cluster,
                "dominant_primitive_cluster",
            ),
        )
        object.__setattr__(self, "incumbent", _nonempty(self.incumbent, "incumbent"))
        object.__setattr__(self, "reserve", _nonempty(self.reserve, "reserve"))
        if self.incumbent == self.reserve:
            raise ValueError("slot incumbent and reserve must differ")

    @property
    def slot_id(self) -> str:
        return f"{self.family}::{self.dominant_primitive_cluster}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "slot_id": self.slot_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorSlot":
        return cls(
            schema_version=str(payload.get("schema_version", FACTOR_SLOT_SCHEMA_VERSION)),
            family=str(payload.get("family", "")),
            dominant_primitive_cluster=str(
                payload.get("dominant_primitive_cluster", "")
            ),
            incumbent=str(payload.get("incumbent", "")),
            reserve=str(payload.get("reserve", "")),
        )


@dataclass(frozen=True)
class FactorEvidenceWindow:
    window_id: str
    snapshot_id: str
    data_hash: str
    code_hash: str
    cost_hash: str
    forward_cohort_ids: list[str]
    month_end_rankic_dates: list[str]
    evaluation_hash: str
    purge_days: int
    embargo_days: int
    walk_forward_fold_count: int
    observed_at: str
    forward_cohorts: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = EVIDENCE_WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_WINDOW_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FactorEvidenceWindow schema {self.schema_version!r}"
            )
        for field_name in ("window_id", "snapshot_id", "observed_at"):
            object.__setattr__(self, field_name, _nonempty(getattr(self, field_name), field_name))
        for field_name in ("data_hash", "code_hash", "cost_hash", "evaluation_hash"):
            object.__setattr__(
                self,
                field_name,
                _sha256_hex(getattr(self, field_name), field_name),
            )
        cohorts = [str(item).strip() for item in self.forward_cohort_ids if str(item).strip()]
        month_ends = [
            str(item).strip() for item in self.month_end_rankic_dates if str(item).strip()
        ]
        object.__setattr__(self, "forward_cohort_ids", list(dict.fromkeys(cohorts)))
        object.__setattr__(self, "month_end_rankic_dates", list(dict.fromkeys(month_ends)))
        object.__setattr__(
            self,
            "forward_cohorts",
            [copy.deepcopy(dict(item)) for item in self.forward_cohorts],
        )
        if self.forward_cohorts:
            cohort_record_ids = [
                str(item.get("cohort_id", "") or "").strip()
                for item in self.forward_cohorts
            ]
            if (
                any(not item for item in cohort_record_ids)
                or len(set(cohort_record_ids)) != len(cohort_record_ids)
                or set(cohort_record_ids) != set(self.forward_cohort_ids)
            ):
                raise ValueError(
                    "forward_cohort_ids must exactly match distinct forward_cohorts"
                )
        object.__setattr__(self, "purge_days", int(self.purge_days))
        object.__setattr__(self, "embargo_days", int(self.embargo_days))
        object.__setattr__(self, "walk_forward_fold_count", int(self.walk_forward_fold_count))
        if self.purge_days < MIN_PURGE_DAYS:
            raise ValueError(f"purge_days must be >= {MIN_PURGE_DAYS}")
        if self.embargo_days != REQUIRED_EMBARGO_DAYS:
            raise ValueError(f"embargo_days must equal {REQUIRED_EMBARGO_DAYS}")
        if self.walk_forward_fold_count <= 0:
            raise ValueError("walk_forward_fold_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorEvidenceWindow":
        return cls(
            schema_version=str(
                payload.get("schema_version", EVIDENCE_WINDOW_SCHEMA_VERSION)
            ),
            window_id=str(payload.get("window_id", "")),
            snapshot_id=str(payload.get("snapshot_id", "")),
            data_hash=str(payload.get("data_hash", "")),
            code_hash=str(payload.get("code_hash", "")),
            cost_hash=str(payload.get("cost_hash", "")),
            forward_cohort_ids=list(payload.get("forward_cohort_ids", []) or []),
            month_end_rankic_dates=list(
                payload.get("month_end_rankic_dates", []) or []
            ),
            evaluation_hash=str(payload.get("evaluation_hash", "")),
            purge_days=int(payload.get("purge_days", 0) or 0),
            embargo_days=int(payload.get("embargo_days", 0) or 0),
            walk_forward_fold_count=int(
                payload.get("walk_forward_fold_count", 0) or 0
            ),
            observed_at=str(payload.get("observed_at", "")),
            forward_cohorts=[
                dict(item)
                for item in payload.get("forward_cohorts", []) or []
                if isinstance(item, Mapping)
            ],
        )


@dataclass(frozen=True)
class FactorTransitionPlan:
    transition_id: str
    as_of: str
    slot: FactorSlot
    incumbent: str
    challenger: str
    evidence_window: FactorEvidenceWindow
    arm_evidence: dict[str, Any]
    before_weights: dict[str, float]
    after_weights: dict[str, float]
    blockers: list[str]
    rollback: dict[str, Any]
    evidence_hash: str
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = TRANSITION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSITION_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FactorTransitionPlan schema {self.schema_version!r}"
            )
        object.__setattr__(
            self, "transition_id", _nonempty(self.transition_id, "transition_id")
        )
        object.__setattr__(self, "as_of", _date(self.as_of, "as_of").isoformat())
        object.__setattr__(self, "incumbent", _nonempty(self.incumbent, "incumbent"))
        object.__setattr__(self, "challenger", _nonempty(self.challenger, "challenger"))
        object.__setattr__(
            self,
            "evidence_hash",
            _sha256_hex(self.evidence_hash, "evidence_hash"),
        )
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
        if self.incumbent != self.slot.incumbent or self.challenger != self.slot.reserve:
            raise ValueError("transition names must match slot incumbent/reserve")
        before = {
            str(key): _finite(value, f"before_weights[{key}]")
            for key, value in self.before_weights.items()
        }
        after = {
            str(key): _finite(value, f"after_weights[{key}]")
            for key, value in self.after_weights.items()
        }
        object.__setattr__(self, "before_weights", before)
        object.__setattr__(self, "after_weights", after)
        expected_weight_names = {self.incumbent, self.challenger}
        if set(before) != expected_weight_names or set(after) != expected_weight_names:
            raise ValueError(
                "before_weights and after_weights must contain exactly incumbent and challenger"
            )
        object.__setattr__(self, "arm_evidence", copy.deepcopy(dict(self.arm_evidence)))
        if self.arm_evidence.get("evidence_hash") != self.evidence_hash:
            raise ValueError("transition canonical evidence hash mismatch")
        object.__setattr__(self, "blockers", list(dict.fromkeys(str(item) for item in self.blockers)))
        object.__setattr__(self, "rollback", copy.deepcopy(dict(self.rollback)))

    @property
    def transition_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("transition_hash", None)
        return _sha256(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "transition_id": self.transition_id,
            "as_of": self.as_of,
            "slot": self.slot.to_dict(),
            "incumbent": self.incumbent,
            "challenger": self.challenger,
            "evidence_window": self.evidence_window.to_dict(),
            "arm_evidence": copy.deepcopy(self.arm_evidence),
            "before_weights": dict(self.before_weights),
            "after_weights": dict(self.after_weights),
            "blockers": list(self.blockers),
            "rollback": copy.deepcopy(self.rollback),
            "evidence_hash": self.evidence_hash,
        }
        payload["transition_hash"] = _sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorTransitionPlan":
        result = cls(
            schema_version=str(
                payload.get("schema_version", TRANSITION_PLAN_SCHEMA_VERSION)
            ),
            protocol_version=str(payload.get("protocol_version", "")),
            transition_id=str(payload.get("transition_id", "")),
            as_of=str(payload.get("as_of", "")),
            slot=FactorSlot.from_dict(dict(payload.get("slot", {}) or {})),
            incumbent=str(payload.get("incumbent", "")),
            challenger=str(payload.get("challenger", "")),
            evidence_window=FactorEvidenceWindow.from_dict(
                dict(payload.get("evidence_window", {}) or {})
            ),
            arm_evidence=dict(payload.get("arm_evidence", {}) or {}),
            before_weights=dict(payload.get("before_weights", {}) or {}),
            after_weights=dict(payload.get("after_weights", {}) or {}),
            blockers=list(payload.get("blockers", []) or []),
            rollback=dict(payload.get("rollback", {}) or {}),
            evidence_hash=str(payload.get("evidence_hash", "")),
        )
        supplied_hash = str(payload.get("transition_hash", "") or "")
        if supplied_hash and supplied_hash != result.transition_hash:
            raise ValueError("FactorTransitionPlan transition_hash mismatch")
        return result


@dataclass(frozen=True)
class RegistryMutationPlan:
    transition: FactorTransitionPlan
    expected_registry_sha256: str
    target_record_names: list[str]
    metadata_updates: dict[str, Any]
    wal_path: str
    budget_ledger_path: str
    evidence_hash: str
    challenger_record_payload: dict[str, Any]
    inverse_patch_required: bool
    protocol_hash: str = field(default_factory=protocol_hash)
    schema_version: str = REGISTRY_MUTATION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_MUTATION_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RegistryMutationPlan schema {self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "expected_registry_sha256",
            _sha256_hex(
                self.expected_registry_sha256,
                "expected_registry_sha256",
            ),
        )
        object.__setattr__(self, "wal_path", _nonempty(self.wal_path, "wal_path"))
        object.__setattr__(
            self,
            "budget_ledger_path",
            _nonempty(self.budget_ledger_path, "budget_ledger_path"),
        )
        object.__setattr__(
            self,
            "evidence_hash",
            _sha256_hex(self.evidence_hash, "evidence_hash"),
        )
        names = sorted(
            {str(item).strip() for item in self.target_record_names if str(item).strip()}
        )
        object.__setattr__(self, "target_record_names", names)
        object.__setattr__(self, "metadata_updates", copy.deepcopy(dict(self.metadata_updates)))
        object.__setattr__(
            self,
            "challenger_record_payload",
            copy.deepcopy(dict(self.challenger_record_payload)),
        )
        if names != sorted([self.transition.incumbent, self.transition.challenger]):
            raise ValueError("target_record_names must be exactly incumbent and challenger")
        if self.protocol_hash != protocol_hash():
            raise ValueError("mutation plan protocol_hash mismatch")
        if self.evidence_hash != self.transition.evidence_hash:
            raise ValueError("mutation plan evidence_hash mismatch")
        if not self.inverse_patch_required:
            raise ValueError("inverse_patch_required must be true")
        challenger_record = FactorRecord.from_dict(self.challenger_record_payload)
        if challenger_record.name != self.transition.challenger:
            raise ValueError("challenger_record_payload name mismatch")

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_hash": self.protocol_hash,
            "expected_registry_sha256": self.expected_registry_sha256,
            "target_record_names": list(self.target_record_names),
            "metadata_updates": copy.deepcopy(self.metadata_updates),
            "wal_path": self.wal_path,
            "budget_ledger_path": self.budget_ledger_path,
            "evidence_hash": self.evidence_hash,
            "challenger_record_payload": copy.deepcopy(
                self.challenger_record_payload
            ),
            "inverse_patch_required": self.inverse_patch_required,
            "transition": self.transition.to_dict(),
        }

    @property
    def mutation_plan_hash(self) -> str:
        return _sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["mutation_plan_hash"] = self.mutation_plan_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistryMutationPlan":
        result = cls(
            schema_version=str(
                payload.get("schema_version", REGISTRY_MUTATION_PLAN_SCHEMA_VERSION)
            ),
            protocol_hash=str(payload.get("protocol_hash", "")),
            transition=FactorTransitionPlan.from_dict(
                dict(payload.get("transition", {}) or {})
            ),
            expected_registry_sha256=str(
                payload.get("expected_registry_sha256", "")
            ),
            target_record_names=list(payload.get("target_record_names", []) or []),
            metadata_updates=dict(payload.get("metadata_updates", {}) or {}),
            wal_path=str(payload.get("wal_path", "")),
            budget_ledger_path=str(payload.get("budget_ledger_path", "")),
            evidence_hash=str(payload.get("evidence_hash", "")),
            challenger_record_payload=dict(
                payload.get("challenger_record_payload", {}) or {}
            ),
            inverse_patch_required=bool(payload.get("inverse_patch_required", False)),
        )
        supplied_hash = str(payload.get("mutation_plan_hash", "") or "")
        if supplied_hash and supplied_hash != result.mutation_plan_hash:
            raise ValueError("RegistryMutationPlan mutation_plan_hash mismatch")
        return result


def assess_candidate_maturity(
    *,
    month_end_rankic_dates: Sequence[str],
    forward_cohorts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    month_ends: list[str] = []
    for raw in month_end_rankic_dates:
        try:
            value = _date(raw, "month_end_rankic_date").isoformat()
        except ValueError:
            continue
        if value not in month_ends:
            month_ends.append(value)

    valid: list[tuple[date, date, str]] = []
    for raw in forward_cohorts:
        try:
            start = _date(raw.get("start"), "forward_cohort.start")
            end = _date(raw.get("end"), "forward_cohort.end")
            horizon = int(raw.get("horizon_days", 0) or 0)
            cohort_id = _nonempty(raw.get("cohort_id"), "forward_cohort.cohort_id")
        except (TypeError, ValueError):
            continue
        if horizon != 30 or end <= start:
            continue
        valid.append((start, end, cohort_id))
    valid.sort()
    nonoverlap: list[tuple[date, date, str]] = []
    last_end: date | None = None
    seen_ids: set[str] = set()
    for start, end, cohort_id in valid:
        if cohort_id in seen_ids or (last_end is not None and start <= last_end):
            continue
        nonoverlap.append((start, end, cohort_id))
        seen_ids.add(cohort_id)
        last_end = end

    mature_by_month_end = len(month_ends) >= MIN_MONTH_END_RANKIC_COUNT
    mature_by_cohort = len(nonoverlap) >= MIN_NONOVERLAP_30D_COHORT_COUNT
    return {
        "mature": mature_by_month_end or mature_by_cohort,
        "maturity_route": (
            "month_end_rankic"
            if mature_by_month_end
            else "nonoverlap_30d_forward_cohort"
            if mature_by_cohort
            else "insufficient"
        ),
        "month_end_rankic_count": len(month_ends),
        "nonoverlap_30d_cohort_count": len(nonoverlap),
        "month_end_rankic_dates": month_ends,
        "nonoverlap_30d_cohort_ids": [item[2] for item in nonoverlap],
        "required_month_end_rankic_count": MIN_MONTH_END_RANKIC_COUNT,
        "required_nonoverlap_30d_cohort_count": MIN_NONOVERLAP_30D_COHORT_COUNT,
    }


def benjamini_hochberg_by_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    q: float = FDR_Q,
) -> list[dict[str, Any]]:
    q = _finite(q, "q")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    families: dict[str, list[dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = dict(raw)
        name = _nonempty(item.get("name"), "name")
        family = _nonempty(item.get("family"), "family")
        p_value = _finite(item.get("p_value"), "p_value")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("p_value must be in [0, 1]")
        item.update(
            {
                "name": name,
                "family": family,
                "p_value": p_value,
                "_input_index": index,
            }
        )
        families.setdefault(family, []).append(item)

    for family, members in families.items():
        ordered = sorted(members, key=lambda item: (item["p_value"], item["name"]))
        count = len(ordered)
        max_passing_rank = 0
        for rank, item in enumerate(ordered, start=1):
            if item["p_value"] <= q * rank / count:
                max_passing_rank = rank
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
                    "fdr_passed": rank <= max_passing_rank,
                    "fdr_method": "benjamini_hochberg_by_family",
                }
            )
            output.append(item)
    output.sort(key=lambda item: item.pop("_input_index"))
    return output


def validate_purged_walk_forward(evidence: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if evidence.get("purged") is not True:
        blockers.append("walk_forward_not_purged")
    try:
        purge_days = int(evidence.get("purge_days", 0) or 0)
    except (TypeError, ValueError):
        purge_days = 0
    if purge_days < MIN_PURGE_DAYS:
        blockers.append("purge_days_below_30")
    try:
        embargo_days = int(evidence.get("embargo_days", 0) or 0)
    except (TypeError, ValueError):
        embargo_days = 0
    if embargo_days != REQUIRED_EMBARGO_DAYS:
        blockers.append("embargo_days_not_30")
    folds = evidence.get("folds")
    if not isinstance(folds, list) or not folds:
        blockers.append("walk_forward_folds_missing")
        folds = []
    for index, raw in enumerate(folds):
        if not isinstance(raw, Mapping):
            blockers.append(f"fold_{index}_not_object")
            continue
        try:
            train_end = _date(raw.get("train_end"), "train_end")
            validation_start = _date(raw.get("validation_start"), "validation_start")
            validation_end = _date(raw.get("validation_end"), "validation_end")
        except ValueError:
            blockers.append(f"fold_{index}_date_invalid")
            continue
        if validation_start <= train_end or validation_end <= validation_start:
            blockers.append(f"fold_{index}_chronology_invalid")
        if (validation_start - train_end).days < purge_days:
            blockers.append(f"fold_{index}_purge_gap_too_short")
        if not str(raw.get("evidence_hash", "") or "").strip():
            blockers.append(f"fold_{index}_evidence_hash_missing")
    return {
        "passed": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "fold_count": len(folds),
    }


def advance_failure_streak(
    failure_window_ids: Sequence[str],
    observation_status: str,
    maturity_window_id: str,
) -> dict[str, Any]:
    active = list(
        dict.fromkeys(
            str(item).strip()
            for item in failure_window_ids
            if str(item).strip() and str(item).strip() not in {"missing", "unknown"}
        )
    )
    status = str(observation_status or "").strip().lower()
    window_id = str(maturity_window_id or "").strip()
    if status == "healthy":
        active = []
    elif status == "failure" and window_id and window_id not in active:
        active.append(window_id)
    elif status not in {"data_blocked", "failure", "healthy"}:
        raise ValueError("observation_status must be failure, healthy or data_blocked")
    return {
        "failure_window_ids": active,
        "failure_count": len(active),
        "health_state": (
            "deprecated_eligible"
            if len(active) >= 3
            else "reduced_eligible"
            if len(active) >= 2
            else "watch"
            if active
            else "healthy"
        ),
        "data_blocked_streak_unchanged": status == "data_blocked",
    }


def block_bootstrap_paired_delta_ci(
    paired_after_cost_daily_deltas: Sequence[float],
    *,
    confidence: float = 0.95,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute an auditable moving-block bootstrap CI for paired daily deltas."""

    values = [
        _finite(value, "paired_after_cost_daily_delta")
        for value in paired_after_cost_daily_deltas
    ]
    confidence = _finite(confidence, "confidence")
    block_length = int(block_length)
    resamples = int(resamples)
    seed = int(seed)
    if len(values) < BOOTSTRAP_MIN_SAMPLES:
        raise ValueError(
            "paired after-cost daily deltas require at least "
            f"{BOOTSTRAP_MIN_SAMPLES} samples"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if block_length <= 0 or block_length > len(values):
        raise ValueError("block_length must be in [1, sample_count]")
    if resamples < 500:
        raise ValueError("resamples must be >= 500")

    rng = random.Random(seed)
    sample_count = len(values)
    max_start = sample_count - block_length
    block_count = math.ceil(sample_count / block_length)
    means: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for _block in range(block_count):
            start = rng.randint(0, max_start)
            sample.extend(values[start:start + block_length])
        means.append(sum(sample[:sample_count]) / sample_count)
    means.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(means) - 1)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return means[lower_index]
        fraction = position - lower_index
        return means[lower_index] * (1.0 - fraction) + means[upper_index] * fraction

    alpha = 1.0 - confidence
    config = {
        "method": "moving_block_bootstrap",
        "confidence": confidence,
        "block_length": block_length,
        "resamples": resamples,
        "seed": seed,
        "sample_count": sample_count,
    }
    return {
        **config,
        "sample_hash": _sha256(values),
        "config_hash": _sha256(config),
        "mean_daily_delta": sum(values) / sample_count,
        "ci_lower": percentile(alpha / 2.0),
        "ci_upper": percentile(1.0 - alpha / 2.0),
    }


def _max_drawdown(returns: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    worst = 0.0
    for raw in returns:
        value = _finite(raw, "after_cost_daily_return")
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        if peak > 0.0:
            worst = min(worst, wealth / peak - 1.0)
    return worst


def control_chain_evidence_hash(evidence: Mapping[str, Any]) -> str:
    """Hash the actual A/B/C/D arrays and coverage counts used by C-arm gates."""

    keys = (
        "source",
        "quant",
        "theme",
        "bayesian",
        "risk_guard",
        "portfolio_constructor",
        "a_arm_hash",
        "b_arm_hash",
        "c_arm_hash",
        "d_arm_hash",
        "paired_after_cost_daily_deltas",
        "paired_after_cost_daily_delta_hash",
        "bootstrap_block_length",
        "bootstrap_resamples",
        "bootstrap_seed",
        "incumbent_after_cost_daily_returns",
        "challenger_after_cost_daily_returns",
        "candidate_covered_observations",
        "incumbent_covered_observations",
        "coverage_total_observations",
    )
    return _sha256({key: copy.deepcopy(evidence.get(key)) for key in keys})


def evaluate_c_arm(evidence: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    canonical_evidence = copy.deepcopy(dict(evidence))
    try:
        # Local import avoids the module-load cycle: the canonical producer
        # imports these protocol contracts, while apply time must re-run the
        # producer verifier before trusting any array or hash.
        from quant_investor.factors.governance_evidence import (
            EVIDENCE_SCHEMA_VERSION,
            verify_governance_replay_evidence,
        )

        canonical_evidence = verify_governance_replay_evidence(
            canonical_evidence
        )
    except (OSError, TypeError, ValueError) as exc:
        EVIDENCE_SCHEMA_VERSION = "factor-governance-replay-evidence.v2"
        blockers.append(f"canonical_governance_evidence_invalid:{exc}")
    if canonical_evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        blockers.append("canonical_governance_evidence_missing")
    producer_evidence_hash = str(
        canonical_evidence.get("evidence_hash", "") or ""
    ).strip()
    if not producer_evidence_hash:
        blockers.append("producer_evidence_hash_missing")

    snapshot = dict(
        canonical_evidence.get("snapshot_evidence", {}) or {}
    )
    valid_days = [
        str(item) for item in snapshot.get("valid_trading_days", []) or []
    ]
    if snapshot.get("source") != "strict_parquet_snapshot":
        blockers.append("strict_snapshot_evidence_missing")
    if not str(snapshot.get("snapshot_id", "") or "").strip():
        blockers.append("snapshot_id_missing")
    if not str(snapshot.get("manifest_sha256", "") or "").strip():
        blockers.append("snapshot_manifest_hash_missing")
    if (
        not valid_days
        or valid_days != sorted(valid_days)
        or len(valid_days) != len(set(valid_days))
    ):
        blockers.append("valid_trading_days_invalid")
    if snapshot.get("valid_trading_days_sha256") != _sha256(valid_days):
        blockers.append("valid_trading_days_hash_mismatch")

    raw_arms = canonical_evidence.get("arms")
    arms = dict(raw_arms) if isinstance(raw_arms, Mapping) else {}
    arm_names = ("A", "B", "C", "D")
    stage_names = {
        "quant",
        "theme",
        "bayesian",
        "risk_guard",
        "portfolio_constructor",
    }
    normalized_arms: dict[str, dict[str, Any]] = {}
    for arm_name in arm_names:
        arm = dict(arms.get(arm_name, {}) or {})
        dates = [str(item) for item in arm.get("trading_dates", []) or []]
        returns = list(arm.get("after_cost_daily_returns", []) or [])
        stages = dict(arm.get("stage_artifact_hashes", {}) or {})
        if (
            not dates
            or len(dates) != len(returns)
            or dates != sorted(dates)
            or len(dates) != len(set(dates))
        ):
            blockers.append(f"arm_{arm_name}_date_return_contract_invalid")
        if any(item not in set(valid_days) for item in dates):
            blockers.append(f"arm_{arm_name}_non_snapshot_trading_date")
        if set(stages) != stage_names or any(
            not str(value or "").strip() for value in stages.values()
        ):
            blockers.append(f"arm_{arm_name}_control_chain_incomplete")
        arm_payload = {
            "trading_dates": dates,
            "after_cost_daily_returns": returns,
            "stage_artifact_hashes": stages,
        }
        computed_arm_hash = _sha256(arm_payload)
        if arm.get("arm_hash") != computed_arm_hash:
            blockers.append(f"arm_{arm_name}_hash_mismatch")
        normalized_arms[arm_name] = {
            **arm_payload,
            "arm_hash": computed_arm_hash,
        }
    if normalized_arms and any(
        normalized_arms[name]["trading_dates"]
        != normalized_arms["A"]["trading_dates"]
        for name in arm_names
    ):
        blockers.append("arm_trading_date_contract_mismatch")

    a_returns = normalized_arms.get("A", {}).get(
        "after_cost_daily_returns", []
    )
    c_returns = normalized_arms.get("C", {}).get(
        "after_cost_daily_returns", []
    )
    total_observations = max(len(a_returns), len(c_returns))
    incumbent_covered = sum(item is not None for item in a_returns)
    candidate_covered = sum(item is not None for item in c_returns)
    incumbent_returns: list[float] = []
    challenger_returns: list[float] = []
    for raw, label, target in (
        (a_returns, "incumbent_after_cost_return", incumbent_returns),
        (c_returns, "challenger_after_cost_return", challenger_returns),
    ):
        for item in raw:
            if item is None:
                continue
            try:
                target.append(_finite(item, label))
            except ValueError:
                blockers.append("after_cost_return_array_invalid")
    paired_deltas: list[float] = []
    if len(a_returns) == len(c_returns):
        for incumbent_raw, challenger_raw in zip(a_returns, c_returns):
            if incumbent_raw is None or challenger_raw is None:
                continue
            try:
                incumbent_value = _finite(
                    incumbent_raw, "incumbent_after_cost_return"
                )
                challenger_value = _finite(
                    challenger_raw, "challenger_after_cost_return"
                )
            except ValueError:
                blockers.append("after_cost_return_array_invalid")
                continue
            paired_deltas.append(challenger_value - incumbent_value)
    else:
        blockers.append("paired_after_cost_return_length_mismatch")

    try:
        bootstrap = block_bootstrap_paired_delta_ci(
            paired_deltas,
            confidence=0.95,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        ci_lower = float(bootstrap["ci_lower"])
    except (TypeError, ValueError):
        bootstrap = {}
        ci_lower = float("nan")
        blockers.append("paired_delta_bootstrap_evidence_invalid")
    if not math.isfinite(ci_lower) or ci_lower <= 0.0:
        blockers.append("paired_delta_ci95_lower_not_positive")
    improvement = (
        float(bootstrap["mean_daily_delta"]) * 252.0
        if bootstrap
        else float("nan")
    )
    if (
        not math.isfinite(improvement)
        or improvement < MIN_ANNUALIZED_NET_EXCESS_IMPROVEMENT
    ):
        blockers.append("annualized_net_excess_improvement_below_1pp")
    try:
        incumbent_drawdown = _max_drawdown(incumbent_returns)
        challenger_drawdown = _max_drawdown(challenger_returns)
        drawdown_delta = abs(challenger_drawdown) - abs(incumbent_drawdown)
    except ValueError:
        incumbent_drawdown = float("nan")
        challenger_drawdown = float("nan")
        drawdown_delta = float("nan")
        blockers.append("after_cost_drawdown_samples_invalid")
    if not math.isfinite(drawdown_delta) or drawdown_delta > MAX_DRAWDOWN_WORSENING:
        blockers.append("max_drawdown_worsening_above_2pp")
    coverage_ratio = (
        candidate_covered / incumbent_covered
        if incumbent_covered > 0
        else float("nan")
    )
    if (
        total_observations <= 0
        or not math.isfinite(coverage_ratio)
        or coverage_ratio < MIN_COVERAGE_RATIO_TO_INCUMBENT
    ):
        blockers.append("coverage_below_95pct_of_incumbent")

    limits = dict(
        canonical_evidence.get("limits_evidence", {}) or {}
    )
    for key in ("turnover", "slippage", "tail_risk"):
        item = dict(limits.get(key, {}) or {})
        try:
            measured = _finite(item.get("measured"), f"{key}_measured")
            limit = _finite(item.get("limit"), f"{key}_limit")
        except ValueError:
            blockers.append(f"{key}_limit_evidence_invalid")
            continue
        if not str(item.get("artifact_hash", "") or "").strip():
            blockers.append(f"{key}_evidence_hash_missing")
        if measured > limit:
            blockers.append(f"{key}_strategy_limit_exceeded")

    health_evidence = dict(
        canonical_evidence.get("health_evidence", {}) or {}
    )
    failure_ids = list(
        dict.fromkeys(
            str(item).strip()
            for item in health_evidence.get("failure_window_ids", []) or []
            if str(item).strip()
        )
    )
    failure_count = len(failure_ids)
    if not str(health_evidence.get("artifact_hash", "") or "").strip():
        blockers.append("health_evidence_hash_missing")
    if failure_count < 3:
        blockers.append("incumbent_not_deprecation_eligible")
    selection_evidence = dict(
        canonical_evidence.get("selection_evidence", {}) or {}
    )
    try:
        fdr_q_value = _finite(
            selection_evidence.get("family_fdr_q_value"), "fdr"
        )
    except ValueError:
        fdr_q_value = float("nan")
    if not str(selection_evidence.get("artifact_hash", "") or "").strip():
        blockers.append("selection_evidence_hash_missing")
    if not math.isfinite(fdr_q_value) or fdr_q_value > FDR_Q:
        blockers.append("challenger_family_fdr_above_0.10")

    arm_hashes = {
        name: normalized_arms.get(name, {}).get("arm_hash", "")
        for name in arm_names
    }
    control_chain_hash = _sha256(
        {
            "snapshot_evidence_hash": snapshot.get(
                "snapshot_evidence_hash", ""
            ),
            "arm_hashes": arm_hashes,
            "producer_evidence_hash": producer_evidence_hash,
        }
    )
    return {
        "passed": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "producer_evidence_hash": producer_evidence_hash,
        "control_chain_hash": control_chain_hash,
        "arm_hashes": arm_hashes,
        "paired_after_cost_daily_delta_hash": _sha256(paired_deltas),
        "paired_after_cost_sample_count": len(paired_deltas),
        "after_cost_paired_delta_ci95_lower": (
            ci_lower if math.isfinite(ci_lower) else None
        ),
        "paired_delta_bootstrap": bootstrap,
        "annualized_net_excess_improvement": (
            improvement if math.isfinite(improvement) else None
        ),
        "max_drawdown_delta": (
            drawdown_delta if math.isfinite(drawdown_delta) else None
        ),
        "incumbent_max_drawdown": (
            incumbent_drawdown if math.isfinite(incumbent_drawdown) else None
        ),
        "challenger_max_drawdown": (
            challenger_drawdown if math.isfinite(challenger_drawdown) else None
        ),
        "candidate_covered_observations": candidate_covered,
        "incumbent_covered_observations": incumbent_covered,
        "coverage_total_observations": total_observations,
        "coverage_ratio_to_incumbent": (
            coverage_ratio if math.isfinite(coverage_ratio) else None
        ),
        "incumbent_distinct_mature_failure_count": failure_count,
        "challenger_family_fdr_q_value": (
            fdr_q_value if math.isfinite(fdr_q_value) else None
        ),
    }


def build_slot_risk_budget(
    *,
    previous_weights: Mapping[str, float],
    family_by_factor: Mapping[str, str],
    incumbent: str,
    challenger: str,
    evidence_sufficient: bool,
) -> dict[str, Any]:
    previous = {
        str(name): _finite(weight, f"weight[{name}]")
        for name, weight in previous_weights.items()
    }
    if not evidence_sufficient:
        return {
            "status": "retained_previous_weights",
            "weights": previous,
            "blockers": ["evidence_insufficient"],
        }
    if incumbent not in previous or challenger not in previous:
        return {
            "status": "blocked",
            "weights": previous,
            "blockers": ["slot_weight_missing"],
        }
    proposed = dict(previous)
    proposed[challenger] = abs(previous[incumbent])
    proposed[incumbent] = 0.0
    total = sum(abs(value) for value in proposed.values())
    if total <= 0.0:
        return {
            "status": "blocked",
            "weights": previous,
            "blockers": ["zero_total_abs_weight"],
        }
    normalized = {name: abs(value) / total for name, value in proposed.items()}
    blockers: list[str] = []
    if any(value > MAX_FACTOR_ABS_WEIGHT + 1e-12 for value in normalized.values()):
        blockers.append("factor_abs_weight_above_0.20")
    family_totals: dict[str, float] = {}
    for name, value in normalized.items():
        if value <= 1e-15:
            continue
        family = str(family_by_factor.get(name, "") or "")
        if not family:
            blockers.append(f"factor_family_missing:{name}")
            continue
        family_totals[family] = family_totals.get(family, 0.0) + value
    if any(value > MAX_FAMILY_ABS_WEIGHT + 1e-12 for value in family_totals.values()):
        blockers.append("family_abs_weight_above_0.35")
    return {
        "status": "blocked" if blockers else "ready",
        "weights": proposed if not blockers else previous,
        "normalized_abs_weights": normalized,
        "family_normalized_abs_weights": family_totals,
        "blockers": list(dict.fromkeys(blockers)),
    }


def governance_runtime_status(registry: MinedFactorRegistry) -> dict[str, Any]:
    """Validate the complete v2 runtime contract, not merely factor count.

    A selectable v1 record is useful historical evidence, but it is not enough
    to authorize production scoring under protocol v2.  Runtime readiness is
    therefore bound to the registry manifest, local protocol hash, explicit
    slot identities, slot/family risk budgets, and readback-bound canonical
    evidence.  Any missing or inconsistent field fails closed.
    """

    selectable = registry.selectable_factors()
    manifest = registry.selectable_manifest()
    metadata = dict(registry.metadata or {})
    blockers: list[str] = []
    if metadata.get("missing"):
        blockers.append("registry_missing")
    if metadata.get("load_error"):
        blockers.append("registry_load_error")
    if metadata.get("strict_load_error"):
        blockers.append("registry_strict_load_error")
    if metadata.get("strict_loader") is not True:
        blockers.append("registry_not_strictly_loaded")
    if not selectable:
        blockers.append("no_selectable_production_factors")
    if len(manifest["production_factor_names"]) != len(
        set(manifest["production_factor_names"])
    ):
        blockers.append("production_factor_names_not_unique")

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

    slot_records: dict[str, list[str]] = {}
    family_weights: dict[str, float] = {}
    normalized_weights: dict[str, float] = {}
    numeric_weights: dict[str, float] = {}
    raw_abs_weights: list[float] = []
    for record in selectable:
        try:
            weight = float(record.weight)
        except (TypeError, ValueError):
            weight = math.nan
        try:
            direction = float(record.direction)
        except (TypeError, ValueError):
            direction = math.nan
        raw_abs_weights.append(abs(weight))
        if not math.isfinite(weight):
            blockers.append(f"factor_weight_non_finite:{record.name}")
        elif abs(weight) <= 1e-15:
            blockers.append(f"factor_weight_zero_or_negligible:{record.name}")
        else:
            numeric_weights[record.name] = weight
        if not math.isfinite(direction):
            blockers.append(f"factor_direction_non_finite:{record.name}")
        elif abs(direction) <= 1e-15:
            blockers.append(f"factor_direction_zero_or_negligible:{record.name}")

    total_abs_weight = sum(raw_abs_weights)
    if selectable and not math.isfinite(total_abs_weight):
        blockers.append("production_factor_total_abs_weight_non_finite")
    elif selectable and total_abs_weight <= 1e-15:
        blockers.append("production_factor_total_abs_weight_zero")
    for record in selectable:
        family, cluster = _slot_identity(record)
        if not family:
            blockers.append(f"factor_family_missing:{record.name}")
        if not cluster:
            blockers.append(f"factor_slot_cluster_missing:{record.name}")
        if family and cluster:
            slot_records.setdefault(f"{family}::{cluster}", []).append(record.name)
        if (
            math.isfinite(total_abs_weight)
            and total_abs_weight > 1e-15
            and record.name in numeric_weights
        ):
            normalized = abs(numeric_weights[record.name]) / total_abs_weight
            normalized_weights[record.name] = normalized
            if normalized > MAX_FACTOR_ABS_WEIGHT + 1e-12:
                blockers.append(f"factor_abs_weight_above_0.20:{record.name}")
            if family:
                family_weights[family] = family_weights.get(family, 0.0) + normalized
    for slot_id, names in slot_records.items():
        if len(names) != 1:
            blockers.append(f"factor_slot_multiple_incumbents:{slot_id}")
    for family, weight in family_weights.items():
        if weight > MAX_FAMILY_ABS_WEIGHT + 1e-12:
            blockers.append(f"family_abs_weight_above_0.35:{family}")

    producer_control = canonical_replay_producer_control()
    if not producer_control.get("producer_available"):
        blockers.append(str(producer_control["blocker"]))
    if not producer_control.get("artifact_bytes_readback_bound"):
        blockers.append("canonical_evidence_not_readback_bound")
    if not producer_control.get("production_apply_eligible"):
        blockers.append("canonical_evidence_not_production_eligible")
    last_evidence_hash = str(
        metadata.get("factor_governance_last_evidence_hash") or ""
    ).strip()
    if (
        len(last_evidence_hash) != 64
        or any(char not in "0123456789abcdef" for char in last_evidence_hash)
    ):
        blockers.append("registry_canonical_evidence_hash_missing_or_invalid")
    if metadata.get("factor_governance_production_apply_eligible") is not True:
        blockers.append("registry_canonical_evidence_not_production_eligible")
    if metadata.get("factor_governance_evidence_schema") != (
        "factor-governance-replay-evidence.v2"
    ):
        blockers.append("registry_canonical_evidence_schema_mismatch")
    last_evaluation_hash = str(
        metadata.get("factor_governance_last_evaluation_hash") or ""
    ).strip()
    if (
        len(last_evaluation_hash) != 64
        or any(char not in "0123456789abcdef" for char in last_evaluation_hash)
    ):
        blockers.append("registry_canonical_evaluation_hash_missing_or_invalid")
    if str(
        metadata.get("factor_governance_production_apply_blocker") or ""
    ).strip():
        blockers.append("registry_canonical_evidence_has_apply_blocker")

    runtime_contract_status = validate_production_runtime_contracts(
        selectable,
        metadata,
    )
    blockers.extend(runtime_contract_status.get("blockers", []))
    activation_status = validate_quant_production_activation(
        metadata,
        manifest,
        str(runtime_contract_status.get("contracts_sha256") or ""),
        implementation_code_sha256s=dict(
            runtime_contract_status.get("implementation_code_sha256s", {}) or {}
        ),
        protocol_version=PROTOCOL_VERSION,
        protocol_hash_value=protocol_hash(),
    )
    blockers.extend(activation_status.get("blockers", []))

    blockers = list(dict.fromkeys(blockers))
    ready = not blockers
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "status": "ready" if ready else "governance_blocked",
        "factor_mode": (
            "governed_mined_factors" if ready else "governance_blocked"
        ),
        "confidence_multiplier": 1.0 if ready else 0.0,
        "legacy_fallback_allowed": False,
        "production_factor_count": len(selectable),
        "production_factor_names": manifest["production_factor_names"],
        "production_factor_set_sha256": manifest[
            "production_factor_set_sha256"
        ],
        "normalized_abs_weights": normalized_weights,
        "family_normalized_abs_weights": family_weights,
        "slot_incumbents": slot_records,
        "canonical_replay_producer_control": producer_control,
        "factor_runtime_contracts": dict(
            runtime_contract_status.get("contracts", {}) or {}
        ),
        "factor_runtime_contracts_sha256": str(
            runtime_contract_status.get("contracts_sha256") or ""
        ),
        "factor_runtime_implementation_code_sha256s": dict(
            runtime_contract_status.get("implementation_code_sha256s", {}) or {}
        ),
        "quant_production_activation": dict(activation_status),
        "blockers": blockers,
    }


def _budget_entry_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("entry_hash", None)
    return _sha256(body)


def load_mutation_budget_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Strictly read the append-only monthly mutation reservation ledger."""

    resolved = Path(path).expanduser()
    if not resolved.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"mutation budget ledger line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise ValueError(
                f"mutation budget ledger line {line_number} must be an object"
            )
        item = dict(row)
        if item.get("schema_version") != MUTATION_BUDGET_LEDGER_SCHEMA_VERSION:
            raise ValueError(
                f"mutation budget ledger line {line_number} has unsupported schema"
            )
        required = (
            "reservation_id",
            "month",
            "transition_id",
            "transition_hash",
            "mutation_plan_hash",
            "evidence_hash",
            "before_registry_sha256",
            "reserved_at",
            "entry_hash",
        )
        if any(not str(item.get(key, "") or "").strip() for key in required):
            raise ValueError(
                f"mutation budget ledger line {line_number} is incomplete"
            )
        if item["entry_hash"] != _budget_entry_hash(item):
            raise ValueError(
                f"mutation budget ledger line {line_number} hash mismatch"
            )
        rows.append(item)
    reservation_ids = [row["reservation_id"] for row in rows]
    if len(reservation_ids) != len(set(reservation_ids)):
        raise ValueError("mutation budget ledger has duplicate reservation_id")
    return rows


def monthly_mutation_budget_blockers(
    path: str | Path,
    month: str,
) -> list[str]:
    try:
        rows = load_mutation_budget_ledger(path)
    except (OSError, ValueError) as exc:
        return [f"mutation_budget_ledger_invalid:{exc}"]
    if any(str(row.get("month", "")) == month for row in rows):
        return ["monthly_transition_budget_exhausted"]
    return []


def reserve_monthly_mutation_budget(
    path: str | Path,
    *,
    month: str,
    transition_id: str,
    transition_hash: str,
    mutation_plan_hash: str,
    evidence_hash: str,
    before_registry_sha256: str,
) -> dict[str, Any]:
    """Append one irreversible monthly reservation before registry mutation."""

    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ValueError("month must be YYYY-MM") from exc
    transition_hash = _sha256_hex(transition_hash, "transition_hash")
    mutation_plan_hash = _sha256_hex(
        mutation_plan_hash,
        "mutation_plan_hash",
    )
    evidence_hash = _sha256_hex(evidence_hash, "evidence_hash")
    before_registry_sha256 = _sha256_hex(
        before_registry_sha256,
        "before_registry_sha256",
    )
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.with_name(f".{resolved.name}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            blockers = monthly_mutation_budget_blockers(resolved, month)
            if blockers:
                raise ValueError(blockers[0])
            reserved_at = (
                datetime.now(timezone.utc)
                .astimezone()
                .replace(microsecond=0)
                .isoformat()
            )
            reservation_id = _sha256(
                {
                    "month": month,
                    "transition_id": transition_id,
                    "transition_hash": transition_hash,
                    "mutation_plan_hash": mutation_plan_hash,
                    "evidence_hash": evidence_hash,
                    "before_registry_sha256": before_registry_sha256,
                }
            )
            row = {
                "schema_version": MUTATION_BUDGET_LEDGER_SCHEMA_VERSION,
                "reservation_id": reservation_id,
                "month": month,
                "transition_id": transition_id,
                "transition_hash": transition_hash,
                "mutation_plan_hash": mutation_plan_hash,
                "evidence_hash": evidence_hash,
                "before_registry_sha256": before_registry_sha256,
                "reserved_at": reserved_at,
                "status": "reserved_budget_consumed",
            }
            row["entry_hash"] = _budget_entry_hash(row)
            raw = _canonical_json(row) + b"\n"
            fd = os.open(
                resolved,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.fchmod(fd, 0o600)
                written = os.write(fd, raw)
                if written != len(raw):
                    raise OSError("mutation budget ledger short append")
                os.fsync(fd)
            finally:
                os.close(fd)
            directory_fd = os.open(resolved.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            readback = load_mutation_budget_ledger(resolved)
            if not readback or readback[-1]["entry_hash"] != row["entry_hash"]:
                raise OSError("mutation budget ledger readback mismatch")
            return row
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _slot_identity(record: FactorRecord) -> tuple[str, str]:
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
        if isinstance(dominant, list) and dominant:
            cluster = "+".join(sorted(str(item) for item in dominant if str(item)))
    return family, cluster


def _month_end_blockers(
    *,
    as_of: str,
    valid_trading_days: Sequence[str],
    budget_ledger_path: str | Path,
) -> list[str]:
    blockers: list[str] = []
    as_of_date = _date(as_of, "as_of")
    valid: list[date] = []
    for raw in valid_trading_days:
        try:
            valid.append(_date(raw, "valid_trading_day"))
        except ValueError:
            continue
    valid = sorted({item for item in valid if item.year == as_of_date.year and item.month == as_of_date.month})
    if not valid or as_of_date != valid[-1]:
        blockers.append("not_last_valid_trading_day")
    month_key = as_of_date.strftime("%Y-%m")
    blockers.extend(
        monthly_mutation_budget_blockers(budget_ledger_path, month_key)
    )
    return blockers


def apply_governed_transition(
    registry_path: str | Path,
    plan: RegistryMutationPlan,
    *,
    expected_protocol_hash: str,
    valid_trading_days: Sequence[str],
    write: bool = False,
) -> dict[str, Any]:
    """Validate and optionally apply one month-end one-for-one slot swap."""

    path = Path(registry_path).expanduser()
    snapshot = load_registry_snapshot_strict(path)
    blockers: list[str] = []
    producer_control = canonical_replay_producer_control()
    if write and not producer_control["production_apply_eligible"]:
        blockers.append(str(producer_control["blocker"]))
    if expected_protocol_hash != protocol_hash():
        blockers.append("expected_protocol_hash_mismatch")
    if plan.protocol_hash != protocol_hash():
        blockers.append("plan_protocol_hash_mismatch")
    if plan.transition.evidence_hash != plan.evidence_hash:
        blockers.append("transition_evidence_hash_mismatch")
    if plan.transition.arm_evidence.get("evidence_hash") != plan.evidence_hash:
        blockers.append("producer_evidence_hash_mismatch")
    snapshot_evidence = dict(
        plan.transition.arm_evidence.get("snapshot_evidence", {}) or {}
    )
    canonical_valid_days = [
        str(item)
        for item in snapshot_evidence.get("valid_trading_days", []) or []
    ]
    if canonical_valid_days != [str(item) for item in valid_trading_days]:
        blockers.append("valid_trading_days_not_from_evidence_artifact")
    if snapshot_evidence.get("valid_trading_days_sha256") != _sha256(
        canonical_valid_days
    ):
        blockers.append("valid_trading_days_evidence_hash_mismatch")
    if snapshot.registry_sha256 != plan.expected_registry_sha256:
        blockers.append("expected_registry_sha256_mismatch")
    blockers.extend(plan.transition.blockers)
    blockers.extend(
        _month_end_blockers(
            as_of=plan.transition.as_of,
            valid_trading_days=valid_trading_days,
            budget_ledger_path=plan.budget_ledger_path,
        )
    )
    maturity = assess_candidate_maturity(
        month_end_rankic_dates=plan.transition.evidence_window.month_end_rankic_dates,
        forward_cohorts=plan.transition.evidence_window.forward_cohorts,
    )
    if not maturity["mature"]:
        blockers.append("challenger_not_mature")
    if plan.transition.evidence_window.purge_days < MIN_PURGE_DAYS:
        blockers.append("purge_days_below_30")
    if plan.transition.evidence_window.embargo_days != REQUIRED_EMBARGO_DAYS:
        blockers.append("embargo_days_not_30")
    walk_forward = validate_purged_walk_forward(
        dict(plan.transition.arm_evidence.get("walk_forward_evidence", {}) or {})
    )
    blockers.extend(walk_forward["blockers"])
    c_arm = evaluate_c_arm(plan.transition.arm_evidence)
    blockers.extend(c_arm["blockers"])

    incumbent = snapshot.registry.factors[
        next(
            (
                index
                for index, record in enumerate(snapshot.registry.factors)
                if record.name == plan.transition.incumbent
            ),
            -1,
        )
    ] if any(record.name == plan.transition.incumbent for record in snapshot.registry.factors) else None
    challenger_existing = snapshot.registry.factors[
        next(
            (
                index
                for index, record in enumerate(snapshot.registry.factors)
                if record.name == plan.transition.challenger
            ),
            -1,
        )
    ] if any(record.name == plan.transition.challenger for record in snapshot.registry.factors) else None
    challenger_from_evidence = FactorRecord.from_dict(
        plan.challenger_record_payload
    )
    challenger = challenger_existing or challenger_from_evidence
    if incumbent is None:
        blockers.append("incumbent_record_missing")
    if incumbent is not None:
        if not math.isclose(
            float(incumbent.weight),
            float(plan.transition.before_weights[incumbent.name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append("incumbent_before_weight_mismatch")
        if not incumbent.selectable_in_quant_branch():
            blockers.append("incumbent_not_selectable_production_factor")
        if _slot_identity(incumbent) != (
            plan.transition.slot.family,
            plan.transition.slot.dominant_primitive_cluster,
        ):
            blockers.append("incumbent_slot_identity_mismatch")
    if challenger is not None:
        if challenger_existing is not None and (
            factor_record_sha256(challenger_existing)
            != factor_record_sha256(challenger_from_evidence)
        ):
            blockers.append("challenger_record_evidence_mismatch")
        if not math.isclose(
            float(challenger.weight),
            float(plan.transition.before_weights[challenger.name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append("challenger_before_weight_mismatch")
        if challenger.state not in {
            FactorLifecycleState.SHADOW,
            FactorLifecycleState.MATURE_CANDIDATE,
            FactorLifecycleState.PRODUCTION_CANDIDATE,
        }:
            blockers.append("challenger_lifecycle_not_eligible")
        if not challenger.all_gates_passed():
            blockers.append("challenger_8gate_not_passed")
        if _slot_identity(challenger) != (
            plan.transition.slot.family,
            plan.transition.slot.dominant_primitive_cluster,
        ):
            blockers.append("challenger_slot_identity_mismatch")

    weights = {record.name: float(record.weight) for record in snapshot.registry.factors}
    weights.setdefault(challenger.name, float(challenger.weight))
    families = {record.name: _slot_identity(record)[0] for record in snapshot.registry.factors}
    families.setdefault(challenger.name, _slot_identity(challenger)[0])
    budget = build_slot_risk_budget(
        previous_weights=weights,
        family_by_factor=families,
        incumbent=plan.transition.incumbent,
        challenger=plan.transition.challenger,
        evidence_sufficient=not blockers,
    )
    if not blockers:
        blockers.extend(budget["blockers"])
        if not math.isclose(
            float(plan.transition.after_weights[plan.transition.incumbent]),
            float(budget["weights"][plan.transition.incumbent]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append("incumbent_after_weight_mismatch")
        if not math.isclose(
            float(plan.transition.after_weights[plan.transition.challenger]),
            float(budget["weights"][plan.transition.challenger]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append("challenger_after_weight_mismatch")
    if blockers:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "transition_id": plan.transition.transition_id,
            "transition_hash": plan.transition.transition_hash,
            "mutation_plan_hash": plan.mutation_plan_hash,
            "evidence_hash": plan.evidence_hash,
            "status": "blocked",
            "apply_requested": bool(write),
            "blockers": list(dict.fromkeys(blockers)),
            "before_registry_sha256": snapshot.registry_sha256,
            "after_registry_sha256": snapshot.registry_sha256,
            "inverse_wal_path": plan.wal_path,
            "mutation_budget_ledger_path": plan.budget_ledger_path,
            "changed_record_names": [],
            "maturity": maturity,
            "c_arm": c_arm,
            "walk_forward": walk_forward,
            "risk_budget": budget,
            "canonical_replay_producer_control": producer_control,
        }

    assert incumbent is not None and challenger is not None
    updated_incumbent = FactorRecord.from_dict(incumbent.to_dict())
    updated_challenger = FactorRecord.from_dict(challenger.to_dict())
    target_weight = float(budget["weights"][challenger.name])
    updated_incumbent.state = FactorLifecycleState.DEPRECATED
    updated_incumbent.weight = 0.0
    updated_incumbent.deprecated_reason = (
        f"protocol_v2_targeted_swap:{plan.transition.transition_id}"
    )
    updated_challenger.state = FactorLifecycleState.PRODUCTION_FACTOR
    updated_challenger.weight = target_weight
    updated_challenger.deprecated_reason = ""
    updated_challenger.admission_decision = FactorAdmissionDecision.PRODUCTION_CANDIDATE
    for record, role in (
        (updated_incumbent, "incumbent_deprecated"),
        (updated_challenger, "challenger_promoted"),
    ):
        record.metadata = {
            **dict(record.metadata or {}),
            "factor_governance_protocol": {
                "version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(),
                "transition_id": plan.transition.transition_id,
                "transition_hash": plan.transition.transition_hash,
                "mutation_plan_hash": plan.mutation_plan_hash,
                "evidence_hash": plan.evidence_hash,
                "evaluation_hash": (
                    plan.transition.evidence_window.evaluation_hash
                ),
                "role": role,
            },
        }

    month_key = plan.transition.as_of[:7]
    next_production_names = sorted(
        {
            record.name
            for record in snapshot.registry.selectable_factors()
            if record.name != incumbent.name
        }
        | {challenger.name}
    )
    next_production_count = len(next_production_names)
    metadata_updates = {
        **dict(plan.metadata_updates),
        "factor_governance_protocol_version": PROTOCOL_VERSION,
        "factor_governance_protocol_hash": protocol_hash(),
        "factor_governance_last_transition_month": month_key,
        "factor_governance_last_transition_id": plan.transition.transition_id,
        "factor_governance_last_transition_hash": plan.transition.transition_hash,
        "factor_governance_last_evidence_hash": plan.evidence_hash,
        "factor_governance_last_evaluation_hash": (
            plan.transition.evidence_window.evaluation_hash
        ),
        "production_factor_count": next_production_count,
        "production_factor_names": next_production_names,
        "production_factor_set_sha256": production_factor_set_sha256(
            next_production_names
        ),
    }
    budget_reservation: dict[str, Any] | None = None
    if write:
        try:
            budget_reservation = reserve_monthly_mutation_budget(
                plan.budget_ledger_path,
                month=month_key,
                transition_id=plan.transition.transition_id,
                transition_hash=plan.transition.transition_hash,
                mutation_plan_hash=plan.mutation_plan_hash,
                evidence_hash=plan.evidence_hash,
                before_registry_sha256=snapshot.registry_sha256,
            )
        except (OSError, ValueError) as exc:
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(),
                "transition_id": plan.transition.transition_id,
                "transition_hash": plan.transition.transition_hash,
                "mutation_plan_hash": plan.mutation_plan_hash,
                "evidence_hash": plan.evidence_hash,
                "status": "blocked",
                "apply_requested": True,
                "blockers": [f"mutation_budget_reservation_failed:{exc}"],
                "before_registry_sha256": snapshot.registry_sha256,
                "after_registry_sha256": snapshot.registry_sha256,
                "inverse_wal_path": plan.wal_path,
                "mutation_budget_ledger_path": plan.budget_ledger_path,
                "changed_record_names": [],
                "maturity": maturity,
                "c_arm": c_arm,
                "walk_forward": walk_forward,
                "risk_budget": budget,
                "canonical_replay_producer_control": producer_control,
            }
    mutation = apply_factor_record_patch(
        path,
        {
            incumbent.name: updated_incumbent,
            challenger.name: updated_challenger,
        },
        expected_registry_sha256=snapshot.registry_sha256,
        expected_record_sha256s={
            incumbent.name: snapshot.record_sha256s[incumbent.name],
            challenger.name: snapshot.record_sha256s.get(challenger.name),
        },
        metadata_updates=metadata_updates,
        expected_metadata_values={
            key: snapshot.metadata_payload.get(key, METADATA_ABSENT)
            for key in metadata_updates
        },
        mutation_id=f"factor-governance-v2:{plan.transition.transition_id}",
        reason="FactorGovernanceProtocol v2 targeted month-end slot swap",
        manifest_metadata={
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "transition_hash": plan.transition.transition_hash,
            "mutation_plan_hash": plan.mutation_plan_hash,
            "evidence_hash": plan.evidence_hash,
            "evaluation_hash": plan.transition.evidence_window.evaluation_hash,
            "target_record_names": list(plan.target_record_names),
            "mutation_budget_ledger_path": plan.budget_ledger_path,
            "mutation_budget_reservation": budget_reservation,
        },
        journal_path=plan.wal_path if write else None,
        write=write,
    )
    readback_blockers: list[str] = []
    if write:
        readback = load_registry_snapshot_strict(path)
        if readback.registry_sha256 != mutation.get("after_registry_sha256"):
            readback_blockers.append("registry_after_sha_readback_mismatch")
        for name in (incumbent.name, challenger.name):
            record_payload = readback.record_payloads.get(name, {})
            protocol_metadata = dict(
                dict(record_payload.get("metadata", {}) or {}).get(
                    "factor_governance_protocol", {}
                )
                or {}
            )
            expected_fields = {
                "protocol_hash": protocol_hash(),
                "transition_hash": plan.transition.transition_hash,
                "mutation_plan_hash": plan.mutation_plan_hash,
                "evidence_hash": plan.evidence_hash,
            }
            for key, expected in expected_fields.items():
                if protocol_metadata.get(key) != expected:
                    readback_blockers.append(
                        f"record_protocol_readback_mismatch:{name}:{key}"
                    )
        metadata_expected = {
            "factor_governance_protocol_hash": protocol_hash(),
            "factor_governance_last_transition_hash": (
                plan.transition.transition_hash
            ),
            "factor_governance_last_evidence_hash": plan.evidence_hash,
        }
        for key, expected in metadata_expected.items():
            if readback.metadata_payload.get(key) != expected:
                readback_blockers.append(
                    f"registry_metadata_readback_mismatch:{key}"
                )
        try:
            budget_rows = load_mutation_budget_ledger(plan.budget_ledger_path)
        except (OSError, ValueError) as exc:
            readback_blockers.append(
                f"mutation_budget_ledger_readback_invalid:{exc}"
            )
        else:
            if not budget_reservation or not any(
                row.get("entry_hash") == budget_reservation.get("entry_hash")
                for row in budget_rows
            ):
                readback_blockers.append(
                    "mutation_budget_reservation_readback_missing"
                )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "transition_id": plan.transition.transition_id,
        "transition_hash": plan.transition.transition_hash,
        "mutation_plan_hash": plan.mutation_plan_hash,
        "evidence_hash": plan.evidence_hash,
        "status": (
            "blocked_readback"
            if readback_blockers
            else "applied"
            if mutation.get("applied")
            else "report_only_ready"
        ),
        "apply_requested": bool(write),
        "blockers": readback_blockers,
        "before_registry_sha256": mutation["before_registry_sha256"],
        "after_registry_sha256": mutation["after_registry_sha256"],
        "inverse_wal_path": plan.wal_path,
        "mutation_budget_ledger_path": plan.budget_ledger_path,
        "mutation_budget_reservation": budget_reservation,
        "changed_record_names": sorted([incumbent.name, challenger.name]),
        "maturity": maturity,
        "c_arm": c_arm,
        "walk_forward": walk_forward,
        "risk_budget": budget,
        "registry_mutation_manifest": mutation,
        "canonical_replay_producer_control": producer_control,
    }


__all__ = [
    "CANONICAL_FULL_CHAIN_PRODUCER_AVAILABLE",
    "CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER",
    "PROTOCOL_VERSION",
    "PROTOCOL_HASH",
    "PROTOCOL_SCHEMA_VERSION",
    "FactorEvidenceWindow",
    "FactorSlot",
    "FactorTransitionPlan",
    "RegistryMutationPlan",
    "advance_failure_streak",
    "apply_governed_transition",
    "assess_candidate_maturity",
    "benjamini_hochberg_by_family",
    "block_bootstrap_paired_delta_ci",
    "build_slot_risk_budget",
    "canonical_replay_producer_control",
    "control_chain_evidence_hash",
    "evaluate_c_arm",
    "governance_runtime_status",
    "load_mutation_budget_ledger",
    "monthly_mutation_budget_blockers",
    "protocol_hash",
    "protocol_policy",
    "reserve_monthly_mutation_budget",
    "validate_purged_walk_forward",
]
