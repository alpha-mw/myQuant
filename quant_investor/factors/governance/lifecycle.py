"""Governed Factor lifecycle policy and monthly decision artifacts.

These artifacts are non-authorizing. They record sealed parameters and a pure
state transition over matured monitor evidence. Applying a weight proposal
still requires the existing Factor production writers and explicit
authorization. Probation scale never enters production under v1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, seal_artifact

from .common import (
    artifact_ref,
    business_identity,
    canonical_identifier,
    canonical_timestamp,
    decimal_text,
    decimal_value,
    exact_payload,
    validate_artifact_ref,
)
from .errors import FactorGovernanceError

LIFECYCLE_POLICY_KIND: Final = "factor.lifecycle_policy"
LIFECYCLE_DECISION_KIND: Final = "factor.lifecycle_decision"
LIFECYCLE_POLICY_ID: Final = "factor-lifecycle-policy-v1"
LIFECYCLE_AUTHORITY: Final = "NON_AUTHORIZING"

PREREGISTERED: Final = "PREREGISTERED"
PROBATION: Final = "PROBATION"
ACTIVE: Final = "ACTIVE"
WATCH: Final = "WATCH"
RETIRED: Final = "RETIRED"
INSUFFICIENT_EVIDENCE: Final = "INSUFFICIENT_EVIDENCE"
SOURCE_BLOCKED: Final = "SOURCE_BLOCKED"

_POLICY_STATES: Final = frozenset(
    {PREREGISTERED, PROBATION, ACTIVE, WATCH, RETIRED, INSUFFICIENT_EVIDENCE, SOURCE_BLOCKED}
)
_DECISION_STATES: Final = frozenset(
    {PREREGISTERED, PROBATION, ACTIVE, WATCH, RETIRED, INSUFFICIENT_EVIDENCE, SOURCE_BLOCKED}
)

_MAX_POLICY_BYTES: Final = 64 * 1024
_MAX_DECISION_BYTES: Final = 512 * 1024

_POLICY_FIELDS: Final = {
    "lifecycle_policy_id",
    "policy_version",
    "decision_half_life_months",
    "probation_months",
    "probation_posterior",
    "watch_posterior",
    "retire_posterior",
    "retire_consecutive_months",
    "reentry_cooldown_months",
    "probation_weight_scale",
    "active_weight_scale",
    "watch_weight_scale",
    "retired_weight_scale",
    "max_factor_weight",
    "max_absolute_weight_step",
    "weight_rebalance_months",
    "min_origin_sessions",
    "min_disjoint_cohorts",
    "cusum_slack",
    "cusum_threshold",
    "prior_mean",
    "prior_sd",
    "observation_sd_floor",
    "publication_haircut",
    "probation_enters_production",
    "bootstrap_weight_change_requires_authorization",
    "cost_evidence_required_to_apply",
    "capacity_evidence_required_to_apply",
    "trial_grid_size",
    "authority",
}

_DECISION_FIELDS: Final = {
    "lifecycle_decision_id",
    "lifecycle_policy_ref",
    "decision_month",
    "previous_decision_ref",
    "factor_rows",
    "weight_proposal",
    "source_gap_counts",
    "outcome_head_refs",
    "blockers",
    "authority",
}

_FACTOR_ROW_FIELDS: Final = {
    "factor_id",
    "lane",
    "previous_state",
    "state",
    "posterior_mean",
    "posterior_p_positive",
    "cusum_alarm",
    "below_retire_months",
    "weight_scale",
    "production_eligible",
    "reasons",
}

_WEIGHT_PROPOSAL_FIELDS: Final = {
    "eligible_factor_ids",
    "target_weights",
    "capped",
    "uninvested_weight",
    "absolute_step_from_previous",
    "application_authorized",
}

# Sealed v1 parameters chosen from the research grid in
# docs/plans/factor_lifecycle_policy_v1.md (6-month decision half-life, 0.35 cap,
# conservative probation/watch, DSR-surviving after 64 trials).
_V1_POLICY_BODY: Final = {
    "lifecycle_policy_id": LIFECYCLE_POLICY_ID,
    "policy_version": "v1",
    "decision_half_life_months": "6.000000000000",
    "probation_months": 12,
    "probation_posterior": "0.800000000000",
    "watch_posterior": "0.700000000000",
    "retire_posterior": "0.500000000000",
    "retire_consecutive_months": 3,
    "reentry_cooldown_months": 12,
    "probation_weight_scale": "0.500000000000",
    "active_weight_scale": "1.000000000000",
    "watch_weight_scale": "0.500000000000",
    "retired_weight_scale": "0.000000000000",
    "max_factor_weight": "0.350000000000",
    "max_absolute_weight_step": "0.500000000000",
    "weight_rebalance_months": 3,
    "min_origin_sessions": 60,
    "min_disjoint_cohorts": 8,
    "cusum_slack": "0.500000000000",
    "cusum_threshold": "4.000000000000",
    "prior_mean": "0.000000000000",
    "prior_sd": "0.050000000000",
    "observation_sd_floor": "0.010000000000",
    "publication_haircut": "0.500000000000",
    "probation_enters_production": False,
    "bootstrap_weight_change_requires_authorization": True,
    "cost_evidence_required_to_apply": True,
    "capacity_evidence_required_to_apply": True,
    "trial_grid_size": 64,
    "authority": LIFECYCLE_AUTHORITY,
}


def _check_size(envelope: Mapping[str, Any], *, limit: int, label: str) -> None:
    if len(canonical_json_bytes(dict(envelope))) > limit:
        raise FactorGovernanceError(f"{label} exceeds its canonical byte limit")


def _decimal_field(value: Any, *, label: str) -> str:
    return decimal_text(decimal_value(value, label=label), label=label)


def _blockers(values: Any) -> list[str]:
    if type(values) is not list or len(values) != len(set(values)):
        raise FactorGovernanceError("lifecycle blockers are not a unique list")
    rows: list[str] = []
    for value in values:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or not value.replace("_", "").isalnum()
            or value != value.upper()
        ):
            raise FactorGovernanceError("lifecycle blocker code is invalid")
        rows.append(value)
    if rows != sorted(rows):
        raise FactorGovernanceError("lifecycle blockers are not canonical")
    return rows


def _reasons(values: Any) -> list[str]:
    return _blockers(values)


def weight_scale_for_state(state: str, *, policy: Mapping[str, Any]) -> str:
    """Return the sealed scale for one lifecycle state."""

    if state == PROBATION:
        return str(policy["probation_weight_scale"])
    if state == ACTIVE:
        return str(policy["active_weight_scale"])
    if state == WATCH:
        return str(policy["watch_weight_scale"])
    if state in {RETIRED, PREREGISTERED, INSUFFICIENT_EVIDENCE, SOURCE_BLOCKED}:
        return str(policy["retired_weight_scale"])
    raise FactorGovernanceError(f"unknown lifecycle state {state}")


def production_eligible(state: str, *, policy: Mapping[str, Any]) -> bool:
    """Whether the state may enter a production weight proposal under v1."""

    if state == ACTIVE:
        return True
    if state == WATCH:
        return True
    if state == PROBATION:
        return bool(policy["probation_enters_production"]) is True
    return False


def _validate_transition_inputs(
    previous_state: str,
    *,
    available_origin_sessions: int,
    cohort_count: int,
    posterior_p_positive: float,
    cusum_alarm: bool,
    below_retire_months: int,
) -> None:
    if previous_state not in _POLICY_STATES:
        raise FactorGovernanceError("previous lifecycle state is invalid")
    if type(available_origin_sessions) is not int or available_origin_sessions < 0:
        raise FactorGovernanceError("available_origin_sessions is invalid")
    if type(cohort_count) is not int or cohort_count < 0:
        raise FactorGovernanceError("cohort_count is invalid")
    if type(below_retire_months) is not int or below_retire_months < 0:
        raise FactorGovernanceError("below_retire_months is invalid")
    if type(cusum_alarm) is not bool:
        raise FactorGovernanceError("cusum_alarm must be boolean")
    if not (0.0 <= float(posterior_p_positive) <= 1.0):
        raise FactorGovernanceError("posterior_p_positive is out of range")


def _advance_from_preregistered(
    *,
    cohort_count: int,
    probation_months: int,
    posterior_p_positive: float,
    probation_p: float,
    residual_ic_positive: bool | None,
    next_below: int,
    reasons: list[str],
) -> tuple[str, int, list[str]]:
    if cohort_count >= probation_months and posterior_p_positive >= probation_p:
        if residual_ic_positive is False:
            return PREREGISTERED, next_below, ["RESIDUAL_IC_NOT_POSITIVE"]
        return PROBATION, next_below, ["ENTERED_PROBATION"]
    return PREREGISTERED, next_below, reasons or ["AWAITING_PROBATION"]


def _advance_governed_state(
    previous_state: str,
    *,
    next_below: int,
    retire_n: int,
    posterior_p_positive: float,
    probation_p: float,
    watch_p: float,
    cusum_alarm: bool,
    admitted: bool,
    reasons: list[str],
) -> tuple[str, int, list[str]]:
    if previous_state == PROBATION:
        if next_below >= retire_n:
            return RETIRED, next_below, reasons + ["RETIRED_FROM_PROBATION"]
        if admitted and posterior_p_positive >= probation_p:
            return ACTIVE, next_below, ["ADMISSION_PASSED"]
        return PROBATION, next_below, reasons or ["REMAINS_PROBATION"]
    if previous_state == ACTIVE:
        if next_below >= retire_n:
            return RETIRED, next_below, reasons + ["RETIRED_FROM_ACTIVE"]
        if cusum_alarm or posterior_p_positive < watch_p:
            return WATCH, next_below, reasons or ["ENTERED_WATCH"]
        return ACTIVE, next_below, ["REMAINS_ACTIVE"]
    if previous_state == WATCH:
        if next_below >= retire_n:
            return RETIRED, next_below, reasons + ["RETIRED_FROM_WATCH"]
        if posterior_p_positive >= probation_p and not cusum_alarm:
            return ACTIVE, next_below, ["RETURNED_TO_ACTIVE"]
        return WATCH, next_below, reasons or ["REMAINS_WATCH"]
    raise FactorGovernanceError(f"unhandled lifecycle state {previous_state}")


def transition_lifecycle_state(
    previous_state: str,
    *,
    policy: Mapping[str, Any],
    available_origin_sessions: int,
    cohort_count: int,
    posterior_p_positive: float,
    cusum_alarm: bool,
    below_retire_months: int,
    residual_ic_positive: bool | None = None,
    admitted: bool = False,
) -> tuple[str, int, list[str]]:
    """Pure next-state function of previous decision, policy and new evidence."""

    _validate_transition_inputs(
        previous_state,
        available_origin_sessions=available_origin_sessions,
        cohort_count=cohort_count,
        posterior_p_positive=posterior_p_positive,
        cusum_alarm=cusum_alarm,
        below_retire_months=below_retire_months,
    )
    reasons: list[str] = []
    min_origins = int(policy["min_origin_sessions"])
    min_cohorts = int(policy["min_disjoint_cohorts"])
    if available_origin_sessions == 0 and cohort_count == 0:
        return SOURCE_BLOCKED, 0, ["NO_AVAILABLE_RANK_IC"]
    if available_origin_sessions < min_origins or cohort_count < min_cohorts:
        return INSUFFICIENT_EVIDENCE, 0, ["EVIDENCE_BELOW_MINIMUM"]

    retire_p = float(Decimal(str(policy["retire_posterior"])))
    watch_p = float(Decimal(str(policy["watch_posterior"])))
    probation_p = float(Decimal(str(policy["probation_posterior"])))
    retire_n = int(policy["retire_consecutive_months"])
    probation_months = int(policy["probation_months"])
    next_below = below_retire_months + 1 if posterior_p_positive < retire_p else 0
    if cusum_alarm:
        reasons.append("CUSUM_BELOW_REFERENCE")
    if posterior_p_positive < retire_p:
        reasons.append("POSTERIOR_BELOW_RETIRE")
    elif posterior_p_positive < watch_p:
        reasons.append("POSTERIOR_BELOW_WATCH")

    if previous_state == RETIRED:
        return RETIRED, next_below, reasons or ["REMAINS_RETIRED"]
    if previous_state in {PREREGISTERED, INSUFFICIENT_EVIDENCE, SOURCE_BLOCKED}:
        return _advance_from_preregistered(
            cohort_count=cohort_count,
            probation_months=probation_months,
            posterior_p_positive=posterior_p_positive,
            probation_p=probation_p,
            residual_ic_positive=residual_ic_positive,
            next_below=next_below,
            reasons=reasons,
        )
    return _advance_governed_state(
        previous_state,
        next_below=next_below,
        retire_n=retire_n,
        posterior_p_positive=posterior_p_positive,
        probation_p=probation_p,
        watch_p=watch_p,
        cusum_alarm=cusum_alarm,
        admitted=admitted,
        reasons=reasons,
    )


def _empty_weight_proposal() -> dict[str, Any]:
    return {
        "eligible_factor_ids": [],
        "target_weights": {},
        "capped": False,
        "uninvested_weight": "1.000000000000",
        "absolute_step_from_previous": "0.000000000000",
        "application_authorized": False,
    }


def _cap_weights(
    weights: dict[str, Decimal], *, max_weight: Decimal
) -> tuple[dict[str, Decimal], bool]:
    capped = False
    result = dict(weights)
    for _ in range(len(result)):
        over = {
            factor_id: weight - max_weight
            for factor_id, weight in result.items()
            if weight > max_weight
        }
        if not over:
            break
        capped = True
        excess = sum(over.values(), Decimal("0"))
        for factor_id in over:
            result[factor_id] = max_weight
        room = {
            factor_id: weight
            for factor_id, weight in result.items()
            if Decimal("0") < weight < max_weight
        }
        if not room:
            break
        room_total = sum(room.values(), Decimal("0"))
        for factor_id in room:
            result[factor_id] += excess * room[factor_id] / room_total
    return result, capped


def _step_limited_weights(
    weights: dict[str, Decimal],
    previous: Mapping[str, Decimal],
    *,
    max_step: Decimal,
) -> tuple[dict[str, Decimal], Decimal, bool]:
    all_ids = sorted(set(previous) | set(weights), key=lambda value: value.encode("utf-8"))
    step = Decimal("0")
    for factor_id in all_ids:
        step += abs(weights.get(factor_id, Decimal("0")) - previous.get(factor_id, Decimal("0")))
    if step <= max_step:
        return weights, step, False
    scale = max_step / step
    adjusted = {
        factor_id: value
        for factor_id in all_ids
        for value in [
            previous.get(factor_id, Decimal("0"))
            + scale * (weights.get(factor_id, Decimal("0")) - previous.get(factor_id, Decimal("0")))
        ]
        if value > 0
    }
    return adjusted, max_step, True


def propose_lifecycle_weights(
    factor_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    previous_weights: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a non-authorizing capped weight proposal from decision rows."""

    max_weight = decimal_value(policy["max_factor_weight"], label="max_factor_weight")
    max_step = decimal_value(policy["max_absolute_weight_step"], label="max_absolute_weight_step")
    scores: dict[str, Decimal] = {}
    for row in factor_rows:
        factor_id = canonical_identifier(row["factor_id"], label="factor_id")
        if not production_eligible(str(row["state"]), policy=policy):
            continue
        posterior = decimal_value(row["posterior_mean"], label=f"posterior_mean[{factor_id}]")
        scale = decimal_value(row["weight_scale"], label=f"weight_scale[{factor_id}]")
        score = max(posterior, Decimal("0")) * scale
        if score > 0:
            scores[factor_id] = score
    eligible = sorted(scores, key=lambda value: value.encode("utf-8"))
    if not eligible:
        return _empty_weight_proposal()
    total = sum(scores.values(), Decimal("0"))
    weights, capped = _cap_weights(
        {factor_id: scores[factor_id] / total for factor_id in eligible},
        max_weight=max_weight,
    )
    invested = sum(weights.values(), Decimal("0"))
    if invested > 1:
        raise FactorGovernanceError("lifecycle weights exceed one")
    previous = {
        str(factor_id): decimal_value(value, label=f"previous_weight[{factor_id}]")
        for factor_id, value in (previous_weights or {}).items()
    }
    weights, step, stepped = _step_limited_weights(weights, previous, max_step=max_step)
    capped = capped or stepped
    invested = sum(weights.values(), Decimal("0"))
    return {
        "eligible_factor_ids": sorted(weights, key=lambda value: value.encode("utf-8")),
        "target_weights": {
            factor_id: decimal_text(weight, label=f"target_weight[{factor_id}]")
            for factor_id, weight in sorted(weights.items(), key=lambda item: item[0].encode())
        },
        "capped": capped,
        "uninvested_weight": decimal_text(Decimal("1") - invested, label="uninvested_weight"),
        "absolute_step_from_previous": decimal_text(step, label="absolute_step"),
        "application_authorized": False,
    }


def build_lifecycle_policy_v1(*, trusted_at: str) -> dict[str, Any]:
    """Seal the reviewed v1 lifecycle policy parameters."""

    envelope = seal_artifact(
        LIFECYCLE_POLICY_KIND,
        dict(_V1_POLICY_BODY),
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )
    _check_size(envelope, limit=_MAX_POLICY_BYTES, label="lifecycle policy")
    return validate_lifecycle_policy(envelope)


def validate_lifecycle_policy(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate and replay the sealed lifecycle policy."""

    envelope, payload = exact_payload(document, kind=LIFECYCLE_POLICY_KIND, fields=_POLICY_FIELDS)
    expected = dict(_V1_POLICY_BODY)
    if payload["lifecycle_policy_id"] != LIFECYCLE_POLICY_ID:
        raise FactorGovernanceError("lifecycle policy id differs")
    if payload != expected:
        raise FactorGovernanceError("lifecycle policy does not replay exactly")
    if payload["authority"] != LIFECYCLE_AUTHORITY:
        raise FactorGovernanceError("lifecycle policy authority differs")
    _check_size(envelope, limit=_MAX_POLICY_BYTES, label="lifecycle policy")
    return envelope


def _normalize_factor_row(row: Mapping[str, Any], *, policy: Mapping[str, Any]) -> dict[str, Any]:
    if set(row) != _FACTOR_ROW_FIELDS:
        raise FactorGovernanceError("lifecycle factor row fields are not exact")
    factor_id = canonical_identifier(row["factor_id"], label="factor_id")
    state = str(row["state"])
    previous_state = str(row["previous_state"])
    if state not in _DECISION_STATES or previous_state not in _DECISION_STATES:
        raise FactorGovernanceError("lifecycle factor state is invalid")
    lane = canonical_identifier(row["lane"], label="lane")
    if type(row["cusum_alarm"]) is not bool:
        raise FactorGovernanceError("cusum_alarm must be boolean")
    if type(row["below_retire_months"]) is not int or row["below_retire_months"] < 0:
        raise FactorGovernanceError("below_retire_months is invalid")
    if type(row["production_eligible"]) is not bool:
        raise FactorGovernanceError("production_eligible must be boolean")
    expected_scale = weight_scale_for_state(state, policy=policy)
    if str(row["weight_scale"]) != expected_scale:
        raise FactorGovernanceError("weight_scale does not match state")
    if row["production_eligible"] != production_eligible(state, policy=policy):
        raise FactorGovernanceError("production_eligible does not match policy")
    return {
        "factor_id": factor_id,
        "lane": lane,
        "previous_state": previous_state,
        "state": state,
        "posterior_mean": _decimal_field(row["posterior_mean"], label="posterior_mean"),
        "posterior_p_positive": _decimal_field(
            row["posterior_p_positive"], label="posterior_p_positive"
        ),
        "cusum_alarm": row["cusum_alarm"],
        "below_retire_months": row["below_retire_months"],
        "weight_scale": expected_scale,
        "production_eligible": row["production_eligible"],
        "reasons": _reasons(row["reasons"]),
    }


def build_lifecycle_decision(
    *,
    lifecycle_policy: Mapping[str, Any] | bytes,
    decision_month: str,
    factor_rows: Sequence[Mapping[str, Any]],
    outcome_head_refs: Mapping[str, Mapping[str, Any]],
    source_gap_counts: Mapping[str, int],
    previous_decision: Mapping[str, Any] | bytes | None = None,
    previous_weights: Mapping[str, str] | None = None,
    blockers: Sequence[str] = (),
    trusted_at: str,
) -> dict[str, Any]:
    """Seal one monthly lifecycle decision from exact matured evidence refs."""

    policy = validate_lifecycle_policy(lifecycle_policy)
    policy_payload = policy["payload"]
    if type(decision_month) is not str or len(decision_month) != 7 or decision_month[4] != "-":
        raise FactorGovernanceError("decision_month must be YYYY-MM")
    previous_ref = None
    if previous_decision is not None:
        previous = validate_lifecycle_decision(previous_decision)
        previous_ref = artifact_ref(previous)
        if previous["payload"]["decision_month"] >= decision_month:
            raise FactorGovernanceError("lifecycle decision month is not increasing")
    rows = [
        _normalize_factor_row(row, policy=policy_payload)
        for row in sorted(factor_rows, key=lambda item: str(item["factor_id"]).encode())
    ]
    if [row["factor_id"] for row in rows] != sorted(
        (row["factor_id"] for row in rows), key=lambda value: value.encode("utf-8")
    ):
        raise FactorGovernanceError("lifecycle factor rows are not in UTF-8 order")
    if len({row["factor_id"] for row in rows}) != len(rows):
        raise FactorGovernanceError("lifecycle factor rows are duplicated")
    gaps = {
        canonical_identifier(factor_id, label="source_gap_counts"): int(count)
        for factor_id, count in source_gap_counts.items()
    }
    if any(count < 0 for count in gaps.values()):
        raise FactorGovernanceError("source_gap_counts must be non-negative")
    heads = {
        canonical_identifier(factor_id, label="outcome_head_refs"): validate_artifact_ref(
            dict(reference),
            label=f"outcome_head_refs[{factor_id}]",
        )
        for factor_id, reference in outcome_head_refs.items()
    }
    weight_proposal = propose_lifecycle_weights(
        rows, policy=policy_payload, previous_weights=previous_weights
    )
    if set(weight_proposal) != _WEIGHT_PROPOSAL_FIELDS:
        raise FactorGovernanceError("weight proposal fields are not exact")
    identity = {
        "lifecycle_policy_ref": artifact_ref(policy),
        "decision_month": decision_month,
        "previous_decision_ref": previous_ref,
        "factor_rows": rows,
        "weight_proposal": weight_proposal,
        "source_gap_counts": {
            factor_id: gaps[factor_id]
            for factor_id in sorted(gaps, key=lambda value: value.encode("utf-8"))
        },
        "outcome_head_refs": {
            factor_id: heads[factor_id]
            for factor_id in sorted(heads, key=lambda value: value.encode("utf-8"))
        },
        "blockers": _blockers(list(blockers)),
        "authority": LIFECYCLE_AUTHORITY,
    }
    payload = {
        "lifecycle_decision_id": business_identity("factor-lifecycle-decision", identity),
        **identity,
    }
    envelope = seal_artifact(
        LIFECYCLE_DECISION_KIND,
        payload,
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )
    _check_size(envelope, limit=_MAX_DECISION_BYTES, label="lifecycle decision")
    return validate_lifecycle_decision(envelope)


def _validate_decision_payload_shape(payload: Mapping[str, Any]) -> dict[str, str]:
    if payload["authority"] != LIFECYCLE_AUTHORITY:
        raise FactorGovernanceError("lifecycle decision authority differs")
    policy_ref = validate_artifact_ref(
        payload["lifecycle_policy_ref"],
        label="lifecycle_policy_ref",
        expected_kind=LIFECYCLE_POLICY_KIND,
    )
    if type(payload["decision_month"]) is not str or len(payload["decision_month"]) != 7:
        raise FactorGovernanceError("decision_month is invalid")
    previous = payload["previous_decision_ref"]
    if previous is not None:
        validate_artifact_ref(
            previous,
            label="previous_decision_ref",
            expected_kind=LIFECYCLE_DECISION_KIND,
        )
    if type(payload["factor_rows"]) is not list:
        raise FactorGovernanceError("factor_rows must be a list")
    if type(payload["weight_proposal"]) is not dict or set(payload["weight_proposal"]) != (
        _WEIGHT_PROPOSAL_FIELDS
    ):
        raise FactorGovernanceError("weight_proposal fields are not exact")
    if payload["weight_proposal"]["application_authorized"] is not False:
        raise FactorGovernanceError("lifecycle weight proposal cannot self-authorize")
    _blockers(payload["blockers"])
    if type(payload["source_gap_counts"]) is not dict:
        raise FactorGovernanceError("lifecycle decision refs are invalid")
    if type(payload["outcome_head_refs"]) is not dict:
        raise FactorGovernanceError("lifecycle decision refs are invalid")
    return policy_ref


def validate_lifecycle_decision(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate and replay one sealed lifecycle decision."""

    envelope, payload = exact_payload(
        document, kind=LIFECYCLE_DECISION_KIND, fields=_DECISION_FIELDS
    )
    policy_ref = _validate_decision_payload_shape(payload)
    expected_id = business_identity(
        "factor-lifecycle-decision",
        {field: payload[field] for field in _DECISION_FIELDS if field != "lifecycle_decision_id"},
    )
    if payload["lifecycle_decision_id"] != expected_id:
        raise FactorGovernanceError("lifecycle decision identity differs")
    if policy_ref != payload["lifecycle_policy_ref"]:
        raise FactorGovernanceError("lifecycle policy ref differs")
    _check_size(envelope, limit=_MAX_DECISION_BYTES, label="lifecycle decision")
    return envelope


__all__ = [
    "ACTIVE",
    "INSUFFICIENT_EVIDENCE",
    "LIFECYCLE_AUTHORITY",
    "LIFECYCLE_DECISION_KIND",
    "LIFECYCLE_POLICY_ID",
    "LIFECYCLE_POLICY_KIND",
    "PREREGISTERED",
    "PROBATION",
    "RETIRED",
    "SOURCE_BLOCKED",
    "WATCH",
    "build_lifecycle_decision",
    "build_lifecycle_policy_v1",
    "production_eligible",
    "propose_lifecycle_weights",
    "transition_lifecycle_state",
    "validate_lifecycle_decision",
    "validate_lifecycle_policy",
    "weight_scale_for_state",
]
