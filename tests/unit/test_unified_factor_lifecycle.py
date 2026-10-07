"""Governed Factor lifecycle policy and decision contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.contracts import get_contract
from quant_investor.factors.governance import (
    ACTIVE,
    INSUFFICIENT_EVIDENCE,
    LIFECYCLE_AUTHORITY,
    LIFECYCLE_DECISION_KIND,
    LIFECYCLE_POLICY_ID,
    LIFECYCLE_POLICY_KIND,
    PREREGISTERED,
    PROBATION,
    RETIRED,
    WATCH,
    build_lifecycle_decision,
    build_lifecycle_policy_v1,
    production_eligible,
    propose_lifecycle_weights,
    transition_lifecycle_state,
    validate_lifecycle_decision,
    validate_lifecycle_policy,
    weight_scale_for_state,
)
from quant_investor.factors.governance.bootstrap import BLEND_W80, LOW_DOLLAR_VOLUME
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.factors.governance.implementations import (
    BOOTSTRAP_FACTOR_IDS,
    PROSPECTIVE_FACTOR_IDS,
    compute_installed_signals,
    installed_semantic_row,
)
from quant_investor.factors.lifecycle_candidates import (
    PRICE_VOLUME_CANDIDATES,
    VOLATILITY_PENALTY_5D,
)


def _ref(kind: str, artifact_id: str) -> dict[str, str]:
    return {
        "kind": kind,
        "contract_sha256": get_contract(kind).contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": "1" * 64,
        "byte_sha256": "2" * 64,
    }


def test_policy_v1_seals_and_replays() -> None:
    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")
    assert policy["kind"] == LIFECYCLE_POLICY_KIND
    assert policy["payload"]["lifecycle_policy_id"] == LIFECYCLE_POLICY_ID
    assert policy["payload"]["probation_enters_production"] is False
    assert policy["payload"]["cost_evidence_required_to_apply"] is True
    assert policy["payload"]["capacity_evidence_required_to_apply"] is True
    assert policy["payload"]["authority"] == LIFECYCLE_AUTHORITY
    assert validate_lifecycle_policy(policy)["artifact_id"] == policy["artifact_id"]


def test_probation_is_not_production_eligible() -> None:
    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")["payload"]
    assert production_eligible(PROBATION, policy=policy) is False
    assert production_eligible(ACTIVE, policy=policy) is True
    assert production_eligible(WATCH, policy=policy) is True
    assert weight_scale_for_state(PROBATION, policy=policy) == "0.500000000000"
    assert weight_scale_for_state(ACTIVE, policy=policy) == "1.000000000000"


def test_transition_requires_evidence_then_watch_and_retire() -> None:
    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")["payload"]
    state, below, reasons = transition_lifecycle_state(
        ACTIVE,
        policy=policy,
        available_origin_sessions=20,
        cohort_count=2,
        posterior_p_positive=0.9,
        cusum_alarm=False,
        below_retire_months=0,
    )
    assert state == INSUFFICIENT_EVIDENCE
    assert "EVIDENCE_BELOW_MINIMUM" in reasons

    state, below, reasons = transition_lifecycle_state(
        ACTIVE,
        policy=policy,
        available_origin_sessions=60,
        cohort_count=8,
        posterior_p_positive=0.65,
        cusum_alarm=False,
        below_retire_months=0,
    )
    assert state == WATCH
    assert below == 0

    state, below, reasons = transition_lifecycle_state(
        WATCH,
        policy=policy,
        available_origin_sessions=60,
        cohort_count=8,
        posterior_p_positive=0.40,
        cusum_alarm=True,
        below_retire_months=2,
    )
    assert state == RETIRED
    assert below == 3
    assert "RETIRED_FROM_WATCH" in reasons


def test_preregistered_enters_probation_with_residual_ic() -> None:
    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")["payload"]
    state, _, reasons = transition_lifecycle_state(
        PREREGISTERED,
        policy=policy,
        available_origin_sessions=60,
        cohort_count=12,
        posterior_p_positive=0.85,
        cusum_alarm=False,
        below_retire_months=0,
        residual_ic_positive=True,
    )
    assert state == PROBATION
    assert "ENTERED_PROBATION" in reasons
    state, _, reasons = transition_lifecycle_state(
        PREREGISTERED,
        policy=policy,
        available_origin_sessions=60,
        cohort_count=12,
        posterior_p_positive=0.85,
        cusum_alarm=False,
        below_retire_months=0,
        residual_ic_positive=False,
    )
    assert state == PREREGISTERED
    assert "RESIDUAL_IC_NOT_POSITIVE" in reasons


def test_weight_proposal_caps_and_never_self_authorizes() -> None:
    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")["payload"]
    proposal = propose_lifecycle_weights(
        [
            {
                "factor_id": "alpha",
                "state": ACTIVE,
                "posterior_mean": "0.050000000000",
                "weight_scale": "1.000000000000",
            },
            {
                "factor_id": "beta",
                "state": ACTIVE,
                "posterior_mean": "0.010000000000",
                "weight_scale": "1.000000000000",
            },
            {
                "factor_id": "gamma",
                "state": PROBATION,
                "posterior_mean": "0.080000000000",
                "weight_scale": "0.500000000000",
            },
        ],
        policy=policy,
    )
    assert proposal["application_authorized"] is False
    assert "gamma" not in proposal["target_weights"]
    assert all(float(weight) <= 0.350000000001 for weight in proposal["target_weights"].values())


def test_decision_seals_with_outcome_refs() -> None:
    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")
    decision = build_lifecycle_decision(
        lifecycle_policy=policy,
        decision_month="2026-09",
        factor_rows=[
            {
                "factor_id": "pv_low_dollar_volume_5d",
                "lane": "bootstrap",
                "previous_state": ACTIVE,
                "state": ACTIVE,
                "posterior_mean": "0.020000000000",
                "posterior_p_positive": "0.900000000000",
                "cusum_alarm": False,
                "below_retire_months": 0,
                "weight_scale": "1.000000000000",
                "production_eligible": True,
                "reasons": ["REMAINS_ACTIVE"],
            }
        ],
        outcome_head_refs={
            "pv_low_dollar_volume_5d": _ref("factor.observation_head", "head-low"),
        },
        source_gap_counts={"pv_low_dollar_volume_5d": 0},
        trusted_at="2026-10-07T00:00:00Z",
    )
    assert decision["kind"] == LIFECYCLE_DECISION_KIND
    assert decision["payload"]["weight_proposal"]["application_authorized"] is False
    assert validate_lifecycle_decision(decision)["artifact_id"] == decision["artifact_id"]


def test_decision_rejects_self_authorized_weight_proposal() -> None:
    from quant_investor.contracts import seal_artifact
    from quant_investor.factors.governance.common import artifact_ref, business_identity

    policy = build_lifecycle_policy_v1(trusted_at="2026-10-07T00:00:00Z")
    rows = [
        {
            "factor_id": "pv_low_dollar_volume_5d",
            "lane": "bootstrap",
            "previous_state": ACTIVE,
            "state": ACTIVE,
            "posterior_mean": "0.020000000000",
            "posterior_p_positive": "0.900000000000",
            "cusum_alarm": False,
            "below_retire_months": 0,
            "weight_scale": "1.000000000000",
            "production_eligible": True,
            "reasons": ["REMAINS_ACTIVE"],
        }
    ]
    weight_proposal = propose_lifecycle_weights(rows, policy=policy["payload"])
    weight_proposal["application_authorized"] = True
    identity = {
        "lifecycle_policy_ref": artifact_ref(policy),
        "decision_month": "2026-09",
        "previous_decision_ref": None,
        "factor_rows": rows,
        "weight_proposal": weight_proposal,
        "source_gap_counts": {"pv_low_dollar_volume_5d": 0},
        "outcome_head_refs": {
            "pv_low_dollar_volume_5d": _ref("factor.observation_head", "head-low"),
        },
        "blockers": [],
        "authority": LIFECYCLE_AUTHORITY,
    }
    payload = {
        "lifecycle_decision_id": business_identity("factor-lifecycle-decision", identity),
        **identity,
    }
    broken = seal_artifact(LIFECYCLE_DECISION_KIND, payload, created_at="2026-10-07T00:00:00Z")
    with pytest.raises(FactorGovernanceError, match="self-authorize"):
        validate_lifecycle_decision(broken)


def test_registries_split_bootstrap_from_prospective() -> None:
    assert BOOTSTRAP_FACTOR_IDS == {LOW_DOLLAR_VOLUME, BLEND_W80}
    assert PROSPECTIVE_FACTOR_IDS == set(PRICE_VOLUME_CANDIDATES)
    assert not (BOOTSTRAP_FACTOR_IDS & PROSPECTIVE_FACTOR_IDS)
    semantic = installed_semantic_row(VOLATILITY_PENALTY_5D)
    assert semantic["factor_id"] == VOLATILITY_PENALTY_5D
    assert "NEGATIVE_RETURN_STD" in semantic["normalized_expression"]


def test_compute_installed_signals_covers_prospective_price_volume() -> None:
    dates = pd.bdate_range("2026-01-05", periods=30)
    frames = {
        "000001.SZ": pd.DataFrame(
            {
                "trade_date": dates.strftime("%Y%m%d"),
                "adj_close": [10.0 + 0.1 * index for index in range(30)],
                "amount": [1.0e8] * 30,
                "vol": [1.0e6] * 30,
            }
        ),
        "600000.SH": pd.DataFrame(
            {
                "trade_date": dates.strftime("%Y%m%d"),
                "adj_close": [20.0 - 0.05 * index for index in range(30)],
                "amount": [2.0e8] * 30,
                "vol": [2.0e6] * 30,
            }
        ),
    }
    signals = compute_installed_signals(frames, factor_ids=[VOLATILITY_PENALTY_5D])
    assert set(signals) == {VOLATILITY_PENALTY_5D}
    assert signals[VOLATILITY_PENALTY_5D].notna().all()
