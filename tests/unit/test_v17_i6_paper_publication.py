from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from quant_investor.intelligence_v2._core import canonical_bytes, content_ref
from quant_investor.intelligence_v2.portfolio import (
    build_graduation_policy,
    build_graduation_receipt,
    build_paper_execution_policy,
    build_paper_fill,
    build_paper_ledger,
    build_paper_order,
    build_paper_outcome,
    validate_graduation_receipt,
    validate_paper_fill,
    validate_paper_ledger,
    validate_paper_order,
)
from quant_investor.intelligence_v2.publication import (
    PUBLICATION_CLOSURE_VERSIONS,
    PublicationContractError,
    build_action_permit,
    build_activation_sidecar,
    build_expected_pointer_cas_receipt,
    build_expected_pointer_cas_request,
    build_legacy_marker,
    build_legacy_marker_profile,
    build_publication_closure,
    build_publication_owner_policy,
    build_preactivation_receipt,
    build_quarantine_receipt,
    build_rollback_receipt,
    derive_publication_paths,
    permit_message,
    validate_action_permit,
    validate_activation_sidecar,
    validate_expected_pointer_cas_receipt,
    validate_expected_pointer_cas_request,
    validate_legacy_marker,
    validate_preactivation_receipt,
    validate_publication_closure,
    validate_quarantine_receipt,
    validate_rollback_receipt,
)

AT = "2026-08-09T01:00:00Z"
LATER = "2026-08-09T01:02:00Z"


def _exact_ref(
    name: str,
    *,
    version: str | None = None,
    artifact_id: str | None = None,
    relative_path: str | None = None,
) -> dict[str, str]:
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "artifact_id": artifact_id or name,
        "artifact_version": version or f"myquant.test.{name}.v1",
        "available_at": AT,
        "byte_sha256": digest,
        "cutoff": AT,
        "relative_path": relative_path or f"private/{name}.json",
        "semantic_sha256": digest,
    }


def _paper_policy() -> dict:
    return build_paper_execution_policy(
        created_at=AT,
        effective_from_session="20260801",
        effective_through_session="20260831",
        lot_size=100,
        settlement_rule="T_PLUS_ONE",
        buy_commission_rate="0.0003",
        sell_commission_rate="0.0003",
        minimum_commission_cny="5",
        transfer_fee_rate="0.00001",
        sell_stamp_duty_rate="0.0005",
        slippage_rate="0.001",
        max_fill_adv_participation="0.10",
        fee_rounding_quantum_cny="0.0001",
        fee_rounding_mode="ROUND_HALF_EVEN",
        price_rounding_quantum_cny="0.01",
        allow_partial_fills=True,
        allow_odd_lot_full_exit=True,
        order_expiry_rule="EXPLICIT_SESSION",
        partial_fill_ordering="QUEUE_PRIORITY_THEN_ORDER_ID",
        corporate_action_policy="EXACT_SOURCE_CHRONOLOGY",
        listing_policy="LISTED_NOT_DELISTED",
        exchange_calendar_ref=_exact_ref("calendar"),
        exchange_calendar_sessions=[
            {
                "session": value,
                "source_ref": _exact_ref(f"calendar-{value}"),
                "status": "OPEN",
            }
            for value in ("20260808", "20260809")
        ],
        price_limit_rules=[
            {
                "board": "MAIN",
                "effective_from_session": "20260801",
                "effective_through_session": "20260831",
                "ipo_no_limit_sessions": 5,
                "limit_ratio": "0.10",
                "rule_id": "MAIN_NORMAL",
                "source_ref": _exact_ref("main-limit"),
                "st": False,
            }
        ],
    )


def _order_closure(policy: dict, *, side: str = "PAPER_BUY", shares: int = 200) -> dict:
    return {
        "policy": policy,
        "company": "000001.SZ",
        "side": side,
        "requested_shares": shares,
        "position_shares": shares if side == "PAPER_SELL" else 0,
        "acquired_session": "20260808" if side == "PAPER_SELL" else None,
        "decision_session": "20260808",
        "execution_session": "20260809",
        "expires_session": "20260809",
        "queue_priority": 1,
        "cancellation_ref": None,
        "market_ref": _exact_ref("order-market"),
        "created_at": AT,
    }


def _market(*, price: str = "10", volume: int = 1_500) -> dict:
    return {
        "available_volume_shares": volume,
        "board": "MAIN",
        "company_code": "000001.SZ",
        "corporate_action_refs": [],
        "delisting_session": None,
        "execution_price": price,
        "is_st": False,
        "listing_session": "20200101",
        "lower_limit": "9.00",
        "previous_close": "10",
        "session": "20260809",
        "sessions_since_listing": 100,
        "source_ref": _exact_ref(f"market-{price}-{volume}"),
        "suspended": False,
        "upper_limit": "11.00",
    }


def test_a_share_order_fill_partial_fee_lot_t1_and_no_policy() -> None:
    policy = _paper_policy()
    closure = _order_closure(policy)
    order = build_paper_order(**closure)
    assert order["status"] == "READY"
    assert validate_paper_order(order, **closure) == order

    fill_closure = {
        "order": order,
        "order_validation_closure": closure,
        "policy": policy,
        "market_observation": _market(),
        "filled_at": LATER,
    }
    fill = build_paper_fill(**fill_closure)
    assert fill["status"] == "PARTIALLY_FILLED"
    assert fill["filled_shares"] == 100
    assert fill["fill_price"] == "10.0100"
    assert fill["gross_value_cny"] == "1001.0000"
    assert fill["commission_cny"] == "5.0000"
    assert fill["stamp_duty_cny"] == "0.0000"
    assert validate_paper_fill(fill, **fill_closure) == fill

    no_policy = build_paper_order(**{**closure, "policy": None})
    assert no_policy["status"] == "BLOCKED"
    assert no_policy["blocker_codes"] == ["PAPER_EXECUTION_POLICY_UNAVAILABLE"]
    invalid_lot = build_paper_order(**{**closure, "requested_shares": 150})
    assert invalid_lot["status"] == "BLOCKED"
    assert invalid_lot["blocker_codes"] == ["BUY_LOT_INVALID"]

    same_day_sell = build_paper_order(
        **{
            **_order_closure(policy, side="PAPER_SELL", shares=100),
            "acquired_session": "20260809",
        }
    )
    assert "T_PLUS_ONE_BLOCKED" in same_day_sell["blocker_codes"]


def test_limit_rules_odd_lot_exit_ledger_outcomes_and_graduation() -> None:
    policy = _paper_policy()
    buy_closure = _order_closure(policy)
    buy = build_paper_order(**buy_closure)
    blocked_fill = build_paper_fill(
        order=buy,
        order_validation_closure=buy_closure,
        policy=policy,
        market_observation=_market(price="11"),
        filled_at=LATER,
    )
    assert blocked_fill["status"] == "BLOCKED"
    assert blocked_fill["blocker_codes"] == ["LIMIT_UP_BUY_BLOCKED"]

    sell_closure = _order_closure(policy, side="PAPER_SELL", shares=50)
    sell = build_paper_order(**sell_closure)
    assert sell["status"] == "READY"
    sell_fill_closure = {
        "order": sell,
        "order_validation_closure": sell_closure,
        "policy": policy,
        "market_observation": _market(volume=50),
        "filled_at": LATER,
    }
    sell_fill = build_paper_fill(**sell_fill_closure)
    assert sell_fill["status"] == "FILLED"
    assert sell_fill["filled_shares"] == 50
    assert sell_fill["stamp_duty_cny"] == "0.2498"

    ledger_closure = {
        "fills": [sell_fill],
        "fill_validation_closures": [sell_fill_closure],
        "opening_cash_cny": "1000",
        "opening_positions": {"000001.SZ": 50},
        "created_at": "2026-08-09T01:03:00Z",
    }
    ledger = build_paper_ledger(**ledger_closure)
    assert ledger["positions"] == []
    assert ledger["closing_cash_cny"] == "1494.2452"
    assert validate_paper_ledger(ledger, **ledger_closure) == ledger

    ledger_ref = content_ref(ledger, identity_field="ledger_id")
    outcome_closures = []
    outcomes = []
    for horizon, observed, matured_at in (
        (20, "0.10", "2026-08-09T02:20:00Z"),
        (60, "0.12", "2026-08-09T03:00:00Z"),
    ):
        closure = {
            "ledger_ref": ledger_ref,
            "horizon_sessions": horizon,
            "observed_return": observed,
            "benchmark_return": "0.02",
            "maximum_drawdown": "0.10",
            "turnover": "0.20",
            "cost_ratio": "0.005",
            "hard_risk_breach": False,
            "benchmark_ref": _exact_ref("benchmark"),
            "entry_price_ref": _exact_ref(f"entry-{horizon}"),
            "outcome_price_ref": _exact_ref(f"outcome-{horizon}"),
            "regime_ref": _exact_ref(f"regime-{horizon}", artifact_id="NORMAL"),
            "matured_at": matured_at,
        }
        outcome_closures.append(closure)
        outcomes.append(build_paper_outcome(**closure))
    graduation_policy = build_graduation_policy(
        created_at=AT,
        required_horizons=[20, 60],
        benchmark_ref=_exact_ref("benchmark"),
        minimum_matured_observations=2,
        minimum_coverage="1",
        minimum_cost_adjusted_excess_return="0.05",
        maximum_drawdown="0.20",
        maximum_regime_changes=0,
        require_no_hard_risk_breach=True,
    )
    graduation_closure = {
        "policy": graduation_policy,
        "outcomes": outcomes,
        "outcome_validation_closures": outcome_closures,
        "evaluated_at": "2026-08-09T04:00:00Z",
    }
    graduation = build_graduation_receipt(**graduation_closure)
    assert graduation["status"] == "ELIGIBLE_FOR_OWNER_REVIEW"
    assert graduation["production"] is False
    assert validate_graduation_receipt(graduation, **graduation_closure) == graduation
    unavailable = build_graduation_receipt(**{**graduation_closure, "policy": None})
    assert unavailable["status"] == "NOT_ELIGIBLE"
    assert unavailable["blocker_codes"] == ["GRADUATION_POLICY_UNAVAILABLE"]


def test_calendar_cancellation_ipo_and_corporate_action_boundaries() -> None:
    policy = _paper_policy()
    uncovered = build_paper_order(
        **{
            **_order_closure(policy),
            "execution_session": "20260810",
            "expires_session": "20260810",
        }
    )
    assert "CALENDAR_SESSION_COVERAGE_UNAVAILABLE" in uncovered["blocker_codes"]

    cancelled = build_paper_order(
        **{
            **_order_closure(policy),
            "cancellation_ref": _exact_ref("cancelled-order"),
        }
    )
    assert cancelled["status"] == "BLOCKED"
    assert "ORDER_CANCELLED" in cancelled["blocker_codes"]

    closure = _order_closure(policy)
    order = build_paper_order(**closure)
    corporate_market = _market()
    corporate_market["corporate_action_refs"] = [_exact_ref("cash-dividend")]
    corporate = build_paper_fill(
        order=order,
        order_validation_closure=closure,
        policy=policy,
        market_observation=corporate_market,
        filled_at=LATER,
    )
    assert corporate["filled_shares"] == 0
    assert "CORPORATE_ACTION_SIMULATION_UNAVAILABLE" in corporate["blocker_codes"]

    ipo_market = _market(price="20")
    ipo_market["sessions_since_listing"] = 1
    ipo_market["lower_limit"] = None
    ipo_market["upper_limit"] = None
    ipo_fill = build_paper_fill(
        order=order,
        order_validation_closure=closure,
        policy=policy,
        market_observation=ipo_market,
        filled_at=LATER,
    )
    assert "LIMIT_UP_BUY_BLOCKED" not in ipo_fill["blocker_codes"]
    assert ipo_fill["filled_shares"] == 100


def _publication_graph() -> tuple[dict, dict, dict, dict]:
    profile = build_legacy_marker_profile(created_at=AT, canonical_strategy_id="research_portfolio")
    marker_closure = {
        "profile": profile,
        "transaction_id": "tx-001",
        "legacy_run_ref": _exact_ref("legacy-run", version="myquant.v17.v4.mainline-run.v1"),
        "target_pointer_ref": _exact_ref(
            "legacy-pointer", version="myquant.v17.v4.mainline-active-pointer.v1"
        ),
        "portfolio_ref": content_ref(_paper_policy(), identity_field="policy_id"),
        "risk_ref": {
            key: value
            for key, value in _exact_ref(
                "evidence-graph",
                version=PUBLICATION_CLOSURE_VERSIONS["EVIDENCE_GRAPH_V2"],
            ).items()
            if key
            in {
                "artifact_id",
                "artifact_version",
                "byte_sha256",
                "semantic_sha256",
            }
        },
        "graduation_ref": content_ref(
            build_graduation_policy(
                created_at=AT,
                required_horizons=[20, 60],
                benchmark_ref=_exact_ref("benchmark"),
                minimum_matured_observations=2,
                minimum_coverage="1",
                minimum_cost_adjusted_excess_return="0",
                maximum_drawdown="1",
                maximum_regime_changes=1,
                require_no_hard_risk_breach=True,
            ),
            identity_field="policy_id",
        ),
        "built_at": AT,
    }
    marker = build_legacy_marker(**marker_closure)
    closure_refs = {
        key: _exact_ref(
            key.lower(),
            version=version,
            relative_path=(f"results/v17_intelligence_v2/fixtures/{key.lower()}.json"),
        )
        for key, version in PUBLICATION_CLOSURE_VERSIONS.items()
        if key != "PREACTIVATION"
    }
    profile_bytes = canonical_bytes(profile)
    closure_refs["LEGACY_MARKER_PROFILE"] = {
        "artifact_id": profile["profile_id"],
        "artifact_version": profile["version"],
        "available_at": AT,
        "byte_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "cutoff": profile["timestamp"],
        "relative_path": ("results/v17_intelligence_v2/fixtures/legacy-marker-profile.json"),
        "semantic_sha256": profile["semantic_sha256"],
    }
    candidates = sorted(
        [closure_refs["DECISION_V2"], closure_refs["GRADUATION"], closure_refs["PORTFOLIO"]],
        key=lambda row: (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        ),
    )
    preactivation = build_preactivation_receipt(
        candidate_refs=candidates,
        expected_pointer_sha256=marker["target_pointer_ref"]["byte_sha256"],
        rollback_target_ref=marker["target_pointer_ref"],
        blocker_codes=[],
        evaluated_at=LATER,
    )
    preactivation_bytes = canonical_bytes(preactivation)
    closure_refs["PREACTIVATION"] = {
        "artifact_id": preactivation["preactivation_id"],
        "artifact_version": preactivation["version"],
        "available_at": LATER,
        "byte_sha256": hashlib.sha256(preactivation_bytes).hexdigest(),
        "cutoff": preactivation["timestamp"],
        "relative_path": ("results/v17_intelligence_v2/fixtures/preactivation-receipt.json"),
        "semantic_sha256": preactivation["semantic_sha256"],
    }
    publication_closure = build_publication_closure(
        canonical_strategy_id="research_portfolio",
        transaction_id="tx-001",
        closure_refs=closure_refs,
        outcome_refs=[
            _exact_ref(
                "paper-outcome",
                relative_path="results/v17_intelligence_v2/fixtures/paper-outcome.json",
            )
        ],
        built_at=LATER,
    )
    sidecar_closure = {
        "marker": marker,
        "marker_validation_closure": marker_closure,
        "publication_closure": publication_closure,
        "built_at": LATER,
    }
    sidecar = build_activation_sidecar(**sidecar_closure)
    return marker, marker_closure, sidecar, sidecar_closure


def test_preactivation_is_real_replayable_and_fail_closed() -> None:
    marker, _, _, closure = _publication_graph()
    candidates = sorted(
        [
            closure["publication_closure"]["nodes"]["DECISION_V2"],
            closure["publication_closure"]["nodes"]["GRADUATION"],
        ],
        key=lambda row: (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        ),
    )
    receipt_closure = {
        "candidate_refs": candidates,
        "expected_pointer_sha256": marker["target_pointer_ref"]["byte_sha256"],
        "rollback_target_ref": marker["target_pointer_ref"],
        "blocker_codes": [],
        "evaluated_at": LATER,
    }
    receipt = build_preactivation_receipt(**receipt_closure)
    assert receipt["readiness"] is True
    assert receipt["status"] == "READY"
    assert receipt["write_performed"] is False
    assert validate_preactivation_receipt(receipt, **receipt_closure) == receipt

    blocked = build_preactivation_receipt(
        **{**receipt_closure, "blocker_codes": ["GRADUATION_NOT_READY"]}
    )
    assert blocked["readiness"] is False
    assert blocked["status"] == "NOT_READY"
    tampered = dict(receipt)
    tampered["status"] = "NOT_READY"
    with pytest.raises(ValueError, match="mismatch"):
        validate_preactivation_receipt(tampered, **receipt_closure)


def test_publication_sidecar_paths_dag_cas_quarantine_and_rollback() -> None:
    marker, marker_closure, sidecar, sidecar_closure = _publication_graph()
    assert validate_legacy_marker(marker, **marker_closure) == marker
    assert validate_activation_sidecar(sidecar, **sidecar_closure) == sidecar
    assert "sidecar_ref" not in marker
    assert sidecar["marker_ref"]["relative_path"] == marker["marker_path"]
    assert validate_publication_closure(sidecar_closure["publication_closure"]) == (
        sidecar_closure["publication_closure"]
    )
    paths = derive_publication_paths(
        strategy_id="research_portfolio",
        transaction_id="tx-001",
        target_pointer_sha256=marker["target_pointer_ref"]["byte_sha256"],
        run_id=marker["legacy_run_ref"]["artifact_id"],
    )
    assert paths["activation_sidecar"].startswith("results/v17_intelligence_v2/")
    assert paths["activation_permit"].endswith(
        f"/activation_permits/{marker['target_pointer_ref']['byte_sha256']}.json"
    )
    assert sidecar["publication_profile"] == "INTELLIGENCE_V2_PUBLICATION_V1"

    placeholder_permit_ref = marker["graduation_ref"]
    request_closure = {
        "sidecar_ref": content_ref(sidecar, identity_field="sidecar_id"),
        "permit_ref": placeholder_permit_ref,
        "canonical_strategy_id": "research_portfolio",
        "transaction_id": "tx-001",
        "run_id": marker["legacy_run_ref"]["artifact_id"],
        "expected_pointer_sha256": "EMPTY",
        "target_pointer_ref": marker["target_pointer_ref"],
        "requested_at": LATER,
    }
    request = build_expected_pointer_cas_request(**request_closure)
    assert request["write_performed"] is False
    assert validate_expected_pointer_cas_request(request, **request_closure) == request
    receipt_closure = {
        "request_ref": content_ref(request, identity_field="request_id"),
        "expected_pointer_sha256": "EMPTY",
        "observed_pointer_sha256": "EMPTY",
        "target_pointer_sha256": marker["target_pointer_ref"]["byte_sha256"],
        "status": "NOT_ATTEMPTED",
        "received_at": LATER,
    }
    receipt = build_expected_pointer_cas_receipt(**receipt_closure)
    assert validate_expected_pointer_cas_receipt(receipt, **receipt_closure) == receipt

    quarantine_closure = {
        "sidecar_ref": request_closure["sidecar_ref"],
        "permit_ref": placeholder_permit_ref,
        "canonical_strategy_id": "research_portfolio",
        "transaction_id": "tx-001",
        "target_pointer_sha256": marker["target_pointer_ref"]["byte_sha256"],
        "run_id": marker["legacy_run_ref"]["artifact_id"],
        "reason_codes": ["OWNER_REVIEW"],
        "quarantined_at": LATER,
    }
    quarantine = build_quarantine_receipt(**quarantine_closure)
    assert validate_quarantine_receipt(quarantine, **quarantine_closure) == quarantine
    rollback_closure = {
        "sidecar_ref": request_closure["sidecar_ref"],
        "permit_ref": placeholder_permit_ref,
        "expected_current_pointer_sha256": marker["target_pointer_ref"]["byte_sha256"],
        "rollback_target_ref": marker["target_pointer_ref"],
        "status": "REQUESTED",
        "rolled_back_at": LATER,
    }
    rollback = build_rollback_receipt(**rollback_closure)
    assert rollback["write_performed"] is False
    assert validate_rollback_receipt(rollback, **rollback_closure) == rollback


def test_publication_closure_rejects_missing_unknown_and_version_substitution() -> None:
    _, _, _, closure = _publication_graph()
    publication = closure["publication_closure"]
    base = {
        "canonical_strategy_id": publication["canonical_strategy_id"],
        "transaction_id": publication["transaction_id"],
        "closure_refs": publication["nodes"],
        "outcome_refs": publication["outcome_refs"],
        "built_at": publication["timestamp"],
    }
    missing = dict(base)
    missing["closure_refs"] = dict(base["closure_refs"])
    missing["closure_refs"].pop("PREACTIVATION")
    with pytest.raises(ValueError, match="shape"):
        build_publication_closure(**missing)

    unknown = dict(base)
    unknown["closure_refs"] = dict(base["closure_refs"])
    unknown["closure_refs"]["UNKNOWN"] = _exact_ref("unknown")
    with pytest.raises(ValueError, match="shape"):
        build_publication_closure(**unknown)

    substituted = dict(base)
    substituted["closure_refs"] = dict(base["closure_refs"])
    substituted["closure_refs"]["DECISION_V2"] = _exact_ref(
        "wrong-decision",
        version="myquant.wrong.v1",
        relative_path="results/v17_intelligence_v2/fixtures/wrong-decision.json",
    )
    with pytest.raises(PublicationContractError, match="artifact version"):
        build_publication_closure(**substituted)

    duplicated = dict(base)
    duplicated["closure_refs"] = dict(base["closure_refs"])
    duplicated_portfolio = dict(duplicated["closure_refs"]["PORTFOLIO"])
    duplicated_portfolio["byte_sha256"] = duplicated["closure_refs"]["DECISION_V2"]["byte_sha256"]
    duplicated_portfolio["semantic_sha256"] = duplicated["closure_refs"]["DECISION_V2"][
        "semantic_sha256"
    ]
    duplicated["closure_refs"]["PORTFOLIO"] = duplicated_portfolio
    with pytest.raises(PublicationContractError, match="duplicate refs"):
        build_publication_closure(**duplicated)


def _owner_policy(*, public_key: bytes, revoked_at: str | None = None) -> tuple[dict, str]:
    key_id = hashlib.sha256(public_key).hexdigest()
    return (
        build_publication_owner_policy(
            created_at=AT,
            maximum_permit_lifetime_seconds=600,
            keys=[
                {
                    "actions": ["ACTIVATE", "QUARANTINE", "ROLLBACK"],
                    "algorithm": "ED25519",
                    "key_id": key_id,
                    "not_after": "2026-08-10T01:00:00Z",
                    "not_before": AT,
                    "public_key_base64": base64.b64encode(public_key).decode(),
                    "revoked_at": revoked_at,
                }
            ],
        ),
        key_id,
    )


def test_action_specific_permit_expiry_revocation_and_real_ed25519() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _, _, sidecar, _ = _publication_graph()
    subject_ref = content_ref(sidecar, identity_field="sidecar_id")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    owner_policy, key_id = _owner_policy(public_key=public_key)
    target_sha = sidecar["target_pointer_ref"]["byte_sha256"]
    permit_args = {
        "action": "ACTIVATE",
        "canonical_strategy_id": "research_portfolio",
        "subject_ref": subject_ref,
        "expected_pointer_sha256": "EMPTY",
        "target_pointer_sha256": target_sha,
        "issued_at": AT,
        "not_before": AT,
        "expires_at": "2026-08-09T01:05:00Z",
        "nonce": hashlib.sha256(b"nonce").hexdigest(),
        "signer_key_id": key_id,
    }
    unsigned = build_action_permit(
        **permit_args,
        signature_base64=base64.b64encode(bytes(64)).decode(),
    )
    signature = private_key.sign(permit_message(unsigned["claims"]))
    permit = build_action_permit(
        **permit_args,
        signature_base64=base64.b64encode(signature).decode(),
    )
    validated = validate_action_permit(
        permit,
        owner_policy=owner_policy,
        expected_action="ACTIVATE",
        expected_subject_ref=subject_ref,
        expected_strategy_id="research_portfolio",
        expected_pointer_sha256="EMPTY",
        target_pointer_sha256=target_sha,
        verified_at=LATER,
    )
    assert validated == permit
    assert permit_message(permit["claims"]).startswith(
        b"myquant.v17.intelligence-v2.publication-permit.v1:ACTIVATE\x00"
    )
    with pytest.raises(PublicationContractError, match="claims"):
        validate_action_permit(
            permit,
            owner_policy=owner_policy,
            expected_action="ROLLBACK",
            expected_subject_ref=subject_ref,
            expected_strategy_id="research_portfolio",
            expected_pointer_sha256="EMPTY",
            target_pointer_sha256=target_sha,
            verified_at=LATER,
        )
    with pytest.raises(PublicationContractError, match="validity window"):
        validate_action_permit(
            permit,
            owner_policy=owner_policy,
            expected_action="ACTIVATE",
            expected_subject_ref=subject_ref,
            expected_strategy_id="research_portfolio",
            expected_pointer_sha256="EMPTY",
            target_pointer_sha256=target_sha,
            verified_at="2026-08-09T01:06:00Z",
        )
    revoked_policy, _ = _owner_policy(public_key=public_key, revoked_at="2026-08-09T01:01:00Z")
    with pytest.raises(PublicationContractError, match="revoked"):
        validate_action_permit(
            permit,
            owner_policy=revoked_policy,
            expected_action="ACTIVATE",
            expected_subject_ref=subject_ref,
            expected_strategy_id="research_portfolio",
            expected_pointer_sha256="EMPTY",
            target_pointer_sha256=target_sha,
            verified_at=LATER,
        )


def test_i6_builders_do_not_write_mainline_data_or_results() -> None:
    roots = [Path("data"), Path("results")]

    def inventory() -> list[tuple[str, int, int]]:
        return sorted(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for root in roots
            if root.exists()
            for path in root.rglob("*")
            if path.is_file()
        )

    before = inventory()
    _publication_graph()
    _paper_policy()
    assert inventory() == before
