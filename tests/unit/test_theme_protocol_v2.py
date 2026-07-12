from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.themes.protocol_v2 import (
    ThemeLifecycle,
    ThemeProtocolConfig,
    build_theme_protocol_hash,
    evaluate_theme_protocol_v2,
    reconcile_theme_protocol_v2,
    tactical_lane_cap,
    transition_theme_lifecycle,
    write_theme_formal_reconciliation_artifact,
)
from quant_investor.themes.taxonomy import ThemeTaxonomy


def _taxonomy() -> ThemeTaxonomy:
    return ThemeTaxonomy.load()


def _score(theme_id: str, *, attention: float = 0.80) -> dict[str, object]:
    return {
        "theme_id": theme_id,
        "theme_name": theme_id,
        "score": 80.0,
        "smoothed_score": 78.0,
        "attention": attention,
        "attention_5d": 0.75,
        "attention_20d": 0.80,
        "attention_60d": 0.82,
        "attention_120d": 0.78,
        "attention_turnover_share": 0.12,
        "new_high_rate": 0.60,
        "leader_persistence": 0.70,
        "attention_history_coverage": 1.0,
        "market_confirmation": 0.80,
        "breadth": 0.80,
        "momentum": 0.80,
        "volume_confirmation": 0.80,
        "acceleration": 0.80,
        "crowding_risk": 0.20,
        "crowding_status": "success",
        "valuation_risk": 0.20,
        "valuation_risk_status": "fresh",
        "member_count": 10,
    }


def _event(theme_id: str, event_id: str, *, available_at: str = "2026-07-01") -> dict[str, object]:
    return {
        "schema_version": "theme_evidence_event.v1",
        "event_id": event_id,
        "theme_id": theme_id,
        "event_type": "order",
        "event_date": "2026-06-30",
        "available_at": available_at,
        "direction": 1,
        "strength": 0.90,
        "confidence": 0.90,
    }


def _gates(*theme_ids: str) -> dict[str, dict[str, bool]]:
    return {
        theme_id: {
            "data_pass": True,
            "tradability_pass": True,
            "liquidity_pass": True,
            "positive_edge_or_buy": True,
            "risk_guard_pass": True,
            "portfolio_constructor_pass": True,
        }
        for theme_id in theme_ids
    }


def _membership_details(*theme_ids: str) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "theme_membership.v2",
            "membership_id": f"membership-{index}",
            "theme_id": theme_id,
            "theme_name": theme_id,
            "theme_type": "technology" if theme_id.startswith("tech::") else "concept",
            "symbol": f"S{index:03d}.SZ",
            "taxonomy_node_id": theme_id,
            "supply_chain_role": "test",
            "revenue_exposure": None,
            "effective_from": "2026-01-01",
            "available_at": "2026-07-01",
            "confidence": 0.9,
            "source_type": "local_test",
            "source_ref": f"test:{theme_id}",
        }
        for index, theme_id in enumerate(theme_ids, start=1)
    ]


def test_initial_taxonomy_covers_locked_technology_themes_and_supply_chains() -> None:
    taxonomy = _taxonomy()

    for alias in ("AI", "半导体", "光模块", "商业航天", "人形机器人"):
        node = taxonomy.resolve(alias)
        assert node is not None
        assert node.tradable_node is True
        assert node.mandate in {"technology", "advanced_manufacturing"}
        assert node.supply_chain_roles


def test_protocol_hash_is_stable_and_observer_default_has_empty_formal_pool() -> None:
    taxonomy = _taxonomy()
    config = ThemeProtocolConfig()
    first_hash = build_theme_protocol_hash(taxonomy=taxonomy, config=config)
    second_hash = build_theme_protocol_hash(taxonomy=taxonomy, config=config)

    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=taxonomy,
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-1")],
        valid_membership_theme_ids=["tech::ai"],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        downstream_gates=_gates("tech::ai"),
    )

    assert first_hash == second_hash == result["protocol_hash"]
    assert result["status"] == "observer"
    assert result["formal_enabled"] is False
    assert result["formal_pool"] == []
    assert result["forced_theme_count"] == 0
    assert result["rollback_reason"] == "formal_not_enabled_observer_only"
    state = result["states"]["tech::ai"]
    assert state["attention_5d"] == 0.75
    assert state["attention_20d"] == 0.80
    assert state["attention_60d"] == 0.82
    assert state["attention_120d"] == 0.78


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_min": -0.01},
        {"crowding_block": 1.01},
        {"long_horizon_attention_min_coverage": 1.01},
        {"pevc_max_percentile_adjustment": 0.100001},
        {"evidence_stale_days": 0},
        {"upward_confirmation_days": 0},
        {"cooling_confirmation_days": 1.5},
    ],
)
def test_protocol_config_rejects_invalid_thresholds_and_windows(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ThemeProtocolConfig(**kwargs)


def test_protocol_config_pevc_adjustment_default_is_ten_percentile() -> None:
    assert ThemeProtocolConfig().pevc_max_percentile_adjustment == 0.10


def test_formal_kill_switch_rolls_back_to_observer_even_when_all_gates_pass() -> None:
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-1")],
        valid_membership_theme_ids=["tech::ai"],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        downstream_gates=_gates("tech::ai"),
        formal_enabled=True,
        formal_kill_switch=True,
    )

    assert result["status"] == "observer"
    assert result["formal_pool"] == []
    assert result["rollback_status"] == "observer_only"
    assert result["rollback_reason"] == "formal_kill_switch_active"


def test_prequalified_pool_is_reachable_without_claiming_missing_downstream_gates_passed() -> None:
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-1")],
        valid_membership_theme_ids=["tech::ai"],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
        formal_kill_switch=False,
    )

    state = result["states"]["tech::ai"]
    assert result["status"] == "prequalified"
    assert result["prequalified_pool"] == ["tech::ai"]
    assert result["formal_pool"] == []
    assert result["downstream_gates_supplied"] is False
    assert state["prequalification_blockers"] == []
    assert "positive_edge_buy_gate_blocked" in state["downstream_blockers"]
    assert state["lane"] == "tech_thesis_watch"


def test_pevc_prior_adjusts_pit_prequalified_rank_by_at_most_ten_percentile() -> None:
    theme_ids = ("tech::ai", "tech::semiconductor")
    result = evaluate_theme_protocol_v2(
        theme_scores={theme_id: _score(theme_id) for theme_id in theme_ids},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[
            _event("tech::ai", "event-ai"),
            _event("tech::semiconductor", "event-semi"),
        ],
        pevc_theses=[
            {
                "status": "approved",
                "thesis_id": "ai-thesis",
                "theme_id": "tech::ai",
                "version": "1.0.0",
                "available_at": "2026-07-01",
                "approved_at": "2026-07-01",
                "review_by": "2026-12-31",
                "prior_score": 1.0,
            }
        ],
        valid_membership_theme_ids=theme_ids,
        theme_membership_details=_membership_details(*theme_ids),
        previous_states={theme_id: {"lifecycle": "validated_trend"} for theme_id in theme_ids},
        downstream_gates=_gates(*theme_ids),
        formal_enabled=True,
    )

    ai = result["states"]["tech::ai"]
    assert set(result["prequalified_pool"]) == set(theme_ids)
    assert result["formal_pool"] == []
    assert ai["lane"] == "tech_thesis_watch"
    assert 0.0 <= ai["pevc_rank_adjustment"] <= 0.10
    assert abs(ai["adjusted_percentile_rank"] - ai["base_percentile_rank"]) <= 0.10


def test_pevc_prior_cannot_bypass_attention_or_market_gates() -> None:
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai", attention=0.10)},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-ai")],
        pevc_theses=[
            {
                "status": "approved",
                "thesis_id": "ai-thesis",
                "theme_id": "tech::ai",
                "version": "1.0.0",
                "available_at": "2026-07-01",
                "approved_at": "2026-07-01",
                "review_by": "2026-12-31",
                "prior_score": 1.0,
            }
        ],
        valid_membership_theme_ids=["tech::ai"],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        downstream_gates=_gates("tech::ai"),
        formal_enabled=True,
    )

    state = result["states"]["tech::ai"]
    assert "attention_below_gate" in state["eligibility_blockers"]
    assert state["pevc_rank_adjustment"] == 0.0
    assert state["lane"] == "tech_thesis_watch"
    assert result["formal_pool"] == []


def test_future_pevc_approval_cannot_backdate_prior_into_prequalified_rank() -> None:
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-ai")],
        pevc_theses=[
            {
                "status": "approved",
                "thesis_id": "future-approval",
                "theme_id": "tech::ai",
                "version": "1.0.0",
                "available_at": "2020-01-01",
                "approved_at": "2026-07-11",
                "review_by": "2026-12-31",
                "prior_score": 1.0,
            }
        ],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )

    state = result["states"]["tech::ai"]
    assert state["pevc_prior"] == 0.0
    assert state["pevc_rank_adjustment"] == 0.0


def test_latest_pevc_thesis_prefers_natural_version_before_approval_date() -> None:
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-ai")],
        pevc_theses=[
            {
                "status": "approved",
                "thesis_id": "ai-v10",
                "theme_id": "tech::ai",
                "version": "v10",
                "available_at": "2026-07-01",
                "approved_at": "2026-07-01",
                "review_by": "2026-12-31",
                "prior_score": 0.2,
                "content_hash": "a" * 64,
            },
            {
                "status": "approved",
                "thesis_id": "ai-v9",
                "theme_id": "tech::ai",
                "version": "v9",
                "available_at": "2026-07-02",
                "approved_at": "2026-07-02",
                "review_by": "2026-12-31",
                "prior_score": 1.0,
                "content_hash": "b" * 64,
            },
        ],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )

    state = result["states"]["tech::ai"]
    assert state["thesis_id"] == "ai-v10"
    assert state["thesis_version"] == "v10"
    assert state["pevc_prior"] == pytest.approx(0.2)


def test_future_available_evidence_is_not_used_for_pit_eligibility() -> None:
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "future", available_at="2026-07-11")],
        valid_membership_theme_ids=["tech::ai"],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        downstream_gates=_gates("tech::ai"),
        formal_enabled=True,
    )

    state = result["states"]["tech::ai"]
    assert state["industrial_validation"] == 0.0
    assert "industrial_validation_below_gate" in state["eligibility_blockers"]
    assert "stale_industrial_evidence" in state["eligibility_blockers"]


def test_non_pit_industry_map_membership_cannot_prequalify() -> None:
    taxonomy = ThemeTaxonomy.from_mapping(
        {
            "schema_version": "theme_taxonomy.v2",
            "taxonomy_id": "industry-test",
            "version": "1",
            "nodes": [
                {
                    "theme_id": "industry::ai",
                    "name": "AI Industry",
                    "mandate": "tactical",
                    "tradable_node": True,
                    "supply_chain_roles": [],
                }
            ],
        }
    )
    result = evaluate_theme_protocol_v2(
        theme_scores={"industry::ai": _score("industry::ai")},
        taxonomy=taxonomy,
        as_of="2026-07-10",
        evidence_events=[_event("industry::ai", "industry-event")],
        valid_membership_theme_ids=["industry::ai"],
        theme_membership_details=[
            {
                "schema_version": "industry_map.v1",
                "theme_id": "industry::ai",
                "symbol": "S001.SZ",
                "pit_membership": False,
            }
        ],
        previous_states={"industry::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )

    state = result["states"]["industry::ai"]
    assert result["pit_membership_status"] == "coverage_blocked"
    assert result["prequalified_pool"] == []
    assert "pit_membership_missing" in state["prequalification_blockers"]
    assert state["lane"] == "market_observation"


def test_future_available_v2_membership_cannot_prequalify() -> None:
    details = _membership_details("tech::ai")
    details[0]["available_at"] = "2026-07-11"
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event")],
        theme_membership_details=details,
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )

    assert result["prequalified_pool"] == []
    assert "pit_membership_missing" in result["states"]["tech::ai"][
        "prequalification_blockers"
    ]


def test_latest_membership_tombstone_immediately_breaks_theme() -> None:
    active_revision = _membership_details("tech::ai")[0]
    active_revision["membership_id"] = "ai-active-v1"
    active_revision["updated_at"] = "2026-06-01T00:00:00Z"
    tombstone = {
        **active_revision,
        "membership_id": "ai-invalidated-v2",
        "membership_status": "invalidated",
        "available_at": "2026-07-01",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-ai")],
        theme_membership_details=[active_revision, tombstone],
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
        formal_kill_switch=False,
        valid_trading_dates=["2026-07-10"],
    )

    state = result["states"]["tech::ai"]
    assert result["pit_membership_count"] == 0
    assert result["pit_membership_invalidated_theme_ids"] == ["tech::ai"]
    assert state["lifecycle"] == ThemeLifecycle.BROKEN.value
    assert "pit_membership_missing" in state["prequalification_blockers"]


def test_missing_updated_at_uses_available_at_for_immediate_tombstone_kill() -> None:
    active_revision = _membership_details("tech::ai")[0]
    active_revision.update(
        {
            "membership_id": "zzz-active",
            "available_at": "2026-01-01",
            "updated_at": "",
        }
    )
    tombstone = {
        **active_revision,
        "membership_id": "aaa-tombstone",
        "membership_status": "invalidated",
        "available_at": "2026-07-01",
        "updated_at": "",
    }
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event-ai")],
        theme_membership_details=[active_revision, tombstone],
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
        formal_kill_switch=False,
        valid_trading_dates=["2026-07-10"],
    )

    state = result["states"]["tech::ai"]
    assert result["pit_membership_count"] == 0
    assert result["pit_membership_invalidated_theme_ids"] == ["tech::ai"]
    assert state["lifecycle"] == ThemeLifecycle.BROKEN.value


def test_unknown_crowding_and_valuation_axes_fail_prequalification_closed() -> None:
    score = _score("tech::ai")
    score.pop("crowding_status")
    score.pop("valuation_risk_status")
    result = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": score},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event")],
        theme_membership_details=_membership_details("tech::ai"),
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )

    state = result["states"]["tech::ai"]
    assert state["crowding"] is None
    assert state["valuation_risk"] is None
    assert "crowding_unavailable" in state["prequalification_blockers"]
    assert "valuation_risk_unavailable" in state["prequalification_blockers"]
    assert result["prequalified_pool"] == []


def test_post_control_reconciliation_is_only_formal_pool_producer_and_is_immutable(
    tmp_path: Path,
) -> None:
    details = _membership_details("tech::ai")
    prequalification = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event")],
        theme_membership_details=details,
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )
    assert prequalification["formal_pool"] == []
    assert prequalification["prequalified_pool"] == ["tech::ai"]

    outcomes = {
        "S001.SZ": {
            "data_pass": True,
            "tradability_pass": True,
            "liquidity_pass": True,
            "positive_edge_or_buy": True,
            "risk_guard_pass": True,
            "portfolio_constructor_pass": True,
            "decision_id": "decision-1",
            "portfolio_weight": 0.05,
        }
    }
    artifact = reconcile_theme_protocol_v2(
        prequalification=prequalification,
        symbol_membership_details={"S001.SZ": details},
        symbol_outcomes=outcomes,
        as_of="2026-07-10",
        expected_protocol_hash=prequalification["protocol_hash"],
        run_id="run-1",
    )

    assert artifact["source_stage"] == "post_control_chain"
    assert artifact["only_formal_producer"] is True
    assert artifact["formal_pool"] == ["tech::ai"]
    assert artifact["formal_symbols"] == ["S001.SZ"]
    target = tmp_path / "reconciliation.json"
    write_theme_formal_reconciliation_artifact(target, artifact)
    with pytest.raises(FileExistsError):
        write_theme_formal_reconciliation_artifact(target, artifact)


def test_post_control_reconciliation_fails_closed_on_missing_buy_gate() -> None:
    details = _membership_details("tech::ai")
    prequalification = evaluate_theme_protocol_v2(
        theme_scores={"tech::ai": _score("tech::ai")},
        taxonomy=_taxonomy(),
        as_of="2026-07-10",
        evidence_events=[_event("tech::ai", "event")],
        theme_membership_details=details,
        previous_states={"tech::ai": {"lifecycle": "validated_trend"}},
        formal_enabled=True,
    )
    artifact = reconcile_theme_protocol_v2(
        prequalification=prequalification,
        symbol_membership_details={"S001.SZ": details},
        symbol_outcomes={
            "S001.SZ": {
                "data_pass": True,
                "tradability_pass": True,
                "liquidity_pass": True,
                "positive_edge_or_buy": False,
                "risk_guard_pass": True,
                "portfolio_constructor_pass": True,
            }
        },
        as_of="2026-07-10",
        expected_protocol_hash=prequalification["protocol_hash"],
    )

    assert artifact["status"] == "valid_empty"
    assert artifact["formal_pool"] == []
    assert "positive_edge_buy_gate_blocked" in artifact["per_symbol"]["S001.SZ"][
        "blockers"
    ]


def test_post_control_reconciliation_rejects_non_tech_nav_cap_breach() -> None:
    theme_id = "tactical::gold"
    taxonomy = ThemeTaxonomy.from_mapping(
        {
            "schema_version": "theme_taxonomy.v2",
            "taxonomy_id": "tactical-test",
            "version": "1",
            "nodes": [
                {
                    "theme_id": theme_id,
                    "name": "Gold",
                    "mandate": "tactical",
                    "tradable_node": True,
                    "supply_chain_roles": [],
                }
            ],
        }
    )
    details = _membership_details(theme_id)
    prequalification = evaluate_theme_protocol_v2(
        theme_scores={theme_id: _score(theme_id)},
        taxonomy=taxonomy,
        as_of="2026-07-10",
        evidence_events=[_event(theme_id, "event")],
        theme_membership_details=details,
        previous_states={theme_id: {"lifecycle": "validated_trend"}},
        markov_regime="震荡低波",
        formal_enabled=True,
    )
    artifact = reconcile_theme_protocol_v2(
        prequalification=prequalification,
        symbol_membership_details={"S001.SZ": details},
        symbol_outcomes={
            "S001.SZ": {
                "data_pass": True,
                "tradability_pass": True,
                "liquidity_pass": True,
                "positive_edge_or_buy": True,
                "risk_guard_pass": True,
                "portfolio_constructor_pass": True,
                "portfolio_weight": 0.12,
            }
        },
        as_of="2026-07-10",
        expected_protocol_hash=prequalification["protocol_hash"],
    )

    assert artifact["formal_pool"] == []
    assert artifact["tactical_non_tech_nav_used"] == pytest.approx(0.12)
    assert artifact["tactical_non_tech_nav_cap"] == pytest.approx(0.10)
    assert "tactical_nav_cap_exceeded" in artifact["per_symbol"]["S001.SZ"][
        "blockers"
    ]


def test_lifecycle_counts_distinct_dates_and_hard_kill_is_immediate() -> None:
    config = ThemeProtocolConfig(upward_confirmation_days=3, cooling_confirmation_days=2)
    valid_dates = ("2026-07-01", "2026-07-02", "2026-07-03")
    first = transition_theme_lifecycle(
        {},
        desired="warming",
        as_of="2026-07-01",
        confirmation_date="2026-07-01",
        valid_trading_dates=valid_dates,
        config=config,
    )
    duplicate = transition_theme_lifecycle(
        first,
        desired="warming",
        as_of="2026-07-01",
        confirmation_date="2026-07-01",
        valid_trading_dates=valid_dates,
        config=config,
    )
    second = transition_theme_lifecycle(
        duplicate,
        desired="warming",
        as_of="2026-07-02",
        confirmation_date="2026-07-02",
        valid_trading_dates=valid_dates,
        config=config,
    )
    third = transition_theme_lifecycle(
        second,
        desired="warming",
        as_of="2026-07-03",
        confirmation_date="2026-07-03",
        valid_trading_dates=valid_dates,
        config=config,
    )
    killed = transition_theme_lifecycle(
        third, desired="broken", as_of="2026-07-03", hard_kill=True, config=config
    )

    assert first["pending_confirmation_dates"] == ["2026-07-01"]
    assert duplicate["pending_confirmation_dates"] == ["2026-07-01"]
    assert second["lifecycle"] == "discovery"
    assert third["lifecycle"] == "warming"
    assert killed["lifecycle"] == ThemeLifecycle.BROKEN.value


def test_lifecycle_ignores_weekend_and_future_confirmation_dates() -> None:
    pending = transition_theme_lifecycle(
        {
            "lifecycle": "discovery",
            "pending_transition": "warming",
            "pending_confirmation_dates": [
                "2026-07-03",
                "2026-07-04",
                "2026-07-07",
            ],
        },
        desired="warming",
        as_of="2026-07-06",
        confirmation_date="2026-07-06",
        valid_trading_dates=(
            "2026-07-03",
            "2026-07-04",
            "2026-07-06",
            "2026-07-07",
        ),
        config=ThemeProtocolConfig(upward_confirmation_days=3),
    )

    assert pending["lifecycle"] == "discovery"
    assert pending["pending_confirmation_dates"] == [
        "2026-07-03",
        "2026-07-06",
    ]


def test_non_tech_tactical_caps_match_all_markov_states() -> None:
    assert tactical_lane_cap("趋势上涨") == {
        "regime": "趋势上涨",
        "non_tech_nav_cap": 0.15,
        "non_tech_max_positions": 2,
        "enabled": True,
    }
    assert tactical_lane_cap("震荡低波")["non_tech_nav_cap"] == 0.10
    assert tactical_lane_cap("震荡高波")["non_tech_nav_cap"] == 0.05
    assert tactical_lane_cap("趋势下跌")["enabled"] is False


@pytest.mark.parametrize(
    ("regime", "expected_count"),
    [
        ("趋势上涨", 2),
        ("震荡低波", 1),
        ("震荡高波", 1),
        ("趋势下跌", 0),
    ],
)
def test_protocol_enforces_non_tech_tactical_position_cap_in_prequalified_pool(
    regime: str,
    expected_count: int,
) -> None:
    theme_ids = ("tactical::a", "tactical::b", "tactical::c")
    taxonomy = ThemeTaxonomy.from_mapping(
        {
            "schema_version": "theme_taxonomy.v2",
            "taxonomy_id": "tactical-test",
            "version": "1",
            "nodes": [
                {
                    "theme_id": theme_id,
                    "name": theme_id,
                    "mandate": "tactical",
                    "tradable_node": True,
                    "supply_chain_roles": [],
                }
                for theme_id in theme_ids
            ],
        }
    )
    result = evaluate_theme_protocol_v2(
        theme_scores={theme_id: _score(theme_id) for theme_id in theme_ids},
        taxonomy=taxonomy,
        as_of="2026-07-10",
        evidence_events=[
            _event(theme_id, f"event-{theme_id}")
            for theme_id in theme_ids
        ],
        valid_membership_theme_ids=theme_ids,
        theme_membership_details=_membership_details(*theme_ids),
        previous_states={
            theme_id: {"lifecycle": "validated_trend"}
            for theme_id in theme_ids
        },
        downstream_gates=_gates(*theme_ids),
        markov_regime=regime,
        formal_enabled=True,
    )

    assert len(result["prequalified_pool"]) == expected_count
    assert result["formal_pool"] == []
    excluded = set(theme_ids) - set(result["prequalified_pool"])
    expected_blocker = (
        "tactical_lane_closed_by_markov"
        if expected_count == 0
        else "tactical_position_cap_exceeded"
    )
    for theme_id in excluded:
        assert expected_blocker in result["states"][theme_id]["eligibility_blockers"]
