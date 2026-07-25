from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = ROOT / "quant_investor" / "v17_v2_contract" / "resources"
RESOURCE_NAMES = (
    "deep_research_template.v1.json",
    "quant_factor_set.v1.json",
    "shadow_policy.v1.json",
)
PROTOCOL_VERSION = "myquant.v17.v2"
DEEP_SOURCE_SHA256 = "7dbda6970ea058aa656f1e7c12a0be24feb5ea7ce99d3e33e885dcc84f438e0e"
FACTOR_DEFINITION_SHA256S = (
    "a412e07a6c5df48f250577d821209670ffce36794c61b514c8eb8b3b412a499d",
    "17863a088f0bebaaf68ea137f13ab3fb7576bb24514f6c87735838b9643467df",
    "8bb6170626d59c45aa2d21171e1660ad06fdc5efc322006869bac529e79b4152",
)
RESOURCE_SHA256S = {
    "deep_research_template.v1.json": (
        "15143aa020f7e78a5ac6772ab5c1125caf989bb7a2a0e9c072511f7c68871c85"
    ),
    "quant_factor_set.v1.json": (
        "8c46282029fe07f677a1a3a7efe1241a0a2b2b817f2e6db51082a84bc0b09c3d"
    ),
    "shadow_policy.v1.json": ("9a1b16d1ea3f131a3842bdde2955615c3c14563cad262641d0f608542b2b662a"),
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(name: str) -> tuple[bytes, dict[str, Any]]:
    raw = (RESOURCE_ROOT / name).read_bytes()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            AssertionError(f"non-finite JSON value: {value}")
        ),
    )
    assert isinstance(payload, dict)
    return raw, payload


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _array_paths(value: Any, *, path: str = "") -> set[str]:
    if isinstance(value, list):
        return {path}
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            if key == "array_order_semantics":
                continue
            result.update(_array_paths(child, path=f"{path}/{key}"))
        return result
    return set()


@pytest.mark.parametrize("name", RESOURCE_NAMES)
def test_policy_resource_identity_hash_and_canonical_bytes(name: str) -> None:
    raw, payload = _load(name)
    stem = {
        "deep_research_template.v1.json": "deep-research-template",
        "quant_factor_set.v1.json": "quant-factor-set",
        "shadow_policy.v1.json": "shadow-policy",
    }[name]
    expected_version = f"myquant.v17.v2.{stem}.v1"

    assert payload["artifact_id"] == expected_version
    assert payload["version"] == expected_version
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["authority"] is False
    assert raw == _canonical_bytes(payload)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == RESOURCE_SHA256S[name]

    ordering = payload["array_order_semantics"]
    assert isinstance(ordering, dict)
    assert set(ordering) == _array_paths(payload)
    assert all(isinstance(item, str) and item for item in ordering.values())


def test_shadow_policy_freezes_scoring_eligibility_overlay_and_permissions() -> None:
    _, policy = _load("shadow_policy.v1.json")
    scoring = policy["fundamental_scoring"]
    eligibility = policy["fundamental_eligibility"]
    calibration = policy["forward_calibration"]
    overlay = policy["portfolio_overlay"]
    optimizer = policy["optimizer"]
    permissions = policy["permission_policy"]
    pretrade = policy["pretrade_checks"]
    transaction_cost = policy["transaction_cost"]
    authority_requirements = policy["authority_requirements"]
    holdings = policy["holdings_availability"]
    trade_truth_table = policy["trade_truth_table"]

    assert scoring["top_n"] == 24
    assert scoring["holdings_append_without_consuming_top_n"] is True
    assert scoring["failed_deep_research_may_backfill"] is False
    assert (
        scoring["fcf_formula"] == "free_cash_flow=CFO-capex; capex uses POSITIVE_OUTFLOW convention"
    )
    assert scoring["flow_statement_basis"] == "PIT_LATEST_TTM"
    assert scoring["flow_basis"] == "LATEST_TTM"
    assert scoring["balance_sheet_basis"] == "PIT_LATEST_REPORT_PERIOD"
    assert (
        scoring["fcf_to_price_denominator"] == "market_cap_AVAILABLE_and_strictly_greater_than_zero"
    )
    assert scoring["positive_denominator_requirements"] == {
        "fcf_to_price": "market_cap_AVAILABLE_and_strictly_greater_than_zero",
        "fin_fcf_to_profit": "parent_net_profit_TTM_AVAILABLE_and_strictly_greater_than_zero",
        "fin_ocf_to_profit": "parent_net_profit_TTM_AVAILABLE_and_strictly_greater_than_zero",
    }
    assert scoring["industry_winsor_quantiles"] == [0.01, 0.99]
    assert scoring["industry_percentile_weight"] == 0.7
    assert scoring["self_percentile_weight"] == 0.3
    assert (
        scoring["industry_self_blend_formula"]
        == "score=0.70*industry_percentile_after_1_99_winsor+0.30*self_percentile"
    )
    assert scoring["minimum_industry_sample"] == 20
    assert scoring["self_history_open_days"] == 756
    assert scoring["minimum_self_history_open_days"] == 252
    assert scoring["metric_unavailable_criteria"] == [
        "required_raw_input_missing",
        "denominator_missing_or_non_positive",
        "PIT_timestamp_after_cutoff",
        "history_trade_date_after_cutoff",
        "industry_unknown",
        "industry_sample_below_minimum",
        "self_history_below_minimum",
        "unknown_or_conflicting_membership",
    ]
    assert scoring["top24_selection_order"] == "total_score_descending_then_security_code_ascending"
    assert scoring["main_metrics"] == [
        "fin_roe",
        "fin_ocf_to_profit",
        "fin_net_profit_yoy",
        "fin_debt_to_assets",
        "fcf_to_price",
    ]
    assert scoring["optional_metrics"] == [
        "fin_roa",
        "fin_fcf_to_profit",
        "forecast_revision",
    ]
    assert scoring["pillar_order"] == [
        "profitability",
        "cash_conversion",
        "growth_expectations",
        "balance_sheet_resilience",
        "valuation",
    ]
    assert math.isclose(
        sum(scoring["pillars"][name]["weight"] for name in scoring["pillar_order"]),
        1.0,
    )

    assert calibration["target"] == "horizon_excess_total_return_q25_q50_q75"
    assert (
        calibration["excess_total_return_definition"]
        == "stock_forward_total_return_over_exact_trade_dates_minus_benchmark_forward_total_return_over_same_trade_dates; dividends_reinvested_when_official_adjusted_total_return_available_else_cell_unavailable"
    )
    assert calibration["reported_quantiles"] == ["q25", "q50", "q75"]

    assert eligibility == {
        "base_q25_requirement": (
            "AVAILABLE_and_strictly_greater_than_zero_for_all_required_horizons_before_optimizer_adjustment"
        ),
        "deep_research_complete_required": True,
        "fail_closed": True,
        "optimizer_adjusted_q25_requirement": "optimizer_uses_adjusted_q25_for_252_open_day_horizon_only",
        "optimizer_q25_horizon_open_days": 252,
        "required_horizons_open_days": [120, 252, 378],
        "severe_red_flag_required_value": False,
        "status_eligible": "F_ELIGIBLE",
        "status_ineligible": "F_INELIGIBLE",
        "strict_boolean_controls": True,
        "truth_table": [
            {
                "base_q25_all_required_horizons_positive": True,
                "deep_research_complete": True,
                "fundamental_eligibility": "F_ELIGIBLE",
                "severe_red_flag": False,
            },
            {
                "base_q25_all_required_horizons_positive": False,
                "deep_research_complete": True,
                "fundamental_eligibility": "F_INELIGIBLE",
                "severe_red_flag": False,
            },
            {
                "base_q25_all_required_horizons_positive": True,
                "deep_research_complete": False,
                "fundamental_eligibility": "F_INELIGIBLE",
                "severe_red_flag": False,
            },
            {
                "base_q25_all_required_horizons_positive": True,
                "deep_research_complete": True,
                "fundamental_eligibility": "F_INELIGIBLE",
                "severe_red_flag": True,
            },
        ],
    }
    assert "severe_red_flags_required" not in eligibility

    assert holdings == {
        "AVAILABLE": "snapshot contains identity plus positive NAV OR explicit all-cash declaration",
        "UNAVAILABLE": (
            "snapshot contains identity+reason only; NAV and positions are prohibited; must not be "
            "interpreted as empty portfolio"
        ),
        "snapshot_shape": {
            "AVAILABLE": "positive NAV OR explicit all-cash declaration",
            "UNAVAILABLE": "identity+reason only; NAV/positions prohibited; not empty portfolio",
        },
    }

    assert authority_requirements == {
        "defaults_prohibited": True,
        "missing_input_behavior": "UNAVAILABLE_or_terminal_fail_closed",
        "prohibited_default_inputs": ["risk", "holdings", "macro", "markov", "cost"],
    }
    assert "required_to_authorize_execution" not in authority_requirements
    assert "zero_defaults_are_authoritative" not in authority_requirements

    assert pretrade == {
        "ordered_checks": [
            "tradability",
            "ADV20",
            "industry",
            "beta",
            "cluster",
            "stress",
            "cost",
        ],
        "restrictions_may_only_shrink": True,
        "unavailable_check_behavior": "cancel_affected_buy_or_sell_fail_closed",
    }

    assert transaction_cost == {
        "buy_applied_components": [
            "buy_commission",
            "buy_transfer_fee",
            "buy_slippage",
            "market_impact",
        ],
        "global_parameter_order": [
            "buy_commission",
            "sell_commission",
            "sell_stamp_tax",
            "buy_transfer_fee",
            "sell_transfer_fee",
            "buy_slippage",
            "sell_slippage",
            "market_impact",
        ],
        "market_impact_formula": (
            "impact_bps=market_impact_coefficient*sqrt(abs(traded_notional)/ADV20_notional); "
            "unavailable_ADV20_makes_trade_UNAVAILABLE"
        ),
        "sell_applied_components": [
            "sell_commission",
            "sell_stamp_tax",
            "sell_transfer_fee",
            "sell_slippage",
            "market_impact",
        ],
        "unit": "fraction_of_traded_notional",
    }

    assert overlay["component_order"] == ["base", "macro", "markov"]
    assert overlay["gross_cap_aggregation"] == "minimum_of_enabled_AVAILABLE_components"
    assert overlay["cash_floor_aggregation"] == "maximum_of_enabled_AVAILABLE_components"
    assert overlay["effective_gross"] == "min(gross_cap,1-cash_floor)"
    assert (
        overlay["formula"]
        == "gross_cap=min(enabled_AVAILABLE_component_gross_caps); cash_floor=max(enabled_AVAILABLE_component_cash_floors); effective_gross=min(gross_cap,1-cash_floor)"
    )
    assert overlay["gross_cap_min"] == 0
    assert overlay["gross_cap_max"] == 1
    assert overlay["cash_floor_min"] == 0
    assert overlay["cash_floor_max"] == 1
    assert overlay["may_reorder_or_select_security"] is False
    assert overlay["unavailable_terminal"] == "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"

    assert optimizer["objective_order"] == [
        "maximize_adjusted_252d_q25_net_of_cost",
        "minimize_turnover",
        "ascending_security_code",
    ]
    assert (
        optimizer["lexicographic_objective"]
        == "first maximize adjusted_252d_q25_net_of_cost, then minimize turnover, then choose ascending security_code"
    )
    assert optimizer["may_create_permission"] is False
    assert (
        optimizer["risk_and_optimizer_restriction_authority"]
        == "may_only_cancel_buy_sell_or_reduce_target_weight; may_not_create_permission_or_expand_trade_set"
    )

    assert permissions["restriction_order"] == ["risk", "optimizer"]
    assert permissions["restrictions_may_only_shrink"] is True
    assert permissions["risk_and_optimizer_may_only_cancel"] is True
    assert [rule["basis"] for rule in permissions["rules"]] == [
        "untradable_absolute_block",
        "new_position_buy_permitted",
        "new_position_not_permitted",
        "held_trim_sell_only",
        "held_watch_locked",
        "held_add_permitted",
        "held_buy_now_fundamental_lock",
    ]
    assert permissions["rules"][0]["can_buy"] is False
    assert permissions["rules"][0]["can_sell"] is False
    assert permissions["rules"][3]["can_sell"] is True
    assert permissions["rules"][5]["can_buy"] is True

    assert trade_truth_table["risk_and_optimizer_may_only_cancel"] is True
    assert trade_truth_table["matching"] == "first_match_exhaustive_order"
    assert [row["base_permission"] for row in trade_truth_table["rows"]] == [
        "NO_BUY_NO_SELL_POSITION_LOCKED",
        "NO_BUY_NO_SELL",
        "CAN_BUY_ONLY",
        "NO_BUY_NO_SELL",
        "NO_BUY_NO_SELL",
        "NO_BUY_NO_SELL",
        "NO_BUY_NO_SELL",
        "NO_BUY_NO_SELL",
        "NO_BUY_NO_SELL_POSITION_LOCKED",
        "CAN_SELL_ONLY",
        "NO_BUY_NO_SELL_POSITION_LOCKED",
        "CAN_BUY_ONLY",
        "NO_BUY_NO_SELL_POSITION_LOCKED",
        "NO_BUY_NO_SELL_POSITION_LOCKED",
        "NO_BUY_NO_SELL_POSITION_LOCKED",
    ]
    assert any(
        row["fundamental_eligibility"] == "F_INELIGIBLE" and row["severe_red_flag"] is True
        for row in trade_truth_table["rows"]
    )
    assert any(
        row["quant_timing"] == "UNREADY" and row["held"] is False
        for row in trade_truth_table["rows"]
    )
    assert any(
        row["quant_timing"] == "UNREADY" and row["held"] is True
        for row in trade_truth_table["rows"]
    )


def test_trade_truth_table_is_exhaustive_and_nonoverlapping() -> None:
    _, policy = _load("shadow_policy.v1.json")
    rows = policy["trade_truth_table"]["rows"]
    timings = ("BUY_NOW", "TRIM_TIMING", "WATCH", "UNREADY")
    eligibility_states = ("F_ELIGIBLE", "F_INELIGIBLE")

    for tradable in (False, True):
        for held in (False, True):
            for timing in timings:
                for eligibility in eligibility_states:
                    for red_flag in (False, True):
                        facts = {
                            "tradable": tradable,
                            "held": held,
                            "quant_timing": timing,
                            "fundamental_eligibility": eligibility,
                            "severe_red_flag": red_flag,
                        }
                        matches = [
                            row
                            for row in rows
                            if all(
                                row[field] == "ANY" or row[field] == value
                                for field, value in facts.items()
                            )
                        ]
                        assert len(matches) == 1, facts
                        permission = matches[0]["base_permission"]
                        if not tradable:
                            assert permission == "NO_BUY_NO_SELL_POSITION_LOCKED"
                        elif not held:
                            expected = (
                                "CAN_BUY_ONLY"
                                if timing == "BUY_NOW"
                                and eligibility == "F_ELIGIBLE"
                                and not red_flag
                                else "NO_BUY_NO_SELL"
                            )
                            assert permission == expected
                        elif timing == "TRIM_TIMING":
                            assert permission == "CAN_SELL_ONLY"
                        elif timing == "BUY_NOW" and eligibility == "F_ELIGIBLE" and not red_flag:
                            assert permission == "CAN_BUY_ONLY"
                        else:
                            assert permission == "NO_BUY_NO_SELL_POSITION_LOCKED"


def test_deep_research_template_binds_source_scope_and_adjustment() -> None:
    _, template = _load("deep_research_template.v1.json")

    assert template["source"]["sha256"] == DEEP_SOURCE_SHA256
    assert template["scope_rules"] == {
        "evidence": "sealed_evidence_only",
        "failed_deep_research_may_backfill": False,
        "security_set": "sealed_top24_plus_holdings_only",
    }
    assert template["signal_order"] == [
        "financial",
        "business_model",
        "industry",
        "competitiveness",
        "management",
        "valuation",
    ]
    assert math.isclose(sum(template["signals"].values()), 1.0)
    assert template["allowed_signal_values"] == [-1, -0.5, 0, 0.5, 1]
    assert template["signal_adjustment"] == {
        "clamp_lower": -0.1,
        "clamp_upper": 0.1,
        "delta_formula": "clamp(0.10*sum(weight*signal),-0.10,0.10)",
        "result_formula": "adjusted_q25_252=base_q25_252*(1+delta)",
        "target_horizon_open_days": 252,
    }
    assert len(template["coverage"]) == 16
    assert len(template["severe_red_flags"]) == 10


def test_quant_policy_freezes_source_factor_definitions_and_timing_states() -> None:
    _, policy = _load("quant_factor_set.v1.json")
    source = ROOT / policy["implementation_source"]

    assert hashlib.sha256(source.read_bytes()).hexdigest() == policy["implementation_source_sha256"]
    assert tuple(item["definition_sha256"] for item in policy["factors"]) == (
        FACTOR_DEFINITION_SHA256S
    )
    assert [item["name"] for item in policy["factors"]] == [
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "pv_short_reversal_25d",
        "pv_downside_volatility_15d",
    ]
    assert math.isclose(sum(item["weight"] for item in policy["factors"]), 1.0)
    assert policy["calibration"] == {
        "deciles": 10,
        "horizons_open_days": [20, 60],
        "lookback_open_days": 1260,
        "minimum_cross_section_dates_per_decile": 24,
        "minimum_observations_per_decile": 200,
        "monotonic_fit": "weighted_deterministic_pava_non_decreasing",
        "pit_mature_samples_only": True,
        "probability": "(wins+0.5)/(n+1)",
        "target": "horizon_excess_return_gt_zero",
    }
    assert policy["states"] == {
        "BUY_NOW": "p20>=0.60 and p60>=0.60",
        "TRIM_TIMING": "p20<=0.40 and p60<=0.40",
        "UNREADY": "any factor or calibration requirement unavailable",
        "WATCH": "otherwise",
    }


@pytest.mark.parametrize("name", RESOURCE_NAMES)
def test_policy_resources_are_self_contained_and_non_authorizing(name: str) -> None:
    raw, payload = _load(name)
    boundaries = payload["authority_boundaries"]

    assert b"quant_investor/v17/resources" not in raw
    assert b'"$ref"' not in raw
    assert boundaries["advisory_only"] is True
    for key, value in boundaries.items():
        if key.startswith("may_") and key != "may_revoke_buy_for_severe_red_flag":
            assert value is False

    if name == "shadow_policy.v1.json":
        assert payload["mode"] == "research_shadow_only"
        assert payload["production_protocol"] == "v15"
        assert all(value is False for value in payload["side_effects"].values())
    elif name == "deep_research_template.v1.json":
        assert boundaries["may_revoke_buy_for_severe_red_flag"] is True
        assert boundaries["may_create_sell_instruction"] is False
    else:
        assert boundaries["timing_only_within_sealed_security_set"] is True
