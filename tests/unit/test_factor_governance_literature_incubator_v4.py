from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_candidate_preregistration_v4_4 as exact_five
from quant_investor.factors import governance_literature_incubator_v4 as subject
from quant_investor.factors import governance_screening_v4 as screening


def _axes(rows: int = 280, columns: int = 10) -> tuple[pd.DatetimeIndex, pd.Index]:
    dates = pd.bdate_range("2025-01-02", periods=rows, name="trade_date")
    symbols = pd.Index(
        [f"{index:06d}.SZ" for index in range(1, columns + 1)],
        name="ts_code",
    )
    return dates, symbols


def _frame(
    values: np.ndarray | list[list[float]],
    *,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates, columns=symbols, dtype=float)


def test_catalog_is_source_bound_unique_and_non_authorizing() -> None:
    ontology = subject.candidate_ontology_v4()
    catalog_artifact = subject.candidate_catalog_artifact_v4()
    assert screening.validate_primitive_ontology_v4(ontology) == ontology
    assert (
        screening.validate_candidate_catalog_v4(
            catalog_artifact,
            ontology=ontology,
        )
        == catalog_artifact
    )

    catalog = subject.candidate_catalog_v4()
    assert [row["order"] for row in catalog] == list(range(1, 12))
    assert len({row["name"] for row in catalog}) == 11
    assert len({row["family"] for row in catalog}) == 11
    assert len({row["slot"] for row in catalog}) == 11
    assert len({row["definition_identity_sha256"] for row in catalog}) == 11
    assert all(len(row["definition_identity_sha256"]) == 64 for row in catalog)
    assert all(not any(row["authority"].values()) for row in catalog)
    assert all(not any(row["side_effects"].values()) for row in catalog)

    sources = subject.literature_idea_catalog_v4()
    assert len(sources) == 26
    assert {row["source_id"] for row in sources if row["local_data_status"] == "IMPLEMENTABLE"} == {
        "gao_han_xiong_2021",
        "liu_stambaugh_yuan_2019",
        "frazzini_pedersen_2014",
        "meng_du_shu_2024",
        "wang_wang_wu_2023",
        "zhen_ruan_zhang_2020",
    }
    assert {
        row["source_id"]
        for row in sources
        if row["local_data_status"] == "IMPLEMENTABLE_CAUSAL_ROLLING_TRANSLATION"
    } == {"qian_sun_yu_2017"}
    assert {
        row["source_id"]
        for row in sources
        if row["local_data_status"] == "IMPLEMENTABLE_SIGNAL_BLOCKED_PIT_EPU_REGIME"
    } == {"zhou_liu_guo_2021"}
    assert {row["idea"] for row in sources if row["local_data_status"].startswith("BLOCKED_")} == {
        "china_beta_anomaly_conditional",
        "gross_profitability",
        "investment_and_profitability",
        "china_residual_momentum",
        "china_idiosyncratic_momentum_pls",
        "china_factor_momentum",
        "china_change_in_salience",
    }
    assessments = {
        row["candidate_name"]: row for row in subject.candidate_literature_assessments_v4()
    }
    assert set(assessments) == {row["name"] for row in catalog}
    assert assessments["cn_low_beta_252d"]["future_preregistration_eligible"] is False
    assert assessments["cn_52_week_high_momentum_12m"]["future_preregistration_eligible"] is True
    assert assessments["cn_high_price_delay_d1_52w"]["future_preregistration_eligible"] is True
    assert assessments["cn_low_max_return_20d"]["future_preregistration_eligible"] is True
    assert assessments["cn_low_total_skewness_20d"]["future_preregistration_eligible"] is True
    assert assessments["cn_fip_continuous_direction_12m"]["future_preregistration_eligible"] is True
    assert assessments["cn_low_left_tail_var1_250d"]["future_preregistration_eligible"] is True
    assert assessments["cn_low_left_tail_var1_250d"]["status"] == (
        "DIRECT_CHINA_SUPPORT_EXACT_VAR1_SIGNAL"
    )
    assert assessments["cn_fip_continuous_direction_12m"]["status"] == (
        "FOUNDATIONAL_FIP_WITH_AGGREGATE_CHINA_RELEVANCE"
    )
    assert (
        assessments["cn_low_market_adjusted_tail_asymmetry_252d"]["future_preregistration_eligible"]
        is True
    )
    assert assessments["cn_low_beta_252d"]["adverse_source_ids"] == [
        "blitz_hanauer_van_vliet_2021",
        "zhao_lin_2022",
    ]

    policy = subject.low_max_future_preregistration_policy_v4()
    assert policy["status"] == "DRAFT_FUTURE_PREREGISTRATION_POLICY"
    assert policy["formal_preregistration_created"] is False
    assert policy["candidate"]["name"] == "cn_low_max_return_20d"
    assert policy["candidate"]["initial_weight"] == 0
    assert policy["candidate"]["definition_identity_sha256"] == next(
        row["definition_identity_sha256"]
        for row in catalog
        if row["name"] == "cn_low_max_return_20d"
    )
    assert policy["future_sample_contract"] == {
        "universe": "strict_parquet_pit_full_a_after_canonical_eligibility",
        "embargo_strictly_later_open_sessions": 30,
        "measurement_starts_on_later_open_session": 31,
        "minimum_post_embargo_open_sessions": 240,
        "minimum_distinct_closed_month_ends": 12,
        "publication_day_excluded": True,
        "pre_publication_observations_are_diagnostic_only": True,
    }
    assert policy["multiple_testing_contract"]["maximum_q_value"] == 0.10
    assert policy["dedup_contract"]["maximum_allowed"] == 0.70
    assert policy["dedup_contract"]["required_incubator_candidate_names"] == [
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_total_skewness_20d",
    ]
    assert policy["dedup_contract"]["required_protected_candidate_names"] == list(
        subject.PROTECTED_EXACT_FIVE_CANDIDATE_NAMES
    )
    assert not any(policy["authority"].values())
    assert not any(policy["side_effects"].values())
    assert len(policy["semantic_sha256"]) == 64

    skew_policy = subject.low_total_skewness_future_preregistration_policy_v4()
    assert skew_policy["status"] == "DRAFT_FUTURE_PREREGISTRATION_POLICY"
    assert skew_policy["formal_preregistration_created"] is False
    assert skew_policy["candidate"]["name"] == "cn_low_total_skewness_20d"
    assert skew_policy["candidate"]["initial_weight"] == 0
    assert (
        skew_policy["selection_provenance"]["conventional_total_skewness_definition_acknowledged"]
        is True
    )
    assert (
        skew_policy["selection_provenance"]["distributional_tail_asymmetry_replication_claimed"]
        is False
    )
    assert skew_policy["dedup_contract"]["required_incubator_candidate_names"] == [
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_max_return_20d",
    ]
    assert skew_policy["dedup_contract"]["required_protected_candidate_names"] == list(
        subject.PROTECTED_EXACT_FIVE_CANDIDATE_NAMES
    )
    assert (
        skew_policy["anchor_state_contract"]["required_interaction_direction"]
        == "low_skewness_spread_far_below_must_exceed_near_52_week_high"
    )
    assert skew_policy["multiple_testing_contract"]["maximum_q_value"] == 0.10
    assert not any(skew_policy["authority"].values())
    assert not any(skew_policy["side_effects"].values())
    assert len(skew_policy["semantic_sha256"]) == 64

    tail_policy = subject.tail_asymmetry_future_preregistration_policy_v4()
    assert tail_policy["status"] == "DRAFT_FUTURE_PREREGISTRATION_POLICY"
    assert tail_policy["formal_preregistration_created"] is False
    assert tail_policy["candidate"]["name"] == "cn_low_market_adjusted_tail_asymmetry_252d"
    assert tail_policy["candidate"]["initial_weight"] == 0
    assert tail_policy["selection_provenance"]["proxy_definition_acknowledged"] is True
    assert (
        tail_policy["selection_provenance"]["exact_CH3_or_CH4_idiosyncratic_IE_replication_claimed"]
        is False
    )
    assert tail_policy["dedup_contract"]["required_incubator_candidate_names"] == [
        "cn_low_max_return_20d",
        "cn_low_total_skewness_20d",
    ]
    assert tail_policy["dedup_contract"]["required_protected_candidate_names"] == list(
        subject.PROTECTED_EXACT_FIVE_CANDIDATE_NAMES
    )
    assert tail_policy["multiple_testing_contract"]["maximum_q_value"] == 0.10
    assert not any(tail_policy["authority"].values())
    assert not any(tail_policy["side_effects"].values())
    assert len(tail_policy["semantic_sha256"]) == 64

    fip_policy = subject.fip_future_preregistration_policy_v4()
    assert fip_policy["status"] == "DRAFT_FUTURE_PREREGISTRATION_POLICY"
    assert fip_policy["formal_preregistration_created"] is False
    assert fip_policy["candidate"]["name"] == "cn_fip_continuous_direction_12m"
    assert fip_policy["candidate"]["initial_weight"] == 0
    assert (
        fip_policy["selection_provenance"]["foundational_source_double_sort_acknowledged"] is True
    )
    assert (
        fip_policy["selection_provenance"][
            "China_aggregate_evidence_treated_as_cross_sectional_support"
        ]
        is False
    )
    assert (
        fip_policy["source_translation_contract"]["source_monthly_double_sort_replication_claimed"]
        is False
    )
    assert fip_policy["dedup_contract"]["required_incubator_candidate_names"] == [
        "cn_52_week_high_momentum_12m"
    ]
    assert fip_policy["dedup_contract"]["required_protected_candidate_names"] == list(
        subject.PROTECTED_EXACT_FIVE_CANDIDATE_NAMES
    )
    assert fip_policy["multiple_testing_contract"]["maximum_q_value"] == 0.10
    assert not any(fip_policy["authority"].values())
    assert not any(fip_policy["side_effects"].values())
    assert len(fip_policy["semantic_sha256"]) == 64

    var1_policy = subject.left_tail_var1_future_preregistration_policy_v4()
    assert var1_policy["status"] == "DRAFT_FUTURE_PREREGISTRATION_POLICY"
    assert var1_policy["formal_preregistration_created"] is False
    assert var1_policy["candidate"]["name"] == "cn_low_left_tail_var1_250d"
    assert var1_policy["candidate"]["initial_weight"] == 0
    assert var1_policy["source_translation_contract"]["VaR1_definition"] == (
        "negative_one_times_first_percentile_of_daily_returns"
    )
    assert var1_policy["source_translation_contract"]["primary_window_open_sessions"] == 250
    assert var1_policy["source_translation_contract"]["primary_minimum_observations"] == 200
    assert var1_policy["source_translation_contract"]["primary_quantile"] == 0.01
    assert var1_policy["dedup_contract"]["required_incubator_candidate_names"] == [
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_max_return_20d",
        "cn_low_total_skewness_20d",
    ]
    assert var1_policy["parameter_robustness_contract"]["secondary_risk_statistics"] == [
        "VaR5",
        "ES1",
    ]
    assert not any(var1_policy["authority"].values())
    assert not any(var1_policy["side_effects"].values())
    assert len(var1_policy["semantic_sha256"]) == 64


def test_protected_exact_five_audit_uses_frozen_identities_and_stops_slot_collision() -> None:
    audit = subject.build_protected_exact_five_audit_v4(
        protected_candidates=exact_five.EXPECTED_CANDIDATE_ROWS,
    )
    assert audit["protected_candidate_names"] == list(subject.PROTECTED_EXACT_FIVE_CANDIDATE_NAMES)
    assert audit["protected_candidate_count"] == 5
    assert all(
        row["protected_structural_collision_passed_diagnostic"]
        for row in audit["candidate_results"]
    )
    assert not any(audit["authority"].values())
    assert not any(audit["side_effects"].values())

    tampered = copy.deepcopy(list(exact_five.EXPECTED_CANDIDATE_ROWS))
    tampered[0]["slot"] = "primitive:total_return_skewness"
    tampered_audit = subject.build_protected_exact_five_audit_v4(
        protected_candidates=tampered,
    )
    skewness = next(
        row
        for row in tampered_audit["candidate_results"]
        if row["candidate_name"] == "cn_low_total_skewness_20d"
    )
    assert skewness["slot_collision_names"] == ["alpha_range_position_momentum_20d"]
    assert skewness["protected_structural_collision_passed_diagnostic"] is False


def test_structural_audit_routes_exact_primitive_and_overlap_collisions() -> None:
    ontology = screening.build_primitive_ontology_v4(
        [
            {"primitive_id": "earnings_yield", "family": "value"},
            {"primitive_id": "fin_roe", "family": "quality"},
            {"primitive_id": "volatility", "family": "volatility"},
        ]
    )
    catalog = screening.build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            {
                "name": "existing_earnings_yield",
                "implementation": "test:earnings_yield",
                "expression": "earnings_yield",
                "direction": 1.0,
                "params": {},
                "lookback": 1,
                "slot": "primitive:earnings_yield",
                "input_fields": ["pe"],
                "primitive_ids": ["earnings_yield"],
            },
            {
                "name": "existing_quality_volatility",
                "implementation": "test:quality_volatility",
                "expression": "fin_roe - volatility",
                "direction": 1.0,
                "params": {},
                "lookback": 60,
                "slot": "test:quality_volatility",
                "input_fields": ["adj_close", "fin_roe"],
                "primitive_ids": ["fin_roe", "volatility"],
            },
        ],
    )
    audit = subject.build_structural_audit_v4(
        comparison_ontology=ontology,
        comparison_catalog=catalog,
    )
    by_name = {row["candidate_name"]: row for row in audit["candidate_results"]}
    earnings = by_name["cn_earnings_yield_ex_shell_30pct"]
    assert earnings["exact_primitive_collision_names"] == ["existing_earnings_yield"]
    assert earnings["structural_collision_passed_diagnostic"] is False
    quality = by_name["cn_quality_cash_low_leverage"]
    assert quality["exact_primitive_collision_names"] == []
    assert quality["primitive_overlap_rows"] == [
        {
            "existing_factor_name": "existing_quality_volatility",
            "shared_primitive_ids": ["fin_roe"],
            "primitive_jaccard": 0.25,
        }
    ]
    assert quality["structural_collision_passed_diagnostic"] is True
    assert audit["formal_dedup_evidence"] is False
    assert not any(audit["authority"].values())


def test_structural_audit_rejects_tampered_comparison_catalog() -> None:
    ontology = screening.build_primitive_ontology_v4(
        [{"primitive_id": "volatility", "family": "volatility"}]
    )
    catalog = screening.build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            {
                "name": "existing_volatility",
                "implementation": "test:volatility",
                "expression": "-volatility",
                "direction": 1.0,
                "params": {},
                "lookback": 60,
                "slot": "primitive:volatility",
                "input_fields": ["adj_close"],
                "primitive_ids": ["volatility"],
            }
        ],
    )
    tampered = copy.deepcopy(catalog)
    tampered["candidates"][0]["lookback"] = 20
    with pytest.raises(
        screening.FactorGovernanceScreeningV4Error,
        match="definition SHA mismatch",
    ):
        subject.build_structural_audit_v4(
            comparison_ontology=ontology,
            comparison_catalog=tampered,
        )


def test_china_earnings_yield_excludes_smallest_30_percent_and_nonpositive_pe() -> None:
    dates, symbols = _axes(rows=2, columns=10)
    pit = _frame(np.ones((2, 10), dtype=bool), dates=dates, symbols=symbols).astype(bool)
    total_mv = _frame(
        np.tile(np.arange(1.0, 11.0), (2, 1)),
        dates=dates,
        symbols=symbols,
    )
    pe = _frame(
        np.tile(np.arange(10.0, 20.0), (2, 1)),
        dates=dates,
        symbols=symbols,
    )
    pe.iloc[1, 8] = 0.0
    signal = subject.earnings_yield_ex_shell_v4(pe=pe, total_mv=total_mv, pit_mask=pit)

    assert signal.iloc[:, :3].isna().all().all()
    assert signal.iloc[0, 3:].notna().all()
    assert signal.iloc[0, 3] > signal.iloc[0, 9]
    assert pd.isna(signal.iloc[1, 8])
    assert signal.where(~pit).isna().all().all()


def test_low_beta_is_causal_and_prefers_lower_realized_beta() -> None:
    dates, symbols = _axes(rows=300, columns=3)
    phase = np.arange(300, dtype=float)
    market_return = 0.0005 + 0.01 * np.sin(phase / 8.0)
    idiosyncratic = 0.001 * np.cos(phase / 5.0)
    returns = np.column_stack(
        [
            0.45 * market_return + idiosyncratic,
            1.00 * market_return - idiosyncratic,
            1.60 * market_return + 0.5 * idiosyncratic,
        ]
    )
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adj_close = _frame(close, dates=dates, symbols=symbols)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.low_beta_v4(adj_close=adj_close, pit_mask=pit)
    assert full.iloc[: subject.LOW_BETA_MIN_PERIODS].isna().all().all()
    assert full.iloc[-1, 0] > full.iloc[-1, 1] > full.iloc[-1, 2]

    prefix = subject.low_beta_v4(
        adj_close=adj_close.iloc[:270],
        pit_mask=pit.iloc[:270],
    )
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)


def test_price_delay_d1_is_causal_and_prefers_lagged_market_response() -> None:
    dates = pd.bdate_range("2024-01-01", periods=390, name="trade_date")
    wednesdays = dates[dates.weekday == subject.PRICE_DELAY_WEEKDAY]
    rng = np.random.default_rng(20260728)
    market_weekly_return = rng.normal(0.001, 0.018, len(wednesdays) - 1)
    current_prices = np.r_[100.0, 100.0 * np.cumprod(1.0 + market_weekly_return)]
    delayed_weekly_return = np.r_[0.0, market_weekly_return[:-1]]
    delayed_prices = np.r_[100.0, 100.0 * np.cumprod(1.0 + delayed_weekly_return)]
    symbols = pd.Index(
        [*(f"{index:06d}.SZ" for index in range(1, 21)), "000999.SZ"],
        name="ts_code",
    )
    weekly_prices = pd.DataFrame(
        np.column_stack(
            [current_prices * (1.0 + 0.00001 * index) for index in range(20)] + [delayed_prices]
        ),
        index=wednesdays,
        columns=symbols,
    )
    adj_close = weekly_prices.reindex(dates, method="ffill").bfill()
    total_mv = pd.DataFrame(
        np.tile([*[1_000_000.0] * 20, 1.0], (len(dates), 1)),
        index=dates,
        columns=symbols,
    )
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.high_price_delay_d1_v4(
        adj_close=adj_close,
        total_mv=total_mv,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_high_price_delay_d1_52w",
        inputs={"adj_close": adj_close, "total_mv": total_mv},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[-1, -1] > full.iloc[-1, :-1].max()

    prefix_end = dates[-25]
    prefix = subject.high_price_delay_d1_v4(
        adj_close=adj_close.loc[:prefix_end],
        total_mv=total_mv.loc[:prefix_end],
        pit_mask=pit.loc[:prefix_end],
    )
    pd.testing.assert_frame_equal(full.loc[:prefix_end], prefix)


def test_52_week_high_is_causal_and_prefers_price_nearest_trailing_high() -> None:
    dates, symbols = _axes(rows=300, columns=3)
    first = np.linspace(100.0, 150.0, len(dates))
    second = np.r_[np.linspace(100.0, 150.0, 220), np.linspace(149.0, 110.0, 80)]
    third = np.r_[np.linspace(100.0, 150.0, 220), np.linspace(149.0, 75.0, 80)]
    adj_close = _frame(
        np.column_stack([first, second, third]),
        dates=dates,
        symbols=symbols,
    )
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.high_52_week_momentum_v4(
        adj_close=adj_close,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_52_week_high_momentum_12m",
        inputs={"adj_close": adj_close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[-1, 0] > full.iloc[-1, 1] > full.iloc[-1, 2]

    prefix = subject.high_52_week_momentum_v4(
        adj_close=adj_close.iloc[:270],
        pit_mask=pit.iloc[:270],
    )
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)


def test_low_max_is_causal_and_prefers_stocks_without_lottery_spikes() -> None:
    dates, symbols = _axes(rows=80, columns=3)
    returns = np.full((80, 3), 0.001, dtype=float)
    returns[55, 0] = 0.20
    returns[56, 1] = 0.08
    returns[57, 2] = 0.02
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adj_close = _frame(close, dates=dates, symbols=symbols)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.low_max_return_v4(adj_close=adj_close, pit_mask=pit)
    dispatched = subject.evaluate_candidate_v4(
        name="cn_low_max_return_20d",
        inputs={"adj_close": adj_close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[57, 2] > full.iloc[57, 1] > full.iloc[57, 0]

    prefix = subject.low_max_return_v4(
        adj_close=adj_close.iloc[:70],
        pit_mask=pit.iloc[:70],
    )
    pd.testing.assert_frame_equal(full.iloc[:70], prefix)


def test_total_skewness_is_causal_and_prefers_lower_prior_month_skewness() -> None:
    dates, symbols = _axes(rows=80, columns=3)
    returns = np.full((80, 3), 0.001, dtype=float)
    returns[65, 0] = 0.20
    returns[66, 1] = 0.10
    returns[67, 1] = -0.10
    returns[68, 2] = -0.20
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adj_close = _frame(close, dates=dates, symbols=symbols)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.low_total_skewness_v4(
        adj_close=adj_close,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_low_total_skewness_20d",
        inputs={"adj_close": adj_close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[-1, 2] > full.iloc[-1, 1] > full.iloc[-1, 0]

    prefix = subject.low_total_skewness_v4(
        adj_close=adj_close.iloc[:70],
        pit_mask=pit.iloc[:70],
    )
    pd.testing.assert_frame_equal(full.iloc[:70], prefix)


def test_tail_asymmetry_is_causal_and_prefers_lower_upside_tail_probability() -> None:
    dates, symbols = _axes(rows=300, columns=3)
    returns = np.zeros((300, 3), dtype=float)
    for position in range(60, 300, 12):
        returns[position, 0] = 0.12
    for position in range(66, 300, 12):
        returns[position, 2] = -0.12
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adj_close = _frame(close, dates=dates, symbols=symbols)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.low_market_adjusted_tail_asymmetry_v4(
        adj_close=adj_close,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_low_market_adjusted_tail_asymmetry_252d",
        inputs={"adj_close": adj_close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[-1, 2] > full.iloc[-1, 1] > full.iloc[-1, 0]

    prefix = subject.low_market_adjusted_tail_asymmetry_v4(
        adj_close=adj_close.iloc[:270],
        pit_mask=pit.iloc[:270],
    )
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)


def test_left_tail_var1_is_causal_and_prefers_lower_one_percent_var() -> None:
    dates, symbols = _axes(rows=300, columns=3)
    returns = np.full((300, 3), 0.001, dtype=float)
    for position in (100, 150, 200, 250):
        returns[position] = [-0.25, -0.10, -0.03]
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adj_close = _frame(close, dates=dates, symbols=symbols)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.low_left_tail_var1_v4(
        adj_close=adj_close,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_low_left_tail_var1_250d",
        inputs={"adj_close": adj_close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[: subject.LEFT_TAIL_VAR_MIN_PERIODS].isna().all().all()
    assert full.iloc[-1, 2] > full.iloc[-1, 1] > full.iloc[-1, 0]

    prefix = subject.low_left_tail_var1_v4(
        adj_close=adj_close.iloc[:270],
        pit_mask=pit.iloc[:270],
    )
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)

    with pytest.raises(
        subject.FactorGovernanceLiteratureIncubatorV4Error,
        match="within \\(0, 0.5\\)",
    ):
        subject.low_left_tail_var1_v4(
            adj_close=adj_close,
            pit_mask=pit,
            quantile=0.5,
        )


def test_quality_proxy_uses_complete_cases_and_low_leverage_direction() -> None:
    dates, symbols = _axes(rows=2, columns=4)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    roe = _frame(
        np.tile([0.05, 0.10, 0.15, 0.20], (2, 1)),
        dates=dates,
        symbols=symbols,
    )
    cash = roe.copy()
    debt = _frame(
        np.tile([0.80, 0.60, 0.40, 0.20], (2, 1)),
        dates=dates,
        symbols=symbols,
    )
    cash.iloc[1, 3] = np.nan
    signal = subject.quality_cash_low_leverage_v4(
        fin_roe=roe,
        fin_ocf_to_profit=cash,
        fin_debt_to_assets=debt,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_quality_cash_low_leverage",
        inputs={
            "fin_roe": roe,
            "fin_ocf_to_profit": cash,
            "fin_debt_to_assets": debt,
        },
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(signal, dispatched)
    assert signal.iloc[0].is_monotonic_increasing
    assert pd.isna(signal.iloc[1, 3])


def test_same_month_seasonality_is_causal_and_prefers_prior_same_month_winner() -> None:
    dates = pd.bdate_range(
        "2020-01-02",
        "2026-07-31",
        name="trade_date",
    )
    symbols = pd.Index(["000001.SZ", "000002.SZ"], name="ts_code")
    returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=symbols,
        dtype=float,
    )
    for year in range(2021, 2026):
        july_dates = dates[(dates.year == year) & (dates.month == 7)]
        returns.loc[july_dates[-1], "000001.SZ"] = 0.10
        returns.loc[july_dates[-1], "000002.SZ"] = -0.05
    close = 100.0 * returns.add(1.0).cumprod()
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.same_month_seasonality_v4(
        adj_close=close,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_same_month_seasonality_5y",
        inputs={"adj_close": close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    july_2026 = dates[(dates.year == 2026) & (dates.month == 7)]
    assert full.loc[july_2026, "000001.SZ"].notna().all()
    assert (full.loc[july_2026, "000001.SZ"] > full.loc[july_2026, "000002.SZ"]).all()

    prefix_date = pd.Timestamp("2026-07-15")
    prefix = subject.same_month_seasonality_v4(
        adj_close=close.loc[:prefix_date],
        pit_mask=pit.loc[:prefix_date],
    )
    pd.testing.assert_frame_equal(full.loc[:prefix_date], prefix)


def test_fip_continuous_direction_is_causal_and_rewards_gradual_trends() -> None:
    dates, symbols = _axes(rows=300, columns=4)
    returns = np.zeros((300, 4), dtype=float)
    returns[1:280, 0] = 0.001
    returns[1:280, 1] = -0.001
    returns[1:280, 2] = -0.0005
    returns[120, 2] = 0.50
    returns[1:280, 3] = 0.0005
    returns[120, 3] = -0.30
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adj_close = _frame(close, dates=dates, symbols=symbols)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    full = subject.fip_continuous_direction_v4(
        adj_close=adj_close,
        pit_mask=pit,
    )
    dispatched = subject.evaluate_candidate_v4(
        name="cn_fip_continuous_direction_12m",
        inputs={"adj_close": adj_close},
        pit_mask=pit,
    )
    pd.testing.assert_frame_equal(full, dispatched)
    assert full.iloc[-1, 0] > full.iloc[-1, 2]
    assert full.iloc[-1, 3] > full.iloc[-1, 1]

    prefix = subject.fip_continuous_direction_v4(
        adj_close=adj_close.iloc[:270],
        pit_mask=pit.iloc[:270],
    )
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)


def test_dispatch_rejects_extra_fields_unknown_candidates_and_axis_drift() -> None:
    dates, symbols = _axes(rows=3, columns=3)
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    values = _frame(np.ones((3, 3)), dates=dates, symbols=symbols)
    with pytest.raises(
        subject.FactorGovernanceLiteratureIncubatorV4Error,
        match="exact required fields",
    ):
        subject.evaluate_candidate_v4(
            name="cn_low_beta_252d",
            inputs={"adj_close": values, "unexpected": values},
            pit_mask=pit,
        )
    with pytest.raises(
        subject.FactorGovernanceLiteratureIncubatorV4Error,
        match="not allowlisted",
    ):
        subject.evaluate_candidate_v4(
            name="unknown",
            inputs={"adj_close": values},
            pit_mask=pit,
        )
    with pytest.raises(
        subject.FactorGovernanceLiteratureIncubatorV4Error,
        match="strictly ordered",
    ):
        subject.low_beta_v4(
            adj_close=values.iloc[::-1],
            pit_mask=pit.iloc[::-1],
        )
