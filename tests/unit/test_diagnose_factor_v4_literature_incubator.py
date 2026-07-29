from __future__ import annotations

import stat

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import governance_screening_v4 as screening
from scripts import diagnose_factor_v4_literature_incubator as subject


def test_exclusive_writer_requires_owner_only_parent_and_never_overwrites(
    tmp_path,
) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    output = private_parent / "report.json"
    subject._write_exclusive(output, {"status": "test"})
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(subject.FactorV4LiteratureDiagnosticError, match="already exists"):
        subject._write_exclusive(output, {"status": "replacement"})

    broad_parent = tmp_path / "broad"
    broad_parent.mkdir(mode=0o755)
    with pytest.raises(
        subject.FactorV4LiteratureDiagnosticError,
        match="owner-only 0700",
    ):
        subject._write_exclusive(
            broad_parent / "report.json",
            {"status": "test"},
        )


def test_signal_summary_uses_last_computable_session_not_last_market_session() -> None:
    dates = pd.bdate_range("2026-01-02", periods=4, name="trade_date")
    symbols = pd.Index(["000001.SZ", "000002.SZ"], name="ts_code")
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    signal = pd.DataFrame(
        [[1.0, np.nan], [1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan]],
        index=dates,
        columns=symbols,
    )
    result = subject._signal_summary(signal, pit_mask=pit)
    assert result["status"] == "COMPUTABLE_RESEARCH_ONLY"
    assert result["last_computable_session"] == dates[-2].date().isoformat()
    assert result["latest_finite_count"] == 1
    assert result["latest_eligible_count"] == 2
    assert result["latest_coverage_ratio"] == 0.5


def test_pairwise_correlation_is_diagnostic_only() -> None:
    dates = pd.bdate_range("2026-01-02", periods=3, name="trade_date")
    symbols = pd.Index(
        [f"{index:06d}.SZ" for index in range(1, 121)],
        name="ts_code",
    )
    ascending = pd.DataFrame(
        np.tile(np.arange(120, dtype=float), (3, 1)),
        index=dates,
        columns=symbols,
    )
    descending = -ascending
    rows = subject._pairwise_correlation_diagnostic(
        {"ascending": ascending, "descending": descending}
    )
    assert rows == [
        {
            "left": "ascending",
            "right": "descending",
            "valid_session_count": 3,
            "mean_cross_sectional_spearman": -1.0,
            "median_cross_sectional_spearman": -1.0,
            "formal_dedup_evidence": False,
        }
    ]


def test_momentum_comparators_preserve_full_axes_and_pit_mask() -> None:
    dates = pd.bdate_range("2025-01-02", periods=130, name="trade_date")
    symbols = pd.Index(["000001.SZ", "000002.SZ"], name="ts_code")
    close = pd.DataFrame(
        np.column_stack(
            [
                100.0 + np.arange(len(dates), dtype=float),
                200.0 + np.arange(len(dates), dtype=float),
            ]
        ),
        index=dates,
        columns=symbols,
    )
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    pit.iloc[-1, 1] = False
    signals = subject._build_momentum_comparison_signals(
        adj_close=close,
        pit_mask=pit,
    )
    assert set(signals) == {"pv_momentum_20d", "pv_momentum_120d"}
    assert all(signal.index.equals(dates) for signal in signals.values())
    assert signals["pv_momentum_20d"].iloc[:20].isna().all().all()
    assert signals["pv_momentum_120d"].iloc[:120].isna().all().all()
    assert signals["pv_momentum_20d"].iloc[-1, 0] > 0.0
    assert pd.isna(signals["pv_momentum_20d"].iloc[-1, 1])


def test_protected_exact_five_signals_require_source_local_equivalence() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80, name="trade_date")
    symbols = pd.Index(
        [f"{index:06d}.SZ" for index in range(1, 31)],
        name="ts_code",
    )
    phase = np.arange(len(dates), dtype=float)
    close_values = np.column_stack(
        [
            100.0 + 0.1 * phase + 0.2 * np.sin(phase / (3.0 + column / 10.0))
            for column in range(len(symbols))
        ]
    )
    close = pd.DataFrame(close_values, index=dates, columns=symbols)
    market = {
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]).mul(1.0005),
        "vol": pd.DataFrame(
            1_000_000.0 + np.add.outer(phase * 100.0, np.arange(len(symbols)) * 10.0),
            index=dates,
            columns=symbols,
        ),
        "adj_close": close.mul(1.01),
    }
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    signals, binding = subject._build_protected_exact_five_comparison_signals(
        market=market,
        pit_mask=pit,
    )
    assert tuple(signals) == subject.PROTECTED_EXACT_FIVE_NAMES
    assert binding["source_local_engine_equivalence_proven"] is True
    assert binding["labels_loaded"] is False
    assert binding["forward_returns_loaded"] is False
    assert binding["factor_v4_authority"] is False
    assert signals["pv_low_overnight_gap_20d"].iloc[-1].median() < 0.0
    assert signals["alpha_range_position_momentum_20d"].iloc[-1].median() > 0.0


def test_seasonality_coverage_requires_all_twelve_calendar_months() -> None:
    dates = pd.date_range(
        "2025-01-31",
        periods=13,
        freq="BME",
        name="trade_date",
    )
    symbols = pd.Index(
        [f"{index:06d}.SZ" for index in range(1, 11)],
        name="ts_code",
    )
    pit = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    signal = pd.DataFrame(1.0, index=dates, columns=symbols, dtype=float)
    signal.loc[signal.index.month == 9] = np.nan
    result = subject._seasonality_calendar_coverage_diagnostic(
        signal=signal,
        pit_mask=pit,
    )
    assert result["stable_calendar_month_count"] == 11
    assert result["missing_calendar_months"] == [9]
    assert result["all_calendar_months_covered"] is False
    assert result["formal_coverage_evidence"] is False


def test_monthly_dedup_uses_closed_months_absolute_spearman_only() -> None:
    dates = pd.DatetimeIndex(
        [
            "2026-01-30",
            "2026-02-27",
            "2026-03-31",
            "2026-04-30",
            "2026-05-15",
        ],
        name="trade_date",
    )
    symbols = pd.Index(
        [f"{index:06d}.SZ" for index in range(1, 31)],
        name="ts_code",
    )
    ascending = pd.DataFrame(
        np.tile(np.arange(30, dtype=float), (len(dates), 1)),
        index=dates,
        columns=symbols,
    )
    descending = -ascending
    candidate_signals = {name: ascending for name in subject.DEDUP_COMPARISON_ROUTES}
    comparison_names = sorted(
        {
            name
            for candidate_name in subject.DEDUP_COMPARISON_ROUTES
            for name in subject._dedup_route_names(candidate_name)
        }
    )
    comparison_signals = {name: descending for name in comparison_names}
    incubator_names = {row["name"] for row in subject.incubator.candidate_catalog_v4()}
    protected_names = set(subject.PROTECTED_EXACT_FIVE_NAMES)
    formal_comparison_names = [
        name
        for name in comparison_names
        if name not in incubator_names and name not in protected_names
    ]
    primitives = [
        {"primitive_id": f"p{index}", "family": f"f{index}"}
        for index in range(len(formal_comparison_names))
    ]
    ontology = screening.build_primitive_ontology_v4(primitives)
    catalog = screening.build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            {
                "name": name,
                "implementation": f"test:{name}",
                "expression": name,
                "direction": 1.0,
                "params": {},
                "lookback": 1,
                "slot": f"slot:{index}",
                "input_fields": [f"input_{index}"],
                "primitive_ids": [f"p{index}"],
            }
            for index, name in enumerate(formal_comparison_names)
        ],
    )
    result = subject._monthly_dedup_diagnostic(
        candidate_signals=candidate_signals,
        comparison_signals=comparison_signals,
        comparison_catalog=catalog,
        protected_candidates=list(subject.exact_five_prereg.EXPECTED_CANDIDATE_ROWS),
    )
    assert len(result["rows"]) == sum(
        len(subject._dedup_route_names(candidate_name))
        for candidate_name in subject.DEDUP_COMPARISON_ROUTES
    )
    assert result["formal_dedup_evidence"] is False
    assert all(row["valid_common_date_count"] == 4 for row in result["rows"])
    assert all(row["abs_correlation"] == 1.0 for row in result["rows"])
    assert all(row["status"] == "THRESHOLD_BREACHED_DIAGNOSTIC" for row in result["rows"])
    assert all(
        row["closed_month_end_rows"][-1]["month_end"] == "2026-04-30" for row in result["rows"]
    )


def test_monthly_dedup_rejects_formal_incubator_namespace_collision() -> None:
    candidate_name = "cn_low_max_return_20d"
    ontology = screening.build_primitive_ontology_v4(
        [{"primitive_id": "collision", "family": "collision"}]
    )
    catalog = screening.build_candidate_catalog_v4(
        ontology=ontology,
        candidates=[
            {
                "name": candidate_name,
                "implementation": "test:collision",
                "expression": "collision",
                "direction": 1.0,
                "params": {},
                "lookback": 1,
                "slot": "slot:collision",
                "input_fields": ["collision"],
                "primitive_ids": ["collision"],
            }
        ],
    )
    with pytest.raises(
        subject.FactorV4LiteratureDiagnosticError,
        match="comparison namespaces collide",
    ):
        subject._monthly_dedup_diagnostic(
            candidate_signals={},
            comparison_signals={},
            comparison_catalog=catalog,
            protected_candidates=list(subject.exact_five_prereg.EXPECTED_CANDIDATE_ROWS),
        )


def test_candidate_routing_stops_breaches_and_waits_for_samples() -> None:
    candidate_names = sorted(subject.DEDUP_COMPARISON_ROUTES)
    structural = {
        "candidate_results": [
            {
                "candidate_name": name,
                "structural_collision_passed_diagnostic": True,
            }
            for name in candidate_names
        ]
    }
    monthly = {
        "rows": [
            {
                "candidate_name": "cn_earnings_yield_ex_shell_30pct",
                "existing_factor_name": "fund_fcf_to_price",
                "status": "INSUFFICIENT_MONTHS_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_low_beta_252d",
                "existing_factor_name": "pv_volatility_penalty_60d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_52_week_high_momentum_12m",
                "existing_factor_name": "pv_momentum_120d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_low_max_return_20d",
                "existing_factor_name": "pv_volatility_penalty_60d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_low_total_skewness_20d",
                "existing_factor_name": "cn_low_max_return_20d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_low_market_adjusted_tail_asymmetry_252d",
                "existing_factor_name": "cn_low_max_return_20d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_low_left_tail_var1_250d",
                "existing_factor_name": "pv_downside_volatility_60d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_quality_cash_low_leverage",
                "existing_factor_name": "fund_quality_cash_combo",
                "status": "THRESHOLD_BREACHED_DIAGNOSTIC",
            },
            {
                "candidate_name": "cn_same_month_seasonality_5y",
                "existing_factor_name": "pv_momentum_120d",
                "status": "BELOW_THRESHOLD_DIAGNOSTIC",
            },
        ]
    }
    decisions = subject._candidate_routing_decisions(
        structural_audit=structural,
        protected_exact_five_audit={
            "candidate_results": [
                {
                    "candidate_name": name,
                    "protected_structural_collision_passed_diagnostic": True,
                    "definition_identity_collision_names": [],
                    "slot_collision_names": [],
                }
                for name in candidate_names
            ]
        },
        monthly_dedup=monthly,
        seasonality_calendar_coverage={
            "all_calendar_months_covered": False,
            "missing_calendar_months": [1, 2],
        },
        literature_assessments=(subject.incubator.candidate_literature_assessments_v4()),
        candidate_computability={
            name: {"status": "COMPUTABLE_RESEARCH_ONLY"} for name in candidate_names
        },
        candidate_input_history_diagnostics={
            "cn_52_week_high_momentum_12m": {"status": "BLOCKED_MISSING_PIT_CHINA_EPU_SERIES"}
        },
    )
    by_name = {row["candidate_name"]: row for row in decisions}
    assert (
        by_name["cn_earnings_yield_ex_shell_30pct"]["status"] == "WAITING_FOR_COMMON_CLOSED_MONTHS"
    )
    assert by_name["cn_low_beta_252d"]["status"] == "CONTROL_ONLY_LITERATURE_MECHANISM_CONFLICT"
    assert by_name["cn_52_week_high_momentum_12m"]["status"] == "WAITING_FOR_PIT_CHINA_EPU_SERIES"
    assert by_name["cn_52_week_high_momentum_12m"]["reasons"] == [
        "input_history_status:BLOCKED_MISSING_PIT_CHINA_EPU_SERIES"
    ]
    assert by_name["cn_low_max_return_20d"]["status"] == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
    assert by_name["cn_low_total_skewness_20d"]["status"] == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
    assert (
        by_name["cn_low_market_adjusted_tail_asymmetry_252d"]["status"]
        == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
    )
    assert by_name["cn_low_left_tail_var1_250d"]["status"] == "ELIGIBLE_FOR_FUTURE_PREREGISTRATION"
    assert by_name["cn_quality_cash_low_leverage"]["status"] == "STOP_HIGH_CORRELATION"
    assert (
        by_name["cn_same_month_seasonality_5y"]["status"] == "WAITING_FOR_12_STABLE_CALENDAR_MONTHS"
    )
    assert all(row["formal_preregistration_created"] is False for row in decisions)


def test_candidate_routing_stops_protected_exact_five_collision() -> None:
    candidate_names = sorted(subject.DEDUP_COMPARISON_ROUTES)
    decisions = subject._candidate_routing_decisions(
        structural_audit={
            "candidate_results": [
                {
                    "candidate_name": name,
                    "structural_collision_passed_diagnostic": True,
                }
                for name in candidate_names
            ]
        },
        protected_exact_five_audit={
            "candidate_results": [
                {
                    "candidate_name": name,
                    "protected_structural_collision_passed_diagnostic": (
                        name != "cn_low_max_return_20d"
                    ),
                    "definition_identity_collision_names": [],
                    "slot_collision_names": (
                        ["pv_low_overnight_gap_20d"] if name == "cn_low_max_return_20d" else []
                    ),
                }
                for name in candidate_names
            ]
        },
        monthly_dedup={"rows": []},
        seasonality_calendar_coverage={
            "all_calendar_months_covered": True,
            "missing_calendar_months": [],
        },
        literature_assessments=(subject.incubator.candidate_literature_assessments_v4()),
        candidate_computability={
            name: {"status": "COMPUTABLE_RESEARCH_ONLY"} for name in candidate_names
        },
        candidate_input_history_diagnostics={},
    )
    low_max = next(row for row in decisions if row["candidate_name"] == "cn_low_max_return_20d")
    assert low_max["status"] == "STOP_PROTECTED_EXACT_FIVE_COLLISION"
    assert low_max["reasons"] == ["protected_slot_collision:pv_low_overnight_gap_20d"]


def test_candidate_routing_blocks_noncomputable_signal_before_dedup() -> None:
    candidate_names = sorted(subject.DEDUP_COMPARISON_ROUTES)
    computability = {name: {"status": "COMPUTABLE_RESEARCH_ONLY"} for name in candidate_names}
    computability["cn_high_price_delay_d1_52w"] = {"status": "NOT_COMPUTABLE"}
    decisions = subject._candidate_routing_decisions(
        structural_audit={
            "candidate_results": [
                {
                    "candidate_name": name,
                    "structural_collision_passed_diagnostic": True,
                }
                for name in candidate_names
            ]
        },
        protected_exact_five_audit={
            "candidate_results": [
                {
                    "candidate_name": name,
                    "protected_structural_collision_passed_diagnostic": True,
                    "definition_identity_collision_names": [],
                    "slot_collision_names": [],
                }
                for name in candidate_names
            ]
        },
        monthly_dedup={"rows": []},
        seasonality_calendar_coverage={
            "all_calendar_months_covered": True,
            "missing_calendar_months": [],
        },
        literature_assessments=(subject.incubator.candidate_literature_assessments_v4()),
        candidate_computability=computability,
        candidate_input_history_diagnostics={
            "cn_high_price_delay_d1_52w": {
                "status": "BLOCKED_INSUFFICIENT_PIT_VALUE_WEIGHTED_MARKET_HISTORY"
            }
        },
    )
    price_delay = next(
        row for row in decisions if row["candidate_name"] == "cn_high_price_delay_d1_52w"
    )
    assert price_delay["status"] == "BLOCKED_NOT_COMPUTABLE"
    assert price_delay["reasons"] == [
        "candidate_signal_status:NOT_COMPUTABLE",
        ("input_history_status:" "BLOCKED_INSUFFICIENT_PIT_VALUE_WEIGHTED_MARKET_HISTORY"),
    ]


def test_future_policy_applicability_retires_stopped_draft() -> None:
    routing = [
        {
            "candidate_name": "cn_fip_continuous_direction_12m",
            "status": "ELIGIBLE_FOR_FUTURE_PREREGISTRATION",
            "reasons": ["diagnostic_routes_below_dedup_threshold"],
        },
        {
            "candidate_name": "cn_low_max_return_20d",
            "status": "STOP_HIGH_CORRELATION",
            "reasons": ["dedup_threshold_breached:pv_low_vol_of_vol_20d"],
        },
        {
            "candidate_name": "cn_low_total_skewness_20d",
            "status": "ELIGIBLE_FOR_FUTURE_PREREGISTRATION",
            "reasons": ["diagnostic_routes_below_dedup_threshold"],
        },
        {
            "candidate_name": "cn_low_market_adjusted_tail_asymmetry_252d",
            "status": "ELIGIBLE_FOR_FUTURE_PREREGISTRATION",
            "reasons": ["diagnostic_routes_below_dedup_threshold"],
        },
        {
            "candidate_name": "cn_low_left_tail_var1_250d",
            "status": "ELIGIBLE_FOR_FUTURE_PREREGISTRATION",
            "reasons": ["diagnostic_routes_below_dedup_threshold"],
        },
    ]
    applicability = subject._future_policy_applicability(routing)
    assert applicability["cn_low_max_return_20d"] == {
        "applicability": "INAPPLICABLE_STOP_HIGH_CORRELATION",
        "routing_status": "STOP_HIGH_CORRELATION",
        "routing_reasons": ["dedup_threshold_breached:pv_low_vol_of_vol_20d"],
        "formal_preregistration_created": False,
        "factor_v4_authority": False,
    }
    assert (
        applicability["cn_low_total_skewness_20d"]["applicability"]
        == "DIAGNOSTICALLY_ELIGIBLE_DRAFT_ONLY"
    )
    assert (
        applicability["cn_fip_continuous_direction_12m"]["applicability"]
        == "DIAGNOSTICALLY_ELIGIBLE_DRAFT_ONLY"
    )
    assert (
        applicability["cn_low_left_tail_var1_250d"]["applicability"]
        == "DIAGNOSTICALLY_ELIGIBLE_DRAFT_ONLY"
    )
