from __future__ import annotations

import ast
from decimal import Decimal
import json
from pathlib import Path

import pandas as pd

from quant_investor.v17_v2_runtime.algorithms.deep_research import (
    COVERAGE_SECTIONS,
    LAYER_NAMES,
    SEVERE_RED_FLAGS,
    SIGNAL_WEIGHTS,
    evaluate_deep_research,
)
from quant_investor.v17_v2_runtime.algorithms.forward_calibration import (
    assess_fundamental_eligibility,
    calibrate_forward_returns,
)
from quant_investor.v17_v2_runtime.algorithms.fundamental_scoring import (
    MAIN_METRICS,
    score_fundamental_universe,
)
from quant_investor.v17_v2_runtime.algorithms.optimizer import (
    FeasiblePortfolio,
    ProposedTrade,
    optimize_lexicographic,
)
from quant_investor.v17_v2_runtime.algorithms.permissions import (
    apply_permission_restrictions,
    build_permission_restriction,
    determine_trade_permission,
)
from quant_investor.v17_v2_runtime.algorithms.quant_timing import (
    calibrate_timing_probabilities,
    pava_non_decreasing,
)
from quant_investor.v17_v2_runtime.algorithms.regime_overlay import (
    build_available_overlay_input,
    compute_regime_portfolio_overlay,
)
from quant_investor.v17_v2_runtime.algorithms.transaction_cost import (
    estimate_transaction_cost,
)
from quant_investor.v17_v2_runtime.pipeline import (
    PipelineInput,
    PipelineResult,
    RankOutput,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "quant_investor" / "v17_v2_runtime"
VECTORS = json.loads(
    (REPO_ROOT / "tests/fixtures/v17_v2_algorithm_vectors.json").read_text(encoding="utf-8")
)


def test_runtime_static_imports_never_reference_legacy_v17() -> None:
    for path in RUNTIME_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("quant_investor.v17.")
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("quant_investor.v17.") for alias in node.names
                )


def test_square_root_cost_frozen_vector() -> None:
    vector = VECTORS["transaction_cost"]
    result = estimate_transaction_cost(
        coefficient=vector["coefficient"],
        notional=vector["notional"],
        adv20=vector["adv20"],
    )
    assert result.fraction == Decimal(vector["fraction"])
    assert result.amount == Decimal(vector["amount"])
    assert result.authority is False


def test_pava_frozen_vector() -> None:
    vector = VECTORS["pava"]
    assert pava_non_decreasing(vector["values"], vector["weights"]) == vector["expected"]


def test_jeffreys_probability_formula_parity() -> None:
    rows = []
    for index, excess_return in enumerate((0.1, -0.1, 0.2)):
        start = pd.Timestamp("2025-01-01T07:00:00Z") + pd.Timedelta(days=index)
        end = start + pd.Timedelta(days=20)
        rows.append(
            {
                "horizon": 20,
                "symbol": f"{index + 1:06d}.SZ",
                "score_decile": 1,
                "cross_section_date": start,
                "availability": end,
                "age_open_days": index,
                "target_start_trade_date": start,
                "target_end_trade_date": end,
                "realized_open_days": 20,
                "is_mature": True,
                "is_pit": True,
                "target_definition": "EXCESS_RETURN_GT_ZERO",
                "excess_return": excess_return,
            }
        )
    calibration = calibrate_timing_probabilities(
        pd.DataFrame(rows),
        cutoff="2026-07-01T07:00:00Z",
    )
    cell = calibration.cells.loc[
        (calibration.cells["horizon"] == 20)
        & (calibration.cells["score_decile"] == 1)
    ].iloc[0]
    assert cell["wins"] == 2
    assert cell["jeffreys_probability"] == (2.0 + 0.5) / (3.0 + 1.0)


def test_five_pillar_top24_and_appended_holding_parity() -> None:
    snapshot: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    metric_names = (*MAIN_METRICS, "fin_fcf_to_profit")
    for symbol_index in range(1, 26):
        symbol = f"{symbol_index:06d}.SZ"
        snapshot.append(
            {
                "symbol": symbol,
                "industry": "industry-a",
                "in_universe": True,
                "research_eligible": True,
                "membership_conflict": False,
                "membership_is_pit": True,
                "universe_id": "CN/full_a",
                "availability": "2026-06-30T07:00:00Z",
                "flow_basis": "LATEST_TTM",
                "balance_sheet_basis": "LATEST_REPORT_PERIOD",
                "capex_sign_convention": "POSITIVE_OUTFLOW",
                "net_profit_ttm": 10.0,
                "market_cap": 1000.0,
                "cfo_ttm": 100.0 + symbol_index,
                "capex_ttm": 10.0,
                "fin_roe": float(symbol_index),
                "fin_ocf_to_profit": float(symbol_index),
                "fin_net_profit_yoy": float(symbol_index),
                "fin_debt_to_assets": float(26 - symbol_index),
            }
        )
        for metric in metric_names:
            for day in range(252):
                history.append(
                    {
                        "symbol": symbol,
                        "trade_date": pd.Timestamp("2025-01-01T07:00:00Z")
                        + pd.Timedelta(days=day),
                        "availability": pd.Timestamp("2025-01-01T07:00:00Z")
                        + pd.Timedelta(days=day),
                        "is_open_day": True,
                        "metric": metric,
                        "value": day / 10.0,
                    }
                )
    result = score_fundamental_universe(
        pd.DataFrame(snapshot),
        pd.DataFrame(history),
        cutoff="2026-07-01T07:00:00Z",
        holdings=("000001.SZ",),
    )
    assert len(result.ranked_symbols) == 24
    assert result.ranked_symbols[0] == "000025.SZ"
    assert result.appended_holdings == ("000001.SZ",)
    assert len(result.sealed_symbols) == 25


def test_regime_min_cap_max_floor_frozen_vector() -> None:
    overlay = compute_regime_portfolio_overlay(
        base=build_available_overlay_input(name="base", gross_cap=0.9, cash_floor=0.1),
        macro=build_available_overlay_input(name="macro", gross_cap=0.7, cash_floor=0.3),
        markov=build_available_overlay_input(name="markov", gross_cap=0.65, cash_floor=0.2),
    )
    expected = VECTORS["regime"]
    assert overlay["gross_cap"] == expected["gross_cap"]
    assert overlay["cash_floor"] == expected["cash_floor"]
    assert overlay["effective_gross"] == expected["effective_gross"]
    assert overlay["authority"] is False


def test_permission_truth_table_and_restrictions_are_shrink_only() -> None:
    buy = determine_trade_permission(
        symbol="000001.SZ",
        held=False,
        tradable=True,
        fundamental_eligibility="F_ELIGIBLE",
        severe_red_flag=False,
        quant_timing="BUY_NOW",
    )
    assert (buy["can_buy"], buy["can_sell"], buy["position_locked"]) == (True, False, False)
    restricted = apply_permission_restrictions(
        buy,
        restrictions=(
            build_permission_restriction(
                gate="risk",
                allow_buy=False,
                allow_sell=True,
                reason="risk_cap",
            ),
            build_permission_restriction(
                gate="optimizer",
                allow_buy=True,
                allow_sell=True,
                reason="candidate_mask",
            ),
        ),
    )
    assert restricted["can_buy"] is False
    assert restricted["can_sell"] is False

    trim = determine_trade_permission(
        symbol="000001.SZ",
        held=True,
        tradable=True,
        fundamental_eligibility="F_INELIGIBLE",
        severe_red_flag=True,
        quant_timing="TRIM_TIMING",
    )
    assert (trim["can_buy"], trim["can_sell"], trim["position_locked"]) == (False, True, False)


def test_decimal_optimizer_frozen_lexicographic_vector() -> None:
    trades = (ProposedTrade("000001.SZ", "BUY", "0.20"),)
    candidates = (
        FeasiblePortfolio(
            "candidate-a",
            {"000001.SZ": "0.20"},
            trades,
            "0.13",
            "0.02",
            "0.20",
        ),
        FeasiblePortfolio(
            "candidate-b",
            {"000001.SZ": "0.20"},
            trades,
            "0.12",
            "0.01",
            "0.20",
        ),
    )
    result = optimize_lexicographic(
        candidates,
        permission_mask={"000001.SZ": {"BUY"}},
        current_weights={},
        effective_gross="0.80",
    )
    assert result.selected is not None
    expected = VECTORS["optimizer"]
    assert result.selected.candidate_id == expected["expected_selected"]
    assert result.selected.net_adjusted_q25 == Decimal(expected["net_q25"])
    assert Decimal(str(result.selected.turnover)) == Decimal(expected["turnover"])


def _forward_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cutoff = pd.Timestamp("2026-07-01T07:00:00Z")
    for horizon in (120, 252, 378):
        for date_index in range(20):
            start = pd.Timestamp("2023-01-02T07:00:00Z") + pd.Timedelta(days=date_index)
            end = start + pd.Timedelta(days=horizon)
            for symbol_index in range(5):
                symbol = f"{date_index * 5 + symbol_index:06d}.SZ"
                rows.append(
                    {
                        "symbol": symbol,
                        "industry": "industry-a",
                        "score_decile": 10,
                        "horizon": horizon,
                        "cross_section_date": start,
                        "availability": end,
                        "age_open_days": date_index,
                        "realized_open_days": horizon,
                        "is_pit_month_end": True,
                        "is_mature": True,
                        "stock_start_trade_date": start,
                        "stock_end_trade_date": end,
                        "benchmark_start_trade_date": start,
                        "benchmark_end_trade_date": end,
                        "stock_total_return": 0.10 + symbol_index / 1000,
                        "benchmark_total_return": 0.02,
                        "benchmark_symbol": "H00300.CSI",
                        "stock_return_includes_dividends": True,
                        "benchmark_return_is_pre_tax_total_return": True,
                        "delisted": False,
                        "official_terminal_cash_settlement": False,
                    }
                )
    assert max(row["availability"] for row in rows) < cutoff
    return pd.DataFrame(rows)


def _deep_response() -> dict[str, object]:
    evidence = ["evidence-1"]
    return {
        "symbol": "000001.SZ",
        "layers": {
            layer: [{"layer": layer, "content": f"{layer} conclusion", "evidence_ids": evidence}]
            for layer in LAYER_NAMES
        },
        "coverage": {
            section: {"conclusion": f"{section} conclusion", "evidence_ids": evidence}
            for section in COVERAGE_SECTIONS
        },
        "signals": {
            dimension: {"signal": 1.0, "evidence_ids": evidence}
            for dimension in SIGNAL_WEIGHTS
        },
        "severe_red_flags": {
            flag: {"triggered": False, "evidence_ids": []} for flag in SEVERE_RED_FLAGS
        },
    }


def test_forward_eligibility_and_deep_q25_adjustment_parity() -> None:
    calibration = calibrate_forward_returns(
        _forward_observations(),
        cutoff="2026-07-01T07:00:00Z",
    )
    eligibility = assess_fundamental_eligibility(
        calibration,
        industry="industry-a",
        score_decile=10,
        deep_research_complete=True,
        severe_red_flags=False,
    )
    assert eligibility.eligible
    evaluated = evaluate_deep_research(
        _deep_response(),
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("evidence-1",),
        base_q25_by_horizon=eligibility.base_q25_by_horizon,
        base_eligible=True,
    )
    assert evaluated.weighted_signal == 1.0
    assert evaluated.delta == 0.1
    assert evaluated.adjusted_q25_252 == eligibility.base_q25_by_horizon[252] * 1.1


def test_pipeline_wire_contract_is_typed_canonical_friendly_and_authority_false() -> None:
    empty_rank = RankOutput((), (), (), ())
    result = PipelineResult(
        rank_output=empty_rank,
        portfolio_output=None,
        terminal_state="SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        blockers=("no_portfolio_candidates",),
    )
    assert result.to_wire() == {
        "version": "myquant.v17.v2.pipeline-result.v1",
        "rank_output": {
            "version": "myquant.v17.v2.rank-output.v1",
            "initial_ranked_symbols": [],
            "eligible_ranked_symbols": [],
            "sealed_symbols": [],
            "rows": [],
            "authority": False,
        },
        "portfolio_output": None,
        "terminal_state": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "blockers": ["no_portfolio_candidates"],
        "authority": False,
    }
    assert "to_wire" in PipelineInput.__dict__
