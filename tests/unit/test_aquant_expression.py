from __future__ import annotations

import pytest
import pandas as pd

from quant_investor.factors.aquant_expression import (
    build_aquant_expression_inputs,
    cs_rank,
    evaluate_aquant_expression,
    ts_mean,
)
from quant_investor.factors.governance import FactorLifecycleState, FactorRecord, GateResult
from quant_investor.factors.pit_fundamentals import PIT_COLUMNS, write_fundamental_pit_series
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    REPORT_ONLY_SHADOW_RUNTIME_MODE,
    score_with_mined_factors,
)
from quant_investor.market.fundamental_mart import write_fundamental_mart


def _write_pit(metadata_dir):
    metadata_dir.mkdir()
    rows = []
    for symbol, ocf, profit in [("000001.SZ", 100.0, 50.0), ("000002.SZ", 80.0, 80.0)]:
        rows.append(
            {
                "ts_code": symbol,
                "report_period": "20231231",
                "availability_date": "2024-01-01",
                "metric_name": "operating_cashflow",
                "value": ocf,
                "source": "fixture",
                "fetched_at": "",
                "raw_table": "cashflow",
                "raw_field": "n_cashflow_act",
            }
        )
        rows.append(
            {
                "ts_code": symbol,
                "report_period": "20231231",
                "availability_date": "2024-01-01",
                "metric_name": "net_income",
                "value": profit,
                "source": "fixture",
                "fetched_at": "",
                "raw_table": "income",
                "raw_field": "n_income",
            }
        )
    write_fundamental_pit_series(
        pd.DataFrame(rows, columns=PIT_COLUMNS).to_dict("records"),
        metadata_dir=metadata_dir,
    )


def _frames():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    return {
        "000001.SZ": pd.DataFrame(
            {
                "symbol": ["000001.SZ"] * len(dates),
                "trade_date": dates,
                "close": [10, 11, 12, 13, 14, 15, 16, 17],
                "adj_close": [10, 11, 12, 13, 14, 15, 16, 17],
                "volume": [100] * len(dates),
                "amount": [102, 113, 126, 139, 151, 160, 170, 180],
            }
        ),
        "000002.SZ": pd.DataFrame(
            {
                "symbol": ["000002.SZ"] * len(dates),
                "trade_date": dates,
                "close": [20, 20, 21, 21, 22, 23, 23, 24],
                "adj_close": [20, 20, 21, 21, 22, 23, 23, 24],
                "volume": [100] * len(dates),
                "amount": [198, 199, 208, 210, 218, 228, 230, 238],
            }
        ),
    }


def test_aquant_expression_matches_manual_pandas(tmp_path):
    metadata_dir = tmp_path / "metadata"
    _write_pit(metadata_dir)
    inputs = build_aquant_expression_inputs(
        _frames(),
        metadata_dir=metadata_dir,
        fundamental_mart_root=tmp_path / "missing_mart",
    )
    expression = (
        "cs_rank(0.5 * cs_rank(ts_mean((vwap - close) / close, 3)) "
        "+ 0.5 * cs_rank(fin_ocf_to_profit))"
    )

    actual = evaluate_aquant_expression(expression, inputs)
    manual = cs_rank(
        0.5 * cs_rank(ts_mean((inputs.vwap - inputs.close) / inputs.close, 3))
        + 0.5 * cs_rank(inputs.fin_ocf_to_profit)
    )

    pd.testing.assert_frame_equal(actual, manual.astype(float))


def test_aquant_expression_rejects_unknown_name(tmp_path):
    metadata_dir = tmp_path / "metadata"
    _write_pit(metadata_dir)
    inputs = build_aquant_expression_inputs(
        _frames(),
        metadata_dir=metadata_dir,
        fundamental_mart_root=tmp_path / "missing_mart",
    )

    with pytest.raises(ValueError, match="unsupported expression name"):
        evaluate_aquant_expression("cs_rank(close + evil_metric)", inputs)


def test_aquant_expression_reads_full_financial_fields_from_mart(tmp_path):
    mart_root = tmp_path / "parquet" / "cn"
    raw_tables = {
        "fina_indicator": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240102",
                    "f_ann_date": "20240102",
                    "roe_dt": 10.0,
                    "roa": 4.0,
                    "debt_to_assets": 40.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240102",
                    "f_ann_date": "20240102",
                    "roe_dt": 20.0,
                    "roa": 8.0,
                    "debt_to_assets": 30.0,
                },
            ]
        ),
        "income": pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "end_date": "20231231", "ann_date": "20240102", "f_ann_date": "20240102", "n_income_attr_p": 50.0},
                {"ts_code": "000002.SZ", "end_date": "20231231", "ann_date": "20240102", "f_ann_date": "20240102", "n_income_attr_p": 80.0},
            ]
        ),
        "cashflow": pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "end_date": "20231231", "ann_date": "20240102", "f_ann_date": "20240102", "n_cashflow_act": 100.0, "c_pay_acq_const_fiolta": 10.0},
                {"ts_code": "000002.SZ", "end_date": "20231231", "ann_date": "20240102", "f_ann_date": "20240102", "n_cashflow_act": 120.0, "c_pay_acq_const_fiolta": 20.0},
            ]
        ),
        "daily_basic": pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "total_mv": 100000.0, "sector": "bank"},
                {"ts_code": "000002.SZ", "trade_date": "20240102", "total_mv": 200000.0, "sector": "tech"},
            ]
        ),
    }
    write_fundamental_mart(
        raw_tables,
        data_root=mart_root,
        raw_snapshot_root=tmp_path / "snapshots",
        reports_root=tmp_path / "reports",
        run_id="fixture",
    )

    inputs = build_aquant_expression_inputs(
        _frames(),
        fundamental_mart_root=mart_root,
        allow_legacy_fundamental_fallback=False,
    )
    result = evaluate_aquant_expression(
        "cs_rank(fin_roe + fin_roa + fin_debt_to_assets + fin_fcf_to_profit + fcf_to_price)",
        inputs,
    )

    assert result.loc[pd.Timestamp("2024-01-02")].notna().all()


def test_runtime_route_consumes_only_production_aquant_expression(tmp_path):
    metadata_dir = tmp_path / "metadata"
    _write_pit(metadata_dir)
    passed_gates = [
        GateResult(gate_id=i, gate_key=f"gate_{i}", title=f"Gate {i}", passed=True)
        for i in range(1, 9)
    ]
    expression = "cs_rank(fin_ocf_to_profit)"
    production = FactorRecord(
        name="alpha_mix_vwap40_50_ocfprofit_50",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="aquant_expression:alpha_mix_vwap40_50_ocfprofit_50",
        weight=1.0,
        gate_results=passed_gates,
        metadata={
            "expression": expression,
            "metadata_dir": str(metadata_dir),
            "fundamental_mart_root": str(tmp_path / "missing_mart"),
        },
    )
    paper = FactorRecord(
        name="paper_alpha_mix_vwap40_50_ocfprofit_50",
        state=FactorLifecycleState.PAPER_FACTOR,
        implementation="aquant_expression:alpha_mix_vwap40_50_ocfprofit_50",
        weight=1.0,
        gate_results=passed_gates,
        metadata={
            "expression": expression,
            "metadata_dir": str(metadata_dir),
            "fundamental_mart_root": str(tmp_path / "missing_mart"),
        },
    )

    result = score_with_mined_factors(
        _frames(),
        registry=MinedFactorRegistry.from_records([production, paper]),
        runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE,
    )

    assert result.factor_count == 1
    assert result.factors_used == ["alpha_mix_vwap40_50_ocfprofit_50"]
    assert "paper_alpha_mix_vwap40_50_ocfprofit_50" in result.skipped_factors
    assert result.symbol_scores["000001.SZ"] > result.symbol_scores["000002.SZ"]


def test_default_registry_has_aquant_shadow_registration_not_selectable():
    registry = MinedFactorRegistry.load()
    factors = {factor.name: factor for factor in registry.factors}
    factor = factors["alpha_mix_vwap40_50_ocfprofit_50"]

    assert factor.implementation == "aquant_expression:alpha_mix_vwap40_50_ocfprofit_50"
    assert factor.state == FactorLifecycleState.RESEARCH_CANDIDATE
    assert factor.weight == 0.0
    assert factor.metadata["fundamental_mart_root"] == "data/parquet/cn"
    assert factor.metadata["allow_legacy_fundamental_fallback"] is False
    assert not factor.selectable_in_quant_branch()
