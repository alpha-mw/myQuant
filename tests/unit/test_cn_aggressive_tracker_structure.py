from __future__ import annotations

import importlib
import importlib.util

import quant_investor.monitoring.cn_aggressive_portfolio_tracker as tracker


def test_cn_aggressive_tracker_helpers_are_split_and_reexported() -> None:
    module_names = [
        "quant_investor.monitoring.cn_aggressive_utils",
        "quant_investor.monitoring.cn_aggressive_review_layer",
        "quant_investor.monitoring.cn_aggressive_review_runtime",
        "quant_investor.monitoring.cn_aggressive_rebalance",
        "quant_investor.monitoring.cn_aggressive_report_renderer",
        "quant_investor.monitoring.cn_aggressive_reporting",
    ]
    for module_name in module_names:
        assert importlib.util.find_spec(module_name) is not None

    utils = importlib.import_module("quant_investor.monitoring.cn_aggressive_utils")
    review_layer = importlib.import_module(
        "quant_investor.monitoring.cn_aggressive_review_layer"
    )
    review_runtime = importlib.import_module(
        "quant_investor.monitoring.cn_aggressive_review_runtime"
    )
    rebalance = importlib.import_module(
        "quant_investor.monitoring.cn_aggressive_rebalance"
    )
    report_renderer = importlib.import_module(
        "quant_investor.monitoring.cn_aggressive_report_renderer"
    )
    reporting = importlib.import_module(
        "quant_investor.monitoring.cn_aggressive_reporting"
    )

    assert tracker._jsonable is utils._jsonable
    assert tracker._safe_float is utils._safe_float
    assert tracker._build_dag_four_branch_compliance is review_layer._build_dag_four_branch_compliance
    assert tracker._render_dag_compliance_markdown is review_layer._render_dag_compliance_markdown
    assert (
        tracker._run_unified_review_mainline_impl
        is review_runtime.run_unified_review_mainline_for_holdings
    )
    assert tracker.ProposedOrder is rebalance.ProposedOrder
    assert tracker.INDEX_QUOTES is rebalance.INDEX_QUOTES
    assert tracker._build_rebalance_plan is rebalance._build_rebalance_plan
    assert tracker._apply_orders is rebalance._apply_orders
    assert tracker._format_holding_advice_line is reporting._format_holding_advice_line
    assert tracker._build_formal_report_text is report_renderer.build_formal_report_text
    assert tracker._build_data_status_summary is reporting._build_data_status_summary
    assert tracker._write_outputs is reporting._write_outputs
