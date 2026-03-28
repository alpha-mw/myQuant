from __future__ import annotations

from tests.integration._helpers import run_stubbed_quant_path


def test_single_symbol_flow_with_review_layer_enabled(monkeypatch):
    run = run_stubbed_quant_path(
        monkeypatch,
        symbols=["000001.SZ"],
        thesis_prefix="review-single",
        veto=False,
        review_enabled=True,
    )

    result = run.result

    assert result.agent_layer_enabled is True
    assert result.master_review_output is not None
    assert result.branch_review_outputs["macro"] is not None
    assert result.final_strategy.target_weights == run.artifacts["portfolio_plan"].target_weights
    assert result.agent_report_bundle.portfolio_plan.target_weights == result.final_strategy.target_weights
    assert max(result.final_strategy.target_weights.values(), default=0.0) < 0.95
    assert result.final_strategy.metadata["agent_layer_enabled"] is True
    assert result.final_strategy.provenance_summary["agent_layer_enabled"] is True


def test_shortlist_flow_with_review_layer_enabled(monkeypatch):
    run = run_stubbed_quant_path(
        monkeypatch,
        symbols=["000001.SZ", "600519.SH", "000858.SZ"],
        thesis_prefix="review-shortlist",
        veto=False,
        review_enabled=True,
    )

    result = run.result

    assert result.agent_layer_enabled is True
    assert result.master_review_output is not None
    assert result.final_strategy.target_weights == run.artifacts["portfolio_plan"].target_weights
    assert set(result.final_strategy.target_weights) == set(run.artifacts["portfolio_plan"].target_weights)
    assert result.agent_report_bundle.portfolio_plan.target_weights == result.final_strategy.target_weights
    assert max(result.final_strategy.target_weights.values(), default=0.0) < 0.95


def test_fallback_flow_without_review_layer(monkeypatch):
    run = run_stubbed_quant_path(
        monkeypatch,
        symbols=["000001.SZ", "600519.SH"],
        thesis_prefix="fallback",
        veto=False,
        review_enabled=False,
    )

    result = run.result

    assert result.agent_layer_enabled is False
    assert result.master_review_output is None
    assert result.branch_review_outputs == {}
    assert result.final_strategy.target_weights == run.artifacts["portfolio_plan"].target_weights
    assert result.agent_report_bundle.portfolio_plan.target_weights == result.final_strategy.target_weights
    assert result.final_strategy.metadata["agent_layer_enabled"] is False
    assert result.final_strategy.provenance_summary["agent_layer_enabled"] is False
