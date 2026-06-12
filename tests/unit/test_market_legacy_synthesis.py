"""Legacy market-report synthesis boundary tests."""

from __future__ import annotations

from types import SimpleNamespace

from quant_investor.market.legacy_synthesis import (
    synthesize_legacy_analysis_results_from_dag,
)


class _Payload(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def test_synthesize_legacy_results_filters_retired_branches():
    packet = _Payload(
        symbol="000001.SZ",
        company_name="平安银行",
        category="hs300",
        branch_scores={
            "quant": 0.4,
            "fundamental": 0.2,
            "intelligence": 0.1,
            "macro": 0.0,
            "kline": 0.9,
        },
        branch_confidences={
            "quant": 0.8,
            "fundamental": 0.7,
            "intelligence": 0.6,
            "macro": 0.5,
            "kline": 0.9,
        },
        branch_theses={"quant": "量化正向", "kline": "retired"},
        metadata={"latest_close": 10.0},
        diagnostic_notes=[],
        risk_flags=[],
    )
    shortlist_item = _Payload(
        symbol="000001.SZ",
        company_name="平安银行",
        rank_score=0.3,
        confidence=0.7,
        suggested_weight=0.12,
        rationale=["进入 shortlist"],
        risk_flags=[],
        action="buy",
    )
    portfolio_decision = _Payload(
        target_exposure=0.4,
        target_weights={"000001.SZ": 0.12},
        target_positions={"000001.SZ": 1200},
        metadata={"style_bias": "均衡"},
        risk_constraints={"risk_decision": {"hard_veto": False}},
    )
    dag_artifacts = {
        "symbol_research_packets": {"000001.SZ": packet},
        "shortlist": [shortlist_item],
        "portfolio_decision": portfolio_decision,
        "branch_summaries": {
            "quant": {"score": 0.4, "confidence": 0.8},
            "kline": {"score": 0.9, "confidence": 0.9},
        },
        "execution_trace": _Payload(steps=[]),
        "data_quality_issues": [],
        "global_context": _Payload(
            metadata={"data_snapshot": {"market": "CN"}}
        ),
        "data_snapshot": {"market": "CN"},
    }

    results = synthesize_legacy_analysis_results_from_dag(
        dag_artifacts=dag_artifacts,
        market="CN",
        universe="hs300",
        categories=["hs300"],
        total_capital=1_000_000,
    )

    batch = results["hs300"][0]
    recommendation = batch["recommendations"][0]
    assert set(batch["branches"]) == {"quant"}
    assert recommendation["symbol"] == "000001.SZ"
    assert recommendation["company_name"] == "平安银行"
    assert recommendation["portfolio_amount"] == 120_000
    assert recommendation["branch_positive_count"] == 3
