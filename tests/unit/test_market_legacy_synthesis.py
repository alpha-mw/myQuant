"""Legacy market-report synthesis boundary tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_investor.market.full_report import MarketArtifactContractError
from quant_investor.market.legacy_synthesis import (
    synthesize_legacy_analysis_results_from_dag,
)
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)


class _Payload(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _current_report_bundle(**overrides):
    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "ic_protocol_version": IC_PROTOCOL_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
    }
    payload.update(overrides)
    return _Payload(**payload)


def _branch_summaries():
    return {
        "quant": {"score": 0.4, "confidence": 0.8},
        "fundamental": {"score": 0.2, "confidence": 0.7},
        "macro": {"score": 0.0, "confidence": 0.5},
    }


def _dag_artifacts():
    packet = _Payload(
        symbol="000001.SZ",
        company_name="平安银行",
        category="hs300",
        branch_scores={
            "quant": 0.4,
            "fundamental": 0.2,
            "macro": 0.0,
        },
        branch_confidences={
            "quant": 0.8,
            "fundamental": 0.7,
            "macro": 0.5,
        },
        branch_theses={
            "quant": "量化正向",
            "fundamental": "基本面正向",
            "macro": "宏观中性",
        },
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
    return {
        "symbol_research_packets": {"000001.SZ": packet},
        "shortlist": [shortlist_item],
        "portfolio_decision": portfolio_decision,
        "branch_summaries": _branch_summaries(),
        "report_bundle": _current_report_bundle(),
        "execution_trace": _Payload(steps=[]),
        "data_quality_issues": [],
        "global_context": _Payload(
            metadata={"data_snapshot": {"market": "CN"}}
        ),
        "data_snapshot": {"market": "CN"},
    }


def test_synthesize_legacy_results_emits_current_three_branch_contract():
    dag_artifacts = _dag_artifacts()

    results = synthesize_legacy_analysis_results_from_dag(
        dag_artifacts=dag_artifacts,
        market="CN",
        universe="hs300",
        categories=["hs300"],
        total_capital=1_000_000,
    )

    batch = results["hs300"][0]
    recommendation = batch["recommendations"][0]
    assert list(batch["branches"]) == ["quant", "fundamental", "macro"]
    assert batch["architecture_version"] == ARCHITECTURE_VERSION
    assert batch["analysis_meta"]["likelihood_schema_version"] == (
        LIKELIHOOD_SCHEMA_VERSION
    )
    assert batch["analysis_meta"]["ic_protocol_version"] == IC_PROTOCOL_VERSION
    assert recommendation["symbol"] == "000001.SZ"
    assert recommendation["company_name"] == "平安银行"
    assert recommendation["portfolio_amount"] == 120_000
    assert recommendation["branch_positive_count"] == 2


def test_synthesis_rejects_retired_branch_instead_of_filtering_it():
    dag_artifacts = _dag_artifacts()
    dag_artifacts["branch_summaries"]["intelligence"] = {
        "score": 0.9,
        "confidence": 0.9,
    }

    with pytest.raises(MarketArtifactContractError, match="intelligence"):
        synthesize_legacy_analysis_results_from_dag(
            dag_artifacts=dag_artifacts,
            market="CN",
            universe="hs300",
            categories=["hs300"],
            total_capital=1_000_000,
        )


def test_synthesis_rejects_stale_report_bundle():
    dag_artifacts = _dag_artifacts()
    dag_artifacts["report_bundle"] = _current_report_bundle(
        report_protocol_version="report-protocol.v13.four-branch"
    )

    with pytest.raises(MarketArtifactContractError, match="report_protocol_version"):
        synthesize_legacy_analysis_results_from_dag(
            dag_artifacts=dag_artifacts,
            market="CN",
            universe="hs300",
            categories=["hs300"],
            total_capital=1_000_000,
        )
