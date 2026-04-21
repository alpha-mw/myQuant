from __future__ import annotations

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.market.dag_executor import (
    _build_master_evidence_pack,
    _build_shortlist_from_bayesian_records,
)


def _posterior(symbol: str, company_name: str, score: float, rank: int) -> PosteriorResult:
    return PosteriorResult(
        symbol=symbol,
        company_name=company_name,
        prior=PriorSet(composite_prior=0.55),
        likelihoods=LikelihoodSet(),
        posterior_win_rate=0.50 + score * 0.2,
        posterior_expected_alpha=score * 0.1,
        posterior_confidence=0.70,
        posterior_action_score=score,
        rank=rank,
        evidence_sources=["quant", "kline"],
        action_threshold_used=0.55,
        metadata={"posterior_edge_after_costs": score * 0.08, "posterior_capacity_penalty": 0.02},
    )


def test_bayesian_shortlist_is_ranked_before_master_pack_build():
    records = [
        _posterior("000003.SZ", "国农科技", 0.71, 3),
        _posterior("000001.SZ", "平安银行", 0.93, 1),
        _posterior("000002.SZ", "万科A", 0.84, 2),
    ]

    shortlist = _build_shortlist_from_bayesian_records(
        posterior_results=records,
        company_name_map={record.symbol: record.company_name for record in records},
        top_k=2,
    )

    assert [item.symbol for item in shortlist] == ["000001.SZ", "000002.SZ"]
    assert [item.company_name for item in shortlist] == ["平安银行", "万科A"]


def test_master_evidence_pack_uses_bayesian_shortlist_fields():
    records = [
        _posterior("000001.SZ", "平安银行", 0.93, 1),
        _posterior("000002.SZ", "万科A", 0.84, 2),
    ]
    shortlist = _build_shortlist_from_bayesian_records(
        posterior_results=records,
        company_name_map={record.symbol: record.company_name for record in records},
        top_k=2,
    )

    evidence_pack = _build_master_evidence_pack(
        shortlist=shortlist,
        branch_summaries={
            "macro": BranchVerdict(agent_name="macro", thesis="macro stable", final_score=0.1, final_confidence=0.7)
        },
        macro_verdict=BranchVerdict(agent_name="macro", thesis="macro stable", final_score=0.1, final_confidence=0.7),
        risk_constraints={"gross_exposure_cap": 0.6, "target_exposure_cap": 0.5},
        model_roles={"branch_model": "deepseek-reasoner", "master_model": "moonshot-v1-128k"},
        resolver_snapshot={"resolution_strategy": "logical_full_a"},
        data_quality_issues=[],
        company_name_map={record.symbol: record.company_name for record in records},
        top_k=2,
    )

    assert evidence_pack["shortlist"][0]["symbol"] == "000001.SZ"
    assert evidence_pack["shortlist"][0]["company_name"] == "平安银行"
    assert evidence_pack["shortlist"][0]["posterior_action_score"] == 0.93
    assert evidence_pack["shortlist"][0]["posterior_win_rate"] > evidence_pack["shortlist"][1]["posterior_win_rate"]

