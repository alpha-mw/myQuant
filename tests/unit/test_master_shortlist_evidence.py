"""Tests for ShortlistEvidencePack and master agent shortlist integration."""

from __future__ import annotations

import pytest

from quant_investor.agents.agent_contracts import (
    MasterAgentInput,
    ShortlistEvidencePack,
)


class TestShortlistEvidencePack:
    def test_pack_construction(self):
        pack = ShortlistEvidencePack(
            symbol="000001.SZ",
            company_name="平安银行",
            bayesian_record={
                "posterior_win_rate": 0.68,
                "posterior_action_score": 0.62,
                "posterior_confidence": 0.75,
            },
            branch_verdicts_summary={
                "kline": {"score": 0.3, "confidence": 0.7, "direction": "bullish"},
                "quant": {"score": 0.5, "confidence": 0.8, "direction": "bullish"},
            },
            risk_flags=["sector_concentration"],
            key_catalysts=["Q1 earnings beat"],
            macro_summary="震荡低波",
        )
        assert pack.symbol == "000001.SZ"
        assert pack.company_name == "平安银行"
        assert pack.bayesian_record["posterior_win_rate"] == 0.68

    def test_master_input_with_shortlist_evidence(self):
        packs = [
            ShortlistEvidencePack(
                symbol=f"SYM{i}",
                company_name=f"Company {i}",
                bayesian_record={"posterior_action_score": 0.7 - i * 0.05},
            )
            for i in range(5)
        ]
        master_input = MasterAgentInput(
            market_regime="震荡低波",
            shortlist_evidence=packs,
            candidate_symbols=[p.symbol for p in packs],
        )
        assert len(master_input.shortlist_evidence) == 5
        assert master_input.shortlist_evidence[0].symbol == "SYM0"

    def test_master_input_backward_compatible_without_shortlist(self):
        master_input = MasterAgentInput(
            branch_results={"kline": {"score": 0.5}},
            market_regime="default",
            candidate_symbols=["A", "B"],
        )
        assert master_input.shortlist_evidence == []
        assert master_input.branch_results["kline"]["score"] == 0.5

    def test_pack_carries_review_overlays(self):
        pack = ShortlistEvidencePack(
            symbol="600519.SH",
            review_overlays={
                "kline": {"score_delta": 0.05, "thesis": "momentum aligned"},
            },
        )
        assert "kline" in pack.review_overlays
        assert pack.review_overlays["kline"]["score_delta"] == 0.05
