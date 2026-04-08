from quant_investor.agent_protocol import ActionLabel, BranchVerdict, Direction
from quant_investor.market.dag_executor import _build_shortlist


def _verdict(score: float, confidence: float, thesis: str) -> BranchVerdict:
    return BranchVerdict(
        agent_name="branch",
        thesis=thesis,
        direction=Direction.BULLISH if score >= 0 else Direction.BEARISH,
        action=ActionLabel.BUY if score >= 0 else ActionLabel.SELL,
        final_score=score,
        final_confidence=confidence,
        investment_risks=["parse_error"],
        coverage_notes=["covered"],
        diagnostic_notes=["diagnostic"],
    )


def test_quarantined_symbol_is_hard_excluded_from_shortlist():
    research_by_symbol = {
        "000001.SZ": {
            "kline": _verdict(0.8, 0.7, "good"),
            "fundamental": _verdict(0.7, 0.6, "good"),
        },
        "600519.SH": {
            "kline": _verdict(0.9, 0.8, "bad"),
            "fundamental": _verdict(0.85, 0.75, "bad"),
        },
    }

    shortlist = _build_shortlist(
        research_by_symbol,
        ic_hints_by_symbol={},
        macro_verdict=_verdict(0.1, 0.5, "macro"),
        company_name_map={"000001.SZ": "Ping An", "600519.SH": "Kweichow Moutai"},
        top_k=10,
        allowed_symbols={"000001.SZ", "600519.SH"},
        blocked_symbols={"600519.SH"},
    )

    symbols = [item.symbol for item in shortlist]
    assert "000001.SZ" in symbols
    assert "600519.SH" not in symbols
