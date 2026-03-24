"""
设计意图 7：coverage / investment_risk / diagnostic 必须严格分桶。

当前预期：通过。
"""

from __future__ import annotations

from quant_investor.agents.quant_agent import QuantAgent
from quant_investor.branch_contracts import BranchResult
from quant_investor.reporting.diagnostics_bucketizer import DiagnosticsBucketizer


def test_bucketization_reroutes_coverage_and_diagnostic_notes_out_of_investment_risk() -> None:
    verdict = QuantAgent().branch_result_to_verdict(
        BranchResult(
            branch_name="quant",
            score=0.25,
            confidence=0.70,
            explanation="量化分支完成结构化判断。",
            symbol_scores={"AAA": 0.25},
            risks=[
                "provider_missing",
                "Traceback: hidden stack",
                "流动性恶化可能压缩持仓上限。",
            ],
        )
    )

    assert verdict.investment_risks == ["流动性恶化可能压缩持仓上限。"]
    assert verdict.coverage_notes == ["provider_missing"]
    assert verdict.diagnostic_notes == ["Traceback: hidden stack"]

    bucketed = DiagnosticsBucketizer({"quant": verdict}).bucket()

    assert "provider_missing" not in bucketed["investment_risks"]
    assert any("覆盖说明" in line for line in bucketed["coverage_summary"][:1])
    assert any("工程异常" in line for line in bucketed["appendix_diagnostics"][1:])

