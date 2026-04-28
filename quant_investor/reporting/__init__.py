"""
新报告层组件。
"""

from quant_investor.reporting.action_consistency_guard import ActionConsistencyGuard
from quant_investor.reporting.conclusion_renderer import ConclusionRenderer
from quant_investor.reporting.diagnostics_bucketizer import DiagnosticsBucketizer
from quant_investor.reporting.executive_summary import ExecutiveSummaryBuilder
from quant_investor.reporting.formal_diagnostics import (
    HoldingDecisionDiagnostic,
    ReportDecisionGuardrailResult,
    ReportWarning,
)

__all__ = [
    "ActionConsistencyGuard",
    "ConclusionRenderer",
    "DiagnosticsBucketizer",
    "ExecutiveSummaryBuilder",
    "HoldingDecisionDiagnostic",
    "ReportDecisionGuardrailResult",
    "ReportWarning",
]
