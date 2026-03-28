"""
FundamentalAgent：对现有基本面分支做 agent 化包装。
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.fundamental_branch import FundamentalBranch
from quant_investor.agents.base import BaseAgent


class FundamentalAgent(BaseAgent):
    """基本面 research agent。"""

    agent_name = "FundamentalAgent"

    def run(self, payload: Mapping[str, Any]) -> Any:
        envelope = self.ensure_payload(payload)
        data_bundle = envelope.get("data_bundle")
        if not isinstance(data_bundle, UnifiedDataBundle):
            raise TypeError("FundamentalAgent 需要 `data_bundle: UnifiedDataBundle`")

        stock_pool = list(envelope.get("stock_pool") or data_bundle.symbols)
        data_layer = envelope.get("data_layer")
        if data_layer is None:
            data_layer = EnhancedDataLayer(
                market=str(envelope.get("market", data_bundle.market or "CN")),
                verbose=bool(envelope.get("verbose", False)),
            )

        branch = FundamentalBranch(
            data_layer=data_layer,
            stock_pool=stock_pool,
            enable_document_semantics=bool(envelope.get("enable_document_semantics", True)),
        )
        result = branch.run(data_bundle)

        thesis = self._build_thesis(result)
        verdict = self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={
                "module_coverage": dict(result.module_coverage),
                "data_quality": dict(result.data_quality),
            },
        )
        verdict.investment_risks = [
            item
            for item in verdict.investment_risks
            if "provider_missing" not in item.lower() and "snapshot_missing" not in item.lower()
        ]
        return verdict

    @staticmethod
    def _build_thesis(result) -> str:
        if str(result.conclusion or "").strip():
            return str(result.conclusion).strip()

        active_modules = [
            str(info.get("label", name))
            for name, info in dict(result.module_coverage).items()
            if info.get("status") == "active" and int(info.get("available_symbols", 0)) > 0
        ]
        module_text = "、".join(active_modules[:4]) if active_modules else "可用模块有限"
        return f"基本面分支当前由 {module_text} 参与评分并形成结构化判断。"
