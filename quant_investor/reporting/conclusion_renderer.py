"""
报告层 markdown 渲染器。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quant_investor.agent_protocol import BranchVerdict, ICDecision, PortfolioPlan
from quant_investor.reporting.action_consistency_guard import ActionConsistencyGuard
from quant_investor.reporting.diagnostics_bucketizer import dedupe_texts, sanitize_report_text
from quant_investor.reporting.executive_summary import confidence_label

BRANCH_LABELS = {
    "kline": "K线",
    "quant": "量化",
    "fundamental": "基本面",
    "intelligence": "智能融合",
    "macro": "宏观",
}


class ConclusionRenderer:
    """渲染市场观点、分支结论、股票卡片和 markdown。"""

    @classmethod
    def render_market_view(
        cls,
        macro_verdict: BranchVerdict,
        ic_decisions: Sequence[ICDecision],
        portfolio_plan: PortfolioPlan,
    ) -> str:
        regime = str(macro_verdict.metadata.get("regime", "neutral"))
        style_bias = str(macro_verdict.metadata.get("style_bias", "balanced"))
        target_gross = float(portfolio_plan.target_gross_exposure)
        ic_actions = [decision.action for decision in ic_decisions]
        guard = ActionConsistencyGuard.guard(
            action=ic_decisions[0].action if ic_decisions else "watch",
            conclusion=(
                f"当前宏观环境处于 {regime}，组合应在 {target_gross:.1%} 总暴露附近保持"
                f" {style_bias} 风格。"
            ),
            ic_actions=ic_actions,
            risk_action_cap=portfolio_plan.metadata.get("action_cap")
            or portfolio_plan.metadata.get("risk_action_cap"),
            subject="当前市场",
        )
        return guard["conclusion"]

    @classmethod
    def render_branch_conclusions(
        cls,
        branch_summaries: Mapping[str, BranchVerdict | Mapping[str, Any]],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for branch_name in sorted(branch_summaries):
            branch = branch_summaries[branch_name]
            if isinstance(branch, BranchVerdict):
                thesis = branch.thesis
                confidence = float(branch.final_confidence)
                risk_candidates = dedupe_texts(
                    sanitize_report_text(item) for item in branch.investment_risks
                )
                coverage_candidates = dedupe_texts(
                    sanitize_report_text(item) for item in branch.coverage_notes
                )
            else:
                thesis = str(branch.get("thesis") or branch.get("conclusion") or "").strip()
                confidence = float(branch.get("final_confidence", branch.get("confidence", 0.0)))
                risk_candidates = dedupe_texts(
                    sanitize_report_text(item)
                    for item in branch.get("investment_risks", branch.get("risks", []))
                )
                coverage_candidates = dedupe_texts(
                    sanitize_report_text(item)
                    for item in branch.get("coverage_notes", [])
                )
            label = BRANCH_LABELS.get(branch_name, branch_name)
            risks = [
                item for item in risk_candidates
                if "数据覆盖摘要" not in item and "缺少覆盖" not in item and "数据接口本轮不可用" not in item
            ][:2]
            coverage_hints = [
                item for item in (risk_candidates + coverage_candidates)
                if "数据覆盖摘要" in item or "缺少覆盖" in item or "数据接口本轮不可用" in item
            ][:2]
            risk_tail = f"；投资风险包括 {'；'.join(risks)}" if risks else ""
            coverage_tail = f"；数据覆盖提示包括 {'；'.join(coverage_hints)}" if coverage_hints else ""
            result[branch_name] = (
                f"{label}分支结论：{sanitize_report_text(thesis)}"
                f"；可信度为{confidence_label(confidence)}{risk_tail}{coverage_tail}。"
            )
        return result

    @classmethod
    def render_stock_cards(
        cls,
        ic_decisions: Sequence[ICDecision],
        portfolio_plan: PortfolioPlan,
    ) -> list[dict[str, Any]]:
        symbol_meta = cls._collect_symbol_meta(ic_decisions)
        cards: list[dict[str, Any]] = []
        for symbol, weight in sorted(
            portfolio_plan.target_positions.items(),
            key=lambda item: (-float(item[1]), item[0]),
        ):
            meta = symbol_meta.get(symbol, {})
            ic_actions = meta.get("ic_actions", [meta.get("action", "buy")])
            raw_action = meta.get("action", "buy" if weight > 0 else "watch")
            default_conclusion = (
                f"{symbol} 当前进入目标仓位 {float(weight):.1%}，按纪律分批执行。"
                if float(weight) >= 0.08
                else f"{symbol} 当前进入观察后的轻仓配置区间，需继续验证。"
            )
            guard = ActionConsistencyGuard.guard(
                action=raw_action,
                conclusion=str(meta.get("one_line_conclusion") or default_conclusion),
                ic_actions=ic_actions,
                risk_action_cap=portfolio_plan.metadata.get("action_cap")
                or portfolio_plan.metadata.get("risk_action_cap"),
                subject=symbol,
            )
            cards.append(
                {
                    "symbol": symbol,
                    "target_weight": float(weight),
                    "display_action": guard["display_action"],
                    "one_line_conclusion": guard["conclusion"],
                    "confidence": float(meta.get("confidence", 0.0)),
                }
            )
        return cards

    @classmethod
    def render_markdown(
        cls,
        executive_summary: Sequence[str],
        market_view: str,
        branch_conclusions: Mapping[str, str],
        stock_cards: Sequence[Mapping[str, Any]],
        coverage_summary: Sequence[str],
        appendix_diagnostics: Sequence[str],
    ) -> str:
        lines = [
            "# 投资研究执行报告",
            "",
            "## 三句话执行摘要",
        ]
        lines.extend(f"- {line}" for line in list(executive_summary)[:3])
        lines.extend(
            [
                "",
                "## 市场观点",
                f"- {market_view}",
                "",
                "## 分支结论",
            ]
        )

        for branch_name in ["kline", "quant", "fundamental", "intelligence", "macro"]:
            if branch_name not in branch_conclusions:
                continue
            lines.extend(
                [
                    f"### {BRANCH_LABELS.get(branch_name, branch_name)}分支",
                    f"- 结论: {branch_conclusions[branch_name]}",
                    "",
                ]
            )

        lines.append("## 推荐标的卡片")
        if stock_cards:
            for card in stock_cards:
                lines.extend(
                    [
                        f"### {card['symbol']}",
                        f"- 一句话结论: {card['one_line_conclusion']}",
                        f"- 展示动作: {card['display_action']}",
                        f"- 目标权重: {float(card['target_weight']):.1%}",
                        "",
                    ]
                )
        else:
            lines.extend(["- 当前没有进入目标仓位的标的。", ""])

        lines.append("## 数据覆盖摘要")
        lines.extend(f"- {line}" for line in coverage_summary)
        lines.extend(["", "## 附录：工程诊断"])
        lines.extend(f"- {line}" for line in appendix_diagnostics)
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _collect_symbol_meta(ic_decisions: Sequence[ICDecision]) -> dict[str, dict[str, Any]]:
        metadata_by_symbol: dict[str, dict[str, Any]] = {}
        for decision in ic_decisions:
            metadata = decision.metadata if isinstance(decision.metadata, Mapping) else {}
            candidates = metadata.get("symbol_candidates", [])
            if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        continue
                    symbol = str(candidate.get("symbol", "")).strip()
                    if not symbol:
                        continue
                    bucket = metadata_by_symbol.setdefault(symbol, {"ic_actions": []})
                    action = candidate.get("action", decision.action.value)
                    bucket["action"] = action
                    bucket["confidence"] = float(candidate.get("confidence", decision.final_confidence))
                    bucket["one_line_conclusion"] = str(
                        candidate.get("one_line_conclusion")
                        or candidate.get("thesis")
                        or candidate.get("conclusion")
                        or ""
                    ).strip()
                    bucket["ic_actions"].append(action)
            symbol_actions = metadata.get("symbol_actions", {})
            symbol_confidences = metadata.get("symbol_confidences", {})
            symbol_conclusions = metadata.get("symbol_conclusions", {})
            if isinstance(symbol_actions, Mapping):
                for symbol, action in symbol_actions.items():
                    text = str(symbol).strip()
                    if not text:
                        continue
                    bucket = metadata_by_symbol.setdefault(text, {"ic_actions": []})
                    bucket["action"] = action
                    if isinstance(symbol_confidences, Mapping):
                        bucket["confidence"] = float(
                            symbol_confidences.get(text, decision.final_confidence)
                        )
                    if isinstance(symbol_conclusions, Mapping):
                        bucket["one_line_conclusion"] = str(symbol_conclusions.get(text, "")).strip()
                    bucket["ic_actions"].append(action)
            for symbol in decision.selected_symbols:
                text = str(symbol).strip()
                if not text:
                    continue
                bucket = metadata_by_symbol.setdefault(text, {"ic_actions": []})
                bucket.setdefault("action", decision.action.value)
                bucket.setdefault("confidence", float(decision.final_confidence))
                bucket.setdefault("one_line_conclusion", "")
                bucket["ic_actions"].append(bucket["action"])
        for symbol in metadata_by_symbol:
            metadata_by_symbol[symbol]["ic_actions"] = list(
                dedupe_texts(str(item) for item in metadata_by_symbol[symbol].get("ic_actions", []))
            )
        return metadata_by_symbol
